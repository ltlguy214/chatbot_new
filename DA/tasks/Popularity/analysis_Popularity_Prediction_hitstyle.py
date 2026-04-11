'''
TimeSeriesSplit(spotify_release_date) for Popularity Prediction (Regression)
optuna for selected regressors
shap cache artifact for external runner

Hit-style code structure (same def blocks):
- LOAD DATA
- FEATURE ENGINEERING
- TRAIN/EVAL: baseline -> pick best by Test MAE -> optuna (cached params) -> rollback

Notes (Windows): default CV_N_JOBS=1 to avoid loky WinError 1455.
Set CV_N_JOBS=-1 if your machine has enough paging file.
'''

import sys

# When this file is executed as a script, its module name is '__main__'.
# We alias it to its canonical import path so joblib/pickle can reliably
# save/load objects that reference functions/classes defined in this file.
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
_DA_DIR = next((p for p in _THIS_FILE.parents if p.name == "DA"), None)
_CANONICAL_MODULE = 'DA.tasks.Popularity.analysis_Popularity_Prediction_hitstyle'
if __name__ == '__main__':
    sys.modules[_CANONICAL_MODULE] = sys.modules[__name__]

# --- Console encoding (Windows-safe) ---
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

import os
import json
import warnings
import atexit
from datetime import datetime

# --- Ensure repo root is on sys.path (so `import DA...` works when run by file path) ---
if _DA_DIR is not None:
    _REPO_ROOT = _DA_DIR.parent
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd

# Use a non-interactive backend to avoid Tkinter/thread teardown errors on Windows.
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import joblib
import optuna

from optuna.trial import TrialState

from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, VotingRegressor, StackingRegressor
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import BayesianRidge, ElasticNet, HuberRegressor, Lasso, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.model_selection import TimeSeriesSplit, KFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from DA.utils.sklearn_utils import sparse_to_dense

from sklearn.base import BaseEstimator

try:
    from imblearn.pipeline import Pipeline as ImbPipeline
    _HAS_IMBLEARN = True
except Exception:
    ImbPipeline = None  # type: ignore[assignment]
    _HAS_IMBLEARN = False

import xgboost as xgb
import lightgbm as lgb

_HAS_SHAP_ARTIFACT = True
try:
    # When executed as a script: `python DA/analysis_*.py`, DA/ is on sys.path.
    from DA.SHAP_explain.shap_artifact import ShapCacheConfig, build_shap_cache
except ModuleNotFoundError:
    try:
        # When executed as a module: `python -m DA.analysis_*`, import via package path.
        from DA.SHAP_explain.shap_artifact import ShapCacheConfig, build_shap_cache
    except ModuleNotFoundError:
        _HAS_SHAP_ARTIFACT = False
        ShapCacheConfig = None  # type: ignore[assignment]
        build_shap_cache = None  # type: ignore[assignment]

warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# Run logging (save full console output to file)
# -----------------------------------------------------------------------------
class _Tee:
    def __init__(self, *streams):
        self._streams = streams
        self.encoding = 'utf-8'

    def write(self, s):
        if isinstance(s, bytes):
            try:
                s = s.decode('utf-8', errors='replace')
            except Exception:
                s = str(s)
        for st in self._streams:
            try:
                st.write(s)
            except Exception:
                pass

    def flush(self):
        for st in self._streams:
            try:
                st.flush()
            except Exception:
                pass

    def isatty(self):
        for st in self._streams:
            try:
                if hasattr(st, 'isatty') and st.isatty():
                    return True
            except Exception:
                pass
        return False

def _wrap_skewed_target_regressor(regressor):
    mode = os.getenv('P1_TARGET_TRANSFORM', 'log1p').strip().lower()
    if mode in {'', '0', 'off', 'none', 'false'}:
        return regressor
    if mode in {'log1p', 'log'}:
        return TransformedTargetRegressor(
            regressor=regressor,
            func=np.log1p,
            inverse_func=np.expm1,
        )
    raise ValueError(f"Invalid P1_TARGET_TRANSFORM='{mode}'. Use: none|log1p")


class OutlierFilterSampler(BaseEstimator):
    """Leakage-safe outlier filtering for regression (TRAIN/CV only).

    Implemented as an imblearn-style sampler (fit_resample) so it runs only
    during Pipeline.fit (and per CV fold), never during predict.
    """

    def __init__(
        self,
        *,
        method: str = 'isoforest',
        contamination: float = 0.02,
        n_neighbors: int = 35,
        min_keep_frac: float = 0.80,
        random_state: int = 42,
    ):
        self.method = str(method)
        self.contamination = float(contamination)
        self.n_neighbors = int(n_neighbors)
        self.min_keep_frac = float(min_keep_frac)
        self.random_state = int(random_state)

    def fit(self, X, y=None):
        return self

    def fit_resample(self, X, y):
        X_arr = X.toarray() if hasattr(X, 'toarray') else np.asarray(X)
        y_arr = np.asarray(y)

        n = int(len(X_arr))
        if n == 0:
            return X, y

        method = str(self.method).strip().lower()
        if method in {'', '0', 'off', 'none', 'false'}:
            return X, y

        if not (0.0 < float(self.contamination) < 0.5):
            raise ValueError('contamination must be in (0, 0.5)')

        try:
            if method in {'isoforest', 'isolation_forest', 'iforest'}:
                det = IsolationForest(
                    n_estimators=250,
                    contamination=float(self.contamination),
                    random_state=int(self.random_state),
                    n_jobs=-1,
                )
                flags = det.fit_predict(X_arr)
            elif method in {'lof', 'local_outlier_factor'}:
                det = LocalOutlierFactor(
                    n_neighbors=int(self.n_neighbors),
                    contamination=float(self.contamination),
                    novelty=False,
                )
                flags = det.fit_predict(X_arr)
            else:
                raise ValueError(f"Invalid outlier method '{self.method}'. Use: none|isoforest|lof")
        except Exception:
            # Fail-safe: never crash training for outlier filter.
            return X, y

        keep_mask = (np.asarray(flags).reshape(-1) == 1)
        keep_n = int(np.sum(keep_mask))
        min_keep_n = int(np.ceil(float(self.min_keep_frac) * n))
        if keep_n < min_keep_n:
            # Too aggressive: keep all to avoid harming CV folds.
            return X, y

        if hasattr(X, 'iloc'):
            X_new = X.iloc[np.flatnonzero(keep_mask)].reset_index(drop=True)
        else:
            X_new = X_arr[keep_mask]
        y_new = y_arr[keep_mask]
        return X_new, y_new


OutlierFilterSampler.__module__ = _CANONICAL_MODULE

def _enable_task_logging(*, task_dir: Path, task_tag: str) -> str:
    task_dir.mkdir(parents=True, exist_ok=True)
    latest_path = task_dir / f'{task_tag}_run_latest.log'

    # Line-buffered so logs still contain content if the run is interrupted.
    latest_f = open(latest_path, 'w', encoding='utf-8', buffering=1)

    orig_out, orig_err = sys.stdout, sys.stderr
    sys.stdout = _Tee(orig_out, latest_f)
    sys.stderr = _Tee(orig_err, latest_f)

    def _close_files():
        try:
            latest_f.flush()
            latest_f.close()
        except Exception:
            pass

    atexit.register(_close_files)
    print(f"📝 Log file: {latest_path}", flush=True)
    print(
        f"🕒 Started: {datetime.now().isoformat(timespec='seconds')} | Script: {Path(__file__).name}",
        flush=True,
    )
    return str(latest_path)


def _configure_optuna_logging() -> None:
    """Silence Optuna INFO logs like: 'A new study created in memory ...'."""
    try:
        optuna.logging.disable_default_handler()
        optuna.logging.disable_propagation()
        optuna.logging.set_verbosity(optuna.logging.ERROR)
    except Exception:
        pass


_configure_optuna_logging()


# SHAP is intentionally separated to a standalone runner for speed.
# Some environments may not have SHAP deps or can crash on large allocations.
# Keep this on by default, but never crash the whole training if it fails.
BUILD_SHAP_CACHE = (os.getenv('BUILD_SHAP_CACHE', '1') == '1') and _HAS_SHAP_ARTIFACT

# Avoid joblib/loky crashes on Windows by default.
CV_N_JOBS = int(os.getenv('CV_N_JOBS', '1'))

# -----------------------------------------------------------------------------
# Supervised feature selection (optional)
# - Must be inside Pipeline/CV to avoid leakage.
# - Selection is trained on y (TRAIN folds), then applied to X.
# -----------------------------------------------------------------------------
# Default is ON. Set env `P1_FEATURE_SELECTION=none` to disable.
P1_FEATURE_SELECTION = os.getenv('P1_FEATURE_SELECTION', 'tree').strip().lower()  # none|lasso|elasticnet|tree
P1_SFM_ALPHA = float(os.getenv('P1_SFM_ALPHA', '0.001'))
P1_SFM_L1_RATIO = float(os.getenv('P1_SFM_L1_RATIO', '0.5'))
P1_SFM_THRESHOLD = os.getenv('P1_SFM_THRESHOLD', 'median').strip()
_P1_SFM_MAX_FEATURES_RAW = os.getenv('P1_SFM_MAX_FEATURES', '').strip()
P1_SFM_MAX_FEATURES = int(_P1_SFM_MAX_FEATURES_RAW) if _P1_SFM_MAX_FEATURES_RAW.isdigit() else None

P1_TREE_N_ESTIMATORS = int(os.getenv('P1_TREE_N_ESTIMATORS', '600'))
_P1_TREE_MAX_DEPTH_RAW = os.getenv('P1_TREE_MAX_DEPTH', '').strip()
P1_TREE_MAX_DEPTH = int(_P1_TREE_MAX_DEPTH_RAW) if _P1_TREE_MAX_DEPTH_RAW.isdigit() else None

# -----------------------------------------------------------------------------
# Outlier filtering (optional, TRAIN/CV only via Pipeline sampler)
# -----------------------------------------------------------------------------
P1_OUTLIER_FILTER = os.getenv('P1_OUTLIER_FILTER', 'isoforest').strip().lower()  # none|isoforest|lof
P1_OUTLIER_CONTAMINATION = float(os.getenv('P1_OUTLIER_CONTAMINATION', '0.02'))
P1_OUTLIER_N_NEIGHBORS = int(os.getenv('P1_OUTLIER_N_NEIGHBORS', '35'))
P1_OUTLIER_MIN_KEEP_FRAC = float(os.getenv('P1_OUTLIER_MIN_KEEP_FRAC', '0.80'))

# StackingRegressor uses internal CV that is not time-aware; disable by default.
P1_ENABLE_STACKING = os.getenv('P1_ENABLE_STACKING', '0') == '1'

OPTUNA_TRIALS = int(os.getenv('OPTUNA_TRIALS', '20'))
SHOW_OPTUNA_PROGRESS_BAR = os.getenv('OPTUNA_PROGRESS_BAR', '0') == '1'

RANDOM_STATE = 42

TASK_ID = 'P1'
TASK_LABEL = 'P1 - Popularity Prediction'

FINAL_DATA_DIR = Path('DA') / 'final_data'
MODELS_DIR = Path('DA') / 'models'

TASK_DIR = Path('DA') / 'tasks' / 'Popularity'

# User request: store ALL artifacts under the task folder.
SAVE_DIR = TASK_DIR
IMG_DIR = TASK_DIR

FEATURE_NAMES_PATH = SAVE_DIR / 'feature_names_p1.json'

MODEL_PATH = MODELS_DIR / 'best_model_p1.pkl'
LEGACY_MODEL_PATH = MODELS_DIR / 'popularity_model.pkl'

MODEL_COMPARISON_IMG_CSV = FINAL_DATA_DIR / 'p1_model_comparison_results.csv'
MODEL_COMPARISON_CSV = FINAL_DATA_DIR / 'model_comparison_results_p1.csv'

OPTUNA_PARAMS_DIR = IMG_DIR / 'optuna_history_json'

# Mirror P0: Optuna history charts for each tuned model.
OPTUNA_HISTORY_IMG_DIR = IMG_DIR / 'optuna_history_image'

# Persistent Optuna studies (sqlite) to allow resume trials.
OPTUNA_DB_PATH = IMG_DIR / 'optuna_studies_p1.db'

# Legacy: the older P1 script used numbered model names and saved Optuna params as
# DA/tasks/Popularity/optuna_history_json/p1_<model_name>.json (e.g. p1_13._Stacking_...).
LEGACY_OPTUNA_JSON_GLOB = 'p1_*.json'

DPI = 300

def _sanitize_name_for_file(name: str) -> str:
    safe = name.lower().strip()
    safe = safe.replace(' ', '_')
    safe = safe.replace('(', '').replace(')', '')
    safe = safe.replace('+', '_')
    safe = safe.replace('->', 'to')
    safe = safe.replace('/', '-')
    safe = safe.replace('\\', '-')
    return safe

def _optuna_storage_url() -> str:
    # sqlite:/// needs forward slashes even on Windows.
    try:
        OPTUNA_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return f"sqlite:///{OPTUNA_DB_PATH.resolve().as_posix()}"

def _study_name_for_model(model_key: str) -> str:
    # Keep stable across runs; bump suffix if search space changes.
    return f"{TASK_ID}__{_sanitize_name_for_file(model_key)}__mae_v1"

def _get_or_create_study(*, model_key: str, direction: str):
    return optuna.create_study(
        study_name=_study_name_for_model(model_key),
        storage=_optuna_storage_url(),
        load_if_exists=True,
        direction=direction,
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )

def _try_load_legacy_optuna_params(*, contains: str) -> dict | None:

    try:
        if not OPTUNA_PARAMS_DIR.exists():
            return None

        want = contains.lower().strip()
        candidates = [p for p in OPTUNA_PARAMS_DIR.glob(LEGACY_OPTUNA_JSON_GLOB) if want in p.name.lower()]
        if not candidates:
            return None

        # Prefer the newest file.
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        path = candidates[0]
        with open(path, 'r', encoding='utf-8') as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            return None
        print(f"✅ Legacy Optuna params detected → reuse: {path}")
        return payload
    except Exception:
        return None

def _fix_date(val: object) -> str:
    val = str(val).strip()
    if val == '' or val.lower() in {'nan', 'nat', 'none'}:
        return '1900-01-01'
    if len(val) == 4 and val.isdigit():
        return val + '-01-01'
    if len(val) == 7 and val[:4].isdigit() and val[4] == '-' and val[5:7].isdigit():
        return val + '-01'
    return val

def _save_feature_names(pipeline: Pipeline, numeric_feats: list[str], cat_feats: list[str]) -> None:
    """Persist feature names for external reporting."""
    try:
        pre = pipeline.named_steps['preprocessor']
        ohe_names: list[str] = []
        if cat_feats:
            try:
                ohe = pre.named_transformers_['cat']
                ohe_names = ohe.get_feature_names_out(cat_feats).tolist()
            except Exception:
                ohe_names = []

        names = list(numeric_feats) + [str(x) for x in ohe_names]

        SAVE_DIR.mkdir(parents=True, exist_ok=True)
        with open(FEATURE_NAMES_PATH, 'w', encoding='utf-8') as f:
            json.dump(names, f, ensure_ascii=False, indent=2)
        print(f"✅ Đã lưu feature names tại: {FEATURE_NAMES_PATH}")
    except Exception as e:
        print(f"⚠️  Không thể lưu feature names: {e}")

def _plot_custom_optuna_history_minimize(*, study: optuna.Study, model_name: str, baseline_value: float | None, save_path: Path) -> None:
    """P0-style Optuna history plot for minimize objective (CV MAE)."""
    try:
        df = study.trials_dataframe()
        if df.empty or 'value' not in df.columns:
            return
        trial_values = pd.to_numeric(df['value'], errors='coerce').dropna().reset_index(drop=True)
        if len(trial_values) == 0:
            return

        best_values = trial_values.cummin()

        # Save trials to CSV (to reuse without rerunning Optuna)
        try:
            FINAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
            df_csv = df.copy()
            df_csv['best_value_so_far'] = pd.to_numeric(df_csv['value'], errors='coerce').cummin()
            csv_path = FINAL_DATA_DIR / f"{TASK_ID.lower()}_{save_path.stem}.csv"
            df_csv.to_csv(csv_path, index=False, encoding='utf-8-sig')
        except Exception:
            pass

        plt.figure(figsize=(12, 6))
        plt.scatter(
            range(len(trial_values)),
            trial_values,
            color='#add8e6',
            alpha=0.7,
            edgecolors='none',
            label='Từng trial',
            s=35,
        )
        plt.step(
            range(len(best_values)),
            best_values,
            where='post',
            color='red',
            linewidth=2,
            label='CV MAE tốt nhất',
        )

        if baseline_value is not None and np.isfinite(baseline_value):
            plt.axhline(
                y=float(baseline_value),
                color='forestgreen',
                linestyle='--',
                linewidth=1.5,
                label=f'Baseline CV MAE ({float(baseline_value):.4f})',
            )

        title = (
            f"Lịch sử tối ưu hóa {model_name} (mục tiêu: minimize CV MAE)\n"
            f"Best: {float(study.best_value):.4f}" + (f" | Baseline: {float(baseline_value):.4f}" if baseline_value is not None else "")
        )
        plt.title(title, fontsize=14, fontweight='bold', pad=15)
        plt.xlabel('Số lượt thử (Trial)', fontsize=12)
        plt.ylabel('CV MAE (lower is better)', fontsize=12)
        y_min = float(min(trial_values.min(), baseline_value)) if baseline_value is not None else float(trial_values.min())
        y_max = float(max(trial_values.max(), baseline_value)) if baseline_value is not None else float(trial_values.max())
        pad = max(0.01, (y_max - y_min) * 0.15)
        plt.ylim(max(0.0, y_min - pad), y_max + pad)
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.legend(loc='upper right', frameon=True)
        plt.tight_layout()

        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(save_path), dpi=DPI, bbox_inches='tight')
        plt.close()
    except Exception:
        try:
            plt.close()
        except Exception:
            pass

def _plot_baseline_vs_optimized_mae(*, df: pd.DataFrame, save_path: Path) -> None:
    """P0-style grouped bar chart comparing baseline vs optimized (Test MAE)."""
    if df is None or df.empty:
        return

    df_plot = df.copy()
    for c in ['Baseline', 'Optimized', 'Improvement']:
        df_plot[c] = pd.to_numeric(df_plot[c], errors='coerce')
    df_plot = df_plot.replace([np.inf, -np.inf], np.nan).dropna(subset=['Baseline', 'Optimized'])
    if len(df_plot) == 0:
        return

    df_plot = df_plot.sort_values(by='Improvement', ascending=False)
    fig, ax = plt.subplots(figsize=(16, max(8, int(len(df_plot) * 0.55))))
    x = np.arange(len(df_plot))
    width = 0.38

    ax.barh(
        x - width / 2,
        df_plot['Baseline'],
        width,
        label='Baseline (Default params)',
        color='#87CEEB',
        edgecolor='white',
        linewidth=1.5,
    )
    ax.barh(
        x + width / 2,
        df_plot['Optimized'],
        width,
        label='Optimized (Optuna tuned)',
        color='#FF8C42',
        edgecolor='white',
        linewidth=1.5,
    )

    for i, (_, row) in enumerate(df_plot.iterrows()):
        base_v = float(row['Baseline'])
        opt_v = float(row['Optimized'])
        ax.text(base_v + 0.01 * max(1.0, df_plot[['Baseline', 'Optimized']].max().max()), i - width / 2, f"{base_v:.2f}", va='center', fontsize=10, fontweight='bold')
        ax.text(opt_v + 0.01 * max(1.0, df_plot[['Baseline', 'Optimized']].max().max()), i + width / 2, f"{opt_v:.2f}", va='center', fontsize=10, fontweight='bold')

        # Improvement means MAE reduced.
        imp = float(row.get('Improvement', np.nan))
        if np.isfinite(imp) and imp > 0:
            pct = (imp / max(1e-9, base_v)) * 100.0
            ax.text(max(base_v, opt_v) + 0.02 * max(1.0, df_plot[['Baseline', 'Optimized']].max().max()), i, f"↓ -{pct:.2f}%", va='center', color='green', fontweight='bold', fontsize=9)

    ax.set_yticks(x)
    ax.set_yticklabels(df_plot['Model'], fontsize=11)
    ax.set_xlabel('Test MAE (lower is better)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Model', fontsize=13, fontweight='bold')
    ax.set_title('So sánh Baseline vs Optimized - Impact của Optuna Tuning (P1)', fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='lower right', fontsize=11, frameon=True, shadow=True)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=DPI, bbox_inches='tight')
    plt.close(fig)

def _plot_mae_comparison(df_results: pd.DataFrame) -> None:
    """MAE lower is better → best should be on top."""
    if df_results.empty or 'Test_MAE' not in df_results.columns:
        print('⚠️  Không có dữ liệu để vẽ biểu đồ MAE.')
        return

    df_plot = df_results.copy()
    df_plot['Test_MAE'] = pd.to_numeric(df_plot['Test_MAE'], errors='coerce')
    df_plot = df_plot.replace([np.inf, -np.inf], np.nan).dropna(subset=['Test_MAE'])
    df_plot = df_plot.sort_values(by='Test_MAE', ascending=True)

    if len(df_plot) == 0:
        print('⚠️  Không có MAE hợp lệ để vẽ biểu đồ.')
        return

    fig, ax = plt.subplots(figsize=(14, max(8, len(df_plot) * 0.55)))
    palette = sns.color_palette('viridis', len(df_plot))

    ax.barh(df_plot['Model'], df_plot['Test_MAE'], color=palette, edgecolor='white', linewidth=1.5)
    ax.invert_yaxis()  # first (best) row on top

    max_val = float(df_plot['Test_MAE'].max())
    for i, (_, row) in enumerate(df_plot.iterrows()):
        v = float(row['Test_MAE'])
        ax.text(v + 0.01 * max(1.0, max_val), i, f"{v:.2f}", va='center', ha='left', fontweight='bold', fontsize=10)

    ax.set_xlabel('MAE (lower is better)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Mô hình', fontsize=13, fontweight='bold')
    ax.set_title('So sánh mô hình Regression - Spotify Popularity (P1) — Tiêu chí: Test MAE', fontsize=16, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_xlim(0, max_val * 1.15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    IMG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = IMG_DIR / 'p1_model_comparison_mae.png'
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Đã lưu biểu đồ MAE tại: {out_path}")

def _plot_actual_vs_predicted(*, y_true, y_pred, save_path: Path, model_label: str) -> None:
    """Scatter plot: Actual vs Predicted with 45° reference line (regression sanity check)."""
    try:
        y_t = pd.to_numeric(pd.Series(y_true), errors='coerce').astype(float).values
        y_p = pd.to_numeric(pd.Series(y_pred), errors='coerce').astype(float).values
        mask = np.isfinite(y_t) & np.isfinite(y_p)
        y_t = y_t[mask]
        y_p = y_p[mask]
        if len(y_t) == 0:
            print('⚠️  Không có dữ liệu hợp lệ để vẽ Actual vs Predicted.')
            return

        v_min = float(min(np.min(y_t), np.min(y_p)))
        v_max = float(max(np.max(y_t), np.max(y_p)))
        pad = max(1.0, (v_max - v_min) * 0.05)
        lo = v_min - pad
        hi = v_max + pad

        fig, ax = plt.subplots(figsize=(9, 9))
        ax.scatter(y_t, y_p, s=22, alpha=0.55, color='#1f77b4', edgecolor='none')
        ax.plot([lo, hi], [lo, hi], linestyle='--', color='black', linewidth=1.5, alpha=0.8, label='Dự đoán hoàn hảo (45°)')

        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('Popularity thực tế (Actual)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Popularity dự đoán (Predicted)', fontsize=12, fontweight='bold')
        ax.set_title(
            'Actual vs Predicted (Thực tế vs Dự đoán)\n'
            f'{model_label}',
            fontsize=14,
            fontweight='bold',
            pad=14,
        )
        ax.grid(alpha=0.25, linestyle='--')
        ax.legend(loc='upper left', frameon=True)

        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(str(save_path), dpi=DPI, bbox_inches='tight')
        plt.close(fig)
        print(f"✅ Đã lưu biểu đồ Actual vs Predicted tại: {save_path}")
    except Exception as e:
        try:
            plt.close('all')
        except Exception:
            pass
        print(f"⚠️  Không thể vẽ Actual vs Predicted: {e}")


# =============================================================================
# 1. LOAD DỮ LIỆU
# =============================================================================
FILE_DATA = Path('final_data') / 'data_prepared_for_ML.csv'
TARGET_COLUMN = 'spotify_popularity'


def load_data():
    print(f"⏳ Đang tải dữ liệu từ {FILE_DATA}...")
    try:
        path = Path(FILE_DATA)
        if not path.exists():
            print(f"❌ Không tìm thấy file: {path.as_posix()}")
            return None

        df = pd.read_csv(path)
        print(f"✅ Đã tải thành công: {len(df)} dòng dữ liệu.")
        return df
    except Exception as e:
        print(f"❌ Lỗi load data: {e}")
        return None

# =============================================================================
# 2. FEATURE ENGINEERING
# =============================================================================
def prepare_data(df: pd.DataFrame):
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"❌ Thiếu cột target '{TARGET_COLUMN}'!")

    df = df.copy()
    df['target'] = pd.to_numeric(df[TARGET_COLUMN], errors='coerce')

    cols_ignore = [
        'spotify_track_id', 'title', 'artists', 'spotify_release_date', 'genres',
        'spotify_popularity',  # target
        'is_hit', 'target',
        'final_sentiment',
        # multicollinearity drops (aligned with other tasks)
        'mfcc2_mean',
        'spectral_rolloff',
        'noun_count',
        'verb_count',
        'tempo_stability',
        'spectral_contrast_band3_mean',
        'spectral_contrast_band4_mean',
        'spectral_contrast_band5_mean',
    ]

    candidate_numeric = [
        c for c in df.columns
        if c not in cols_ignore and pd.api.types.is_numeric_dtype(df[c])
    ]

    binary_feats: list[str] = []
    numeric_cont_feats: list[str] = []
    for col in candidate_numeric:
        values = pd.Series(df[col].dropna().unique())
        try:
            is_binary = len(values) <= 2 and set(values.astype(int).tolist()).issubset({0, 1})
        except Exception:
            is_binary = False

        if is_binary:
            binary_feats.append(col)
        else:
            numeric_cont_feats.append(col)

    numeric_feats = numeric_cont_feats + binary_feats

    if 'final_sentiment' in df.columns:
        df['final_sentiment'] = df['final_sentiment'].fillna('Neutral')
    cat_feats = ['final_sentiment'] if 'final_sentiment' in df.columns else []

    # TimeSeries safety: normalize + sort by spotify_release_date
    if 'spotify_release_date' in df.columns:
        df['spotify_release_date'] = df['spotify_release_date'].apply(_fix_date)
        df['spotify_release_date'] = pd.to_datetime(df['spotify_release_date'], errors='coerce')
        df['spotify_release_date'] = df['spotify_release_date'].fillna(pd.Timestamp('1900-01-01'))
        df = df.sort_values('spotify_release_date').reset_index(drop=True)

    X = df[numeric_feats + cat_feats]
    y = df['target']

    return df, X, y, numeric_feats, numeric_cont_feats, binary_feats, cat_feats


# =============================================================================
# 3. QUY TRÌNH HUẤN LUYỆN
# =============================================================================
def run_full_analysis_task_p1():
    _enable_task_logging(task_dir=TASK_DIR, task_tag=TASK_ID)
    # === SET GLOBAL SEED ĐỂ ĐẢM BẢO KẾT QUẢ NHẤT QUÁN ===
    np.random.seed(RANDOM_STATE)

    print("\n" + "=" * 80)
    print(f"🚀 {TASK_LABEL} (Regression) — Time-based holdout 80/20")
    print("=" * 80)
    print(f"   • Data: {FILE_DATA}")
    print("   • Split: Time-based 80/20 (spotify_release_date)")
    print(f"   • CV_N_JOBS: {CV_N_JOBS} (env CV_N_JOBS=-1 để dùng đa lõi)")

    df = load_data()
    if df is None:
        return

    df_clean, X, y, numeric_feats, numeric_cont_feats, binary_feats, cat_feats = prepare_data(df)

    y_arr = pd.to_numeric(y, errors='coerce').values
    print(f"🎯 Target: {TARGET_COLUMN} (min={np.nanmin(y_arr):.1f}, max={np.nanmax(y_arr):.1f}, mean={np.nanmean(y_arr):.1f})")

    # Sanity check: master data nên không còn missing.
    if X.isna().any().any() or y.isna().any():
        na_cols = X.columns[X.isna().any()].tolist()
        raise ValueError(
            "❌ Dữ liệu đầu vào vẫn còn missing values. "
            "Hãy chạy lại scripts/data_prepared_for_ML.py để fill median trước khi train. "
            f"Cột bị thiếu: {na_cols[:20]}" + (" ..." if len(na_cols) > 20 else "")
        )

    num_cat_ohe = df_clean['final_sentiment'].nunique() if 'final_sentiment' in df_clean.columns else 0
    print("\n" + "=" * 60)
    print("🔍 KIỂM TRA CHI TIẾT CÁC BIẾN ĐẦU VÀO (TASK P1 - POPULARITY)")
    print("=" * 60)
    print(f"1️⃣ Số lượng biến số (Continuous - sẽ Scale): {len(numeric_cont_feats)}")
    print(np.array(numeric_cont_feats))
    print("-" * 60)
    print(f"1️⃣b Số lượng biến nhị phân (0/1 - KHÔNG Scale): {len(binary_feats)}")
    print(np.array(binary_feats))
    print("-" * 60)
    print(f"1️⃣c Tổng số biến số dùng cho model: {len(numeric_feats)}")
    print("-" * 60)
    print(f"2️⃣ Số lượng biến phân loại (Sentiment OHE): {num_cat_ohe}")
    if 'final_sentiment' in df_clean.columns:
        print(f"   Các nhãn tìm thấy: {df_clean['final_sentiment'].unique()}")
    print("-" * 60)
    total_feats = len(numeric_feats) + int(num_cat_ohe)
    print(f"✅ TỔNG CỘNG SỐ BIẾN ĐƯA VÀO MODEL: {total_feats}")
    print("=" * 60 + "\n")

    # HOLDOUT theo thời gian: 80/20 (không shuffle)
    n_total = len(df_clean)
    split_point = int(n_total * 0.8)
    if split_point <= 0 or split_point >= n_total:
        raise ValueError(f"❌ Split 80/20 không hợp lệ với n_total={n_total}")

    X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
    y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]

    print(f"📌 Holdout 80/20 (time-based) → Train={len(y_train)} | Test={len(y_test)}")
    if 'spotify_release_date' in df_clean.columns:
        print("-" * 60)
        print(
            f"📅 Tập Train: từ {df_clean.iloc[:split_point]['spotify_release_date'].min().date()} "
            f"đến {df_clean.iloc[:split_point]['spotify_release_date'].max().date()} ({len(y_train)} bài)"
        )
        print(
            f"📅 Tập Test : từ {df_clean.iloc[split_point:]['spotify_release_date'].min().date()} "
            f"đến {df_clean.iloc[split_point:]['spotify_release_date'].max().date()} ({len(y_test)} bài)"
        )
        print("-" * 60)

    # Transformer cho biến số
    # - Continuous numeric: Scale trong CV/pipeline (không leak)
    # - Binary 0/1 one-hot: passthrough (KHÔNG scale)
    numeric_transformer = StandardScaler()
    # Ensure dense output (important when supervised feature selection uses Lasso/ElasticNet).
    try:
        categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    except TypeError:  # sklearn<1.2
        categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse=False)

    transformers: list[tuple[str, object, list[str]]] = []
    if numeric_cont_feats:
        transformers.append(('num', numeric_transformer, numeric_cont_feats))
    if binary_feats:
        transformers.append(('bin', 'passthrough', binary_feats))
    if cat_feats:
        transformers.append(('cat', categorical_transformer, cat_feats))

    preprocessor = ColumnTransformer(transformers=transformers)

    # Optional: supervised feature selection (fit only on TRAIN folds via Pipeline/CV)
    feature_selector = None
    selector_requires_dense = False
    if P1_FEATURE_SELECTION in {'lasso', 'elasticnet'}:
        if P1_FEATURE_SELECTION == 'lasso':
            selector_estimator = Lasso(alpha=P1_SFM_ALPHA, max_iter=10000)
        else:
            selector_estimator = ElasticNet(alpha=P1_SFM_ALPHA, l1_ratio=P1_SFM_L1_RATIO, max_iter=10000)
        feature_selector = SelectFromModel(
            estimator=selector_estimator,
            threshold=P1_SFM_THRESHOLD,
            max_features=P1_SFM_MAX_FEATURES,
        )
    elif P1_FEATURE_SELECTION in {'tree', 'extratrees', 'extra_trees'}:
        selector_estimator = ExtraTreesRegressor(
            n_estimators=int(P1_TREE_N_ESTIMATORS),
            max_depth=P1_TREE_MAX_DEPTH,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        feature_selector = SelectFromModel(
            estimator=selector_estimator,
            threshold=P1_SFM_THRESHOLD,
            max_features=P1_SFM_MAX_FEATURES,
        )
        selector_requires_dense = True
    elif P1_FEATURE_SELECTION in {'', '0', 'off', 'none', 'false'}:
        feature_selector = None
    else:
        raise ValueError(f"Invalid P1_FEATURE_SELECTION='{P1_FEATURE_SELECTION}'. Use: none|lasso|elasticnet|tree")

    outlier_enabled = P1_OUTLIER_FILTER not in {'', '0', 'off', 'none', 'false'}
    if outlier_enabled and not _HAS_IMBLEARN:
        raise ImportError('P1_OUTLIER_FILTER requires imbalanced-learn (imblearn). Please install imbalanced-learn.')

    print(
        f"🧩 Feature selection: {P1_FEATURE_SELECTION}"
        + (
            f" | selector={P1_FEATURE_SELECTION}(threshold={P1_SFM_THRESHOLD}, max_features={P1_SFM_MAX_FEATURES})"
            if feature_selector is not None
            else ""
        )
        + (
            f" | outlier_filter={P1_OUTLIER_FILTER}(contam={P1_OUTLIER_CONTAMINATION}, min_keep={P1_OUTLIER_MIN_KEEP_FRAC})"
            if outlier_enabled
            else " | outlier_filter=none"
        )
        + f" | target_transform={os.getenv('P1_TARGET_TRANSFORM', 'log1p')}",
        flush=True,
    )

    def _build_pipe_p1(regressor_ttr):
        pipeline_cls = ImbPipeline if outlier_enabled else Pipeline

        steps = [('preprocessor', preprocessor)]
        if outlier_enabled or selector_requires_dense:
            steps.append(('to_dense', FunctionTransformer(sparse_to_dense, validate=False)))
        if outlier_enabled:
            steps.append(
                (
                    'outliers',
                    OutlierFilterSampler(
                        method=P1_OUTLIER_FILTER,
                        contamination=P1_OUTLIER_CONTAMINATION,
                        n_neighbors=P1_OUTLIER_N_NEIGHBORS,
                        min_keep_frac=P1_OUTLIER_MIN_KEEP_FRAC,
                        random_state=RANDOM_STATE,
                    ),
                )
            )
        if feature_selector is not None:
            steps.append(('select', feature_selector))
        steps.append(('regressor', regressor_ttr))
        return pipeline_cls(steps)

    # TimeSeries CV chỉ dùng trong TRAIN (tránh leakage)
    tscv_inner = TimeSeriesSplit(n_splits=5)

    # =============================================================================
    # 3. BƯỚC 1: CHẠY BASELINE - SO SÁNH BAN ĐẦU
    # =============================================================================
    print("\n" + "=" * 80)
    print("📊 BƯỚC 1: CHẠY BASELINE - SO SÁNH BAN ĐẦU")
    print("=" * 80)

    rf = RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)
    xgb_reg = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        objective='reg:squarederror',
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    lgbm_reg = lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        random_state=RANDOM_STATE,
        verbose=-1,
        n_jobs=-1,
    )
    svr = SVR(kernel='rbf')

    baseline_models = {
        'Ridge Regression': Ridge(),
        'Lasso Regression': Lasso(),
        'ElasticNet': ElasticNet(),
        'Bayesian Ridge': BayesianRidge(),
        'Huber Regressor': HuberRegressor(max_iter=500),
        'KNN Regressor': KNeighborsRegressor(n_jobs=-1),
        'Decision Tree': DecisionTreeRegressor(random_state=RANDOM_STATE),
        'SVR (RBF Kernel)': SVR(kernel='rbf'),
        'Random Forest': rf,
        'XGBoost': xgb_reg,
        'LightGBM': lgbm_reg,
    }

    # Ensemble baselines
    baseline_models['Voting (RF+XGB+LGBM)'] = VotingRegressor(
        estimators=[('rf', rf), ('xgb', xgb_reg), ('lgbm', lgbm_reg)]
    )

    # NOTE: StackingRegressor internally uses cross_val_predict, which requires a
    # partitioning CV (each sample appears in exactly one test fold). TimeSeriesSplit
    # is not partitioning over the full dataset (early samples never appear in test),
    # so we use KFold internally.
    stacking_inner_cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    if P1_ENABLE_STACKING:
        baseline_models['Stacking (RF+XGB+SVR -> Ridge)'] = StackingRegressor(
            estimators=[('rf', rf), ('xgb', xgb_reg), ('svr', svr)],
            final_estimator=Ridge(),
            cv=stacking_inner_cv,
            n_jobs=None,
        )

    baseline_results: list[dict] = []
    baseline_pipelines: dict[str, Pipeline] = {}

    print(f"\n{'MODEL':<40} | {'CV_MAE':<10} | {'TEST_MAE':<10} | {'TEST_RMSE':<10}")
    print('-' * 85)

    for name, model in baseline_models.items():
        try:
            model_ttr = _wrap_skewed_target_regressor(model)
            pipe = _build_pipe_p1(model_ttr)
            cv_scores = cross_val_score(
                pipe,
                X_train,
                y_train,
                cv=tscv_inner,
                scoring='neg_mean_absolute_error',
                n_jobs=CV_N_JOBS,
            )
            cv_mae = float(-np.mean(cv_scores))

            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            test_mae = float(mean_absolute_error(y_test, y_pred))
            test_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))


            baseline_results.append({
                'Model': name,
                'CV_MAE': cv_mae,
                'Test_MAE': test_mae,
                'Test_RMSE': test_rmse,
            })
            baseline_pipelines[name] = pipe
            print(f"{name:<40} | {cv_mae:<10.2f} | {test_mae:<10.2f} | {test_rmse:<10.2f}")
        except Exception as e:
            print(f"❌ Lỗi {name}: {str(e)[:200]}")

    if not baseline_results:
        raise RuntimeError('❌ Không có baseline model nào chạy thành công. Kiểm tra lại dữ liệu/feature/thiết lập model.')

    baseline_df = pd.DataFrame(baseline_results).replace([np.inf, -np.inf], np.nan)
    # Leakage-safe model selection: choose by CV_MAE (TRAIN only). Keep TEST only for reporting.
    baseline_df = baseline_df.sort_values(by='CV_MAE', ascending=True, na_position='last')

    # =============================================================================
    # BƯỚC 1.5: XÁC ĐỊNH BEST MODEL TỪ BASELINE
    # =============================================================================
    print("\n" + "=" * 80)
    print("🏆 XÁC ĐỊNH BEST MODEL TỪ BASELINE")
    print("=" * 80)

    baseline_df_ok = baseline_df.dropna(subset=['CV_MAE'])
    if len(baseline_df_ok) == 0:
        raise RuntimeError('❌ Không có baseline model nào có CV_MAE hợp lệ để chọn best model.')

    best_baseline_name = str(baseline_df_ok.iloc[0]['Model'])
    best_baseline_cv = float(baseline_df_ok.iloc[0]['CV_MAE'])
    best_baseline_mae = float(baseline_df_ok.iloc[0]['Test_MAE'])
    best_baseline_rmse = float(baseline_df_ok.iloc[0]['Test_RMSE'])


    print(
        f"\n✨ BEST BASELINE MODEL (by CV MAE): {best_baseline_name} "
        f"(CV MAE: {best_baseline_cv:.2f} | Test MAE: {best_baseline_mae:.2f} | RMSE: {best_baseline_rmse:.2f})"
    )

    # Save baseline comparison tables
    FINAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    baseline_df.to_csv(str(MODEL_COMPARISON_IMG_CSV), index=False)
    baseline_df.to_csv(str(MODEL_COMPARISON_CSV), index=False)
    print(f"✅ Đã lưu kết quả so sánh tại: {MODEL_COMPARISON_IMG_CSV}")
    print(f"✅ Đã lưu kết quả so sánh tại: {MODEL_COMPARISON_CSV}")
    _plot_mae_comparison(baseline_df_ok)

    # =============================================================================
    # 4. BƯỚC 2: TỐI ƯU CHỈ BEST MODEL BẰNG OPTUNA (CACHED PARAMS)
    # =============================================================================
    print("\n" + "=" * 80)
    print("🔧 BƯỚC 2: TỐI ƯU CHỈ BEST MODEL BẰNG OPTUNA")
    print("=" * 80)

    OPTUNA_PARAMS_DIR.mkdir(parents=True, exist_ok=True)
    optimized_params: dict[str, dict] = {}

    legacy_stacking_params: dict | None = None

    # Hit-style separation: if ensemble wins, optimize base models separately.
    if 'Voting' in best_baseline_name:
        models_to_optimize = ['Random Forest', 'XGBoost', 'LightGBM']
        print(f"📋 Ensemble model detected → Sẽ tối ưu 3 base models: {', '.join(models_to_optimize)}")

    elif 'Stacking' in best_baseline_name:
        # If the legacy script already optimized Stacking and saved a JSON, reuse it
        # to avoid rerunning Optuna (user request).
        legacy_stacking_params = _try_load_legacy_optuna_params(contains='stacking')

        if legacy_stacking_params is not None:
            # Derive base params from legacy stacking payload (rf_* and xgb_* keys).
            # SVR params were not optimized in the legacy payload; keep default SVR.
            models_to_optimize = []
            rf_legacy = {
                k.replace('rf_', ''): v
                for k, v in legacy_stacking_params.items()
                if isinstance(k, str) and k.startswith('rf_')
            }
            xgb_legacy = {
                k.replace('xgb_', ''): v
                for k, v in legacy_stacking_params.items()
                if isinstance(k, str) and k.startswith('xgb_')
            }

            if rf_legacy:
                optimized_params['Random Forest'] = rf_legacy
            if xgb_legacy:
                optimized_params['XGBoost'] = xgb_legacy

            print("📦 Found legacy Stacking Optuna params → skip new Optuna trials.")
            if rf_legacy:
                print("   • Reuse RF params from legacy Stacking")
            if xgb_legacy:
                print("   • Reuse XGBoost params from legacy Stacking")
            print("   • SVR: keep default (no legacy params)")

        else:
            models_to_optimize = ['Random Forest', 'XGBoost', 'SVR']
            print(f"📋 Ensemble model detected → Sẽ tối ưu 3 base models: {', '.join(models_to_optimize)}")

    else:
        # Normalize names so Optuna keys stay stable.
        single_key = 'SVR' if best_baseline_name == 'SVR (RBF Kernel)' else best_baseline_name
        models_to_optimize = [single_key]
        print(f"📋 Single model detected → Sẽ tối ưu: {single_key}")

    def load_cached_params(model_key: str) -> dict | None:
        params_file = OPTUNA_PARAMS_DIR / f"{_sanitize_name_for_file(model_key)}_params.json"
        if params_file.exists():
            try:
                print("✅ Đã tồn tại params, đang load...")
                with open(params_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️  Không load được params cũ ({model_key}), sẽ optimize lại: {e}")
        return None

    def save_params(model_key: str, params: dict) -> None:
        params_file = OPTUNA_PARAMS_DIR / f"{_sanitize_name_for_file(model_key)}_params.json"
        with open(params_file, 'w', encoding='utf-8') as f:
            json.dump(params, f, ensure_ascii=False, indent=2)

    def objective_factory(model_key: str):
        def objective(trial: optuna.Trial) -> float:
            if model_key == 'Random Forest':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 600, step=100),
                    'max_depth': trial.suggest_int('max_depth', 5, 40),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 15),
                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                }
                reg = RandomForestRegressor(**params, random_state=RANDOM_STATE, n_jobs=-1)
            elif model_key == 'XGBoost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 200, 800, step=100),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                }
                reg = xgb.XGBRegressor(
                    **params,
                    objective='reg:squarederror',
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                )
            elif model_key == 'LightGBM':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 200, 800, step=100),
                    'max_depth': trial.suggest_int('max_depth', 3, 12),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'num_leaves': trial.suggest_int('num_leaves', 20, 140),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                }
                reg = lgb.LGBMRegressor(**params, random_state=RANDOM_STATE, verbose=-1, n_jobs=-1)
            elif model_key == 'SVR':
                params = {
                    'C': trial.suggest_float('C', 0.1, 200.0, log=True),
                    'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
                    'epsilon': trial.suggest_float('epsilon', 0.01, 2.0, log=True),
                    'kernel': 'rbf',
                }
                reg = SVR(**params)
            else:
                return float('inf')

            reg_ttr = _wrap_skewed_target_regressor(reg)
            pipe_trial = _build_pipe_p1(reg_ttr)
            cv_scores = cross_val_score(
                pipe_trial,
                X_train,
                y_train,
                cv=tscv_inner,
                scoring='neg_mean_absolute_error',
                n_jobs=CV_N_JOBS,
            )
            return float(-np.mean(cv_scores))

        return objective

    for idx, model_key in enumerate(models_to_optimize, 1):
        print(f"\n{'=' * 70}")
        print(f"🔧 [{idx}/{len(models_to_optimize)}] ĐANG TỐI ƯU: {model_key}")
        print(f"{'=' * 70}")

        cached = load_cached_params(model_key)
        if cached is not None:
            # Still keep sqlite study as the source-of-truth for resume;
            # cached JSON is just a fast-path.
            optimized_params[model_key] = cached
            print("ℹ️  Đã có cached JSON params → dùng ngay (không chạy thêm trials).")
            continue

        # Some models are not optuna-supported in this script.
        supported = model_key in {'Random Forest', 'XGBoost', 'LightGBM', 'SVR'}
        if not supported:
            print(f"📋 Không hỗ trợ Optuna cho: {model_key}")
            continue

        study = _get_or_create_study(model_key=model_key, direction='minimize')

        n_complete = sum(1 for t in study.trials if t.state == TrialState.COMPLETE)
        remaining = max(0, OPTUNA_TRIALS - n_complete)

        if remaining <= 0:
            print(f"✅ Study đã đủ trials (complete={n_complete} / target={OPTUNA_TRIALS}) → skip optimize")
        else:
            print(
                f"📦 Optuna sqlite resume: {OPTUNA_DB_PATH} | study={study.study_name} | "
                f"complete={n_complete} → run thêm {remaining} trials"
            )
            study.optimize(
                objective_factory(model_key),
                n_trials=remaining,
                show_progress_bar=SHOW_OPTUNA_PROGRESS_BAR,
            )

        optimized_params[model_key] = dict(study.best_params)
        save_params(model_key, optimized_params[model_key])
        print(f"✅ Best CV MAE: {study.best_value:.4f}")

        # Save Optuna history chart (P0-style)
        try:
            baseline_cv_val = None
            baseline_row_name = 'SVR (RBF Kernel)' if model_key == 'SVR' else model_key
            match = baseline_df.loc[baseline_df['Model'] == baseline_row_name]
            if len(match) > 0:
                baseline_cv_val = float(match.iloc[0]['CV_MAE'])

            OPTUNA_HISTORY_IMG_DIR.mkdir(parents=True, exist_ok=True)
            out_hist = OPTUNA_HISTORY_IMG_DIR / f"{_sanitize_name_for_file(model_key)}_history.png"
            _plot_custom_optuna_history_minimize(
                study=study,
                model_name=model_key,
                baseline_value=baseline_cv_val,
                save_path=out_hist,
            )
        except Exception:
            pass

    print("\n" + "=" * 80)
    print("🎉 HOÀN TẤT TỐI ƯU HÓA")
    print("=" * 80)

    # =============================================================================
    # 5. BƯỚC 3: XÂY DỰNG & ĐÁNH GIÁ OPTIMIZED BEST MODEL
    # =============================================================================
    print("\n" + "=" * 80)
    print("🔨 BƯỚC 3: XÂY DỰNG BEST MODEL VỚI THAM SỐ TỐI ƯU")
    print("=" * 80)

    baseline_pipe = baseline_pipelines.get(best_baseline_name)
    if baseline_pipe is None:
        raise RuntimeError(f"❌ Không tìm thấy pipeline baseline cho: {best_baseline_name}")

    best_pipe = baseline_pipe
    best_tag = 'BASELINE'
    best_model_name = best_baseline_name

    def build_regressor_from_params(model_key: str, params: dict):
        if model_key == 'Random Forest':
            return RandomForestRegressor(**params, random_state=RANDOM_STATE, n_jobs=-1)
        if model_key == 'XGBoost':
            return xgb.XGBRegressor(**params, objective='reg:squarederror', random_state=RANDOM_STATE, n_jobs=-1)
        if model_key == 'LightGBM':
            return lgb.LGBMRegressor(**params, random_state=RANDOM_STATE, verbose=-1, n_jobs=-1)
        if model_key == 'SVR':
            p = dict(params)
            p.setdefault('kernel', 'rbf')
            return SVR(**p)
        return None

    # Build optimized candidate (only for the winning structure)
    optimized_candidate = None
    if 'Voting' in best_baseline_name:
        rf_p = optimized_params.get('Random Forest')
        xgb_p = optimized_params.get('XGBoost')
        lgb_p = optimized_params.get('LightGBM')
        if rf_p and xgb_p and lgb_p:
            rf_opt = build_regressor_from_params('Random Forest', rf_p)
            xgb_opt = build_regressor_from_params('XGBoost', xgb_p)
            lgb_opt = build_regressor_from_params('LightGBM', lgb_p)
            optimized_candidate = VotingRegressor([('rf', rf_opt), ('xgb', xgb_opt), ('lgbm', lgb_opt)])

    elif 'Stacking' in best_baseline_name:
        rf_p = optimized_params.get('Random Forest')
        xgb_p = optimized_params.get('XGBoost')
        svr_p = optimized_params.get('SVR')
        # Allow legacy stacking reuse where SVR params may be missing.
        if rf_p and xgb_p:
            rf_opt = build_regressor_from_params('Random Forest', rf_p)
            xgb_opt = build_regressor_from_params('XGBoost', xgb_p)
            if svr_p:
                svr_opt = build_regressor_from_params('SVR', svr_p)
            else:
                svr_opt = SVR(kernel='rbf')
            optimized_candidate = StackingRegressor(
                estimators=[('rf', rf_opt), ('xgb', xgb_opt), ('svr', svr_opt)],
                final_estimator=Ridge(),
                cv=stacking_inner_cv,
                n_jobs=None,
            )

    else:
        # Single model optimization
        opt_key = 'SVR' if best_baseline_name == 'SVR (RBF Kernel)' else best_baseline_name
        p = optimized_params.get(opt_key)
        if isinstance(p, dict) and len(p) > 0:
            if opt_key in {'Random Forest', 'XGBoost', 'LightGBM'}:
                optimized_candidate = build_regressor_from_params(opt_key, p)
            elif opt_key == 'SVR':
                optimized_candidate = build_regressor_from_params('SVR', p)

    # Evaluate optimized candidate and rollback if worse
    if optimized_candidate is not None:
        optimized_candidate_ttr = _wrap_skewed_target_regressor(optimized_candidate)
        opt_pipe = _build_pipe_p1(optimized_candidate_ttr)
        opt_pipe.fit(X_train, y_train)
        opt_pred = opt_pipe.predict(X_test)

        opt_mae = float(mean_absolute_error(y_test, opt_pred))
        opt_rmse = float(np.sqrt(mean_squared_error(y_test, opt_pred)))

        base_pred = baseline_pipe.predict(X_test)
        base_mae = float(mean_absolute_error(y_test, base_pred))
        base_rmse = float(np.sqrt(mean_squared_error(y_test, base_pred)))

        print("\n" + "=" * 80)
        print("⚖️ BƯỚC 3.3: KIỂM TRA VÀ ROLLBACK NẾU CẦN")
        print("=" * 80)
        # Leakage-safe decision: compare CV MAE on TRAIN only.
        # (TEST metrics are reported, but NOT used for rollback/model selection.)
        opt_cv_scores = cross_val_score(
            opt_pipe,
            X_train,
            y_train,
            cv=tscv_inner,
            scoring='neg_mean_absolute_error',
            n_jobs=CV_N_JOBS,
        )
        opt_cv_mae = float(-np.mean(opt_cv_scores))

        print("📊 So sánh Performance (mục tiêu: minimize CV MAE on TRAIN; Test chỉ để báo cáo):")
        print(f"   • Baseline:  {best_baseline_name} (CV MAE: {best_baseline_cv:.2f} | Test MAE: {base_mae:.2f} | RMSE: {base_rmse:.2f})")
        print(f"   • Optimized: {best_baseline_name} (CV MAE: {opt_cv_mae:.2f} | Test MAE: {opt_mae:.2f} | RMSE: {opt_rmse:.2f})")
        print(f"   • Chênh lệch CV MAE: {opt_cv_mae - best_baseline_cv:+.2f} (thấp hơn là tốt hơn)")

        if opt_cv_mae > best_baseline_cv:
            print(f"\n⚠️ PHÁT HIỆN: Optimized CV MAE ({opt_cv_mae:.2f}) > Baseline CV MAE ({best_baseline_cv:.2f})")
            print("✅ QUYẾT ĐỊNH: ROLLBACK về Baseline model")
            best_pipe = baseline_pipe
            best_tag = 'BASELINE'
        else:
            print(f"\n✅ Optuna cải thiện (CV): {best_baseline_cv:.2f} → {opt_cv_mae:.2f} ({opt_cv_mae - best_baseline_cv:+.2f})")
            print("✅ QUYẾT ĐỊNH: Giữ lại Optimized model")
            best_pipe = opt_pipe
            best_tag = 'OPTIMIZED'

    # Optional: Baseline vs Optimized chart for tuned base models (P0-style)
    try:
        comparison_rows = []
        for model_key, params in optimized_params.items():
            if not isinstance(params, dict) or len(params) == 0:
                continue

            baseline_row_name = 'SVR (RBF Kernel)' if model_key == 'SVR' else model_key
            base_match = baseline_df.loc[baseline_df['Model'] == baseline_row_name]
            if len(base_match) == 0:
                continue
            baseline_test_mae = float(base_match.iloc[0]['Test_MAE'])

            reg = build_regressor_from_params(model_key, params)
            if reg is None:
                continue
            reg_ttr = _wrap_skewed_target_regressor(reg)
            tmp_pipe = _build_pipe_p1(reg_ttr)
            tmp_pipe.fit(X_train, y_train)
            tmp_pred = tmp_pipe.predict(X_test)
            opt_test_mae = float(mean_absolute_error(y_test, tmp_pred))

            comparison_rows.append({
                'Model': model_key,
                'Baseline': baseline_test_mae,
                'Optimized': opt_test_mae,
                'Improvement': baseline_test_mae - opt_test_mae,
            })

        if comparison_rows:
            comp_df = pd.DataFrame(comparison_rows)
            out_cmp = IMG_DIR / 'p1_baseline_vs_optimized.png'
            _plot_baseline_vs_optimized_mae(df=comp_df, save_path=out_cmp)
            print(f"✅ Đã lưu biểu đồ so sánh tại: {out_cmp}")
    except Exception as e:
        print(f"⚠️  Không thể vẽ baseline_vs_optimized: {e}")

    # =============================================================================
    # 6. FINAL EVALUATION
    # =============================================================================
    print("\n" + "=" * 80)
    print("📊 BƯỚC 4: ĐÁNH GIÁ FINAL")
    print("=" * 80)

    y_pred_final = best_pipe.predict(X_test)
    final_mae = float(mean_absolute_error(y_test, y_pred_final))
    final_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_final)))

    print(f"🏆 BEST FINAL MODEL (by Test MAE): {best_model_name} [{best_tag}]")
    print(f"Test MAE : {final_mae:.2f}")
    print(f"Test RMSE: {final_rmse:.2f}")

    # Plot: Actual vs Predicted ("biểu đồ sống còn" của Regression)
    try:
        out_avp = IMG_DIR / 'p1_actual_vs_predicted.png'
        _plot_actual_vs_predicted(
            y_true=y_test,
            y_pred=y_pred_final,
            save_path=out_avp,
            model_label=f"{best_model_name} [{best_tag}] | MAE={final_mae:.2f} | RMSE={final_rmse:.2f}",
        )
    except Exception as e:
        print(f"⚠️  Không thể tạo biểu đồ Actual vs Predicted: {e}")

    # =============================================================================
    # 7. SAVE ARTIFACTS
    # =============================================================================
    print("\n" + "=" * 80)
    print("💾 ĐANG LƯU MODELS VÀ KẾT QUẢ")
    print("=" * 80)

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    shap_cache_payload = None
    if BUILD_SHAP_CACHE:
        try:
            shap_cache_payload = build_shap_cache(
                X_train,
                X_test,
                config=ShapCacheConfig(n_background=200, n_explain=200, random_state=RANDOM_STATE),
            )
        except Exception as e:
            print(f"⚠️  build_shap_cache failed → continue without SHAP cache: {e}")

    pkl_data = {
        'pkl_schema_version': 1,
        'task_id': TASK_ID,
        'data_source': str(FILE_DATA),
        'target_col': TARGET_COLUMN,
        'pipeline': best_pipe,
        'model_name': f"{best_model_name} [{best_tag}]",
        'training_config': {
            'target_transform': str(os.getenv('P1_TARGET_TRANSFORM', 'log1p')),
            'feature_selection': str(P1_FEATURE_SELECTION),
            'sfm_threshold': str(P1_SFM_THRESHOLD),
            'sfm_max_features': P1_SFM_MAX_FEATURES,
            'tree_n_estimators': int(P1_TREE_N_ESTIMATORS),
            'tree_max_depth': P1_TREE_MAX_DEPTH,
            'outlier_filter': str(P1_OUTLIER_FILTER),
            'outlier_contamination': float(P1_OUTLIER_CONTAMINATION),
            'outlier_n_neighbors': int(P1_OUTLIER_N_NEIGHBORS),
            'outlier_min_keep_frac': float(P1_OUTLIER_MIN_KEEP_FRAC),
            'enable_stacking': bool(P1_ENABLE_STACKING),
        },
        'metrics': {
            'test_mae': final_mae,
            'test_rmse': final_rmse,
        },
        'best_params': optimized_params,
        'cv_n_jobs': CV_N_JOBS,
        'optuna_storage': _optuna_storage_url(),
        'optuna_db_path': str(OPTUNA_DB_PATH),
        'shap_cache': shap_cache_payload,
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pkl_data, str(MODEL_PATH))
    try:
        joblib.dump(pkl_data, str(LEGACY_MODEL_PATH))
    except Exception:
        pass

    _save_feature_names(best_pipe, numeric_feats=numeric_feats, cat_feats=cat_feats)

    print("\n💾 FILES ĐÃ LƯU:")
    print(f"   • Best model:         {MODEL_PATH}")
    print(f"   • Legacy model:       {LEGACY_MODEL_PATH}")
    print(f"   • Feature names:      {FEATURE_NAMES_PATH}")
    print(f"   • Optuna params dir:  {OPTUNA_PARAMS_DIR}")
    print(f"   • Model comparison:   {MODEL_COMPARISON_CSV}")
    print(f"   • Model chart:        {IMG_DIR / 'p1_model_comparison_mae.png'}")
    print(f"   • Actual vs Predicted:{IMG_DIR / 'p1_actual_vs_predicted.png'}")

    print("\n" + "=" * 80)
    print("✅ HOÀN TẤT PHÂN TÍCH P1!")
    print("=" * 80)


if __name__ == '__main__':
    try:
        run_full_analysis_task_p1()
    except BrokenPipeError:
        # Common on Windows when piping output and downstream closes early.
        try:
            sys.stdout.close()
        except Exception:
            pass
        raise SystemExit(0)
