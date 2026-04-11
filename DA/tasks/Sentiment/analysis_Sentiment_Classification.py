'''
TimeSeriesSplit(spotify_release_date) for Sentiment Classification (3 Labels)
Workflow: Baseline → Optuna → Optimized → Best Model Analysis  
Target: Negative (0), Neutral (1), Positive (2)
'''
import sys
import atexit
from datetime import datetime

# --- Ensure repo root is on sys.path (so `import DA...` works when run by file path) ---
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
_DA_DIR = next((p for p in _THIS_FILE.parents if p.name == "DA"), None)
if _DA_DIR is not None:
    _REPO_ROOT = _DA_DIR.parent
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

# --- Console encoding (Windows-safe) ---
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

import optuna
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import pandas as pd
import numpy as np
import warnings

# Use a non-interactive backend to avoid Tkinter/thread teardown errors on Windows.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
import json
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel

from DA.utils.sklearn_utils import sparse_to_dense
from DA.utils.imblearn_utils import make_ratio_sampling_strategy, parse_ratios_spec
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, 
    ExtraTreesClassifier, VotingClassifier, StackingClassifier
)
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

try:
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.under_sampling import TomekLinks, RandomUnderSampler
    _HAS_IMBLEARN = True
except Exception:
    ImbPipeline = None  # type: ignore[assignment]
    TomekLinks = None  # type: ignore[assignment]
    RandomUnderSampler = None  # type: ignore[assignment]
    _HAS_IMBLEARN = False

warnings.filterwarnings('ignore')

SENTIMENT_CLASS_NAMES = ['Negative', 'Neutral', 'Positive']
SENTIMENT_CLASS_IDS = [0, 1, 2]

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

BUILD_SHAP_CACHE = (os.getenv('BUILD_SHAP_CACHE', '1') == '1') and _HAS_SHAP_ARTIFACT

FILE_DATA = 'final_data/data_prepared_for_ML.csv'

TASK_ID = 'P3'
TASK_LABEL = 'P3 - Sentiment Classification'
RANDOM_STATE = 42

FINAL_DATA_DIR = Path('DA') / 'final_data'
MODELS_DIR = Path('DA') / 'models'

TASK_DIR = Path('DA') / 'tasks' / 'Sentiment'

# User request: store ALL artifacts under the task folder.
SAVE_DIR = TASK_DIR
IMG_DIR = TASK_DIR
OPTUNA_DIR = TASK_DIR / 'optuna_history_json'
MODEL_PATH = MODELS_DIR / 'best_model_p3.pkl'
LEGACY_MODEL_PATH = MODELS_DIR / 'sentiment_model.pkl'
FEATURE_NAMES_PATH = SAVE_DIR / 'feature_names_p3.json'
MODEL_COMPARISON_CSV = FINAL_DATA_DIR / 'model_comparison_results_p3.csv'
MODEL_COMPARISON_IMG_CSV = FINAL_DATA_DIR / 'p3_model_comparison_results.csv'

# -----------------------------------------------------------------------------
# Leakage-safe feature selection & resampling (inside Pipeline/CV)
# -----------------------------------------------------------------------------
P3_FEATURE_SELECTION = os.getenv('P3_FEATURE_SELECTION', 'l1').strip().lower()
P3_RESAMPLING = os.getenv('P3_RESAMPLING', 'tomek_rus').strip().lower()
P3_RUS_STRATEGY_RAW = os.getenv('P3_RUS_STRATEGY', 'ratios:1.2,1.2,1.0').strip()
P3_SFM_C = float(os.getenv('P3_SFM_C', '0.5'))
P3_SFM_THRESHOLD = os.getenv('P3_SFM_THRESHOLD', 'median')
P3_SFM_MAX_FEATURES = int(os.getenv('P3_SFM_MAX_FEATURES', '0')) or None
P3_TREE_N_ESTIMATORS = int(os.getenv('P3_TREE_N_ESTIMATORS', '500'))
_P3_TREE_MAX_DEPTH_RAW = os.getenv('P3_TREE_MAX_DEPTH', '').strip()
P3_TREE_MAX_DEPTH = int(_P3_TREE_MAX_DEPTH_RAW) if _P3_TREE_MAX_DEPTH_RAW else None

try:
    # For multiclass, default 'auto' is the closest to "1:1" across classes.
    _tmp = float(P3_RUS_STRATEGY_RAW)
    P3_RUS_STRATEGY = _tmp
except Exception:
    P3_RUS_STRATEGY = P3_RUS_STRATEGY_RAW


def _resolve_p3_rus_strategy():
    """Return a RandomUnderSampler sampling_strategy.

    Supports controlled ratios via env:
      - P3_RUS_STRATEGY='ratios:1.2,1.2,1.0'
      - P3_RUS_STRATEGY='Negative=1.2,Neutral=1.2,Positive=1.0'
      - P3_RUS_STRATEGY='{"0":1.2,"1":1.2,"2":1.0}'
    Otherwise falls back to imblearn defaults ('auto', float, dict, ...).
    """

    raw = str(P3_RUS_STRATEGY_RAW).strip()
    ratios = None
    try:
        ratios = parse_ratios_spec(raw, class_names=SENTIMENT_CLASS_NAMES, class_ids=SENTIMENT_CLASS_IDS)
    except Exception:
        ratios = None

    if ratios is not None and len(ratios) == len(SENTIMENT_CLASS_IDS):
        return make_ratio_sampling_strategy(class_ids=SENTIMENT_CLASS_IDS, ratios=ratios)

    return P3_RUS_STRATEGY

P3_RESAMPLING_ENABLED = P3_RESAMPLING not in {'', '0', 'off', 'none', 'false'}
MODEL_CLASS_WEIGHT = None if P3_RESAMPLING_ENABLED else 'balanced'


def _save_feature_names_p3(best_pipe: Pipeline) -> None:
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        pre = best_pipe.named_steps.get('preprocessor')
        if pre is None:
            return
        names = pre.get_feature_names_out().tolist()
        with open(FEATURE_NAMES_PATH, 'w', encoding='utf-8') as f:
            json.dump(names, f, ensure_ascii=False, indent=2)
        print(f"✅ Đã lưu feature names tại: {FEATURE_NAMES_PATH}")
    except Exception as e:
        print(f"⚠️  Không thể lưu feature names: {e}")

def load_data():
    print(f"⏳ Đang tải dữ liệu từ {FILE_DATA}...")
    try:
        if not os.path.exists(FILE_DATA):
            path = f'final_data/{FILE_DATA}'
            if not os.path.exists(path):
                print(f"❌ Không tìm thấy file: {FILE_DATA}")
                return None
        else:
            path = FILE_DATA
        
        df = pd.read_csv(path)
        print(f"✅ Đã tải thành công: {len(df)} dòng dữ liệu.")
        return df
    except Exception as e:
        print(f"❌ Lỗi load data: {e}")
        return None


def _fix_and_sort_release_date(df: pd.DataFrame) -> pd.DataFrame:
    if 'spotify_release_date' not in df.columns:
        return df

    def fix_date(val):
        val = str(val).strip()
        if val == '' or val.lower() in {'nan', 'nat', 'none'}:
            return '1900-01-01'
        if len(val) == 4 and val.isdigit():
            return val + '-01-01'
        if len(val) == 7 and val[:4].isdigit() and val[4] == '-' and val[5:7].isdigit():
            return val + '-01'
        return val

    df = df.copy()
    df['spotify_release_date'] = df['spotify_release_date'].apply(fix_date)
    df['spotify_release_date'] = pd.to_datetime(df['spotify_release_date'], errors='coerce')
    df['spotify_release_date'] = df['spotify_release_date'].fillna(pd.Timestamp('1900-01-01'))
    df = df.sort_values('spotify_release_date').reset_index(drop=True)
    return df


def _map_sentiment_to_target(series: pd.Series) -> pd.Series:
    def map_sentiment(x):
        x = str(x).lower().strip()
        if 'positive' in x:
            return 2
        if 'negative' in x:
            return 0
        return 1

    return series.apply(map_sentiment)


def _build_features_p3(df: pd.DataFrame):
    if 'final_sentiment' not in df.columns:
        raise ValueError("❌ Thiếu cột label 'final_sentiment'!")

    df = df.copy()
    y = _map_sentiment_to_target(df['final_sentiment'])

    cols_ignore = {
        'spotify_track_id', 'title', 'artists', 'spotify_release_date', 'genres', #IDs
        'is_hit', 'spotify_popularity', # không liên quan đến sentiment classification
        'target', 'final_sentiment', #target leakage
        # 8. BIẾN LOẠI BỎ DO ĐA CỘNG TUYẾN
        'mfcc2_mean',           # Tương quan cực cao với rms_energy/âm lượng
        'spectral_rolloff',     # Trùng lặp thông tin với spectral_centroid
        'noun_count',           # Đã có lyric_total_words đại diện
        'verb_count',           # Đã có lyric_total_words đại diện
        'tempo_stability',        # Tương quan cao với tempo
        'spectral_contrast_band3_mean', # VIF ~2000 (Dư thừa dải tần giữa)
        'spectral_contrast_band4_mean', # VIF ~2200
        'spectral_contrast_band5_mean'  # VIF ~2000
    }

    # Prefer topic probability features (15 dims) if present.
    topic_candidates = [
        c for c in df.columns
        if isinstance(c, str) and c.startswith('topic_prob') and pd.api.types.is_numeric_dtype(df[c])
    ]
    topic_candidates = sorted(topic_candidates)
    topic_feats = topic_candidates[:15]

    candidate_numeric = [
        c for c in df.columns
        if c not in cols_ignore and c not in topic_feats and pd.api.types.is_numeric_dtype(df[c])
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

    X = df[numeric_cont_feats + binary_feats + topic_feats]

    return X, y, numeric_cont_feats, binary_feats, topic_feats


def _create_preprocessor_p3(numeric_cont_feats: list[str], binary_feats: list[str], topic_feats: list[str]):
    transformers = []
    if numeric_cont_feats:
        transformers.append(('num', StandardScaler(), numeric_cont_feats))
    if binary_feats:
        transformers.append(('bin', 'passthrough', binary_feats))
    if topic_feats:
        transformers.append(('topic', 'passthrough', topic_feats))
    return ColumnTransformer(transformers=transformers)


def _build_pipeline_p3(preprocessor, clf):
    resampling_enabled = bool(P3_RESAMPLING_ENABLED)
    if resampling_enabled and not _HAS_IMBLEARN:
        raise ImportError('P3_RESAMPLING requires imbalanced-learn (imblearn). Please install imbalanced-learn.')

    sampler_steps = []
    if resampling_enabled:
        if P3_RESAMPLING in {'tomek', 'tomek_rus', 'tomek+rus'}:
            sampler_steps.append(('tomek', TomekLinks()))
        if P3_RESAMPLING in {'rus', 'tomek_rus', 'tomek+rus'}:
            sampler_steps.append(
                (
                    'rus',
                    RandomUnderSampler(
                        sampling_strategy=_resolve_p3_rus_strategy(),
                        random_state=RANDOM_STATE,
                    ),
                )
            )

    feature_selector = None
    selector_requires_dense = False
    if P3_FEATURE_SELECTION in {'', '0', 'off', 'none', 'false'}:
        feature_selector = None
    elif P3_FEATURE_SELECTION in {'l1', 'logreg', 'logistic', 'lr'}:
        base = LogisticRegression(
            penalty='l1',
            solver='saga',
            C=float(P3_SFM_C),
            class_weight=MODEL_CLASS_WEIGHT,
            random_state=RANDOM_STATE,
            max_iter=4000,
            n_jobs=None,
        )
        feature_selector = SelectFromModel(
            estimator=base,
            threshold=P3_SFM_THRESHOLD,
            max_features=P3_SFM_MAX_FEATURES,
        )
    elif P3_FEATURE_SELECTION in {'tree', 'extratrees', 'extra_trees'}:
        base = ExtraTreesClassifier(
            n_estimators=int(P3_TREE_N_ESTIMATORS),
            max_depth=P3_TREE_MAX_DEPTH,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight=MODEL_CLASS_WEIGHT,
        )
        feature_selector = SelectFromModel(
            estimator=base,
            threshold=P3_SFM_THRESHOLD,
            max_features=P3_SFM_MAX_FEATURES,
        )
        selector_requires_dense = True
    else:
        raise ValueError(f"Invalid P3_FEATURE_SELECTION='{P3_FEATURE_SELECTION}'. Use: none|l1|tree")

    pipeline_cls = ImbPipeline if resampling_enabled else Pipeline

    steps = [('preprocessor', preprocessor)]
    if resampling_enabled or selector_requires_dense:
        steps.append(('to_dense', FunctionTransformer(sparse_to_dense, validate=False)))
    if resampling_enabled:
        steps.extend(list(sampler_steps))
    if feature_selector is not None:
        steps.append(('select', feature_selector))
    steps.append(('clf', clf))
    return pipeline_cls(steps)

def _optuna_optimize_p3(model_name: str, X_train, y_train, preprocessor, cv_splitter):
    OPTUNA_REUSE = os.getenv('OPTUNA_REUSE', '1') == '1'
    OPTUNA_DIR.mkdir(parents=True, exist_ok=True)
    safe = model_name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('+', '').replace('->', 'to')
    out_path = OPTUNA_DIR / f'p3_{safe}_best_params.json'
    if OPTUNA_REUSE and out_path.exists():
        try:
            print(f"✅ Optuna params đã tồn tại → skip trials và dùng lại: {out_path}")
            with open(out_path, 'r', encoding='utf-8') as f:
                cached = json.load(f)
            if isinstance(cached, dict) and len(cached) > 0:
                return cached
        except Exception as e:
            print(f"⚠️  Không load được Optuna params cũ → sẽ optimize lại: {e}")

    trials = int(os.getenv('OPTUNA_TRIALS', '20'))
    print(f"So luong trials: {trials}")

    def objective(trial):
        params = {}
        if model_name == 'Logistic Regression':
            params = {
                'C': trial.suggest_float('C', 0.001, 50.0, log=True),
                'solver': trial.suggest_categorical('solver', ['lbfgs', 'saga']),
            }
            clf = LogisticRegression(
                max_iter=2000,
                class_weight=MODEL_CLASS_WEIGHT,
                random_state=RANDOM_STATE,
                **params,
            )
        elif model_name == 'Random Forest':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 600, step=100),
                'max_depth': trial.suggest_int('max_depth', 5, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            }
            clf = RandomForestClassifier(
                class_weight=MODEL_CLASS_WEIGHT,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                **params,
            )
        elif model_name == 'XGBoost':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 200, 800, step=200),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            }
            clf = XGBClassifier(
                objective='multi:softprob',
                num_class=3,
                eval_metric='mlogloss',
                random_state=RANDOM_STATE,
                n_jobs=-1,
                **params,
            )
        elif model_name == 'LightGBM':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 200, 800, step=200),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 20, 120),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            }
            clf = LGBMClassifier(
                random_state=RANDOM_STATE,
                class_weight=MODEL_CLASS_WEIGHT,
                verbose=-1,
                n_jobs=-1,
                **params,
            )
        elif model_name == 'SVM':
            params = {
                'C': trial.suggest_float('C', 0.1, 100.0, log=True),
            }
            clf = SVC(
                kernel='linear',
                probability=False,
                class_weight=MODEL_CLASS_WEIGHT,
                random_state=RANDOM_STATE,
                max_iter=5000,
                **params,
            )
        elif model_name == 'AdaBoost':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500, step=50),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 1.0, log=True),
            }
            clf = AdaBoostClassifier(
                estimator=DecisionTreeClassifier(max_depth=1),
                random_state=RANDOM_STATE,
                **params,
            )
        elif model_name == 'Gradient Boosting':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 600, step=50),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 2, 5),
            }
            clf = GradientBoostingClassifier(random_state=RANDOM_STATE, **params)
        elif model_name == 'MLP (Neural Net)':
            params = {
                'hidden_layer_sizes': trial.suggest_categorical('hidden_layer_sizes', [(128, 64), (256, 128), (128, 64, 32)]),
                'alpha': trial.suggest_float('alpha', 1e-6, 1e-2, log=True),
                'learning_rate_init': trial.suggest_float('learning_rate_init', 1e-4, 5e-3, log=True),
            }
            clf = MLPClassifier(max_iter=500, random_state=RANDOM_STATE, **params)
        else:
            return -1.0

        pipe = _build_pipeline_p3(preprocessor, clf)
        scores = cross_val_score(pipe, X_train, y_train, cv=cv_splitter, scoring='f1_macro', n_jobs=-1)
        return float(np.mean(scores))

    study = optuna.create_study(direction='maximize', study_name=f'P3_{model_name}_opt')
    study.optimize(objective, n_trials=trials, show_progress_bar=True)

    IMG_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(study.best_params, f, ensure_ascii=False, indent=2)

    print(f"\nBest CV F1-macro: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
    print(f"✅ Đã lưu Optuna params tại: {out_path}")
    return study.best_params           

def _plot_cm_multiclass(cm: np.ndarray, *, class_names: list[str], title: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8.5, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title, fontsize=13, fontweight='bold')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=300, bbox_inches='tight')
    plt.close()


def _plot_per_label_ovr_cms(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    class_ids: list[int],
    class_names: list[str],
    title: str,
    out_path: Path,
) -> None:
    """Plot one-vs-rest 2x2 confusion matrices for each class in a single figure."""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    fig, axes = plt.subplots(1, len(class_ids), figsize=(5.6 * len(class_ids), 4.8))
    if len(class_ids) == 1:
        axes = [axes]

    for ax, cid, cname in zip(axes, class_ids, class_names):
        yt = (y_true == int(cid)).astype(int)
        yp = (y_pred == int(cid)).astype(int)
        cm = confusion_matrix(yt, yp, labels=[0, 1])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                    xticklabels=['Not ' + cname, cname], yticklabels=['Not ' + cname, cname])
        ax.set_title(cname, fontweight='bold')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')

    fig.suptitle(title, fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(str(out_path), dpi=300, bbox_inches='tight')
    plt.close()


def _plot_baseline_vs_optimized_metrics(
    baseline_metrics: dict,
    optimized_metrics: dict,
    *,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {'Metric': 'Accuracy', 'Baseline': float(baseline_metrics.get('accuracy', 0.0)), 'Optimized': float(optimized_metrics.get('accuracy', 0.0))},
        {'Metric': 'F1-weighted', 'Baseline': float(baseline_metrics.get('f1_weighted', 0.0)), 'Optimized': float(optimized_metrics.get('f1_weighted', 0.0))},
        {'Metric': 'F1-macro', 'Baseline': float(baseline_metrics.get('f1_macro', 0.0)), 'Optimized': float(optimized_metrics.get('f1_macro', 0.0))},
    ]
    df = pd.DataFrame(rows)
    df['Improvement'] = df['Optimized'] - df['Baseline']

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(df))
    width = 0.38
    ax.barh(x - width/2, df['Baseline'], width, label='Baseline',
            color='#87CEEB', edgecolor='white', linewidth=1.5)
    ax.barh(x + width/2, df['Optimized'], width, label='Optimized (Optuna)',
            color='#FF8C42', edgecolor='white', linewidth=1.5)

    for i, row in df.iterrows():
        ax.text(row['Baseline'] + 0.003, i - width/2, f"{row['Baseline']:.4f}", va='center', fontsize=10, fontweight='bold')
        ax.text(row['Optimized'] + 0.003, i + width/2, f"{row['Optimized']:.4f}", va='center', fontsize=10, fontweight='bold')
        if row['Improvement'] > 0:
            ax.text(max(row['Baseline'], row['Optimized']) + 0.02, i, f"↑ +{row['Improvement']*100:.2f}%",
                    va='center', color='green', fontweight='bold', fontsize=9)

    ax.set_yticks(x)
    ax.set_yticklabels(df['Metric'], fontsize=11)
    ax.set_xlabel('Score (Test)', fontsize=12, fontweight='bold')
    ax.set_title('P3 Sentiment — Baseline vs Optimized', fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='lower right', fontsize=10, frameon=True)
    ax.set_xlim(0.0, float(df[['Baseline', 'Optimized']].max().max()) + 0.10)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=300, bbox_inches='tight')
    plt.close()


def run_analysis_p3_timeholdout_no_tfidf():
    _enable_task_logging(task_dir=TASK_DIR, task_tag=TASK_ID)
    """P3 standardized like P1: no TF-IDF/lyric, time-based holdout, leakage-safe CV+Optuna+rollback."""

    print("\n" + "="*80)
    print(f"🚀 {TASK_LABEL} — Time-based holdout 80/20 | NO TF-IDF/lyric")
    print("="*80)
    print(f"   • Data: {FILE_DATA}")
    print("   • Target: final_sentiment (3 labels)")

    df = load_data()
    if df is None:
        return

    df = _fix_and_sort_release_date(df)
    X, y, numeric_cont_feats, binary_feats, topic_feats = _build_features_p3(df)

    # Sanity check: master data nên không còn missing.
    if X.isna().any().any() or y.isna().any():
        na_cols = X.columns[X.isna().any()].tolist()
        raise ValueError(
            "❌ Dữ liệu đầu vào vẫn còn missing values. "
            "Hãy chạy lại scripts/data_prepared_for_ML.py để fill median trước khi train. "
            f"Cột bị thiếu: {na_cols[:20]}" + (" ..." if len(na_cols) > 20 else "")
        )

    print("\n" + "="*60)
    print("🔍 KIỂM TRA CHI TIẾT CÁC BIẾN ĐẦU VÀO (TASK 3 - SENTIMENT)")
    print("="*60)
    print(f"1️⃣ Số lượng biến số (Continuous - sẽ Scale): {len(numeric_cont_feats)}")
    print(np.array(numeric_cont_feats))
    print("-" * 60)
    print(f"1️⃣b Số lượng biến nhị phân (0/1 - KHÔNG Scale): {len(binary_feats)}")
    print(np.array(binary_feats))
    print("-" * 60)
    print(f"1️⃣c Số lượng biến topic_prob (Passthrough): {len(topic_feats)}")
    if topic_feats:
        print(np.array(topic_feats))
    print("-" * 60)
    print(f"1️⃣d Tổng số biến số dùng cho model: {len(numeric_cont_feats) + len(binary_feats) + len(topic_feats)}")
    print("-" * 60)
    print("2️⃣ Biến phân loại: 0 (P3 không dùng OHE; final_sentiment là target)")
    print("-" * 60)
    print(f"✅ TỔNG CỘNG SỐ BIẾN ĐƯA VÀO MODEL: {len(numeric_cont_feats) + len(binary_feats) + len(topic_feats)}")
    print("="*60 + "\n")

    # Holdout 80/20 by time (no shuffle)
    n_total = len(df)
    if n_total < 10:
        raise ValueError('Dataset quá nhỏ để split 80/20 theo thời gian')
    split_point = int(n_total * 0.8)
    X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
    y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]

    print("\n" + "="*80)
    print("🧪 BƯỚC 1: CHIA DỮ LIỆU (Time-based Holdout 80/20)")
    print("="*80)
    print(f"📌 Holdout 80/20 (time-based) → Train={len(X_train)} | Test={len(X_test)}")
    if 'spotify_release_date' in df.columns:
        print("-" * 60)
        print(f"📅 Tập Train: từ {df.iloc[:split_point]['spotify_release_date'].min().date()} đến {df.iloc[:split_point]['spotify_release_date'].max().date()} ({len(X_train)} bài)")
        print(f"📅 Tập Test : từ {df.iloc[split_point:]['spotify_release_date'].min().date()} đến {df.iloc[split_point:]['spotify_release_date'].max().date()} ({len(X_test)} bài)")
        print("-" * 60)

    # Label distribution (mirrors Hit task style)
    labels_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    train_counts = y_train.value_counts().to_dict()
    test_counts = y_test.value_counts().to_dict()
    print("📊 Phân bố nhãn theo split:")
    print(
        "   • Train: "
        + ", ".join([f"{labels_map.get(k, k)}={int(train_counts.get(k, 0))}" for k in [0, 1, 2]])
    )
    print(
        "   • Test : "
        + ", ".join([f"{labels_map.get(k, k)}={int(test_counts.get(k, 0))}" for k in [0, 1, 2]])
    )

    if P3_RESAMPLING_ENABLED:
        print(
            f"⚖️  Resampling (TRAIN/CV only, leakage-safe via Pipeline): {P3_RESAMPLING} | "
            f"RUS strategy={P3_RUS_STRATEGY_RAW} | class_weight disabled to avoid double-balancing"
        )
        _raw = str(P3_RUS_STRATEGY_RAW).strip().lower()
        if _raw.startswith(('ratio:', 'ratios:')) or ('=' in _raw) or (_raw.startswith('{') and _raw.endswith('}')):
            print("   ℹ️  Controlled RUS: target counts computed per CV fold from class prevalence")
    else:
        print("⚙️  Không dùng undersample/oversample; model tự xử lý imbalance bằng class_weight (nếu hỗ trợ)")

    preprocessor = _create_preprocessor_p3(numeric_cont_feats, binary_feats, topic_feats)
    tscv_inner = TimeSeriesSplit(n_splits=5)

    # Stacking must also be time-aware to avoid future leakage.
    stacking_inner_cv = TimeSeriesSplit(n_splits=5)

    print("\n" + "="*80)
    print("🏁 BƯỚC 2: BASELINE MODELS (CV on TRAIN only)")
    print("="*80)

    rf = RandomForestClassifier(n_estimators=300, class_weight=MODEL_CLASS_WEIGHT, random_state=RANDOM_STATE, n_jobs=-1)
    xgb_clf = XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        objective='multi:softprob',
        num_class=3,
        eval_metric='mlogloss',
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    lgbm_clf = LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        random_state=RANDOM_STATE,
        class_weight=MODEL_CLASS_WEIGHT,
        verbose=-1,
        n_jobs=-1,
    )
    # RBF+probability is extremely slow on this dataset; use a fast linear SVM.
    svm = SVC(
        kernel='linear',
        probability=False,
        class_weight=MODEL_CLASS_WEIGHT,
        random_state=RANDOM_STATE,
        max_iter=5000,
    )

    baseline_models = {
        'Logistic Regression': LogisticRegression(max_iter=2000, class_weight=MODEL_CLASS_WEIGHT, random_state=RANDOM_STATE),
        'Random Forest': rf,
        'Extra Trees': ExtraTreesClassifier(n_estimators=400, class_weight=MODEL_CLASS_WEIGHT, random_state=RANDOM_STATE, n_jobs=-1),
        'SVM': svm,
        'AdaBoost': AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1, class_weight=MODEL_CLASS_WEIGHT), n_estimators=300, learning_rate=0.05, random_state=RANDOM_STATE),
        'Gradient Boosting': GradientBoostingClassifier(random_state=RANDOM_STATE),
        'MLP (Neural Net)': MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-4, learning_rate_init=1e-3, max_iter=500, random_state=RANDOM_STATE),
        'XGBoost': xgb_clf,
        'LightGBM': lgbm_clf,
    }

    baseline_models['Voting (LGBM+RF+XGB) Soft Voting'] = VotingClassifier(
        estimators=[('lgbm', lgbm_clf), ('rf', rf), ('xgb', xgb_clf)],
        voting='soft',
    )

    baseline_models['Stacking (RF+XGB+SVM -> LR)'] = StackingClassifier(
        estimators=[('rf', rf), ('xgb', xgb_clf), ('svm', svm)],
        final_estimator=LogisticRegression(max_iter=2000, class_weight=MODEL_CLASS_WEIGHT, random_state=RANDOM_STATE),
        cv=stacking_inner_cv,
        n_jobs=None,
        passthrough=False,
    )

    rows = []
    pipelines = {}
    print(f"\n{'MODEL':<34} | {'CV_F1M':<8} | {'TEST_ACC':<8} | {'TEST_F1M':<8} | {'TEST_F1W':<8}")
    print("-" * 70)
    for name, clf in baseline_models.items():
        try:
            pipe = _build_pipeline_p3(preprocessor, clf)
            cv_scores = cross_val_score(pipe, X_train, y_train, cv=tscv_inner, scoring='f1_macro', n_jobs=-1)
            cv_f1m = float(np.mean(cv_scores))
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            test_acc = float(accuracy_score(y_test, y_pred))
            test_f1m = float(f1_score(y_test, y_pred, average='macro'))
            test_f1w = float(f1_score(y_test, y_pred, average='weighted'))
            rows.append({'Model': name, 'CV_F1_Macro': cv_f1m, 'Test_Accuracy': test_acc, 'Test_F1_Macro': test_f1m, 'Test_F1_Weighted': test_f1w})
            pipelines[name] = pipe
            print(f"{name:<34} | {cv_f1m:.4f}  | {test_acc:.4f}  | {test_f1m:.4f}  | {test_f1w:.4f}")
        except Exception as e:
            print(f"❌ {name}: {e}")

    results_df = pd.DataFrame(rows).sort_values(by='CV_F1_Macro', ascending=False)
    best_baseline = results_df.iloc[0]
    best_name = str(best_baseline['Model'])
    print(
        f"\n🏆 Best Baseline (by CV F1-macro): {best_name} "
        f"(CV F1-macro={best_baseline['CV_F1_Macro']:.4f} | Test F1-macro={best_baseline['Test_F1_Macro']:.4f})"
    )

    # --- Baseline CM (best baseline on Test) ---
    baseline_pipe = pipelines.get(best_name)
    baseline_metrics = {'accuracy': 0.0, 'f1_macro': 0.0, 'f1_weighted': 0.0}
    if baseline_pipe is not None:
        try:
            y_pred_baseline = baseline_pipe.predict(X_test)
            baseline_metrics = {
                'accuracy': float(accuracy_score(y_test, y_pred_baseline)),
                'f1_macro': float(f1_score(y_test, y_pred_baseline, average='macro', zero_division=0)),
                'f1_weighted': float(f1_score(y_test, y_pred_baseline, average='weighted', zero_division=0)),
            }
            cm_base = confusion_matrix(y_test, y_pred_baseline, labels=SENTIMENT_CLASS_IDS)
            out_cm_base = TASK_DIR / 'p3_confusion_matrix_baseline.png'
            _plot_cm_multiclass(
                cm_base,
                class_names=SENTIMENT_CLASS_NAMES,
                title=f"Confusion Matrix — Baseline ({best_name})\nAcc={baseline_metrics['accuracy']:.4f} | F1m={baseline_metrics['f1_macro']:.4f}",
                out_path=out_cm_base,
            )
            print(f"✅ Đã lưu Confusion Matrix baseline tại: {out_cm_base}")
        except Exception as e:
            print(f"⚠️  Không thể vẽ CM baseline: {e}")

    # Optuna
    print("\n" + "="*80)
    print(f"🧠 BƯỚC 3: OPTUNA (maximize CV F1-macro) — {best_name}")
    print("="*80)

    optimized_params: dict[str, dict] = {}

    if 'Voting' in best_name:
        models_to_optimize = ['Random Forest', 'XGBoost', 'LightGBM']
        print(f"📋 Ensemble model detected → Sẽ tối ưu 3 base models: {', '.join(models_to_optimize)}")
    elif 'Stacking' in best_name:
        models_to_optimize = ['Random Forest', 'XGBoost', 'SVM']
        print(f"📋 Ensemble model detected → Sẽ tối ưu 3 base models: {', '.join(models_to_optimize)}")
    else:
        models_to_optimize = [best_name]
        print(f"📋 Single model detected → Sẽ tối ưu: {best_name}")

    for model_key in models_to_optimize:
        optimized_params[model_key] = _optuna_optimize_p3(model_key, X_train, y_train, preprocessor, tscv_inner)

    # Train optimized + rollback
    final_pipe = pipelines[best_name]
    final_model_name = best_name

    def _build_clf_from_params(model_key: str, params: dict):
        if model_key == 'Logistic Regression':
            return LogisticRegression(max_iter=2000, class_weight=MODEL_CLASS_WEIGHT, random_state=RANDOM_STATE, **params)
        if model_key == 'Random Forest':
            return RandomForestClassifier(class_weight=MODEL_CLASS_WEIGHT, random_state=RANDOM_STATE, n_jobs=-1, **params)
        if model_key == 'XGBoost':
            return XGBClassifier(objective='multi:softprob', num_class=3, eval_metric='mlogloss', random_state=RANDOM_STATE, n_jobs=-1, **params)
        if model_key == 'LightGBM':
            return LGBMClassifier(random_state=RANDOM_STATE, class_weight=MODEL_CLASS_WEIGHT, verbose=-1, n_jobs=-1, **params)
        if model_key == 'SVM':
            p = dict(params)
            p.setdefault('kernel', 'linear')
            p.setdefault('probability', False)
            p.setdefault('class_weight', MODEL_CLASS_WEIGHT)
            p.setdefault('max_iter', 5000)
            return SVC(random_state=RANDOM_STATE, **p)
        if model_key == 'AdaBoost':
            return AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1), random_state=RANDOM_STATE, **params)
        if model_key == 'Gradient Boosting':
            return GradientBoostingClassifier(random_state=RANDOM_STATE, **params)
        if model_key == 'MLP (Neural Net)':
            return MLPClassifier(max_iter=500, random_state=RANDOM_STATE, **params)
        return None

    optimized_candidate = None
    if 'Voting' in best_name:
        rf_p = optimized_params.get('Random Forest')
        xgb_p = optimized_params.get('XGBoost')
        lgb_p = optimized_params.get('LightGBM')
        if isinstance(rf_p, dict) and isinstance(xgb_p, dict) and isinstance(lgb_p, dict):
            rf_opt = _build_clf_from_params('Random Forest', rf_p)
            xgb_opt = _build_clf_from_params('XGBoost', xgb_p)
            lgb_opt = _build_clf_from_params('LightGBM', lgb_p)
            if rf_opt is not None and xgb_opt is not None and lgb_opt is not None:
                optimized_candidate = VotingClassifier(
                    estimators=[('lgbm', lgb_opt), ('rf', rf_opt), ('xgb', xgb_opt)],
                    voting='soft',
                )

    elif 'Stacking' in best_name:
        rf_p = optimized_params.get('Random Forest')
        xgb_p = optimized_params.get('XGBoost')
        svm_p = optimized_params.get('SVM')
        if isinstance(rf_p, dict) and isinstance(xgb_p, dict):
            rf_opt = _build_clf_from_params('Random Forest', rf_p)
            xgb_opt = _build_clf_from_params('XGBoost', xgb_p)
            svm_opt = _build_clf_from_params('SVM', svm_p) if isinstance(svm_p, dict) else svm
            if rf_opt is not None and xgb_opt is not None and svm_opt is not None:
                optimized_candidate = StackingClassifier(
                    estimators=[('rf', rf_opt), ('xgb', xgb_opt), ('svm', svm_opt)],
                    final_estimator=LogisticRegression(max_iter=2000, class_weight=MODEL_CLASS_WEIGHT, random_state=RANDOM_STATE),
                    cv=stacking_inner_cv,
                    n_jobs=None,
                    passthrough=False,
                )

    else:
        p = optimized_params.get(best_name)
        if isinstance(p, dict) and len(p) > 0:
            optimized_candidate = _build_clf_from_params(best_name, p)

    if optimized_candidate is not None:
        opt_pipe = _build_pipeline_p3(preprocessor, optimized_candidate)
        # Leakage-safe decision: compare CV (TRAIN only), not Test.
        base_cv_f1m = float(best_baseline['CV_F1_Macro'])
        opt_cv_scores = cross_val_score(opt_pipe, X_train, y_train, cv=tscv_inner, scoring='f1_macro', n_jobs=-1)
        opt_cv_f1m = float(np.mean(opt_cv_scores))

        # Fit once for reporting on TEST
        opt_pipe.fit(X_train, y_train)
        y_pred_opt = opt_pipe.predict(X_test)
        opt_test_f1m = float(f1_score(y_test, y_pred_opt, average='macro', zero_division=0))
        base_test_f1m = float(best_baseline['Test_F1_Macro'])

        print("\n" + "="*80)
        print("⚖️ BƯỚC 3.3: KIỂM TRA VÀ ROLLBACK NẾU CẦN")
        print("="*80)
        print(f"   • Baseline CV F1-macro (TRAIN):  {base_cv_f1m:.4f}")
        print(f"   • Optimized CV F1-macro (TRAIN): {opt_cv_f1m:.4f}")
        print(f"   • Baseline Test F1-macro (report):  {base_test_f1m:.4f}")
        print(f"   • Optimized Test F1-macro (report): {opt_test_f1m:.4f}")
        if opt_cv_f1m < base_cv_f1m:
            print(f"\n⚠️ Optimized CV F1-macro ({opt_cv_f1m:.4f}) < Baseline ({base_cv_f1m:.4f})")
            print("✅ QUYẾT ĐỊNH: ROLLBACK về baseline pipeline")
        else:
            final_pipe = opt_pipe
            final_model_name = best_name + ' (OPTIMIZED)'
            print(f"\n✅ Optuna cải thiện (CV): {base_cv_f1m:.4f} → {opt_cv_f1m:.4f}")

    # Final evaluation: dùng predict/argmax mặc định của scikit-learn
    y_pred_final = final_pipe.predict(X_test)
    final_metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred_final)),
        'f1_macro': float(f1_score(y_test, y_pred_final, average='macro', zero_division=0)),
        'f1_weighted': float(f1_score(y_test, y_pred_final, average='weighted', zero_division=0)),
    }
    print("\n" + "="*80)
    print("📌 BƯỚC 4: ĐÁNH GIÁ FINAL")
    print("="*80)
    print(f"Model: {final_model_name}")

    print(f"Test Accuracy : {final_metrics['accuracy']:.4f}")
    print(f"Test F1-macro : {final_metrics['f1_macro']:.4f}")
    print(f"Test F1-weight: {final_metrics['f1_weighted']:.4f}")

    # Save artifacts
    print("\n" + "="*80)
    print("💾 BƯỚC 5: LƯU ARTIFACTS")
    print("="*80)
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    pkl_data = {
        'pkl_schema_version': 1,
        'task_id': TASK_ID,
        'data_source': str(Path(FILE_DATA)),
        'pipeline': final_pipe,
        'model_name': final_model_name,
        'training_config': {
            'feature_selection': str(P3_FEATURE_SELECTION),
            'resampling': str(P3_RESAMPLING),
            'rus_strategy': str(P3_RUS_STRATEGY_RAW),
            'model_class_weight': MODEL_CLASS_WEIGHT,
        },
        'metrics': {
            'test_accuracy': float(final_metrics['accuracy']),
            'test_f1_macro': float(final_metrics['f1_macro']),
            'test_f1_weighted': float(final_metrics['f1_weighted']),
        },
        'best_params': optimized_params,
        'shap_cache': None,
    }

    # --- Confusion matrices (per user request) ---
    try:
        # Per-label (OvR)
        out_ovr_before = TASK_DIR / 'p3_cm_per_label.png'
        _plot_per_label_ovr_cms(
            np.asarray(y_test, dtype=int),
            np.asarray(y_pred_final, dtype=int),
            class_ids=SENTIMENT_CLASS_IDS,
            class_names=SENTIMENT_CLASS_NAMES,
            title=f"Per-label CM (OvR)\n{final_model_name}",
            out_path=out_ovr_before,
        )
        print(f"✅ Đã lưu CM theo từng nhãn tại: {out_ovr_before}")
    except Exception as e:
        print(f"⚠️  Không thể vẽ CM theo từng nhãn: {e}")

    try:
        # Baseline vs optimized plot (like Hit)
        out_cmp = TASK_DIR / 'p3_baseline_vs_optimized.png'
        _plot_baseline_vs_optimized_metrics(
            baseline_metrics,
            final_metrics,
            out_path=out_cmp,
        )
        print(f"✅ Đã lưu baseline_vs_optimized tại: {out_cmp}")
    except Exception as e:
        print(f"⚠️  Không thể vẽ baseline_vs_optimized: {e}")

    if BUILD_SHAP_CACHE:
        try:
            pkl_data['shap_cache'] = build_shap_cache(
                X_train,
                X_test,
                config=ShapCacheConfig(n_background=200, n_explain=200, random_state=RANDOM_STATE),
            )
        except Exception as e:
            print(f"⚠️  build_shap_cache failed → continue without SHAP cache: {e}")

    joblib.dump(pkl_data, str(MODEL_PATH))
    try:
        joblib.dump(pkl_data, str(LEGACY_MODEL_PATH))
    except Exception:
        pass
    _save_feature_names_p3(final_pipe)

    # Save model comparison CSVs
    results_df.to_csv(str(MODEL_COMPARISON_IMG_CSV), index=False)
    results_df.to_csv(str(MODEL_COMPARISON_CSV), index=False)

    # Simple comparison plot (Test F1-macro)
    try:
        df_plot = results_df.sort_values(by='Test_F1_Macro', ascending=False).reset_index(drop=True)

        fig, ax = plt.subplots(figsize=(16, max(8, int(len(df_plot) * 1.2))))
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(df_plot)))
        ax.barh(df_plot['Model'], df_plot['Test_F1_Macro'], color=colors, edgecolor='white', linewidth=1.5)

        for i, row in df_plot.iterrows():
            val = float(row['Test_F1_Macro'])
            ax.text(val + 0.005, i, f"{val:.4f}", va='center', fontsize=11, fontweight='bold', color='black')

        ax.set_xlabel('Test F1-macro (higher is better)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Model', fontsize=13, fontweight='bold')
        ax.set_title('P3 Sentiment — Model Comparison (Test F1-macro)', fontsize=16, fontweight='bold', pad=20)
        ax.invert_yaxis()
        ax.set_xlim(0.0, float(df_plot['Test_F1_Macro'].max()) + 0.10)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()
        out_plot = IMG_DIR / 'p3_model_comparison_f1.png'
        plt.savefig(str(out_plot), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Đã lưu biểu đồ tại: {out_plot}")
    except Exception as e:
        print(f"⚠️  Không thể lưu biểu đồ so sánh: {e}")

    print(f"\n💾 FILES ĐÃ LƯU:")
    print(f"   • Best model:         {MODEL_PATH}")
    print(f"   • Legacy model alias: {LEGACY_MODEL_PATH}")
    print(f"   • Feature names:      {FEATURE_NAMES_PATH}")
    print(f"   • Model comparison (img): {MODEL_COMPARISON_IMG_CSV}")
    print(f"   • Model comparison (pkl): {MODEL_COMPARISON_CSV}")
    print(f"   • Optuna params (pkl): {OPTUNA_DIR}/p3_*_best_params.json")
    print(f"   • Plot:               {IMG_DIR}/p3_model_comparison_f1.png")


# NOTE: The legacy TF-IDF/lyric-based workflow (and its helper functions) has
# been moved into `analysis_Sentiment_Classification_legacy_tfidf.py` to keep
# this file focused on the leakage-safe, no-lyric pipeline.

if __name__ == "__main__":
    run_analysis_p3_timeholdout_no_tfidf()