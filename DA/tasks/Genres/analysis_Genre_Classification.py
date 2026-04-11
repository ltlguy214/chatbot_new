'''
Genre Classification (P4) — Multi-label (4 labels - Ballad, Indie, Pop, Rap/Hip-hop)
Workflow (aligned with P1/P3): Time-based Holdout (80/20) → Baseline → Optuna (CV on TRAIN) → Rollback → Report

Features (NO TF-IDF / lyric text):
  - Continuous numeric (scaled)
  - Binary numeric (passthrough)
  - topic_prob_0..topic_prob_14 (passthrough)
  - final_sentiment (OneHotEncoder → 3 dims)

CV & Optimization:
  - TimeSeriesSplit CV only on TRAIN
    - Optuna maximizes F1-macro

  best model = F1-macro
  Tối ưu ngưỡng theo từng genre (trên validation trong TRAIN để tránh leakage)

PKL kèm thresholds
'''

import sys

# --- Ensure repo root is on sys.path (so `import DA...` works when run by file path) ---
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
_DA_DIR = next((p for p in _THIS_FILE.parents if p.name == "DA"), None)
if _DA_DIR is not None:
    _REPO_ROOT = _DA_DIR.parent
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

# When this file is executed as a script, its module name is '__main__'.
# We alias it to its canonical import path so joblib/pickle can save/load
# objects that reference classes defined in this file.
_CANONICAL_MODULE = 'DA.tasks.Genres.analysis_Genre_Classification'
if __name__ == '__main__':
    sys.modules[_CANONICAL_MODULE] = sys.modules[__name__]

# --- Console encoding (Windows-safe) ---
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

import json
import os
import warnings
import atexit
from datetime import datetime

import joblib
import numpy as np
import optuna
import pandas as pd

# Use a non-interactive backend to avoid Tkinter/thread teardown errors on Windows.
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from DA.models.topic_mapping import rename_topics_in_feature_names

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    hamming_loss,
    jaccard_score,
    make_scorer,
    multilabel_confusion_matrix,
)
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.feature_selection import SelectFromModel
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight

from DA.utils.sklearn_utils import sparse_to_dense

from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

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


class PerLabelMultiOutputClassifier(BaseEstimator, ClassifierMixin):
    """Multi-label wrapper with optional per-label imbalance weighting.

    If `USE_PER_LABEL_WEIGHTS=1`:
    - For estimators supporting `class_weight`, sets a per-label `{0: w0, 1: w1}`.
      If `POS_WEIGHT_MULTIPLIER>1`, increases the positive-class weight.
    - Otherwise, uses per-label balanced `sample_weight` (and can upweight positives).

    This is computed inside each `fit`, so it is safe with time-based CV.
    """

    def __init__(
        self,
        estimator,
        *,
        pos_weight_multiplier=1.0,
        rare_pos_rate_threshold: float = 0.35,
        use_per_label_weights: bool = True,
    ):
        self.estimator = estimator
        # Can be either:
        # - float/int: same multiplier applied to any "rare" label
        # - sequence of floats (len = n_labels): per-label multipliers
        self.pos_weight_multiplier = pos_weight_multiplier
        self.rare_pos_rate_threshold = float(rare_pos_rate_threshold)
        self.use_per_label_weights = bool(use_per_label_weights)

    def _multiplier_for_label(self, j: int) -> float:
        m = self.pos_weight_multiplier
        if isinstance(m, (list, tuple, np.ndarray)):
            try:
                return float(m[j])
            except Exception:
                return 1.0
        try:
            return float(m)
        except Exception:
            return 1.0

    def fit(self, X, y, sample_weight=None):
        y = np.asarray(y)
        if y.ndim != 2:
            raise ValueError('Expected y as 2D array for multi-label classification')

        self.estimators_ = []
        base_sw = None if sample_weight is None else np.asarray(sample_weight, dtype=float)

        for j in range(y.shape[1]):
            yj = np.asarray(y[:, j], dtype=int)
            # Some CV folds / resampled folds can end up with only one class for a label.
            # In that case, fit a constant predictor to avoid hard crashes in estimators.
            try:
                uniq = np.unique(yj)
            except Exception:
                uniq = None
            if uniq is not None and getattr(uniq, 'size', 0) < 2:
                from sklearn.dummy import DummyClassifier

                constant = int(uniq[0]) if getattr(uniq, 'size', 0) == 1 else 0
                dummy = DummyClassifier(strategy='constant', constant=constant)
                dummy.fit(X, yj)
                self.estimators_.append(dummy)
                continue

            pos_rate = float(np.mean(yj)) if yj.size else 0.0
            mult = float(self._multiplier_for_label(j))
            # Keep the original "rare-label gating" behavior to avoid blowing up FP on common labels.
            if mult > 1.0 and pos_rate > float(self.rare_pos_rate_threshold):
                mult = 1.0
            est = clone(self.estimator)

            est_params = {}
            try:
                est_params = est.get_params(deep=False)
            except Exception:
                est_params = {}

            used_class_weight = False
            if self.use_per_label_weights and 'class_weight' in est_params:
                try:
                    cw = compute_class_weight(class_weight='balanced', classes=np.array([0, 1]), y=yj)
                    w0, w1 = float(cw[0]), float(cw[1])
                    if mult > 1.0:
                        w1 *= mult
                    est.set_params(class_weight={0: w0, 1: w1})
                    used_class_weight = True
                except Exception:
                    used_class_weight = False

            # Does estimator.fit accept sample_weight?
            fit_sig_ok = False
            try:
                import inspect

                fit_sig_ok = 'sample_weight' in inspect.signature(est.fit).parameters
            except Exception:
                fit_sig_ok = False

            if fit_sig_ok:
                sw = base_sw
                if (not used_class_weight) and self.use_per_label_weights:
                    try:
                        sw_lbl = compute_sample_weight(class_weight='balanced', y=yj).astype(float)
                        if mult > 1.0:
                            sw_lbl[yj == 1] *= mult
                        m = float(np.mean(sw_lbl))
                        if m > 0:
                            sw_lbl /= m
                        sw = sw_lbl if sw is None else (sw * sw_lbl)
                    except Exception:
                        pass

                if sw is None:
                    est.fit(X, yj)
                else:
                    est.fit(X, yj, sample_weight=sw)
            else:
                est.fit(X, yj)

            self.estimators_.append(est)

        return self

    def predict(self, X):
        preds = [np.asarray(est.predict(X)).reshape(-1) for est in self.estimators_]
        return np.column_stack(preds).astype(int)

    def predict_proba(self, X):
        probas = []
        for est in self.estimators_:
            if not hasattr(est, 'predict_proba'):
                raise AttributeError('Base estimator does not support predict_proba')
            p = np.asarray(est.predict_proba(X))
            # Normalize binary probabilities to shape (n_samples, 2) even for constant/degenerate labels.
            if p.ndim == 2 and p.shape[1] == 1:
                classes = getattr(est, 'classes_', None)
                out = np.zeros((p.shape[0], 2), dtype=float)
                if classes is not None:
                    try:
                        cls = int(np.asarray(classes).reshape(-1)[0])
                    except Exception:
                        cls = 0
                else:
                    cls = 0
                if cls == 0:
                    out[:, 0] = p[:, 0]
                    out[:, 1] = 0.0
                else:
                    out[:, 0] = 0.0
                    out[:, 1] = p[:, 0]
                p = out
            probas.append(p)
        return probas


# Make the class importable for pickle/joblib even when this file is executed as a script.
# Without this, joblib may store it as '__main__.PerLabelMultiOutputClassifier' and later loads fail.
PerLabelMultiOutputClassifier.__module__ = _CANONICAL_MODULE


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

TASK_ID = 'P4'
TASK_LABEL = 'P4 - Genre Classification (Multi-label)'
RANDOM_STATE = 42

DATA_FILE = 'final_data/data_prepared_for_ML.csv'

LABEL_COLS = [
    'genre_Ballad',
    'genre_Indie',
    'genre_Pop',
    'genre_Rap/Hip-hop',
]

CAT_FEATS = ['final_sentiment']
SENTIMENT_CATEGORIES = ['Negative', 'Neutral', 'Positive']

FINAL_DATA_DIR = Path('DA') / 'final_data'
MODELS_DIR = Path('DA') / 'models'

TASK_DIR = Path('DA') / 'tasks' / 'Genres'

# Optional: penalize false negatives more for rare genres.
# Example: set POS_WEIGHT_MULTIPLIER=2.0 to upweight positive samples.
# Optional: penalize false negatives more for rare genres.
# [Ballad, Indie, Pop, Rap] -> Ép Indie x4.0 lần trọng số, Rap x1.5
POS_WEIGHT_MULTIPLIER = [1.0, 4.0, 1.0, 1.5]
POS_WEIGHT_RARE_THRESHOLD = 0.35
USE_PER_LABEL_WEIGHTS = True

# Tắt tự động để dùng mảng tỷ lệ cứng ở trên
AUTO_POS_WEIGHT = False

# Stronger imbalance option: compute per-label multipliers automatically from TRAIN label prevalence.
# This still respects POS_WEIGHT_RARE_THRESHOLD as a gating threshold.
AUTO_POS_WEIGHT_ALPHA = float(os.getenv('AUTO_POS_WEIGHT_ALPHA', '0.7'))
AUTO_POS_WEIGHT_MAX = float(os.getenv('AUTO_POS_WEIGHT_MAX', '6.0'))

# Stronger imbalance option: reduce dominant label counts by downsampling rows that have ONLY that label.
DOWNSAMPLE_DOMINANT_SINGLES = os.getenv('DOWNSAMPLE_DOMINANT_SINGLES', '0') == '1'
DOWNSAMPLE_LABEL_POS_RATE_MIN = float(os.getenv('DOWNSAMPLE_LABEL_POS_RATE_MIN', '0.50'))
DOWNSAMPLE_TARGET_POS_RATE = float(os.getenv('DOWNSAMPLE_TARGET_POS_RATE', '0.40'))
DOWNSAMPLE_MIN_KEEP_FRAC = float(os.getenv('DOWNSAMPLE_MIN_KEEP_FRAC', '0.20'))

# Leakage-safe feature selection & resampling (inside Pipeline/CV)
P4_FEATURE_SELECTION = os.getenv('P4_FEATURE_SELECTION', 'tree').strip().lower()

# Multi-label resampling via TomekLinks/RUS is not well-defined; we instead support
# fold-safe downsampling of dominant single-label rows.
#
# Additionally (user request): support a controlled TomekLinks + RUS workflow by
# restricting training rows to *single-label* samples (drop hybrids like Pop-Ballad),
# then applying TomekLinks + RandomUnderSampler on the derived single-label target.
# This runs inside each fit/CV fold via an imblearn-compatible sampler to avoid leakage.
P4_RESAMPLING = os.getenv('P4_RESAMPLING', '').strip().lower()
if not P4_RESAMPLING:
    # Default to the user-requested boundary cleaning + controlled undersampling.
    P4_RESAMPLING = 'tomek_rus_controlled'

# If user wants resampling but imblearn isn't available, safely disable it.
if (P4_RESAMPLING not in {'', '0', 'off', 'none', 'false'}) and (not _HAS_IMBLEARN):
    print(
        "⚠️  P4_RESAMPLING is enabled but missing package `imbalanced-learn`. "
        "Cài bằng: pip install imbalanced-learn. Tạm thời chạy KHÔNG resampling.",
        flush=True,
    )
    P4_RESAMPLING = 'none'
P4_RESAMPLING_ENABLED = P4_RESAMPLING not in {'', '0', 'off', 'none', 'false'}

# Match P3 behavior: when we do explicit resampling inside the Pipeline, avoid
# double-compensating imbalance via model-level class_weight.
MODEL_CLASS_WEIGHT = None if P4_RESAMPLING_ENABLED else 'balanced'

# Controlled Tomek+RUS parameters (only used when P4_RESAMPLING in {'tomek_rus_controlled', 'tomek+rus_controlled'}).
P4_DROP_NON_SINGLE_LABEL = False

P4_RUS_POP_MAX_MULT = float(os.getenv('P4_RUS_POP_MAX_MULT', '2.0'))
P4_RUS_REF_LABELS_RAW = os.getenv('P4_RUS_REF_LABELS', 'genre_Rap/Hip-hop,genre_Indie').strip()
P4_RUS_REF_LABELS = [s.strip() for s in P4_RUS_REF_LABELS_RAW.split(',') if s.strip()]

# Giảm độ gắt khi lọc biến để giữ lại thông tin cho Indie
P4_SFM_THRESHOLD = '0.5*mean'
P4_SFM_MAX_FEATURES = int(os.getenv('P4_SFM_MAX_FEATURES', '0')) or None

P4_TREE_N_ESTIMATORS = int(os.getenv('P4_TREE_N_ESTIMATORS', '500'))
_P4_TREE_MAX_DEPTH_RAW = os.getenv('P4_TREE_MAX_DEPTH', '').strip()
P4_TREE_MAX_DEPTH = int(_P4_TREE_MAX_DEPTH_RAW) if _P4_TREE_MAX_DEPTH_RAW else None

# User request: store ALL artifacts under the task folder.
SAVE_DIR = TASK_DIR
IMG_DIR = TASK_DIR
OPTUNA_DIR = IMG_DIR / 'optuna_history_json'
MODEL_PATH = MODELS_DIR / 'best_model_p4.pkl'
LEGACY_MODEL_PATH = MODELS_DIR / 'genre_model.pkl'
FEATURE_NAMES_PATH = SAVE_DIR / 'feature_names_p4.json'
MODEL_COMPARISON_CSV = FINAL_DATA_DIR / 'model_comparison_results_p4.csv'
MODEL_COMPARISON_IMG_CSV = FINAL_DATA_DIR / 'p4_model_comparison_results.csv'

# SHAP cache is optional.
_HAS_SHAP_ARTIFACT = True
try:
    from DA.SHAP_explain.shap_artifact import ShapCacheConfig, build_shap_cache
except ModuleNotFoundError:
    try:
        from DA.SHAP_explain.shap_artifact import ShapCacheConfig, build_shap_cache
    except ModuleNotFoundError:
        _HAS_SHAP_ARTIFACT = False
        ShapCacheConfig = None  # type: ignore[assignment]
        build_shap_cache = None  # type: ignore[assignment]

BUILD_SHAP_CACHE = (os.getenv('BUILD_SHAP_CACHE', '1') == '1') and _HAS_SHAP_ARTIFACT


def _compute_multilabel_sample_weight(y: np.ndarray) -> np.ndarray:
    """Fallback weighting for estimators without class_weight.

    For multi-label targets (n_samples, n_labels), compute a per-label balanced
    sample_weight then average across labels.
    """

    y = np.asarray(y)
    if y.ndim != 2 or y.shape[0] == 0:
        return np.ones((y.shape[0],), dtype=float)

    n_samples, n_labels = y.shape
    weights = np.zeros((n_samples,), dtype=float)
    for j in range(n_labels):
        weights += compute_sample_weight(class_weight='balanced', y=y[:, j])
    weights /= float(max(1, n_labels))
    return weights


def _compute_auto_pos_multipliers(
    y_train: np.ndarray,
    *,
    rare_pos_rate_threshold: float,
    alpha: float,
    max_mult: float,
) -> np.ndarray:
    """Compute per-label positive multipliers from TRAIN label prevalence.

    For each label with pos_rate <= rare_pos_rate_threshold:
        mult = min(max_mult, (rare_pos_rate_threshold / max(pos_rate, eps)) ** alpha)
    Otherwise mult = 1.
    """

    y_train = np.asarray(y_train, dtype=int)
    if y_train.ndim != 2 or y_train.shape[1] == 0:
        return np.ones((y_train.shape[1],), dtype=float)

    eps = 1e-6
    mults = np.ones((y_train.shape[1],), dtype=float)
    thr = float(rare_pos_rate_threshold)
    a = float(alpha)
    mmax = float(max_mult)

    for j in range(y_train.shape[1]):
        pos_rate = float(np.mean(y_train[:, j]))
        if pos_rate <= thr:
            raw = (thr / max(pos_rate, eps)) ** a
            mults[j] = float(min(mmax, max(1.0, raw)))

    return mults


def _downsample_dominant_single_label_rows(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    *,
    label_names: list[str],
    label_pos_rate_min: float,
    target_pos_rate: float,
    min_keep_frac: float,
    random_state: int,
) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """Downsample TRAIN rows that are (only) a dominant label to reduce imbalance.

    - Only touches TRAIN (caller responsibility).
    - Preserves time order by keeping original row order after sampling.
    """

    y_train = np.asarray(y_train, dtype=int)
    n = len(X_train)
    if n == 0:
        return X_train, y_train, {'enabled': False, 'reason': 'empty_train'}
    if y_train.ndim != 2 or y_train.shape[0] != n:
        raise ValueError('y_train must be shape (n_samples, n_labels) aligned to X_train')

    label_names = list(label_names)
    n_labels = y_train.shape[1]
    if len(label_names) != n_labels:
        label_names = [f'label_{i}' for i in range(n_labels)]

    sums = np.sum(y_train, axis=1)
    rng = np.random.RandomState(int(random_state))

    keep_mask = np.ones((n,), dtype=bool)
    report: dict = {
        'enabled': True,
        'dropped_rows': 0,
        'labels': {},
    }

    for j, lbl in enumerate(label_names):
        pos_rate = float(np.mean(y_train[:, j]))
        if pos_rate < float(label_pos_rate_min):
            continue

        # Candidate rows: ONLY this label is positive.
        cand = (sums == 1) & (y_train[:, j] == 1)
        cand_idx = np.flatnonzero(cand)
        if cand_idx.size == 0:
            continue

        # Split into candidates and others for this label.
        other_mask = ~cand
        O = int(np.sum(other_mask))
        P_other = int(np.sum(y_train[other_mask, j]))
        C = int(cand_idx.size)

        t = float(target_pos_rate)
        # r = (tO - P_other) / (C(1 - t))
        denom = float(C) * float(max(1e-6, 1.0 - t))
        r = float((t * O - P_other) / denom)
        r = float(min(1.0, max(float(min_keep_frac), r)))

        keep_count = int(np.ceil(r * C))
        keep_count = max(0, min(C, keep_count))
        if keep_count >= C:
            report['labels'][lbl] = {'pos_rate_before': pos_rate, 'candidates': C, 'kept': C, 'dropped': 0, 'keep_frac': 1.0}
            continue

        chosen = rng.choice(cand_idx, size=keep_count, replace=False)
        drop_idx = np.setdiff1d(cand_idx, chosen, assume_unique=False)
        keep_mask[drop_idx] = False

        report['labels'][lbl] = {
            'pos_rate_before': pos_rate,
            'candidates': C,
            'kept': keep_count,
            'dropped': int(C - keep_count),
            'keep_frac': float(keep_count / max(1, C)),
        }

    kept_idx = np.flatnonzero(keep_mask)
    report['dropped_rows'] = int(n - kept_idx.size)

    # Preserve time order.
    X_new = X_train.iloc[kept_idx].reset_index(drop=True)
    y_new = y_train[kept_idx]
    return X_new, y_new, report


class DominantSingleLabelUnderSampler(BaseEstimator):
    """Fold-safe downsampler for multi-label targets.

    Drops a fraction of rows that have ONLY a dominant label (one-hot single).
    Designed to be used inside an imblearn Pipeline so it runs only during `fit`
    (and per-CV fold), avoiding leakage from validation folds.
    """

    def __init__(
        self,
        *,
        label_names: list[str] | None = None,
        label_pos_rate_min: float = 0.50,
        target_pos_rate: float = 0.40,
        min_keep_frac: float = 0.20,
        random_state: int = 42,
    ):
        self.label_names = label_names
        self.label_pos_rate_min = float(label_pos_rate_min)
        self.target_pos_rate = float(target_pos_rate)
        self.min_keep_frac = float(min_keep_frac)
        self.random_state = int(random_state)

    def fit(self, X, y):
        return self

    def fit_resample(self, X, y):
        y_arr = np.asarray(y, dtype=int)
        n = len(X)
        if n == 0:
            return X, y_arr
        if y_arr.ndim != 2 or y_arr.shape[0] != n:
            raise ValueError('y must be shape (n_samples, n_labels) aligned to X')

        label_names = list(self.label_names) if self.label_names is not None else [f'label_{i}' for i in range(y_arr.shape[1])]
        if len(label_names) != int(y_arr.shape[1]):
            label_names = [f'label_{i}' for i in range(y_arr.shape[1])]

        sums = np.sum(y_arr, axis=1)
        rng = np.random.RandomState(int(self.random_state))
        keep_mask = np.ones((n,), dtype=bool)

        for j, _lbl in enumerate(label_names):
            pos_rate = float(np.mean(y_arr[:, j]))
            if pos_rate < float(self.label_pos_rate_min):
                continue

            cand = (sums == 1) & (y_arr[:, j] == 1)
            cand_idx = np.flatnonzero(cand)
            if cand_idx.size == 0:
                continue

            other_mask = ~cand
            O = int(np.sum(other_mask))
            P_other = int(np.sum(y_arr[other_mask, j]))
            C = int(cand_idx.size)

            t = float(self.target_pos_rate)
            denom = float(C) * float(max(1e-6, 1.0 - t))
            r = float((t * O - P_other) / denom)
            r = float(min(1.0, max(float(self.min_keep_frac), r)))

            keep_count = int(np.ceil(r * C))
            keep_count = max(0, min(C, keep_count))
            if keep_count >= C:
                continue

            chosen = rng.choice(cand_idx, size=keep_count, replace=False)
            drop_idx = np.setdiff1d(cand_idx, chosen, assume_unique=False)
            keep_mask[drop_idx] = False

        kept_idx = np.flatnonzero(keep_mask)
        # Preserve time order (kept_idx is sorted).
        if hasattr(X, 'iloc'):
            X_new = X.iloc[kept_idx].reset_index(drop=True)
        else:
            X_new = X[kept_idx]
        y_new = y_arr[kept_idx]
        return X_new, y_new


# Make the sampler importable for pickle/joblib even when executed as a script.
DominantSingleLabelUnderSampler.__module__ = _CANONICAL_MODULE


class SingleLabelTomekRUSSampler(BaseEstimator):
    """Fold-safe boundary cleaning + controlled undersampling for multi-label genre targets.

    Strategy:
    - Convert multi-label y (n_samples, n_labels) into a single-label target by keeping
      only rows with exactly one positive label (sum==1). This drops hybrids like Pop-Ballad.
    - Apply TomekLinks to clean decision boundaries.
    - Apply RandomUnderSampler with a controlled target count for Pop:
        Pop_target = min(Pop_count, ceil(P4_RUS_POP_MAX_MULT * min(ref_counts)))
      where ref_counts are counts of reference labels (e.g., Rap/Indie).
    - Convert y back to 2D one-hot for downstream multi-label wrapper.

    Designed to live inside an imblearn Pipeline so it runs only during fit/CV folds.
    """

    def __init__(
        self,
        *,
        label_names: list[str],
        drop_non_single_label: bool = True,
        pop_label: str = 'genre_Pop',
        ref_labels: list[str] | None = None,
        pop_max_mult: float = 2.0,
        random_state: int = 42,
    ):
        # IMPORTANT: keep parameters exactly as passed in.
        # sklearn.clone() requires __init__ not to copy/mutate parameters.
        self.label_names = label_names
        self.drop_non_single_label = drop_non_single_label
        self.pop_label = pop_label
        self.ref_labels = ref_labels
        self.pop_max_mult = pop_max_mult
        self.random_state = random_state

    def fit(self, X, y):
        return self

    def fit_resample(self, X, y):
        if not _HAS_IMBLEARN or TomekLinks is None or RandomUnderSampler is None:
            raise ImportError('P4 Tomek+RUS resampling requires imbalanced-learn (imblearn).')

        y_arr = np.asarray(y, dtype=int)
        if y_arr.ndim != 2:
            raise ValueError('Expected y as 2D array (n_samples, n_labels)')

        n_samples, n_labels = y_arr.shape
        if n_labels <= 0:
            return X, y_arr

        sums = np.sum((y_arr > 0).astype(int), axis=1)
        
        # 1. Sửa lỗi logic giữ bài lai
        if self.drop_non_single_label:
            keep = sums == 1
        else:
            keep = sums >= 1

        kept_idx = np.flatnonzero(keep)
        if kept_idx.size == 0:
            return X, y_arr

        if hasattr(X, 'iloc'):
            X1 = X.iloc[kept_idx]
        else:
            X1 = X[kept_idx]
        y1 = y_arr[kept_idx]

        # 2. Áp dụng thuật toán "Gán theo độ hiếm" (Rarity Assignment)
        label_counts = np.sum(y1, axis=0) 
        rarity_weights = 1.0 / (label_counts + 1e-6) 
        weighted_y = y1 * rarity_weights
        
        y_single = np.argmax(weighted_y, axis=1).astype(int)

        # 1) TomekLinks boundary cleaning
        try:
            X2, y2 = TomekLinks().fit_resample(X1, y_single)
        except Exception:
            X2, y2 = X1, y_single

        # 2) Controlled RUS (reduce Pop while keeping other labels intact)
        label_names = list(self.label_names) if self.label_names else [f'label_{i}' for i in range(n_labels)]
        pop_id = None
        try:
            pop_id = int(label_names.index(self.pop_label))
        except Exception:
            pop_id = None

        ref_ids: list[int] = []
        ref_labels = list(self.ref_labels) if self.ref_labels is not None else []
        for nm in ref_labels:
            try:
                ref_ids.append(int(label_names.index(nm)))
            except Exception:
                pass

        counts = {c: int(np.sum(np.asarray(y2) == c)) for c in range(n_labels)}
        # Reference count = min count among ref labels present; fallback to min among non-pop labels.
        ref_present = [cid for cid in ref_ids if counts.get(cid, 0) > 0]
        if ref_present:
            ref_count = min(counts[cid] for cid in ref_present)
        else:
            non_pop = [cid for cid in range(n_labels) if cid != pop_id and counts.get(cid, 0) > 0]
            ref_count = min((counts[cid] for cid in non_pop), default=0)

        strategy: dict[int, int] = {cid: int(counts.get(cid, 0)) for cid in range(n_labels) if counts.get(cid, 0) > 0}
        if pop_id is not None and counts.get(pop_id, 0) > 0 and ref_count > 0:
            pop_target = int(np.ceil(float(self.pop_max_mult) * float(ref_count)))
            pop_target = max(1, min(int(counts[pop_id]), int(pop_target)))
            strategy[int(pop_id)] = int(pop_target)

        try:
            X3, y3 = RandomUnderSampler(sampling_strategy=strategy, random_state=int(self.random_state)).fit_resample(X2, y2)
        except Exception:
            X3, y3 = X2, y2

        y3 = np.asarray(y3, dtype=int).reshape(-1)
        y_multi = np.zeros((y3.shape[0], n_labels), dtype=int)
        for i, cls in enumerate(y3.tolist()):
            if 0 <= int(cls) < n_labels:
                y_multi[i, int(cls)] = 1

        return X3, y_multi


# Make the sampler importable for pickle/joblib even when executed as a script.
SingleLabelTomekRUSSampler.__module__ = _CANONICAL_MODULE


def _tscv_f1_micro(pipe_template: Pipeline, X: pd.DataFrame, y: np.ndarray, cv_splitter, sample_weight=None) -> float:
    """Manual TimeSeriesSplit CV to support sample_weight."""

    scores: list[float] = []
    for tr_idx, va_idx in cv_splitter.split(X):
        pipe = clone(pipe_template)
        X_tr = X.iloc[tr_idx]
        X_va = X.iloc[va_idx]
        y_tr = y[tr_idx]
        y_va = y[va_idx]

        # IMPORTANT: when resampling is enabled (imblearn Pipeline), sample counts change inside the pipeline,
        # so externally provided sample_weight would no longer align. In that case, ignore sample_weight.
        if (sample_weight is not None) and (not P4_RESAMPLING_ENABLED):
            sw_tr = np.asarray(sample_weight)[tr_idx]
            pipe.fit(X_tr, y_tr, clf__sample_weight=sw_tr)
        else:
            pipe.fit(X_tr, y_tr)

        y_pred = pipe.predict(X_va)
        scores.append(float(f1_score(y_va, y_pred, average='micro', zero_division=0)))

    return float(np.mean(scores)) if scores else 0.0


def _make_adaboost_balanced() -> AdaBoostClassifier:
    # A shallower stump is more stable for highly imbalanced labels.
    stump = DecisionTreeClassifier(
        max_depth=1,
        min_samples_leaf=20,
        class_weight=MODEL_CLASS_WEIGHT,
        random_state=RANDOM_STATE,
    )
    # sklearn version compatibility matrix is messy here:
    # - newer: `estimator=...`, no `algorithm`
    # - older: `base_estimator=...`
    try:
        return AdaBoostClassifier(
            estimator=stump,
            n_estimators=200,
            learning_rate=0.2,
            algorithm='SAMME',
            random_state=RANDOM_STATE,
        )
    except TypeError:
        try:
            return AdaBoostClassifier(
                estimator=stump,
                n_estimators=200,
                learning_rate=0.2,
                random_state=RANDOM_STATE,
            )
        except TypeError:
            return AdaBoostClassifier(
                base_estimator=stump,
                n_estimators=200,
                learning_rate=0.2,
                random_state=RANDOM_STATE,
            )


def _make_sentiment_ohe() -> OneHotEncoder:
    """Fixed 3-category OHE for final_sentiment (stabilizes feature count)."""
    try:
        return OneHotEncoder(
            handle_unknown='ignore',
            sparse_output=False,
            categories=[SENTIMENT_CATEGORIES],
        )
    except TypeError:
        # sklearn<1.2
        return OneHotEncoder(
            handle_unknown='ignore',
            sparse=False,
            categories=[SENTIMENT_CATEGORIES],
        )


def f1_micro_multilabel(y_true, y_pred) -> float:
    try:
        return float(f1_score(y_true, y_pred, average='micro', zero_division=0))
    except Exception:
        return 0.0


f1_micro_scorer = make_scorer(f1_micro_multilabel, greater_is_better=True)


def f1_macro_multilabel(y_true, y_pred) -> float:
    try:
        return float(f1_score(y_true, y_pred, average='macro', zero_division=0))
    except Exception:
        return 0.0


f1_macro_scorer = make_scorer(f1_macro_multilabel, greater_is_better=True)


def _extract_multilabel_positive_proba(pipe: Pipeline, X: pd.DataFrame) -> np.ndarray | None:
    """Return per-label positive-class probabilities as (n_samples, n_labels).

    Works with Pipeline(..., PerLabelMultiOutputClassifier(...)) when estimator supports predict_proba.
    """

    if not hasattr(pipe, 'predict_proba'):
        return None
    try:
        proba_list = pipe.predict_proba(X)
    except Exception:
        return None

    if not isinstance(proba_list, (list, tuple)):
        return None

    cols = []
    for p in proba_list:
        arr = np.asarray(p)
        if arr.ndim == 2 and arr.shape[1] >= 2:
            cols.append(arr[:, 1])
        elif arr.ndim == 1:
            cols.append(arr)
        else:
            return None
    try:
        return np.column_stack(cols)
    except Exception:
        return None


def _predict_with_thresholds(proba_pos: np.ndarray, thresholds: dict[str, float], label_names: list[str]) -> np.ndarray:
    thr = np.array([float(thresholds.get(lbl, 0.5)) for lbl in label_names], dtype=float)
    return (proba_pos >= thr.reshape(1, -1)).astype(int)


def _tune_thresholds_per_label(
    *,
    pipe_template: Pipeline,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    label_names: list[str],
    sample_weight_train: np.ndarray | None,
    val_fraction: float = 0.2,
    grid: np.ndarray | None = None,
) -> tuple[dict[str, float] | None, dict[str, dict[str, float]]]:
    """Tune per-label thresholds on a time-based validation slice inside TRAIN.

    Selection rule per label:
    - Choose threshold that maximizes F1 for that label.
    - Tie-break by higher accuracy.
    - If best F1 is not better than default (0.5), choose threshold with highest accuracy
      (tie-break by higher F1).  # "ưu tiên tỉ lệ đoán trúng" when tuning not effective
    """

    X_train = X_train.reset_index(drop=True)
    y_train = np.asarray(y_train, dtype=int)

    n = len(X_train)
    if n < 50:
        return None, {}

    if grid is None:
        grid = np.linspace(0.05, 0.95, 91)

    cut = int(n * (1.0 - float(val_fraction)))
    cut = max(10, min(cut, n - 10))
    X_tr, X_va = X_train.iloc[:cut], X_train.iloc[cut:]
    y_tr, y_va = y_train[:cut], y_train[cut:]

    pipe = clone(pipe_template)

    # Try with sample_weight if provided; fallback if estimator doesn't accept it.
    # When resampling is enabled, do NOT pass sample_weight (it won't align after resampling).
    if (sample_weight_train is not None) and (not P4_RESAMPLING_ENABLED):
        sw = np.asarray(sample_weight_train)
        sw_tr = sw[:cut]
        try:
            pipe.fit(X_tr, y_tr, clf__sample_weight=sw_tr)
        except Exception:
            pipe.fit(X_tr, y_tr)
    else:
        pipe.fit(X_tr, y_tr)

    proba_va = _extract_multilabel_positive_proba(pipe, X_va)
    if proba_va is None:
        return None, {}

    thresholds: dict[str, float] = {}
    report: dict[str, dict[str, float]] = {}

    for j, label in enumerate(label_names):
        p = np.asarray(proba_va[:, j], dtype=float)
        y_true = np.asarray(y_va[:, j], dtype=int)

        default_pred = (p >= 0.5).astype(int)
        default_f1 = float(f1_score(y_true, default_pred, zero_division=0))
        default_acc = float(accuracy_score(y_true, default_pred))

        best_thr = 0.5
        best_f1 = default_f1
        best_acc = default_acc

        best_acc_thr = 0.5
        best_acc_val = default_acc
        best_acc_f1 = default_f1

        for thr in grid:
            pred = (p >= float(thr)).astype(int)
            f1v = float(f1_score(y_true, pred, zero_division=0))
            accv = float(accuracy_score(y_true, pred))

            if (f1v > best_f1) or (f1v == best_f1 and accv > best_acc):
                best_f1, best_acc, best_thr = f1v, accv, float(thr)

            if (accv > best_acc_val) or (accv == best_acc_val and f1v > best_acc_f1):
                best_acc_val, best_acc_f1, best_acc_thr = accv, f1v, float(thr)

        chosen_thr = best_thr
        chosen_reason = 'maximize_f1'
        
        # Nếu F1 quá thấp hoặc bằng 0, TUYỆT ĐỐI KHÔNG đẩy ngưỡng lên cao để lấy Accuracy.
        # Hạ ngưỡng xuống 0.20 để ép mô hình nhả Recall cho lớp thiểu số.
        if best_f1 <= default_f1 + 1e-12:
            chosen_thr = 0.20
            chosen_reason = 'force_recall_for_minority'

        thresholds[label] = float(chosen_thr)

        report[label] = {
            'default_threshold': 0.5,
            'default_f1': default_f1,
            'default_accuracy': default_acc,
            'chosen_threshold': float(chosen_thr),
            'chosen_reason': 1.0 if chosen_reason == 'maximize_f1' else 0.0,
            'best_f1': float(best_f1),
            'best_f1_accuracy': float(best_acc),
            'best_accuracy': float(best_acc_val),
            'best_accuracy_f1': float(best_acc_f1),
        }

    return thresholds, report


def plot_custom_optuna_history(study, model_name: str, baseline_f1: float, save_path: str) -> None:
    """Vẽ lịch sử tối ưu Optuna (giữ lại theo yêu cầu)."""
    df = study.trials_dataframe()
    if df.empty or 'value' not in df.columns:
        return

    trial_values = df['value']
    best_values = trial_values.cummax()

    # LƯU CSV để tái sử dụng nếu ảnh lỗi/không ưng ý (khỏi chạy lại Optuna)
    try:
        import os
        import pandas as pd
        from pathlib import Path

        FINAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
        stem = Path(save_path).stem
        csv_path = str(FINAL_DATA_DIR / f"{TASK_ID.lower()}_{stem}.csv")
        df_csv = df.copy()
        df_csv['best_value_so_far'] = pd.to_numeric(df_csv['value'], errors='coerce').cummax()
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
        label='F1-macro tốt nhất',
    )
    plt.axhline(
        y=baseline_f1,
        color='forestgreen',
        linestyle='--',
        linewidth=1.5,
        label=f'Baseline ({baseline_f1:.4f})',
    )

    plt.title(
        f'Lịch sử tối ưu hóa {model_name} (P4 Multi-Label)\n'
        f'Baseline: {baseline_f1:.4f} → Best CV: {study.best_value:.4f}',
        fontsize=14,
        fontweight='bold',
        pad=15,
    )
    plt.xlabel('Số lượt thử (Trial)', fontsize=12)
    plt.ylabel('F1-Macro Score', fontsize=12)
    y_min = float(min(trial_values.min(), baseline_f1) - 0.02)
    plt.ylim(max(0.0, y_min), 1.0)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend(loc='upper right', frameon=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def _save_feature_names_p4(best_pipe) -> None:
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        pre = getattr(best_pipe, 'named_steps', {}).get('preprocessor')
        if pre is None:
            return
        names = pre.get_feature_names_out().tolist()
        names = [str(n).split('__')[-1] for n in names]
        names = rename_topics_in_feature_names(names)
        with open(FEATURE_NAMES_PATH, 'w', encoding='utf-8') as f:
            json.dump(names, f, ensure_ascii=False, indent=2)
        print(f"✅ Đã lưu feature names tại: {FEATURE_NAMES_PATH}")
    except Exception as e:
        print(f"⚠️  Không thể lưu feature names: {e}")


def _fix_and_sort_release_date_p4(df: pd.DataFrame) -> pd.DataFrame:
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


def _build_features_p4(df: pd.DataFrame):
    """Build P4 multi-label target + features (NO TF-IDF)."""

    df = df.copy()

    missing = [c for c in LABEL_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"❌ Thiếu các cột genre one-hot: {missing}")

    # Target multi-label matrix (0/1)
    y_df = df[LABEL_COLS].fillna(0)
    y_df = (y_df.astype(float) > 0).astype(int)

    # Sentiment feature
    if 'final_sentiment' not in df.columns:
        df['final_sentiment'] = 'Neutral'
    df['final_sentiment'] = df['final_sentiment'].fillna('Neutral').astype(str)

    cols_ignore = {
        'spotify_track_id', 'title', 'artists', 'spotify_release_date', 'genres',
        'is_hit', 'spotify_popularity',
        'target',
        *LABEL_COLS,
        # Multi-collinearity removals
        'mfcc2_mean',
        'spectral_rolloff',
        'noun_count',
        'verb_count',
        'tempo_stability',
        'spectral_contrast_band3_mean',
        'spectral_contrast_band4_mean',
        'spectral_contrast_band5_mean',
    }

    topic_candidates = [
        c for c in df.columns
        if isinstance(c, str) and c.startswith('topic_prob') and pd.api.types.is_numeric_dtype(df[c])
    ]
    topic_feats = sorted(topic_candidates)[:15]

    candidate_numeric = [
        c for c in df.columns
        if c not in cols_ignore
        and c not in topic_feats
        and c not in CAT_FEATS
        and pd.api.types.is_numeric_dtype(df[c])
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

    X = df[numeric_cont_feats + binary_feats + topic_feats + CAT_FEATS]
    y = y_df.to_numpy(dtype=int)
    return X, y, numeric_cont_feats, binary_feats, topic_feats, list(CAT_FEATS)


def _create_preprocessor_p4(
    numeric_cont_feats: list[str],
    binary_feats: list[str],
    topic_feats: list[str],
    cat_feats: list[str],
):
    transformers = []
    if numeric_cont_feats:
        transformers.append(('num', StandardScaler(), numeric_cont_feats))
    if binary_feats:
        transformers.append(('bin', 'passthrough', binary_feats))
    if topic_feats:
        transformers.append(('topic', 'passthrough', topic_feats))
    if cat_feats:
        transformers.append(('cat', _make_sentiment_ohe(), cat_feats))
    return ColumnTransformer(transformers=transformers)


def _build_pipeline_p4(preprocessor, clf):
    resampling_enabled = bool(P4_RESAMPLING_ENABLED) and (P4_RESAMPLING not in {'none', 'off', '0', 'false', ''})
    if resampling_enabled and not _HAS_IMBLEARN:
        raise ImportError('P4_RESAMPLING requires imbalanced-learn (imblearn). Please install imbalanced-learn.')

    sampler_step = None
    if resampling_enabled:
        if P4_RESAMPLING in {'dominant_single', 'dominant_single_downsample', 'dominant_singles'}:
            sampler_step = (
                'rebalance',
                DominantSingleLabelUnderSampler(
                    label_names=list(LABEL_COLS),
                    label_pos_rate_min=DOWNSAMPLE_LABEL_POS_RATE_MIN,
                    target_pos_rate=DOWNSAMPLE_TARGET_POS_RATE,
                    min_keep_frac=DOWNSAMPLE_MIN_KEEP_FRAC,
                    random_state=RANDOM_STATE,
                ),
            )
        elif P4_RESAMPLING in {'tomek_rus_controlled', 'tomek+rus_controlled', 'tomek_rus', 'tomek+rus'}:
            sampler_step = (
                'rebalance',
                SingleLabelTomekRUSSampler(
                    label_names=list(LABEL_COLS),
                    drop_non_single_label=bool(P4_DROP_NON_SINGLE_LABEL),
                    pop_label='genre_Pop',
                    ref_labels=list(P4_RUS_REF_LABELS),
                    pop_max_mult=float(P4_RUS_POP_MAX_MULT),
                    random_state=RANDOM_STATE,
                ),
            )
        else:
            raise ValueError(
                f"Invalid P4_RESAMPLING='{P4_RESAMPLING}'. Use: none|dominant_single|tomek_rus_controlled"
            )

    feature_selector = None
    selector_requires_dense = False
    if P4_FEATURE_SELECTION in {'', '0', 'off', 'none', 'false'}:
        feature_selector = None
    elif P4_FEATURE_SELECTION in {'tree', 'extratrees', 'extra_trees'}:
        base = ExtraTreesClassifier(
            n_estimators=int(P4_TREE_N_ESTIMATORS),
            max_depth=P4_TREE_MAX_DEPTH,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight=None,
        )
        feature_selector = SelectFromModel(
            estimator=base,
            threshold=P4_SFM_THRESHOLD,
            max_features=P4_SFM_MAX_FEATURES,
        )
        selector_requires_dense = True
    else:
        raise ValueError(f"Invalid P4_FEATURE_SELECTION='{P4_FEATURE_SELECTION}'. Use: none|tree")

    pipeline_cls = ImbPipeline if resampling_enabled else Pipeline

    steps = [('preprocessor', preprocessor)]
    if resampling_enabled or selector_requires_dense:
        steps.append(('to_dense', FunctionTransformer(sparse_to_dense, validate=False)))
    if sampler_step is not None:
        steps.append(sampler_step)
    if feature_selector is not None:
        steps.append(('select', feature_selector))
    steps.append(('clf', clf))
    return pipeline_cls(steps)


def _optuna_optimize_p4(model_name: str, X_train, y_train, preprocessor, cv_splitter, baseline_f1_macro: float):
    # User request: if Optuna already ran and params exist, do not rerun.
    OPTUNA_REUSE = os.getenv('OPTUNA_REUSE', '1') == '1'
    trials = int(os.getenv('OPTUNA_TRIALS', '20'))
    print(f"So luong trials: {trials}")

    def objective(trial):
        params = {}
        if model_name == 'Random Forest':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 200, 800, step=200),
                'max_depth': trial.suggest_int('max_depth', 5, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            }
            clf = RandomForestClassifier(class_weight=MODEL_CLASS_WEIGHT, random_state=RANDOM_STATE, n_jobs=-1, **params)
        elif model_name == 'Extra Trees':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 200, 800, step=200),
                'max_depth': trial.suggest_int('max_depth', 5, 40),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            }
            clf = ExtraTreesClassifier(class_weight=MODEL_CLASS_WEIGHT, random_state=RANDOM_STATE, n_jobs=-1, **params)
        elif model_name == 'SVM':
            params = {
                'C': trial.suggest_float('C', 0.1, 100.0, log=True),
                'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
            }
            clf = SVC(
                kernel='rbf',
                probability=True,
                class_weight=MODEL_CLASS_WEIGHT,
                random_state=RANDOM_STATE,
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
                objective='binary:logistic',
                eval_metric='logloss',
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
                objective='binary',
                random_state=RANDOM_STATE,
                class_weight=MODEL_CLASS_WEIGHT,
                verbose=-1,
                n_jobs=-1,
                **params,
            )
        elif model_name == 'AdaBoost':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 600, step=100),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 1.0, log=True),
            }
            # class_weight is applied via the balanced decision stump.
            clf = _make_adaboost_balanced()
            clf.set_params(**params)
        elif model_name == 'Gradient Boosting':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=100),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 2, 5),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            }
            clf = GradientBoostingClassifier(random_state=RANDOM_STATE, **params)
        elif model_name == 'Logistic Regression':
            params = {
                'C': trial.suggest_float('C', 0.01, 10.0, log=True),
            }
            clf = LogisticRegression(
                class_weight=MODEL_CLASS_WEIGHT,
                max_iter=2000,
                solver='liblinear',
                random_state=RANDOM_STATE,
                **params,
            )
        elif model_name == 'MLP (Neural Net)':
            params = {
                'alpha': trial.suggest_float('alpha', 1e-5, 1e-2, log=True),
                'learning_rate_init': trial.suggest_float('learning_rate_init', 1e-4, 5e-2, log=True),
            }
            clf = MLPClassifier(
                hidden_layer_sizes=(128, 64),
                max_iter=1000,
                early_stopping=True,
                random_state=RANDOM_STATE,
                **params,
            )
        else:
            return -1.0

        mo = PerLabelMultiOutputClassifier(
            clf,
            pos_weight_multiplier=POS_WEIGHT_MULTIPLIER,
            rare_pos_rate_threshold=POS_WEIGHT_RARE_THRESHOLD,
            use_per_label_weights=USE_PER_LABEL_WEIGHTS,
        )
        pipe = _build_pipeline_p4(preprocessor, mo)
        scores = cross_val_score(pipe, X_train, y_train, cv=cv_splitter, scoring=f1_macro_scorer, n_jobs=-1)
        return float(np.mean(scores))

    OPTUNA_DIR.mkdir(parents=True, exist_ok=True)
    safe = model_name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('+', '').replace('->', 'to')
    out_path = OPTUNA_DIR / f'p4_{safe}_best_params.json'

    if OPTUNA_REUSE and out_path.exists():
        try:
            print(f"✅ Optuna params đã tồn tại → skip trials và dùng lại: {out_path}")
            with open(out_path, 'r', encoding='utf-8') as f:
                cached = json.load(f)
            if isinstance(cached, dict) and len(cached) > 0:
                return cached
        except Exception as e:
            print(f"⚠️  Không load được Optuna params cũ → sẽ optimize lại: {e}")

    study = optuna.create_study(direction='maximize', study_name=f'P4_{model_name}_opt')
    study.optimize(objective, n_trials=trials, show_progress_bar=True)

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(study.best_params, f, ensure_ascii=False, indent=2)

    # User request: keep everything under the task folder only.

    try:
        opt_img_dir = IMG_DIR / 'optuna_history_image'
        opt_img_dir.mkdir(parents=True, exist_ok=True)
        out_img = opt_img_dir / f'p4_optuna_history_{safe}.png'
        plot_custom_optuna_history(study, model_name=model_name, baseline_f1=baseline_f1_macro, save_path=str(out_img))
    except Exception:
        pass

    print(f"\nBest CV F1-macro: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
    print(f"✅ Đã lưu Optuna params tại: {out_path}")
    return study.best_params


def run_analysis_p4_timeholdout_no_tfidf():
    _enable_task_logging(task_dir=TASK_DIR, task_tag=TASK_ID)
    print("\n" + "=" * 80)
    print(f"🚀 {TASK_LABEL} — Time-based holdout 80/20 | NO TF-IDF/lyric")
    print("=" * 80)
    print(f"   • Data: {DATA_FILE}")
    print(f"   • Target: {LABEL_COLS} (multi-label)")

    df = pd.read_csv(DATA_FILE)
    df = _fix_and_sort_release_date_p4(df)
    X, y, numeric_cont_feats, binary_feats, topic_feats, cat_feats = _build_features_p4(df)

    # Sanity check: master data nên không còn missing.
    if X.isna().any().any():
        na_cols = X.columns[X.isna().any()].tolist()
        raise ValueError(
            "❌ Dữ liệu đầu vào vẫn còn missing values. "
            "Hãy chạy lại scripts/data_prepared_for_ML.py để fill median trước khi train. "
            f"Cột bị thiếu: {na_cols[:20]}" + (" ..." if len(na_cols) > 20 else "")
        )

    print("\n" + "=" * 60)
    print("🔍 KIỂM TRA CHI TIẾT CÁC BIẾN ĐẦU VÀO (TASK 4 - MULTI-LABEL)")
    print("=" * 60)
    print(f"1️⃣ Continuous (Scale): {len(numeric_cont_feats)}")
    print(np.array(numeric_cont_feats))
    print("-" * 60)
    print(f"2️⃣ Binary (Passthrough): {len(binary_feats)}")
    print(np.array(binary_feats))
    print("-" * 60)
    print(f"3️⃣ Topic_prob (Passthrough): {len(topic_feats)}")
    if topic_feats:
        print(np.array(topic_feats))
    print("-" * 60)
    print(f"4️⃣ Categorical (Sentiment): {cat_feats} → OHE = 3 dims")
    print("-" * 60)

    raw_total = len(numeric_cont_feats) + len(binary_feats) + len(topic_feats) + 3
    print(f"✅ TỔNG SỐ BIẾN ĐẦU VÀO (EXPECTED=102, gồm 3 sentiment OHE): {raw_total}")
    if raw_total == 102:
        print("✅ PASS: Tổng số biến = 102")
    else:
        print(f"⚠️  LỆCH: Tổng số biến != 102 (hiện tại = {raw_total})")
    print("=" * 60 + "\n")

    n_total = len(df)
    if n_total < 10:
        raise ValueError('Dataset quá nhỏ để split 80/20 theo thời gian')

    split_point = int(n_total * 0.8)
    X_train, X_test = X.iloc[:split_point].reset_index(drop=True), X.iloc[split_point:].reset_index(drop=True)
    y = np.asarray(y, dtype=int)
    y_train, y_test = y[:split_point], y[split_point:]

    print("\n" + "=" * 80)
    print("🧪 BƯỚC 1: CHIA DỮ LIỆU (Time-based Holdout 80/20)")
    print("=" * 80)
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")
    if 'spotify_release_date' in df.columns:
        print(
            f"📅 Train: {df.iloc[:split_point]['spotify_release_date'].min().date()} → {df.iloc[:split_point]['spotify_release_date'].max().date()}"
        )
        print(
            f"📅 Test : {df.iloc[split_point:]['spotify_release_date'].min().date()} → {df.iloc[split_point:]['spotify_release_date'].max().date()}"
        )

    print("\n" + "=" * 80)
    print("📊 PHÂN BỐ NHÃN (tỉ lệ genre=1) — dùng để cân nhắc phạt label hiếm")
    print("=" * 80)
    for i, lbl in enumerate(LABEL_COLS):
        tr_rate = float(np.mean(y_train[:, i]))
        te_rate = float(np.mean(y_test[:, i]))
        print(f"   • {lbl}: train_pos={tr_rate:.3f} | test_pos={te_rate:.3f}")

    # Helpful diagnostics for the controlled Tomek+RUS mode (which keeps single-label rows only).
    if P4_RESAMPLING in {'tomek_rus_controlled', 'tomek+rus_controlled', 'tomek_rus', 'tomek+rus'}:
        tr_sums = np.sum(y_train, axis=1)
        te_sums = np.sum(y_test, axis=1)
        tr_single = int(np.sum(tr_sums == 1))
        tr_hybrid = int(np.sum(tr_sums > 1))
        tr_zero = int(np.sum(tr_sums == 0))
        te_single = int(np.sum(te_sums == 1))
        te_hybrid = int(np.sum(te_sums > 1))
        te_zero = int(np.sum(te_sums == 0))
        print("-" * 80)
        print(
            "🧼 Single-label diagnostic (sum(labels) per row): "
            f"TRAIN single={tr_single} | hybrid(>1)={tr_hybrid} | none(0)={tr_zero} ; "
            f"TEST single={te_single} | hybrid(>1)={te_hybrid} | none(0)={te_zero}"
        )

    if P4_RESAMPLING_ENABLED:
        print("\n" + "=" * 80)
        print("⚖️  REBALANCE (TRAIN/CV only, leakage-safe via Pipeline)")
        print("=" * 80)
        if P4_RESAMPLING in {'dominant_single', 'dominant_single_downsample', 'dominant_singles'}:
            print(
                f"Enabled: P4_RESAMPLING={P4_RESAMPLING} | "
                f"label_pos_rate_min={DOWNSAMPLE_LABEL_POS_RATE_MIN} | target_pos_rate={DOWNSAMPLE_TARGET_POS_RATE} | "
                f"min_keep_frac={DOWNSAMPLE_MIN_KEEP_FRAC}"
            )
            print("ℹ️  Downsampling will run inside each fit/CV fold (not applied to X_train upfront).")
        elif P4_RESAMPLING in {'tomek_rus_controlled', 'tomek+rus_controlled', 'tomek_rus', 'tomek+rus'}:
            print(
                f"Enabled: P4_RESAMPLING={P4_RESAMPLING} | drop_non_single_label={int(P4_DROP_NON_SINGLE_LABEL)} | "
                f"pop_max_mult={P4_RUS_POP_MAX_MULT} | ref_labels={P4_RUS_REF_LABELS}"
            )
            print("ℹ️  TomekLinks + controlled RUS runs inside each fit/CV fold (single-label rows only).")
        else:
            print(f"Enabled: P4_RESAMPLING={P4_RESAMPLING}")

    pos_weight_multipliers = POS_WEIGHT_MULTIPLIER
    if USE_PER_LABEL_WEIGHTS and AUTO_POS_WEIGHT:
        pos_weight_multipliers = _compute_auto_pos_multipliers(
            y_train,
            rare_pos_rate_threshold=POS_WEIGHT_RARE_THRESHOLD,
            alpha=AUTO_POS_WEIGHT_ALPHA,
            max_mult=AUTO_POS_WEIGHT_MAX,
        )

    if USE_PER_LABEL_WEIGHTS and AUTO_POS_WEIGHT:
        print(
            f"✅ Per-label weights AUTO: alpha={AUTO_POS_WEIGHT_ALPHA} | max={AUTO_POS_WEIGHT_MAX} "
            f"(áp cho label có train_pos ≤ {POS_WEIGHT_RARE_THRESHOLD})"
        )
        for i, lbl in enumerate(LABEL_COLS):
            try:
                m = float(pos_weight_multipliers[i])
            except Exception:
                m = 1.0
            print(f"   • {lbl}: pos_mult={m:.3f}")
    elif USE_PER_LABEL_WEIGHTS and (isinstance(POS_WEIGHT_MULTIPLIER, list) or float(POS_WEIGHT_MULTIPLIER) > 1.0):
        print(
            f"✅ Per-label weights ON: POS_WEIGHT_MULTIPLIER={POS_WEIGHT_MULTIPLIER} "
            f"(áp cho label có train_pos ≤ {POS_WEIGHT_RARE_THRESHOLD})"
        )
    else:
        print(
            "ℹ️  Nếu label hiếm bị miss nhiều (FN lớn trong CM), thử: "
            "POS_WEIGHT_MULTIPLIER=2.0 và POS_WEIGHT_RARE_THRESHOLD=0.35 (giữ USE_PER_LABEL_WEIGHTS=1)"
        )

    preprocessor = _create_preprocessor_p4(numeric_cont_feats, binary_feats, topic_feats, cat_feats)
    tscv_inner = TimeSeriesSplit(n_splits=5)

    # Fallback weighting for estimators without class_weight
    sample_weight_train = _compute_multilabel_sample_weight(y_train)

    print("\n" + "=" * 80)
    print("🏁 BƯỚC 2: BASELINE MODELS (CV on TRAIN only)")
    print("=" * 80)

    baseline = {
        'Random Forest': RandomForestClassifier(
            n_estimators=400,
            class_weight=MODEL_CLASS_WEIGHT,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        'Extra Trees': ExtraTreesClassifier(
            n_estimators=500,
            class_weight=MODEL_CLASS_WEIGHT,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        'AdaBoost': _make_adaboost_balanced(),
        'HistGradientBoosting': HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_depth=6,
            max_iter=300,
            random_state=RANDOM_STATE,
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.1,
            subsample=0.8,
            random_state=RANDOM_STATE,
        ),
        'Logistic Regression': LogisticRegression(
            class_weight=MODEL_CLASS_WEIGHT,
            max_iter=2000,
            solver='liblinear',
            random_state=RANDOM_STATE,
        ),
        'MLP (Neural Net)': MLPClassifier(
            hidden_layer_sizes=(128, 64),
            max_iter=1000,
            early_stopping=True,
            random_state=RANDOM_STATE,
        ),
        'SVM': SVC(
            kernel='rbf',
            probability=True,
            class_weight=MODEL_CLASS_WEIGHT,
            random_state=RANDOM_STATE,
        ),
        'XGBoost': XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=6,
            objective='binary:logistic',
            eval_metric='logloss',
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        'LightGBM': LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=31,
            objective='binary',
            random_state=RANDOM_STATE,
            class_weight=MODEL_CLASS_WEIGHT,
            verbose=-1,
            n_jobs=-1,
        ),
    }

    rows = []
    pipes = {}
    print(f"\n{'MODEL':<16} | {'CV_F1MIC':<9} | {'CV_F1MAC':<9} | {'TEST_F1MIC':<10} | {'TEST_F1MAC':<10} | {'HAMMING':<8} | {'JACCARD':<8}")
    print("-" * 85)

    for name, est in baseline.items():
        try:
            mo = PerLabelMultiOutputClassifier(
                est,
                pos_weight_multiplier=pos_weight_multipliers,
                rare_pos_rate_threshold=POS_WEIGHT_RARE_THRESHOLD,
                use_per_label_weights=USE_PER_LABEL_WEIGHTS,
            )
            pipe = _build_pipeline_p4(preprocessor, mo)

            # CV is always computed without sample_weight (fast/parallel).
            # For estimators that don't support class_weight, we still apply a balanced sample_weight on final fit.
            fit_with_sw = (not P4_RESAMPLING_ENABLED) and (name in {'Gradient Boosting', 'MLP (Neural Net)', 'XGBoost'})
            try:
                cv_scores = cross_val_score(
                    pipe,
                    X_train,
                    y_train,
                    cv=tscv_inner,
                    scoring=f1_micro_scorer,
                    n_jobs=-1,
                    error_score=0.0,
                )
                cv_f1mic = float(np.mean(cv_scores))
            except Exception:
                # Fallback to a sequential CV to surface issues as score=0 rather than NaN.
                cv_f1mic = _tscv_f1_micro(pipe, X_train, y_train, cv_splitter=tscv_inner)

            # CV F1-macro (selection/Optuna objective). Prefer parallel CV; fallback to score=0 on failure.
            try:
                cv_scores_mac = cross_val_score(
                    pipe,
                    X_train,
                    y_train,
                    cv=tscv_inner,
                    scoring=f1_macro_scorer,
                    n_jobs=-1,
                    error_score=0.0,
                )
                cv_f1mac = float(np.mean(cv_scores_mac))
            except Exception:
                cv_f1mac = 0.0

            if fit_with_sw:
                pipe.fit(X_train, y_train, clf__sample_weight=sample_weight_train)
            else:
                pipe.fit(X_train, y_train)

            y_pred = pipe.predict(X_test)

            test_f1mic = float(f1_score(y_test, y_pred, average='micro', zero_division=0))
            test_f1mac = float(f1_score(y_test, y_pred, average='macro', zero_division=0))
            hl = float(hamming_loss(y_test, y_pred))
            jac = float(jaccard_score(y_test, y_pred, average='samples', zero_division=0))

            rows.append(
                {
                    'Model': name,
                    'CV_F1_Micro': cv_f1mic,
                    'CV_F1_Macro': cv_f1mac,
                    'Test_F1_Micro': test_f1mic,
                    'Test_F1_Macro': test_f1mac,
                    'Hamming_Loss': hl,
                    'Jaccard_Samples': jac,
                }
            )
            pipes[name] = pipe
            print(f"{name:<16} | {cv_f1mic:.4f}   | {cv_f1mac:.4f}   | {test_f1mic:.4f}     | {test_f1mac:.4f}     | {hl:.4f}  | {jac:.4f}")
        except Exception as e:
            print(f"❌ {name}: {e}")

    # Leakage-safe selection: pick best baseline by CV F1-macro (TRAIN only).
    results_df = pd.DataFrame(rows).sort_values(by='CV_F1_Macro', ascending=False)
    best_baseline = results_df.iloc[0]
    best_name = str(best_baseline['Model'])
    base_cv_f1mac = float(best_baseline['CV_F1_Macro'])
    base_test_f1mac = float(best_baseline['Test_F1_Macro'])
    print(
        f"\n🏆 Best Baseline (by CV F1-macro): {best_name} "
        f"(CV F1-macro={base_cv_f1mac:.4f} | Test F1-macro(report)={base_test_f1mac:.4f})"
    )

    print("\n" + "=" * 80)
    print(f"🧠 BƯỚC 3: OPTUNA (maximize CV F1-macro) — {best_name}")
    print("=" * 80)

    best_params = None
    if best_name in {
        'Random Forest',
        'Extra Trees',
        'AdaBoost',
        'Gradient Boosting',
        'Logistic Regression',
        'MLP (Neural Net)',
        'SVM',
        'XGBoost',
        'LightGBM',
    }:
        best_params = _optuna_optimize_p4(best_name, X_train, y_train, preprocessor, tscv_inner, baseline_f1_macro=base_cv_f1mac)
    else:
        print("ℹ️ Model này chưa hỗ trợ Optuna trong workflow chuẩn → dùng baseline.")

    final_pipe = pipes[best_name]
    final_model_name = best_name
    if isinstance(best_params, dict) and len(best_params) > 0:
        est = None
        if best_name == 'Random Forest':
            est = RandomForestClassifier(class_weight=MODEL_CLASS_WEIGHT, random_state=RANDOM_STATE, n_jobs=-1, **best_params)
        elif best_name == 'Extra Trees':
            est = ExtraTreesClassifier(class_weight=MODEL_CLASS_WEIGHT, random_state=RANDOM_STATE, n_jobs=-1, **best_params)
        elif best_name == 'AdaBoost':
            est = _make_adaboost_balanced()
            est.set_params(**best_params)
        elif best_name == 'Gradient Boosting':
            est = GradientBoostingClassifier(random_state=RANDOM_STATE, **best_params)
        elif best_name == 'Logistic Regression':
            est = LogisticRegression(
                class_weight=MODEL_CLASS_WEIGHT,
                max_iter=2000,
                solver='liblinear',
                random_state=RANDOM_STATE,
                **best_params,
            )
        elif best_name == 'MLP (Neural Net)':
            est = MLPClassifier(
                hidden_layer_sizes=(128, 64),
                max_iter=1000,
                early_stopping=True,
                random_state=RANDOM_STATE,
                **best_params,
            )
        elif best_name == 'SVM':
            p = dict(best_params)
            p.setdefault('kernel', 'rbf')
            p.setdefault('probability', True)
            p.setdefault('class_weight', MODEL_CLASS_WEIGHT)
            est = SVC(random_state=RANDOM_STATE, **p)
        elif best_name == 'XGBoost':
            est = XGBClassifier(objective='binary:logistic', eval_metric='logloss', random_state=RANDOM_STATE, n_jobs=-1, **best_params)
        elif best_name == 'LightGBM':
            est = LGBMClassifier(objective='binary', class_weight=MODEL_CLASS_WEIGHT, random_state=RANDOM_STATE, verbose=-1, n_jobs=-1, **best_params)

        if est is not None:
            mo = PerLabelMultiOutputClassifier(
                est,
                pos_weight_multiplier=pos_weight_multipliers,
                rare_pos_rate_threshold=POS_WEIGHT_RARE_THRESHOLD,
                use_per_label_weights=USE_PER_LABEL_WEIGHTS,
            )
            opt_pipe = _build_pipeline_p4(preprocessor, mo)

            needs_sw = (not P4_RESAMPLING_ENABLED) and (best_name in {'Gradient Boosting', 'MLP (Neural Net)', 'XGBoost'})
            if needs_sw:
                opt_pipe.fit(X_train, y_train, clf__sample_weight=sample_weight_train)
            else:
                opt_pipe.fit(X_train, y_train)

            y_pred_opt = opt_pipe.predict(X_test)
            opt_test_f1mac = float(f1_score(y_test, y_pred_opt, average='macro', zero_division=0))

            # Leakage-safe rollback: compare CV F1-macro on TRAIN only.
            try:
                opt_cv_scores = cross_val_score(
                    opt_pipe,
                    X_train,
                    y_train,
                    cv=tscv_inner,
                    scoring=f1_macro_scorer,
                    n_jobs=-1,
                    error_score=0.0,
                )
                opt_cv_f1mac = float(np.mean(opt_cv_scores))
            except Exception:
                opt_cv_f1mac = 0.0

            print("\n" + "=" * 80)
            print("⚖️ ROLLBACK CHECK (CV on TRAIN; Test chỉ để báo cáo)")
            print("=" * 80)
            print(f"   • Baseline CV F1-macro : {base_cv_f1mac:.4f}")
            print(f"   • Optimized CV F1-macro: {opt_cv_f1mac:.4f}")
            print(f"   • Baseline Test F1-macro(report): {base_test_f1mac:.4f}")
            print(f"   • Optimized Test F1-macro(report): {opt_test_f1mac:.4f}")

            if opt_cv_f1mac < base_cv_f1mac:
                print(f"\n⚠️ Optimized CV F1-macro ({opt_cv_f1mac:.4f}) < Baseline ({base_cv_f1mac:.4f})")
                print("✅ QUYẾT ĐỊNH: ROLLBACK về baseline pipeline")
            else:
                final_pipe = opt_pipe
                final_model_name = best_name + ' (OPTIMIZED)'
                print(f"\n✅ Optuna cải thiện (CV): {base_cv_f1mac:.4f} → {opt_cv_f1mac:.4f}")

    # --- Per-label threshold tuning (on validation slice inside TRAIN to avoid leakage) ---
    thresholds, threshold_report = _tune_thresholds_per_label(
        pipe_template=final_pipe,
        X_train=X_train,
        y_train=y_train,
        label_names=LABEL_COLS,
        sample_weight_train=sample_weight_train,
        val_fraction=0.2,
    )

    y_pred_default = final_pipe.predict(X_test)
    y_pred_thr = None
    y_pred_final = y_pred_default

    if thresholds is not None:
        proba_test = _extract_multilabel_positive_proba(final_pipe, X_test)
        if proba_test is None:
            print("\n⚠️ Không lấy được predict_proba → bỏ qua tối ưu ngưỡng (dùng predict mặc định).")
            thresholds = None
            threshold_report = {}
        else:
            y_pred_thr = _predict_with_thresholds(proba_test, thresholds, LABEL_COLS)

            default_f1mac = float(f1_score(y_test, y_pred_default, average='macro', zero_division=0))
            thr_f1mac = float(f1_score(y_test, y_pred_thr, average='macro', zero_division=0))

            print("\n" + "=" * 80)
            print("🎚️  BƯỚC 3.5: TỐI ƯU NGƯỠNG THEO TỪNG THỂ LOẠI (on TRAIN-val only)")
            print("=" * 80)
            print(f"Default Test F1-macro (threshold=0.5): {default_f1mac:.4f}")
            print(f"Thresholded Test F1-macro:             {thr_f1mac:.4f}")

            for lbl in LABEL_COLS:
                thr = thresholds.get(lbl, 0.5)
                print(f"   • {lbl}: threshold={thr:.3f}")

            y_pred_final = y_pred_thr

    test_f1mic = float(f1_score(y_test, y_pred_final, average='micro', zero_division=0))
    test_f1mac = float(f1_score(y_test, y_pred_final, average='macro', zero_division=0))
    test_hl = float(hamming_loss(y_test, y_pred_final))
    test_jac = float(jaccard_score(y_test, y_pred_final, average='samples', zero_division=0))

    print("\n" + "=" * 80)
    print("📌 BƯỚC 4: ĐÁNH GIÁ FINAL")
    print("=" * 80)
    print(f"Model: {final_model_name}")
    print(f"Test F1-micro : {test_f1mic:.4f}")
    print(f"Test F1-macro : {test_f1mac:.4f}")
    print(f"Hamming Loss  : {test_hl:.4f}")
    print(f"Jaccard(samples): {test_jac:.4f}")

    try:
        print("\n📋 Classification Report (Test):")
        print(classification_report(y_test, y_pred_final, target_names=LABEL_COLS, zero_division=0))
    except Exception as e:
        print(f"⚠️  Không thể in classification report: {e}")

    try:
        IMG_DIR.mkdir(parents=True, exist_ok=True)

        # BEFORE threshold (default predict)
        cm_before = multilabel_confusion_matrix(y_test, y_pred_default)
        for i, label in enumerate(LABEL_COLS):
            safe_label = str(label).replace('/', '_').replace(' ', '_')
            plt.figure(figsize=(6, 5))
            sns.heatmap(
                cm_before[i],
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=['0', '1'],
                yticklabels=['0', '1'],
            )
            plt.xlabel('Pred', fontsize=11, fontweight='bold')
            plt.ylabel('True', fontsize=11, fontweight='bold')
            plt.title(f'P4 Genre — CM BEFORE threshold ({label})', fontsize=12, fontweight='bold')
            plt.tight_layout()
            out_cm = IMG_DIR / f'p4_cm_{safe_label}_before_threshold.png'
            plt.savefig(str(out_cm), dpi=300, bbox_inches='tight')
            plt.close()

        if y_pred_thr is not None and isinstance(thresholds, dict) and len(thresholds) > 0:
            cm_after = multilabel_confusion_matrix(y_test, y_pred_thr)
            for i, label in enumerate(LABEL_COLS):
                safe_label = str(label).replace('/', '_').replace(' ', '_')
                thr = float(thresholds.get(label, 0.5))
                plt.figure(figsize=(6, 5))
                sns.heatmap(
                    cm_after[i],
                    annot=True,
                    fmt='d',
                    cmap='Blues',
                    xticklabels=['0', '1'],
                    yticklabels=['0', '1'],
                )
                plt.xlabel('Pred', fontsize=11, fontweight='bold')
                plt.ylabel('True', fontsize=11, fontweight='bold')
                plt.title(f'P4 Genre — CM AFTER threshold ({label}) (thr={thr:.2f})', fontsize=12, fontweight='bold')
                plt.tight_layout()
                out_cm = IMG_DIR / f'p4_cm_{safe_label}_after_threshold.png'
                plt.savefig(str(out_cm), dpi=300, bbox_inches='tight')
                plt.close()

            print(f"✅ Đã lưu CM BEFORE/AFTER threshold tại: {IMG_DIR}/p4_cm_[label]_[before|after]_threshold.png")
        else:
            print(f"✅ Đã lưu CM BEFORE threshold tại: {IMG_DIR}/p4_cm_[label]_before_threshold.png")
    except Exception as e:
        print(f"⚠️  Không thể lưu Confusion Matrix BEFORE/AFTER threshold: {e}")

    # Confirm transformed feature count after fit
    try:
        pre = final_pipe.named_steps['preprocessor']
        n_transformed = len(pre.get_feature_names_out())
        print(f"\n🔎 Transformed feature count: {n_transformed}")
        if n_transformed == 102:
            print("✅ PASS: Transformed features = 102")
        else:
            print("⚠️  NOTE: Transformed features != 102")
    except Exception:
        pass

    print("\n" + "=" * 80)
    print("💾 BƯỚC 5: LƯU ARTIFACTS")
    print("=" * 80)

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    pkl_data = {
        'pkl_schema_version': 1,
        'task_id': TASK_ID,
        'data_source': str(Path(DATA_FILE)),
        'pipeline': final_pipe,
        'model_name': final_model_name,
        'label_names': list(LABEL_COLS),
        'training_config': {
            'feature_selection': str(P4_FEATURE_SELECTION),
            'resampling': str(P4_RESAMPLING),
            'downsample_label_pos_rate_min': float(DOWNSAMPLE_LABEL_POS_RATE_MIN),
            'downsample_target_pos_rate': float(DOWNSAMPLE_TARGET_POS_RATE),
            'downsample_min_keep_frac': float(DOWNSAMPLE_MIN_KEEP_FRAC),
            'use_per_label_weights': bool(USE_PER_LABEL_WEIGHTS),
            'auto_pos_weight': bool(AUTO_POS_WEIGHT),
            'pos_weight_rare_threshold': float(POS_WEIGHT_RARE_THRESHOLD),
            'pos_weight_multiplier': POS_WEIGHT_MULTIPLIER,
        },
        'metrics': {
            'test_f1_micro': test_f1mic,
            'test_f1_macro': test_f1mac,
            'test_hamming_loss': test_hl,
            'test_jaccard_samples': test_jac,
        },
        'best_params': best_params if isinstance(best_params, dict) else {},
        'thresholds': thresholds if isinstance(thresholds, dict) else None,
        'threshold_report': threshold_report if isinstance(threshold_report, dict) else {},
        'shap_cache': None,
    }

    if BUILD_SHAP_CACHE:
        try:
            pkl_data['shap_cache'] = build_shap_cache(
                X_train,
                X_test,
                config=ShapCacheConfig(n_background=200, n_explain=200, random_state=RANDOM_STATE),
            )
        except Exception as e:
            print(f"⚠️  build_shap_cache failed → continue without SHAP cache: {e}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pkl_data, str(MODEL_PATH))
    try:
        joblib.dump(pkl_data, str(LEGACY_MODEL_PATH))
    except Exception:
        pass

    _save_feature_names_p4(final_pipe)

    # CSVs
    FINAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
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
        ax.set_title('P4 Genre — Model Comparison (Test F1-macro)', fontsize=16, fontweight='bold', pad=20)
        ax.invert_yaxis()
        ax.set_xlim(0.0, float(df_plot['Test_F1_Macro'].max()) + 0.10)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()
        out_plot = IMG_DIR / 'p4_model_comparison_f1_macro.png'
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
    print(f"   • Optuna params (pkl): {OPTUNA_DIR}/p4_*_best_params.json")
    print(f"   • Plot:               {IMG_DIR}/p4_model_comparison_f1_macro.png")


if __name__ == '__main__':
    run_analysis_p4_timeholdout_no_tfidf()
