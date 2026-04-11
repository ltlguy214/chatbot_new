# -*- coding: utf-8 -*-
"""Style Clustering — 5 Vibes (PCA - Hierarchical)

Mục tiêu:
- Phân loại nhạc V-Pop thành 5 sắc thái cảm xúc:
    1) Bùng nổ / Sôi động
    2) Tươi mới / Yêu đời
    3) Kịch tính / Da diết
    4) Sâu lắng / Thấu cảm
    5) Bình yên / Chữa lành

Thiết kế pipeline:
- Feature selection: dùng toàn bộ biến số từ dataset nhưng loại bỏ `genre_*` và nhãn `is_hit`.
    (Trong dataset hiện tại: 115 numeric features nếu tính cả `genre_*`; khi loại `genre_*` thì còn 111.)
- Weighting: tempo_bpm x1.5, rms_energy x1.5, sentiment_* x1.5.
- PCA: n_components = 0.60 ()
- k-auto = kneed => k=5 (25 biến)
- Visualization: Radar chart cho 5 nhóm. 
- Export: lưu .pkl (scaler + pca + 2 kmeans) và xuất VPop_5_Vibes_Final.csv.
"""

import sys

# --- Console encoding (Windows-safe) ---
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

import pandas as pd
import numpy as np

# Use a non-interactive backend to avoid Tkinter/thread teardown errors on Windows.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import atexit
import warnings
import joblib
import json
from datetime import datetime
from scipy.optimize import linear_sum_assignment
from types import SimpleNamespace

from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import optuna
from sklearn.metrics import adjusted_rand_score

# Auto-elbow detection
try:
    from kneed import KneeLocator
except Exception:  # pragma: no cover
    KneeLocator = None

# Import đầy đủ các thuật toán phân cụm
from sklearn.cluster import (
    KMeans, BisectingKMeans, Birch, AgglomerativeClustering,
    MiniBatchKMeans, SpectralClustering
)
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# Cấu hình hệ thống
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
TASK_ID = 'P2'
TASK_LABEL = 'P2 - 5 Vibes Style Clustering'
RANDOM_STATE = 42

def _configure_optuna_logging() -> None:
    """Silence Optuna INFO logs like Hit-style tasks."""
    try:
        optuna.logging.disable_default_handler()
        optuna.logging.disable_propagation()
        optuna.logging.set_verbosity(optuna.logging.ERROR)
    except Exception:
        pass


_configure_optuna_logging()

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


def _enable_task_logging(*, task_dir: str, task_tag: str) -> str:
    os.makedirs(task_dir, exist_ok=True)
    latest_path = os.path.join(task_dir, f'{task_tag}_run_latest.log')

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
    print(f"📝 Log file: {_show_path(latest_path)}", flush=True)
    print(
        f"🕒 Started: {datetime.now().isoformat(timespec='seconds')} | Script: {os.path.basename(__file__)}",
        flush=True,
    )
    return latest_path

# -----------------------------------------------------------------------------
# Paths (robust to current working directory)
# -----------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))

def _show_path(path: str) -> str:
    try:
        return os.path.relpath(path, PROJECT_ROOT)
    except Exception:
        return path

# Keep a dedicated image folder like other Hit-style tasks.
FINAL_DATA_DIR = os.path.join(PROJECT_ROOT, 'DA', 'final_data')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'DA', 'models')

IMG_DIR = os.path.join(PROJECT_ROOT, 'DA', 'tasks', 'Style')
# User request: store task artifacts under DA/tasks/Style.
OUTPUT_DIR = IMG_DIR

MODEL_PATH = os.path.join(MODEL_DIR, 'best_model_p2.pkl')

FEATURE_NAMES_PATH = os.path.join(MODEL_DIR, 'feature_names_p2.json')
# Keep only a BEST-model CSV (requested).
MODEL_COMPARISON_IMG_CSV = os.path.join(FINAL_DATA_DIR, 'p2_best_model_results.csv')

FINAL_CSV_PATH = os.path.join(FINAL_DATA_DIR, 'VPop_5_Vibes_Final.csv')

os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(FINAL_DATA_DIR, exist_ok=True)

# =============================================================================
# GLOBAL CONFIG (Hit-style toggles)
# =============================================================================
OPTUNA_TRIALS = int(os.getenv('OPTUNA_TRIALS', '20'))
SHOW_OPTUNA_PROGRESS_BAR = os.getenv('OPTUNA_PROGRESS_BAR', '0') == '1'

# Reuse cached Optuna best_params if present (do not rerun trials).
# Default: enabled, aligned with other tasks.
OPTUNA_REUSE = os.getenv('OPTUNA_REUSE', '1') == '1'

# Print per-trial logs like Optuna INFO output (requested).
# Note: we keep Optuna's own INFO logs silenced and print equivalent lines ourselves.
P2_OPTUNA_TRIAL_LOGS = os.getenv('P2_OPTUNA_TRIAL_LOGS', '1') == '1'

# Toggle feature weighting (Tempo/Energy/Sentiment)
APPLY_FEATURE_WEIGHTS = os.getenv('P2_APPLY_FEATURE_WEIGHTS', '1') == '1'

# Optimize feature weights via Optuna (requested)
P2_OPTIMIZE_FEATURE_WEIGHTS = os.getenv('P2_OPTIMIZE_FEATURE_WEIGHTS', '1') == '1'
P2_WEIGHT_OPT_TRIALS = int(os.getenv('P2_WEIGHT_OPT_TRIALS', str(OPTUNA_TRIALS)))
P2_WEIGHT_BOUNDS_LOW = float(os.getenv('P2_WEIGHT_BOUNDS_LOW', '0.6'))
P2_WEIGHT_BOUNDS_HIGH = float(os.getenv('P2_WEIGHT_BOUNDS_HIGH', '2.2'))

# Consensus clustering (ensemble) (requested)
P2_CONSENSUS_CLUSTERING = os.getenv('P2_CONSENSUS_CLUSTERING', '1') == '1'
P2_CONSENSUS_TOP_K = int(os.getenv('P2_CONSENSUS_TOP_K', '3'))

# Stability validation (bootstrap) (requested)
P2_BOOTSTRAP_STABILITY = os.getenv('P2_BOOTSTRAP_STABILITY', '1') == '1'
P2_BOOTSTRAP_N = int(os.getenv('P2_BOOTSTRAP_N', '25'))
P2_BOOTSTRAP_SAMPLE_FRAC = float(os.getenv('P2_BOOTSTRAP_SAMPLE_FRAC', '0.90'))

# Optimize ideal vectors for vibe mapping (requested)
P2_OPTIMIZE_IDEAL_VECTORS = os.getenv('P2_OPTIMIZE_IDEAL_VECTORS', '1') == '1'
P2_IDEAL_OPT_TRIALS = int(os.getenv('P2_IDEAL_OPT_TRIALS', '30'))
P2_IDEAL_DELTA = float(os.getenv('P2_IDEAL_DELTA', '0.25'))

# PCA variance threshold (requested: 0.60)
P2_PCA_N_COMPONENTS = float(os.getenv('P2_PCA_N_COMPONENTS', '0.60'))

# Toggle Optuna history images
SAVE_OPTUNA_PLOTS = os.getenv('P2_SAVE_OPTUNA_PLOTS', '1') == '1'

# Reduce clutter: don't dump full trials JSON unless explicitly enabled.
SAVE_OPTUNA_TRIALS_JSON = os.getenv('P2_SAVE_OPTUNA_TRIALS_JSON', '0') == '1'

# Legacy: optionally also export extra backward-compatibility PKLs.
SAVE_LEGACY_PKLS = os.getenv('P2_SAVE_LEGACY_PKLS', '0') == '1'

OPTUNA_IMG_DIR = os.path.join(IMG_DIR, 'optuna_history_image')
OPTUNA_JSON_DIR = os.path.join(IMG_DIR, 'optuna_history_json')
os.makedirs(OPTUNA_IMG_DIR, exist_ok=True)
os.makedirs(OPTUNA_JSON_DIR, exist_ok=True)

# =============================================================================
# 1. CÁC HÀM HỖ TRỢ (UTILS)
# =============================================================================
def plot_custom_optuna_history(study, model_name: str, baseline_score: float, save_path: str) -> None:
    """Hit-style Optuna history: trials scatter + best step + baseline line."""
    try:
        df = study.trials_dataframe()
    except Exception:
        return
    if df is None or df.empty or 'value' not in df.columns:
        return

    trial_values = df['value']
    best_values = trial_values.cummax()

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
        label='CH tốt nhất',
    )
    plt.axhline(
        y=baseline_score,
        color='forestgreen',
        linestyle='--',
        linewidth=1.5,
        label=f'Baseline ({baseline_score:.2f})',
    )

    ymin = float(min(trial_values.min(), baseline_score))
    ymax = float(max(trial_values.max(), baseline_score))
    # Keep y-range tight so small differences are readable.
    pad = max(0.002, (ymax - ymin) * 0.10)

    plt.title(
        f'Lịch sử tối ưu hóa {model_name} (mục tiêu: Silhouette)\n'
        f'Baseline: {baseline_score:.4f} → Best: {study.best_value:.4f}',
        fontsize=14,
        fontweight='bold',
        pad=15,
    )
    plt.xlabel('Số lượt thử (Trial)', fontsize=12)
    plt.ylabel('Silhouette (SIL)', fontsize=12)
    plt.ylim(ymin - pad, ymax + pad)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend(loc='upper right', frameon=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# =============================================================================
# 2. XỬ LÝ DỮ LIỆU & PCA
# =============================================================================
def _ensure_sentiment_ohe(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure fixed 3-dim sentiment one-hot: negative/neutral/positive."""
    df = df.copy()
    if 'final_sentiment' in df.columns:
        df['final_sentiment'] = df['final_sentiment'].fillna('Neutral').astype(str)
        df['_final_sentiment_norm'] = df['final_sentiment'].str.strip().str.lower()
        df['_final_sentiment_norm'] = df['_final_sentiment_norm'].replace({'neg': 'negative', 'pos': 'positive'})
        try:
            ohe = OneHotEncoder(
                sparse_output=False,
                handle_unknown='ignore',
                categories=[['negative', 'neutral', 'positive']],
            )
        except TypeError:
            ohe = OneHotEncoder(
                sparse=False,
                handle_unknown='ignore',
                categories=[['negative', 'neutral', 'positive']],
            )
        sent_encoded = ohe.fit_transform(df[['_final_sentiment_norm']])
        df['sentiment_negative'] = sent_encoded[:, 0]
        df['sentiment_neutral'] = sent_encoded[:, 1]
        df['sentiment_positive'] = sent_encoded[:, 2]
    else:
        df['sentiment_negative'] = 0.0
        df['sentiment_neutral'] = 1.0
        df['sentiment_positive'] = 0.0
    return df

def compute_feature_weights(feature_names: list[str], *, group_weights: dict | None = None) -> np.ndarray:
    """Strategic weighting (Librosa/NLP) based on available features.

    Default (legacy) weights:
    - tempo_bpm & sentiment_*: 1.5
    - beat_strength_mean: 1.4
    - onset_rate: 1.3
    - rms_energy: 1.2
    - spectral_flatness_mean: 1.2

    If `group_weights` is provided, overrides those group multipliers.
    """

    gw = dict(group_weights or {})
    tempo_w = float(gw.get('tempo', 1.5))
    sentiment_w = float(gw.get('sentiment', 1.5))
    beat_w = float(gw.get('beat', 1.4))
    onset_w = float(gw.get('onset', 1.3))
    energy_w = float(gw.get('energy', 1.2))
    flatness_w = float(gw.get('flatness', 1.2))

    weights = np.ones(len(feature_names), dtype=float)
    for i, col in enumerate(feature_names):
        c = str(col).lower()
        if 'tempo_bpm' in c:
            weights[i] = tempo_w
        elif c.startswith('sentiment_'):
            weights[i] = sentiment_w
        elif 'beat_strength_mean' in c:
            weights[i] = beat_w
        elif 'onset_rate' in c:
            weights[i] = onset_w
        elif 'rms_energy' in c:
            weights[i] = energy_w
        elif 'spectral_flatness_mean' in c:
            weights[i] = flatness_w
    return weights


def optimize_feature_weights(
    *,
    X_scaled: np.ndarray,
    feature_cols: list[str],
    k: int,
    n_trials: int,
    verbose: bool = True,
) -> tuple[dict, np.ndarray]:
    """Optuna search for feature-group weights to maximize Silhouette.

    Uses the same PCA+KMeans objective as the main pipeline, but keeps it
    lightweight by reusing X_scaled and only re-weighting features.
    """

    cache_path = os.path.join(OPTUNA_JSON_DIR, 'p2_feature_weights_best_params.json')
    if OPTUNA_REUSE and os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cached = json.load(f)
            if isinstance(cached, dict) and cached:
                w = compute_feature_weights(feature_cols, group_weights=cached)
                if verbose:
                    print("\n" + "=" * 80)
                    print("OPTUNA REUSE (skip trials): Feature Weights")
                    print("=" * 80)
                    print(f"   ✅ Loaded: {_show_path(cache_path)}")
                    print(f"   ✅ Params: {cached}")
                    print("=" * 80 + "\n")
                return dict(cached), w
        except Exception:
            pass

    def objective(trial: optuna.Trial) -> float:
        low = float(P2_WEIGHT_BOUNDS_LOW)
        high = float(P2_WEIGHT_BOUNDS_HIGH)
        params = {
            'tempo': trial.suggest_float('tempo', low, high),
            'energy': trial.suggest_float('energy', low, high),
            'sentiment': trial.suggest_float('sentiment', low, high),
            'beat': trial.suggest_float('beat', low, high),
            'onset': trial.suggest_float('onset', low, high),
            'flatness': trial.suggest_float('flatness', low, high),
        }
        w = compute_feature_weights(feature_cols, group_weights=params)
        X_w = X_scaled * w
        X_pca, _pca = run_pca(X_w, n_components=P2_PCA_N_COMPONENTS)
        km = KMeans(n_clusters=int(k), random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X_pca)
        return _safe_silhouette(X_pca, labels)

    study = optuna.create_study(
        direction='maximize',
        study_name='p2_feature_weights_opt',
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    if verbose and P2_OPTUNA_TRIAL_LOGS:
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]
        print(f"So luong trials: {n_trials}")
        print(f"[I {ts}] A new study created in memory with name: {study.study_name}")

    study.optimize(objective, n_trials=int(n_trials), show_progress_bar=SHOW_OPTUNA_PROGRESS_BAR)

    best = dict(study.best_params)
    try:
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(best, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    w_best = compute_feature_weights(feature_cols, group_weights=best)
    if verbose:
        print("\n" + "=" * 80)
        print("OPTUNA BEST: Feature Weights")
        print("=" * 80)
        print(f"   ✅ Best SIL: {study.best_value:.4f}")
        print(f"   ✅ Best params: {best}")
        print("=" * 80 + "\n")
    return best, w_best

def preprocess_data(df: pd.DataFrame):
    """Module 1: Preprocessing + strategic weighting.

    - Uses numeric features from `data_prepared_for_ML.csv` (incl. `genre_*` if present).
    - Excludes metadata/labels.
    - Does NOT scale one-hot/binary columns (0/1).
    - Applies requested Librosa/NLP weights after scaling.
    """
    print("🚀 BẮT ĐẦU TIỀN XỬ LÝ DỮ LIỆU...\n")

    df_proc = _ensure_sentiment_ohe(df)
    sentiment_cols = ['sentiment_negative', 'sentiment_neutral', 'sentiment_positive']

    drop_cols = [
        'spotify_track_id', 'title', 'artists', 'spotify_release_date', 'genres',
        'is_hit', 'target', 'final_sentiment',
    ]

    X_raw = df_proc.drop(columns=drop_cols, errors='ignore')
    feature_cols = X_raw.select_dtypes(include=[np.number]).columns.tolist()

    # Ensure sentiment OHE exists in the feature list
    for c in sentiment_cols:
        if c in df_proc.columns and c not in feature_cols:
            feature_cols.append(c)

    print(f"   ✅ Sentiment OHE included: {sentiment_cols}")
    print(f"   📊 Input Features: {len(feature_cols)} biến")

    X_df = df_proc[feature_cols].copy()

    # Detect binary 0/1 columns (don't scale them)
    binary_cols: list[str] = []
    continuous_cols: list[str] = []
    for c in feature_cols:
        s = X_df[c]
        vals = pd.unique(s.dropna())
        if len(vals) <= 2 and set(map(float, vals.tolist())) <= {0.0, 1.0}:
            binary_cols.append(c)
        else:
            continuous_cols.append(c)

    print(f"   🔎 Binary (no-scale) cols: {len(binary_cols)} | Continuous cols: {len(continuous_cols)}")

    # Build sklearn-only imputing/scaling objects that preserve original feature order
    feature_index = {c: i for i, c in enumerate(feature_cols)}
    cont_idx = [feature_index[c] for c in continuous_cols]
    bin_idx = [feature_index[c] for c in binary_cols]
    out_order = continuous_cols + binary_cols
    inv_perm = [out_order.index(c) for c in feature_cols]

    imputer_ct = ColumnTransformer(
        transformers=[
            ('cont', SimpleImputer(strategy='median'), cont_idx),
            ('bin', SimpleImputer(strategy='most_frequent'), bin_idx),
        ],
        remainder='drop',
        sparse_threshold=0.0,
    )
    imputer = Pipeline(
        steps=[
            ('ct', imputer_ct),
            ('reorder', FunctionTransformer(np.take, kw_args={'indices': inv_perm, 'axis': 1})),
        ]
    )

    scaler_ct = ColumnTransformer(
        transformers=[
            ('cont', StandardScaler(), cont_idx),
            ('bin', 'passthrough', bin_idx),
        ],
        remainder='drop',
        sparse_threshold=0.0,
    )
    scaler = Pipeline(
        steps=[
            ('ct', scaler_ct),
            ('reorder', FunctionTransformer(np.take, kw_args={'indices': inv_perm, 'axis': 1})),
        ]
    )

    X_imputed = imputer.fit_transform(X_df.values)
    X_scaled = scaler.fit_transform(X_imputed)

    # Weighting (post-scaling)
    # - If Optuna weight optimization is enabled, we defer weighting to main().
    if APPLY_FEATURE_WEIGHTS and not P2_OPTIMIZE_FEATURE_WEIGHTS:
        print("   ⚖️  Applying Strategic Weights (fixed, legacy)...")
        weights = compute_feature_weights(feature_cols)
        X_weighted = X_scaled * weights
    elif APPLY_FEATURE_WEIGHTS and P2_OPTIMIZE_FEATURE_WEIGHTS:
        print("   ⚖️  Feature Weights: Optuna enabled → defer weighting to optimizer")
        weights = np.ones(len(feature_cols), dtype=float)
        X_weighted = X_scaled
    else:
        print("   ⚖️  Skipping Feature Weights (disabled)")
        weights = np.ones(len(feature_cols), dtype=float)
        X_weighted = X_scaled

    artifacts = {
        'imputer': imputer,
        'scaler': scaler,
        'binary_cols': binary_cols,
        'continuous_cols': continuous_cols,
        'feature_weights': weights,
    }

    return df_proc, feature_cols, X_scaled, X_weighted, artifacts


def run_pca(X_weighted: np.ndarray, n_components: float = 0.60) -> tuple[np.ndarray, PCA]:
    """Module 3: PCA analytics + dynamic PCA embedding."""
    
    # Tính toán mốc % tự động dựa trên tham số đầu vào
    target_pct = int(n_components * 100) 
    print(f"   📉 PCA ANALYTICS + PCA(n_components={n_components}) ...")

    pca_full = PCA().fit(X_weighted)
    eigenvalues = pca_full.explained_variance_
    cum_var = np.cumsum(pca_full.explained_variance_ratio_)
    
    # Tìm số lượng PC đạt mốc variance mục tiêu (ví dụ 70%)
    n_target = int(np.argmax(cum_var >= n_components) + 1)
    print(f"   📊 n_{target_pct} (>={target_pct}%): {n_target}")

    # Chạy mô hình PCA chính thức
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_weighted)
    selected_n = int(getattr(pca, 'n_components_', X_pca.shape[1]))
    actual_var = float(np.sum(pca.explained_variance_ratio_))
    print(f"   Tổng phương sai giải thích {actual_var*100:.1f}% với {selected_n} chiều")

    # ==========================================
    # VẼ BIỂU ĐỒ VỚI MỐC MỚI
    # ==========================================
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))

    # (1) Eigenvalues (top 50)
    ax = axes[0]
    top_n = min(50, len(eigenvalues))
    ax.bar(range(1, top_n + 1), eigenvalues[:top_n], color='steelblue', alpha=0.75)
    if n_target <= top_n:
        ax.axvline(x=n_target, color='green', linestyle='--', linewidth=2.0, label=f'n_{target_pct}={n_target}')
    ax.set_title('Eigenvalues', fontweight='bold')
    ax.set_xlabel('PC')
    ax.set_ylabel('Eigenvalue')
    ax.grid(True, linestyle=':', alpha=0.5, axis='y')
    ax.legend()

    # (2) Scree plot (first 40)
    ax = axes[1]
    max_pc = min(40, len(eigenvalues))
    ax.plot(range(1, max_pc + 1), eigenvalues[:max_pc], marker='o', linestyle='-', color='#1f77b4', linewidth=2)
    ax.axhline(y=1, color='red', linestyle='--', label='Kaiser (y=1)')
    ax.axvline(x=n_target, color='green', linestyle=':', label=f'{target_pct}% (PC{n_target})')
    ax.set_title('Scree Plot', fontweight='bold')
    ax.set_xlabel('PC')
    ax.set_ylabel('Eigenvalue')
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend()

    # (3) Cumulative variance
    ax = axes[2]
    ax.plot(range(1, len(cum_var) + 1), cum_var * 100, 'go-', linewidth=2, markersize=3)
    ax.axhline(y=target_pct, color='green', linestyle='--', linewidth=1.5, alpha=0.9, label=f'{target_pct}%')
    ax.axvline(x=n_target, color='green', linestyle=':', linewidth=2)
    ax.set_title('Cumulative Variance (%)', fontweight='bold')
    ax.set_xlabel('PC')
    ax.set_ylabel('Cumulative %')
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend()
    ax.set_xlim(0, min(100, len(cum_var)))

    fig.suptitle(f'P2 PCA Analysis (Weighted) + PCA({target_pct}% variance)', fontsize=14, fontweight='bold')
    out = os.path.join(OUTPUT_DIR, 'pca_analysis.png')
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"   ✅ Đã lưu PCA analytics tại: {_show_path(out)}")
    return X_pca, pca


def run_auto_k(X_pca: np.ndarray, k_min: int = 2, k_max: int = 10) -> dict:
    """Module 2: Automated Elbow Detection using `kneed` on KMeans inertia."""
    print("\n" + "=" * 80)
    print("AUTO-K MODULE: k=2..10 (KneeLocator / Elbow on inertia)")
    print("=" * 80)

    ks = list(range(k_min, k_max + 1))
    inertias: list[float] = []
    silhouettes: list[float] = []

    for k in ks:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X_pca)
        inertias.append(float(km.inertia_))
        silhouettes.append(float(_safe_silhouette(X_pca, labels)))
        # keep per-k logs concise but readable
        print(f"   k={k:2d} | inertia={inertias[-1]:.2f} | silhouette={silhouettes[-1]:.4f}")

    optimal_k = None
    if KneeLocator is not None:
        try:
            kl = KneeLocator(ks, inertias, curve='convex', direction='decreasing')
            optimal_k = int(kl.knee) if kl.knee is not None else None
        except Exception:
            optimal_k = None

    if optimal_k is None:
        # Fallback to a simple heuristic: best silhouette
        optimal_k = int(ks[int(np.argmax(silhouettes))])
        print(f"\n   ⚠️  KneeLocator không xác định được knee → fallback theo Silhouette: k={optimal_k}")
    else:
        print(f"\n   ✅ Auto-Elbow (KneeLocator): k={optimal_k}")

    # Plot elbow with auto vertical marker
    plt.figure(figsize=(10, 6))
    plt.plot(ks, inertias, marker='o', linestyle='--', color='red', linewidth=2, label='Inertia (WCSS)')
    plt.axvline(x=optimal_k, color='green', linestyle='-', linewidth=2, label=f'Auto Elbow k={optimal_k}')
    plt.title('Auto Elbow Detection (kneed) — KMeans Inertia', fontweight='bold')
    plt.xlabel('Số lượng cụm (k)')
    plt.ylabel('WCSS (Inertia)')
    plt.xticks(ks)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    out = os.path.join(OUTPUT_DIR, 'k_selection_auto.png')
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Đã lưu Auto-Elbow plot tại: {_show_path(out)}")

    return {
        'k': ks,
        'inertia': inertias,
        'silhouette': silhouettes,
        'optimal_k': int(optimal_k),
    }

def assign_5_vibes_direct(df, cluster_col='cluster_main'):
    return assign_5_vibes_with_ideals(df, cluster_col=cluster_col, ideal_vibes=None)


DEFAULT_IDEAL_VIBES = {
    # Structure: [Beat, Energy, Sentiment_Negative] (scaled 0..1)
    "Bùng nổ / Sôi động":   np.array([1.0, 1.0, 0.0]),
    "Tươi mới / Yêu đời":   np.array([0.6, 0.7, 0.0]),
    "Kịch tính / Da diết":  np.array([0.7, 0.8, 1.0]),
    "Sâu lắng / Thấu cảm":  np.array([0.0, 0.0, 1.0]),
    "Bình yên / Chữa lành": np.array([0.2, 0.1, 0.0]),
}


def _align_labels_to_reference(ref_labels: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    """Align cluster ids to a reference labeling (solves label permutation).

    Uses Hungarian assignment to maximize overlap between clusters.
    """
    ref = np.asarray(ref_labels).astype(int).reshape(-1)
    lab = np.asarray(labels).astype(int).reshape(-1)
    k = int(k)
    if k <= 1:
        return lab

    # Build contingency matrix counts[ref, lab]
    counts = np.zeros((k, k), dtype=int)
    mask = (ref >= 0) & (ref < k) & (lab >= 0) & (lab < k)
    r = ref[mask]
    l = lab[mask]
    for rr, ll in zip(r.tolist(), l.tolist()):
        counts[int(rr), int(ll)] += 1

    # Hungarian minimizes cost; we maximize overlap => cost = max - counts
    maxv = int(counts.max()) if counts.size else 0
    cost = (maxv - counts).astype(float)
    row_ind, col_ind = linear_sum_assignment(cost)

    mapping = {int(col): int(row) for row, col in zip(row_ind, col_ind)}
    aligned = np.array([mapping.get(int(x), int(x)) for x in lab], dtype=int)
    return aligned


def _vote_consensus_labels(ref_labels: np.ndarray, aligned_label_runs: list[np.ndarray], k: int) -> np.ndarray:
    """Majority vote over already-aligned label runs."""
    ref = np.asarray(ref_labels).astype(int).reshape(-1)
    runs = [ref] + [np.asarray(x).astype(int).reshape(-1) for x in aligned_label_runs]
    mat = np.vstack(runs)
    # vote per column
    out = np.zeros((mat.shape[1],), dtype=int)
    for i in range(mat.shape[1]):
        vals = mat[:, i]
        # bincount for 0..k-1
        bc = np.bincount(np.clip(vals, 0, k - 1), minlength=k)
        out[i] = int(np.argmax(bc))
    return out


def _fit_consensus_kmeans(X_pca: np.ndarray, labels: np.ndarray, k: int) -> KMeans:
    """Fit a predict-capable KMeans initialized from consensus labels."""
    k = int(k)
    X = np.asarray(X_pca)
    labs = np.asarray(labels).astype(int).reshape(-1)
    centers = []
    for c in range(k):
        idx = np.flatnonzero(labs == c)
        if idx.size == 0:
            centers.append(np.mean(X, axis=0))
        else:
            centers.append(np.mean(X[idx], axis=0))
    init = np.vstack(centers)
    km = KMeans(n_clusters=k, init=init, n_init=1, random_state=RANDOM_STATE, max_iter=300)
    km.fit(X)
    return km


def bootstrap_cluster_stability(
    *,
    clusterer_builder,
    X_pca: np.ndarray,
    ref_labels: np.ndarray,
    k: int,
    n_boot: int,
    sample_frac: float,
) -> dict:
    """Bootstrap stability: fit on resampled data, predict all points, align to ref."""
    X = np.asarray(X_pca)
    n = int(X.shape[0])
    n_boot = int(max(0, n_boot))
    if n_boot <= 0 or n < 10:
        return {'enabled': False, 'reason': 'n_boot<=0_or_small_n'}

    if not (0.2 <= float(sample_frac) <= 1.0):
        raise ValueError('P2_BOOTSTRAP_SAMPLE_FRAC must be in [0.2, 1.0]')

    rng = np.random.RandomState(RANDOM_STATE)
    m = int(np.ceil(float(sample_frac) * n))
    ref = np.asarray(ref_labels).astype(int).reshape(-1)

    aligned_runs: list[np.ndarray] = []
    agreements: list[float] = []
    for b in range(n_boot):
        idx = rng.choice(np.arange(n), size=m, replace=True)
        model = clusterer_builder()
        try:
            model.fit(X[idx])
            if hasattr(model, 'predict'):
                pred = model.predict(X)
            else:
                pred = model.fit_predict(X)
        except Exception:
            continue

        aligned = _align_labels_to_reference(ref, pred, k)
        aligned_runs.append(aligned)
        agreements.append(float(np.mean(aligned == ref)))

    if not aligned_runs:
        return {'enabled': False, 'reason': 'no_successful_bootstrap_fits'}

    per_point = np.mean(np.vstack([(r == ref).astype(float) for r in aligned_runs]), axis=0)
    return {
        'enabled': True,
        'n_boot_success': int(len(aligned_runs)),
        'mean_agreement': float(np.mean(agreements)),
        'p10_agreement': float(np.percentile(agreements, 10)),
        'p50_agreement': float(np.percentile(agreements, 50)),
        'p90_agreement': float(np.percentile(agreements, 90)),
        'per_point_agreement_mean': float(np.mean(per_point)),
        'per_point_agreement_p10': float(np.percentile(per_point, 10)),
        'per_point_agreement_p50': float(np.percentile(per_point, 50)),
        'per_point_agreement_p90': float(np.percentile(per_point, 90)),
    }


def bootstrap_collect_aligned_label_runs(
    *,
    clusterer_builder,
    X_pca: np.ndarray,
    ref_labels: np.ndarray,
    k: int,
    n_boot: int,
    sample_frac: float,
) -> list[np.ndarray]:
    """Collect aligned label runs for downstream stability analyses."""
    X = np.asarray(X_pca)
    n = int(X.shape[0])
    n_boot = int(max(0, n_boot))
    if n_boot <= 0 or n < 10:
        return []
    rng = np.random.RandomState(RANDOM_STATE)
    m = int(np.ceil(float(sample_frac) * n))
    ref = np.asarray(ref_labels).astype(int).reshape(-1)

    runs: list[np.ndarray] = []
    for _b in range(n_boot):
        idx = rng.choice(np.arange(n), size=m, replace=True)
        model = clusterer_builder()
        try:
            model.fit(X[idx])
            if not hasattr(model, 'predict'):
                continue
            pred = model.predict(X)
        except Exception:
            continue
        runs.append(_align_labels_to_reference(ref, pred, k))
    return runs


def _vibe_labels_from_cluster_labels(
    *,
    cluster_labels: np.ndarray,
    X_map01: np.ndarray,
    k: int,
    ideal_vibes: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[int, str]]:
    """Map cluster ids -> vibe names then return per-song vibe labels."""
    labels = np.asarray(cluster_labels).astype(int).reshape(-1)
    k = int(k)
    vibe_names = list(ideal_vibes.keys())
    cluster_ids = list(range(k))

    # cluster profiles in mapping feature space
    prof = []
    for cid in cluster_ids:
        idx = np.flatnonzero(labels == int(cid))
        if idx.size == 0:
            prof.append(np.zeros((X_map01.shape[1],), dtype=float))
        else:
            prof.append(np.mean(X_map01[idx], axis=0))
    prof = np.vstack(prof)

    cost = np.zeros((k, len(vibe_names)), dtype=float)
    for i in range(k):
        for j, vn in enumerate(vibe_names):
            cost[i, j] = float(np.linalg.norm(prof[i] - np.asarray(ideal_vibes[vn], dtype=float).reshape(-1)))
    row_ind, col_ind = linear_sum_assignment(cost)
    mapping: dict[int, str] = {}
    for r, c in zip(row_ind, col_ind):
        mapping[int(cluster_ids[int(r)])] = str(vibe_names[int(c)])
    vibe_labels = np.array([mapping.get(int(x), 'UNKNOWN') for x in labels], dtype=object)
    return vibe_labels, mapping


def optimize_ideal_vibes(
    *,
    X_map01: np.ndarray,
    ref_cluster_labels: np.ndarray,
    aligned_cluster_label_runs: list[np.ndarray],
    k: int,
    n_trials: int,
    verbose: bool = True,
) -> dict[str, np.ndarray]:
    """Optuna optimize ideal vibe vectors around defaults to maximize bootstrap stability."""

    cache_path = os.path.join(OPTUNA_JSON_DIR, 'p2_ideal_vibes_best_params.json')
    if OPTUNA_REUSE and os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cached = json.load(f)
            if isinstance(cached, dict) and cached:
                ideals = {k: np.asarray(v, dtype=float) for k, v in cached.items()}
                if verbose:
                    print("\n" + "=" * 80)
                    print("OPTUNA REUSE (skip trials): Ideal Vectors")
                    print("=" * 80)
                    print(f"   ✅ Loaded: {_show_path(cache_path)}")
                    print("=" * 80 + "\n")
                return ideals
        except Exception:
            pass

    base = {k: np.asarray(v, dtype=float).reshape(-1) for k, v in DEFAULT_IDEAL_VIBES.items()}
    vibe_names = list(base.keys())
    delta = float(P2_IDEAL_DELTA)

    # Precompute reference vibe labels for each trial quickly.
    def build_ideals_from_trial(trial: optuna.Trial) -> dict[str, np.ndarray]:
        ideals = {}
        for vn in vibe_names:
            vec0 = base[vn]
            vec = []
            for d_i, dim_name in enumerate(['beat', 'energy', 'neg']):
                key = f"{vn}__{dim_name}"
                dv = trial.suggest_float(key, -delta, delta)
                vv = float(np.clip(vec0[d_i] + dv, 0.0, 1.0))
                vec.append(vv)
            ideals[vn] = np.asarray(vec, dtype=float)
        return ideals

    def objective(trial: optuna.Trial) -> float:
        ideals = build_ideals_from_trial(trial)
        ref_vibes, _ = _vibe_labels_from_cluster_labels(
            cluster_labels=np.asarray(ref_cluster_labels),
            X_map01=X_map01,
            k=k,
            ideal_vibes=ideals,
        )
        if len(aligned_cluster_label_runs) == 0:
            return 0.0

        agrees = []
        for run_labels in aligned_cluster_label_runs:
            vibes, _ = _vibe_labels_from_cluster_labels(
                cluster_labels=np.asarray(run_labels),
                X_map01=X_map01,
                k=k,
                ideal_vibes=ideals,
            )
            agrees.append(float(np.mean(vibes == ref_vibes)))

        stability = float(np.mean(agrees))

        # Regularize: keep ideals separated to avoid degenerate solutions.
        vecs = np.vstack([ideals[vn] for vn in vibe_names])
        pen = 0.0
        for i in range(len(vibe_names)):
            for j in range(i + 1, len(vibe_names)):
                dist = float(np.linalg.norm(vecs[i] - vecs[j]))
                pen += max(0.0, 0.15 - dist)
        return stability - 0.25 * pen

    study = optuna.create_study(
        direction='maximize',
        study_name='p2_ideal_vibes_opt',
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(objective, n_trials=int(n_trials), show_progress_bar=SHOW_OPTUNA_PROGRESS_BAR)

    # Rebuild best ideals
    best_params = dict(study.best_params)
    ideals_best = {}
    for vn in vibe_names:
        vec0 = base[vn]
        vec = []
        for d_i, dim_name in enumerate(['beat', 'energy', 'neg']):
            key = f"{vn}__{dim_name}"
            dv = float(best_params.get(key, 0.0))
            vec.append(float(np.clip(vec0[d_i] + dv, 0.0, 1.0)))
        ideals_best[vn] = np.asarray(vec, dtype=float)

    try:
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump({k: v.tolist() for k, v in ideals_best.items()}, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    if verbose:
        print("\n" + "=" * 80)
        print("OPTUNA BEST: Ideal Vectors")
        print("=" * 80)
        print(f"   ✅ Best objective: {study.best_value:.4f}")
        print(f"   ✅ Saved: {_show_path(cache_path)}")
        print("=" * 80 + "\n")
    return ideals_best


def assign_5_vibes_with_ideals(
    df: pd.DataFrame,
    *,
    cluster_col: str = 'cluster_main',
    ideal_vibes: dict[str, np.ndarray] | None,
    features: list[str] | None = None,
) -> tuple[pd.Series, dict]:
    """Assign 5 vibes using Hungarian mapping from cluster profiles to ideal vectors."""

    df = df.copy()
    feats = features or ['beat_strength_mean', 'rms_energy', 'sentiment_negative']
    ideals = dict(DEFAULT_IDEAL_VIBES) if ideal_vibes is None else dict(ideal_vibes)

    # Global scaling of features to [0, 1] improves stability vs per-cluster scaling.
    scaler = MinMaxScaler()
    X_map = df[feats].astype(float).fillna(0.0).values
    X_map01 = scaler.fit_transform(X_map)

    labels = df[cluster_col].astype(int).values
    cluster_ids = sorted(pd.unique(labels).tolist())
    vibe_names = list(ideals.keys())

    prof = []
    for cid in cluster_ids:
        idx = np.flatnonzero(labels == int(cid))
        if idx.size == 0:
            prof.append(np.zeros((len(feats),), dtype=float))
        else:
            prof.append(np.mean(X_map01[idx], axis=0))
    prof = np.vstack(prof)

    cost_matrix = np.zeros((len(cluster_ids), len(vibe_names)), dtype=float)
    for i, _cid in enumerate(cluster_ids):
        cluster_vector = prof[i]
        for j, vibe_name in enumerate(vibe_names):
            ideal_vector = np.asarray(ideals[vibe_name], dtype=float).reshape(-1)
            cost_matrix[i, j] = float(np.linalg.norm(cluster_vector - ideal_vector))

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    mapping: dict[int, str] = {}
    for r, c in zip(row_ind, col_ind):
        mapping[int(cluster_ids[int(r)])] = str(vibe_names[int(c)])

    df['vibe'] = df[cluster_col].map(mapping)
    return df['vibe'], mapping
def _safe_silhouette(X: np.ndarray, labels: np.ndarray) -> float:
    """Silhouette with guards for degenerate clustering and heavy noise."""
    labels = np.asarray(labels)
    unique = set(labels.tolist())
    if len(unique) <= 1:
        return -1.0

    # If there is noise label -1, compute silhouette only on non-noise points when possible.
    if -1 in unique:
        mask = labels != -1
        if mask.sum() < 10:
            return -1.0
        labels_n = labels[mask]
        if len(set(labels_n.tolist())) <= 1:
            return -1.0
        return float(silhouette_score(X[mask], labels_n))

    return float(silhouette_score(X, labels))


def _safe_calinski_harabasz(X: np.ndarray, labels: np.ndarray) -> float:
    """Calinski-Harabasz with guards for degenerate clustering and heavy noise."""
    labels = np.asarray(labels)
    unique = set(labels.tolist())
    if len(unique) <= 1:
        return -1.0

    # If there is noise label -1, compute CH only on non-noise points when possible.
    if -1 in unique:
        mask = labels != -1
        if mask.sum() < 10:
            return -1.0
        labels_n = labels[mask]
        if len(set(labels_n.tolist())) <= 1:
            return -1.0
        try:
            return float(calinski_harabasz_score(X[mask], labels_n))
        except Exception:
            return -1.0

    try:
        return float(calinski_harabasz_score(X, labels))
    except Exception:
        return -1.0


def _safe_davies_bouldin(X: np.ndarray, labels: np.ndarray) -> float:
    """Davies-Bouldin with guards for degenerate clustering and heavy noise."""
    labels = np.asarray(labels)
    unique = set(labels.tolist())
    if len(unique) <= 1:
        return float('inf')

    if -1 in unique:
        mask = labels != -1
        if mask.sum() < 10:
            return float('inf')
        labels_n = labels[mask]
        if len(set(labels_n.tolist())) <= 1:
            return float('inf')
        try:
            return float(davies_bouldin_score(X[mask], labels_n))
        except Exception:
            return float('inf')

    try:
        return float(davies_bouldin_score(X, labels))
    except Exception:
        return float('inf')


def optimize_params(
    algorithm: str,
    X_pca: np.ndarray,
    k: int | None = None,
    n_trials: int = 25,
    verbose: bool = True,
) -> tuple[dict, optuna.study.Study]:
    """Generic Optuna optimizer for clustering hyper-parameters."""
    algo = str(algorithm).strip().lower()

    def _safe_algo_name(name: str) -> str:
        return (
            str(name)
            .lower()
            .replace(' ', '_')
            .replace('(', '')
            .replace(')', '')
            .replace('+', '')
            .replace('->', 'to')
        )

    def _evaluate_params(params: dict) -> float:
        """Evaluate objective value (Silhouette) for a given param set."""
        try:
            if algo == 'kmeans':
                if k is None:
                    return -1.0
                model = KMeans(n_clusters=k, random_state=RANDOM_STATE, **params)
                labels = model.fit_predict(X_pca)
                return _safe_silhouette(X_pca, labels)

            if algo == 'kmeans_random':
                if k is None:
                    return -1.0
                params = dict(params)
                params.pop('init', None)
                model = KMeans(n_clusters=k, init='random', n_init=10, random_state=RANDOM_STATE, **params)
                labels = model.fit_predict(X_pca)
                return _safe_silhouette(X_pca, labels)

            if algo == 'bisecting_kmeans':
                if k is None:
                    return -1.0
                model = BisectingKMeans(n_clusters=k, random_state=RANDOM_STATE, **params)
                labels = model.fit_predict(X_pca)
                return _safe_silhouette(X_pca, labels)

            if algo == 'agglomerative_ward':
                if k is None:
                    return -1.0
                model = AgglomerativeClustering(n_clusters=k, linkage='ward')
                labels = model.fit_predict(X_pca)
                return _safe_silhouette(X_pca, labels)

            if algo == 'birch':
                if k is None:
                    return -1.0
                model = Birch(n_clusters=k, **params)
                labels = model.fit_predict(X_pca)
                return _safe_silhouette(X_pca, labels)

            if algo == 'gaussian_mixture':
                if k is None:
                    return -1.0
                model = GaussianMixture(n_components=k, covariance_type='spherical', random_state=RANDOM_STATE, **params)
                labels = model.fit_predict(X_pca)
                return _safe_silhouette(X_pca, labels)

            if algo == 'spectral':
                if k is None:
                    return -1.0
                model = SpectralClustering(n_clusters=k, random_state=RANDOM_STATE, **params)
                labels = model.fit_predict(X_pca)
                return _safe_silhouette(X_pca, labels)

            if algo == 'spectral_rbf':
                if k is None:
                    return -1.0
                params = dict(params)
                params.pop('affinity', None)
                model = SpectralClustering(n_clusters=k, affinity='rbf', random_state=RANDOM_STATE, **params)
                labels = model.fit_predict(X_pca)
                return _safe_silhouette(X_pca, labels)

            if algo == 'spectral_knn':
                if k is None:
                    return -1.0
                params = dict(params)
                params.pop('affinity', None)
                model = SpectralClustering(n_clusters=k, affinity='nearest_neighbors', random_state=RANDOM_STATE, **params)
                labels = model.fit_predict(X_pca)
                return _safe_silhouette(X_pca, labels)
        except Exception:
            return -1.0
        return -1.0

    safe = _safe_algo_name(algo)
    cached_best_params_path = os.path.join(OPTUNA_JSON_DIR, f'p2_{safe}_best_params.json')
    if OPTUNA_REUSE and os.path.exists(cached_best_params_path):
        try:
            with open(cached_best_params_path, 'r', encoding='utf-8') as f:
                cached_params = json.load(f)
            # Backward-compat: older caches may contain params we now enforce.
            if algo == 'gaussian_mixture' and isinstance(cached_params, dict):
                cached_params.pop('covariance_type', None)
            cached_score = float(_evaluate_params(cached_params))
            if verbose:
                print("\n" + "=" * 80)
                print(f"OPTUNA REUSE (skip trials): {algorithm}")
                print("=" * 80)
                print(f"   ✅ Loaded best params: {_show_path(cached_best_params_path)}")
                print(f"   ✅ Cached SIL (re-eval): {cached_score:.4f}")
                print("=" * 80 + "\n")
            pseudo_study = SimpleNamespace(
                best_params=dict(cached_params),
                best_value=cached_score,
                study_name=f'p2_{algo}_reuse',
            )
            return dict(cached_params), pseudo_study
        except Exception:
            # If cache is corrupt, fall back to running Optuna.
            pass
    study = optuna.create_study(
        direction='maximize',
        study_name=f'p2_{algo}_optimization',
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )

    if verbose and P2_OPTUNA_TRIAL_LOGS:
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]
        print(f"So luong trials: {n_trials}")
        print(f"[I {ts}] A new study created in memory with name: {study.study_name}")

    def _trial_logger(st: optuna.study.Study, tr: optuna.trial.FrozenTrial) -> None:
        if not (verbose and P2_OPTUNA_TRIAL_LOGS):
            return
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]
        val = tr.value
        try:
            best = st.best_trial
            best_num = best.number
            best_val = best.value
        except Exception:
            best_num = tr.number
            best_val = val
        print(
            f"[I {ts}] Trial {tr.number} finished with value: {val} and parameters: {tr.params}. "
            f"Best is trial {best_num} with value: {best_val}."
        )

    def objective(trial: optuna.Trial) -> float:
        try:
            if algo == 'kmeans':
                if k is None:
                    return -1.0
                params = {
                    'n_init': trial.suggest_int('n_init', 10, 50),
                    'max_iter': trial.suggest_int('max_iter', 200, 600),
                    'tol': trial.suggest_float('tol', 1e-5, 1e-3, log=True),
                    'init': trial.suggest_categorical('init', ['k-means++', 'random']),
                }
                model = KMeans(n_clusters=k, random_state=RANDOM_STATE, **params)
                labels = model.fit_predict(X_pca)
                return _safe_silhouette(X_pca, labels)

            if algo == 'kmeans_random':
                if k is None:
                    return -1.0
                params = {
                    'max_iter': trial.suggest_int('max_iter', 200, 600),
                    'tol': trial.suggest_float('tol', 1e-5, 1e-3, log=True),
                }
                model = KMeans(n_clusters=k, init='random', n_init=10, random_state=RANDOM_STATE, **params)
                labels = model.fit_predict(X_pca)
                return _safe_silhouette(X_pca, labels)

            if algo == 'bisecting_kmeans':
                if k is None:
                    return -1.0
                params = {
                    'bisecting_strategy': trial.suggest_categorical(
                        'bisecting_strategy',
                        ['biggest_inertia', 'largest_cluster'],
                    ),
                    'n_init': trial.suggest_int('n_init', 10, 50),
                    'max_iter': trial.suggest_int('max_iter', 200, 600),
                    'tol': trial.suggest_float('tol', 1e-5, 1e-3, log=True),
                    'init': trial.suggest_categorical('init', ['k-means++', 'random']),
                }
                model = BisectingKMeans(n_clusters=k, random_state=RANDOM_STATE, **params)
                labels = model.fit_predict(X_pca)
                return _safe_silhouette(X_pca, labels)

            if algo == 'agglomerative_ward':
                if k is None:
                    return -1.0
                # Ward has essentially no tunable hyper-params that change labels.
                model = AgglomerativeClustering(n_clusters=k, linkage='ward')
                labels = model.fit_predict(X_pca)
                return _safe_silhouette(X_pca, labels)

            if algo == 'birch':
                if k is None:
                    return -1.0
                params = {
                    'threshold': trial.suggest_float('threshold', 0.1, 1.5),
                    'branching_factor': trial.suggest_int('branching_factor', 30, 150),
                }
                model = Birch(n_clusters=k, **params)
                labels = model.fit_predict(X_pca)
                return _safe_silhouette(X_pca, labels)

            if algo == 'gaussian_mixture':
                if k is None:
                    return -1.0
                params = {
                    'reg_covar': trial.suggest_float('reg_covar', 1e-7, 1e-3, log=True),
                    'max_iter': trial.suggest_int('max_iter', 100, 400),
                    'n_init': trial.suggest_int('n_init', 1, 10),
                    'init_params': trial.suggest_categorical('init_params', ['kmeans', 'random']),
                }
                model = GaussianMixture(n_components=k, covariance_type='spherical', random_state=RANDOM_STATE, **params)
                labels = model.fit_predict(X_pca)
                return _safe_silhouette(X_pca, labels)

            if algo == 'spectral_rbf':
                if k is None:
                    return -1.0
                params = {
                    'gamma': trial.suggest_float('gamma', 0.1, 2.0),
                }
                model = SpectralClustering(n_clusters=k, affinity='rbf', random_state=RANDOM_STATE, **params)
                labels = model.fit_predict(X_pca)
                return _safe_silhouette(X_pca, labels)

            if algo == 'spectral_knn':
                if k is None:
                    return -1.0
                params = {
                    'n_neighbors': trial.suggest_int('n_neighbors', 5, 25),
                }
                model = SpectralClustering(n_clusters=k, affinity='nearest_neighbors', random_state=RANDOM_STATE, **params)
                labels = model.fit_predict(X_pca)
                return _safe_silhouette(X_pca, labels)

        except Exception:
            return -1.0

        return -1.0

    # Hit-style: progress bar controlled via env.
    callbacks = [_trial_logger] if (verbose and P2_OPTUNA_TRIAL_LOGS) else None
    study.optimize(
        objective,
        n_trials=n_trials,
        callbacks=callbacks,
        show_progress_bar=SHOW_OPTUNA_PROGRESS_BAR,
    )

    save_optuna_history(study, algo, verbose=False)
    if verbose:
        print("\n" + "=" * 80)
        print(f"OPTUNA BEST: {algorithm}")
        print("=" * 80)
        print(f"   ✅ Best SIL: {study.best_value:.4f}")
        print(f"   ✅ Best params: {study.best_params}")
        print("=" * 80 + "\n")

    return dict(study.best_params), study


def print_benchmark_table(results: dict, title: str, *, silent: bool = False) -> pd.DataFrame:
    """Print benchmark results like Hit tasks and return sorted DataFrame."""
    df_res = pd.DataFrame(results).T
    # Default sort aligns with best-model criterion (SIL).
    if 'Silhouette' in df_res.columns:
        df_res = df_res.sort_values(by='Silhouette', ascending=False)
    elif 'Calinski-Harabasz' in df_res.columns:
        df_res = df_res.sort_values(by='Calinski-Harabasz', ascending=False)

    if not silent:
        print("\n" + "=" * 80)
        print(title)
        print("=" * 80)
        try:
            print(df_res.to_string(float_format=lambda x: f"{x:.4f}"))
        except Exception:
            print(df_res)
    return df_res

def save_optuna_history(study, algorithm_name, verbose: bool = False):
    """Save Optuna artifacts (Hit-style): best_params JSON (always) + optional full trials JSON."""
    safe = str(algorithm_name).lower().replace(' ', '_').replace('(', '').replace(')', '').replace('+', '').replace('->', 'to')

    # Hit-style: write Optuna JSON under images folder.
    json_dir = OPTUNA_JSON_DIR
    os.makedirs(json_dir, exist_ok=True)

    # Always save best params
    best_params_path = os.path.join(json_dir, f'p2_{safe}_best_params.json')
    with open(best_params_path, 'w', encoding='utf-8') as f:
        json.dump(study.best_params, f, indent=2, ensure_ascii=False)

    # Always save trials to CSV (to reuse without rerunning Optuna)
    try:
        os.makedirs(FINAL_DATA_DIR, exist_ok=True)
        df_trials = study.trials_dataframe()
        if df_trials is not None and (not df_trials.empty) and ('value' in df_trials.columns):
            values = pd.to_numeric(df_trials['value'], errors='coerce')
            df_trials = df_trials.copy()
            df_trials['best_value_so_far'] = values.cummax()
            csv_path = os.path.join(FINAL_DATA_DIR, f'p2_{safe}_optimization.csv')
            df_trials.to_csv(csv_path, index=False, encoding='utf-8-sig')
            if verbose:
                print(f"   💾 Đã lưu Optuna trials CSV: {_show_path(csv_path)}")
    except Exception:
        pass

    filepath = None
    if SAVE_OPTUNA_TRIALS_JSON:
        history = {
            'algorithm': algorithm_name,
            'best_value': study.best_value,
            'best_params': study.best_params,
            'n_trials': len(study.trials),
            'trials': [
                {
                    'number': trial.number,
                    'value': trial.value,
                    'params': trial.params,
                    'state': trial.state.name,
                }
                for trial in study.trials
            ],
        }
        filepath = os.path.join(json_dir, f'p2_{safe}_optimization.json')
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
    
    if verbose:
        print(f"   💾 Đã lưu best params: {_show_path(best_params_path)}")
        if filepath is not None:
            print(f"   💾 Đã lưu lịch sử Optuna: {_show_path(filepath)}")


def run_benchmark_optuna(
    X_pca: np.ndarray,
    k_primary: int,
    n_trials: int = OPTUNA_TRIALS,
) -> tuple[dict, dict[str, dict], str]:
    """Module 4: Chỉ chạy Optuna cho duy nhất thuật toán tốt nhất từ Baseline."""
    results_base, _ = benchmark_10_algorithms(X_pca, k_primary, optimized_params=None, verbose=True)
    _ = print_benchmark_table(results_base, title=f"📊 BASELINE - 9 THUẬT TOÁN (k={k_primary})", silent=False)

    # TỰ ĐỘNG TÌM MODEL TỐT NHẤT (Dựa trên Silhouette cao nhất)
    # Chỉ chọn các model có số cụm cố định theo k (để còn map ra 5 vibes)
    fixed_k_models = {
        'K-Means',
        'K-Means (Random)',
        'MiniBatch KMeans',
        'Bisecting KMeans',
        'Agglomerative (Ward)',
        'Birch',
        'Spectral (KNN)',
    }
    valid_results = {
        name: vals
        for name, vals in results_base.items()
        if name in fixed_k_models and float(vals.get('Silhouette', 0.0)) > 0
    }

    if not valid_results:
        # Fallback: pick the best among fixed-k even if CH <= 0
        valid_results = {name: vals for name, vals in results_base.items() if name in fixed_k_models}

    if not valid_results:
        # Ultimate fallback
        best_model_name = 'K-Means'
        baseline_sil = float(results_base.get(best_model_name, {}).get('Silhouette', 0.0))
        baseline_ch = float(results_base.get(best_model_name, {}).get('Calinski-Harabasz', 0.0))
    else:
        best_model_name = max(valid_results, key=lambda x: float(valid_results[x].get('Silhouette', -1e18)))
        baseline_sil = float(valid_results[best_model_name].get('Silhouette', 0.0))
        baseline_ch = float(valid_results[best_model_name].get('Calinski-Harabasz', 0.0))
    
    print(
        f"   🏆 Best fixed-k model (đủ điều kiện map 5 vibes): {best_model_name} "
        f"(Silhouette: {baseline_sil:.4f} | CH: {baseline_ch:.2f})"
    )
    print("=" * 80)
    print(f"   🔥 BƯỚC 2: Bắt đầu tối ưu hóa Optuna ({n_trials} lượt) cho {best_model_name}...")

    # Mapping tên model sang key của hàm optimize_params
    name_map: dict[str, str] = {
        'K-Means': 'kmeans',
        'K-Means (Random)': 'kmeans_random',
        'Bisecting KMeans': 'bisecting_kmeans',
        'Agglomerative (Ward)': 'agglomerative_ward',
        'Birch': 'birch',
        'Spectral (KNN)': 'spectral_knn',
        # Treat MiniBatch as KMeans-like space
        'MiniBatch KMeans': 'kmeans',
    }
    
    algo_key = name_map.get(best_model_name, 'kmeans')
    
    # CHỈ CHẠY OPTUNA CHO 1 MODEL DUY NHẤT
    optimized_params = {}
    tuned_params, study = optimize_params(algo_key, X_pca, k=k_primary, n_trials=n_trials, verbose=True)

    # Hit-style rollback: only accept tuned params if they improve baseline.
    tuned_score = float(getattr(study, 'best_value', -1.0))
    if tuned_score + 1e-9 < baseline_sil:
        print(
            f"   ⚠️  Optuna không cải thiện (baseline_SIL={baseline_sil:.4f} > optuna_SIL={tuned_score:.4f}) "
            "→ GIỮ BASELINE params"
        )
        tuned_params = {}
    else:
        print(
            f"   ✅ Optuna cải thiện/không tệ hơn baseline (SIL): {baseline_sil:.4f} → {tuned_score:.4f} "
            "→ DÙNG params Optuna"
        )

    optimized_params[algo_key] = tuned_params

    # Xuất ảnh Optuna cho model chiến thắng
    if SAVE_OPTUNA_PLOTS:
        img_path = os.path.join(OPTUNA_IMG_DIR, f'p2_{algo_key}_optimization.png')
        plot_custom_optuna_history(study, best_model_name, baseline_sil, img_path)
        print(f"   ✅ Đã xuất ảnh lịch sử Optuna tại: {_show_path(img_path)}")

    print("\n" + "=" * 80)
    print("📊 BƯỚC 3: LƯU KẾT QUẢ BEST MODEL")
    print("=" * 80)

    # Only evaluate + save the best model (requested).
    try:
        best_clusterer = build_clusterer_from_best(best_model_name, k_primary, optimized_params)
        labels = best_clusterer.fit_predict(X_pca)
        n_clusters = len(set(labels)) - (1 if -1 in set(labels.tolist()) else 0)
        if n_clusters > 1:
            best_sil = float(_safe_silhouette(X_pca, labels))
            best_db = float(_safe_davies_bouldin(X_pca, labels))
            best_ch = float(_safe_calinski_harabasz(X_pca, labels))
        else:
            best_sil, best_db, best_ch = 0.0, float('inf'), 0.0

        used_optuna = bool(optimized_params.get(algo_key))
        row = {
            'Model': best_model_name,
            'AlgoKey': algo_key,
            'k': int(k_primary),
            'UsedOptunaParams': int(used_optuna),
            'Baseline_CH': float(baseline_ch),
            'Baseline_Silhouette': float(baseline_sil),
            'Final_CH': float(best_ch),
            'Final_Silhouette': float(best_sil),
            'Final_DaviesBouldin': float(best_db if np.isfinite(best_db) else np.nan),
            'BestParams': json.dumps(optimized_params.get(algo_key, {}), ensure_ascii=False),
        }
        df_best = pd.DataFrame([row])
        df_best.to_csv(MODEL_COMPARISON_IMG_CSV, index=False, encoding='utf-8-sig')
        print(f"✅ Đã lưu best-model CSV tại: {_show_path(MODEL_COMPARISON_IMG_CSV)}")
    except Exception as e:
        print(f"⚠️  Không thể lưu best-model CSV: {e}")

    # Return minimal results (kept for compatibility; caller doesn't use it)
    return {best_model_name: results_base.get(best_model_name, {})}, optimized_params, best_model_name


def build_clusterer_from_best(best_model_name: str, k: int, optimized_params: dict[str, dict]):
    """Construct the final clustering model (fixed-k) using tuned params when available."""
    name = str(best_model_name)

    if name == 'K-Means':
        params = optimized_params.get('kmeans', {})
        return KMeans(n_clusters=k, random_state=RANDOM_STATE, **params)

    if name == 'K-Means (Random)':
        params = optimized_params.get('kmeans_random', {}).copy()
        # Enforce init=random and fixed n_init.
        params.pop('init', None)
        params.pop('n_init', None)
        return KMeans(n_clusters=k, init='random', n_init=10, random_state=RANDOM_STATE, **params)

    if name == 'MiniBatch KMeans':
        params = optimized_params.get('kmeans', {})
        mb_params = {kk: vv for kk, vv in params.items() if kk != 'n_init'}
        mb_params['n_init'] = params.get('n_init', 10)
        return MiniBatchKMeans(n_clusters=k, random_state=RANDOM_STATE, **mb_params)

    if name == 'Bisecting KMeans':
        params = optimized_params.get('bisecting_kmeans', {})
        return BisectingKMeans(n_clusters=k, random_state=RANDOM_STATE, **params)

    if name == 'Agglomerative (Ward)':
        # Ward linkage requires Euclidean metric.
        return AgglomerativeClustering(n_clusters=k, linkage='ward')

    if name == 'Birch':
        params = optimized_params.get('birch', {})
        return Birch(n_clusters=k, **params)

    if name == 'Spectral (KNN)':
        params = optimized_params.get('spectral_knn', {}).copy()
        params.pop('affinity', None)
        return SpectralClustering(n_clusters=k, affinity='nearest_neighbors', random_state=RANDOM_STATE, **params)

    # fallback
    params = optimized_params.get('kmeans', {})
    return KMeans(n_clusters=k, random_state=RANDOM_STATE, **params)

def benchmark_10_algorithms(X_pca, k, optimized_params=None, verbose: bool = True):
    """Chạy thử nghiệm các thuật toán với tham số đã tối ưu"""
    if verbose:
        print("\n" + "=" * 80)
        print(f"BENCHMARK 9 THUẬT TOÁN (k={k})")
        print("=" * 80)
        print(f"\n{'MODEL':<30} | {'SIL':<8} | {'DB':<8} | {'CH':<12} | {'N_CLUST':<7}")
        print("-" * 80)
    
    # Sử dụng optimized_params nếu có, nếu không dùng default
    if optimized_params is None:
        optimized_params = {}
    
    # K-Means
    if 'kmeans' in optimized_params:
        kmeans_model = KMeans(n_clusters=k, random_state=42, **optimized_params['kmeans'])
    else:
        kmeans_model = KMeans(n_clusters=k, random_state=42, n_init=10)

    # K-Means (Random)
    if 'kmeans_random' in optimized_params:
        kr_params = dict(optimized_params['kmeans_random'])
        kr_params.pop('init', None)
        kr_params.pop('n_init', None)
        kmeans_random_model = KMeans(n_clusters=k, init='random', n_init=10, random_state=42, **kr_params)
    else:
        kmeans_random_model = KMeans(n_clusters=k, init='random', n_init=10, random_state=42)
    
    # MiniBatch KMeans (dùng params tương tự KMeans)
    if 'kmeans' in optimized_params:
        mb_params = {k: v for k, v in optimized_params['kmeans'].items() if k != 'n_init'}
        mb_params['n_init'] = optimized_params['kmeans'].get('n_init', 10)
        minibatch_model = MiniBatchKMeans(n_clusters=k, random_state=42, **mb_params)
    else:
        minibatch_model = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=10)

    # Agglomerative (Ward)
    # Note: Ward has no meaningful tunables here.
    agg_ward_model = AgglomerativeClustering(n_clusters=k, linkage='ward')
    
    # Birch
    if 'birch' in optimized_params:
        birch_model = Birch(n_clusters=k, **optimized_params['birch'])
    else:
        birch_model = Birch(n_clusters=k)
    
    # Spectral (KNN)
    if 'spectral_knn' in optimized_params:
        sk_params = dict(optimized_params['spectral_knn'])
        sk_params.pop('affinity', None)
        spectral_knn_model = SpectralClustering(n_clusters=k, affinity='nearest_neighbors', random_state=42, **sk_params)
    else:
        spectral_knn_model = SpectralClustering(n_clusters=k, affinity='nearest_neighbors', random_state=42)

    models = {
        # 1. Centroid-based (Optimized)
        'K-Means': kmeans_model,
        'K-Means (Random)': kmeans_random_model,
        'MiniBatch KMeans': minibatch_model,
        'Bisecting KMeans': (
            BisectingKMeans(n_clusters=k, random_state=42, **optimized_params['bisecting_kmeans'])
            if 'bisecting_kmeans' in optimized_params
            else BisectingKMeans(n_clusters=k, random_state=42)
        ),
        'Agglomerative (Ward)': agg_ward_model,
        'Birch': birch_model,
        'Spectral (KNN)': spectral_knn_model,
    }
    results = {}
    trained_models = {}
    for name, model in models.items():
        try:
            labels = model.fit_predict(X_pca)
            # Kiểm tra số lượng cụm
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            
            # Chỉ tính điểm nếu tìm được > 1 cụm
            if n_clusters > 1:
                sil = _safe_silhouette(X_pca, labels)
                db = _safe_davies_bouldin(X_pca, labels)
                ch = _safe_calinski_harabasz(X_pca, labels)
                results[name] = {'Silhouette': sil, 'Davies-Bouldin': db, 'Calinski-Harabasz': ch}
                trained_models[name] = (model, labels)
                if verbose:
                    db_disp = db if np.isfinite(db) else float('nan')
                    print(f"{name:<30} | {sil:>8.4f} | {db_disp:>8.4f} | {ch:>12.2f} | {n_clusters:>7d}")
            else:
                if verbose:
                    print(f"{name:<30} | {'NA':>8} | {'NA':>8} | {'NA':>12} | {n_clusters:>7d}")
                results[name] = {'Silhouette': 0, 'Davies-Bouldin': 100, 'Calinski-Harabasz': 0}
        except Exception as e:
            if verbose:
                msg = str(e).replace('\n', ' ')
                print(f"{name:<30} | {'ERR':>8} | {'ERR':>8} | {'ERR':>12} | {'?':>7}  ({msg[:60]})")
            results[name] = {'Silhouette': 0, 'Davies-Bouldin': 100, 'Calinski-Harabasz': 0}
    
    if verbose:
        print("\n" + "=" * 80 + "\n")
    return results, trained_models

def plot_radar_5_vibes(df, vibe_labels, output_dir):
    """Radar chart for the 5 vibe groups."""
    df_plot = df.copy()
    df_plot['vibe'] = vibe_labels

    features = {
        'Tempo': 'tempo_bpm',
        'Energy': 'rms_energy',
        'Beat': 'beat_strength_mean',
        'Sentiment +': 'sentiment_positive',
        'Sentiment -': 'sentiment_negative',
    }
    features = {k: v for k, v in features.items() if v in df_plot.columns}
    if len(features) < 3:
        print("⚠️  Không đủ feature để vẽ radar.")
        return

    vibe_order = [
        "Bùng nổ / Sôi động", 
        "Tươi mới / Yêu đời", 
        "Kịch tính / Da diết", 
        "Sâu lắng / Thấu cảm", 
        "Bình yên / Chữa lành"
    ]
    
    vibe_order = [v for v in vibe_order if v in df_plot['vibe'].unique()]

    prof = df_plot.groupby('vibe')[list(features.values())].mean().reindex(vibe_order)
    scaler_minmax = MinMaxScaler()
    prof_norm = scaler_minmax.fit_transform(prof.values)

    categories = list(features.keys())
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(11, 11), subplot_kw=dict(projection='polar'))
    colors = sns.color_palette("bright", len(prof))
    for i, vibe in enumerate(prof.index.tolist()):
        row = prof_norm[i].tolist()
        values = row + row[:1]
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=vibe, color=colors[i])
        ax.fill(angles, values, alpha=0.1, color=colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=12, fontweight='bold')
    ax.set_yticklabels([])

    plt.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=11)
    plt.title("Radar Chart: 5 Vibes V-Pop (PCA - Direct)", size=16, fontweight='bold', y=1.08)
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, '5vibes_radar.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Đã lưu radar chart tại: {_show_path(save_path)}")

# =============================================================================
# MAIN
# =============================================================================
def main():
    _enable_task_logging(task_dir=IMG_DIR, task_tag=TASK_ID)
    print("\n" + "=" * 80)
    print("BƯỚC 1: TẢI & PREPROCESS")
    print("=" * 80)
    df = pd.read_csv(os.path.join(PROJECT_ROOT, 'final_data', 'data_prepared_for_ML.csv'))
    df_proc, numeric_cols, X_scaled, X_weighted, preproc_artifacts = preprocess_data(df)

    weight_opt_params: dict | None = None
    if APPLY_FEATURE_WEIGHTS and P2_OPTIMIZE_FEATURE_WEIGHTS:
        print("\n" + "=" * 80)
        print("BƯỚC 1B: OPTUNA FEATURE WEIGHTS")
        print("=" * 80)
        # Optimize towards the final objective (5 vibes) to keep weights aligned with business goal.
        weight_opt_params, w_best = optimize_feature_weights(
            X_scaled=X_scaled,
            feature_cols=numeric_cols,
            k=5,
            n_trials=P2_WEIGHT_OPT_TRIALS,
            verbose=True,
        )
        X_weighted = X_scaled * w_best
        preproc_artifacts['feature_weights'] = w_best
        preproc_artifacts['feature_weight_params'] = dict(weight_opt_params or {})

    print("\n" + "=" * 80)
    pct = int(P2_PCA_N_COMPONENTS * 100)
    print(f"BƯỚC 2: PCA (Dynamic {pct}% Variance)")
    print("=" * 80)
    X_pca, pca = run_pca(X_weighted, n_components=P2_PCA_N_COMPONENTS)

    print("\n" + "=" * 80)
    print("BƯỚC 3: AUTO-ELBOW (kneed)")
    print("=" * 80)
    k_sel = run_auto_k(X_pca, 2, 10)
    k_primary = int(k_sel.get('optimal_k', 5)) # Lấy K chuẩn từ dữ liệu
    print(f"✅ Kết quả Auto-Kneed chuẩn: k = {k_primary}")
    
    print("\n" + "=" * 80)
    print("BƯỚC 4: CHẠY BASELINE - SO SÁNH BAN ĐẦU + OPTUNA")
    print("=" * 80)
    results_final, optimized_params, best_name = run_benchmark_optuna(X_pca, k_primary, n_trials=OPTUNA_TRIALS)

    # Primary clustering: use best model from baseline (then tuned by Optuna)
    base_model = build_clusterer_from_best(best_name, k_primary, optimized_params)
    ref_labels = base_model.fit_predict(X_pca)

    consensus_meta: dict | None = None
    if P2_CONSENSUS_CLUSTERING:
        print("\n" + "=" * 80)
        print("BƯỚC 4B: CONSENSUS CLUSTERING (Ensemble)")
        print("=" * 80)
        # Re-benchmark with tuned params to select top-K models for the ensemble.
        results_base2, trained_models = benchmark_10_algorithms(
            X_pca,
            k_primary,
            optimized_params=optimized_params,
            verbose=False,
        )
        fixed_k_models = [
            'K-Means',
            'K-Means (Random)',
            'MiniBatch KMeans',
            'Bisecting KMeans',
            'Agglomerative (Ward)',
            'Birch',
            'Spectral (KNN)',
        ]
        rows = []
        for name in fixed_k_models:
            vals = results_base2.get(name, {})
            rows.append((name, float(vals.get('Silhouette', 0.0))))
        rows.sort(key=lambda t: t[1], reverse=True)
        top_names = [n for n, s in rows if s > 0][: max(1, int(P2_CONSENSUS_TOP_K))]
        if best_name not in top_names:
            top_names = [best_name] + [n for n in top_names if n != best_name]
            top_names = top_names[: max(1, int(P2_CONSENSUS_TOP_K))]

        aligned_label_runs = []
        used = []
        for nm in top_names:
            if nm == best_name:
                continue
            if nm in trained_models:
                _m, lab = trained_models[nm]
                aligned = _align_labels_to_reference(ref_labels, lab, k_primary)
                aligned_label_runs.append(aligned)
                used.append(nm)

        if len(aligned_label_runs) >= 1:
            consensus_labels = _vote_consensus_labels(ref_labels, aligned_label_runs, k_primary)
            consensus_model = _fit_consensus_kmeans(X_pca, consensus_labels, k_primary)
            final_labels = _align_labels_to_reference(ref_labels, consensus_model.predict(X_pca), k_primary)
            main_model = consensus_model
            main_labels = final_labels
            consensus_meta = {
                'enabled': True,
                'top_k': int(P2_CONSENSUS_TOP_K),
                'models_used': [best_name] + used,
            }
            print(f"✅ Consensus built from: {[best_name] + used}")
        else:
            main_model = base_model
            main_labels = _align_labels_to_reference(ref_labels, ref_labels, k_primary)
            consensus_meta = {
                'enabled': False,
                'reason': 'not_enough_valid_models_for_voting',
                'top_k': int(P2_CONSENSUS_TOP_K),
            }
            print("⚠️  Consensus skipped (not enough valid models).")
    else:
        main_model = base_model
        main_labels = _align_labels_to_reference(ref_labels, ref_labels, k_primary)

    stability_report: dict | None = None
    aligned_boot_runs: list[np.ndarray] = []
    if P2_BOOTSTRAP_STABILITY:
        print("\n" + "=" * 80)
        print("BƯỚC 4C: BOOTSTRAP STABILITY VALIDATION")
        print("=" * 80)
        # Ensure we bootstrap a predict-capable model.
        if hasattr(main_model, 'cluster_centers_'):
            centers = np.asarray(main_model.cluster_centers_)
        else:
            tmp = _fit_consensus_kmeans(X_pca, main_labels, k_primary)
            centers = np.asarray(tmp.cluster_centers_)

        def _builder():
            return KMeans(n_clusters=k_primary, init=centers, n_init=1, random_state=RANDOM_STATE)

        stability_report = bootstrap_cluster_stability(
            clusterer_builder=_builder,
            X_pca=X_pca,
            ref_labels=main_labels,
            k=k_primary,
            n_boot=P2_BOOTSTRAP_N,
            sample_frac=P2_BOOTSTRAP_SAMPLE_FRAC,
        )
        print(f"✅ Bootstrap stability: mean agreement = {stability_report.get('mean_agreement', float('nan')):.4f}")

        if P2_OPTIMIZE_IDEAL_VECTORS:
            aligned_boot_runs = bootstrap_collect_aligned_label_runs(
                clusterer_builder=_builder,
                X_pca=X_pca,
                ref_labels=main_labels,
                k=k_primary,
                n_boot=P2_BOOTSTRAP_N,
                sample_frac=P2_BOOTSTRAP_SAMPLE_FRAC,
            )

    print("\n" + "=" * 80)
    print("BƯỚC 5: DIRECT MAPPING 5 VIBES")
    print("=" * 80)
    df_proc = df_proc.copy()
    df_proc['cluster_main'] = main_labels

    ideal_vibes_used = DEFAULT_IDEAL_VIBES
    ideal_opt_enabled = False
    if P2_OPTIMIZE_IDEAL_VECTORS and len(aligned_boot_runs) > 0:
        # Prepare mapping-space matrix (global scaling) once.
        map_features = []
        for c in ['beat_strength_mean', 'rms_energy', 'sentiment_negative']:
            if c in df_proc.columns:
                map_features.append(c)
        if len(map_features) >= 3:
            X_map = df_proc[map_features].fillna(df_proc[map_features].median(numeric_only=True)).values
            X_map01 = MinMaxScaler().fit_transform(X_map)
            print("\n" + "=" * 80)
            print("BƯỚC 5B: OPTUNA IDEAL VECTORS (Vibe Mapping)")
            print("=" * 80)
            ideal_vibes_used = optimize_ideal_vibes(
                X_map01=X_map01,
                ref_cluster_labels=main_labels,
                aligned_cluster_label_runs=aligned_boot_runs,
                k=k_primary,
                n_trials=P2_IDEAL_OPT_TRIALS,
                verbose=True,
            )
            ideal_opt_enabled = True
        else:
            print("⚠️  Skip ideal-vectors opt (missing mapping features).")

    final_vibes, vibe_map = assign_5_vibes_with_ideals(
        df_proc,
        cluster_col='cluster_main',
        ideal_vibes=ideal_vibes_used,
    )
    df_proc['vibe'] = final_vibes

    # Metrics on final 5 vibes
    vibe_codes = pd.Categorical(df_proc['vibe']).codes
    sil = float(silhouette_score(X_pca, vibe_codes))
    db = float(davies_bouldin_score(X_pca, vibe_codes))
    ch = float(calinski_harabasz_score(X_pca, vibe_codes))
    print("📌 METRICS (FINAL 5 VIBES)")
    print(f"Silhouette:      {sil:.4f}")
    print(f"Davies-Bouldin:  {db:.4f}")
    print(f"Calinski-Harabasz: {ch:.2f}")

    # Radar chart
    plot_radar_5_vibes(df_proc, df_proc['vibe'], IMG_DIR)

    # Export CSV
    os.makedirs(IMG_DIR, exist_ok=True)
    df_proc.to_csv(FINAL_CSV_PATH, index=False, encoding='utf-8-sig')
    print(f"✅ Đã xuất CSV tại: {_show_path(FINAL_CSV_PATH)}")

    meta = {
        'k_primary': int(k_primary),
        'auto_k': k_sel,
        'vibe_map': vibe_map,
        'best_model_name': best_name,
        'feature_weight_optuna': {
            'enabled': bool(APPLY_FEATURE_WEIGHTS and P2_OPTIMIZE_FEATURE_WEIGHTS),
            'params': weight_opt_params,
        },
        'consensus': consensus_meta,
        'bootstrap_stability': stability_report,
        'ideal_vectors_optuna': {
            'enabled': bool(ideal_opt_enabled),
            'n_boot_used': int(len(aligned_boot_runs)) if aligned_boot_runs is not None else 0,
        },
    }

    model_data = {
        'pkl_schema_version': 3,
        'task_id': TASK_ID,
        'data_source': 'final_data/data_prepared_for_ML.csv',
        'preprocess_artifacts': preproc_artifacts,
        'imputer': preproc_artifacts.get('imputer'),
        'scaler': preproc_artifacts.get('scaler'),
        'pca': pca,
        'numeric_features': numeric_cols,
        'best_algorithm': best_name,
        'clusterer': main_model,
        'clusterer_main_name': best_name,
        'kmeans_main': None,
        'kmeans_sub_ballad': None,
        'meta': meta,
        'optuna_best_params': optimized_params,
        'metrics': {
            'silhouette': sil,
            'davies_bouldin': db,
            'calinski_harabasz': ch,
        },
        'ideal_vibes': {k: np.asarray(v, dtype=float) for k, v in ideal_vibes_used.items()},
    }

    # Save feature names
    try:
        with open(FEATURE_NAMES_PATH, 'w', encoding='utf-8') as f:
            json.dump(list(numeric_cols), f, ensure_ascii=False, indent=2)
        print(f"✅ Đã lưu feature names tại: {_show_path(FEATURE_NAMES_PATH)}")
    except Exception as e:
        print(f"⚠️  Không thể lưu feature names: {e}")


    print(f"\n💾 FILES ĐÃ LƯU:")
    try:
        joblib.dump(model_data, MODEL_PATH)
        print(f"✅ Đã lưu best model (PKL) tại: {_show_path(MODEL_PATH)}")
    except Exception as e:
        print(f"⚠️  Không thể lưu PKL: {e}")
    print(f"   • Best model:        {_show_path(MODEL_PATH)}")
    print(f"   • Feature names:     {_show_path(FEATURE_NAMES_PATH)}")
    print(f"   • Final CSV:         {_show_path(FINAL_CSV_PATH)}\n")

if __name__ == "__main__":
    main()