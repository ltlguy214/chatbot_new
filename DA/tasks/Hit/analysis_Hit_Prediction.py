'''
Prediction nên không thực hiện tối ưu ngưỡng
TimeSeriesSplit(spotify_release_date) for Hit Prediction
CalibratedClassifierCV 
confusion_matrix

best model = HIT_F1  
optuna = average_precision
tối ưu ngưỡng = Hit F1

Mô hình tốt nhất: Logistic Regression (Baseline với Threshold Tuning).

Ngưỡng cắt tối ưu (Threshold): 0.4396.

Hit F1-Score: 0.5209.

Hit Recall: 0.7457 (74.57%).

Hit Precision: 0.4002 (40.02%).

Accuracy: 0.5356.
'''
import sys
import atexit
from datetime import datetime
from sklearn.base import clone

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
import json  # THÊM: Để lưu params

from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    average_precision_score,
    precision_recall_curve,
)
# Scikit-learn Imports
from sklearn.model_selection import TimeSeriesSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel

from DA.utils.sklearn_utils import sparse_to_dense

# Optional: resampling (must be inside Pipeline/CV to avoid leakage)
_HAS_IMBLEARN = True
try:
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.under_sampling import RandomUnderSampler, TomekLinks
except Exception:
    _HAS_IMBLEARN = False

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, 
    ExtraTreesClassifier, VotingClassifier, StackingClassifier
)
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

# Metrics
from sklearn.model_selection import cross_val_score

warnings.filterwarnings('ignore')

# Optuna scoring (objective metric)
# Default: 'f1' (the user's primary KPI)
# Option:  'average_precision' (PR-AUC) for ranking-focused optimization
OPTUNA_SCORING = os.getenv('OPTUNA_SCORING', 'f1').strip()
if OPTUNA_SCORING not in {'average_precision', 'f1'}:
    print(f"⚠️ OPTUNA_SCORING='{OPTUNA_SCORING}' không hợp lệ → fallback 'average_precision'")
    OPTUNA_SCORING = 'average_precision'

# -----------------------------------------------------------------------------
# Supervised feature selection (optional)
# - Must be inside Pipeline/CV to avoid leakage.
# - Keeps binary 0/1 cols untouched (selection happens after preprocessing).
# -----------------------------------------------------------------------------
# Default is ON. Set env `P0_FEATURE_SELECTION=none` to disable.
P0_FEATURE_SELECTION = os.getenv('P0_FEATURE_SELECTION', 'tree').strip().lower()
P0_SFM_C = float(os.getenv('P0_SFM_C', '0.05'))
P0_SFM_THRESHOLD = os.getenv('P0_SFM_THRESHOLD', 'median').strip()
_P0_SFM_MAX_FEATURES_RAW = os.getenv('P0_SFM_MAX_FEATURES', '').strip()
P0_SFM_MAX_FEATURES = int(_P0_SFM_MAX_FEATURES_RAW) if _P0_SFM_MAX_FEATURES_RAW.isdigit() else None

# Tree-based selector params (used when P0_FEATURE_SELECTION in {'tree', 'extratrees', 'et'})
P0_TREE_N_ESTIMATORS = int(os.getenv('P0_TREE_N_ESTIMATORS', '500'))
_P0_TREE_MAX_DEPTH_RAW = os.getenv('P0_TREE_MAX_DEPTH', '').strip()
P0_TREE_MAX_DEPTH = int(_P0_TREE_MAX_DEPTH_RAW) if _P0_TREE_MAX_DEPTH_RAW.isdigit() else None
P0_TREE_MIN_SAMPLES_LEAF = int(os.getenv('P0_TREE_MIN_SAMPLES_LEAF', '5'))

# -----------------------------------------------------------------------------
# Train-only resampling (optional)
# - Apply ONLY on TRAIN folds via imblearn Pipeline.
# - Default: TomekLinks (clean boundary) + RandomUnderSampler to 1:1.
# -----------------------------------------------------------------------------
P0_RESAMPLING = os.getenv('P0_RESAMPLING', 'tomek_rus').strip().lower()  # none|tomek|rus|tomek_rus
try:
    P0_RUS_STRATEGY = float(os.getenv('P0_RUS_STRATEGY', '1.0'))
except Exception:
    P0_RUS_STRATEGY = 1.0

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
    print(f"📝 Log file: {latest_path}", flush=True)
    print(
        f"🕒 Started: {datetime.now().isoformat(timespec='seconds')} | Script: {os.path.basename(__file__)}",
        flush=True,
    )
    return latest_path

# SHAP is intentionally separated to a standalone runner for speed.
# Use: DA/SHAP_explain/run_shap_all_tasks.py

def _get_y_score_for_pr_auc(model, X):
    """Return a continuous score for PR-AUC (prefer proba; fallback to decision_function; else predict)."""
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, 'decision_function'):
        return model.decision_function(X)
    return model.predict(X)


def _metric_label_for_optuna() -> str:
    return 'PR-AUC' if OPTUNA_SCORING == 'average_precision' else 'F1'


def compute_scale_pos_weight(y):
    """Return ratio = count(non-hit) / count(hit) for binary labels {0,1}."""
    y = pd.Series(y)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos <= 0:
        return 1.0
    return float(n_neg) / float(n_pos)

from DA.SHAP_explain.shap_artifact import ShapCacheConfig, build_shap_cache
from DA.models.topic_mapping import rename_topics_for_report, rename_topics_in_feature_names

# =============================================================================
# 1. LOAD DỮ LIỆU
# =============================================================================
FILE_DATA = Path('final_data') / 'data_prepared_for_ML.csv'

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
def prepare_data(df):
    # 1. TẠO TARGET: IS_HIT
    target_col = 'is_hit'
    if target_col not in df.columns:
        raise ValueError("❌ Thiếu cột label 'is_hit'!")
        
    df['target'] = df[target_col].apply(lambda x: 1 if (x == 1 or str(x).lower() == 'true') else 0)

    # 2. Lọc Feature (Cập nhật danh sách loại bỏ để tránh Leakage)
    # Thêm các tên cột biến mới của bạn vào danh sách ignore nếu chúng là ID hoặc text
    cols_ignore = [  #sym:cols_ignore
        'spotify_track_id', 'title', 'artists', 'spotify_release_date', 'genres', #IDs
        'is_hit', 'target', 'spotify_popularity', #target leakage
        'final_sentiment',
        # 8. BIẾN LOẠI BỎ DO ĐA CỘNG TUYẾN
        'mfcc2_mean',           # Tương quan cực cao với rms_energy/âm lượng
        'spectral_rolloff',     # Trùng lặp thông tin với spectral_centroid
        'noun_count',           # Đã có lyric_total_words đại diện
        'verb_count',           # Đã có lyric_total_words đại diện
        'tempo_stability',        # Tương quan cao với tempo
        'spectral_contrast_band3_mean', # VIF ~2000 (Dư thừa dải tần giữa)
        'spectral_contrast_band4_mean', # VIF ~2200
        'spectral_contrast_band5_mean'  # VIF ~2000
    ]

    # Strict leakage guardrails: exclude any post-release success proxies if present.
    leakage_patterns = (
        'spotify_popularity',
        'popularity',
        'stream',
        'streams',
        'view',
        'views',
        'like',
        'likes',
        'chart',
        'rank',
    )
    dynamic_leakage_cols = [
        c for c in df.columns
        if any(pat in str(c).lower() for pat in leakage_patterns)
    ]
    if dynamic_leakage_cols:
        cols_ignore = sorted(set(cols_ignore) | set(dynamic_leakage_cols))
        print(
            "\n🛡️  Leakage guard: loại bỏ các cột nghi ngờ target leakage (pattern match): "
            + ", ".join(dynamic_leakage_cols[:25])
            + (" ..." if len(dynamic_leakage_cols) > 25 else "")
        )
    
    # Tự động lấy các cột số còn lại làm numeric features
    # IMPORTANT: Không scale các cột one-hot 0/1 (key_*, genre_*), vì scaling sẽ làm mất bản chất nhị phân.
    candidate_numeric = [
        c for c in df.columns
        if c not in cols_ignore
        and pd.api.types.is_numeric_dtype(df[c])
        and not str(c).startswith('genre_')
    ]

    binary_feats = []
    numeric_cont_feats = []
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

    # Giữ `numeric_feats` để tương thích phần code còn lại (đúng thứ tự transformer)
    numeric_feats = numeric_cont_feats + binary_feats
    
    df['final_sentiment'] = df['final_sentiment'].fillna('Neutral')
    return df, numeric_feats, numeric_cont_feats, binary_feats

# =============================================================================
# 3. QUY TRÌNH HUẤN LUYỆN
# =============================================================================
def run_full_analysis_task1():
    _enable_task_logging(task_dir=os.path.join('DA', 'tasks', 'Hit'), task_tag='Hit')
    print(
        f"ℹ️  OPTUNA_SCORING={OPTUNA_SCORING!r} | Selection/Rollback: CV-on-TRAIN only (TEST is report-only)",
        flush=True,
    )

    # Fast path: reuse the last successfully saved model to avoid re-running all baselines/Optuna.
    # Default ON. Set `P0_REUSE_MODEL=0` to force retraining.
    try:
        reuse_model = os.getenv('P0_REUSE_MODEL', '1') == '1'
    except Exception:
        reuse_model = True
    cached_model_path = Path('DA') / 'models' / 'best_model_p0.pkl'
    if reuse_model and cached_model_path.exists():
        print(
            f"✅ Found cached model: {cached_model_path} → skip training. "
            "(Set P0_REUSE_MODEL=0 để train lại)",
            flush=True,
        )
        return

    # === SET GLOBAL SEED ĐỂ ĐẢM BẢO KẾT QUẢ NHẤT QUÁN ===
    np.random.seed(42)
    
    df = load_data()
    if df is None: return

    df_clean, numeric_feats, numeric_cont_feats, binary_feats = prepare_data(df)

    n_hit = len(df_clean[df_clean['target'] == 1])
    n_non_hit = len(df_clean[df_clean['target'] == 0])

    print(f"📊 Phân bố tập dữ liệu (GỐC - real-world prevalence): {n_hit} Hit và {n_non_hit} Non-hit")
    print(f"   Tổng: {len(df_clean)} bài hát")
    print(
        "   ⚙️  Mặc định: resampling chỉ trên TRAIN/fold (TomekLinks + RandomUnderSampler 1:1). "
        "Nếu tắt resampling: dùng cost-sensitive learning: class_weight='balanced' + scale_pos_weight."
    )
    # 2. Biến phân loại
    cat_feats = ['final_sentiment'] if 'final_sentiment' in df_clean.columns else []

    # X và y (Dataset mới không còn cột lyric/TF-IDF)
    X = df_clean[numeric_feats + cat_feats]
    y = df_clean['target']

    # TimeSeries safety: chuẩn hóa + sort theo spotify_release_date trước khi holdout
    if 'spotify_release_date' in df_clean.columns:
        def fix_date(val):
            val = str(val).strip()
            if val == '' or val.lower() in {'nan', 'nat', 'none'}:
                return '1900-01-01'
            if len(val) == 4 and val.isdigit():
                return val + '-01-01'
            if len(val) == 7 and val[:4].isdigit() and val[4] == '-' and val[5:7].isdigit():
                return val + '-01'
            return val

        df_clean['spotify_release_date'] = df_clean['spotify_release_date'].apply(fix_date)
        df_clean['spotify_release_date'] = pd.to_datetime(df_clean['spotify_release_date'], errors='coerce')
        df_clean['spotify_release_date'] = df_clean['spotify_release_date'].fillna(pd.Timestamp('1900-01-01'))
        df_clean = df_clean.sort_values('spotify_release_date').reset_index(drop=True)
        X = df_clean[numeric_feats + cat_feats]
        y = df_clean['target']

    # Robustness: allow NaNs (they will be imputed inside the pipeline).
    if X.isna().any().any():
        na_cols = X.columns[X.isna().any()].tolist()
        print(
            "⚠️  Phát hiện missing values trong dữ liệu đầu vào. "
            "Pipeline sẽ tự SimpleImputer(strategy='median') cho biến numeric continuous. "
            f"Cột bị thiếu: {na_cols[:20]}" + (" ..." if len(na_cols) > 20 else "")
        )

    num_cat_ohe = df_clean['final_sentiment'].nunique() if 'final_sentiment' in df_clean.columns else 0
    
    print("\n" + "="*60)
    print("🔍 KIỂM TRA CHI TIẾT CÁC BIẾN ĐẦU VÀO (TASK 1 - HIT PREDICTION)")
    print("="*60)
    print(f"1️⃣ Số lượng biến số (Continuous - sẽ Scale): {len(numeric_cont_feats)}")
    print(np.array(numeric_cont_feats))
    print("-" * 60)
    print(f"1️⃣b Số lượng biến nhị phân (0/1 - KHÔNG Scale): {len(binary_feats)}")
    print(np.array(binary_feats))
    print("-" * 60)
    print(f"1️⃣c Tổng số biến số dùng cho model: {len(numeric_feats)}")
    print("-" * 60)
    
    # In thêm phần biến phân loại
    print(f"2️⃣ Số lượng biến phân loại (Sentiment OHE): {num_cat_ohe}")
    print(f"   Các nhãn tìm thấy: {df_clean['final_sentiment'].unique()}")
    print("-" * 60)

    total_feats = len(numeric_feats) + num_cat_ohe
    
    print("-" * 60)
    print(f"✅ TỔNG CỘNG SỐ BIẾN ĐƯA VÀO MODEL: {total_feats}")
    print("="*60 + "\n")

    # HOLDOUT theo thời gian: 80/20 (không shuffle)
    n_total = len(df_clean)
    split_point = int(n_total * 0.8)
    if split_point <= 0 or split_point >= n_total:
        raise ValueError(f"❌ Split 80/20 không hợp lệ với n_total={n_total}")

    train_idx = np.arange(0, split_point)
    test_idx = np.arange(split_point, n_total)

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Real-world ratio (used for logging; and for model weighting only when resampling is OFF)
    scale_pos_weight = compute_scale_pos_weight(y_train)
    print(f"⚖️  non_hit/hit (Train) = {scale_pos_weight:.4f}")

    # Sanity check: phân bố nhãn theo split
    train_hit_rate = float(np.mean(y_train)) if len(y_train) else 0.0
    test_hit_rate = float(np.mean(y_test)) if len(y_test) else 0.0
    print(f"📌 Holdout 80/20 (time-based) → Train={len(y_train)} | Test={len(y_test)}")
    print(f"   • Hit-rate Train: {train_hit_rate:.3f} | Test: {test_hit_rate:.3f}")
    print("-" * 60)
    print(f"📅 Tập Train: từ {df_clean.iloc[train_idx]['spotify_release_date'].min().date()} đến {df_clean.iloc[train_idx]['spotify_release_date'].max().date()} ({len(train_idx)} bài)")
    print(f"📅 Tập Test : từ {df_clean.iloc[test_idx]['spotify_release_date'].min().date()} đến {df_clean.iloc[test_idx]['spotify_release_date'].max().date()} ({len(test_idx)} bài)")
    print("-" * 60)
    # Transformer cho biến số
    # - Continuous numeric: Impute(median) + Scale trong CV/pipeline (không leak)
    # - Binary 0/1 one-hot: passthrough (KHÔNG scale)
    numeric_transformer = Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
        ]
    )
    
    # 3. CẬP NHẬT: Thêm Transformer cho biến phân loại (OHE)
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    transformers = []
    if numeric_cont_feats:
        transformers.append(('num', numeric_transformer, numeric_cont_feats))
    if binary_feats:
        transformers.append(('bin', 'passthrough', binary_feats))
    transformers.append(('cat', categorical_transformer, cat_feats))  # <-- OHE sentiment

    preprocessor = ColumnTransformer(transformers=transformers)

    # -------------------------------------------------------------------------
    # Train-only resampling configuration (inside Pipeline/CV)
    # -------------------------------------------------------------------------
    sampler_steps = []
    _resampling_mode = (P0_RESAMPLING or 'none').strip().lower()
    if _resampling_mode in {'0', 'off', 'false', 'none', 'no'}:
        _resampling_mode = 'none'

    if _resampling_mode != 'none' and not _HAS_IMBLEARN:
        print(
            "⚠️  P0_RESAMPLING được bật nhưng thiếu package `imbalanced-learn`. "
            "Cài bằng: pip install imbalanced-learn. Tạm thời chạy KHÔNG resampling.",
            flush=True,
        )
        _resampling_mode = 'none'

    if _resampling_mode in {'tomek', 'tomek_rus', 'tomek+rus', 'clean'}:
        sampler_steps.append(('tomek', TomekLinks()))
    if _resampling_mode in {'rus', 'tomek_rus', 'tomek+rus', 'under', 'undersample'}:
        sampler_steps.append(('rus', RandomUnderSampler(sampling_strategy=P0_RUS_STRATEGY, random_state=42)))

    resampling_enabled = len(sampler_steps) > 0
    MODEL_CLASS_WEIGHT = None if resampling_enabled else 'balanced'
    MODEL_SCALE_POS_WEIGHT = 1.0 if resampling_enabled else float(scale_pos_weight)

    if resampling_enabled:
        print(
            f"🧪 Resampling (TRAIN-only): mode={_resampling_mode} | TomekLinks + RUS(sampling_strategy={P0_RUS_STRATEGY}) | "
            "Disable class_weight/scale_pos_weight to avoid double-balancing.",
            flush=True,
        )
    else:
        print(
            f"🧪 Resampling: OFF | class_weight='balanced' + scale_pos_weight={MODEL_SCALE_POS_WEIGHT:.4f}",
            flush=True,
        )

    # Optional: supervised feature selection (fit only on TRAIN folds via Pipeline/CV)
    feature_selector = None
    selector_requires_dense = False

    _l1_modes = {'sfm', 'l1', 'l1-logreg', 'selectfrommodel'}
    _tree_modes = {'tree', 'extratrees', 'et'}

    if P0_FEATURE_SELECTION in _l1_modes:
        sfm_estimator = LogisticRegression(
            penalty='l1',
            solver='saga',
            C=P0_SFM_C,
            class_weight=MODEL_CLASS_WEIGHT,
            random_state=42,
            max_iter=4000,
        )
        feature_selector = SelectFromModel(
            estimator=sfm_estimator,
            threshold=P0_SFM_THRESHOLD,
            max_features=P0_SFM_MAX_FEATURES,
        )

    elif P0_FEATURE_SELECTION in _tree_modes:
        # Tree-based selection: robust to non-linearities; tends to be less sparse than L1.
        # Requires dense input for sklearn tree ensembles.
        sfm_estimator = ExtraTreesClassifier(
            n_estimators=P0_TREE_N_ESTIMATORS,
            max_depth=P0_TREE_MAX_DEPTH,
            min_samples_leaf=P0_TREE_MIN_SAMPLES_LEAF,
            class_weight=MODEL_CLASS_WEIGHT,
            random_state=42,
            n_jobs=1,
        )
        feature_selector = SelectFromModel(
            estimator=sfm_estimator,
            threshold=P0_SFM_THRESHOLD,
            max_features=P0_SFM_MAX_FEATURES,
        )
        selector_requires_dense = True

    _selector_desc = ""
    if feature_selector is not None:
        if P0_FEATURE_SELECTION in _l1_modes:
            _selector_desc = (
                f" | selector=L1-LogReg(C={P0_SFM_C}, threshold={P0_SFM_THRESHOLD}, max_features={P0_SFM_MAX_FEATURES})"
            )
        elif P0_FEATURE_SELECTION in _tree_modes:
            _selector_desc = (
                f" | selector=ExtraTrees(n_estimators={P0_TREE_N_ESTIMATORS}, max_depth={P0_TREE_MAX_DEPTH}, "
                f"min_samples_leaf={P0_TREE_MIN_SAMPLES_LEAF}, threshold={P0_SFM_THRESHOLD}, max_features={P0_SFM_MAX_FEATURES})"
            )

    print(
        f"🧩 Feature selection: {P0_FEATURE_SELECTION}" + _selector_desc,
        flush=True,
    )

    def _build_pipe_p0(clf):
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

    # TimeSeries CV chỉ dùng trong TRAIN (tránh leakage)
    tscv_inner = TimeSeriesSplit(n_splits=5)
        
    # =============================================================================
    # 3. BƯỚC 1: CHẠY BASELINE - SO SÁNH BAN ĐẦU
    # =============================================================================
    print("\n" + "="*80)
    print("📊 BƯỚC 1: CHẠY BASELINE - SO SÁNH BAN ĐẦU")
    print("="*80)
    
    # Định nghĩa tất cả models với tham số mặc định
    baseline_models = {
        'Extra Trees': ExtraTreesClassifier(n_estimators=300, class_weight=MODEL_CLASS_WEIGHT, random_state=42),
        'AdaBoost': AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1), n_estimators=300, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=300, class_weight=MODEL_CLASS_WEIGHT, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=300, learning_rate=0.1, subsample=0.8, random_state=42),
        'MLP (Neural Net)': MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=1000, early_stopping=True, random_state=42),
        # 3 "chuyên gia" đa dạng (imbalanced-safe)
        'XGBoost': XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            random_state=42,
            eval_metric='logloss',
            scale_pos_weight=MODEL_SCALE_POS_WEIGHT,
            n_jobs=-1,
        ),
        'SVM': SVC(kernel='rbf', probability=True, class_weight=MODEL_CLASS_WEIGHT, random_state=42),
        'Logistic Regression': LogisticRegression(class_weight=MODEL_CLASS_WEIGHT, random_state=42, max_iter=2000),
        'LightGBM': LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            verbose=-1,
            scale_pos_weight=MODEL_SCALE_POS_WEIGHT,
        ),
    }
    baseline_results = []
    baseline_pipelines = {}  # Lưu pipelines để rollback nếu cần
    print(f"\n{'MODEL':<35} | {'CV_'+_metric_label_for_optuna():<8} | {'TEST_AP':<8} | {'TEST_ACC':<8} | {'HIT_P':<7} | {'HIT_R':<7} | {'HIT_F1':<7}")
    print("-" * 105)
    
    for name, model in baseline_models.items():
        try:
            pipeline = _build_pipe_p0(model)

            cv_scores = cross_val_score(
                pipeline,
                X_train,
                y_train,
                cv=tscv_inner,
                scoring=OPTUNA_SCORING,
                n_jobs=-1,
            )
            cv_pr_auc = float(np.mean(cv_scores))

            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            y_score = _get_y_score_for_pr_auc(pipeline, X_test)
            test_pr_auc = float(average_precision_score(y_test, y_score))
            test_acc = float(accuracy_score(y_test, y_pred))

            # Metrics for Hit class (1)
            hit_prec = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
            hit_rec = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
            hit_f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)

            # Keep weighted metrics (useful for debugging / compatibility)
            f1_w = f1_score(y_test, y_pred, average='weighted')
            prec_w = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            rec_w = recall_score(y_test, y_pred, average='weighted', zero_division=0)

            baseline_results.append({
                'Model': name,
                'CV_PR_AUC': cv_pr_auc,
                'Test_PR_AUC': test_pr_auc,
                'Test_Accuracy': test_acc,
                'Hit_Precision': hit_prec,
                'Hit_Recall': hit_rec,
                'Hit_F1': hit_f1,
                'F1-Score': f1_w,
                'Precision': prec_w,
                'Recall': rec_w,
            })
            baseline_pipelines[name] = pipeline  # Lưu pipeline
            print(f"{name:<35} | {cv_pr_auc:.4f}   | {test_pr_auc:.4f}   | {test_acc:.4f}   | {hit_prec:.4f} | {hit_rec:.4f} | {hit_f1:.4f}")
        except Exception as e:
            print(f"❌ Lỗi {name}: {e}")

    # -----------------------------
    # Baseline Ensembles: XGB + SVM + LR
    # - Weighted soft voting by CV metric on TRAIN
    # - Stacking uses TimeSeriesSplit to prevent future leakage
    # -----------------------------
    def _weights_from_pr_auc(pr_auc_triplet):
        pr_auc_triplet = [float(x) for x in pr_auc_triplet]
        order = list(np.argsort(-np.array(pr_auc_triplet)))  # desc
        weights = [0, 0, 0]
        for w, idx in zip([3, 2, 1], order):
            weights[int(idx)] = w
        return weights

    expert_names = ['XGBoost', 'SVM', 'Logistic Regression']
    expert_cv = [
        next((r['CV_PR_AUC'] for r in baseline_results if r['Model'] == nm), 0.0)
        for nm in expert_names
    ]
    voting_weights = _weights_from_pr_auc(expert_cv)
    print(f"\n🗳️  Voting weights theo CV {_metric_label_for_optuna()} (XGB, SVM, LR) = {expert_cv} → weights={voting_weights}")

    xgb_base = baseline_models['XGBoost']
    svm_base = baseline_models['SVM']
    lr_base = baseline_models['Logistic Regression']

    voting = VotingClassifier(
        estimators=[('xgb', xgb_base), ('svm', svm_base), ('lr', lr_base)],
        voting='soft',
        weights=voting_weights,
    )
    # NOTE: StackingClassifier uses cross_val_predict internally.
    # Some partitioning CVs (e.g., TimeSeriesSplit) may leave early samples never in test,
    # causing "partitions" errors. Use cv=5 (default StratifiedKFold) for stability.
    stacking = StackingClassifier(
        estimators=[('xgb', xgb_base), ('svm', svm_base), ('lr', lr_base)],
        final_estimator=LogisticRegression(class_weight=MODEL_CLASS_WEIGHT, random_state=42, max_iter=2000),
        cv=5,
        n_jobs=-1,
        passthrough=False,
    )

    for name, model in {
        'Voting Ensemble (XGB+SVM+LR)': voting,
        'Stacking Ensemble (XGB+SVM+LR)': stacking,
    }.items():
        try:
            pipeline = _build_pipe_p0(model)
            cv_scores = cross_val_score(
                pipeline,
                X_train,
                y_train,
                cv=tscv_inner,
                scoring=OPTUNA_SCORING,
                n_jobs=-1,
            )
            cv_pr_auc = float(np.mean(cv_scores))

            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            y_score = _get_y_score_for_pr_auc(pipeline, X_test)
            test_pr_auc = float(average_precision_score(y_test, y_score))
            test_acc = float(accuracy_score(y_test, y_pred))

            hit_prec = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
            hit_rec = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
            hit_f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
            f1_w = f1_score(y_test, y_pred, average='weighted')
            prec_w = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            rec_w = recall_score(y_test, y_pred, average='weighted', zero_division=0)

            baseline_results.append({
                'Model': name,
                'CV_PR_AUC': cv_pr_auc,
                'Test_PR_AUC': test_pr_auc,
                'Test_Accuracy': test_acc,
                'Hit_Precision': hit_prec,
                'Hit_Recall': hit_rec,
                'Hit_F1': hit_f1,
                'F1-Score': f1_w,
                'Precision': prec_w,
                'Recall': rec_w,
            })
            baseline_pipelines[name] = pipeline
            print(f"{name:<35} | {cv_pr_auc:.4f}   | {test_pr_auc:.4f}   | {test_acc:.4f}   | {hit_prec:.4f} | {hit_rec:.4f} | {hit_f1:.4f}")
        except Exception as e:
            print(f"❌ Lỗi {name}: {e}")
    
    # Vẽ biểu đồ so sánh baseline (UPDATED: Style giống reference image)
    if not baseline_results:
        raise RuntimeError("❌ Không có baseline model nào chạy thành công. Kiểm tra lại dữ liệu/feature/thiết lập model.")

    # Leakage-safe model selection: choose by CV PR-AUC (TRAIN only).
    # TEST metrics are reported for transparency, but must not drive selection.
    baseline_df = pd.DataFrame(baseline_results).sort_values(by='CV_PR_AUC', ascending=False)
    
    # Lấy best model theo CV PR-AUC để dùng cho CM/Optuna
    best_baseline_name = baseline_df.iloc[0]['Model']
    best_baseline_cv_pr_auc = float(baseline_df.iloc[0]['CV_PR_AUC'])
    best_baseline_acc = float(baseline_df.iloc[0]['Test_Accuracy'])
    best_baseline_hit_f1 = float(baseline_df.iloc[0]['Hit_F1'])
    
    fig, ax = plt.subplots(figsize=(16, 10))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(baseline_df)))
    bars = ax.barh(baseline_df['Model'], baseline_df['Hit_F1'], color=colors, edgecolor='white', linewidth=1.5)
    
    # Thêm giá trị accuracy vào cuối mỗi bar
    for i, (idx, row) in enumerate(baseline_df.iterrows()):
        ax.text(row['Hit_F1'] + 0.005, i, f"{row['Hit_F1']:.4f}",
                va='center', fontsize=11, fontweight='bold', color='black')
    
    ax.set_xlabel('Hit F1-score (Test, threshold=0.5)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Model', fontsize=13, fontweight='bold')
    ax.set_title('So sánh hiệu năng dự đoán HIT (BASELINE) — Tiêu chí: Hit F1-score', fontsize=16, fontweight='bold', pad=20)
    
    ax.invert_yaxis()

    ax.set_xlim(0.0, max(baseline_df['Hit_F1']) + 0.10)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    os.makedirs('DA/tasks/Hit', exist_ok=True)
    plt.savefig('DA/tasks/Hit/p0_baseline_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\n✅ Đã lưu biểu đồ baseline tại: DA/tasks/Hit/p0_baseline_comparison.png")
    
    # Confusion Matrix cho baseline
    # Prefer the already-fitted pipeline (supports ensembles too).
    pipeline_baseline = baseline_pipelines.get(best_baseline_name)
    if pipeline_baseline is None:
        best_baseline_model = baseline_models[best_baseline_name]
        pipeline_baseline = _build_pipe_p0(best_baseline_model)
        pipeline_baseline.fit(X_train, y_train)
    y_pred_baseline = pipeline_baseline.predict(X_test)
    
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred_baseline)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Non-Hit', 'Hit'], yticklabels=['Non-Hit', 'Hit'])
    plt.xlabel('Dự đoán', fontsize=12, fontweight='bold')
    plt.ylabel('Thực tế', fontsize=12, fontweight='bold')
    plt.title(
        f'Confusion Matrix - {best_baseline_name}\n'
        f'(Test Acc: {best_baseline_acc:.4f}, Test Hit_F1: {best_baseline_hit_f1:.4f})',
        fontsize=14,
        fontweight='bold',
    )
    plt.tight_layout()
    plt.savefig('DA/tasks/Hit/p0_confusion_matrix_baseline.png', dpi=300, bbox_inches='tight')
    print(f"✅ Đã lưu Confusion Matrix tại: DA/tasks/Hit/p0_confusion_matrix_baseline.png")
    plt.close()
    # =============================================================================
    # BƯỚC 1.5: XÁC ĐỊNH BEST MODEL TỪ BASELINE
    # =============================================================================
    print("\n" + "="*80)
    print("🏆 XÁC ĐỊNH BEST MODEL TỪ BASELINE")
    print("="*80)
    
    best_baseline_model = baseline_df.iloc[0]['Model']
    best_baseline_cv_pr_auc = float(baseline_df.iloc[0]['CV_PR_AUC'])
    best_baseline_acc = float(baseline_df.iloc[0]['Test_Accuracy'])
    best_baseline_hit_f1 = float(baseline_df.iloc[0]['Hit_F1'])
    print(
        f"\n✨ BEST BASELINE MODEL (by CV PR-AUC): {best_baseline_model} "
        f"(CV PR-AUC: {best_baseline_cv_pr_auc:.4f} | Test Hit_F1(report): {best_baseline_hit_f1:.4f} | Test Acc(report): {best_baseline_acc:.4f})"
    )
    
    # Xác định models cần optimize
    models_to_optimize = []
    if 'Voting' in best_baseline_model or 'Stacking' in best_baseline_model:
        models_to_optimize = ['XGBoost', 'SVM', 'Logistic Regression']
        print(f"📋 Ensemble model detected → Sẽ tối ưu 3 base experts: {', '.join(models_to_optimize)}")
    else:
        models_to_optimize = [best_baseline_model]
        print(f"📋 Single model detected → Sẽ tối ưu: {best_baseline_model}")
    
    # =============================================================================
    # 4. BƯỚC 2: TỐI ƯU CHỈ BEST MODEL BẰNG OPTUNA (CÓ VISUALIZATION)
    # =============================================================================
    print("\n" + "="*80)
    print("🔧 BƯỚC 2: TỐI ƯU CHỈ BEST MODEL BẰNG OPTUNA")
    print("="*80)
    
    # Thư mục lưu Optuna history
    optuna_dir = 'DA/tasks/Hit/optuna_history_json'
    os.makedirs(optuna_dir, exist_ok=True)
    os.makedirs('DA/tasks/Hit/optuna_history_image', exist_ok=True)

    # -------------------------------------------------------------------------
    # DRY helper: build Optuna objectives with consistent CV + Pipeline wrapper
    # -------------------------------------------------------------------------
    def create_objective(*, suggest_params, make_estimator, per_fold: bool = False):
        """Create an Optuna objective.

        - suggest_params(trial) -> dict of hyperparams
        - make_estimator(params, y_tr_or_none) -> sklearn estimator
        - per_fold=True: manually loop folds (useful when scale_pos_weight depends on y_tr)
        """

        def _objective(trial):
            params = suggest_params(trial)

            if not per_fold:
                estimator = make_estimator(params, None)
                pipe_trial = _build_pipe_p0(estimator)
                cv_scores = cross_val_score(
                    pipe_trial,
                    X_train,
                    y_train,
                    cv=tscv_inner,
                    scoring=OPTUNA_SCORING,
                    n_jobs=-1,
                )
                return float(np.mean(cv_scores))

            # Manual fold loop (e.g., dynamic scale_pos_weight)
            fold_scores = []
            for tr_idx, va_idx in tscv_inner.split(X_train, y_train):
                y_tr = y_train.iloc[tr_idx]
                estimator = make_estimator(params, y_tr)
                pipe_trial = _build_pipe_p0(estimator)
                pipe_trial.fit(X_train.iloc[tr_idx], y_tr)

                y_va_score = _get_y_score_for_pr_auc(pipe_trial, X_train.iloc[va_idx])
                if OPTUNA_SCORING == 'average_precision':
                    fold_scores.append(average_precision_score(y_train.iloc[va_idx], y_va_score))
                else:
                    y_va_pred = (y_va_score >= 0.5).astype(int)
                    fold_scores.append(
                        f1_score(y_train.iloc[va_idx], y_va_pred, pos_label=1, zero_division=0)
                    )

            return float(np.mean(fold_scores))

        return _objective
    
    # Dictionary lưu tham số tối ưu
    optimized_params = {}
    optuna_studies = {}  # Lưu study để vẽ chart sau
    
    # CONDITIONAL OPTUNA: Chỉ optimize models_to_optimize
    for idx, model_name in enumerate(models_to_optimize, 1):
        print(f"\n{'='*70}")
        print(f"🔧 [{idx}/{len(models_to_optimize)}] ĐANG TỐI ƯU: {model_name}")
        print(f"{'='*70}")
        
        params_file = f'{optuna_dir}/{model_name.lower().replace(" ", "_")}_params.json'
        
        # EXTRA TREES
        if model_name == 'Extra Trees':
            if os.path.exists(params_file):
                print("✅ Đã tồn tại params, đang load...")
                with open(params_file, 'r') as f:
                    optimized_params[model_name] = json.load(f)
            else:
                objective_et = create_objective(
                    suggest_params=lambda trial: {
                        'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
                        'max_depth': trial.suggest_int('max_depth', 10, 30),
                        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 15),
                        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
                    },
                    make_estimator=lambda params, _y_tr: ExtraTreesClassifier(
                        **params,
                        class_weight=MODEL_CLASS_WEIGHT,
                        random_state=42,
                    ),
                )
                
                study = optuna.create_study(direction='maximize', study_name=f'P0_Extra_Trees')
                study.optimize(objective_et, n_trials=20, show_progress_bar=True)
                optimized_params[model_name] = study.best_params
                optuna_studies[model_name] = study
                
                with open(params_file, 'w') as f:
                    json.dump(study.best_params, f, indent=2)
                print(f"✅ Best CV {_metric_label_for_optuna()}: {study.best_value:.4f}")
                
                # VẼ OPTUNA HISTORY
                print(f"📊 Đang vẽ biểu đồ tối ưu hóa tùy chỉnh cho Extra Trees...")
                plot_custom_optuna_history(
                    study=study, 
                    model_name='Extra Trees', 
                    baseline_acc=best_baseline_cv_pr_auc,
                    save_path='DA/tasks/Hit/optuna_history_image/extra_trees_history.png'
                )
                print(f"✅ Đã lưu biểu đồ tại: DA/tasks/Hit/optuna_history_image/extra_trees_history.png")
        
        # ADABOOST
        elif model_name == 'AdaBoost':
            if os.path.exists(params_file):
                print("✅ Đã tồn tại params, đang load...")
                with open(params_file, 'r') as f:
                    optimized_params[model_name] = json.load(f)
            else:
                objective_ada = create_objective(
                    suggest_params=lambda trial: {
                        'n_estimators': trial.suggest_int('n_estimators', 50, 500, step=50),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 1.0, log=True),
                    },
                    make_estimator=lambda params, _y_tr: AdaBoostClassifier(
                        estimator=DecisionTreeClassifier(max_depth=1),
                        **params,
                        random_state=42,
                    ),
                )
                
                study = optuna.create_study(direction='maximize', study_name='P0_AdaBoost')
                study.optimize(objective_ada, n_trials=20, show_progress_bar=True)
                optimized_params[model_name] = study.best_params
                optuna_studies[model_name] = study
                
                with open(params_file, 'w') as f:
                    json.dump(study.best_params, f, indent=2)
                print(f"✅ Best CV {_metric_label_for_optuna()}: {study.best_value:.4f}")
                
                # VẼ OPTUNA HISTORY
                print(f"📊 Đang vẽ biểu đồ tối ưu hóa tùy chỉnh cho AdaBoost...")
                plot_custom_optuna_history(
                    study=study, 
                    model_name='AdaBoost', 
                    baseline_acc=best_baseline_cv_pr_auc,
                    save_path='DA/tasks/Hit/optuna_history_image/AdaBoost_history.png'
                )
                print(f"✅ Đã lưu biểu đồ tại: DA/tasks/Hit/optuna_history_image/AdaBoost_history.png")
        
        # SVM
        elif model_name == 'SVM':
            if os.path.exists(params_file):
                print("✅ Đã tồn tại params, đang load...")
                with open(params_file, 'r') as f:
                    optimized_params[model_name] = json.load(f)
            else:
                objective_svm = create_objective(
                    suggest_params=lambda trial: {
                        'C': trial.suggest_float('C', 0.1, 100.0, log=True),
                        'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
                    },
                    make_estimator=lambda params, _y_tr: SVC(
                        **params,
                        kernel='rbf',
                        probability=True,
                        class_weight=MODEL_CLASS_WEIGHT,
                        random_state=42,
                    ),
                )
                
                study = optuna.create_study(direction='maximize', study_name='P0_SVM')
                study.optimize(objective_svm, n_trials=20, show_progress_bar=True)
                optimized_params[model_name] = study.best_params
                optuna_studies[model_name] = study
                
                with open(params_file, 'w') as f:
                    json.dump(study.best_params, f, indent=2)
                print(f"✅ Best CV {_metric_label_for_optuna()}: {study.best_value:.4f}")
                
                print(f"📊 Đang vẽ biểu đồ tối ưu hóa tùy chỉnh cho SVM...")
                plot_custom_optuna_history(
                    study=study, 
                    model_name='SVM', 
                    baseline_acc=best_baseline_cv_pr_auc,
                    save_path='DA/tasks/Hit/optuna_history_image/SVM_history.png'
                )
                print(f"✅ Đã lưu biểu đồ tại: DA/tasks/Hit/optuna_history_image/SVM_history.png")
        
        # RANDOM FOREST
        elif model_name == 'Random Forest':
            if os.path.exists(params_file):
                print("✅ Đã tồn tại params, đang load...")
                with open(params_file, 'r') as f:
                    optimized_params[model_name] = json.load(f)
            else:
                objective_rf = create_objective(
                    suggest_params=lambda trial: {
                        'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
                        'max_depth': trial.suggest_int('max_depth', 10, 30),
                        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 15),
                        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
                    },
                    make_estimator=lambda params, _y_tr: RandomForestClassifier(
                        **params,
                        class_weight=MODEL_CLASS_WEIGHT,
                        random_state=42,
                    ),
                )
                
                study = optuna.create_study(direction='maximize', study_name='P0_Random_Forest')
                study.optimize(objective_rf, n_trials=20, show_progress_bar=True)
                optimized_params[model_name] = study.best_params
                optuna_studies[model_name] = study
                
                with open(params_file, 'w') as f:
                    json.dump(study.best_params, f, indent=2)
                print(f"✅ Best {_metric_label_for_optuna()}: {study.best_value:.4f}")
                
                print(f"📊 Đang vẽ biểu đồ tối ưu hóa tùy chỉnh cho Random Forest...")
                plot_custom_optuna_history(
                    study=study, 
                    model_name='Random Forest', 
                    baseline_acc=best_baseline_cv_pr_auc,
                    save_path='DA/tasks/Hit/optuna_history_image/Random_Forest_history.png'
                )
                print(f"✅ Đã lưu biểu đồ tại: DA/tasks/Hit/optuna_history_image/Random_Forest_history.png")
        
        # GRADIENT BOOSTING
        elif model_name == 'Gradient Boosting':
            if os.path.exists(params_file):
                print("✅ Đã tồn tại params, đang load...")
                with open(params_file, 'r') as f:
                    optimized_params[model_name] = json.load(f)
            else:
                objective_gb = create_objective(
                    suggest_params=lambda trial: {
                        'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                        'max_depth': trial.suggest_int('max_depth', 3, 10),
                        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    },
                    make_estimator=lambda params, _y_tr: GradientBoostingClassifier(
                        **params,
                        random_state=42,
                    ),
                )
                
                study = optuna.create_study(direction='maximize', study_name='P0_Gradient_Boosting')
                study.optimize(objective_gb, n_trials=20, show_progress_bar=True)
                optimized_params[model_name] = study.best_params
                optuna_studies[model_name] = study
                
                with open(params_file, 'w') as f:
                    json.dump(study.best_params, f, indent=2)
                print(f"✅ Best {_metric_label_for_optuna()}: {study.best_value:.4f}")
                
                print(f"📊 Đang vẽ biểu đồ tối ưu hóa tùy chỉnh cho Gradient Boosting...")
                plot_custom_optuna_history(
                    study=study, 
                    model_name='Gradient Boosting', 
                    baseline_acc=best_baseline_cv_pr_auc,
                    save_path='DA/tasks/Hit/optuna_history_image/Gradient_Boosting_history.png'
                )
                print(f"✅ Đã lưu biểu đồ tại: DA/tasks/Hit/optuna_history_image/Gradient_Boosting_history.png")
        
        # XGBOOST
        elif model_name == 'XGBoost':
            if os.path.exists(params_file):
                print("✅ Đã tồn tại params, đang load...")
                with open(params_file, 'r') as f:
                    optimized_params[model_name] = json.load(f)
            else:
                objective_xgb = create_objective(
                    suggest_params=lambda trial: {
                        'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                        'max_depth': trial.suggest_int('max_depth', 3, 10),
                        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                        # anti-overfitting regularization
                        'gamma': trial.suggest_float('gamma', 0.0, 10.0),
                        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 100.0, log=True),
                    },
                    make_estimator=lambda params, y_tr: XGBClassifier(
                        **params,
                        random_state=42,
                        eval_metric='logloss',
                        n_jobs=-1,
                        scale_pos_weight=(1.0 if resampling_enabled else compute_scale_pos_weight(y_tr)),
                    ),
                    per_fold=True,
                )
                
                study = optuna.create_study(direction='maximize', study_name='P0_XGBoost')
                study.optimize(objective_xgb, n_trials=20, show_progress_bar=True)
                optimized_params[model_name] = study.best_params
                optuna_studies[model_name] = study
                
                with open(params_file, 'w') as f:
                    json.dump(study.best_params, f, indent=2)
                print(f"✅ Best CV {_metric_label_for_optuna()}: {study.best_value:.4f}")
                
                print(f"📊 Đang vẽ biểu đồ tối ưu hóa tùy chỉnh cho XGBoost...")
                plot_custom_optuna_history(
                    study=study, 
                    model_name='XGBoost', 
                    baseline_acc=best_baseline_cv_pr_auc,
                    save_path='DA/tasks/Hit/optuna_history_image/XGBoost_history.png'
                )
                print(f"✅ Đã lưu biểu đồ tại: DA/tasks/Hit/optuna_history_image/XGBoost_history.png")
        
        # LIGHTGBM
        elif model_name == 'LightGBM':
            if os.path.exists(params_file):
                print("✅ Đã tồn tại params, đang load...")
                with open(params_file, 'r') as f:
                    optimized_params[model_name] = json.load(f)
            else:
                objective_lgbm = create_objective(
                    suggest_params=lambda trial: {
                        'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                        'max_depth': trial.suggest_int('max_depth', 3, 10),
                        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    },
                    make_estimator=lambda params, y_tr: LGBMClassifier(
                        **params,
                        random_state=42,
                        verbose=-1,
                        scale_pos_weight=(1.0 if resampling_enabled else compute_scale_pos_weight(y_tr)),
                    ),
                    per_fold=True,
                )
                
                study = optuna.create_study(direction='maximize', study_name='P0_LightGBM')
                study.optimize(objective_lgbm, n_trials=20, show_progress_bar=True)
                optimized_params[model_name] = study.best_params
                optuna_studies[model_name] = study
                
                with open(params_file, 'w') as f:
                    json.dump(study.best_params, f, indent=2)
                print(f"✅ Best {_metric_label_for_optuna()}: {study.best_value:.4f}")
                
                print(f"📊 Đang vẽ biểu đồ tối ưu hóa tùy chỉnh cho LightGBM...")
                plot_custom_optuna_history(
                    study=study, 
                    model_name='LightGBM', 
                    baseline_acc=best_baseline_cv_pr_auc,
                    save_path='DA/tasks/Hit/optuna_history_image/LightGBM_history.png'
                )
                print(f"✅ Đã lưu biểu đồ tại: DA/tasks/Hit/optuna_history_image/LightGBM_history.png")
        
        # MLP
        elif model_name == 'MLP (Neural Net)':
            if os.path.exists(params_file):
                print("✅ Đã tồn tại params, đang load...")
                with open(params_file, 'r') as f:
                    optimized_params[model_name] = json.load(f)
            else:
                objective_mlp = create_objective(
                    suggest_params=lambda trial: (
                        (lambda n_layers: {
                            'hidden_layer_sizes': tuple(
                                [trial.suggest_int(f'n_units_l{i}', 32, 256) for i in range(n_layers)]
                            ),
                            'activation': trial.suggest_categorical('activation', ['relu', 'tanh']),
                            'alpha': trial.suggest_float('alpha', 1e-5, 1e-2, log=True),
                            'learning_rate_init': trial.suggest_float('learning_rate_init', 1e-4, 1e-2, log=True),
                        })(trial.suggest_int('n_layers', 1, 3))
                    ),
                    make_estimator=lambda params, _y_tr: MLPClassifier(
                        **params,
                        max_iter=1000,
                        early_stopping=True,
                        random_state=42,
                    ),
                )
                
                study = optuna.create_study(direction='maximize', study_name='P0_MLP')
                study.optimize(objective_mlp, n_trials=20, show_progress_bar=True)
                optimized_params[model_name] = study.best_params
                optuna_studies[model_name] = study
                
                with open(params_file, 'w') as f:
                    json.dump(study.best_params, f, indent=2)
                print(f"✅ Best {_metric_label_for_optuna()}: {study.best_value:.4f}")
                
                print(f"📊 Đang vẽ biểu đồ tối ưu hóa tùy chỉnh cho MLP (Neural Net)...")
                plot_custom_optuna_history(
                    study=study, 
                    model_name='MLP (Neural Net)', 
                    baseline_acc=best_baseline_cv_pr_auc,
                    save_path='DA/tasks/Hit/optuna_history_image/MLP_history.png'
                )
                print(f"✅ Đã lưu biểu đồ tại: DA/tasks/Hit/optuna_history_image/MLP_history.png")
        
        # LOGISTIC REGRESSION  
        elif model_name == 'Logistic Regression':
            if os.path.exists(params_file):
                print("✅ Đã tồn tại params, đang load...")
                with open(params_file, 'r') as f:
                    optimized_params[model_name] = json.load(f)
            else:
                objective_lr = create_objective(
                    suggest_params=lambda trial: {
                        'C': trial.suggest_float('C', 1e-4, 100.0, log=True),
                        'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced']),
                    },
                    make_estimator=lambda params, _y_tr: LogisticRegression(
                        **{**params, 'class_weight': (None if resampling_enabled else params.get('class_weight', None))},
                        random_state=42,
                        max_iter=2000,
                    ),
                )

                study = optuna.create_study(direction='maximize', study_name='P0_Logistic_Regression')
                study.optimize(objective_lr, n_trials=20, show_progress_bar=True)
                optimized_params[model_name] = study.best_params
                optuna_studies[model_name] = study

                with open(params_file, 'w') as f:
                    json.dump(study.best_params, f, indent=2)
                print(f"✅ Best CV {_metric_label_for_optuna()}: {study.best_value:.4f}")

                print(f"📊 Đang vẽ biểu đồ tối ưu hóa tùy chỉnh cho Logistic Regression...")
                plot_custom_optuna_history(
                    study=study,
                    model_name='Logistic Regression',
                    baseline_acc=best_baseline_cv_pr_auc,
                    save_path='DA/tasks/Hit/optuna_history_image/Logistic_Regression_history.png'
                )
                print(f"✅ Đã lưu biểu đồ tại: DA/tasks/Hit/optuna_history_image/Logistic_Regression_history.png")
    
    print("\n" + "="*80)
    print("🎉 HOÀN TẤT TỐI ƯU HÓA BEST MODEL")
    print("="*80)
    
    # =============================================================================
    # 5. BƯỚC 3: XÂY DỰNG CHỈ BEST MODEL VỚI THAM SỐ TỐI ƯU
    # =============================================================================
    print("\n" + "="*80)
    print("🔨 BƯỚC 3: XÂY DỰNG BEST MODEL VỚI THAM SỐ TỐI ƯU")
    print("="*80)
    
    # Dictionary để lưu optimized models
    optimized_models = {}
    
    # Xây dựng models dựa trên những gì đã được optimize
    for model_name in models_to_optimize:
        print(f"\n🔨 Đang xây dựng: {model_name}")
        params = optimized_params[model_name]
        
        if model_name == 'Extra Trees':
            optimized_models[model_name] = ExtraTreesClassifier(**params, class_weight=MODEL_CLASS_WEIGHT, random_state=42)
        elif model_name == 'AdaBoost':
            optimized_models[model_name] = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1), **params, random_state=42)
        elif model_name == 'SVM':
            # KHÔNG wrap với CalibratedClassifierCV để giống baseline
            optimized_models[model_name] = SVC(**params, kernel='rbf', probability=True, class_weight=MODEL_CLASS_WEIGHT, random_state=42)
        elif model_name == 'Random Forest':
            optimized_models[model_name] = RandomForestClassifier(**params, class_weight=MODEL_CLASS_WEIGHT, random_state=42)
        elif model_name == 'Gradient Boosting':
            optimized_models[model_name] = GradientBoostingClassifier(**params, random_state=42)
        elif model_name == 'XGBoost':
            # KHÔNG wrap với CalibratedClassifierCV để giống baseline
            optimized_models[model_name] = XGBClassifier(
                **params,
                random_state=42,
                eval_metric='logloss',
                scale_pos_weight=MODEL_SCALE_POS_WEIGHT,
                n_jobs=-1,
            )
        elif model_name == 'LightGBM':
            optimized_models[model_name] = LGBMClassifier(
                **params,
                random_state=42,
                verbose=-1,
                scale_pos_weight=MODEL_SCALE_POS_WEIGHT,
            )
        elif model_name == 'MLP (Neural Net)':
            mlp_params = params.copy()
            n_layers = mlp_params.pop('n_layers', 2)
            hidden_layers = tuple([mlp_params.pop(f'n_units_l{i}', 128) for i in range(n_layers)])
            optimized_models[model_name] = MLPClassifier(hidden_layer_sizes=hidden_layers, **mlp_params, max_iter=1000, early_stopping=True, random_state=42)
        elif model_name == 'Logistic Regression':
            optimized_models[model_name] = LogisticRegression(
                C=params.get('C', 1.0),
                class_weight=(None if resampling_enabled else params.get('class_weight', 'balanced')),
                max_iter=int(params.get('max_iter', 2000)),
                random_state=42,
            )
    
    # Nếu ensemble model win thì tạo ensemble từ 3 base experts đã optimize
    if 'Voting' in best_baseline_model or 'Stacking' in best_baseline_model:
        print("\n🔗 Đang xây dựng Ensemble System với optimized base models...")

        xgb_opt = optimized_models.get('XGBoost')
        svm_opt = optimized_models.get('SVM')
        lr_opt = optimized_models.get('Logistic Regression')

        if xgb_opt is not None and svm_opt is not None and lr_opt is not None:
            # Compute PR-AUC weights on TRAIN for each expert (to weight soft voting)
            expert_models = [('xgb', xgb_opt), ('svm', svm_opt), ('lr', lr_opt)]
            expert_pr_auc = []
            for nm, mdl in expert_models:
                pipe_tmp = _build_pipe_p0(mdl)
                scores_tmp = cross_val_score(
                    pipe_tmp,
                    X_train,
                    y_train,
                    cv=tscv_inner,
                    scoring='average_precision',
                    n_jobs=-1,
                )
                expert_pr_auc.append(float(np.mean(scores_tmp)))

            voting_weights_opt = _weights_from_pr_auc(expert_pr_auc)
            print(f"🗳️  Optimized Voting weights theo CV PR-AUC = {expert_pr_auc} → weights={voting_weights_opt}")

            voting_opt = VotingClassifier(
                estimators=expert_models,
                voting='soft',
                weights=voting_weights_opt,
            )
            optimized_models['Voting Ensemble (XGB+SVM+LR)'] = voting_opt

            stacking_opt = StackingClassifier(
                estimators=expert_models,
                final_estimator=LogisticRegression(class_weight=MODEL_CLASS_WEIGHT, random_state=42, max_iter=2000),
                cv=5,
                n_jobs=-1,
                passthrough=False,
            )
            optimized_models['Stacking Ensemble (XGB+SVM+LR)'] = stacking_opt
            print("✅ Đã xây dựng Voting và Stacking Classifier với optimized base experts")
    
    print(f"\n✅ Đã xây dựng {len(optimized_models)} model(s)")

    # =============================================================================
    # 6. ĐÁNH GIÁ OPTIMIZED BEST MODEL  
    # =============================================================================
    print("\n" + "="*80)
    print("📊 BƯỚC 4: ĐÁNH GIÁ OPTIMIZED BEST MODEL")
    print("="*80)
    
    results_list = []
    fitted_pipelines = {}
    predictions_dict = {}

    print(f"\n{'MODEL':<40} | {'CV_'+_metric_label_for_optuna():<8} | {'TEST_AP':<8} | {'TEST_ACC':<8} | {'HIT_P':<7} | {'HIT_R':<7} | {'HIT_F1':<7}")
    print("-" * 115)

    for name, model in optimized_models.items():
        try:
            pipeline = _build_pipe_p0(model)

            cv_scores = cross_val_score(pipeline, X_train, y_train, cv=tscv_inner, scoring=OPTUNA_SCORING, n_jobs=-1)
            cv_pr_auc = float(np.mean(cv_scores))

            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            y_score = _get_y_score_for_pr_auc(pipeline, X_test)
            test_pr_auc = float(average_precision_score(y_test, y_score))
            test_acc = float(accuracy_score(y_test, y_pred))
            hit_prec = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
            hit_rec = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
            hit_f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)

            f1 = f1_score(y_test, y_pred, average='weighted')
            prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)

            results_list.append({
                'Model': name,
                'CV_PR_AUC': cv_pr_auc,
                'Test_PR_AUC': test_pr_auc,
                'Test_Accuracy': test_acc,
                'Hit_Precision': hit_prec,
                'Hit_Recall': hit_rec,
                'Hit_F1': hit_f1,
                'F1-Score': f1,
                'Precision': prec,
                'Recall': rec,
            })
            fitted_pipelines[name] = pipeline
            predictions_dict[name] = y_pred
            
            print(f"{name:<40} | {cv_pr_auc:.4f}   | {test_pr_auc:.4f}   | {test_acc:.4f}   | {hit_prec:.4f} | {hit_rec:.4f} | {hit_f1:.4f}")
        except Exception as e:
            print(f"❌ Lỗi {name}: {e}")

    optimized_results_list = results_list.copy()
    # Leakage-safe: choose the best optimized model by CV PR-AUC (TRAIN only)
    optimized_df = pd.DataFrame(optimized_results_list).sort_values(by='CV_PR_AUC', ascending=False)
    final_best_model_name = str(optimized_df.iloc[0]['Model'])
    best_optimized_test_acc = float(optimized_df.iloc[0]['Test_Accuracy'])
    best_optimized_hit_f1 = float(optimized_df.iloc[0]['Hit_F1'])
    best_optimized_cv_pr_auc = float(optimized_df.iloc[0]['CV_PR_AUC'])
    print(
        f"\n🏆 BEST OPTIMIZED MODEL (by CV PR-AUC): {final_best_model_name} "
        f"(CV PR-AUC: {best_optimized_cv_pr_auc:.4f} | Test Hit_F1(report): {best_optimized_hit_f1:.4f} | Test Acc(report): {best_optimized_test_acc:.4f})"
    )
    
    # =============================================================================
    # BƯỚC 4.3: LOGIC ROLLBACK - SO SÁNH VỚI BASELINE
    # =============================================================================
    print("\n" + "="*80)
    print("⚖️ BƯỚC 4.3: KIỂM TRA VÀ ROLLBACK NẾU CẦN")
    print("="*80)
    
    # Leakage-safe rollback criterion: compare CV PR-AUC (TRAIN only).
    # Threshold tuning + Test F1 are reported for transparency.
    best_optimized_name = str(optimized_df.iloc[0]['Model'])

    baseline_pipe = baseline_pipelines.get(best_baseline_name)
    optimized_pipe = fitted_pipelines.get(best_optimized_name)

    if baseline_pipe is None or optimized_pipe is None:
        raise RuntimeError("❌ Không tìm thấy pipeline baseline/optimized để so sánh rollback.")

    y_base_thr_pred, thr_base, f1_base_thr, p_base_thr, r_base_thr, acc_base_thr = optimize_and_evaluate_threshold(
        baseline_pipe,
        X_train,
        y_train,
        X_test,
        y_test,
        cv_splitter=tscv_inner,
        tag='baseline',
    )

    y_opt_thr_pred, thr_opt, f1_opt_thr, p_opt_thr, r_opt_thr, acc_opt_thr = optimize_and_evaluate_threshold(
        optimized_pipe,
        X_train,
        y_train,
        X_test,
        y_test,
        cv_splitter=tscv_inner,
        tag='optimized',
    )

    print(f"\n📊 So sánh Performance (threshold-aware, mục tiêu: Max Hit F1):")
    print(
        f"   • Baseline:  {best_baseline_name} | thr={thr_base:.4f} | F1={f1_base_thr:.4f} | P={p_base_thr:.4f} | R={r_base_thr:.4f} | Acc={acc_base_thr:.4f}"
    )
    print(
        f"   • Optimized: {best_optimized_name} | thr={thr_opt:.4f} | F1={f1_opt_thr:.4f} | P={p_opt_thr:.4f} | R={r_opt_thr:.4f} | Acc={acc_opt_thr:.4f}"
    )
    print(f"   • Chênh lệch F1: {f1_opt_thr - f1_base_thr:+.4f}")

    if best_optimized_cv_pr_auc < best_baseline_cv_pr_auc:
        print(
            f"\n⚠️ PHÁT HIỆN: Optimized CV PR-AUC ({best_optimized_cv_pr_auc:.4f}) < "
            f"Baseline CV PR-AUC ({best_baseline_cv_pr_auc:.4f})"
        )
        print("✅ QUYẾT ĐỊNH: ROLLBACK về Baseline model")
        print(f"   → Sử dụng: {best_baseline_name} (thr={thr_base:.4f})\n")

        final_best_model_name = best_baseline_name
        best_pipe = baseline_pipe
        best_model_name = best_baseline_name
        best_thr = float(thr_base)
        is_improved = False
        best_acc = float(acc_base_thr)
        best_hit_f1 = float(f1_base_thr)
        best_hit_p = float(p_base_thr)
        best_hit_r = float(r_base_thr)
        y_pred_final_for_reporting = y_base_thr_pred
    else:
        print(
            f"\n✅ Giữ Optimized theo CV: Baseline CV PR-AUC={best_baseline_cv_pr_auc:.4f} "
            f"→ Optimized CV PR-AUC={best_optimized_cv_pr_auc:.4f} (+{best_optimized_cv_pr_auc - best_baseline_cv_pr_auc:+.4f})"
        )
        print(f"✅ QUYẾT ĐỊNH: Giữ lại Optimized model (thr={thr_opt:.4f})\n")

        best_pipe = optimized_pipe
        best_model_name = best_optimized_name
        best_thr = float(thr_opt)
        is_improved = True
        best_acc = float(acc_opt_thr)
        best_hit_f1 = float(f1_opt_thr)
        best_hit_p = float(p_opt_thr)
        best_hit_r = float(r_opt_thr)
        y_pred_final_for_reporting = y_opt_thr_pred
    
    print("="*80 + "\n")
    
    # Lưu ý về hiệu năng sau Optuna
    print("\n⚠️  LƯU Ý VỀ OPTUNA OPTIMIZATION:")
    print(f"   - Optuna tối ưu dựa trên CV ({OPTUNA_SCORING}) trên TRAIN set")
    print("   - Quyết định rollback dựa trên CV PR-AUC (TRAIN); Test F1_max chỉ để báo cáo")
    print("   - Nếu optimized < baseline: tham số mặc định + threshold có thể generalize tốt hơn\n")
    # best_model_name đã được quyết định ở bước rollback ở trên
    
    # =============================================================================
    # 7. VẼ BIỂU ĐỒ SO SÁNH BASELINE VS OPTIMIZED
    # =============================================================================
    print("\n" + "="*80)
    print("📊 BƯỚC 5: VẼ BIỂU ĐỒ SO SÁNH")
    print("="*80)
    
    # So sánh baseline vs optimized theo Hit_F1 (class 1)
    optimized_df = pd.DataFrame(optimized_results_list).sort_values(by='Hit_F1', ascending=False)
    baseline_df = pd.DataFrame(baseline_results).sort_values(by='Hit_F1', ascending=False)
    
    fig, ax = plt.subplots(figsize=(16, 10))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(baseline_df)))
    bars = ax.barh(baseline_df['Model'], baseline_df['Hit_F1'], color=colors, edgecolor='white', linewidth=1.5)
    
    # Thêm giá trị accuracy vào cuối mỗi bar
    for i, (idx, row) in enumerate(baseline_df.iterrows()):
        ax.text(row['Hit_F1'] + 0.005, i, f"{row['Hit_F1']:.4f}", 
                va='center', fontsize=11, fontweight='bold', color='black')
    
    ax.set_xlabel('Hit F1-score (Test, threshold=0.5)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Model', fontsize=13, fontweight='bold')
    ax.set_title('So sánh hiệu năng các mô hình dự đoán HIT (BASELINE)', fontsize=16, fontweight='bold', pad=20)
    
    ax.invert_yaxis()

    ax.set_xlim(0.0, max(baseline_df['Hit_F1']) + 0.10)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    os.makedirs('DA/tasks/Hit', exist_ok=True)
    plt.savefig('DA/tasks/Hit/p0_baseline_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\n✅ Đã lưu biểu đồ baseline tại: DA/tasks/Hit/p0_baseline_comparison.png")

    # Optimized comparison chart
    fig, ax = plt.subplots(figsize=(16, 10))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(optimized_df)))
    bars = ax.barh(optimized_df['Model'], optimized_df['Hit_F1'], color=colors, edgecolor='white', linewidth=1.5)
    for i, (idx, row) in enumerate(optimized_df.iterrows()):
        ax.text(row['Hit_F1'] + 0.005, i, f"{row['Hit_F1']:.4f}", va='center', fontsize=11, fontweight='bold', color='black')
    ax.set_xlabel('Hit F1-score (Test, threshold=0.5)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Model', fontsize=13, fontweight='bold')
    ax.set_title('So sánh hiệu năng các mô hình dự đoán HIT (OPTIMIZED)', fontsize=16, fontweight='bold', pad=20)
    ax.invert_yaxis()
    ax.set_xlim(0.0, max(optimized_df['Hit_F1']) + 0.10)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig('DA/tasks/Hit/p0_optimized_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Đã lưu biểu đồ optimized tại: DA/tasks/Hit/p0_optimized_comparison.png")

    
    # Tạo biểu đồ so sánh trực tiếp (chỉ với models có trong cả 2)
    common_models = set(baseline_df['Model']) & set(optimized_df['Model'])
    comparison_data = []
    
    for model in common_models:
        baseline_score = baseline_df[baseline_df['Model'] == model]['Hit_F1'].values[0] if len(baseline_df[baseline_df['Model'] == model]) > 0 else 0
        optimized_score = optimized_df[optimized_df['Model'] == model]['Hit_F1'].values[0] if len(optimized_df[optimized_df['Model'] == model]) > 0 else 0
        comparison_data.append({'Model': model, 'Baseline': baseline_score, 'Optimized': optimized_score, 'Improvement': optimized_score - baseline_score})
    
    comparison_df = pd.DataFrame(comparison_data).sort_values(by='Improvement', ascending=False)
    
    # Vẽ biểu đồ grouped bar chart (UPDATED: Better styling)
    fig, ax = plt.subplots(figsize=(16, 10))
    x = np.arange(len(comparison_df))
    width = 0.38
    
    bars1 = ax.barh(x - width/2, comparison_df['Baseline'], width, label='Baseline (Default params)', 
                    color='#87CEEB', edgecolor='white', linewidth=1.5)
    bars2 = ax.barh(x + width/2, comparison_df['Optimized'], width, label='Optimized (Optuna tuned)', 
                    color='#FF8C42', edgecolor='white', linewidth=1.5)
    
    # Thêm giá trị accuracy cho cả 2 bars
    for i, (idx, row) in enumerate(comparison_df.iterrows()):
        ax.text(row['Baseline'] + 0.003, i - width/2, f"{row['Baseline']:.4f}", 
                va='center', fontsize=10, fontweight='bold', color='black')
        ax.text(row['Optimized'] + 0.003, i + width/2, f"{row['Optimized']:.4f}", 
                va='center', fontsize=10, fontweight='bold', color='black')
        
        # Improvement indicator
        if row['Improvement'] > 0:
            ax.text(max(row['Baseline'], row['Optimized']) + 0.015, i, 
                   f"↑ +{row['Improvement']*100:.2f}%", va='center', 
                   color='green', fontweight='bold', fontsize=9)
    
    ax.set_yticks(x)
    ax.set_yticklabels(comparison_df['Model'], fontsize=11)
    ax.set_xlabel('Hit F1-score (Test, threshold=0.5)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Model', fontsize=13, fontweight='bold')
    ax.set_title('So sánh Baseline vs Optimized - Impact của Optuna Tuning', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='lower right', fontsize=11, frameon=True, shadow=True)
    ax.set_xlim(0.0, float(comparison_df[['Baseline', 'Optimized']].max().max()) + 0.10)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('DA/tasks/Hit/p0_baseline_vs_optimized.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Đã lưu biểu đồ so sánh tại: DA/tasks/Hit/p0_baseline_vs_optimized.png")
    # =============================================================================
    # 10. GIẢI THÍCH MÔ HÌNH TỐT NHẤT (CHỈ CHO BEST MODEL)
    # =============================================================================
    print(f"\n" + "="*80)
    print(f"📊 PHÂN TÍCH CHI TIẾT CHO MÔ HÌNH TỐT NHẤT: {best_model_name}")
    print("="*80)
    
    # best_pipe đã được xác định ở BƯỚC 4.3 (hoặc optimized hoặc baseline)
    pre = best_pipe.named_steps['preprocessor']
    clf = best_pipe.named_steps['clf']
    # Step 2: Predict probability (P(Hit))
    if hasattr(best_pipe, 'predict_proba'):
        y_prob_final = best_pipe.predict_proba(X_test)[:, 1]
    else:
        y_prob_final = None

    # Step 3: Tune threshold to maximize Hit F1 (already decided in BƯỚC 4.3)
    # Use tuned predictions/threshold for final reporting.
    if 'y_pred_final_for_reporting' in locals():
        y_pred_final = y_pred_final_for_reporting
    elif y_prob_final is not None:
        y_pred_final = (y_prob_final >= float(best_thr)).astype(int)
    else:
        y_pred_final = best_pipe.predict(X_test)
    
    # A. VẼ CONFUSION MATRIX
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred_final)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Non-Hit', 'Hit'], yticklabels=['Non-Hit', 'Hit'])
    plt.xlabel('Dự đoán', fontsize=12, fontweight='bold')
    plt.ylabel('Thực tế', fontsize=12, fontweight='bold')
    plt.title(
        f'Confusion Matrix - {best_model_name}\n'
        f'(thr={best_thr:.4f} | Hit_F1={best_hit_f1:.4f} | Hit_P={best_hit_p:.4f} | Hit_R={best_hit_r:.4f} | Acc={best_acc:.4f})',
        fontsize=14,
        fontweight='bold',
    )
    plt.tight_layout()
    plt.savefig('DA/tasks/Hit/p0_confusion_matrix.png', dpi=300)
    print(f"✅ Đã lưu Confusion Matrix tại: DA/tasks/Hit/p0_confusion_matrix.png")
    plt.close()
    
    # Step 4: Evaluate (F1 main; Precision/Recall support)
    print(f"\n📋 Báo cáo phân loại (Threshold tuned, thr={best_thr:.4f}):")
    print(classification_report(y_test, y_pred_final, target_names=['Non-Hit', 'Hit']))

    hit_prec_final = precision_score(y_test, y_pred_final, pos_label=1, zero_division=0)
    hit_rec_final = recall_score(y_test, y_pred_final, pos_label=1, zero_division=0)
    hit_f1_final = f1_score(y_test, y_pred_final, pos_label=1, zero_division=0)
    print(f"🎯 MAIN (Hit=1) → F1={hit_f1_final:.4f} | Precision={hit_prec_final:.4f} | Recall={hit_rec_final:.4f}")

    # === PR-CURVE (Best vs Baseline) ===
    try:
        plt.figure(figsize=(9, 7))
        # Best model curve
        if y_prob_final is not None:
            p_best, r_best, _ = precision_recall_curve(y_test, y_prob_final)
            ap_best = float(average_precision_score(y_test, y_prob_final))
            plt.plot(r_best, p_best, linewidth=2.2, label=f"Best ({best_model_name}) AP={ap_best:.4f}")

        # Baseline-best curve (for comparison)
        try:
            baseline_pipe = baseline_pipelines.get(best_baseline_name)
            if baseline_pipe is not None:
                y_score_base = _get_y_score_for_pr_auc(baseline_pipe, X_test)
                p_base, r_base, _ = precision_recall_curve(y_test, y_score_base)
                ap_base = float(average_precision_score(y_test, y_score_base))
                plt.plot(r_base, p_base, linewidth=2.0, linestyle='--', label=f"Baseline ({best_baseline_name}) AP={ap_base:.4f}")
        except Exception:
            pass

        plt.xlabel('Recall', fontsize=12, fontweight='bold')
        plt.ylabel('Precision', fontsize=12, fontweight='bold')
        plt.title('Precision-Recall Curve (Test)', fontsize=14, fontweight='bold', pad=12)
        plt.grid(alpha=0.25, linestyle='--')
        plt.legend(loc='lower left', fontsize=10)
        plt.tight_layout()
        plt.savefig('DA/tasks/Hit/p0_precision_recall_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Đã lưu PR-Curve tại: DA/tasks/Hit/p0_precision_recall_curve.png")
    except Exception as e:
        print(f"⚠️  Không thể vẽ PR-Curve: {e}")
    
    # Threshold tuning already executed in BƯỚC 4.3; don't re-tune on the same TEST again.
    
    # B. FEATURE IMPORTANCE / SHAP CHỈ CHO BEST MODEL
    ohe_names = pre.named_transformers_['cat'].get_feature_names_out(cat_feats).tolist() if cat_feats else []
    all_feats = numeric_feats + ohe_names

    print(f"\n🔍 Tổng số features (num + OHE): {len(all_feats)}")
    
    try:
        if 'SVM' in best_model_name:
            print(
                "\nℹ️ SHAP cho SVM (KernelExplainer) đã được tách ra ngoài vì rất chậm. "
                "Hãy dùng DA/SHAP_explain/run_shap_all_tasks.py để chạy SHAP khi cần."
            )
        
        # C. VẼ FEATURE IMPORTANCE CHO CÁC MÔ HÌNH TREE-BASED (CHỈ BEST MODEL)
        elif any(keyword in best_model_name for keyword in ['Random Forest', 'Extra Trees', 'XGBoost', 'LightGBM', 'Gradient']):
            print(f"\n🌳 Đang vẽ Feature Importance cho {best_model_name}...")
            try:
                # Xử lý CalibratedClassifierCV: Lấy trung bình từ các fold
                if hasattr(clf, 'calibrated_classifiers_'):
                    print("   ℹ️ Model được calibrate - lấy trung bình feature importances từ các fold")
                    importances = np.mean([
                        model.estimator.feature_importances_ 
                        for model in clf.calibrated_classifiers_
                    ], axis=0)
                # Model thường (không calibrated)
                elif hasattr(clf, 'feature_importances_'):
                    importances = clf.feature_importances_
                else:
                    raise AttributeError("Model không có feature_importances_")
                
                # Lấy tên features
                ohe_names = pre.named_transformers_['cat'].get_feature_names_out(cat_feats).tolist() if cat_feats else []
                all_feats_imp = numeric_feats + ohe_names
                
                # Tạo DataFrame và sort
                feat_imp_df = pd.DataFrame({
                    'Feature': all_feats_imp,
                    'Importance': importances
                }).sort_values('Importance', ascending=False).head(20)

                feat_imp_df = rename_topics_for_report(feat_imp_df, column_name='Feature')
                
                print(f"\n🔥 TOP 20 Features ({best_model_name}):")
                print(feat_imp_df.to_string(index=False))
                
                # Vẽ biểu đồ
                fig, ax = plt.subplots(figsize=(12, 10))
                feat_imp_df_sorted = feat_imp_df.sort_values('Importance', ascending=True)
                colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(feat_imp_df_sorted)))
                
                bars = ax.barh(range(len(feat_imp_df_sorted)), feat_imp_df_sorted['Importance'],
                              color=colors, edgecolor='black', linewidth=0.8)
                
                ax.set_yticks(range(len(feat_imp_df_sorted)))
                ax.set_yticklabels(feat_imp_df_sorted['Feature'], fontsize=10)
                
                max_val = feat_imp_df_sorted['Importance'].max()
                for i, val in enumerate(feat_imp_df_sorted['Importance']):
                    ax.text(val + max_val*0.01, i, f'{val:.4f}',
                           va='center', ha='left', fontsize=9)
                
                ax.set_xlim(0, max_val * 1.15)
                ax.set_xlabel('Importance', fontsize=12)
                ax.set_ylabel('Feature', fontsize=12)
                ax.set_title(f'Top 20 Features - {best_model_name}',
                           fontsize=14, fontweight='bold', pad=20)
                ax.grid(axis='x', alpha=0.3, linestyle='--')
                
                plt.tight_layout()
                plt.savefig('DA/tasks/Hit/p0_feature_importance.png', dpi=300, bbox_inches='tight')
                plt.close()
                print(f"✅ Đã lưu Feature Importance tại: DA/tasks/Hit/p0_feature_importance.png")
                
            except Exception as e:
                print(f"❌ Lỗi vẽ Feature Importance: {e}")
                print("ℹ️ Bỏ qua SHAP fallback trong training. Dùng DA/SHAP_explain/run_shap_all_tasks.py để vẽ SHAP HD.")
    except Exception as e:
        print(f"❌ Lỗi phân tích mô hình tốt nhất: {e}")
        import traceback
        traceback.print_exc()
        has_text_features_in_top = False
    # =============================================================================
    # 12. LƯU MODELS VÀ KẾT QUẢ
    # =============================================================================
    print("\n" + "="*80)
    print("💾 ĐANG LƯU MODELS VÀ KẾT QUẢ")
    print("="*80)
    
    save_dir = 'DA/tasks/Hit'
    model_dir = 'DA/models'
    final_data_dir = 'DA/final_data'
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(final_data_dir, exist_ok=True)

    # Lưu danh sách feature names (sau transform) để phục vụ SHAP/plot ở file khác
    feature_names_path = f'{save_dir}/feature_names_p0.json'
    feature_names_legacy_path = f'{save_dir}/feature_names.json'
    try:
        feature_names = best_pipe.named_steps['preprocessor'].get_feature_names_out().tolist()
        with open(feature_names_path, 'w', encoding='utf-8') as f:
            json.dump(feature_names, f, ensure_ascii=False, indent=2)
        # Legacy/simple filename for standalone SHAP flows
        with open(feature_names_legacy_path, 'w', encoding='utf-8') as f:
            json.dump(feature_names, f, ensure_ascii=False, indent=2)
        print(f"✅ Đã lưu feature names tại: {feature_names_path}")
        print(f"✅ Đã lưu feature names tại: {feature_names_legacy_path}")
    except Exception as e:
        print(f"⚠️  Không thể lưu feature names: {e}")
    
    # Lưu best model
    best_model_path = f'{model_dir}/best_model_p0.pkl'
    _shap_cache = None
    try:
        _shap_cache = build_shap_cache(
            X_train,
            X_test,
            config=ShapCacheConfig(n_background=200, n_explain=300, random_state=42),
        )
    except Exception as e:
        print(f"⚠️  build_shap_cache failed → continue without SHAP cache: {e}")

    pkl_data = {
        'pkl_schema_version': 1,
        'task_id': 'P0',
        'data_source': str(Path('final_data') / 'data_prepared_for_ML.csv'),
        'pipeline': best_pipe,  # Sử dụng best_pipe từ BƯỚC 4.3
        'model_name': best_model_name,
        'accuracy': best_acc,
        'optimal_threshold': best_thr,
        'threshold_improved': is_improved,
        'training_config': {
            'optuna_scoring': str(OPTUNA_SCORING),
            'feature_selection': str(P0_FEATURE_SELECTION),
            'resampling': str(P0_RESAMPLING),
            'rus_strategy': str(P0_RUS_STRATEGY),
            'model_class_weight': MODEL_CLASS_WEIGHT,
            'model_scale_pos_weight': float(MODEL_SCALE_POS_WEIGHT),
        },
        'shap_cache': _shap_cache,
    }
    joblib.dump(pkl_data, best_model_path)

    # Legacy/simple filename for standalone SHAP flows
    legacy_model_path = f'{model_dir}/hit_model.pkl'
    try:
        joblib.dump(pkl_data, legacy_model_path)
        print(f"✅ Đã lưu model tại: {legacy_model_path}")
    except Exception as e:
        print(f"⚠️  Không thể lưu legacy model: {e}")
    
    print(f"\n✅ Đã lưu best model tại: {best_model_path}")
    print(f"📋 Thông tin đã lưu:")
    print(f"   • Model Name: {best_model_name}")
    print(f"   • Accuracy: {best_acc:.4f}")
    print(f"   • Optimal Threshold: {best_thr:.4f}")
    print(f"   • Threshold Improved: {'Yes' if is_improved else 'No (Rollback)'}")
    print(f"   • Pipeline size: {len(str(best_pipe))} characters")
    
    # Lưu kết quả so sánh
    results_df = pd.DataFrame(optimized_results_list)
    results_df.to_csv(f'{final_data_dir}/p0_model_comparison_results.csv', index=False)
    print(f"✅ Đã lưu kết quả so sánh tại: {final_data_dir}/p0_model_comparison_results.csv")
    
    
    print(f"\n📊 CẤU HÌNH PIPELINE:")
    print(f"   • Số features: {len(numeric_feats)} numeric + {num_cat_ohe} categorical")
    print(f"   • Holdout (time-based): {len(train_idx)} train / {len(test_idx)} test")
    print(f"   • Calibration: {'Có' if 'Calibrated' in best_model_name else 'Không'}")
    
    print(f"\n💾 FILES ĐÃ LƯU:")
    print(f"   • Best model:         {best_model_path}")
    print(f"   • Optuna params:      DA/tasks/Hit/optuna_history_json/*.json")
    print(f"   • Model comparison:   DA/final_data/p0_model_comparison_results.csv")
    print(f"   • Baseline chart:     DA/tasks/Hit/p0_baseline_comparison.png")
    print(f"   • Optimized chart:    DA/tasks/Hit/p0_optimized_comparison.png")
    print(f"   • Baseline vs Opt:    DA/tasks/Hit/p0_baseline_vs_optimized.png")
    print(f"   • Confusion Matrix:   DA/tasks/Hit/p0_cm_optimized.png")
    print(f"   • Feature Importance: DA/tasks/Hit/p0_feature_importance*.png")
    print(f"   • Correlation Matrix: DA/tasks/Hit/p0_prediction_correlation.png")
    
    print("\n" + "="*80)
    print("✅ HOÀN TẤT PHÂN TÍCH!")
    print("="*80 + "\n")


def optimize_and_evaluate_threshold(model, X_train, y_train, X_test, y_test, cv_splitter, *, tag: str = 'final'):
    """Leakage-safe threshold tuning (avoid optimism bias).

    Step 1 (TRAIN only):
      - Get out-of-fold probabilities on X_train using cross_val_predict.
      - Find threshold that maximizes Hit F1 on the TRAIN OOF predictions.

    Step 2 (BLIND TEST):
      - Apply the chosen threshold to P(Hit) on X_test.
      - Report metrics computed ONLY on the test set.

    Returns:
        (y_pred_final, final_threshold, final_hit_f1, final_hit_p, final_hit_r, final_acc)
    """
    print("\n" + "=" * 80)
    print(f"⚖️ TUNE THRESHOLD (OOF-train → apply-on-test, maximize Hit F1, tag={tag})")
    print("=" * 80)

    if not hasattr(model, 'predict_proba'):
        raise ValueError("Model/pipeline không hỗ trợ predict_proba, không thể tune threshold theo yêu cầu.")

    # -------------------------
    # Step 1: OOF probabilities on TRAIN
    # -------------------------
    print(f"⏳ Đang tính toán xác suất OOF trên tập Train (tag={tag})...")
    oof_probs_full = np.full(shape=(len(X_train),), fill_value=np.nan, dtype=float)
    
    # Chạy vòng lặp thủ công để né lỗi cross_val_predict
    for tr_idx, va_idx in cv_splitter.split(X_train, y_train):
        model_fold = clone(model)
        model_fold.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
        # Lấy xác suất của lớp Hit (cột 1)
        probs = model_fold.predict_proba(X_train.iloc[va_idx])[:, 1]
        oof_probs_full[va_idx] = probs

    # Loại bỏ các giá trị NaN (những bài cũ quá không nằm trong tập test của fold nào)
    valid_mask = np.isfinite(oof_probs_full)
    oof_probs = oof_probs_full[valid_mask]
    y_train_eff = pd.Series(y_train).iloc[np.where(valid_mask)[0]].to_numpy()

    def _hit_metrics(y_true, y_pred):
        hit_prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        hit_rec = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        hit_f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        return hit_prec, hit_rec, hit_f1

    # Threshold selection from TRAIN OOF probabilities
    precision, recall, thresholds = precision_recall_curve(y_train_eff, oof_probs, pos_label=1)
    if thresholds is None or len(thresholds) == 0:
        print("⚠️ precision_recall_curve không trả thresholds trên TRAIN → fallback threshold=0.5")
        final_threshold = 0.5
    else:
        precision_t = precision[:-1]
        recall_t = recall[:-1]
        denom = (precision_t + recall_t)
        f1_vals = np.where(denom > 0, 2 * precision_t * recall_t / denom, 0.0)
        best_local = int(np.argmax(f1_vals))
        final_threshold = float(thresholds[int(best_local)])

    # -------------------------
    # Step 2: Apply threshold to BLIND TEST
    # -------------------------
    y_test_probs = model.predict_proba(X_test)[:, 1]
    y_pred_final = (y_test_probs >= float(final_threshold)).astype(int)

    reason = "Chọn ngưỡng theo TRAIN OOF (maximize Hit F1), áp dụng nguyên ngưỡng cho TEST"

    final_hit_p, final_hit_r, final_hit_f1 = _hit_metrics(y_test, y_pred_final)
    final_acc = float(accuracy_score(y_test, y_pred_final))

    print(f"\n🎯 Threshold (tuned on TRAIN OOF): {final_threshold:.4f}")
    print(f"🧠 Lý do chọn ngưỡng: {reason}")

    print("\n📋 Classification report (theo threshold đã chọn):")
    print(classification_report(y_test, y_pred_final, target_names=['Non-Hit', 'Hit']))

    # Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm_new = confusion_matrix(y_test, y_pred_final)
    sns.heatmap(
        cm_new,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Non-Hit', 'Hit'],
        yticklabels=['Non-Hit', 'Hit'],
    )
    plt.xlabel('Dự đoán', fontsize=12, fontweight='bold')
    plt.ylabel('Thực tế', fontsize=12, fontweight='bold')
    plt.title(f'Confusion Matrix - Threshold {final_threshold:.4f}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    os.makedirs('DA/tasks/Hit', exist_ok=True)
    cm_path = 'DA/tasks/Hit/p0_cm_threshold_optimized.png' if tag == 'final' else f'DA/tasks/Hit/p0_cm_threshold_{tag}.png'
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Đã lưu CM tại: {cm_path}")
    print("=" * 80 + "\n")

    return y_pred_final, float(final_threshold), float(final_hit_f1), float(final_hit_p), float(final_hit_r), float(final_acc)


def plot_custom_optuna_history(study, model_name, baseline_acc, save_path):
    import matplotlib.pyplot as plt
    import os
    
    # Lấy dữ liệu từ study
    df = study.trials_dataframe()
    # trial_values follow OPTUNA_SCORING
    trial_values = df['value']
    # best_values là giá trị tốt nhất tích lũy (đường bậc thang)
    best_values = trial_values.cummax()

    # LƯU CSV để tái sử dụng nếu ảnh lỗi/không ưng ý (khỏi chạy lại Optuna)
    try:
        final_data_dir = os.path.join('DA', 'final_data')
        os.makedirs(final_data_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(save_path))[0]
        csv_path = os.path.join(final_data_dir, f"p0_{stem}.csv")
        df_csv = df.copy()
        df_csv['best_value_so_far'] = df_csv['value'].cummax()
        df_csv.to_csv(csv_path, index=False, encoding='utf-8-sig')
    except Exception:
        pass
    
    plt.figure(figsize=(12, 6))
    
    # 1. Vẽ các điểm trial (màu xanh nhạt)
    plt.scatter(range(len(trial_values)), trial_values, 
                color='#add8e6', alpha=0.7, edgecolors='none', label='Từng trial', s=35)
    
    # 2. Vẽ đường metric tốt nhất (màu đỏ, dạng bậc thang)
    plt.step(range(len(best_values)), best_values, 
             where='post', color='red', linewidth=2, label=f"{_metric_label_for_optuna()} tốt nhất")
    
    # 3. Vẽ đường Baseline (màu xanh lá đứt đoạn)
    plt.axhline(y=baseline_acc, color='forestgreen', linestyle='--', 
                linewidth=1.5, label=f'Baseline ({baseline_acc:.4f})')
    
    # Định dạng trục và tiêu đề
    plt.title(f'Lịch sử tối ưu hóa {model_name} (mục tiêu: {_metric_label_for_optuna()})\nBaseline: {baseline_acc:.4f} → Best: {study.best_value:.4f}', 
              fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Số lượt thử (Trial)', fontsize=12)
    plt.ylabel(_metric_label_for_optuna(), fontsize=12)
    plt.ylim(max(0.0, min(trial_values.min(), baseline_acc) - 0.02), 1.0)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend(loc='upper right', frameon=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
if __name__ == "__main__":
    try:
        run_full_analysis_task1()
    except KeyboardInterrupt:
        print("\n⚠️  Đã dừng chương trình theo yêu cầu (KeyboardInterrupt).")
        sys.exit(0)