'''
Legacy Sentiment Classification (TF-IDF + numeric features).

This file keeps the older TF-IDF/lyric-based workflow that used to live in
`analysis_Sentiment_Classification.py`. The main script was cleaned up to focus
on the leakage-safe, no-lyric pipeline.

Run:
  python DA/tasks/Sentiment/analysis_Sentiment_Classification_legacy_tfidf.py
'''

import sys
import warnings
import os
import json
from pathlib import Path

import joblib
import optuna
import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Use a non-interactive backend to avoid Tkinter/thread teardown errors on Windows.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings('ignore')

# --- Ensure repo root is on sys.path (so `import DA...` works when run by file path) ---
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

# SHAP is run in a standalone script by default for speed.
RUN_SHAP = os.getenv('RUN_SHAP', '0') == '1'

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

from DA.models.topic_mapping import rename_topics_for_report, rename_topics_in_feature_names

FILE_DATA = 'final_data/data_prepared_for_ML.csv'

TASK_ID = 'P3'
TASK_LABEL = 'P3 - Sentiment Classification (LEGACY TF-IDF)'
RANDOM_STATE = 42

FINAL_DATA_DIR = Path('DA') / 'final_data'
MODELS_DIR = Path('DA') / 'models'

TASK_DIR = Path('DA') / 'tasks' / 'Sentiment'

# Artifacts (kept compatible with the main task layout)
SAVE_DIR = TASK_DIR
IMG_DIR = TASK_DIR
OPTUNA_DIR = TASK_DIR / 'optuna_history_json'
MODEL_PATH = MODELS_DIR / 'best_model_p3.pkl'
LEGACY_MODEL_PATH = MODELS_DIR / 'sentiment_model.pkl'
FEATURE_NAMES_PATH = SAVE_DIR / 'feature_names_p3.json'
MODEL_COMPARISON_CSV = FINAL_DATA_DIR / 'model_comparison_results_p3.csv'
MODEL_COMPARISON_IMG_CSV = FINAL_DATA_DIR / 'p3_model_comparison_results.csv'


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


def prepare_data_p3(df: pd.DataFrame):
    # 1. TẠO TARGET: SENTIMENT (3 nhãn)
    target_col = 'final_sentiment'
    if target_col not in df.columns:
        raise ValueError(f"❌ Thiếu cột label '{target_col}'!")

    # Ánh xạ 3 nhãn: Negative (0), Neutral (1), Positive (2)
    def map_sentiment(x):
        x = str(x).lower().strip()
        if 'positive' in x:
            return 2
        if 'negative' in x:
            return 0
        return 1  # Neutral

    df = df.copy()
    df['target'] = df[target_col].apply(map_sentiment)

    # 2. Không sampling/undersampling: dùng toàn bộ dữ liệu thật
    print("\n" + "="*80)
    print("📊 PHÂN BỐ DỮ LIỆU (KHÔNG CÂN BẰNG BẰNG SAMPLING) - TASK 3")
    print("="*80)
    labels_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    counts = df['target'].value_counts().to_dict()
    for val, name in labels_map.items():
        count = int(counts.get(val, 0))
        pct = (count / len(df) * 100.0) if len(df) else 0.0
        print(f"   • {name:<10}: {count:>5} bài ({pct:>5.2f}%)")
    print("="*80 + "\n")

    # 3. Lọc Feature (giống P0)
    cols_ignore = [
        'file_name', 'title', 'artists', 'spotify_release_date', 'spotify_track_id',
        'final_sentiment', 'target', 'is_hit', 'spotify_popularity',
        # 8. BIẾN LOẠI BỎ DO ĐA CỘNG TUYẾN (Kết quả từ EDA/VIF)
        'mfcc2_mean',
        'spectral_rolloff',
        'noun_count',
        'verb_count',
        'tempo_stability',
        'spectral_contrast_band3_mean',
        'spectral_contrast_band4_mean',
        'spectral_contrast_band5_mean',
    ]

    numeric_feats = [c for c in df.columns if c not in cols_ignore and pd.api.types.is_numeric_dtype(df[c])]
    if 'lyric' in df.columns:
        df['lyric'] = df['lyric'].fillna('')
    else:
        df['lyric'] = ''

    return df, numeric_feats


def plot_custom_optuna_history(study, model_name, baseline_score, save_dir):
    # Lấy dữ liệu từ study
    df = study.trials_dataframe()
    trial_values = df['value']
    best_values = trial_values.cummax()

    # LƯU CSV để tái sử dụng nếu ảnh lỗi/không ưng ý (khỏi chạy lại Optuna)
    try:
        os.makedirs(str(FINAL_DATA_DIR), exist_ok=True)
        safe_name = str(model_name).replace(' ', '_')
        csv_name = f"{TASK_ID.lower()}_optuna_history_{safe_name}.csv"
        csv_path = os.path.join(str(FINAL_DATA_DIR), csv_name)
        df_csv = df.copy()
        df_csv['best_value_so_far'] = df_csv['value'].cummax()
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
        y=baseline_score,
        color='forestgreen',
        linestyle='--',
        linewidth=1.5,
        label=f'Baseline ({baseline_score:.4f})',
    )

    plt.title(
        f'Lịch sử tối ưu hóa {model_name} (F1-macro)\nBaseline: {baseline_score:.4f} → Best: {study.best_value:.4f}',
        fontsize=14,
        fontweight='bold',
        pad=15,
    )
    plt.xlabel('Số lượt thử (Trial)', fontsize=12)
    plt.ylabel('F1-macro', fontsize=12)
    plt.ylim(min(float(trial_values.min()), float(baseline_score)) - 0.02, 1.0)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend(loc='upper right', frameon=True)
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    file_name = f"optuna_history_{model_name.replace(' ', '_')}.png"
    plt.savefig(os.path.join(save_dir, file_name), dpi=300, bbox_inches='tight')
    plt.close()


def analyze_specific_bigrams_p3(df, keywords=None):
    if keywords is None:
        keywords = ['hai', 'hay', 'nhớ', 'những', 'mắt', 'nay', 'cho', 'chẳng']
    print(f"\n🔍 Đang phân tích ngữ cảnh cho các từ khóa: {keywords}")

    bg_vectorizer = TfidfVectorizer(ngram_range=(2, 2), max_features=2000)
    bg_matrix = bg_vectorizer.fit_transform(df['lyric'])
    bg_names = bg_vectorizer.get_feature_names_out()

    clf = ExtraTreesClassifier(n_estimators=100, random_state=RANDOM_STATE)
    clf.fit(bg_matrix, df['target'])

    bg_imp = pd.DataFrame({'Bigram': bg_names, 'Importance': clf.feature_importances_})
    filtered_bg = bg_imp[bg_imp['Bigram'].str.contains('|'.join(keywords))]
    top_specific_bg = filtered_bg.sort_values(by='Importance', ascending=False).head(20)

    print("🔥 TOP CỤM TỪ NGỮ CẢNH CỦA TỪ KHÓA QUAN TRỌNG:")
    print(top_specific_bg)

    plt.figure(figsize=(12, 10))
    sns.barplot(data=top_specific_bg, x='Importance', y='Bigram', palette='viridis')
    plt.title('Top 20 Bi-grams ngữ cảnh của các từ khóa quan trọng (Sentiment)', fontsize=14, fontweight='bold')
    plt.xlabel('Tầm quan trọng (Importance)', fontsize=12)
    plt.ylabel('Cụm từ (Bi-gram)', fontsize=12)
    plt.tight_layout()

    TASK_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(TASK_DIR / 'p3_specific_bigrams.png'), dpi=300)
    plt.close()
    print(f"✅ Đã lưu biểu đồ Bi-grams tại: {TASK_DIR / 'p3_specific_bigrams.png'}")

    return top_specific_bg


def run_analysis_p3_3_labels():
    # === SET GLOBAL SEED ===
    np.random.seed(RANDOM_STATE)

    df = load_data()
    if df is None:
        return

    df_clean, numeric_feats = prepare_data_p3(df)

    # Sử dụng toàn bộ dữ liệu thay vì sampling để tránh mất thông tin
    df_balanced = df_clean.copy()

    # In phân bố nhãn
    n_neg = len(df_balanced[df_balanced['target'] == 0])
    n_neu = len(df_balanced[df_balanced['target'] == 1])
    n_pos = len(df_balanced[df_balanced['target'] == 2])

    print(f"📊 Phân bố tập dữ liệu (TOÀN BỘ): {n_neg} Negative, {n_neu} Neutral, {n_pos} Positive")
    print(f"   Tổng: {len(df_balanced)} bài hát")
    print("   ⚙️  Sử dụng class_weight='balanced' để tự động cân bằng nhãn")

    # Chuẩn hóa thời gian (giống P0)
    if 'spotify_release_date' in df_balanced.columns:
        def fix_date(val):
            val = str(val)
            if len(val) == 4 and val.isdigit():
                return val + '-01-01'
            return val

        df_balanced['spotify_release_date'] = df_balanced['spotify_release_date'].apply(fix_date)
        df_balanced['spotify_release_date'] = pd.to_datetime(df_balanced['spotify_release_date'], errors='coerce')
        df_balanced = df_balanced.dropna(subset=['spotify_release_date']).sort_values('spotify_release_date')

    X = df_balanced[numeric_feats + ['lyric']]
    y = df_balanced['target']

    print("\n" + "="*60)
    print("🔍 KIỂM TRA CHI TIẾT CÁC BIẾN ĐẦU VÀO (TASK 3 - SENTIMENT)")
    print("="*60)
    print(f"1️⃣ Số lượng biến số (Numeric Features): {len(numeric_feats)}")
    print(np.array(numeric_feats))
    print("-" * 60)
    print("2️⃣ Số lượng biến văn bản (TF-IDF): 100")
    print("-" * 60)
    print(f"✅ TỔNG CỘNG SỐ BIẾN: {len(numeric_feats) + 100}")
    print("="*60 + "\n")

    # TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=5)
    splits = list(tscv.split(X))
    train_idx, test_idx = splits[-1]
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    print(f"📅 Tập Train: {len(train_idx)} bài")
    print(f"📅 Tập Test : {len(test_idx)} bài")
    print("-" * 60)

    # Preprocessing
    vpop_stopwords = [
        'và', 'là', 'thì', 'mà', 'của', 'những', 'cứ', 'này', 'đã', 'đang', 'rồi', 'nhé',
        'nào', 'sẽ', 'cùng', 'với', 'cho', 'tự', 'vì', 'lại', 'bao', 'như', 'nhưng',
        'cũng', 'có', 'không', 'chỉ', 'làm', 'gì', 'một', 'chi', 'khi', 'hết', 'ra',
        'sao', 'hai', 'hay', 'để', 'được', 'thôi', 'thế', 'cái', 'dù', 'đi', 'vậy',
        'đấy', 'ấy', 'nữa', 'chẳng', 'chưa', 'thể', 'đây', 'từ', 'nếu', 'hẳn', 'vẫn', 'tới',
        'hôm', 'nay', 'ngày', 'giờ', 'lúc', 'nơi', 'chuyện', 'mỗi', 'từng', 'đôi',
        'vài', 'mấy', 'cả', 'ở', 'qua', 'tại', 'trong', 'ngoài', 'trên', 'dưới',
        'giữa', 'bên', 'cạnh', 'quanh', 'xa', 'gần', 'đâu', 'kia', 'kìa',
        'anh', 'em', 'ta', 'tôi', 'mình', 'họ', 'chúng', 'nàng', 'chàng', 'ai', 'người',
        'tớ', 'cậu', 'mày', 'tao', 'nó', 'ông', 'bà', 'ba', 'mẹ', 'con', 'bố', 'thằng',
        'nói', 'biết', 'thấy', 'muốn', 'phải', 'đừng', 'nên', 'về', 'đến', 'nghe',
        'còn', 'mới', 'vừa', 'luôn', 'mãi', 'quá', 'lắm', 'thật', 'bỗng', 'chợt',
        'ngỡ', 'bảo', 'kêu', 'tưởng', 'cần', 'nhau', 'ơi', 'à', 'ôi', 'ừm', 'vâng',
        'da', 'oh', 'yeah', 'ah', 'baby', 'yah', 'yo', 'đó',
        'nhiều', 'hơn', 'ít', 'quá', 'lắm', 'cực', 'hết', 'chỉ', 'mọi', 'tất', 'cả', 'thêm', 'bớt',
        'thế', 'vậy', 'đấy', 'này', 'kia', 'ấy', 'đó', 'nào', 'sao', 'gì', 'đâu', 'nơi', 'you',
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                'num',
                Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler()),
                ]),
                numeric_feats,
            ),
            (
                'text',
                TfidfVectorizer(
                    max_features=100,
                    ngram_range=(1, 2),
                    min_df=20,
                    max_df=0.5,
                    stop_words=vpop_stopwords,
                    sublinear_tf=True,
                ),
                'lyric',
            ),
        ]
    )

    # =============================================================================
    # BƯỚC 1: BASELINE
    # =============================================================================
    print("\n" + "="*80)
    print("📊 BƯỚC 1: CHẠY BASELINE - SO SÁNH BAN ĐẦU")
    print("="*80)

    baseline_models = {
        'Extra Trees': ExtraTreesClassifier(n_estimators=300, class_weight='balanced', random_state=RANDOM_STATE),
        'AdaBoost': AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1, class_weight='balanced'),
            n_estimators=300,
            random_state=RANDOM_STATE,
        ),
        'SVM': SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=RANDOM_STATE),
        'Random Forest': RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=RANDOM_STATE),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.1,
            subsample=0.8,
            random_state=RANDOM_STATE,
        ),
        'Logistic Regression': LogisticRegression(random_state=RANDOM_STATE, max_iter=1000, class_weight='balanced'),
        'MLP (Neural Net)': MLPClassifier(
            hidden_layer_sizes=(128, 64),
            max_iter=1000,
            early_stopping=True,
            random_state=RANDOM_STATE,
        ),
        'XGBoost': XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            random_state=RANDOM_STATE,
            eval_metric='mlogloss',
        ),
        'LightGBM': LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            random_state=RANDOM_STATE,
            verbose=-1,
            class_weight='balanced',
        ),
    }

    lr_base = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000, class_weight='balanced')
    gb_base = GradientBoostingClassifier(n_estimators=300, learning_rate=0.1, subsample=0.8, random_state=RANDOM_STATE)
    lgbm_base = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        random_state=RANDOM_STATE,
        verbose=-1,
        class_weight='balanced',
    )

    voting_clf = VotingClassifier(
        estimators=[('lr', lr_base), ('gb', gb_base), ('lgbm', lgbm_base)],
        voting='soft',
    )
    baseline_models['Voting Ensemble (LR+GB+LGBM)'] = voting_clf

    stacking_clf = StackingClassifier(
        estimators=[('gb', gb_base), ('lgbm', lgbm_base)],
        final_estimator=lr_base,
        cv=3,
    )
    baseline_models['Stacking Ensemble (GB+LGBM->LR)'] = stacking_clf

    baseline_results = []
    baseline_pipelines = {}
    print(f"\n{'MODEL':<30} | {'ACC':<8} | {'F1M':<8} | {'PREC_W':<8} | {'REC_W':<8}")
    print("-" * 75)

    for name, model in baseline_models.items():
        try:
            pipeline = Pipeline([('preprocessor', preprocessor), ('clf', model)])
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            f1m = f1_score(y_test, y_pred, average='macro', zero_division=0)
            prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)

            baseline_results.append({
                'Model': name,
                'Accuracy': acc,
                'F1_Macro': f1m,
                'Precision_Weighted': prec,
                'Recall_Weighted': rec,
            })
            baseline_pipelines[name] = pipeline
            print(f"{name:<30} | {acc:.4f}   | {f1m:.4f}   | {prec:.4f}   | {rec:.4f}")
        except Exception as e:
            print(f"❌ Lỗi {name}: {e}")

    baseline_df = pd.DataFrame(baseline_results).sort_values(by='F1_Macro', ascending=False)

    best_baseline_name = baseline_df.iloc[0]['Model']
    best_baseline_f1m = float(baseline_df.iloc[0]['F1_Macro'])
    best_baseline_acc = float(baseline_df.iloc[0]['Accuracy'])

    fig, ax = plt.subplots(figsize=(16, 10))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(baseline_df)))
    ax.barh(baseline_df['Model'], baseline_df['F1_Macro'], color=colors, edgecolor='white', linewidth=1.5)

    for i, (_, row) in enumerate(baseline_df.iterrows()):
        ax.text(float(row['F1_Macro']) + 0.005, i, f"{float(row['F1_Macro']):.4f}", va='center', fontsize=11, fontweight='bold', color='black')

    ax.set_xlabel('F1-macro', fontsize=13, fontweight='bold')
    ax.set_ylabel('Model', fontsize=13, fontweight='bold')
    ax.set_title('So sánh hiệu năng các mô hình phân loại CẢM XÚC (BASELINE) — F1-macro', fontsize=16, fontweight='bold', pad=20)
    ax.invert_yaxis()
    ax.set_xlim(0.0, float(baseline_df['F1_Macro'].max()) + 0.10)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    TASK_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(TASK_DIR / 'p3_baseline_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✅ Đã lưu biểu đồ baseline tại: {TASK_DIR / 'p3_baseline_comparison.png'}")

    labels = ['Negative', 'Neutral', 'Positive']
    best_baseline_model = baseline_models[str(best_baseline_name)]
    pipeline_baseline = Pipeline([('preprocessor', preprocessor), ('clf', best_baseline_model)])
    pipeline_baseline.fit(X_train, y_train)
    y_pred_baseline = pipeline_baseline.predict(X_test)

    plt.figure(figsize=(10, 8))
    cm_baseline = confusion_matrix(y_test, y_pred_baseline)
    sns.heatmap(cm_baseline, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Dự đoán', fontsize=12, fontweight='bold')
    plt.ylabel('Thực tế', fontsize=12, fontweight='bold')
    plt.title(f'Confusion Matrix (3 Nhãn) - {best_baseline_name}\n(F1-macro: {best_baseline_f1m:.4f})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(str(TASK_DIR / 'p3_confusion_matrix_baseline.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Đã lưu Confusion Matrix tại: {TASK_DIR / 'p3_confusion_matrix_baseline.png'}")

    print("\n" + "="*80)
    print("🏆 XÁC ĐỊNH BEST MODEL TỪ BASELINE")
    print("="*80)

    best_baseline_model_name = best_baseline_name
    print(f"\n✨ BEST BASELINE MODEL: {best_baseline_model_name} (F1-macro: {best_baseline_f1m:.4f} | Acc: {best_baseline_acc:.4f})")

    if 'Voting' in best_baseline_model_name or 'Stacking' in best_baseline_model_name:
        models_to_optimize = ['Logistic Regression', 'Gradient Boosting', 'LightGBM']
        print(f"📋 Ensemble model detected → Sẽ tối ưu 3 base models: {', '.join(models_to_optimize)}")
    else:
        models_to_optimize = [best_baseline_model_name]
        print(f"📋 Single model detected → Sẽ tối ưu: {best_baseline_model_name}")

    print("\n" + "="*80)
    print("🔧 BƯỚC 2: TỐI ƯU CHỈ BEST MODEL BẰNG OPTUNA")
    print("="*80)

    SAVE_DIR_IMG = os.path.join('DA', 'images', 'Sentiment')
    OPTUNA_IMG_DIR = os.path.join(SAVE_DIR_IMG, 'optuna_history_image')
    OPTUNA_JSON_DIR = os.path.join(SAVE_DIR_IMG, 'optuna_history_json')
    MODEL_DIR = SAVE_DIR_IMG

    os.makedirs(OPTUNA_IMG_DIR, exist_ok=True)
    os.makedirs(OPTUNA_JSON_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    optimized_params = {}

    for idx, model_name in enumerate(models_to_optimize, 1):
        print(f"\n{'='*70}")
        print(f"🔧 [{idx}/{len(models_to_optimize)}] ĐANG TỐI ƯU: {model_name}")
        print(f"{'='*70}")

        params_file = os.path.join(OPTUNA_JSON_DIR, f"best_params_{model_name.lower().replace(' ', '_')}.json")

        if os.path.exists(params_file):
            print("✅ Đã tồn tại params, đang load...")
            with open(params_file, 'r', encoding='utf-8') as f:
                optimized_params[model_name] = json.load(f)
            continue

        if model_name == 'Extra Trees':
            def objective_et(trial):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
                    'max_depth': trial.suggest_int('max_depth', 10, 30),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 15),
                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
                    'class_weight': 'balanced',
                    'random_state': RANDOM_STATE,
                }
                clf = ExtraTreesClassifier(**params)
                pipe = Pipeline([('preprocessor', preprocessor), ('clf', clf)])
                from sklearn.model_selection import cross_val_score
                cv_scores = cross_val_score(pipe, X_train, y_train, cv=3, scoring='f1_macro', n_jobs=-1)
                return cv_scores.mean()

            study = optuna.create_study(direction='maximize', study_name='P3_Extra_Trees')
            study.optimize(objective_et, n_trials=50, show_progress_bar=True)
            optimized_params[model_name] = study.best_params

            with open(params_file, 'w', encoding='utf-8') as f:
                json.dump(study.best_params, f, indent=4, ensure_ascii=False)
            print(f"✅ Best CV F1-macro: {study.best_value:.4f}")
            plot_custom_optuna_history(study=study, model_name='Extra Trees', baseline_score=best_baseline_f1m, save_dir=OPTUNA_IMG_DIR)

        elif model_name == 'Random Forest':
            def objective_rf(trial):
                p = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
                    'max_depth': trial.suggest_int('max_depth', 10, 30),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 15),
                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                    'class_weight': 'balanced',
                    'random_state': RANDOM_STATE,
                }
                clf = RandomForestClassifier(**p)
                pipe = Pipeline([('preprocessor', preprocessor), ('clf', clf)])
                from sklearn.model_selection import cross_val_score
                cv_scores = cross_val_score(pipe, X_train, y_train, cv=3, scoring='f1_macro', n_jobs=-1)
                return cv_scores.mean()

            study = optuna.create_study(direction='maximize', study_name='P3_Random_Forest')
            study.optimize(objective_rf, n_trials=50, show_progress_bar=True)
            optimized_params[model_name] = study.best_params

            with open(params_file, 'w', encoding='utf-8') as f:
                json.dump(study.best_params, f, indent=4, ensure_ascii=False)
            print(f"✅ Best CV F1-macro: {study.best_value:.4f}")
            plot_custom_optuna_history(study=study, model_name='Random Forest', baseline_score=best_baseline_f1m, save_dir=OPTUNA_IMG_DIR)

        elif model_name == 'AdaBoost':
            def objective_ada(trial):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 500, step=50),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 1.0, log=True),
                    'random_state': RANDOM_STATE,
                }
                clf = AdaBoostClassifier(
                    estimator=DecisionTreeClassifier(max_depth=1, class_weight='balanced'),
                    **params,
                )
                pipe = Pipeline([('preprocessor', preprocessor), ('clf', clf)])
                from sklearn.model_selection import cross_val_score
                cv_scores = cross_val_score(pipe, X_train, y_train, cv=3, scoring='f1_macro', n_jobs=-1)
                return cv_scores.mean()

            study = optuna.create_study(direction='maximize', study_name='P3_AdaBoost')
            study.optimize(objective_ada, n_trials=50, show_progress_bar=True)
            optimized_params[model_name] = study.best_params

            with open(params_file, 'w', encoding='utf-8') as f:
                json.dump(study.best_params, f, indent=4, ensure_ascii=False)
            print(f"✅ Best CV F1-macro: {study.best_value:.4f}")
            plot_custom_optuna_history(study=study, model_name='AdaBoost', baseline_score=best_baseline_f1m, save_dir=OPTUNA_IMG_DIR)

        elif model_name == 'SVM':
            def objective_svm(trial):
                params = {
                    'C': trial.suggest_float('C', 0.1, 100.0, log=True),
                    'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
                    'kernel': 'rbf',
                    'probability': True,
                    'class_weight': 'balanced',
                    'random_state': RANDOM_STATE,
                }
                clf = SVC(**params)
                pipe = Pipeline([('preprocessor', preprocessor), ('clf', clf)])
                from sklearn.model_selection import cross_val_score
                cv_scores = cross_val_score(pipe, X_train, y_train, cv=3, scoring='f1_macro', n_jobs=-1)
                return cv_scores.mean()

            study = optuna.create_study(direction='maximize', study_name='P3_SVM')
            study.optimize(objective_svm, n_trials=50, show_progress_bar=True)
            optimized_params[model_name] = study.best_params

            with open(params_file, 'w', encoding='utf-8') as f:
                json.dump(study.best_params, f, indent=4, ensure_ascii=False)
            print(f"✅ Best CV F1-macro: {study.best_value:.4f}")
            plot_custom_optuna_history(study=study, model_name='SVM', baseline_score=best_baseline_f1m, save_dir=OPTUNA_IMG_DIR)

        elif model_name == 'Logistic Regression':
            def objective_lr(trial):
                params = {
                    'C': trial.suggest_float('C', 0.001, 100.0, log=True),
                    'solver': trial.suggest_categorical('solver', ['lbfgs', 'saga']),
                    'max_iter': 2000,
                    'class_weight': 'balanced',
                    'random_state': RANDOM_STATE,
                }
                clf = LogisticRegression(**params)
                pipe = Pipeline([('preprocessor', preprocessor), ('clf', clf)])
                from sklearn.model_selection import cross_val_score
                cv_scores = cross_val_score(pipe, X_train, y_train, cv=3, scoring='f1_macro', n_jobs=-1)
                return cv_scores.mean()

            study = optuna.create_study(direction='maximize', study_name='P3_Logistic_Regression')
            study.optimize(objective_lr, n_trials=50, show_progress_bar=True)
            optimized_params[model_name] = study.best_params

            with open(params_file, 'w', encoding='utf-8') as f:
                json.dump(study.best_params, f, indent=4, ensure_ascii=False)
            print(f"✅ Best CV F1-macro: {study.best_value:.4f}")
            plot_custom_optuna_history(study=study, model_name='Logistic Regression', baseline_score=best_baseline_f1m, save_dir=OPTUNA_IMG_DIR)

        elif model_name == 'Gradient Boosting':
            def objective_gb(trial):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'random_state': RANDOM_STATE,
                }
                clf = GradientBoostingClassifier(**params)
                pipe = Pipeline([('preprocessor', preprocessor), ('clf', clf)])
                from sklearn.model_selection import cross_val_score
                cv_scores = cross_val_score(pipe, X_train, y_train, cv=3, scoring='f1_macro', n_jobs=-1)
                return cv_scores.mean()

            study = optuna.create_study(direction='maximize', study_name='P3_Gradient_Boosting')
            study.optimize(objective_gb, n_trials=50, show_progress_bar=True)
            optimized_params[model_name] = study.best_params

            with open(params_file, 'w', encoding='utf-8') as f:
                json.dump(study.best_params, f, indent=4, ensure_ascii=False)
            print(f"✅ Best CV F1-macro: {study.best_value:.4f}")
            plot_custom_optuna_history(study=study, model_name='Gradient Boosting', baseline_score=best_baseline_f1m, save_dir=OPTUNA_IMG_DIR)

        elif model_name == 'LightGBM':
            def objective_lgbm(trial):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'class_weight': 'balanced',
                    'random_state': RANDOM_STATE,
                    'verbose': -1,
                }
                clf = LGBMClassifier(**params)
                pipe = Pipeline([('preprocessor', preprocessor), ('clf', clf)])
                from sklearn.model_selection import cross_val_score
                cv_scores = cross_val_score(pipe, X_train, y_train, cv=3, scoring='f1_macro', n_jobs=-1)
                return cv_scores.mean()

            study = optuna.create_study(direction='maximize', study_name='P3_LightGBM')
            study.optimize(objective_lgbm, n_trials=50, show_progress_bar=True)
            optimized_params[model_name] = study.best_params

            with open(params_file, 'w', encoding='utf-8') as f:
                json.dump(study.best_params, f, indent=4, ensure_ascii=False)
            print(f"✅ Best CV F1-macro: {study.best_value:.4f}")
            plot_custom_optuna_history(study=study, model_name='LightGBM', baseline_score=best_baseline_f1m, save_dir=OPTUNA_IMG_DIR)

        else:
            print(f"⚠️  Legacy Optuna objective chưa hỗ trợ model: {model_name} → skip")

    print("\n" + "="*80)
    print("🎉 HOÀN TẤT TỐI ƯU HÓA BEST MODEL")
    print("="*80)

    # Mirror Optuna params into DA/tasks/Sentiment/optuna_history_json
    try:
        OPTUNA_DIR.mkdir(parents=True, exist_ok=True)
        for m_name, params in optimized_params.items():
            safe = str(m_name).lower().replace(' ', '_').replace('(', '').replace(')', '').replace('+', '').replace('->', 'to')
            out_path = OPTUNA_DIR / f'p3_{safe}_best_params.json'
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(params, f, ensure_ascii=False, indent=2)
        print(f"✅ Đã mirror Optuna params tại: {OPTUNA_DIR}")
    except Exception as e:
        print(f"⚠️  Không thể mirror Optuna params: {e}")

    # =============================================================================
    # Build & evaluate optimized/baseline with rollback
    # =============================================================================
    optimized_models: dict[str, object] = {}

    for model_name in models_to_optimize:
        params = optimized_params.get(model_name)
        if not isinstance(params, dict):
            params = {}

        if model_name == 'Logistic Regression':
            p = {**params}
            p.setdefault('class_weight', 'balanced')
            p.setdefault('random_state', RANDOM_STATE)
            p.setdefault('max_iter', 2000)
            optimized_models[model_name] = LogisticRegression(**p)
        elif model_name == 'Gradient Boosting':
            p = {**params}
            p.setdefault('random_state', RANDOM_STATE)
            optimized_models[model_name] = GradientBoostingClassifier(**p)
        elif model_name == 'LightGBM':
            p = {**params}
            p.setdefault('class_weight', 'balanced')
            p.setdefault('random_state', RANDOM_STATE)
            p.setdefault('verbose', -1)
            optimized_models[model_name] = LGBMClassifier(**p)
        elif model_name == 'Extra Trees':
            p = {**params}
            p.setdefault('class_weight', 'balanced')
            p.setdefault('random_state', RANDOM_STATE)
            p.setdefault('n_estimators', 300)
            optimized_models[model_name] = ExtraTreesClassifier(**p)
        elif model_name == 'Random Forest':
            p = {**params}
            p.setdefault('class_weight', 'balanced')
            p.setdefault('random_state', RANDOM_STATE)
            p.setdefault('n_estimators', 300)
            optimized_models[model_name] = RandomForestClassifier(**p)
        elif model_name == 'AdaBoost':
            p = {**params}
            p.setdefault('random_state', RANDOM_STATE)
            optimized_models[model_name] = AdaBoostClassifier(
                estimator=DecisionTreeClassifier(max_depth=1, class_weight='balanced'),
                **p,
            )
        elif model_name == 'SVM':
            p = {**params}
            p.setdefault('kernel', 'rbf')
            p.setdefault('probability', True)
            p.setdefault('class_weight', 'balanced')
            optimized_models[model_name] = SVC(random_state=RANDOM_STATE, **p)

    if 'Voting' in best_baseline_model_name or 'Stacking' in best_baseline_model_name:
        lr_opt = optimized_models.get('Logistic Regression')
        gb_opt = optimized_models.get('Gradient Boosting')
        lgbm_opt = optimized_models.get('LightGBM')
        if lr_opt is not None and gb_opt is not None and lgbm_opt is not None:
            optimized_models['Voting Ensemble (LR+GB+LGBM)'] = VotingClassifier(
                estimators=[('lr', lr_opt), ('gb', gb_opt), ('lgbm', lgbm_opt)],
                voting='soft',
            )
            optimized_models['Stacking Ensemble (GB+LGBM->LR)'] = StackingClassifier(
                estimators=[('gb', gb_opt), ('lgbm', lgbm_opt)],
                final_estimator=lr_opt,
                cv=5,
            )

    print("\n" + "="*80)
    print("📊 BƯỚC 4: ĐÁNH GIÁ OPTIMIZED BEST MODEL")
    print("="*80)

    optimized_results = []
    best_f1m = -1.0
    best_acc = 0.0
    final_best_model_name = ""

    print(f"\n{'MODEL':<40} | {'F1M':<8} | {'ACC':<8} | {'F1W':<8} | {'PREC_W':<8} | {'REC_W':<8}")
    print("-" * 85)

    for name, model in optimized_models.items():
        try:
            pipeline = Pipeline([('preprocessor', preprocessor), ('clf', model)])
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            f1m = f1_score(y_test, y_pred, average='macro', zero_division=0)
            f1w = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)

            optimized_results.append({
                'Model': name,
                'F1_Macro': f1m,
                'Accuracy': acc,
                'F1_Weighted': f1w,
                'Precision_Weighted': prec,
                'Recall_Weighted': rec,
            })
            print(f"{name:<40} | {f1m:.4f}   | {acc:.4f}   | {f1w:.4f}   | {prec:.4f}   | {rec:.4f}")

            if f1m > best_f1m:
                best_f1m = f1m
                best_acc = acc
                final_best_model_name = name
        except Exception as e:
            print(f"❌ Lỗi {name}: {e}")

    if not optimized_results:
        print("\n⚠️  Không có optimized model nào được đánh giá → dùng baseline pipeline.")
        final_best_model_name = str(best_baseline_name)
        best_pipe = baseline_pipelines[str(best_baseline_name)]
        y_pred_final = best_pipe.predict(X_test)
        acc_final = float(accuracy_score(y_test, y_pred_final))
        f1m_final = float(f1_score(y_test, y_pred_final, average='macro', zero_division=0))
        optimized_results = [r for r in baseline_results if r['Model'] == best_baseline_name]
    else:
        optimized_df = pd.DataFrame(optimized_results).sort_values(by='F1_Macro', ascending=False)
        best_optimized_f1m = float(optimized_df.iloc[0]['F1_Macro'])
        best_optimized_name = optimized_df.iloc[0]['Model']

        print("\n" + "="*80)
        print("⚖️ BƯỚC 4.3: KIỂM TRA VÀ ROLLBACK NẾU CẦN")
        print("="*80)
        print(f"   • Baseline:  {best_baseline_name} | F1m={best_baseline_f1m:.4f} | Acc={best_baseline_acc:.4f}")
        print(f"   • Optimized: {best_optimized_name} | F1m={best_optimized_f1m:.4f}")

        if best_optimized_f1m < best_baseline_f1m:
            print("\n⚠️ Optimized < Baseline → rollback")
            final_best_model_name = best_baseline_name
            best_pipe = baseline_pipelines[best_baseline_name]
            optimized_results = [r for r in baseline_results if r['Model'] == best_baseline_name]
        else:
            final_best_model_name = str(best_optimized_name)
            best_model = optimized_models[final_best_model_name]
            best_pipe = Pipeline([('preprocessor', preprocessor), ('clf', best_model)])
            best_pipe.fit(X_train, y_train)

        y_pred_final = best_pipe.predict(X_test)
        acc_final = float(accuracy_score(y_test, y_pred_final))
        f1m_final = float(f1_score(y_test, y_pred_final, average='macro', zero_division=0))

    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred_final)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Dự đoán', fontsize=12, fontweight='bold')
    plt.ylabel('Thực tế', fontsize=12, fontweight='bold')
    plt.title(
        f'Confusion Matrix (Sau Optuna) - {final_best_model_name}\n(F1-macro: {f1m_final:.4f} | Acc: {acc_final:.4f})',
        fontsize=14,
        fontweight='bold',
    )
    plt.tight_layout()
    TASK_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(TASK_DIR / 'p3_confusion_matrix_optimized.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print("\n📋 Báo cáo phân loại:")
    print(classification_report(y_test, y_pred_final, target_names=labels))

    pre = best_pipe.named_steps['preprocessor']
    clf = best_pipe.named_steps['clf']

    # Feature importance (coefficients or feature_importances_)
    try:
        if hasattr(clf, 'coef_'):
            coef = np.abs(clf.coef_).mean(axis=0)
            tfidf_names = pre.named_transformers_['text'].get_feature_names_out().tolist()
            all_feats = numeric_feats + tfidf_names
            feat_imp_df = pd.DataFrame({'Feature': all_feats, 'Importance': coef}).sort_values('Importance', ascending=False).head(20)
        elif hasattr(clf, 'feature_importances_'):
            importances = clf.feature_importances_
            tfidf_names = pre.named_transformers_['text'].get_feature_names_out().tolist()
            all_feats = numeric_feats + tfidf_names
            feat_imp_df = pd.DataFrame({'Feature': all_feats, 'Importance': importances}).sort_values('Importance', ascending=False).head(20)
        else:
            feat_imp_df = pd.DataFrame(columns=['Feature', 'Importance'])

        feat_imp_df = rename_topics_for_report(feat_imp_df, column_name='Feature')
        if not feat_imp_df.empty:
            fig, ax = plt.subplots(figsize=(12, 10))
            feat_imp_df_sorted = feat_imp_df.sort_values('Importance', ascending=True)
            colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(feat_imp_df_sorted)))
            ax.barh(range(len(feat_imp_df_sorted)), feat_imp_df_sorted['Importance'], color=colors, edgecolor='black', linewidth=0.8)
            ax.set_yticks(range(len(feat_imp_df_sorted)))
            ax.set_yticklabels(feat_imp_df_sorted['Feature'], fontsize=10)
            ax.set_title(f'Top 20 Features - {final_best_model_name}', fontsize=14, fontweight='bold', pad=20)
            plt.tight_layout()
            plt.savefig(str(TASK_DIR / 'p3_feature_importance.png'), dpi=300, bbox_inches='tight')
            plt.close()
    except Exception as e:
        print(f"⚠️  Không thể vẽ feature importance: {e}")

    # Save model & results
    print("\n" + "="*80)
    print("💾 ĐANG LƯU MODELS VÀ KẾT QUẢ")
    print("="*80)

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    pkl_data = {
        'pkl_schema_version': 1,
        'task_id': TASK_ID,
        'data_source': str(Path(FILE_DATA)),
        'pipeline': best_pipe,
        'model_name': final_best_model_name,
        'accuracy': float(acc_final),
        'test_f1_macro': float(f1m_final),
        'labels': labels,
        'shap_cache': None,
    }

    if BUILD_SHAP_CACHE and build_shap_cache is not None and ShapCacheConfig is not None:
        try:
            pkl_data['shap_cache'] = build_shap_cache(
                X_train,
                X_test,
                config=ShapCacheConfig(n_background=200, n_explain=300, random_state=RANDOM_STATE),
            )
        except Exception as e:
            print(f"⚠️  build_shap_cache failed → continue without SHAP cache: {e}")

    joblib.dump(pkl_data, str(MODEL_PATH))
    try:
        joblib.dump(pkl_data, str(LEGACY_MODEL_PATH))
    except Exception:
        pass

    _save_feature_names_p3(best_pipe)

    results_df = pd.DataFrame(optimized_results)
    FINAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(str(MODEL_COMPARISON_IMG_CSV), index=False)
    results_df.to_csv(str(MODEL_COMPARISON_CSV), index=False)

    print(f"\n✅ Đã lưu best model tại: {MODEL_PATH}")
    print(f"✅ Đã lưu kết quả so sánh tại: {MODEL_COMPARISON_IMG_CSV}")
    print(f"✅ Đã lưu kết quả so sánh tại: {MODEL_COMPARISON_CSV}")


if __name__ == '__main__':
    run_analysis_p3_3_labels()
