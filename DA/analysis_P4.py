import pandas as pd
import numpy as np
import ast
import re
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, ClassifierMixin, clone

# Wrapper
from sklearn.multioutput import MultiOutputClassifier

# Models
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import (
    ExtraTreesClassifier, RandomForestClassifier, 
    GradientBoostingClassifier, AdaBoostClassifier,
    VotingClassifier, StackingClassifier
)

# Metrics
from sklearn.metrics import classification_report, f1_score, hamming_loss

warnings.filterwarnings('ignore')

# =============================================================================
# 1. CLASS: SMART STACKING (GIỮ NGUYÊN)
# =============================================================================
class SmartStackingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, estimators, final_estimator, cv, fallback_estimator):
        self.estimators = estimators
        self.final_estimator = final_estimator
        self.cv = cv
        self.fallback_estimator = fallback_estimator
        self.model_ = None
        self.classes_ = None

    def fit(self, X, y):
        try:
            if len(np.unique(y)) < 2: raise ValueError("Single class")
            self.model_ = StackingClassifier(
                estimators=self.estimators,
                final_estimator=self.final_estimator,
                cv=self.cv
            )
            self.model_.fit(X, y)
        except Exception:
            self.model_ = clone(self.fallback_estimator)
            self.model_.fit(X, y)
        
        if hasattr(self.model_, 'classes_'):
            self.classes_ = self.model_.classes_
        else:
            self.classes_ = [0, 1]
        return self

    def predict(self, X):
        return self.model_.predict(X)
    
    def predict_proba(self, X):
        return self.model_.predict_proba(X)

# =============================================================================
# 2. LOAD DỮ LIỆU (Full Feature Audio + NLP)
# =============================================================================
FILE_AUDIO = 'mergerd_balanced_and_features.csv' 
FILE_NLP = 'nlp_analysis.csv'                   

def load_and_merge_data():
    print("⏳ Đang tải dữ liệu TASK 4 (CHECKING FEATURES)...")
    try:
        path_audio = FILE_AUDIO if os.path.exists(FILE_AUDIO) else f'final_data/{FILE_AUDIO}'
        path_nlp = FILE_NLP if os.path.exists(FILE_NLP) else f'Audio_lyric/{FILE_NLP}'
        
        if not os.path.exists(path_audio) or not os.path.exists(path_nlp):
             print(f"❌ Không tìm thấy file dữ liệu.")
             return None

        df_main = pd.read_csv(path_audio)
        df_nlp_source = pd.read_csv(path_nlp)
        
        df_main['file_name'] = df_main['file_name'].astype(str).str.strip().str.lower()
        df_nlp_source['file_name'] = df_nlp_source['file_name'].astype(str).str.strip().str.lower()
        
        # Merge để đảm bảo có đủ cột NLP (nếu file audio chưa có)
        # Lấy tất cả cột từ NLP trừ file_name
        cols_nlp = [c for c in df_nlp_source.columns if c != 'file_name']
        
        # Kiểm tra xem các cột này đã có trong df_main chưa
        cols_to_merge = [c for c in cols_nlp if c not in df_main.columns]
        
        if cols_to_merge:
            print(f"➕ Đang ghép thêm {len(cols_to_merge)} đặc trưng từ file NLP...")
            df = pd.merge(df_main, df_nlp_source[['file_name'] + cols_to_merge], on='file_name', how='inner')
        else:
            print("✅ File chính đã có đủ đặc trưng NLP.")
            df = df_main.copy()
            # Vẫn merge inner để lọc bài thiếu lyric nếu cần
            if 'lyric' in df.columns:
                 df = df[df['lyric'].notna()]

        print(f"✅ Đã load dữ liệu. Tổng số dòng: {len(df)}")
        return df
    except Exception as e:
        print(f"❌ Lỗi load data: {e}")
        return None

# =============================================================================
# 3. XỬ LÝ TARGET & FEATURES (CÓ IN DANH SÁCH FEATURE)
# =============================================================================
def extract_all_genres(genre_raw):
    try:
        if isinstance(genre_raw, str):
            cleaned = genre_raw.replace('[', '').replace(']', '').replace("'", "").replace('"', "")
            genres = [g.strip().lower() for g in cleaned.split(',')]
        elif isinstance(genre_raw, list):
            genres = [str(g).strip().lower() for g in genre_raw]
        else:
            return []
            
        mapped_genres = set()
        for g in genres:
            if not g: continue
            if 'vinahouse' in g: mapped_genres.add('Vinahouse')
            elif 'hip hop' in g or 'rap' in g: mapped_genres.add('Hip-Hop')
            elif 'indie' in g: mapped_genres.add('Indie')
            elif 'v-pop' in g or 'pop' in g: mapped_genres.add('V-Pop')
        return list(mapped_genres)
    except:
        return []

def preprocess_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'[^a-záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def run_final_check_features():
    # 1. Load Data
    df = load_and_merge_data()
    if df is None: return

    # 2. Tạo Target
    df['genres_list'] = df['spotify_genres'].apply(extract_all_genres)
    df = df[df['genres_list'].apply(len) > 0]
    
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(df['genres_list'])
    classes = mlb.classes_
    print(f"✅ Class Labels: {classes}")

    # 3. Features Prep
    df['clean_lyric'] = df['lyric'].fillna('').apply(preprocess_text)
    
    # DANH SÁCH BỎ QUA (Đảm bảo KHÔNG bỏ qua các cột NLP stats)
    cols_ignore = [
        'file_name', 'title', 'artists', 'spotify_release_date', 'track_id',
        'spotify_genres', 'genres_list', 'target_genre', 'primary_genre',
        'is_hit', 'hit', 'target', 'total_plays', 'spotify_streams', 'spotify_popularity',
        'lyric', 'clean_lyric', 'Unnamed: 0',
        'sentiment_confidence','sentiment', 'cluster', 'musical_key'
    ]
    
    # Lọc lấy tất cả cột số (Audio + NLP Stats)
    numeric_feats = [c for c in df.columns if c not in cols_ignore and pd.api.types.is_numeric_dtype(df[c])]
    
    # 1. Lưu số lượng ban đầu sau khi load
    total_initial = len(df)

    # 2. Áp dụng hàm trích xuất nhãn (Hàm này chỉ giữ lại 4 dòng nhạc mục tiêu)
    df['genres_list'] = df['spotify_genres'].apply(extract_all_genres)

    # 3. Lọc bỏ các bài hát không thuộc 4 nhóm trên (Gồm Bolero, Lo-fi, v.v.)
    df_filtered = df[df['genres_list'].apply(len) > 0]
    total_after = len(df_filtered)

    # 4. In thống kê
    print("\n" + "="*50)
    print("📊 THỐNG KÊ SAU KHI SÀNG LỌC NHÃN NHIỄU (BOLERO/LO-FI)")
    print("="*50)
    print(f"🔹 Tổng số dữ liệu ban đầu:         {total_initial} bài hát")
    print(f"🔸 Số lượng sau khi loại bỏ Bolero: {total_after} bài hát")
    print(f"❌ Số lượng bài hát bị loại bỏ:     {total_initial - total_after} bài hát")
    print("="*50)

    # =========================================================================
    # 🔍 KIỂM TRA SỐ LƯỢNG FEATURES
    # =========================================================================
    print("\n" + "="*60)
    print("🔍 KIỂM TRA CHI TIẾT CÁC BIẾN ĐẦU VÀO (FEATURES)")
    print("="*60)
    
    print(f"1️⃣ Số lượng biến số (Numeric Features): {len(numeric_feats)}")
    print("📋 Danh sách chi tiết các biến số:")
    print(np.array(numeric_feats)) # In dạng array cho gọn
    
    print("-" * 60)
    print(f"2️⃣ Số lượng biến văn bản (TF-IDF): 100 (Cấu hình trong Pipeline)")
    
    total_feats = len(numeric_feats) + 100
    print("-" * 60)
    print(f"✅ TỔNG CỘNG SỐ BIẾN ĐƯA VÀO MÔ HÌNH: {total_feats}")
    print("="*60 + "\n")

    
    X = df[numeric_feats + ['clean_lyric']]

    # 4. Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 5. Pipeline
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')), 
        ('scaler', StandardScaler())
    ])
    text_transformer = TfidfVectorizer(max_features=100, ngram_range=(1, 2))

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_feats),
        ('text', text_transformer, 'clean_lyric')
    ])

    # 6. MODEL CONFIG (Voting + Smart Stacking)
    et_base = ExtraTreesClassifier(n_estimators=200, class_weight='balanced', random_state=42)
    rf_base = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
    gb_base = GradientBoostingClassifier(n_estimators=100, random_state=42)
    mlp_base = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
    knn_base = KNeighborsClassifier(n_neighbors=5)
    
    voting_core = VotingClassifier(
        estimators=[('gb', gb_base), ('mlp', mlp_base), ('et', et_base)],
        voting='soft', weights=[2, 1, 1]
    )
    
    stacking_smart = SmartStackingClassifier(
        estimators=[('gb', gb_base), ('rf', rf_base), ('knn', knn_base)],
        final_estimator=LogisticRegression(),
        cv=3,
        fallback_estimator=rf_base 
    )

    models = {
        'Decision Tree': MultiOutputClassifier(DecisionTreeClassifier(max_depth=10, class_weight='balanced', random_state=42)),
        'Naive Bayes': MultiOutputClassifier(BernoulliNB()),
        'Ridge Classifier': MultiOutputClassifier(RidgeClassifier(class_weight='balanced', random_state=42)),
        'KNN': MultiOutputClassifier(knn_base),
        'SVM (RBF)': MultiOutputClassifier(SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42)),
        'Random Forest': MultiOutputClassifier(rf_base),
        'Extra Trees': MultiOutputClassifier(et_base),
        'AdaBoost': MultiOutputClassifier(AdaBoostClassifier(n_estimators=100, random_state=42)),
        'Gradient Boosting': MultiOutputClassifier(gb_base),
        'MLP Neural Net': MultiOutputClassifier(mlp_base),
        'Voting (GB+MLP+ET)': MultiOutputClassifier(voting_core),
        'Stacking (GB+RF+KNN)': MultiOutputClassifier(stacking_smart)
    }

    fitted_pipelines = {}
    
    print("🚀 BẮT ĐẦU HUẤN LUYỆN...")
    print(f"{'MODEL':<25} | {'F1-MICRO':<10} | {'HAMMING':<10}")
    print("-" * 55)

    results = []
    best_score = 0
    best_model_name = ""

    for name, model in models.items():
        try:
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('clf', model)
            ])
            
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            
            f1_mic = f1_score(y_test, y_pred, average='micro')
            hl = hamming_loss(y_test, y_pred)
            
            print(f"{name:<25} | {f1_mic:.4f}     | {hl:.4f}")
            results.append({'Model': name, 'F1-Micro': f1_mic, 'Hamming Loss': hl})

            fitted_pipelines[name] = pipeline
            
            if f1_mic > best_score:
                best_score = f1_mic
                best_model_name = name
                best_pred = y_pred

        except Exception as e:
            print(f"❌ Lỗi {name}: {e}")

    # 7. Kết quả
    print("-" * 65)
    print(f"🏆 QUÁN QUÂN: {best_model_name} (F1-Micro: {best_score:.4f})")
    
    print(f"\n📊 Báo cáo chi tiết của {best_model_name}:")
    print(classification_report(y_test, best_pred, target_names=classes))

    if results:
        res_df = pd.DataFrame(results).sort_values(by='F1-Micro', ascending=False)
        plt.figure(figsize=(14, 8))
        sns.barplot(data=res_df, x='F1-Micro', y='Model', palette='turbo')
        for index, row in enumerate(res_df.iterrows()):
            plt.text(row[1]['F1-Micro'], index, f"{row[1]['F1-Micro']:.4f}", va='center', fontsize=10, fontweight='bold')
        plt.title('BXH 12 Model (Full 168 Features)')
        plt.xlabel('F1-Micro Score')
        plt.tight_layout()
        os.makedirs('DA/images', exist_ok=True)
        plt.savefig('DA/images/P4_big_battle_168_check.png')
        print("\n✅ Đã lưu biểu đồ tại: DA/images/P4_big_battle_168_check.png")
    
    save_dir = 'DA/pkl_file'
    os.makedirs(save_dir, exist_ok=True)
    
    # 💾 FILE 1: Lưu TẤT CẢ model (Dùng cho Streamlit so sánh)
    # File này chứa Dictionary gồm 12 pipeline đã train
    all_models_path = f'{save_dir}/all_models_p4_genre.pkl'
    joblib.dump({
        'models': fitted_pipelines,    # Dict chứa 12 pipeline (Preprocess + Model)
        'classes': classes,            # Danh sách nhãn (Hip-Hop, Indie, V-Pop, Vinahouse)
        'numeric_feats': numeric_feats # Danh sách feature đầu vào để kiểm tra khớp lệnh
    }, all_models_path)
    print(f"\n💾 Đã lưu bộ toàn tập model tại: {all_models_path}")
    
    # 💾 FILE 2: Lưu BEST model (Dùng cho Production/Tương lai)
    # File này chỉ chứa duy nhất 1 pipeline tốt nhất cho nhẹ
    if best_model_name in fitted_pipelines:
        best_model_path = f'{save_dir}/best_model_p4_genre.pkl'
        joblib.dump({
            'pipeline': fitted_pipelines[best_model_name],
            'model_name': best_model_name,
            'classes': classes,
            'score': best_score
        }, best_model_path)
        print(f"💾 Đã lưu model tốt nhất ({best_model_name}) tại: {best_model_path}")
    
    print("\n✅ HOÀN TẤT LƯU FILE!")
if __name__ == "__main__":
    run_final_check_features()