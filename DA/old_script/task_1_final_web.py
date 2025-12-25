'''
Lyrics ✅
Đang lưu tất cả vào file: multi_model_hit_prediction.pkl ...✅

BẮT ĐẦU HUẤN LUYỆN 7 MÔ HÌNH (HYBRID AUDIO + LYRICS)...
   Training Random Forest...
     -> Độ chính xác (Accuracy): 0.5282
   Training XGBoost...
     -> Độ chính xác (Accuracy): 0.5641
   Training Logistic Regression...
     -> Độ chính xác (Accuracy): 0.6000
   Training SVM (Support Vector Machine)...
     -> Độ chính xác (Accuracy): 0.5590
   Training Decision Tree...
     -> Độ chính xác (Accuracy): 0.5077
   Training K-Nearest Neighbors (KNN)...
     -> Độ chính xác (Accuracy): 0.5385
   Training Neural Network (MLP)...
     -> Độ chính xác (Accuracy): 0.5333


'''

import pandas as pd
import numpy as np
import re
import joblib
import warnings

# Sklearn Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# Algorithms (Đầy đủ 7 model như yêu cầu)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb

# Tắt cảnh báo để log gọn gàng
warnings.filterwarnings('ignore')

# =============================================================================
# 1. CẤU HÌNH & DANH SÁCH TỪ VÔ NGHĨA (STOPWORDS)
# =============================================================================
# Danh sách này giúp loại bỏ các từ không mang ý nghĩa phân loại (như "anh", "em")
# để mô hình tập trung vào các từ khóa quan trọng (như "yêu", "đau", "nhớ"...)
vn_stopwords = [
    "anh", "em", "tôi", "ta", "mình", "nó", "chúng_ta", "bạn", "người", 
    "là", "của", "và", "những", "các", "trong", "đã", "đang", "vẫn", 
    "thì", "mà", "có", "một", "được", "tại", "vì", "nên", "hay", "hoặc",
    "cứ", "thôi", "biết", "chưa", "không", "chẳng", "rồi", "lắm", "quá",
    "đi", "ơi", "à", "nhé", "nha", "đây", "nào", "sẽ", "muốn", "phải",
    "từng", "lên", "xuống", "ra", "vào", "để", "lại", "gì", "này", "đó", "đâu"
]

# =============================================================================
# 2. LOAD & GHÉP DỮ LIỆU
# =============================================================================
print("🔄 Đang load dữ liệu...")

try:
    # 2.1 Load Audio Data (File gốc đã cân bằng)
    df_main = pd.read_csv(r'final_data\merged_balanced_500_500.csv')
    # Chuẩn hóa tên bài hát để ghép
    df_main['clean_title'] = df_main['title'].astype(str).str.lower().str.strip()
    
    # 2.2 Load Lyrics Data (File chứa lời bài hát)
    try:
        df_lyrics = pd.read_csv(r'Audio_lyric\lyrics_extracted.csv')
    except:
        # Thử đường dẫn dự phòng nếu file nằm ngay thư mục gốc
        df_lyrics = pd.read_csv('lyrics_extracted.csv') 
        
    df_lyrics['clean_title'] = df_lyrics['file_name'].astype(str).str.replace('.mp3', '', regex=False).str.lower().str.strip()

    # 2.3 Ghép 2 bảng lại với nhau
    df_merged = pd.merge(df_main, df_lyrics[['clean_title', 'lyric']], on='clean_title', how='inner')
    print(f"✅ Đã ghép thành công: {df_merged.shape[0]} bài hát (Audio + Lyrics).")

except Exception as e:
    print(f"❌ Lỗi load file: {e}")
    print("👉 Hãy kiểm tra lại đường dẫn file csv.")
    exit()

# =============================================================================
# 3. CHUẨN BỊ FEATURES (TIỀN XỬ LÝ)
# =============================================================================

# A. Hàm làm sạch Text
def preprocess_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    # Chỉ giữ lại chữ cái tiếng Việt và khoảng trắng
    text = re.sub(r'[^a-záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Áp dụng làm sạch lời bài hát
df_merged['clean_lyric'] = df_merged['lyric'].apply(preprocess_text)

# B. Chọn cột Audio (Loại bỏ các cột "ăn gian" và cột thời gian)
df_numeric = df_merged.select_dtypes(include=[np.number])
target_col = 'is_hit'

cols_to_drop = [
    target_col, 
    'spotify_popularity', 'total_plays', 'spotify_streams', # Leakage (Biết trước kết quả)
    'release_year', 'days_since_release',                   # Time Bias (Năm tháng gây thiên kiến)
    'Unnamed: 0', 
    'sentiment_score', 'sentiment_confidence'               # Đã có trong text raw nên bỏ qua số liệu cũ
]

# Lọc lấy danh sách các cột Audio hợp lệ
audio_features = [c for c in df_numeric.columns if c not in cols_to_drop]

print(f"📊 Số lượng Audio Features sử dụng: {len(audio_features)}")

# Tách Input (X) và Output (y)
# X bao gồm cả cột số (Audio) và cột chữ (Lyrics)
X = df_merged[audio_features + ['clean_lyric']]
y = df_merged[target_col]

# Chia tập Train/Test (80% / 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# =============================================================================
# 4. XÂY DỰNG PIPELINE XỬ LÝ
# =============================================================================

# --- Preprocessor Chung ---
# Đây là bộ xử lý trung tâm, sẽ tự động chia dữ liệu thành 2 nhánh
preprocessor = ColumnTransformer(
    transformers=[
        # Nhánh 1: Xử lý số (Audio) -> Điền giá trị thiếu -> Chuẩn hóa (Scale)
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), audio_features),
        
        # Nhánh 2: Xử lý chữ (Lyrics) -> TF-IDF -> Vector hóa
        # Chỉ lấy 50 từ khóa quan trọng nhất để tránh làm loãng dữ liệu
        ('txt', TfidfVectorizer(stop_words=vn_stopwords, max_features=50, ngram_range=(1, 1)), 'clean_lyric')
    ])

# --- Danh sách đầy đủ 7 thuật toán ---
models_config = {
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "XGBoost": xgb.XGBClassifier(eval_metric='logloss', random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM (Support Vector Machine)": SVC(probability=True, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "K-Nearest Neighbors (KNN)": KNeighborsClassifier(n_neighbors=5),
    "Neural Network (MLP)": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=1000, random_state=42)
}

trained_models = {}

# =============================================================================
# 5. HUẤN LUYỆN VÒNG LẶP & ĐÁNH GIÁ
# =============================================================================
print("\n🚀 BẮT ĐẦU HUẤN LUYỆN 7 MÔ HÌNH (HYBRID AUDIO + LYRICS)...")

for name, clf in models_config.items():
    print(f"   Training {name}...")
    try:
        # Tạo Pipeline riêng cho từng model: (Xử lý dữ liệu) -> (Thuật toán dự đoán)
        model_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', clf)
        ])
        
        # Huấn luyện
        model_pipeline.fit(X_train, y_train)
        
        # Đánh giá nhanh độ chính xác
        y_pred = model_pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"     -> Độ chính xác (Accuracy): {acc:.4f}")
        
        # Lưu pipeline đã train vào dictionary
        trained_models[name] = model_pipeline
        
    except Exception as e:
        print(f"     ❌ Lỗi khi train {name}: {e}")

# =============================================================================
# 6. ĐÓNG GÓI & LƯU FILE KẾT QUẢ
# =============================================================================

# Cấu trúc file dictionary chuẩn cho Web App
final_data_package = {
    "models": trained_models,        # Chứa tất cả 7 model đã train
    "audio_features": audio_features # Danh sách cột Audio cần trích xuất (Metadata)
}

output_filename = 'DA\pkl_file\multi_model_hit_prediction.pkl'
print(f"\n💾 Đang lưu tất cả vào file: {output_filename} ...")
joblib.dump(final_data_package, output_filename)
print("✅ HOÀN TẤT! Hệ thống đã sẵn sàng để triển khai lên Web App.")