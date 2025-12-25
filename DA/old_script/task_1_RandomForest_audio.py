'''
Docstring for DA.task_1_RandomForest_audio
Random Forest ✅
audio, lyrics ✅

Đã ghép thành công: 974 bài hát (fix lấy file_name để đủ 1000)
ĐỘ CHÍNH XÁC SAU KHI LỌC TỪ RÁC: 0.5385
              precision    recall  f1-score   support

         0.0       0.54      0.60      0.57        98
         1.0       0.54      0.47      0.51        97

    accuracy                           0.54       195
   macro avg       0.54      0.54      0.54       195
weighted avg       0.54      0.54      0.54       195


💡 TOP 20 YẾU TỐ (Đã sạch sẽ hơn):
                         Feature  Importance
28                    rms_energy    0.018240
12                     mfcc2_std    0.016467
63             lexical_diversity    0.015346
46                  chroma4_mean    0.014911
7                    mfcc13_mean    0.014895
5                    mfcc12_mean    0.014781
13                    mfcc3_mean    0.013937
62             lyric_total_words    0.013896
26                     mfcc9_std    0.013821
27                  duration_sec    0.013592
31  spectral_contrast_band1_mean    0.013426
52                 chroma10_mean    0.013385
93                      word_yêu    0.013270
3                    mfcc11_mean    0.012940
64                    noun_count    0.012901
60                 tonnetz6_mean    0.012668
17                    mfcc5_mean    0.012557
51                  chroma9_mean    0.012505
32  spectral_contrast_band2_mean    0.012280
10                     mfcc1_std    0.012242
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# =============================================================================
# 1. DANH SÁCH TỪ VÔ NGHĨA (STOPWORDS) TIẾNG VIỆT
# =============================================================================
# Đây là những từ xuất hiện quá nhiều nhưng ít ý nghĩa phân loại
vn_stopwords = [
    "anh", "em", "tôi", "ta", "mình", "nó", "chúng_ta", "bạn", "người", 
    "là", "của", "và", "những", "các", "trong", "đã", "đang", "vẫn", 
    "thì", "mà", "có", "một", "được", "tại", "vì", "nên", "hay", "hoặc",
    "cứ", "thôi", "biết", "chưa", "không", "chẳng", "rồi", "lắm", "quá",
    "đi", "ơi", "à", "nhé", "nha", "đây", "nào", "sẽ", "muốn", "phải",
    "từng", "lên", "xuống", "ra", "vào", "để", "lại", "gì", "này", "đó", "đâu"
]

# =============================================================================
# 2. LOAD VÀ GHÉP DỮ LIỆU
# =============================================================================
print("🔄 Đang load và ghép dữ liệu...")

try:
    df_main = pd.read_csv(r'final_data\merged_balanced_500_500.csv')
    df_main['clean_title'] = df_main['title'].astype(str).str.lower().str.strip()
except:
    print("❌ Không tìm thấy file data chính")
    exit()

try:
    # SỬA LỖI ĐƯỜNG DẪN: Thêm r''
    df_lyrics = pd.read_csv(r'Audio_lyric\lyrics_extracted.csv')
    df_lyrics['clean_title'] = df_lyrics['file_name'].astype(str).str.replace('.mp3', '', regex=False).str.lower().str.strip()
except:
    print("❌ Không tìm thấy file lyrics")
    exit()

# Ghép dữ liệu
df_merged = pd.merge(df_main, df_lyrics[['clean_title', 'lyric']], on='clean_title', how='inner')
print(f"✅ Đã ghép thành công: {df_merged.shape[0]} bài hát")

# =============================================================================
# 3. CHUẨN BỊ FEATURES
# =============================================================================

# Xử lý text
def preprocess_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'[^a-záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df_merged['clean_lyric'] = df_merged['lyric'].apply(preprocess_text)

# Lọc cột số (Audio) - Loại bỏ Leakage & Time Bias
df_numeric = df_merged.select_dtypes(include=[np.number])
target_col = 'is_hit'
cols_to_drop = [
    target_col, 
    'spotify_popularity', 'total_plays', 'spotify_streams', # Leakage
    'release_year', 'days_since_release', 'Unnamed: 0',     # Time Bias
    'sentiment_score', 'sentiment_confidence' 
]
audio_features = [c for c in df_numeric.columns if c not in cols_to_drop]

# Tách X và y
X = df_merged[audio_features + ['clean_lyric']]
y = df_merged[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# =============================================================================
# 4. PIPELINE (ĐÃ CẬP NHẬT STOPWORDS)
# =============================================================================

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# CẬP NHẬT: Thêm stop_words và giảm max_features xuống 30 để tránh nhiễu
text_transformer = TfidfVectorizer(
    stop_words=vn_stopwords,  # <--- QUAN TRỌNG: Loại bỏ từ rác
    max_features=30,          # Chỉ lấy 30 từ "đắt" nhất
    ngram_range=(1, 1)
)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, audio_features),
        ('txt', text_transformer, 'clean_lyric')
    ])

# Dùng Random Forest (Tăng n_estimators lên 300 để học kỹ hơn)
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=300, random_state=42))
])

# =============================================================================
# 5. HUẤN LUYỆN & KẾT QUẢ
# =============================================================================
print("\n🚀 Đang huấn luyện (Đã lọc Stopwords)...")
model_pipeline.fit(X_train, y_train)

y_pred = model_pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\n🏆 ĐỘ CHÍNH XÁC SAU KHI LỌC TỪ RÁC: {acc:.4f}")
print(classification_report(y_test, y_pred))

# Vẽ biểu đồ Feature Importance mới
try:
    num_names = audio_features
    txt_names = model_pipeline.named_steps['preprocessor'].transformers_[1][1].get_feature_names_out()
    all_names = num_names + ['word_' + t for t in txt_names]
    
    importances = model_pipeline.named_steps['classifier'].feature_importances_
    
    fi_df = pd.DataFrame({'Feature': all_names, 'Importance': importances})
    fi_df = fi_df.sort_values(by='Importance', ascending=False).head(20)
    
    print("\n💡 TOP 20 YẾU TỐ (Đã sạch sẽ hơn):")
    print(fi_df)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=fi_df, palette='viridis', hue='Feature', legend=False)
    plt.title('Top Features: Audio vs Meaningful Words')
    plt.tight_layout()
    plt.show()

except Exception as e:
    print(e)