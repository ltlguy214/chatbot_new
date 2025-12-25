'''
Docstring for DA.task4_phobert_pca

Task 4: Phân loại đa nhãn kết hợp Audio + PhoBERT + PCA.
'''
import pandas as pd
import numpy as np
import torch
import os
from transformers import AutoModel, AutoTokenizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import f1_score, classification_report, hamming_loss
from tqdm import tqdm
import warnings

# Tắt các cảnh báo không cần thiết
warnings.filterwarnings('ignore')

# =============================================================================
# 1. CẤU HÌNH ĐƯỜNG DẪN & THIẾT BỊ
# =============================================================================
PATH_AUDIO = r'final_data\mergerd_balanced_and_features.csv'
PATH_LYRIC = r'Audio_lyric\nlp_analysis.csv'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"🚀 Khởi động hệ thống (Sử dụng: {DEVICE})")

# =============================================================================
# 2. TẢI VÀ GỘP DỮ LIỆU
# =============================================================================
if not os.path.exists(PATH_AUDIO) or not os.path.exists(PATH_LYRIC):
    print("❌ Lỗi: Không tìm thấy file dữ liệu. Vui lòng kiểm tra lại đường dẫn!")
    exit()

print("📂 Đang tải dữ liệu và gộp file...")
df_audio = pd.read_csv(PATH_AUDIO)
df_lyric = pd.read_csv(PATH_LYRIC)

# Chuẩn hóa tên file để merge
df_audio['file_name_clean'] = df_audio['file_name'].str.strip().str.lower()
df_lyric['file_name_clean'] = df_lyric['file_name'].str.strip().str.lower()

# Merge lấy Audio làm gốc
df = pd.merge(df_audio, df_lyric[['file_name_clean', 'lyric']], on='file_name_clean', how='inner')

# =============================================================================
# 3. TIỀN XỬ LÝ NHÃN & LỌC DỮ LIỆU (TASK 4 LOGIC)
# =============================================================================
def extract_all_genres(genre_raw):
    try:
        cleaned = str(genre_raw).lower()
        mapped = []
        if 'vinahouse' in cleaned: mapped.append('Vinahouse')
        if 'hip hop' in cleaned or 'rap' in cleaned: mapped.append('Hip-Hop')
        if 'indie' in cleaned: mapped.append('Indie')
        if 'v-pop' in cleaned or 'pop' in cleaned: mapped.append('V-Pop')
        return mapped
    except: return []

print("🧹 Đang lọc nhãn nhiễu (Bolero/None)...")
df['genres_list'] = df['spotify_genres'].apply(extract_all_genres)
df = df[df['genres_list'].apply(len) > 0] # Giữ lại 994 bài sạch
df = df[df['lyric'].notna()] # Đảm bảo có lời nhạc

print(f"✅ Tổng dữ liệu sạch: {len(df)} bài hát.")

# =============================================================================
# 4. TRÍCH XUẤT ĐẶC TRƯNG PHOBERT (VĂN BẢN)
# =============================================================================
print("🔤 Đang khởi tạo PhoBERT v2...")
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
phobert = AutoModel.from_pretrained("vinai/phobert-base-v2").to(DEVICE)

def get_embeddings(text_list, batch_size=8):
    phobert.eval()
    all_embeddings = []
    for i in tqdm(range(0, len(text_list), batch_size), desc="✨ PhoBERT Embedding"):
        batch = [str(t)[:512] for t in text_list[i:i+batch_size]]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = phobert(**inputs)
            # Lấy vector đại diện [CLS]
            all_embeddings.append(outputs.last_hidden_state[:, 0, :].cpu().numpy())
    return np.vstack(all_embeddings)

X_text_768 = get_embeddings(df['lyric'].tolist())

# =============================================================================
# 5. NÉN ĐẶC TRƯNG PCA (FIX LỆCH DATA)
# =============================================================================
print("📉 Đang thực hiện PCA (Nén 768 chiều -> 64 chiều)...")
pca = PCA(n_components=64, random_state=42)
X_text_reduced = pca.fit_transform(X_text_768)

# Chuẩn bị đặc trưng âm thanh (X_audio)
cols_to_drop = [
        'file_name', 'title', 'artists', 'spotify_release_date', 'track_id', 'file_name_clean'
        'spotify_genres', 'genres_list', 'target_genre', 'primary_genre',
        'is_hit', 'hit', 'target', 'total_plays', 'spotify_streams', 'spotify_popularity',
        'lyric', 'clean_lyric', 'Unnamed: 0',
        'sentiment_confidence','sentiment', 'cluster', 'musical_key'
    ]
X_audio = df.drop(columns=[c for c in cols_to_drop if c in df.columns]).values

# Chuẩn hóa Audio Features trước khi gộp
scaler = StandardScaler()
X_audio_scaled = scaler.fit_transform(X_audio)

# Hợp nhất: Audio + Text đã nén
X_final = np.hstack((X_audio_scaled, X_text_reduced))

# Chuyển đổi nhãn sang dạng Multi-label
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df['genres_list'])

print(f"📊 Kích thước tập dữ liệu cuối cùng: {X_final.shape}")

# =============================================================================
# 6. HUẤN LUYỆN MÔ HÌNH VOTING CLASSIFIER (Hệ thống đa nhãn)
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)

# Định nghĩa các mô hình thành phần
clf1 = GradientBoostingClassifier(n_estimators=100, random_state=42)
clf2 = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42)
clf3 = ExtraTreesClassifier(n_estimators=200, class_weight='balanced', random_state=42)

# Kết hợp vào Voting Classifier với trọng số 2-1-1
ensemble = VotingClassifier(
    estimators=[('gb', clf1), ('mlp', clf2), ('et', clf3)],
    voting='soft', weights=[2, 1, 1]
)

# Wrapper để xử lý Đa Nhãn
model = MultiOutputClassifier(ensemble)

print("🏗️ Đang huấn luyện mô hình đa phương thức (Voting Classifier)...")
model.fit(X_train, y_train)

# =============================================================================
# 7. XUẤT BÁO CÁO KẾT QUẢ
# =============================================================================
y_pred = model.predict(X_test)

print("\n" + "="*60)
print("📊 BÁO CÁO HIỆU NĂNG TASK 4: PHOBERT + AUDIO + PCA")
print("="*60)
print(f"🔹 F1-Micro Score:  {f1_score(y_test, y_pred, average='micro'):.4f}")
print(f"🔹 Hamming Loss:    {hamming_loss(y_test, y_pred):.4f}")
print("-" * 60)
print("CHI TIẾT TỪNG DÒNG NHẠC:")
print(classification_report(y_test, y_pred, target_names=mlb.classes_))
print("="*60)