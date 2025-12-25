import pandas as pd
import numpy as np
import torch
import os
from transformers import AutoModel, AutoTokenizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import f1_score, classification_report
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# 1. LOAD VÀ GỘP DỮ LIỆU (AUDIO + LYRIC)
# =============================================================================
PATH_AUDIO = r'final_data\mergerd_balanced_and_features.csv'
PATH_LYRIC = r'Audio_lyric\nlp_analysis.csv'

print("📂 Đang tải và gộp dữ liệu...")
df_audio = pd.read_csv(PATH_AUDIO)
df_lyric = pd.read_csv(PATH_LYRIC)

# Chuẩn hóa tên file để merge chính xác
df_audio['file_name_clean'] = df_audio['file_name'].str.strip().str.lower()
df_lyric['file_name_clean'] = df_lyric['file_name'].str.strip().str.lower()

# Merge: Lấy Audio làm gốc, gộp cột 'lyric' từ file NLP vào
df = pd.merge(df_audio, df_lyric[['file_name_clean', 'lyric']], on='file_name_clean', how='inner')

# =============================================================================
# 2. LỌC NHÃN (CHỈ GIỮ 4 DÒNG NHẠC MỤC TIÊU - LOẠI BOLERO)
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

df['genres_list'] = df['spotify_genres'].apply(extract_all_genres)
df = df[df['genres_list'].apply(len) > 0] # Loại bỏ Bolero/None
df = df[df['lyric'].notna()] # Loại bỏ bài thiếu lời

print(f"✅ Dữ liệu sau khi gộp và lọc: {len(df)} bài hát.")

# =============================================================================
# 3. TRÍCH XUẤT ĐẶC TRƯNG PHOBERT (768 DIM)
# =============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
phobert = AutoModel.from_pretrained("vinai/phobert-base-v2").to(DEVICE)

def get_embeddings(text_list, batch_size=8):
    phobert.eval()
    all_embeddings = []
    for i in tqdm(range(0, len(text_list), batch_size), desc="🔤 PhoBERT Embedding"):
        batch = [str(t)[:512] for t in text_list[i:i+batch_size]]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = phobert(**inputs)
            all_embeddings.append(outputs.last_hidden_state[:, 0, :].cpu().numpy())
    return np.vstack(all_embeddings)

print("🚀 Đang chạy PhoBERT...")
X_text_768 = get_embeddings(df['lyric'].tolist())

# =============================================================================
# 4. GIẢM CHIỀU PCA (FIX LỆCH DATA: 768 -> 64)
# =============================================================================
# 
print("📉 Đang nén 768 chiều văn bản xuống 64 chiều...")
pca = PCA(n_components=64, random_state=42)
X_text_64 = pca.fit_transform(X_text_768)

# Tách đặc trưng âm thanh (loại bỏ các cột metadata)
cols_to_drop = ['file_name', 'title', 'artists', 'lyric', 'spotify_genres', 'genres_list', 'file_name_clean']
X_audio = df.drop(columns=[c for c in cols_to_drop if c in df.columns]).values

# Hợp nhất: Audio (~62-168) + Text (64)
# 
X_final = np.hstack((X_audio, X_text_64))
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df['genres_list'])

# =============================================================================
# 5. TRAIN VOTING CLASSIFIER (TASK 4)
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)

voting_clf = MultiOutputClassifier(VotingClassifier(
    estimators=[
        ('gb', GradientBoostingClassifier(n_estimators=100)),
        ('mlp', MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500)),
        ('et', ExtraTreesClassifier(n_estimators=200, class_weight='balanced'))
    ],
    voting='soft', weights=[2, 1, 1]
))

print("🏗️ Đang huấn luyện mô hình đa phương thức...")
voting_clf.fit(X_train, y_train)

# Kết quả
y_pred = voting_clf.predict(X_test)
print("\n" + "="*50)
print(f"🎯 F1-Micro Score (PhoBERT + PCA): {f1_score(y_test, y_pred, average='micro'):.4f}")
print("="*50)
print(classification_report(y_test, y_pred, target_names=mlb.classes_))