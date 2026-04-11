import pandas as pd

# 1. TẢI DỮ LIỆU
print("⏳ Đang nạp dữ liệu từ 3 file...")
df_songs = pd.read_csv('data/songs.csv', low_memory=False)
df_lib = pd.read_csv('data/librosa_analysis_v2.csv', low_memory=False)
df_nlp = pd.read_csv('data/nlp_metrics_no_lyric.csv', low_memory=False)

print(f"Tổng số bài hát gốc ban đầu: {len(df_songs)} bài.")

# 2. XÓA TRÙNG LẶP VÀ LỌC CỘT NLP
# Đảm bảo ID là duy nhất
df_songs = df_songs.drop_duplicates(subset='spotify_track_id')
df_lib = df_lib.drop_duplicates(subset='spotify_track_id')
df_nlp = df_nlp.drop_duplicates(subset='spotify_track_id')

# Loại bỏ các cột NLP không cần thiết
nlp_cols_to_drop = ['sent_lexicon', 'score_lexicon', 'sent_phobert', 'conf_phobert']
df_nlp.drop(columns=nlp_cols_to_drop, inplace=True, errors='ignore')

# 3. INNER MERGE (Chỉ ghép những bài TỒN TẠI TRONG CẢ 3 FILE)
df_merged = pd.merge(df_songs, df_lib, on='spotify_track_id', how='inner', suffixes=('', '_lib'))
df_merged = pd.merge(df_merged, df_nlp, on='spotify_track_id', how='inner', suffixes=('', '_nlp'))

# Dọn dẹp các cột tên bị trùng lặp do Merge
cols_to_drop = [col for col in df_merged.columns if col.endswith('_lib') or col.endswith('_nlp')]
df_merged.drop(columns=cols_to_drop, inplace=True)

# 4. LỌC THẲNG TAY CÁC BÀI BỊ KHUYẾT DATA (Ngoại trừ featured_artists)
cols_to_check_na = [col for col in df_merged.columns if col != 'featured_artists']

# Lệnh dropna sẽ xóa sổ bất kỳ bài hát nào có chứa NaN ở các cột bắt buộc
df_clean = df_merged.dropna(subset=cols_to_check_na)

# 5. LƯU RA FILE CHÍNH THỨC
FINAL_FILE = 'data/merged_inner_data_final.csv'
df_clean.to_csv(FINAL_FILE, index=False, encoding='utf-8-sig')

print("-" * 50)
print(f"✅ ĐÃ LỌC VÀ MERGE THÀNH CÔNG CỰC MẠNH!")
print(f"🎯 Số bài hát HOÀN HẢO (full 100% data): {len(df_clean)} bài.")
print(f"🧹 Đã loại bỏ hoàn toàn các bài bị khuyết dữ liệu (Librosa, NLP).")
print(f"📁 Dữ liệu siêu sạch được lưu tại: {FINAL_FILE}")
print("-" * 50)