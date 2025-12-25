import pandas as pd
import numpy as np
import os

# =============================================================================
# CẤU HÌNH ĐƯỜNG DẪN
# =============================================================================
# Lưu ý: Chạy script từ thư mục gốc của dự án (D:\Hit_songs_DA)

AUDIO_DIR = "Audio_lyric"
DATA_DIR = "data"
FINAL_DIR = "final_data"

# File đầu vào
LIBROSA_FILE = os.path.join(AUDIO_DIR, "librosa_analysis.csv")
NLP_FILE = os.path.join(AUDIO_DIR, "nlp_analysis.csv")

# File chứa thông tin Metadata (Popularity, Title...)
METADATA_FILE = os.path.join(FINAL_DIR, "balanced_500_500.csv") 

# File chứa Followers
ARTIST_INFO_FILE = os.path.join(DATA_DIR, "artists_info.csv")

# File đầu ra
OUTPUT_FILE = os.path.join(FINAL_DIR, "FINAL_DATASET_TRAIN.csv")

# Cấu hình HIT
THRESHOLD_NATIONAL_HIT = 65
THRESHOLD_MIN_POPULARITY = 40
THRESHOLD_RELATIVE_JUMP = 15

def main():
    print("🚀 BƯỚC 2: ĐÓNG GÓI DỮ LIỆU (CHẾ ĐỘ OFFLINE)...")

    # 1. Đọc file Librosa
    if not os.path.exists(LIBROSA_FILE):
        print(f"❌ Thiếu file Librosa tại: {LIBROSA_FILE}")
        return
    df_lib = pd.read_csv(LIBROSA_FILE)
    print(f"📂 Đã đọc Librosa: {len(df_lib)} dòng.")

    # 2. Đọc file Metadata (Balanced)
    if not os.path.exists(METADATA_FILE):
        print(f"❌ Thiếu file Balanced tại: {METADATA_FILE}")
        return
    df_meta = pd.read_csv(METADATA_FILE)
    print(f"📂 Đã đọc Metadata: {len(df_meta)} dòng.")
    
    # Chuẩn hóa tên cột track_id
    if 'spotify_track_id' in df_meta.columns:
        df_meta = df_meta.rename(columns={'spotify_track_id': 'track_id'})

    # 3. Đọc Artist Info để lấy Followers
    artist_dict = {}
    if os.path.exists(ARTIST_INFO_FILE):
        df_art = pd.read_csv(ARTIST_INFO_FILE)
        # Map: Artist Name -> Followers
        for _, row in df_art.iterrows():
            a_name = str(row.get('artist_name', '')).lower().strip()
            # Ưu tiên lấy theo tên nếu không có ID khớp
            artist_dict[a_name] = row.get('total_followers', 0)

    # 4. GHÉP DỮ LIỆU (Merge theo file_name)
    print("🔗 Đang ghép dữ liệu...")
    
    # Chỉ lấy các cột metadata cần thiết
    meta_cols = ['file_name', 'track_id', 'title', 'artists', 'spotify_popularity', 'spotify_release_date']
    # Lọc những cột thực tế có trong file
    meta_cols = [c for c in meta_cols if c in df_meta.columns]
    
    # Merge: Librosa + Metadata
    df_final = pd.merge(df_lib, df_meta[meta_cols], on='file_name', how='inner')
    
    # 5. Bổ sung Followers (offline)
    print("➕ Đang bổ sung Followers...")
    def get_followers(row):
        # Lấy tên ca sĩ đầu tiên
        a_name = str(row.get('artists', '')).lower().split(',')[0].strip()
        return artist_dict.get(a_name, 0)
    
    df_final['artist_followers'] = df_final.apply(get_followers, axis=1)

    # 6. Merge NLP (Sentiment) nếu có
    if os.path.exists(NLP_FILE):
        print("🔗 Đang ghép NLP...")
        df_nlp = pd.read_csv(NLP_FILE)
        nlp_cols = ['file_name', 'sentiment', 'sentiment_score', 'lyric_total_words']
        nlp_cols = [c for c in nlp_cols if c in df_nlp.columns]
        # Merge
        df_final = pd.merge(df_final, df_nlp[nlp_cols], on='file_name', how='left')

    # 7. Tính nhãn HIT
    print("⚖️ Đang tính nhãn HIT...")
    # Tính popularity trung bình của ca sĩ trong tập dữ liệu này
    artist_avg = df_final.groupby('artists')['spotify_popularity'].transform('mean')
    
    cond_national = df_final['spotify_popularity'] >= THRESHOLD_NATIONAL_HIT
    cond_relative = (
        (df_final['spotify_popularity'] >= THRESHOLD_MIN_POPULARITY) & 
        (df_final['spotify_popularity'] > (artist_avg + THRESHOLD_RELATIVE_JUMP))
    )
    df_final['is_hit'] = np.where(cond_national | cond_relative, 1, 0)

    # 8. Lưu file
    os.makedirs(FINAL_DIR, exist_ok=True)
    
    # Sắp xếp cột ưu tiên lên đầu
    cols = list(df_final.columns)
    priority = ['track_id', 'title', 'artists', 'is_hit', 'spotify_popularity', 'artist_followers', 'file_name']
    final_cols = priority + [c for c in cols if c not in priority]
    
    df_final[final_cols].to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    
    print("\n" + "="*60)
    print(f"🎉 HOÀN TẤT! File chuẩn: {OUTPUT_FILE}")
    print(f"📊 Tổng số bài: {len(df_final)}")
    print(f"🔥 Số lượng HIT: {df_final['is_hit'].sum()}")
    print("="*60)

if __name__ == "__main__":
    main()