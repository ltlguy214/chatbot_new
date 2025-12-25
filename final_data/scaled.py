import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import datetime

# ==============================================================================
# 1. KHU VỰC CẤU HÌNH
# ==============================================================================

INPUT_FILE        = 'final_data\\mergerd_balanced_and_features.csv'
OUTPUT_FILE_CHECK = 'final_data/merged_balanced_500_500_scaled_check.csv'
OUTPUT_FILE_ML    = 'final_data/merged_balanced_500_500_scaled_ml.csv'

GENRE_LIST = [
    'v-pop', 'vinahouse', 'vietnamese hip hop', 'vietnamese lo-fi',
    'vietnam indie', 'bolero', 'hip hop', 'indie', 'lo-fi', 'unknown'
]

# Nhóm Meta (Sẽ đứng đầu tiên, KHÔNG Scale)
META_COLS_BASE = [
    'title', 'artists', 'filename', 'lyrics', 'sentiment_label', # Label chữ thì không scale
    'is_hit',                                 # Target
    'spotify_popularity', 'total_plays', 'spotify_streams'
]

# Các cột cần loại bỏ khi xuất file ML (Model chỉ cần số)
COLS_TO_DROP_FOR_TRAINING = ['title', 'artists', 'filename', 'lyrics', 'sentiment_label']

# ==============================================================================
# 2. XỬ LÝ DỮ LIỆU
# ==============================================================================

def process_pipeline():
    # --- Bước 1: Đọc dữ liệu ---
    print(f"🔄 Đang đọc file: {INPUT_FILE}...")
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print("❌ Lỗi: Không tìm thấy file đầu vào!")
        return

    # --- Bước 2: Xử lý Ngày tháng (Date Engineering) ---
    if 'spotify_release_date' in df.columns:
        print("🛠  Đang xử lý ngày tháng...")
        df['release_date_parsed'] = pd.to_datetime(df['spotify_release_date'], errors='coerce')
        
        today = pd.Timestamp.now()
        df['days_since_release'] = (today - df['release_date_parsed']).dt.days
        df['release_year'] = df['release_date_parsed'].dt.year
        
        # Fill NaN
        df['days_since_release'] = df['days_since_release'].fillna(df['days_since_release'].median())
        df['release_year'] = df['release_year'].fillna(df['release_year'].median())
        
        df = df.drop(columns=['spotify_release_date', 'release_date_parsed'])

    # --- Bước 3: One-Hot Encoding Genre ---
    if 'spotify_genres' in df.columns:
        print("🛠  Đang xử lý One-Hot Encoding Genre...")
        for genre in GENRE_LIST:
            col_name = f'genre_{genre.replace(" ", "_")}'
            df[col_name] = df['spotify_genres'].fillna('').astype(str).apply(
                lambda x: 1 if genre in x.lower() else 0
            )
        df = df.drop(columns=['spotify_genres'])

    # --- Bước 4: Scaling (Scale cả Sentiment Score và Date) ---
    print("⚖️  Đang thực hiện Scaling...")
    
    # Xác định các cột Genre (Không scale)
    genre_cols = [c for c in df.columns if c.startswith('genre_')]
    
    # Những cột KHÔNG Scale: Meta + Genre
    exclude_cols = META_COLS_BASE + genre_cols
    
    # Tự động tìm tất cả cột số còn lại để Scale (Bao gồm Audio, Date, Sentiment Score)
    num_cols_to_scale = [
        col for col in df.columns 
        if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])
    ]
    
    print(f"👉 Các cột được Scale ({len(num_cols_to_scale)}): {num_cols_to_scale}")
    # Bạn sẽ thấy: sentiment_score, days_since_release, release_year, mfcc...
    
    scaler = StandardScaler()
    df_scaled = df.copy()
    
    if num_cols_to_scale:
        df_scaled[num_cols_to_scale] = scaler.fit_transform(df[num_cols_to_scale])

    # --- BƯỚC 5: OHE sentiment (sau scaling, trước khi sắp xếp cột) ---
    if 'sentiment' in df_scaled.columns:
        for val in ['neutral', 'positive', 'negative']:
            col_name = f'sentiment_{val}'
            df_scaled[col_name] = (df_scaled['sentiment'] == val).astype(int)
        df_scaled = df_scaled.drop(columns=['sentiment'])

    # --- SẮP XẾP LẠI THỨ TỰ CỘT (QUAN TRỌNG) ---
    print("🔀 Đang sắp xếp lại thứ tự cột...")
    # 1. Nhóm Meta
    group_meta = [c for c in META_COLS_BASE if c in df_scaled.columns]
    # 2. Nhóm Date
    group_date = [c for c in ['days_since_release', 'release_year'] if c in df_scaled.columns]
    # 3. Nhóm Sentiment (ĐƯA LÊN TRƯỚC GENRE)
    group_sentiment = [
        c for c in df_scaled.columns 
        if ('sentiment' in c or 'sent_' in c) and c not in group_meta
    ]
    # 4. Nhóm Genre OHE
    group_genre = [c for c in df_scaled.columns if c.startswith('genre_')]
    # 5. Nhóm Audio Features (Phần còn lại)
    excluded_for_audio = set(group_meta + group_date + group_sentiment + group_genre)
    group_audio = [c for c in df_scaled.columns if c not in excluded_for_audio]
    # Gộp lại theo thứ tự: Meta -> Date -> Sentiment -> Genre -> Audio
    new_order = group_meta + group_date + group_sentiment + group_genre + group_audio
    df_final = df_scaled[new_order]

    # --- Bước 6: Xuất file ---
    # 6.1 File Check
    df_final.to_csv(OUTPUT_FILE_CHECK, index=False, encoding='utf-8-sig')
    print(f"✅ Đã lưu file Check: {OUTPUT_FILE_CHECK}")

    # 6.2 File ML
    actual_drop = [c for c in COLS_TO_DROP_FOR_TRAINING if c in df_final.columns]
    df_ml = df_final.drop(columns=actual_drop)
    
    df_ml.to_csv(OUTPUT_FILE_ML, index=False, encoding='utf-8')
    print(f"✅ Đã lưu file ML: {OUTPUT_FILE_ML}")

if __name__ == "__main__":
    process_pipeline()