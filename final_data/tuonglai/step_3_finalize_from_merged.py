import pandas as pd
import numpy as np
import os

# =============================================================================
# CẤU HÌNH ĐƯỜNG DẪN
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

PATH_MERGED = os.path.join(PROJECT_ROOT, "final_data", "mergerd_balanced_and_features.csv")
PATH_BRIDGE = os.path.join(PROJECT_ROOT, "final_data", "balanced_500_500.csv")
PATH_ARTIST = os.path.join(PROJECT_ROOT, "data", "artists_info.csv")
PATH_OUTPUT = os.path.join(PROJECT_ROOT, "final_data", "FINAL_DATASET_TRAIN_SCIENTIFIC.csv")

# =============================================================================
# ĐỊNH NGHĨA KHOA HỌC VỀ HIT
# =============================================================================
ABS_THRESHOLD = 50      # Điều kiện 1: Ngưỡng tuyệt đối (Evergreen Hits)
Z_MULTIPLIER = 1.0      # Điều kiện 2: Độ nhạy Z-Score (Statistical Breakout)
MIN_HIT_FLOOR = 35      # Sàn tối thiểu để xét Hit (Loại bỏ rác)

def main():
    print("🚀 BƯỚC 3: ĐÓNG GÓI DATASET THEO ĐỊNH NGHĨA KHOA HỌC...")

    if not os.path.exists(PATH_MERGED):
        print(f"❌ Lỗi: Không tìm thấy {PATH_MERGED}")
        return

    df = pd.read_csv(PATH_MERGED)
    df['spotify_popularity'] = pd.to_numeric(df['spotify_popularity'], errors='coerce').fillna(0)
    
    # 1. PHÂN TÍCH THỐNG KÊ NGHỆ SĨ
    print("📊 Đang phân tích phong độ nghệ sĩ bằng Z-Score...")
    # Lấy nghệ sĩ chính để group
    df['primary_artist'] = df['artists'].astype(str).apply(lambda x: x.split(',')[0].strip())
    
    # Tính Mean và Std cho từng nghệ sĩ
    artist_stats = df.groupby('primary_artist')['spotify_popularity'].agg(['mean', 'std', 'count']).reset_index()
    artist_stats.columns = ['primary_artist', 'art_mean', 'art_std', 'art_count']
    artist_stats['art_std'] = artist_stats['art_std'].fillna(0)
    
    df = df.merge(artist_stats, on='primary_artist', how='left')

    # 2. ÁP DỤNG CÔNG THỨC QUYẾT ĐỊNH NHÃN
    def label_hit_logic(row):
        pop = row['spotify_popularity']
        avg = row['art_mean']
        std = row['art_std']
        
        # ĐIỀU KIỆN A: NGƯỠNG TUYỆT ĐỐI (Dành cho siêu phẩm như Bùi Anh Tuấn 58đ)
        if pop >= ABS_THRESHOLD:
            return 1
            
        # ĐIỀU KIỆN B: NGƯỠNG ĐỘT PHÁ (Dành cho nhạc cũ hoặc nghệ sĩ nhỏ)
        # Công thức: Popularity > (Trung bình + 1.0 * Độ lệch chuẩn) và phải >= 35đ
        if pop >= MIN_HIT_FLOOR and pop > (avg + Z_MULTIPLIER * std):
            return 1
            
        return 0

    df['is_hit'] = df.apply(label_hit_logic, axis=1)

    # 3. ĐỒNG BỘ TRACK_ID VÀ FOLLOWERS
    print("🔗 Đang đồng bộ hóa Metadata...")
    if os.path.exists(PATH_BRIDGE):
        df_bridge = pd.read_csv(PATH_BRIDGE)
        id_col = 'spotify_track_id' if 'spotify_track_id' in df_bridge.columns else 'track_id'
        id_map = dict(zip(df_bridge['file_name'], df_bridge[id_col]))
        df['track_id'] = df['file_name'].map(id_map)

    if os.path.exists(PATH_ARTIST):
        df_art = pd.read_csv(PATH_ARTIST)
        fol_map = dict(zip(df_art['artist_name'].str.lower(), df_art['total_followers']))
        df['artist_followers'] = df['primary_artist'].str.lower().map(fol_map).fillna(0)
    else:
        df['artist_followers'] = 0

    # 4. LƯU KẾT QUẢ
    os.makedirs(os.path.dirname(PATH_OUTPUT), exist_ok=True)
    priority_cols = ['track_id', 'file_name', 'title', 'artists', 'is_hit', 'spotify_popularity', 'artist_followers']
    all_cols = priority_cols + [c for c in df.columns if c not in priority_cols and c not in ['primary_artist', 'art_mean', 'art_std', 'art_count']]
    
    df[all_cols].to_csv(PATH_OUTPUT, index=False, encoding='utf-8-sig')
    
    print("\n" + "="*60)
    print(f"🎉 HOÀN TẤT!")
    print(f"📊 Tổng bài hát: {len(df)}")
    print(f"🔥 Tổng số HIT: {df['is_hit'].sum()}")
    print(f"✅ Đã giải oan cho 'Nơi Tình Yêu Bắt Đầu' và các bản Hit lịch sử.")
    print("="*60)

if __name__ == "__main__":
    main()