import pandas as pd
import numpy as np
import os

# =============================================================================
# CẤU HÌNH
# =============================================================================
# Đường dẫn file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data") # Giả sử folder data cùng cấp với script

INPUT_FILE = os.path.join(DATA_DIR, "raw_vpop_collection.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "vpop_dataset_labeled.csv")

# Ngưỡng (Thresholds) cho Luật HIT
THRESHOLD_NATIONAL_HIT = 65     # Điểm >= 65 là Siêu Hit (bất kể ai hát)
THRESHOLD_MIN_POPULARITY = 40   # Điểm sàn tối thiểu để được xét là Hit
THRESHOLD_RELATIVE_JUMP = 15    # Phải cao hơn trung bình của ca sĩ 15 điểm

# =============================================================================
# LOGIC TÍNH TOÁN
# =============================================================================

def calculate_hit_labels(df):
    print("🔄 Đang tính toán nhãn HIT theo quy tắc Hybrid...")
    
    # 1. Tính Popularity trung bình của từng Ca sĩ (Artist Average)
    # Group theo primary_artist_id để chính xác nhất
    artist_stats = df.groupby('primary_artist_id')['spotify_popularity'].agg(['mean', 'count']).reset_index()
    artist_stats.rename(columns={'mean': 'artist_avg_pop', 'count': 'track_count'}, inplace=True)
    
    # Merge lại vào bảng chính
    df = pd.merge(df, artist_stats, on='primary_artist_id', how='left')
    
    # 2. Áp dụng công thức HIT
    # Điều kiện 1: National Hit (Siêu Hit)
    cond_national = df['spotify_popularity'] >= THRESHOLD_NATIONAL_HIT
    
    # Điều kiện 2: Relative Hit (Hit đột phá)
    # Popularity >= Sàn (40)  VÀ  Popularity > (Trung bình + 15)
    cond_relative = (
        (df['spotify_popularity'] >= THRESHOLD_MIN_POPULARITY) & 
        (df['spotify_popularity'] > (df['artist_avg_pop'] + THRESHOLD_RELATIVE_JUMP))
    )
    
    # Kết hợp: Thỏa mãn 1 trong 2 là HIT
    df['is_hit'] = np.where(cond_national | cond_relative, 1, 0)
    
    return df

def analyze_results(df):
    """Phân tích thống kê bộ dữ liệu sau khi gán nhãn"""
    total = len(df)
    hits = df['is_hit'].sum()
    non_hits = total - hits
    
    print("\n📊 THỐNG KÊ BỘ DỮ LIỆU SAU KHI GÁN NHÃN:")
    print(f"   - Tổng số bài hát: {total}")
    print(f"   - Số lượng HIT (1): {hits} ({hits/total*100:.2f}%)")
    print(f"   - Số lượng NON-HIT (0): {non_hits} ({non_hits/total*100:.2f}%)")
    
    print("\n🔍 Ví dụ các bài được dán nhãn HIT:")
    hit_samples = df[df['is_hit'] == 1][['title', 'artists', 'spotify_popularity', 'artist_avg_pop']].head(10)
    print(hit_samples.to_string(index=False))
    
    print("\n🔍 Ví dụ các bài NON-HIT (Dù popularity > 0):")
    non_hit_samples = df[(df['is_hit'] == 0) & (df['spotify_popularity'] > 30)].head(5)
    print(non_hit_samples[['title', 'artists', 'spotify_popularity', 'artist_avg_pop']].to_string(index=False))

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Lỗi: Không tìm thấy file đầu vào: {INPUT_FILE}")
        print("   Hãy chạy master_data_collector.py trước!")
        return

    # Load dữ liệu
    print(f"📂 Đang đọc file: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)
    
    # Loại bỏ các bài trùng lặp (nếu có)
    df.drop_duplicates(subset=['track_id'], inplace=True)
    
    # Tính toán
    df_labeled = calculate_hit_labels(df)
    
    # Phân tích nhanh
    analyze_results(df_labeled)
    
    # Lưu file
    df_labeled.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print(f"\n✅ ĐÃ XONG! File dán nhãn đã lưu tại: {OUTPUT_FILE}")
    print("   👉 Dùng file này để làm đầu vào cho Librosa (trích xuất Audio).")

if __name__ == "__main__":
    main()