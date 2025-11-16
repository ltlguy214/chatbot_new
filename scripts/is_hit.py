import pandas as pd
import numpy as np
import sys
import io
from pathlib import Path
import re

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# =========================================================================
# --- 1. HẰNG SỐ VỀ FILE VÀ NỀN TẢNG ---
# =========================================================================
MASTER_SONG_LIST = 'data/song_list_info.csv'
OUTPUT_FILE = 'data/is_hit.csv'

PLATFORM_FILES = {
    'APPLE_MUSIC': 'data/apple_music_top100_kworb_vn.csv',
    'SPOTIFY': 'data/spotify_top100_kworb_vn.csv',
    'NCT': 'data/nct_top50.csv',
    'ZINGMP3': 'data/zingmp3_top100.csv'
}
PLATFORM_NAMES = list(PLATFORM_FILES.keys()) # ['APPLE_MUSIC', 'SPOTIFY', 'NCT', 'ZINGMP3']

# =========================================================================
# --- 2. HẰNG SỐ VỀ THÔNG SỐ TÍNH ĐIỂM ---
# =========================================================================
# Ngưỡng lọc chung
RANK_FILTER_TOP_N = 50

# Cửa 1: Ngưỡng điểm chính (Hệ thống Max 17)
FINAL_HIT_THRESHOLD = 9

# Cửa 2: Ngưỡng phao cứu sinh (Spotify Pop)
POPULARITY_LIFELINE = 49

# Trụ cột 1: Điểm Rank (3-2-1)
RANK_SCORE_TIERS = {
    10: 3,  # Top 1-10 -> 3 điểm
    20: 2,  # Top 11-20 -> 2 điểm
    40: 1   # Top 21-40 -> 1 điểm
}

# Trụ cột 2: Điểm Đa nền tảng
PLATFORM_COUNT_THRESHOLD = 3 # Yêu cầu >= 3 nền tảng

# Trụ cột 3: Điểm Bền bỉ
SUSTAIN_APPEARANCES_THRESHOLD = 30 # Yêu cầu >= 30 lần xuất hiện

# Trụ cột 4: Điểm Lịch sử
HISTORICAL_THRESHOLDS = {
    'SPOTIFY_total_streams': 10000000,
    'ZINGMP3_total_plays': 1000000,
    'NCT_total_likes': 30000
}

# =========================================================================
# --- 3. HẰNG SỐ VỀ ĐỊNH DẠNG OUTPUT ---
# =========================================================================
# 3.1. Đổi tên cột (Rename)
RENAME_MAP = {
    'title': 'title',
    'artists': 'artists',
    'featured_artists': 'featured_artists',
    'spotify_release_date': 'release_date',
    'spotify_genres': 'genres',
    'spotify_popularity': 'spotify_popularity',
    'total_platforms': 'total_platforms',
    'score_platform': 'score_platform',
    'APPLE_MUSIC_total_appearances': 'A_total_appearances',
    'ZINGMP3_total_appearances': 'Z_total_appearances',
    'NCT_total_appearances': 'N_total_appearances',
    'SPOTIFY_total_appearances': 'S_total_appearances',
    'total_appearances': 'total_appearances',
    'score_sustain': 'score_sustain',
    'ZINGMP3_best_peak_rank': 'Z_best_rank',
    'SPOTIFY_best_peak_rank': 'S_best_rank',
    'NCT_best_peak_rank': 'N_best_rank',
    'APPLE_MUSIC_best_peak_rank': 'A_best_rank',
    'score_rank': 'score_rank',
    'SPOTIFY_total_streams': 'S_total_streams',
    'ZINGMP3_total_plays': 'Z_total_plays',
    'NCT_total_likes': 'N_total_likes',
    'score_historical': 'score_historical',
    'Base_Score': 'total_score',
    'label': 'is-hit',
    'hit_type': 'hit_type'
}

# 3.2. Sắp xếp (Order) các cột
FINAL_COLUMN_ORDER = [
    'title', 'artists', 'featured_artists', 'release_date', 'genres', 'spotify_popularity',
    'total_platforms', 'score_platform',
    'A_total_appearances', 'Z_total_appearances', 'N_total_appearances', 'S_total_appearances',
    'total_appearances',
    'score_sustain',
    'Z_best_rank', 'S_best_rank', 'N_best_rank', 'A_best_rank',
    'score_rank',
    'S_total_streams', 'Z_total_plays', 'N_total_likes',
    'score_historical',
    'total_score',
    'is-hit',
    'hit_type'
]

# 3.3. Các cột chuyển sang số nguyên
COLS_TO_MAKE_INTEGER = [
    'total_platforms', 'score_platform',
    'A_total_appearances', 'Z_total_appearances', 'N_total_appearances', 'S_total_appearances',
    'total_appearances', 'score_sustain',
    'Z_best_rank', 'S_best_rank', 'N_best_rank', 'A_best_rank',
    'score_rank',
    'S_total_streams', 'Z_total_plays', 'N_total_likes',
    'score_historical',
    'total_score',
    'is-hit'
]

# =========================================================================
# --- 4. CÁC HÀM TÍNH ĐIỂM (Scoring Functions) ---
# =========================================================================

def calculate_rank_score(rank):
    """Áp dụng Bậc 1-2-3 (Top 40) dựa trên HẰNG SỐ"""
    if pd.isna(rank) or rank > RANK_FILTER_TOP_N:
        return 0
    for tier_rank, score in RANK_SCORE_TIERS.items():
        if rank <= tier_rank:
            return score
    return 0 # (VD: Rank 41-50)

def calculate_platform_score(count):
    """Thưởng Đa Nền Tảng (dựa trên HẰNG SỐ)"""
    if pd.notna(count) and count >= PLATFORM_COUNT_THRESHOLD:
        return 1
    return 0

def calculate_sustain_score(count):
    """Thưởng Bền Bỉ (dựa trên HẰNG SỐ)"""
    if pd.notna(count) and count >= SUSTAIN_APPEARANCES_THRESHOLD:
        return 1
    return 0

def calculate_historical_score(spotify_streams, zing_plays, nct_likes):
    """Cộng 3 điểm cho "hit lịch sử" (dựa trên HẰNG SỐ)"""
    values = {
        'SPOTIFY_total_streams': spotify_streams,
        'ZINGMP3_total_plays': zing_plays,
        'NCT_total_likes': nct_likes
    }
    for key, threshold in HISTORICAL_THRESHOLDS.items():
        if pd.notna(values.get(key)) and values.get(key) >= threshold:
            return 3
    return 0

def get_hit_type(row):
    """Gán nhãn chi tiết cho loại hit (dựa trên HẰNG SỐ)"""
    # 1. Hit Thành Tích (Proven Hit)
    if row['Base_Score'] >= FINAL_HIT_THRESHOLD:
        return "Proven Hit"
    
    # 2. Logic Cứu sinh (Nếu trượt Cửa 1)
    if pd.notna(row.get('spotify_popularity')):
        pop = int(row['spotify_popularity'])
        if pop >= POPULARITY_LIFELINE:
            if row['score_historical'] > 0:
                return "Legacy Hit (Saved)"
            if row['Base_Score'] == 0:
                return "Catalog Hit (Saved)"
            return "New Hit (Saved)"
            
    # 3. Trượt
    return "Non-Hit"

# =========================================================================
# --- 5. CÁC HÀM XỬ LÝ DỮ LIỆU (Data Functions) ---
# =========================================================================

def remove_diacritics(text):
    """Dọn dấu tiếng Việt"""
    s = str(text); s = re.sub(r'[àáạảãâầấậẩẫăằắặẳẵ]', 'a', s); s = re.sub(r'[èéẹẻẽêềếệểễ]', 'e', s); s = re.sub(r'[ìíịỉĩ]', 'i', s); s = re.sub(r'[òóọỏõôồốộổỗơờớợởỡ]', 'o', s); s = re.sub(r'[ùúụủũưừứựửữ]', 'u', s); s = re.sub(r'[ỳýỵỷỹ]', 'y', s); s = re.sub(r'[đ]', 'd', s); s = s.upper().replace("Đ", "D"); return s.lower()

def create_match_key(title, artist):
    """Tạo khóa chuẩn hóa để merge"""
    try:
        t_name = str(title).lower()
        t_name = re.sub(r'\(feat\..*?\)|'
                        r'\(from ".*?"\)|'
                        r'\(.*?remix.*?\)|'
                        r'\(.*?live.*?\)|'
                        r'\(.*?version.*?\)', '', t_name).strip()
        t_name = remove_diacritics(t_name)
        t_name = re.sub(r'[^a-z0-9]', '', t_name)
        a_name = str(artist).lower()
        a_name = a_name.split(',')[0].split(' & ')[0].strip() 
        a_name = remove_diacritics(a_name)
        a_name = re.sub(r'[^a-z0-9]', '', a_name)
        if not t_name or not a_name: return None
        return f"{t_name}||{a_name}"
    except Exception:
        return None

def get_platform_stats(df_master, platform_df, platform_name):
    """Trích xuất dữ liệu (Rank, Appearances, Historical) từ 1 file chart"""
    print(f"  - Processing {platform_name} (Top {RANK_FILTER_TOP_N})...", end='')
    
    if platform_df is None or platform_df.empty:
        print(" ⚠️ No data.")
        return None
        
    platform_df = platform_df.rename(columns={'Title': 'track_name', 'title': 'track_name', 'Artist': 'artists', 'artists': 'artists'})
    
    if 'Rank' in platform_df.columns:
        platform_df = platform_df[platform_df['Rank'] <= RANK_FILTER_TOP_N].copy()
    else:
        print(f" ⚠️ No 'Rank' column. Skipping.")
        return None
        
    df_master['match_key'] = df_master.apply(lambda row: create_match_key(row['title'], row['artists']), axis=1)
    platform_df['match_key'] = platform_df.apply(lambda row: create_match_key(row['track_name'], row['artists']), axis=1)
    platform_df = platform_df.dropna(subset=['match_key'])
    
    if platform_df.empty:
        print(f" ⚠️ No songs matched Top {RANK_FILTER_TOP_N}.")
        return None

    best_rank_map = platform_df.groupby('match_key')['Rank'].min()
    appearances_map = platform_df.groupby('match_key').size()
    
    historical_stats = {}
    if platform_name == 'SPOTIFY' and 'Total_Streams' in platform_df.columns:
        historical_stats['SPOTIFY_total_streams'] = platform_df.groupby('match_key')['Total_Streams'].max()
    if platform_name == 'ZINGMP3' and 'Total_Plays' in platform_df.columns:
        historical_stats['ZINGMP3_total_plays'] = platform_df.groupby('match_key')['Total_Plays'].max()
    
    df_master_platform = df_master.copy()
    df_master_platform[f'{platform_name}_best_peak_rank'] = df_master_platform['match_key'].map(best_rank_map)
    df_master_platform[f'{platform_name}_total_appearances'] = df_master_platform['match_key'].map(appearances_map)
    
    for col_name, data_map in historical_stats.items():
        df_master_platform[col_name] = df_master_platform['match_key'].map(data_map)
        
    matched_count = df_master_platform[f'{platform_name}_best_peak_rank'].notna().sum()
    print(f" Matched {matched_count} songs.")
    
    new_cols = [col for col in df_master_platform.columns if platform_name in col or 'total_streams' in col or 'total_plays' in col] + ['match_key']
    return df_master_platform[new_cols]

# =========================================================================
# --- 6. HÀM CHẠY CHÍNH (main) ---
# =========================================================================

def main():
    print("--- 🎵 is_hit (v22 - Final Logic) 🎵 ---")
    
    # --- 6.1. Tải file Master ---
    try:
        df_master = pd.read_csv(MASTER_SONG_LIST, encoding='utf-8-sig')
        print(f"📂 Loading Master List... {len(df_master)} songs loaded.")
    except Exception as e:
        print(f"LỖI: Không thể tải file master '{MASTER_SONG_LIST}'. {e}")
        return
        
    # --- 6.2. Tải 4 file chart (Tối ưu bằng Vòng lặp) ---
    platform_dfs = {}
    print("📂 Loading 4 platform files...")
    for platform_name, file_path in PLATFORM_FILES.items():
        try:
            platform_dfs[platform_name] = pd.read_csv(file_path, encoding='utf-8-sig')
            print(f"  - {platform_name}: {len(platform_dfs[platform_name])} records loaded.")
        except Exception as e:
            print(f"  - {platform_name}: ⚠️ FAILED to load ({e})")
            platform_dfs[platform_name] = None
            
    # --- 6.3. Merge dữ liệu ---
    df_result = df_master.copy()
    df_result['match_key'] = df_result.apply(
        lambda row: create_match_key(row['title'], row['artists']), axis=1
    )
    
    for platform_name, df_platform in platform_dfs.items():
        df_stats = get_platform_stats(df_master.copy(), df_platform, platform_name)
        if df_stats is not None:
            df_result = df_result.merge(
                df_stats,
                on='match_key',
                how='left'
            )
            
    # =========================================================================
    # --- 6.4. TÍNH TOÁN "HỆ THỐNG ĐIỂM V17" (MAX 17) ---
    # =========================================================================
    print("\n📈 Calculating scores (v17 Logic)...")

    # 1. Trụ cột 1: Tính điểm rank
    rank_score_cols = []
    for name in PLATFORM_NAMES:
        rank_col = f'{name}_best_peak_rank'
        score_col = f'rank_score_{name[0]}' # vd: rank_score_A
        if rank_col in df_result.columns:
            df_result[score_col] = df_result[rank_col].apply(calculate_rank_score)
            rank_score_cols.append(score_col)
        else:
            df_result[score_col] = 0
            
    df_result['score_rank'] = df_result[rank_score_cols].sum(axis=1)

    # 2. Tính các trụ cột "Snapshot" còn lại
    appearances_cols = [col for col in df_result.columns if '_total_appearances' in col]
    df_result['total_appearances'] = df_result[appearances_cols].sum(axis=1)
    
    peak_cols = [col for col in df_result.columns if '_best_peak_rank' in col]
    df_result['total_platforms'] = df_result[peak_cols].notna().sum(axis=1)
    
    # 3. Tính điểm cho các trụ cột còn lại
    df_result['score_sustain'] = df_result['total_appearances'].apply(calculate_sustain_score)
    df_result['score_platform'] = df_result['total_platforms'].apply(calculate_platform_score)
    df_result['score_historical'] = df_result.apply(
        lambda row: calculate_historical_score(
            row.get('SPOTIFY_total_streams'), 
            row.get('ZINGMP3_total_plays'),
            row.get('NCT_total_likes')
        ),
        axis=1
    )
    
    # 4. Tính "Điểm Cơ Bản" (Base Score)
    df_result['Base_Score'] = (
        df_result['score_rank'] + 
        df_result['score_sustain'] + 
        df_result['score_platform'] + 
        df_result['score_historical']
    )
    
    # 5. Gán nhãn "True Hit"
    df_result['hit_type'] = df_result.apply(get_hit_type, axis=1)
    df_result['label'] = df_result['hit_type'].apply(lambda x: 1 if x != "Non-Hit" else 0)

    # Sắp xếp
    df_result = df_result.sort_values('Base_Score', ascending=False).reset_index(drop=True)
    
    # =========================================================================
    # --- 6.5. DỌN DẸP VÀ LƯU FILE ---
    # =========================================================================
    print("🧹 Cleaning up and formatting output...")

    # 1. Đổi tên cột
    df_result = df_result.rename(columns=RENAME_MAP)

    # 2. Sắp xếp
    final_cols_to_save = [col for col in FINAL_COLUMN_ORDER if col in df_result.columns]
    # (SỬA LỖI 1) Thêm .copy() để tắt tất cả warning
    df_final_output = df_result[final_cols_to_save].copy() 
    
    # 3. Chuyển sang Số nguyên
    for col in COLS_TO_MAKE_INTEGER:
        if col in df_final_output.columns:
            # (SỬA LỖI 2) Bỏ .loc[] đi
            df_final_output[col] = df_final_output[col].astype('Int64') 
            
    # 4. Save result
    print(f"💾 Saving results to: {OUTPUT_FILE}")
    df_final_output.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    
    # 5. Print summary
    print("\n" + "=" * 70)
    print("📊 SUMMARY STATISTICS")
    print("=" * 70)
    
    hit_count = (df_final_output['is-hit'] == 1).sum()
    non_hit_count = (df_final_output['is-hit'] == 0).sum()
    print(f"Label Classification (Threshold = {FINAL_HIT_THRESHOLD}, Lifeline = {POPULARITY_LIFELINE}):")
    print(f"  - Số 'True Hit' (Label=1): {hit_count} bài")
    print(f"  - Số 'True Non-Hit' (Label=0): {non_hit_count} bài")

    if 'hit_type' in df_final_output.columns:
        print("\nHit Type Breakdown:")
        print(df_final_output['hit_type'].value_counts().to_string())
    
    print(f"\n📈 Top 10 songs by total_score:")
    top_10_cols = ['title', 'artists', 'total_score', 'is-hit', 'hit_type',
                   'Z_best_rank', 'S_best_rank', 'A_best_rank', 'N_best_rank']
    top_10_cols_exist = [col for col in top_10_cols if col in df_final_output.columns]
    
    print(df_final_output.head(10)[top_10_cols_exist].to_string(index=False))
    
    print("\n" + "=" * 70)
    print(f"✅ DONE! Results saved to: {OUTPUT_FILE}")
    print("=" * 70)

# =========================================================================
# --- 7. ĐIỂM BẮT ĐẦU CHẠY SCRIPT ---
# =========================================================================
if __name__ == '__main__':
    main()