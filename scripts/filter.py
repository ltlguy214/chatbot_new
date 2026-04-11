import pandas as pd
import re

def filter_songs_refined_v6(input_path, output_path, min_year=2010):
    # 1. Đọc dữ liệu
    df = pd.read_csv(input_path)
    initial_count = len(df)
    
    # 2. Danh sách từ khóa lọc Title (Bản không gốc + Album filler)
    title_garbage_keywords = [
        r'\bbeat\b', r'\bkaraoke\b', r'\binstrumental\b', r'\bbacking track\b',
        r'\bremix\b', r'\bcover\b', r'\bparody\b', r'\bmashup\b', r'\bmedley\b', 
        r'\bacoustic\b', r'\blive\b',
        r'\bnhạc nền\b', r'\bhát văn\b', r'\btone\b', r'\bspeedup\b', r'\bslowdown\b', 
        r'\bsped up\b', r'\bslowed down\b',
        r'\bintro\b', r'\boutro\b', r'\binterlude\b', r'\bskit\b', r'\bprelude\b', r'\btransition\b',
        r'deep', r'house', r'lofi'

    ]
    title_pattern = '|'.join(title_garbage_keywords)
    
    # 3. Danh sách từ khóa lọc Thể loại (Chỉ nhạc Việt)
    vn_genre_keywords = ['v-pop', 'vietnam', 'vietnamese', 'vinahouse', 'bolero']
    genre_pattern = '|'.join(vn_genre_keywords)

    # 4. Danh sách nghệ sĩ loại bỏ
    excluded_keywords = [
        'christopher wong', 'ian rees', 'ji yeon', 
        'piano', 'pianist', 'instrumental', 'orchestra', 'relaxing', 'yoga', '김현주', '최인희', 
        'oh hye joo', 'king beats', 'yewon', 'lim hyun ji', 'minseo', 'lady gaga', 
        'chihiro onitsuka', 'ライ', 'kinokoteikoku', 'atb', 'avicii', 'meduza', 'maroon 5', 'gwen stefani', 
        'chris lake', 'tiësto', 'hy', 'ヨルシカ', 'zedd', 
        'the beat garden', 'dj snake', 'taylor swift', 'martin jensen', 'acidman', 'ado', 'jax jones', 
        'eric prydz', 'zack tabudlo', 'justin bieber', 'duke dumont', 'ビリー・アイリッシュ', 'topic', 
        'suzukisuzuki', 'olivia rodrigo', 'crazy ken band', 'jonas blue', 'twocolors', 'takashi sorimachi', 
        'weird genius', 'fujii kaze', 'imase', 
        'swedish house mafia', 'キマグレン', 'beni', 'ms.ooja', 'motohiro hata', 'novelbright', 'kungs', 'garrett crosby',
        'calum scott', 'lim jeong hee'
    ]
    artist_pattern = '|'.join(excluded_keywords)
    
    # 5. Tiến hành lọc từng bước
    
    # A. Lọc Title
    is_original_title = ~df['title'].str.contains(title_pattern, case=False, na=False, regex=True)
    
    # B. Lọc Popularity > 5 (ĐIỀU KIỆN MỚI)
    is_popular = df['spotify_popularity'] > 5
    
    # C. Lọc Genre
    is_vn_music = df['spotify_genres'].str.contains(genre_pattern, case=False, na=True, regex=True)

    # D. Lọc Nghệ sĩ
    is_not_excluded_artist = ~(
        df['artists'].str.contains(artist_pattern, case=False, na=False, regex=True) |
        df['featured_artists'].str.contains(artist_pattern, case=False, na=False, regex=True)
    )
    
    # E. Lọc Năm phát hành
    df['release_year'] = pd.to_numeric(df['spotify_release_date'].str[:4], errors='coerce')
    is_modern = (df['release_year'] >= min_year)
    
    # 6. Kết hợp tất cả điều kiện (Thêm is_popular vào đây)
    df_cleaned = df[is_original_title & is_popular & is_vn_music & is_not_excluded_artist & is_modern].copy()
    
    # 7. Lưu kết quả
    df_cleaned.drop(columns=['release_year']).to_csv(output_path, index=False)
    
    # 8. Thống kê
    print(f"--- THỐNG KÊ LỌC NÂNG CAO (V6) ---")
    print(f"Số lượng bài ban đầu: {initial_count}")
    print(f"Số lượng bài sạch cuối cùng: {len(df_cleaned)}")
    print(f"Tổng số bài đã loại bỏ: {initial_count - len(df_cleaned)}")
    print("-" * 40)
    print(f"Chi tiết loại bỏ:")
    print(f"- Do Title/Album filler: {initial_count - len(df[is_original_title])}")
    print(f"- Do Popularity <= 5: {initial_count - len(df[is_popular])}")
    print(f"- Do Nghệ sĩ nhạc cụ/Ngoại lệ: {initial_count - len(df[is_not_excluded_artist])}")
    print(f"- Do Nhạc cũ (trước {min_year}): {len(df[~is_modern & is_modern.notna()])}")
    print("-" * 40)
    print(f"Đã lưu kết quả vào: {output_path}")

# --- Thực thi ---
# Thay đổi đường dẫn file cho đúng với máy của bạn
input_file = 'final_data\songs_missing_from_base.csv'
output_file = 'final_data\songs_missing.csv'

filter_songs_refined_v6(input_file, output_file, min_year=2010)