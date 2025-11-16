import pandas as pd
from typing import List, Set, Dict, Tuple, Optional
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import time
import re
import os
import numpy as np 
from thefuzz import fuzz

try:
    from pyuca import Collator
    pyuca_available = True
except ImportError:
    print("WARNING: Thư viện 'pyuca' chưa được cài đặt. Sắp xếp Tiếng Việt có thể không chuẩn.")
    print("Vui lòng chạy: pip install pyuca")
    pyuca_available = False
    Collator = None

# =============================================================================
# --- BƯỚC 1: THIẾT LẬP CHUNG ---
# =============================================================================
# (Điền ID/Secret của bạn vào đây)
YOUR_CLIENT_ID = "f531846ca30d4dbe8f67c5d2b07f2eca" 
YOUR_CLIENT_SECRET = "a0d355d3c2344b45ac62c106a0382927" 

# (QUAN TRỌNG) Đây là file "Database" DUY NHẤT
# Nó sẽ được TẢI (LOAD) và LƯU (SAVE)
DATABASE_FILE = 'data/song_list_info.csv' 

# =============================================================================
# --- PHẦN 1: CÁC HÀM HỖ TRỢ (Functions) ---
# (Bao gồm các hàm scrape, chuẩn hóa, và gọi API)
# =============================================================================

AUTO_MERGE_THRESHOLD = 95
ASK_THRESHOLD = 75

def normalize_str(s: str) -> str:
    """Chuẩn hóa chuỗi: chữ thường, xóa 'none', gộp khoảng trắng"""
    if pd.isna(s):
        return ''
    s = str(s).strip()
    if s.lower() == 'none':
        return ''
    s = re.sub(r'\s+', ' ', s)
    return s.lower()

def clean_quotes_and_collapse(s: str) -> str:
    """Xóa dấu ngoặc kép và gộp khoảng trắng"""
    if pd.isna(s):
        return ''
    s = str(s)
    s = re.sub(r'[\"\u201C\u201D\u201F\u2033]', '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def normalize_as_tokens(s: str) -> str:
    """Chuẩn hóa cột Artist / Featured_Artists (tách token)"""
    if pd.isna(s):
        return ''
    s = str(s).strip()
    if s == '' or s.lower() == 'none':
        return ''
    
    low = s.lower()
    s_cleaned = re.sub(r'[\(\),&/+\|.]', ' ', low)
    parts = re.split(r'\s+', s_cleaned)
    tokens = sorted(set([p.strip() for p in parts if p.strip()]))
    return ', '.join(tokens)

def normalize_title_key(s: str) -> str:
    """Chuẩn hóa Title để tạo Key (mạnh hơn normalize_str)"""
    s = normalize_str(s) # Bắt đầu bằng chuẩn hóa cơ bản
    if s == '':
        return ''
    s_cleaned = re.sub(r'[\(\)\[\]/\-–—]', ' ', s)
    s_cleaned = re.sub(r'\s+', ' ', s_cleaned).strip()
    return s_cleaned

def aggregate_sources(sources_series: pd.Series) -> str:
    """Gộp các nguồn (vd: 'apple, spotify')"""
    final_set = set()
    cleanup_map = {
        'applemusic': 'apple', 'spotify': 'spotify', 'nct': 'nct',
        'zingmp3': 'zingmp3', 'apple': 'apple', 'zing': 'zingmp3',
    }
    for s in sources_series.dropna().unique():
        if s: 
            tokens = str(s).split(', ') 
            for token in tokens:
                clean_token = token.lower()
                final_set.add(cleanup_map.get(clean_token, clean_token))
    
    if not final_set: return ''
    return ', '.join(sorted(list(final_set)))

def run_unit_tests():
    """Chạy các bài kiểm tra (unit test) nhanh."""
    print("Đang chạy unit tests...")
    assert normalize_as_tokens("Sơn.K (Rap)") == "k, rap, sơn", "Test FAILED: V6 token Sơn.K"
    assert aggregate_sources(pd.Series(['APPLEMUSIC', 'nct', 'apple'])) == "apple, nct", "Test FAILED: V5 aggregate cleanup (APPLEMUSIC)"
    title1 = "còn gì đẹp hơn (mưa đỏ original soundtrack)"
    title2 = "còn gì đẹp hơn - mưa đỏ original soundtrack"
    key1 = normalize_title_key(title1)
    key2 = normalize_title_key(title2)
    assert key1 == "còn gì đẹp hơn mưa đỏ original soundtrack", "Test FAILED: V7 title key ()"
    assert key2 == "còn gì đẹp hơn mưa đỏ original soundtrack", "Test FAILED: V7 title key -"
    assert key1 == key2, "Test FAILED: V7 keys () vs - "
    print("Tất cả tests đã qua. ✔️")
    print("-" * 30)

OUTPUT_COLS = ['title', 'artists', 'featured_artists', 'source']
KEY_COLS = ['title','artists','featured_artists']

def load_source_csv(path: str) -> pd.DataFrame:
    """Tải file nguồn và TỰ ĐỘNG CHUẨN HÓA header (Title -> title)"""
    if not os.path.exists(path):
        return pd.DataFrame(columns=KEY_COLS)
    try:
        df = pd.read_csv(path, encoding='utf-8-sig')
    except Exception as e:
        print(f"⚠️ Lỗi nghiêm trọng khi đọc file {path}: {e}")
        print("   -> Bỏ qua file này.")
        return pd.DataFrame(columns=KEY_COLS)
    col_map = {}
    for col in df.columns:
        low = col.lower()
        if low == 'title': col_map[col] = 'title'
        elif low in ('artist','artists'): col_map[col] = 'artists'
        elif low in ('featured_artists','featured_artists'.lower(),'featured', 'feat', 'featuring'):
            col_map[col] = 'featured_artists'
        elif low == 'source': col_map[col] = 'source'
    df = df.rename(columns=col_map)
    for c in KEY_COLS:
        if c not in df.columns: df[c] = ''
    if 'artists' in df.columns:
        df['artists'] = df['artists'].apply(clean_quotes_and_collapse)
    final_cols_in_source = [c for c in OUTPUT_COLS if c in df.columns]
    return df[final_cols_in_source].copy()

def add_fuzzy_key(df: pd.DataFrame) -> pd.DataFrame:
    """Tạo một "khóa mờ" (có dấu) để so sánh g/t."""
    df_out = df.copy()
    # Đảm bảo các cột key tồn tại (nếu df rỗng)
    if '__key_title_n' not in df_out.columns: df_out['__key_title_n'] = ''
    if '__key_artists_n' not in df_out.columns: df_out['__key_artists_n'] = ''
        
    df_out['__fuzzy_key'] = df_out['__key_title_n'] + ' ' + df_out['__key_artists_n']
    return df_out

def find_best_match(
    new_fuzzy_key: str, 
    master_fuzzy_list: List[Tuple[tuple, str]]
) -> Tuple[int, Optional[tuple]]:
    """Tìm key trong master khớp nhất với key mới."""
    best_score = 0
    best_master_key = None
    for master_key, master_f_key in master_fuzzy_list:
        score = fuzz.token_set_ratio(new_fuzzy_key, master_f_key)
        if score > best_score:
            best_score = score
            best_master_key = master_key
    return best_score, best_master_key

# --- HÀM CHO BƯỚC 2: SPOTIFY API ---

def setup_spotipy(client_id, client_secret):
    """Khởi tạo và xác thực Spotipy."""
    if client_id == "f531846ca30d4dbe8f67c5d2b07f2eca" or client_secret == "a0d355d3c2344b45ac62c106a0382927": 
        print("LỖI: Bạn chưa điền YOUR_CLIENT_ID và YOUR_CLIENT_SECRET.")
        print("-> ĐANG SỬ DỤNG ID MẪU, CÓ THỂ BỊ LỖI.")
    try:
        client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
        sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
        sp.search(q='test', limit=1)
        print("Xác thực Spotify thành công!")
        return sp
    except Exception as e:
        print(f"Lỗi xác thực: {e}")
        return None

def remove_diacritics(text):
    """Chuẩn hoá, xoá dấu tiếng Việt"""
    s = str(text); s = re.sub(r'[àáạảãâầấậẩẫăằắặẳẵ]', 'a', s); s = re.sub(r'[èéẹẻẽêềếệểễ]', 'e', s); s = re.sub(r'[ìíịỉĩ]', 'i', s); s = re.sub(r'[òóọỏõôồốộổỗơờớợởỡ]', 'o', s); s = re.sub(r'[ùúụủũưừứựửữ]', 'u', s); s = re.sub(r'[ỳýỵỷỹ]', 'y', s); s = re.sub(r'[đ]', 'd', s); s = s.upper().replace("Đ", "D"); return s.lower()

def create_match_key(track_name, artist_name):
    """Tạo "key" (khoá) chuẩn hoá (Chỉ dựa trên Nghệ sĩ chính)"""
    try:
        t_name = str(track_name).lower(); t_name = re.sub(r'\(feat\..*?\)', '', t_name); t_name = re.sub(r'\(from ".*?"\)', '', t_name); t_name = re.sub(r'\(.*?remix.*?\)', '', t_name); t_name = re.sub(r'\(.*?live.*?\)', '', t_name); t_name = re.sub(r'\(.*?version.*?\)', '', t_name); t_name = remove_diacritics(t_name); t_name = re.sub(r'[^a-z0-9]', '', t_name)
        a_name = str(artist_name).lower(); a_name = a_name.split(',')[0]; a_name = a_name.split(' & ')[0]; a_name = remove_diacritics(a_name); a_name = re.sub(r'[^a-z0-9]', '', a_name)
        if not t_name or not a_name: return None
        return f"{t_name}||{a_name}"
    except Exception: return None

def call_spotify_api(sp_instance, df_new_songs):
    """Hàm này chỉ gọi API cho các bài hát MỚI (đã được lọc)."""
    print(f"\n--- BẮT ĐẦU BƯỚC 2: GỌI API SPOTIFY ---")
    print(f"Bắt đầu gọi API Spotify cho {len(df_new_songs)} bài hát...")
    
    df_new_songs_copy = df_new_songs.copy()
    
    # Tạo các cột "trống" (empty) nếu nó chưa tồn tại
    for col in ['spotify_popularity', 'spotify_release_date', 'spotify_genres', 'spotify_track_id']:
        if col not in df_new_songs_copy.columns:
            df_new_songs_copy[col] = np.none

    for index, row in df_new_songs_copy.iterrows():
        title = row['title']
        artist = row['artists']
        query = f"track:{title} artist:{artist}"
        
        try:
            results = sp_instance.search(q=query, limit=1, type='track')
            if results['tracks']['items']:
                track = results['tracks']['items'][0]
                track_id = track['id']
                popularity = track['popularity']
                release_date = track['album']['release_date']
                
                genres = []
                if 'artists' in track and track['artists']:
                    artist_id = track['artists'][0]['id']
                    if artist_id:
                        artist_info = sp_instance.artist(artist_id)
                        genres = artist_info['genres']
                
                df_new_songs_copy.at[index, 'spotify_popularity'] = popularity
                df_new_songs_copy.at[index, 'spotify_release_date'] = release_date
                df_new_songs_copy.at[index, 'spotify_genres'] = ", ".join(genres) if genres else np.none
                df_new_songs_copy.at[index, 'spotify_track_id'] = track_id
                print(f"-> Đã lấy thành công: {title}")
            else:
                print(f"-> Không tìm thấy trên Spotify: {title}")
            time.sleep(0.1) # Độ trễ an toàn
        except Exception as e:
            print(f"LỖI API khi xử lý '{title}': {e}")
    
    return df_new_songs_copy

# =============================================================================
# --- HÀM TẠO KEY (Thống nhất) ---
# =============================================================================

def add_all_keys(df: pd.DataFrame) -> pd.DataFrame:
    """
    Hàm này tạo CẢ HAI loại key cần thiết:
    1. __key__: Key chi tiết (có dấu, full feat) để duyệt (y/n)
    2. match_key: Key thô (không dấu, no feat) để call API
    """
    out = df.copy()
    if 'source' not in out.columns: out['source'] = ''
    out['source'] = out['source'].fillna('')

    # Đảm bảo các cột cơ bản là string
    out['title'] = out['title'].astype(str)
    out['artists'] = out['artists'].astype(str)
    out['featured_artists'] = out['featured_artists'].astype(str)

    # BƯỚC 1: TRÍCH XUẤT (Extract) "feat" từ Title
    feat_regex = re.compile(r'\s*\((feat|ft)[\s\.]+(.*?)\)', re.IGNORECASE)
    title_str = out['title'].astype(str) 
    extracted = title_str.str.extract(feat_regex)
    feat_from_title = pd.Series(dtype='str')
    if 1 in extracted.columns: feat_from_title = extracted[1].fillna('')
    else: feat_from_title = pd.Series(dtype='str', index=out.index).fillna('')
    title_cleaned = title_str.str.replace(feat_regex, '', regex=True)

    # BƯỚC 2: HỢP NHẤT (Combine)
    feat_combined = out['featured_artists'].fillna('') + ' ' + feat_from_title

    # BƯỚC 3: TOKEN HÓA (Tạo 2 loại key)
    # Key 1: Key chi tiết (để duyệt y/n)
    out['__key_title_n'] = title_cleaned.apply(normalize_title_key)
    out['__key_artists_n'] = out['artists'].apply(normalize_as_tokens)
    out['__key_featured_n'] = feat_combined.apply(normalize_as_tokens)
    
    # Xử lý lỗi nếu các cột này rỗng
    for col in ['__key_title_n', '__key_artists_n', '__key_featured_n']:
        if col not in out.columns:
            out[col] = ''
    out['__key__'] = list(zip(
        out['__key_title_n'].astype(str), 
        out['__key_artists_n'].astype(str), 
        out['__key_featured_n'].astype(str)
    ))
    
    # Key 2: Key thô (để call API)
    # Tạo 'match_key' ngay cả khi nó đã tồn tại, để đảm bảo tính nhất quán
    out['match_key'] = out.apply(
        lambda row: create_match_key(row['title'], row['artists']), 
        axis=1
    )
    return out

def main():
    collator = Collator() if pyuca_available else None
    run_unit_tests()

    # --- 3.1: Tải "Database" cũ (file song_list_info.csv) ---
    if os.path.exists(DATABASE_FILE):
        print(f"Đang tải 'database' hiện có từ: {DATABASE_FILE}")
        try:
            df_db = pd.read_csv(DATABASE_FILE)
        except Exception as e:
            print(f"Lỗi đọc file database: {e}. Tạo database trống.")
            df_db = pd.DataFrame()
    else:
        print(f"Database '{DATABASE_FILE}' chưa tồn tại. Sẽ tạo mới.")
        df_db = pd.DataFrame()

    # --- 3.2: Chuẩn hóa DB cũ và Lấy "Bộ nhớ" ---
    nm_db = add_all_keys(df_db)
    
    # 1. Bộ nhớ duyệt (y/n): Dùng key CHI TIẾT
    old_detailed_keys = set(nm_db['__key__'])
    
    # 2. Bộ nhớ API: Dùng key THÔ (match_key)
    processed_api_keys = set(nm_db[nm_db['spotify_track_id'].notna()]['match_key'].dropna())
    
    print(f"Đã tải {len(old_detailed_keys)} bài hát đã duyệt từ database.")
    print(f"Trong đó có {len(processed_api_keys)} bài đã có dữ liệu Spotify.")
    print("-" * 30)

    # --- 3.3: Tải và Chuẩn hóa Nguồn (Chart) ---
    sources_map: Dict[str, str] = {
        'apple': 'data/apple_music_top100_kworb_vn.csv',
        'spotify': 'data/spotify_top100_kworb_vn.csv',
        'nct': 'data/nct_top50.csv',
        'zingmp3': 'data/zingmp3_top100.csv',
    }
    dfs = []
    for source_name, p in sources_map.items():
        df = load_source_csv(p)
        if not df.empty:
            df['source'] = source_name
            dfs.append(df)
            
    if not dfs:
        print("Không tìm thấy file nguồn nào. Giữ nguyên database.")
        # Vẫn lưu lại nm_db để chuẩn hóa (nếu file có lỗi)
        nm_db.to_csv(DATABASE_FILE, index=False, encoding='utf-8-sig')
        print("Đã lưu lại database (để chuẩn hóa).")
        return
    else:
        df_sources = pd.concat(dfs, ignore_index=True)

    nm_sources = add_all_keys(df_sources)

    # --- 3.4: BƯỚC 1 - CHẠY LOGIC DUYỆT (y/n) VÀ (g/t) ---
    
    # Danh sách (key, fuzzy_key) của master để so sánh
    master_fuzzy_list = []
    if not nm_db.empty:
        nm_db_fuzzy = add_fuzzy_key(nm_db)
        master_fuzzy_list = list(zip(
            nm_db_fuzzy['__key__'], 
            nm_db_fuzzy['__fuzzy_key']
        ))

    key_remap: Dict[tuple, tuple] = {}
    truly_new_songs_to_ask: List[pd.Series] = []
    approved_new_songs_list: List[pd.Series] = []
    rejected_keys: Set[tuple] = set()

    # Lọc các bài hát MỚI CÓ THỂ CÓ từ sources
    unique_sources = pd.DataFrame()
    if not nm_sources.empty:
        nm_sources_fuzzy = add_fuzzy_key(nm_sources)
        unique_sources = nm_sources_fuzzy.loc[
            ~nm_sources_fuzzy['__key__'].duplicated(keep='first')
        ].copy()

    print("Bắt đầu lọc bài hát mới từ chart...")
    for _, new_song in unique_sources.iterrows():
        new_key = new_song['__key__']
        new_f_key = new_song['__fuzzy_key']

        # 1. [CHECK BỘ NHỚ 1] Bỏ qua nếu key này đã có y hệt trong database
        if new_key in old_detailed_keys:
            continue
            
        # 2. Bỏ qua nếu key này đã được xử lý (trong vòng lặp này)
        if new_key in key_remap:
            continue
            
        # 3. Chạy so sánh mờ
        if not master_fuzzy_list:
            truly_new_songs_to_ask.append(new_song)
            continue

        best_score, best_master_key = find_best_match(new_f_key, master_fuzzy_list)
        
        # 4. Phân loại
        if best_score >= AUTO_MERGE_THRESHOLD:
            master_info = nm_db[nm_db['__key__'] == best_master_key].iloc[0]
            print(f"  [Auto-Merge] '{new_song['title']}' (Nguồn: {new_song['source']})")
            print(f"       -> vào: '{master_info['title']}' (Điểm: {best_score})")
            key_remap[new_key] = best_master_key

        elif best_score >= ASK_THRESHOLD:
            master_info = nm_db[nm_db['__key__'] == best_master_key].iloc[0]
            
            print("\n" + "="*20)
            print(f"  [NGHI NGỜ] (Điểm: {best_score})")
            print(f"  Bài mới:   {new_song['title']} | {new_song['artists']} | {new_song.get('featured_artists','')}")
            print(f"  Giống bài: {master_info['title']} | {master_info['artists']} | {master_info.get('featured_artists','')}")
            print("="*20)
            
            choice = ''
            while choice not in ['g', 't', 'exit']:
                choice = input("  Gộp vào bài cũ (g) / Tạo bài mới (t) / Thoát (exit)?: ").lower().strip()
            
            if choice == 'g':
                print(f"  -> [Đã Gộp] '{new_song['title']}'.")
                key_remap[new_key] = best_master_key
            elif choice == 't':
                print(f"  -> [Tạo Mới] '{new_song['title']}'.")
                truly_new_songs_to_ask.append(new_song)
            elif choice == 'exit':
                print("Đã hủy. Không có thay đổi nào được lưu.")
                return 
        
        else:
            truly_new_songs_to_ask.append(new_song)

    print("-" * 30)

    # 5) HỎI XÁC NHẬN CÁC BÀI "THỰC SỰ MỚI" (y/n/all)
    if truly_new_songs_to_ask:
        print(f"--- Tìm thấy {len(truly_new_songs_to_ask)} bài hát (có vẻ) mới. Vui lòng xác nhận:")
        
        truly_new_songs_df = pd.DataFrame(truly_new_songs_to_ask).sort_values(
            by=['title', 'artists'], 
            key=lambda s: s.astype(str).str.lower()
        )
        
        auto_add_all = False
        
        for _, row in truly_new_songs_df.iterrows():
            if auto_add_all:
                approved_new_songs_list.append(row)
                continue

            title = row.get('title', 'N/A')
            artists = row.get('artists', 'N/A')
            feat = row.get('featured_artists', '')
            source = row.get('source', 'N/A')
            
            print("\n" + "="*10)
            print(f"  Bài hát: {title}")
            print(f"  Nghệ sĩ: {artists}")
            if feat: print(f"  Hợp tác: {feat}")
            print(f"  Nguồn: {source}")
            print("="*10)
            
            choice = ''
            while choice not in ['y', 'n', 'all', 'exit']:
                choice = input("  Đồng ý thêm bài này? (y/n/all/exit): ").lower().strip()
                
            if choice == 'y':
                print(f"  -> [ĐÃ THÊM] {title}")
                approved_new_songs_list.append(row)
            elif choice == 'n':
                print(f"  -> [ĐÃ BỎ QUA] {title}")
                rejected_keys.add(row['__key__'])
            elif choice == 'all':
                print("  -> [THÊM TẤT CẢ] Bắt đầu thêm tất cả các bài còn lại...")
                auto_add_all = True
                approved_new_songs_list.append(row)
            elif choice == 'exit':
                print("Đã hủy. Không có thay đổi nào được lưu.")
                return 
        
        if auto_add_all:
             print("\nĐã hoàn tất thêm tất cả bài hát mới.")
    
    # --- 3.6: Gộp và Cập nhật Source ---
    print("\n--- BƯỚC 1 HOÀN TẤT: ĐANG GỘP DỮ LIỆU ---")
    
    approved_new_df = pd.DataFrame(approved_new_songs_list)
    
    if not nm_sources.empty:
        nm_sources['__key__'] = nm_sources['__key__'].apply(lambda k: key_remap.get(k, k))

    if not nm_sources.empty:
        nm_sources_filtered = nm_sources[~nm_sources['__key__'].isin(rejected_keys)]
    else:
        nm_sources_filtered = nm_sources

    # Gộp tất cả lại:
    combined = pd.concat(
        [nm_db, approved_new_df, nm_sources_filtered], 
        ignore_index=True
    )

    # 4. Gộp (Groupby) và Lấy thông tin (Deduplicate)
    if not combined.empty:
        # Gộp sources
        agg_sources = combined.groupby('__key__')['source'].apply(
            aggregate_sources
        ).reset_index()
        
        # Chọn thông tin bài hát (giữ bản MỚI NHẤT, vd: có API)
        combined['has_api_data'] = combined['spotify_track_id'].notna()
        combined = combined.sort_values(
            by=['__key__', 'has_api_data'], 
            ascending=[True, False] 
        )
        selected_info = combined.loc[
            ~combined['__key__'].duplicated(keep='first')
        ].reset_index(drop=True)
        
        # Merge lại
        df_merged = pd.merge(
            selected_info.drop(columns=['source']), 
            agg_sources, 
            on='__key__', 
            how='left'
        )
    else:
        print("Không có dữ liệu nào. Dừng.")
        return

    # --- 3.7: BƯỚC 2 - LỌC VÀ GỌI API ---
    
    # [CHECK BỘ NHỚ 2] Lọc ra những bài CẦN gọi API
    # 1. Đảm bảo 'match_key' đã được tạo
    if 'match_key' not in df_merged.columns:
         df_merged['match_key'] = df_merged.apply(
            lambda row: create_match_key(row['title'], row['artists']), 
            axis=1
        )
    
    # 2. Lọc: (Có 'match_key') VÀ (key đó KHÔNG có trong bộ nhớ 'processed_api_keys')
    df_new_to_fetch = df_merged[
        df_merged['match_key'].notna() &
        ~df_merged['match_key'].isin(processed_api_keys)
    ].copy()

    df_new_fetched = pd.DataFrame()
    if not df_new_to_fetch.empty:
        print(f"\nPhát hiện {len(df_new_to_fetch)} bài hát cần lấy/cập nhật thông tin API.")
        sp = setup_spotipy(YOUR_CLIENT_ID, YOUR_CLIENT_SECRET)
        if not sp:
            print("Dừng vì lỗi xác thực.")
            return
        
        df_new_fetched = call_spotify_api(sp, df_new_to_fetch)
    else:
        print("\n--- BƯỚC 2: BỎ QUA ---")
        print("Không có bài hát mới nào để gọi API.")
    
    # --- 3.8: BƯỚC 3 - GỘP CUỐI CÙNG VÀ LƯU ---
    print("\n--- BƯỚC 3: TỔNG HỢP VÀ LƯU DATABASE ---")
    
    # Gộp df_merged (chứa bài cũ) và df_new_fetched (bài mới có API)
    df_final_combined = pd.concat([df_merged, df_new_fetched], ignore_index=True)
    
    # Lọc trùng lặp lần cuối (giữ bản MỚI NHẤT, tức là bản có API)
    # Lần này gộp bằng 'match_key'
    df_final_combined['has_api_data'] = df_final_combined['spotify_track_id'].notna()
    df_final_combined = df_final_combined.sort_values(
        by=['match_key', 'has_api_data'], 
        ascending=[True, False] 
    )
    
    df_final = df_final_combined.drop_duplicates(
        subset=['match_key'], 
        keep='first' 
    )
    
    # Sắp xếp theo Tiếng Việt (nếu có)
    if not df_final.empty:
        if collator:
            print("Đang sắp xếp kết quả bằng pyuca (chuẩn Tiếng Việt)...")
            vietnamese_key = lambda s: s.astype(str).apply(collator.sort_key)
            df_final = df_final.sort_values(
                by=['title', 'artists', 'featured_artists'], 
                key=vietnamese_key
            ).reset_index(drop=True)
        else:
            print("Đang sắp xếp (mặc định)...")
            df_final = df_final.sort_values(
                by=['title', 'artists', 'featured_artists']
            ).reset_index(drop=True)

    # --- 3.9: LỌC CỘT VÀ LƯU ---
    
    # Các cột cơ bản
    final_cols = [
        'title', 'artists', 'featured_artists', 'source', 
        'spotify_popularity', 'spotify_release_date', 'spotify_genres', 
        'spotify_track_id', 
        'match_key', '__key__' # Giữ lại key để debug
    ]
    
    cols_to_save = [col for col in final_cols if col in df_final.columns]
    df_to_save = df_final[cols_to_save].copy()  # tốt nhất nên copy rõ ràng
    if 'spotify_popularity' in df_to_save.columns:
        df_to_save['spotify_popularity'] = (
            pd.to_numeric(df_to_save['spotify_popularity'], errors='coerce')
            .astype('Int64')
        )
    df_to_save.to_csv(DATABASE_FILE, index=False, encoding='utf-8-sig')

    print(f"\n--- HOÀN TẤT ---")
    print(f"Đã lưu tổng cộng {len(df_to_save)} bài hát vào '{DATABASE_FILE}'.")
    if len(df_new_fetched) > 0:
        print(f"Đã gọi API và thêm mới/cập nhật {len(df_new_fetched)} bài hát.")

if __name__ == "__main__":
    main()