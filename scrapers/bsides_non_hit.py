import spotipy
import spotipy.exceptions
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import time
import re
import os
import sys
import io
from dotenv import load_dotenv
from tqdm import tqdm 
import numpy as np 

# =============================================================================
# --- 1. THIẾT LẬP CHUNG VÀ BIẾN MÔI TRƯỜNG ---
# =============================================================================
load_dotenv()
YOUR_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
YOUR_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")

# Nguồn Input: File kết quả từ is_hit.py
HIT_MAKER_SOURCE_FILE = 'data/is_hit.csv'

# Tên file Output cho các bài B-Sides
BSIDES_OUTPUT_FILE = 'data/bsides_non_hit.csv'

# --- FILE LOG NGHỆ SĨ ĐÃ XỬ LÝ (Để không chạy lại nghệ sĩ cũ) ---
PROCESSED_ARTISTS_LOG = 'data/processed_artists.txt'
# =============================================================================
# --- 🔴 BẠN CẦN ĐIỀN THÔNG TIN TẠI ĐÂY ---
# =============================================================================
# Danh sách nghệ sĩ bạn muốn chạy. 
# NẾU TRỐNG ([]): Chương trình sẽ tự động lấy TẤT CẢ nghệ sĩ có 'is-hit' = 1 từ HIT_MAKER_SOURCE_FILE.
ARTISTS_MANUAL_LIST = [
    # Thêm tên nghệ sĩ ở đây nếu bạn muốn giới hạn.
]
# =============================================================================

# Các bộ lọc "Thông minh"
MIN_RELEASE_YEAR = 2016 # Chỉ lấy nhạc năm 2016 trở đi
MIN_DURATION_MINUTES = 1 # (Tức là 1 phút)
MIN_DURATION_MS = MIN_DURATION_MINUTES * 60 * 1000 

# --- BỘ LỌC NGỮ CẢNH ---
SHORT_JUNK_DURATION_MS = 60 * 1000 # Intro/Skit chỉ bị loại nếu < 60 giây
# ----------------------------

MAX_ARTISTS_PER_RUN = 50 # Giới hạn 50 nghệ sĩ mỗi lần chạy
MAX_BSIDES_PER_ARTIST = 5 # Giới hạn 5 bài/nghệ sĩ để tránh mất cân bằng dữ liệu

JUNK_KEYWORDS = [
    'remix', 'live', 'acoustic', 'instrumental', 'version', 'edit', 're-release',
    'deluxe', 'anniversary', '(commentary)', 'slowed', 'reverb', 'sped up', 
    'mix', 'house', 'cover', 'recap', 'vinahouse', 'trap', 'vmix',
    'speed up', 'speedup', 'house lak', 'snippet', '(beat)', 'crypto', 
    'remaster', 'radio', 'theme song', 'original soundtrack', 
    'ep. ', 'pt. ', 'part '
]

# --- KEYWORDS CHỈ BỊ LỌC NẾU RẤT NGẮN (<= 60 giây) ---
SHORT_JUNK_KEYWORDS = ['intro', 'outro', 'skit', 'interlude']
# ----------------------------------------

DELAY_PER_ARTIST = 30 # (giây)

# =============================================================================
# --- 2. CÁC HÀM TIỆN ÍCH ---
# =============================================================================

def remove_diacritics(text):
    """Dọn dấu tiếng Việt"""
    s = str(text); s = re.sub(r'[àáạảãâầấậẩẫăằắặẳẵ]', 'a', s); s = re.sub(r'[èéẹẻẽêềếệểễ]', 'e', s); s = re.sub(r'[ìíịỉĩ]', 'i', s); s = re.sub(r'[òóọỏõôồốộổỗơờớợởỡ]', 'o', s); s = re.sub(r'[ùúụủũưừứựửữ]', 'u', s); s = re.sub(r'[ỳýỵỷỹ]', 'y', s); s = re.sub(r'[đ]', 'd', s); s = s.upper().replace("Đ", "D"); return s.lower()

def create_match_key(title, artist):
    """Tạo khóa chuẩn hóa để so sánh"""
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
        
def get_processed_artists(log_file):
    """Đọc file log để lấy danh sách các nghệ sĩ đã được xử lý (theo tên chuẩn hóa)."""
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            return set(line.strip() for line in f)
    except FileNotFoundError:
        return set()
    except Exception as e:
        print(f"⚠️ Lỗi khi đọc file log {log_file}: {e}")
        return set()

def log_processed_artist(log_file, artist_name):
    """Ghi tên nghệ sĩ (chuẩn hóa) vào file log sau khi xử lý."""
    normalized_name = str(artist_name).lower().strip()
    try:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(normalized_name + '\n')
    except Exception as e:
        print(f"❌ Lỗi khi ghi log cho nghệ sĩ {artist_name}: {e}")

# =============================================================================
# --- 3. CÁC HÀM XỬ LÝ ---
# =============================================================================

def setup_spotipy(client_id, client_secret):
    """Khởi tạo và xác thực Spotipy bằng biến môi trường."""
    if not client_id or not client_secret:
        print("LỖI: SPOTIFY_CLIENT_ID hoặc SPOTIFY_CLIENT_SECRET không tồn tại (chưa được nạp từ .env).")
        return None
    try:
        client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
        sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
        sp.search(q='test', limit=1)
        print("✅ Xác thực Spotify thành công!")
        return sp
    except Exception as e:
        print(f"❌ Lỗi xác thực: {e}. Vui lòng kiểm tra Client ID/Secret.")
        return None

def get_hit_makers(file_path):
    """Đọc file is_hit.csv, tìm các nghệ sĩ có is-hit = 1."""
    print(f"📂 Đang tải danh sách 'Hit Makers' từ file: {file_path}...")
    try:
        df = pd.read_csv(file_path)
        if 'is-hit' not in df.columns or 'artists' not in df.columns:
            print(f"   -> ⚠️ File '{file_path}' thiếu cột 'is-hit' hoặc 'artists'.")
            return set()
        
        df_hits = df[df['is-hit'] == 1]
        
        hit_maker_names = set()
        for artist_string in df_hits['artists'].dropna():
            names_comma = [name.strip() for name in artist_string.split(',')]
            for name in names_comma:
                names_ampersand = [n.strip() for n in name.split(' & ')]
                hit_maker_names.update([n.lower() for n in names_ampersand])
        
        print(f"   -> ✅ Tìm thấy {len(hit_maker_names)} 'Hit Makers' (nghệ sĩ có ít nhất 1 hit).")
        return hit_maker_names
            
    except FileNotFoundError:
        print(f"❌ LỖI: Không tìm thấy file {file_path}. Hãy chạy is_hit.py trước.")
        return set()
    except Exception as e:
        print(f"❌ Lỗi khi đọc file {file_path}: {e}")
        return set()

def get_known_chart_songs(file_path):
    """Đọc is_hit.csv và tạo ra một Set (danh sách) các "match_key" của tất cả bài hát đã có trong chart."""
    print(f"📂 Đang tải danh sách 'Known Songs' (để lọc) từ file: {file_path}...")
    try:
        df = pd.read_csv(file_path)
        if 'title' not in df.columns or 'artists' not in df.columns:
            print(f"   -> ⚠️ File '{file_path}' thiếu cột 'title' hoặc 'artists'.")
            return set()
        
        df['match_key'] = df.apply(lambda row: create_match_key(row['title'], row['artists']), axis=1)
        known_songs_set = set(df['match_key'].dropna())
        
        print(f"   -> ✅ Đã có {len(known_songs_set)} bài hát trong chart (sẽ bị lọc bỏ nếu trùng).")
        return known_songs_set
            
    except FileNotFoundError:
        print(f"❌ LỖI: Không tìm thấy file {file_path} khi tải Known Songs.")
        return set()
    except Exception as e:
        print(f"❌ Lỗi khi đọc file {file_path}: {e}")
        return set()

def get_full_track_details(sp_instance, track_ids):
    """Lấy thông tin metadata cơ bản (50 bài/lần)."""
    if not track_ids: return []
    all_details = []
    for i in range(0, len(track_ids), 50):
        batch_ids = track_ids[i:i+50]
        try:
            tracks_info = sp_instance.tracks(tracks=batch_ids)
            for track in tracks_info['tracks']:
                if not track: continue 
                all_details.append({
                    'track_id': track['id'],
                    'track_name': track['name'],
                    'artist_name': track['artists'][0]['name'],
                    'artist_id': track['artists'][0]['id'],
                    'release_date': track['album']['release_date'],
                    'spotify_popularity': track['popularity'],
                    'duration_ms': track['duration_ms']
                })
        except Exception as e:
            try:
                tqdm.write(f"   -> ⚠️ Lỗi khi lấy chi tiết track batch {i}: {e}")
            except NameError:
                 print(f"   -> ⚠️ Lỗi khi lấy chi tiết track batch {i}: {e}")
        time.sleep(0.1) 
    return all_details

def fetch_bsides(sp_instance, artists_to_fetch, known_songs_set, min_year, min_duration, junk_keywords, short_junk_keywords, short_junk_ms, delay, max_bsides, log_file):
    """
    Cào và tự động LỌC B-SIDES với logic thông minh.
    """
    print(f"\n--- 🎶 BẮT ĐẦU THU THẬP B-SIDES TỪ NGHỆ SĨ ---")
    print(f"   Điều kiện: Phát hành >= {min_year}, Thời lượng >= {MIN_DURATION_MINUTES} phút")
    print(f"   Logic lọc: Intro/Skit chỉ bị loại nếu < {round(short_junk_ms/1000)} giây.")
    print(f"   Giới hạn: {max_bsides} bài B-Side/nghệ sĩ")

    newly_fetched_bsides = [] 
    total_junk_filtered, total_short_filtered, total_old_filtered, total_chart_song_filtered, total_zero_duration_filtered = 0, 0, 0, 0, 0

    artist_loop = tqdm(artists_to_fetch, desc="Đang xử lý nghệ sĩ", unit="nghệ sĩ")
    
    for artist_name in artist_loop:
        while True:
            try:
                tqdm.write(f"\nĐang xử lý: {artist_name}")
                artist_loop.set_description(f"Đang xử lý {artist_name}")
                
                # --- 1. Tìm Artist ID ---
                results = sp_instance.search(q=artist_name, limit=1, type='artist')
                if not results['artists']['items']:
                    tqdm.write(f"   -> ⚠️ Không tìm thấy '{artist_name}' trên Spotify. Bỏ qua.")
                    break 
                artist_id = results['artists']['items'][0]['id']

                # --- 2. Lấy tất cả Albums/Singles ---
                albums_response = sp_instance.artist_albums(artist_id, album_type='album,single', country='VN', limit=50)
                albums = albums_response['items']
                while albums_response['next']:
                    albums_response = sp_instance.next(albums_response)
                    albums.extend(albums_response['items'])
                album_ids = [album['id'] for album in albums]

                # --- 3. Lấy tất cả Track IDs từ Albums ---
                track_ids_from_artist = set()
                for album_id in album_ids:
                    try:
                        tracks_response = sp_instance.album_tracks(album_id, limit=50)
                        tracks = tracks_response['items']
                        for track in tracks:
                            if track['artists'] and str(track['artists'][0]['id']) == str(artist_id):
                                track_ids_from_artist.add(track['id'])
                    except Exception: pass
                    time.sleep(0.05)


                if track_ids_from_artist:
                    # --- 4. Lấy chi tiết Track Metadata ---
                    track_details = get_full_track_details(sp_instance, list(track_ids_from_artist))
                    
                    artist_bsides_count = 0 
                    artist_junk, artist_short, artist_old, artist_chart_song, artist_zero_duration = 0, 0, 0, 0, 0
                    
                    # --- 5. Áp dụng Lọc B-Sides (Logic Cải tiến) ---
                    for track in track_details:
                        if artist_bsides_count >= max_bsides:
                            break 
                            
                        track_name_lower = track.get('track_name', '').lower()
                        release_date_str = track.get('release_date', '1990')
                        track_duration = track.get('duration_ms', 0)
                        
                        match = re.match(r'^\d{4}', release_date_str)
                        release_year = int(match.group(0)) if match else 1990
                        
                        track_match_key = create_match_key(track['track_name'], track['artist_name'])

                        # Phân loại cơ bản
                        is_zero_duration = (track_duration < 1000)
                        is_valid_year = (release_year >= min_year)
                        is_valid_duration = (track_duration >= min_duration)
                        is_bside = (track_match_key not in known_songs_set) 
                        
                        # LOGIC LỌC THÔNG MINH
                        is_junk = any(keyword in track_name_lower for keyword in junk_keywords)
                        is_short_junk = any(keyword in track_name_lower for keyword in short_junk_keywords) and (track_duration < short_junk_ms)
                        
                        is_filtered = False
                        
                        if is_zero_duration: 
                            artist_zero_duration += 1
                            is_filtered = True
                        elif is_junk: 
                            artist_junk += 1
                            is_filtered = True
                        elif is_short_junk: # Loại bỏ 'intro/skit' chỉ khi chúng ngắn hơn 60s
                            artist_junk += 1
                            is_filtered = True
                        elif not is_valid_duration: 
                            artist_short += 1
                            is_filtered = True
                        elif not is_valid_year: 
                            artist_old += 1
                            is_filtered = True
                        elif not is_bside: 
                            artist_chart_song += 1
                            is_filtered = True
                        
                        if not is_filtered:
                            newly_fetched_bsides.append(track)
                            artist_bsides_count += 1
                    
                    tqdm.write(f"   -> Thu thập được {artist_bsides_count} bài B-SIDE hợp lệ (Giới hạn {max_bsides}).")
                    tqdm.write(f"   -> [Filter Summary] Loại {artist_chart_song} (On_Chart), {artist_junk} (non_official/Short_Junk), {artist_short} (Duration < {MIN_DURATION_MINUTES}p), {artist_old} (<{min_year}), {artist_zero_duration} (0:00/Lỗi).")
                    
                    total_junk_filtered += artist_junk
                    total_short_filtered += artist_short
                    total_old_filtered += artist_old
                    total_chart_song_filtered += artist_chart_song
                    total_zero_duration_filtered += artist_zero_duration 
                
                # --- GHI LOG NGHỆ SĨ ĐÃ XỬ LÝ ---
                log_processed_artist(log_file, artist_name)
                
                tqdm.write(f"   -> Đã xử lý xong {artist_name}. Ghi log thành công. Nghỉ {delay} giây...")
                time.sleep(delay)
                break 

            except spotipy.exceptions.SpotifyException as e:
                # Xử lý Rate Limit
                if e.http_status == 429:
                    retry_after_seconds = int(e.headers.get('Retry-After', 3600)) + 60
                    tqdm.write(f"   -> ‼️ BỊ CHẶN (Rate Limit). Tự động 'ngủ' {retry_after_seconds} giây...")
                    artist_loop.set_description(f"Bị chặn 429, ngủ {retry_after_seconds}s")
                    time.sleep(retry_after_seconds)
                    tqdm.write(f"   -> Đã 'ngủ' xong. Thử lại với nghệ sĩ {artist_name}...")
                else:
                    tqdm.write(f"   -> ❌ Lỗi Spotify (không phải 429) với {artist_name}: {e}. Bỏ qua nghệ sĩ này.")
                    log_processed_artist(log_file, artist_name)
                    break 
            
            except Exception as e:
                tqdm.write(f"   -> ❌ Lỗi NGHIÊM TRỌNG (Python) với {artist_name}: {e}. Bỏ qua nghệ sĩ này.")
                log_processed_artist(log_file, artist_name)
                break
            
    print(f"\n--- 📊 TỔNG KẾT THU THẬP B-SIDES (LẦN CHẠY NÀY) ---")
    print(f"   Tổng cộng thu thập được {len(newly_fetched_bsides)} bài B-SIDE MỚI hợp lệ.")
    
    return newly_fetched_bsides

def save_bsides_data(new_bsides_list, output_file):
    # ... (Giữ nguyên)
    print(f"\n--- 💾 LƯU FILE B-SIDES ---")
    
    try:
        df_existing = pd.read_csv(output_file)
        print(f"   -> Đã tải {len(df_existing)} B-Sides từ file cũ: {output_file}")
    except FileNotFoundError:
        df_existing = pd.DataFrame()
        print("   -> Không tìm thấy file B-Sides cũ. Sẽ tạo file mới.")

    df_new = pd.DataFrame(new_bsides_list)
    df_final = pd.concat([df_existing, df_new], ignore_index=True)
    
    original_count = len(df_final)
    
    df_final = df_final.drop_duplicates(subset=['artist_name', 'track_name'], keep='last')
    final_count = len(df_final)
    
    if original_count > final_count:
        print(f"   -> Đã loại bỏ {original_count - final_count} bài B-Side trùng lặp.")

    if 'duration_ms' in df_final.columns:
        print("   -> Đang định dạng lại 'duration_ms' -> 'duration_formatted' (M:SS)...")
        
        safe_duration_ms = df_final['duration_ms'].fillna(0)
        safe_duration_ms = safe_duration_ms.replace([np.inf, -np.inf], 0)
        
        total_seconds = safe_duration_ms / 1000
        
        minutes = (total_seconds // 60).astype(int)
        seconds = (total_seconds % 60).astype(int)
        df_final['duration_formatted'] = minutes.astype(str) + ':' + seconds.astype(str).str.zfill(2)
        df_final = df_final.drop(columns=['duration_ms'])
    
    if 'spotify_popularity' in df_final.columns:
        print("   -> Đang định dạng lại 'spotify_popularity' -> (Số nguyên)...")
        
        safe_popularity = df_final['spotify_popularity'].fillna(0)
        safe_popularity = safe_popularity.replace([np.inf, -np.inf], 0)
        
        df_final['spotify_popularity'] = safe_popularity.astype('Int64')

    print("   -> Đang sắp xếp file theo Tên nghệ sĩ (artist_name)...")
    df_final = df_final.sort_values(by=['artist_name', 'track_name'], ascending=True)
    
    df_final.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print(f"\n--- ✅ HOÀN TẤT ---")
    print(f"Đã lưu tổng cộng {len(df_final)} bài B-Side vào file: {output_file}")

# =============================================================================
# --- BƯỚC 4: CHƯƠNG TRÌNH CHÍNH (Đã tối ưu nguồn dữ liệu) ---
# =============================================================================
def main():
    sp = setup_spotipy(YOUR_CLIENT_ID, YOUR_CLIENT_SECRET)
    
    if sp:
        # 1. Xác định Nguồn Danh sách Nghệ sĩ 
        if ARTISTS_MANUAL_LIST:
            # Nếu có danh sách thủ công: chỉ lấy từ đó.
            all_artists_set = set(str(a).lower().strip() for a in ARTISTS_MANUAL_LIST)
            print(f"⭐ Đang sử dụng danh sách nghệ sĩ thủ công: {len(all_artists_set)} nghệ sĩ.")
        else:
            # NẾU TRỐNG: Lấy danh sách Hit Makers từ is_hit.csv
            all_artists_set = get_hit_makers(HIT_MAKER_SOURCE_FILE)
            if not all_artists_set:
                print("❌ Không tìm thấy nghệ sĩ Hit Makers nào để xử lý. Vui lòng kiểm tra file is_hit.csv. Dừng.")
                return

        # 2. (KIỂM TRA) Đọc file LOG để xem đã xử lý những ai
        processed_artists_set = get_processed_artists(PROCESSED_ARTISTS_LOG)
        print(f"\n✅ Đã tìm thấy {len(processed_artists_set)} nghệ sĩ đã xử lý hoàn chỉnh trong file log: {PROCESSED_ARTISTS_LOG}.")

        # 3. (LỌC) Chỉ lấy những nghệ sĩ CHƯA ĐƯỢC XỬ LÝ
        artists_to_process_set = all_artists_set - processed_artists_set
        print(f"  -> {len(all_artists_set)} (Tổng danh sách) - {len(processed_artists_set)} (Đã xử lý) = {len(artists_to_process_set)} nghệ sĩ MỚI cần xử lý.")

        # 4. Giới hạn MAX_ARTISTS_PER_RUN nghệ sĩ MỚI cho lần chạy này
        artists_to_process_list = list(artists_to_process_set)[:MAX_ARTISTS_PER_RUN]
        
        if not artists_to_process_list:
            print(f"\n🎉 HẾT NGHỆ SĨ MỚI ĐỂ XỬ LÝ trong danh sách. Dừng chương trình.")
            return
            
        print(f"  -> Bắt đầu xử lý {len(artists_to_process_list)} nghệ sĩ (Giới hạn: {MAX_ARTISTS_PER_RUN}).")
            
        # 5. Tải các bài hát "ĐÃ BIẾT" (Để lọc B-Sides)
        known_songs_set = get_known_chart_songs(HIT_MAKER_SOURCE_FILE)

        # 6. Chạy chức năng thu thập B-SIDES
        new_bsides_list = fetch_bsides(
            sp, 
            artists_to_process_list, 
            known_songs_set,
            MIN_RELEASE_YEAR, 
            MIN_DURATION_MS, 
            JUNK_KEYWORDS,
            SHORT_JUNK_KEYWORDS,
            SHORT_JUNK_DURATION_MS,
            DELAY_PER_ARTIST,
            MAX_BSIDES_PER_ARTIST,
            PROCESSED_ARTISTS_LOG
        )
        
        # 7. Gộp và lưu
        save_bsides_data(new_bsides_list, BSIDES_OUTPUT_FILE)

if __name__ == "__main__":
    main()