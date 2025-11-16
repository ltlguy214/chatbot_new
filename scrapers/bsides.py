import spotipy
import spotipy.exceptions
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import time
import re
import os
import sys
import io

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# =============================================================================
# --- BƯỚC 1: THIẾT LẬP CHUNG ---
# =============================================================================
YOUR_CLIENT_ID = "f531846ca30d4dbe8f67c5d2b07f2eca" 
YOUR_CLIENT_SECRET = "a0d355d3c2344b45ac62c106a0382927" 

# Nguồn Input: File kết quả từ is_hit.py
HIT_MAKER_SOURCE_FILE = 'data/is_hit.csv'

# Tên file Output cho các bài B-Sides
BSIDES_OUTPUT_FILE = 'data/bsides_non_hit.csv'

MIN_RELEASE_YEAR = 2025
MIN_DURATION_MINUTES = 1
MIN_DURATION_MS = MIN_DURATION_MINUTES * 60 * 1000 

JUNK_KEYWORDS = [
    'remix', 'live', 'acoustic', 'instrumental', 'interlude', 
    'intro', 'outro', 'skit', 'version', 'edit',
    'deluxe', 'anniversary', 're-release', '(commentary)' 
]

DELAY_PER_ARTIST = 30 # (seconds) Độ trễ an toàn giữa các nghệ sĩ

# =============================================================================
# --- BƯỚC 2: CÁC HÀM TIỆN ÍCH (Copy từ is_hit.py) ---
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

# =============================================================================
# --- BƯỚC 3: CÁC HÀM XỬ LÝ  ---
# =============================================================================

def setup_spotipy(client_id, client_secret):
    """Khởi tạo và xác thực Spotipy."""
    if client_id == "ĐIỀN CLIENT ID CỦA BẠN" or client_secret == "ĐIỀN CLIENT SECRET CỦA BẠN": 
        print("LỖI: Bạn chưa điền YOUR_CLIENT_ID và YOUR_CLIENT_SECRET.")
        return None
    try:
        client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
        sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
        sp.search(q='test', limit=1)
        print("✅ Xác thực Spotify thành công!")
        return sp
    except Exception as e:
        print(f"❌ Lỗi xác thực: {e}")
        return None

def get_hit_makers(file_path):
    """
    Đọc file is_hit.csv, tìm các nghệ sĩ có is-hit = 1
    """
    print(f"📂 Đang tải danh sách 'Hit Makers' từ file: {file_path}...")
    try:
        df = pd.read_csv(file_path)
        if 'is-hit' not in df.columns or 'artists' not in df.columns:
            print(f"  -> ⚠️ Bỏ qua: File '{file_path}' thiếu cột 'is-hit' hoặc 'artists'.")
            return set()
        
        # Lọc các nghệ sĩ từ các bài hát LÀ "HIT"
        df_hits = df[df['is-hit'] == 1]
        
        hit_maker_names = set()
        for artist_string in df_hits['artists'].dropna():
            # Xử lý "A, B & C"
            names_comma = [name.strip() for name in artist_string.split(',')]
            for name in names_comma:
                names_ampersand = [n.strip() for n in name.split(' & ')]
                hit_maker_names.update(names_ampersand)
        
        print(f"  -> ✅ Tìm thấy {len(hit_maker_names)} 'Hit Makers' (nghệ sĩ có ít nhất 1 hit).")
        return hit_maker_names
                
    except FileNotFoundError:
        print(f"❌ LỖI: Không tìm thấy file {file_path}. Hãy chạy is_hit.py trước.")
        return set()
    except Exception as e:
        print(f"❌ Lỗi khi đọc file {file_path}: {e}")
        return set()

def get_known_chart_songs(file_path):
    """
    Đọc is_hit.csv và tạo ra một Set (danh sách)
    các "match_key" của tất cả bài hát đã có trong chart (cả hit và non-hit).
    Chúng ta sẽ dùng Set này để lọc, chỉ giữ lại B-Sides.
    """
    print(f"📂 Đang tải danh sách 'Known Songs' (để lọc) từ file: {file_path}...")
    try:
        df = pd.read_csv(file_path)
        if 'title' not in df.columns or 'artists' not in df.columns:
            print(f"  -> ⚠️ Bỏ qua: File '{file_path}' thiếu cột 'title' hoặc 'artists'.")
            return set()
        
        # Tạo match_key cho TẤT CẢ các bài hát trong file is_hit.csv
        df['match_key'] = df.apply(lambda row: create_match_key(row['title'], row['artists']), axis=1)
        known_songs_set = set(df['match_key'].dropna())
        
        print(f"  -> ✅ Đã có {len(known_songs_set)} bài hát trong chart (sẽ bị lọc bỏ nếu trùng).")
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
                    'artist_name': track['artists'][0]['name'], # Chỉ lấy nghệ sĩ chính
                    'artist_id': track['artists'][0]['id'],
                    'release_date': track['album']['release_date'],
                    'spotify_popularity': track['popularity'],
                    'duration_ms': track['duration_ms']
                })
        except Exception as e:
            print(f"  -> ⚠️ Lỗi khi lấy chi tiết track batch {i}: {e}")
        time.sleep(0.1) 
    return all_details

def fetch_bsides(sp_instance, artists_to_fetch, known_songs_set, min_year, min_duration, junk_keywords, delay):
    """
    Cào (crawl) và tự động LỌC B-SIDES
    (Loại bỏ các bài đã có trong 'known_songs_set').
    """
    print(f"\n--- 🎶 BẮT ĐẦU THU THẬP B-SIDES TỪ NGHỆ SĨ ---")
    print(f"  Điều kiện: Phát hành >= {min_year}, Thời lượng >= {MIN_DURATION_MINUTES} phút")
    print(f"  Độ trễ an toàn: {delay} giây/nghệ sĩ")

    newly_fetched_bsides = [] 
    total_junk_filtered, total_short_filtered, total_old_filtered, total_chart_song_filtered = 0, 0, 0, 0

    print(f"  Tổng số nghệ sĩ 'Hit Maker' cần xử lý: {len(artists_to_fetch)}")
    if not artists_to_fetch:
        print("  Không có nghệ sĩ nào để xử lý.")
        return []

    for i, artist_name in enumerate(artists_to_fetch):
        
        while True: # Vòng lặp tự động thử lại (resilience)
            try:
                print(f"\n[ Nghệ sĩ {i+1}/{len(artists_to_fetch)} ] Đang xử lý: {artist_name}")
                
                # 1. Tìm ID nghệ sĩ
                results = sp_instance.search(q=artist_name, limit=1, type='artist')
                if not results['artists']['items']:
                    print(f"  -> ⚠️ Không tìm thấy '{artist_name}' trên Spotify. Bỏ qua.")
                    break 
                artist_id = results['artists']['items'][0]['id']

                # 2. Lấy Albums/Singles
                albums_response = sp_instance.artist_albums(artist_id, album_type='album,single', country='VN', limit=50)
                albums = albums_response['items']
                while albums_response['next']:
                    albums_response = sp_instance.next(albums_response)
                    albums.extend(albums_response['items'])
                album_ids = [album['id'] for album in albums]

                # 3. Lấy Tracks (Chỉ lấy track mà nghệ sĩ là nghệ sĩ chính)
                track_ids_from_artist = set()
                for album_id in album_ids:
                    try:
                        tracks_response = sp_instance.album_tracks(album_id, limit=50)
                        tracks = tracks_response['items']
                        for track in tracks:
                            if track['artists'] and track['artists'][0]['id'] == artist_id:
                                track_ids_from_artist.add(track['id'])
                    except Exception: pass
                    time.sleep(0.05)
                print(f"  -> Tìm thấy {len(track_ids_from_artist)} tracks tổng cộng (full discography).")

                # 4. LỌC THÔNG MINH (Đã thêm lọc B-Sides)
                if track_ids_from_artist:
                    track_details = get_full_track_details(sp_instance, list(track_ids_from_artist))
                    count_valid_bsides = 0
                    artist_junk, artist_short, artist_old, artist_chart_song = 0, 0, 0, 0
                    
                    for track in track_details:
                        track_name_lower = track.get('track_name', '').lower()
                        release_date_str = track.get('release_date', '1990')
                        track_duration = track.get('duration_ms', 0)
                        
                        match = re.match(r'^\d{4}', release_date_str)
                        release_year = int(match.group(0)) if match else 1990
                        
                        track_match_key = create_match_key(track['track_name'], track['artist_name'])

                        # 4 điều kiện lọc
                        is_junk = any(keyword in track_name_lower for keyword in junk_keywords)
                        is_valid_year = (release_year >= min_year)
                        is_valid_duration = (track_duration >= min_duration)
                        is_bside = (track_match_key not in known_songs_set) # <-- LỌC B-SIDES

                        if is_junk: artist_junk += 1
                        elif not is_valid_duration: artist_short += 1
                        elif not is_valid_year: artist_old += 1
                        elif not is_bside: artist_chart_song += 1
                        elif is_valid_year and is_valid_duration and not is_junk and is_bside:
                            newly_fetched_bsides.append(track)
                            count_valid_bsides += 1
                    
                    print(f"  -> [FILTER] Đã loại bỏ {artist_chart_song} bài (vì đã có trong file chart is_hit.csv).")
                    print(f"  -> [FILTER] Đã loại bỏ {artist_junk} bài 'rác' (remix, live...).")
                    print(f"  -> [FILTER] Đã loại bỏ {artist_short} bài quá ngắn (< {MIN_DURATION_MINUTES} phút).")
                    print(f"  -> [FILTER] Đã loại bỏ {artist_old} bài quá cũ (< {min_year}).")
                    print(f"  -> ✅ Thu thập được {count_valid_bsides} bài B-SIDE hợp lệ.")
                    
                    total_junk_filtered += artist_junk
                    total_short_filtered += artist_short
                    total_old_filtered += artist_old
                    total_chart_song_filtered += artist_chart_song
                
                print(f"  -> Đã xử lý xong {artist_name}. Nghỉ {delay} giây...")
                time.sleep(delay)
                break # <-- THOÁT KHỎI VÒNG LẶP THỬ LẠI (while True)

            except spotipy.exceptions.SpotifyException as e:
                if e.http_status == 429:
                    retry_after_seconds = int(e.headers.get('Retry-After', 3600)) + 60
                    print(f"  -> ‼️ BỊ CHẶN (Rate Limit). Tự động 'ngủ' {retry_after_seconds} giây...")
                    time.sleep(retry_after_seconds)
                    print(f"  -> Đã 'ngủ' xong. Thử lại với nghệ sĩ {artist_name}...")
                else:
                    print(f"  -> ❌ Lỗi Spotify (không phải 429) với {artist_name}: {e}. Bỏ qua nghệ sĩ này.")
                    break 
            
            except Exception as e:
                print(f"  -> ❌ Lỗi NGHIÊM TRỌNG (Python) với {artist_name}: {e}. Bỏ qua nghệ sĩ này.")
                break
            
    print(f"\n--- 📊 TỔNG KẾT THU THẬP B-SIDES (LẦN CHẠY NÀY) ---")
    print(f"  Đã lọc (bỏ) {total_chart_song_filtered} bài vì đã có trong chart.")
    print(f"  Đã lọc (bỏ) {total_junk_filtered} bài 'rác'.")
    print(f"  Đã lọc (bỏ) {total_short_filtered} bài quá ngắn.")
    print(f"  Đã lọc (bỏ) {total_old_filtered} bài quá cũ.")
    print(f"  Tổng cộng thu thập được {len(newly_fetched_bsides)} bài B-SIDE MỚI hợp lệ.")
    
    return newly_fetched_bsides

def save_bsides_data(new_bsides_list, output_file):
    """
    (SỬA) Gộp, lọc trùng, đổi 'duration_ms' sang phút, và sắp xếp
    """
    print(f"\n--- 💾 LƯU FILE B-SIDES ---")
    
    # 1. Tải file cũ (nếu có) để GỘP
    try:
        df_existing = pd.read_csv(output_file)
        print(f"  -> Đã tải {len(df_existing)} B-Sides từ file cũ: {output_file}")
    except FileNotFoundError:
        df_existing = pd.DataFrame()
        print("  -> Không tìm thấy file B-Sides cũ. Sẽ tạo file mới.")

    df_new = pd.DataFrame(new_bsides_list)
    
    # 2. Gộp
    df_final = pd.concat([df_existing, df_new], ignore_index=True)
    
    original_count = len(df_final)
    
    # 3. Lọc trùng lặp (Giữ lại bản mới nhất)
    df_final = df_final.drop_duplicates(subset=['artist_name', 'track_name'], keep='last')
    final_count = len(df_final)
    
    if original_count > final_count:
        print(f"  -> Đã loại bỏ {original_count - final_count} bài B-Side trùng lặp.")

    # 4. (MỚI) Chuyển đổi Duration (ms sang M:SS)
    if 'duration_ms' in df_final.columns:
        print("  -> Đang định dạng lại 'duration_ms' -> 'duration_formatted' (M:SS)...")
        # Chuyển ms sang tổng số giây
        total_seconds = df_final['duration_ms'] / 1000
        # Lấy phút (integer)
        minutes = (total_seconds // 60).astype(int)
        # Lấy giây (phần dư)
        seconds = (total_seconds % 60).astype(int)
        # Định dạng thành "M:SS" (ví dụ: 3:05)
        df_final['duration_formatted'] = minutes.astype(str) + ':' + seconds.astype(str).str.zfill(2)
        # Bỏ cột 'duration_ms' cũ
        df_final = df_final.drop(columns=['duration_ms'])

    # 5. (MỚI) Sắp xếp theo Nghệ sĩ
    print("  -> Đang sắp xếp file theo Tên nghệ sĩ (artist_name)...")
    df_final = df_final.sort_values(by=['artist_name', 'track_name'], ascending=True)
    
    # 6. Lưu file
    df_final.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print(f"\n--- ✅ HOÀN TẤT ---")
    print(f"Đã lưu tổng cộng {len(df_final)} bài B-Side vào file: {output_file}")

# =============================================================================
# --- BƯỚC 4: CHƯƠNG TRÌNH CHÍNH (Đã sửa đổi) ---
# =============================================================================
def main():
    sp = setup_spotipy(YOUR_CLIENT_ID, YOUR_CLIENT_SECRET)
    
    if sp:
        # 1. Lấy danh sách nghệ sĩ "HIT MAKERS"
        hit_makers_set = get_hit_makers(HIT_MAKER_SOURCE_FILE)
        if not hit_makers_set:
            print("Không có 'Hit Makers' nào (is-hit=1) trong file. Dừng chương trình.")
            return
            
        # 2. Tải các bài hát "ĐÃ BIẾT" (Để lọc B-Sides)
        known_songs_set = get_known_chart_songs(HIT_MAKER_SOURCE_FILE)

        # 3. Chạy chức năng thu thập B-SIDES
        new_bsides_list = fetch_bsides(
            sp, 
            hit_makers_set,
            known_songs_set,
            MIN_RELEASE_YEAR, 
            MIN_DURATION_MS, 
            JUNK_KEYWORDS,
            DELAY_PER_ARTIST
        )
        
        # 4. Gộp và lưu
        if not new_bsides_list:
            # Nếu không tìm thấy B-Side MỚI nào, chúng ta vẫn nên
            # chạy hàm save để đảm bảo file được sắp xếp và định dạng (format) lại
            print("\nKhông tìm thấy bài B-Side nào mới. Chỉ chạy định dạng file cũ...")
            
        save_bsides_data(new_bsides_list, BSIDES_OUTPUT_FILE)

if __name__ == "__main__":
    main()