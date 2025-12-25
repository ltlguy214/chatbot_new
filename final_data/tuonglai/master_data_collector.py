import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import time
import os
import sys

# =============================================================================
# 1. CẤU HÌNH ĐƯỜNG DẪN (CHỈNH SỬA TẠI ĐÂY)
# =============================================================================
CLIENT_ID = "ea1dc0f8dc364215a097ff9c8892f453"
CLIENT_SECRET = "8d01c6315fe1418cb66662b035ed01bd"

# --- CẤU HÌNH THƯ MỤC ---
# Dùng ".." để lùi lại một cấp thư mục nếu script nằm trong folder con
# Hoặc điền đường dẫn tuyệt đối: r"C:\Users\Name\Project\data"
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # Thư mục chứa script này
DATA_DIR = os.path.join(BASE_DIR, "..", "data")       # Trỏ ra ngoài vào folder data

# File "Cơ sở dữ liệu" chính (Sẽ được đọc và cập nhật liên tục)
DB_ARTISTS_FILE = os.path.join(DATA_DIR, "artists_info.csv")
DB_TRACKS_FILE = os.path.join(DATA_DIR, "raw_vpop_collection.csv")

# Các file nguồn cũ để quét tìm ID mới (Input sources)
SOURCE_FILES = [
    os.path.join(DATA_DIR, "bsides_non_hit.csv"),
    os.path.join(DATA_DIR, "bsides_non_hit_old.csv"),
    os.path.join(DATA_DIR, "new_hits_scraped.csv"),
    os.path.join(DATA_DIR, "song_list_info.csv"),
    os.path.join(DATA_DIR, "vpop_master_metadata.csv")
]

# Cờ kiểm soát (Để tiết kiệm API)
SKIP_EXISTING_ARTISTS_INFO = True  # Nếu artist đã có trong artists_info.csv -> Bỏ qua
SKIP_EXISTING_TRACKS_SCAN = True   # Nếu artist đã có bài trong tracks.csv -> Bỏ qua không quét lại

# =============================================================================
# 2. HÀM HỖ TRỢ
# =============================================================================

def setup_spotify():
    """Kết nối API"""
    try:
        auth = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
        return spotipy.Spotify(auth_manager=auth)
    except Exception as e:
        print(f"❌ Lỗi API Key: {e}")
        sys.exit()

def load_csv_safe(filepath):
    """Đọc file CSV an toàn, trả về DataFrame rỗng nếu file chưa có"""
    if os.path.exists(filepath):
        try:
            return pd.read_csv(filepath)
        except: return pd.DataFrame()
    return pd.DataFrame()

def get_all_source_artist_ids():
    """Quét tất cả file nguồn để lấy danh sách Artist ID tiềm năng"""
    ids = set()
    print("\n🔍 Đang quét ID từ các file nguồn...")
    for f_path in SOURCE_FILES:
        if os.path.exists(f_path):
            try:
                df = pd.read_csv(f_path)
                # Tìm cột chứa ID artist
                cols = [c for c in df.columns if 'artist_id' in c or 'primary_artist_id' in c]
                if cols:
                    current_ids = df[cols[0]].dropna().astype(str).unique().tolist()
                    valid_ids = [i for i in current_ids if len(i) == 22] # ID chuẩn Spotify dài 22 ký tự
                    ids.update(valid_ids)
                    print(f"   - {os.path.basename(f_path)}: Tìm thấy {len(valid_ids)} ID.")
            except Exception as e:
                print(f"   ⚠️ Lỗi đọc {os.path.basename(f_path)}: {e}")
        else:
            print(f"   ⚠️ Không tìm thấy file nguồn: {f_path}")
    return ids

# =============================================================================
# 3. LOGIC CHÍNH
# =============================================================================

def main():
    sp = setup_spotify()
    
    # --- BƯỚC 1: CẬP NHẬT DATABASE NGHỆ SĨ (artists_info.csv) ---
    print(f"\n{'='*60}")
    print("PHẦN 1: CẬP NHẬT THÔNG TIN NGHỆ SĨ (ARTISTS INFO)")
    print(f"{'='*60}")
    
    # 1.1 Load dữ liệu đã có
    df_artists_db = load_csv_safe(DB_ARTISTS_FILE)
    existing_ids = set()
    if not df_artists_db.empty and 'artist_id' in df_artists_db.columns:
        existing_ids = set(df_artists_db['artist_id'].unique().tolist())
        print(f"📚 Đã có sẵn thông tin của {len(existing_ids)} nghệ sĩ trong kho.")

    # 1.2 Tìm ID mới từ nguồn
    source_ids = get_all_source_artist_ids()
    
    # 1.3 Lọc ra những ID chưa có
    if SKIP_EXISTING_ARTISTS_INFO:
        new_ids_to_fetch = list(source_ids - existing_ids)
    else:
        new_ids_to_fetch = list(source_ids) # Fetch hết (update lại)

    print(f"👉 Cần lấy thông tin thêm cho: {len(new_ids_to_fetch)} nghệ sĩ mới.")

    # 1.4 Gọi API lấy thông tin (Batch)
    if new_ids_to_fetch:
        new_artist_data = []
        batch_size = 50
        for i in range(0, len(new_ids_to_fetch), batch_size):
            chunk = new_ids_to_fetch[i:i+batch_size]
            print(f"   ⏳ Đang tải batch {i//batch_size + 1}...")
            try:
                full_artists = sp.artists(chunk)['artists']
                for a in full_artists:
                    if a:
                        new_artist_data.append({
                            'artist_id': a['id'],
                            'artist_name': a['name'],
                            'total_followers': a['followers']['total'],
                            'popularity': a['popularity'],
                            'genres': ", ".join(a['genres']) if a['genres'] else "unknown",
                            'image_url': a['images'][0]['url'] if a['images'] else ""
                        })
            except Exception as e:
                print(f"   ⚠️ Lỗi API: {e}")
            time.sleep(0.2)
        
        # 1.5 Lưu cập nhật
        if new_artist_data:
            df_new_artists = pd.DataFrame(new_artist_data)
            # Gộp với cũ và lưu đè
            df_final_artists = pd.concat([df_artists_db, df_new_artists]).drop_duplicates(subset=['artist_id'], keep='last')
            os.makedirs(os.path.dirname(DB_ARTISTS_FILE), exist_ok=True)
            df_final_artists.to_csv(DB_ARTISTS_FILE, index=False, encoding='utf-8-sig')
            print(f"✅ Đã cập nhật xong artists_info.csv (Tổng: {len(df_final_artists)})")
            # Cập nhật lại biến df_artists_db để dùng cho bước sau
            df_artists_db = df_final_artists
    else:
        print("✅ Dữ liệu nghệ sĩ đã đầy đủ, không cần tải thêm.")

    # --- BƯỚC 2: QUÉT SÂU BÀI HÁT (DEEP SCAN TRACKS) ---
    print(f"\n{'='*60}")
    print("PHẦN 2: QUÉT BÀI HÁT CỦA NGHỆ SĨ (DEEP SCAN)")
    print(f"{'='*60}")

    # 2.1 Load Tracks đã có
    df_tracks_db = load_csv_safe(DB_TRACKS_FILE)
    scanned_artist_ids = set()
    
    # Kiểm tra xem những nghệ sĩ nào ĐÃ ĐƯỢC quét bài hát rồi
    if not df_tracks_db.empty:
        # Kiểm tra cột ID nghệ sĩ trong file track (có thể tên là 'primary_artist_id' hoặc 'artist_id')
        col_check = 'primary_artist_id' if 'primary_artist_id' in df_tracks_db.columns else 'artist_id'
        if col_check in df_tracks_db.columns:
            scanned_artist_ids = set(df_tracks_db[col_check].dropna().unique().tolist())
            print(f"📚 Trong kho track đã có bài hát của {len(scanned_artist_ids)} nghệ sĩ.")

    # 2.2 Lập danh sách nghệ sĩ cần quét
    # Chuyển df_artists_db thành list dict để duyệt
    target_artists = df_artists_db.to_dict('records')
    
    artists_to_scan = []
    for art in target_artists:
        if SKIP_EXISTING_TRACKS_SCAN and art['artist_id'] in scanned_artist_ids:
            continue # Bỏ qua nếu đã quét rồi
        artists_to_scan.append(art)
        
    print(f"👉 Số nghệ sĩ cần quét bài hát: {len(artists_to_scan)} (Tiết kiệm được: {len(scanned_artist_ids)} người)")

    # 2.3 Thực hiện quét (Chỉ quét người chưa có)
    if not artists_to_scan:
        print("🏁 Không có nghệ sĩ mới để quét bài hát. Hoàn tất!")
        return

    all_new_tracks = []
    
    for idx, artist in enumerate(artists_to_scan):
        a_id = artist['artist_id']
        a_name = artist['artist_name']
        print(f"   [{idx+1}/{len(artists_to_scan)}] Đang quét: {a_name}...")
        
        seen_tracks_in_session = set()
        
        try:
            # Lấy Albums + Singles
            albums = sp.artist_albums(a_id, album_type='album,single', limit=50)
            items = albums['items']
            # Phân trang lấy hết album
            while albums['next'] and len(items) < 100: 
                albums = sp.next(albums)
                items.extend(albums['items'])
            
            for alb in items:
                # Gọi track trong album
                tracks = sp.album_tracks(alb['id'])['items']
                for t in tracks:
                    # Logic lọc trùng cơ bản
                    if t['id'] not in seen_tracks_in_session:
                        all_new_tracks.append({
                            'track_id': t['id'],
                            'title': t['name'],
                            'artists': a_name, 
                            'primary_artist_id': a_id,
                            'album_name': alb['name'],
                            'spotify_release_date': alb['release_date'],
                            'album_type': alb['album_type'],
                            'duration_ms': t['duration_ms'],
                            'is_explicit': t['explicit'],
                            # Điền luôn thông tin Artist vào đây (Denormalize) để tiện train
                            'artist_followers': artist.get('total_followers', 0),
                            'artist_genres': artist.get('genres', 'unknown'),
                            'artist_popularity': artist.get('popularity', 0)
                        })
                        seen_tracks_in_session.add(t['id'])
                time.sleep(0.05) # Rate limit
                
        except Exception as e:
            print(f"      ❌ Lỗi quét {a_name}: {e}")
            
        # Lưu cuốn chiếu (Incremental Save) mỗi khi quét xong 5 nghệ sĩ để tránh mất điện/lỗi mạng
        if len(all_new_tracks) > 200:
             _save_tracks_incrementally(all_new_tracks, sp)
             all_new_tracks = [] # Reset buffer

    # Lưu nốt phần còn lại
    if all_new_tracks:
        _save_tracks_incrementally(all_new_tracks, sp)

def _save_tracks_incrementally(new_tracks_list, sp):
    """Hàm phụ: Lấy Popularity, Merge vào DB và Lưu"""
    if not new_tracks_list: return
    
    print(f"      💾 Đang lưu {len(new_tracks_list)} bài hát mới vào kho...")
    
    # 1. Lấy track popularity (Batch 50)
    # Vì album_tracks không trả về popularity của bài hát
    track_ids = [t['track_id'] for t in new_tracks_list]
    pop_map = {}
    for i in range(0, len(track_ids), 50):
        chunk = track_ids[i:i+50]
        try:
            infos = sp.tracks(chunk)['tracks']
            for info in infos:
                if info: pop_map[info['id']] = info['popularity']
        except: pass
    
    # 2. Cập nhật popularity vào list
    for t in new_tracks_list:
        t['spotify_popularity'] = pop_map.get(t['track_id'], 0)
        
    # 3. Load DB cũ, nối thêm và lưu
    df_new = pd.DataFrame(new_tracks_list)
    
    if os.path.exists(DB_TRACKS_FILE):
        df_old = pd.read_csv(DB_TRACKS_FILE)
        # Nối và loại trùng theo track_id
        df_final = pd.concat([df_old, df_new]).drop_duplicates(subset=['track_id'], keep='last')
    else:
        df_final = df_new
        
    df_final.to_csv(DB_TRACKS_FILE, index=False, encoding='utf-8-sig')
    print(f"      ✅ Đã lưu! Tổng số bài trong kho: {len(df_final)}")

if __name__ == "__main__":
    # Đảm bảo thư mục tồn tại
    os.makedirs(DATA_DIR, exist_ok=True)
    main()