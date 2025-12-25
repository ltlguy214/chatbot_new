import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import time
import os

# =============================================================================
# CẤU HÌNH (Điền API Key MỚI nếu cái cũ đang bị limit)
# =============================================================================
CLIENT_ID = "f531846ca30d4dbe8f67c5d2b07f2eca"
CLIENT_SECRET = "806be57341b240369ff885eadf54367b"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# File đích (Kho dữ liệu Spotify mới)
TARGET_FILE = os.path.join(DATA_DIR, "raw_vpop_collection.csv")

# Danh sách file cũ cần vét dữ liệu
OLD_FILES = [
    os.path.join(DATA_DIR, "bsides_non_hit.csv"),
    os.path.join(DATA_DIR, "bsides_non_hit_old.csv"),
    os.path.join(DATA_DIR, "new_hits_scraped.csv"),
    os.path.join(DATA_DIR, "song_list_info.csv"),
    os.path.join(DATA_DIR, "vpop_master_metadata.csv")
]

# =============================================================================
# HÀM XỬ LÝ
# =============================================================================

def setup_spotify():
    auth = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
    return spotipy.Spotify(auth_manager=auth)

def get_track_details_batch(sp, track_ids):
    """Lấy thông tin chi tiết cho danh sách Track ID"""
    tracks_data = []
    # Chia batch 50
    for i in range(0, len(track_ids), 50):
        batch = track_ids[i:i+50]
        try:
            results = sp.tracks(batch)
            for track in results['tracks']:
                if track:
                    # Lấy artist info cơ bản
                    main_artist = track['artists'][0]
                    
                    tracks_data.append({
                        'track_id': track['id'],
                        'title': track['name'],
                        'artists': main_artist['name'],
                        'primary_artist_id': main_artist['id'],
                        'spotify_popularity': track['popularity'],
                        'spotify_release_date': track['album']['release_date'],
                        'album_type': track['album']['album_type'],
                        'duration_ms': track['duration_ms'],
                        'is_explicit': track['explicit'],
                        # Để trống các field artist chi tiết (sẽ update sau hoặc để default)
                        'artist_followers': 0, 
                        'artist_genres': "recovered_from_old_data",
                        'source': "recovered_old_file"
                    })
        except Exception as e:
            print(f"   ⚠️ Lỗi batch: {e}")
        time.sleep(0.1)
    return tracks_data

def main():
    print("🚀 BẮT ĐẦU HỢP NHẤT DỮ LIỆU CŨ VÀO KHO MỚI...")
    sp = setup_spotify()

    # 1. Load danh sách ID đã có trong kho mới
    if os.path.exists(TARGET_FILE):
        df_target = pd.read_csv(TARGET_FILE)
        existing_ids = set(df_target['track_id'].dropna().astype(str).tolist())
        print(f"📚 Kho hiện tại có: {len(existing_ids)} bài.")
    else:
        print("⚠️ Chưa có file kho mới. Sẽ tạo mới từ đầu.")
        df_target = pd.DataFrame()
        existing_ids = set()

    # 2. Quét ID từ các file cũ
    old_ids = set()
    for f_path in OLD_FILES:
        if os.path.exists(f_path):
            try:
                df = pd.read_csv(f_path)
                # Tìm cột ID (có thể tên khác nhau)
                cols = [c for c in df.columns if 'track_id' in c or 'spotify_track_id' in c]
                if cols:
                    ids = df[cols[0]].dropna().astype(str).unique().tolist()
                    clean_ids = [i for i in ids if len(i) == 22]
                    old_ids.update(clean_ids)
                    print(f"   - {os.path.basename(f_path)}: Tìm thấy {len(clean_ids)} ID.")
            except: pass
    
    # 3. Tìm những bài CÓ trong file cũ mà CHƯA CÓ trong kho mới
    missing_ids = list(old_ids - existing_ids)
    print(f"👉 Phát hiện {len(missing_ids)} bài từ dữ liệu cũ chưa được quét vào kho mới.")

    if not missing_ids:
        print("✅ Tuyệt vời! Kho mới đã bao gồm tất cả dữ liệu cũ. Không cần làm gì thêm.")
        return

    # 4. Tải thông tin cho các bài còn thiếu
    print(f"🔄 Đang tải thông tin cho {len(missing_ids)} bài thiếu...")
    recovered_data = get_track_details_batch(sp, missing_ids)
    
    # 5. Gộp và Lưu
    if recovered_data:
        df_recovered = pd.DataFrame(recovered_data)
        
        # Cập nhật thêm thông tin Artist cho các bài này (nếu có trong artists_info.csv)
        # Bước này tùy chọn, giúp data đồng bộ hơn
        artist_file = os.path.join(DATA_DIR, "artists_info.csv")
        if os.path.exists(artist_file):
            df_art = pd.read_csv(artist_file)
            # Map followers và genres vào
            art_dict = df_art.set_index('artist_id')[['total_followers', 'genres']].to_dict('index')
            
            def enrich_artist(row):
                aid = row['primary_artist_id']
                if aid in art_dict:
                    row['artist_followers'] = art_dict[aid]['total_followers']
                    row['artist_genres'] = art_dict[aid]['genres']
                return row
            
            df_recovered = df_recovered.apply(enrich_artist, axis=1)

        # Nối vào file chính
        df_final = pd.concat([df_target, df_recovered]).drop_duplicates(subset=['track_id'])
        df_final.to_csv(TARGET_FILE, index=False, encoding='utf-8-sig')
        
        print(f"\n✅ ĐÃ CỨU THÀNH CÔNG {len(df_recovered)} BÀI HÁT TỪ DỮ LIỆU CŨ!")
        print(f"📊 Tổng số bài trong kho hiện tại: {len(df_final)}")
    else:
        print("⚠️ Không tải được thông tin bài nào (có thể ID cũ bị lỗi/xóa).")

if __name__ == "__main__":
    main()