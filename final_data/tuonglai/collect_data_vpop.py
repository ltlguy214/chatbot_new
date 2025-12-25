import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import time
import os

# =============================================================================
# CẤU HÌNH API
# =============================================================================
CLIENT_ID = "f531846ca30d4dbe8f67c5d2b07f2eca" 
CLIENT_SECRET = "806be57341b240369ff885eadf54367b"

OUTPUT_FILE = "data/raw_vpop_deep_collection.csv"

# =============================================================================
# DANH SÁCH NGHỆ SĨ MỤC TIÊU (V-POP)
# =============================================================================
# Bạn có thể thêm hàng trăm ID vào đây
TARGET_ARTISTS = [
    "5dfZ5uSmzR7VQK0udbAVpf", # Son Tung M-TP
    "1LEtM3AleYg1x72ffgyfMk", # Den
    "2FS9w948Pdb3h3b9aF44aT", # My Tam
    "4d76982f64794c4897f1",   # Hoang Thuy Linh
    "5d115e01691244039169",   # Binz
    "5752c4238b724f119069",   # Chillies
    "20197722784547948218",   # Vu.
    "45592592394042238479",   # JustaTee
    "12330722234044199419",   # Min
    "75184638774747209930",   # Erik
    "6eU0j37D7h23709n3n0062", # Duc Phuc
    "4fl3187243977466632971", # Soobin
    "1n1Q6r3C5b644059088510", # HIEUTHUHAI
    "2SfW4h68f8696229505536", # Tang Duy Tan
    "5lZ0476k27743519803023", # Mono
    # ... Thêm bất kỳ ai bạn muốn
]

# =============================================================================
# HÀM XỬ LÝ CHÍNH
# =============================================================================

def setup_spotify():
    auth_manager = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
    return spotipy.Spotify(auth_manager=auth_manager)

def get_artist_metadata(sp, artist_id):
    """Lấy thông tin chung của Artist (Followers, Genres) một lần"""
    try:
        artist_info = sp.artist(artist_id)
        return {
            'artist_name': artist_info['name'],
            'artist_followers': artist_info['followers']['total'],
            'artist_popularity': artist_info['popularity'],
            'artist_genres': ", ".join(artist_info['genres']) if artist_info['genres'] else "unknown"
        }
    except:
        return None

def get_all_tracks_of_artist(sp, artist_id, artist_meta, seen_ids):
    """
    Quét sạch sành sanh Album và Single của Artist
    """
    tracks_data = []
    
    # 1. Lấy danh sách Albums & Singles
    try:
        # include_groups: lấy cả album, đĩa đơn, và bài hát xuất hiện trong album khác
        albums = sp.artist_albums(artist_id, album_type='album,single', limit=50)
        album_items = albums['items']
        
        # Nếu nghệ sĩ ra nhiều album, phải phân trang lấy tiếp
        while albums['next']:
            albums = sp.next(albums)
            album_items.extend(albums['items'])
            
        print(f"   --> Tìm thấy {len(album_items)} Albums/Singles.")

        # 2. Duyệt từng Album để lấy bài hát
        for album in album_items:
            album_id = album['id']
            album_name = album['name']
            album_date = album['release_date']
            
            # Lấy tracks trong album này
            results = sp.album_tracks(album_id)
            tracks = results['items']
            
            for track in tracks:
                t_id = track['id']
                
                # Bỏ qua nếu đã lấy rồi (tránh trùng bài giữa Single và Album)
                if t_id in seen_ids:
                    continue
                
                # Lưu thông tin
                tracks_data.append({
                    'track_id': t_id,
                    'title': track['name'],
                    'artists': artist_meta['artist_name'], # Tác giả chính
                    'primary_artist_id': artist_id,
                    
                    # Metadata bài hát
                    # Lưu ý: album_tracks không trả về popularity của track, ta phải chấp nhận
                    # hoặc gọi thêm API (tốn quota). Ở đây ta tạm để popularity=0 hoặc gọi batch sau.
                    # Cách tối ưu: Gọi track details cho danh sách track_id sau cùng.
                    'spotify_release_date': album_date,
                    'duration_ms': track['duration_ms'],
                    'is_explicit': track['explicit'],
                    'album_type': album['album_type'],
                    
                    # Metadata Artist (Dùng chung cho tất cả bài của ca sĩ này)
                    'artist_followers': artist_meta['artist_followers'],
                    'artist_popularity': artist_meta['artist_popularity'],
                    'artist_genres': artist_meta['artist_genres'],
                    
                    'source': 'artist_deep_scan'
                })
                seen_ids.add(t_id)
                
            time.sleep(0.1) # Nghỉ cực ngắn để tránh lỗi API
            
    except Exception as e:
        print(f"   ⚠️ Lỗi khi quét artist {artist_id}: {e}")
        
    return tracks_data

def update_tracks_popularity(sp, df_tracks):
    """
    Hàm phụ: Cập nhật Popularity cho các bài hát vừa quét
    (Vì endpoint album_tracks không trả về track popularity)
    """
    print(f"\n🔄 Đang cập nhật Popularity cho {len(df_tracks)} bài hát...")
    
    # Spotify cho phép lấy info tối đa 50 bài/lần
    track_ids = df_tracks['track_id'].tolist()
    pop_map = {}
    
    # Chia thành các batch 50 bài
    batch_size = 50
    for i in range(0, len(track_ids), batch_size):
        batch = track_ids[i:i + batch_size]
        try:
            tracks_info = sp.tracks(batch)
            for t in tracks_info['tracks']:
                if t:
                    pop_map[t['id']] = t['popularity']
        except Exception as e:
            print(f"   Lỗi update batch {i}: {e}")
        time.sleep(0.2)
        
    # Map ngược lại vào DataFrame
    df_tracks['spotify_popularity'] = df_tracks['track_id'].map(pop_map)
    return df_tracks

def main():
    sp = setup_spotify()
    all_collection = []
    seen_ids = set()
    
    print(f"🚀 BẮT ĐẦU QUÉT SÂU (DEEP SCAN) {len(TARGET_ARTISTS)} NGHỆ SĨ...")
    os.makedirs('data', exist_ok=True)
    
    # Nếu file cũ tồn tại, đọc ID cũ để tránh trùng
    if os.path.exists(OUTPUT_FILE):
        try:
            df_old = pd.read_csv(OUTPUT_FILE)
            seen_ids = set(df_old['track_id'].astype(str).tolist())
            print(f"📚 Đã load {len(seen_ids)} bài từ dữ liệu cũ.")
        except: pass

    for artist_id in TARGET_ARTISTS:
        # 1. Lấy thông tin Artist trước
        meta = get_artist_metadata(sp, artist_id)
        if not meta:
            print(f"❌ Không tìm thấy artist {artist_id}")
            continue
            
        print(f"\n🎤 Đang quét: {meta['artist_name']}...")
        
        # 2. Lấy toàn bộ bài hát
        new_tracks = get_all_tracks_of_artist(sp, artist_id, meta, seen_ids)
        if new_tracks:
            all_collection.extend(new_tracks)
            print(f"   ✅ Thêm được {len(new_tracks)} bài.")
        else:
            print("   💤 Không có bài mới.")
            
    # 3. Cập nhật Popularity và Lưu
    if all_collection:
        df_new = pd.DataFrame(all_collection)
        
        # Gọi API để lấy Popularity chính xác cho từng bài
        df_new = update_tracks_popularity(sp, df_new)
        
        if os.path.exists(OUTPUT_FILE):
            df_new.to_csv(OUTPUT_FILE, mode='a', header=False, index=False, encoding='utf-8-sig')
        else:
            df_new.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
            
        print(f"\n🎉 HOÀN TẤT! Đã lưu thêm {len(df_new)} bài vào {OUTPUT_FILE}")
    else:
        print("\n🏁 Hoàn tất. Không có dữ liệu mới.")

if __name__ == "__main__":
    main()