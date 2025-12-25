import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import time
import os
import sys

# =============================================================================
# CẤU HÌNH API
# =============================================================================
CLIENT_ID = "f531846ca30d4dbe8f67c5d2b07f2eca" 
CLIENT_SECRET = "806be57341b240369ff885eadf54367b"

# File đầu ra
OUTPUT_FILE = "data/artists_info.csv"

# Import danh sách nghệ sĩ từ file bạn vừa tạo
try:
    from TARGET_ARTISTS_LIST import TARGET_ARTISTS
except ImportError:
    print("⚠️ Lỗi: Không tìm thấy file TARGET_ARTISTS_LIST.py")
    print("Hãy đảm bảo file này nằm cùng thư mục với script.")
    sys.exit()

# =============================================================================
# HÀM XỬ LÝ
# =============================================================================

def setup_spotify():
    auth_manager = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
    return spotipy.Spotify(auth_manager=auth_manager)

def fetch_artists_batch(sp, artist_ids):
    """
    Gọi API lấy thông tin 50 nghệ sĩ cùng lúc (Batch request)
    """
    artists_data = []
    try:
        # Spotify cho phép tối đa 50 ID mỗi lần gọi
        response = sp.artists(artist_ids)
        
        for artist in response['artists']:
            if artist:
                # Xử lý Genre: Chuyển list thành chuỗi
                genres = ", ".join(artist['genres']) if artist['genres'] else "unknown"
                
                # Xử lý Image: Lấy ảnh đại diện (nếu có)
                image_url = artist['images'][0]['url'] if artist['images'] else None
                
                artists_data.append({
                    'artist_id': artist['id'],
                    'artist_name': artist['name'],
                    'total_followers': artist['followers']['total'],
                    'popularity': artist['popularity'],
                    'genres': genres,
                    'spotify_url': artist['external_urls']['spotify'],
                    'image_url': image_url
                })
    except Exception as e:
        print(f"   ⚠️ Lỗi batch: {e}")
        
    return artists_data

def main():
    sp = setup_spotify()
    all_artist_info = []
    
    # Loại bỏ ID trùng lặp (nếu có) trong list đầu vào
    unique_ids = list(set(TARGET_ARTISTS))
    total_artists = len(unique_ids)
    
    print(f"🚀 BẮT ĐẦU THU THẬP THÔNG TIN {total_artists} NGHỆ SĨ...")
    os.makedirs('data', exist_ok=True)
    
    # Chia nhỏ danh sách thành các batch 50
    batch_size = 50
    for i in range(0, total_artists, batch_size):
        batch_ids = unique_ids[i:i + batch_size]
        
        print(f"   ⏳ Đang xử lý batch {i//batch_size + 1}: {len(batch_ids)} nghệ sĩ...")
        
        batch_data = fetch_artists_batch(sp, batch_ids)
        all_artist_info.extend(batch_data)
        
        time.sleep(0.2) # Nghỉ nhẹ để tránh rate limit

    # Lưu kết quả
    if all_artist_info:
        df = pd.DataFrame(all_artist_info)
        
        # Sắp xếp theo độ nổi tiếng giảm dần
        df = df.sort_values(by='popularity', ascending=False)
        
        df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
        print(f"\n✅ ĐÃ HOÀN TẤT!")
        print(f"   📂 File lưu tại: {OUTPUT_FILE}")
        print(f"   📊 Tổng số nghệ sĩ: {len(df)}")
        print(f"   🌟 Nghệ sĩ hot nhất: {df.iloc[0]['artist_name']} (Pop: {df.iloc[0]['popularity']})")
    else:
        print("❌ Không thu thập được dữ liệu nào.")

if __name__ == "__main__":
    main()