import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd

YOUR_CLIENT_ID = "c0bbdce2760a4ada8fa309691e8efdaf"
YOUR_CLIENT_SECRET = "a584b443859a403689f3a242e470737a"

client_credentials_manager = SpotifyClientCredentials(client_id=YOUR_CLIENT_ID, client_secret=YOUR_CLIENT_SECRET)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# --- BƯỚC 1: ĐỊNH NGHĨA THỂ LOẠI NHẠC VIỆT ---
# Đây là các từ khóa để xác định nghệ sĩ Việt Nam
# Bạn có thể thêm các thể loại khác vào đây (ví dụ: 'nhac tre', 'bolero')
VIETNAMESE_GENRES = [
    'v-pop',
    'vietnamese hip hop',
    'vietnam indie',
    'vietnamese lo-fi',
    'vinahouse'
]

print("Đang lấy 50 bản phát hành mới nhất tại Việt Nam...")

try:
    # --- BƯỚC 2: LẤY NEW RELEASES VÀ CÁC ARTIST ID ---
    response = sp.new_releases(country='VN', limit=50)
    albums = response['albums']['items']
    
    # Thu thập tất cả các artist ID duy nhất từ 50 bản phát hành
    artist_ids_to_check = set()
    for item in albums:
        if item['artists']:
            # Lấy ID của nghệ sĩ chính
            artist_ids_to_check.add(item['artists'][0]['id'])

    # Chuyển sang dạng list để gọi API
    artist_ids_list = list(artist_ids_to_check)
    
    # --- BƯỚC 3: LẤY THÔNG TIN GENRE CỦA TẤT CẢ NGHỆ SĨ (Chỉ 1 lần gọi API) ---
    print(f"Đang kiểm tra thể loại (genre) của {len(artist_ids_list)} nghệ sĩ...")
    
    # Tạo một "bảng tra cứu" (dictionary) để map artist_id -> genres
    artist_genre_map = {}
    
    if artist_ids_list:
        # sp.artists() cho phép lấy thông tin của 50 nghệ sĩ cùng lúc
        artists_info = sp.artists(artists=artist_ids_list)
        for artist in artists_info['artists']:
            artist_genre_map[artist['id']] = artist['genres']

    # --- BƯỚC 4: LỌC DANH SÁCH GỐC ĐỂ CHỈ LẤY NHẠC VIỆT ---
    vietnamese_new_releases = []
    
    for item in albums:
        # Lấy artist ID của album/single này
        artist_id = item['artists'][0]['id'] if item['artists'] else None
        
        if artist_id and artist_id in artist_genre_map:
            # Tra cứu thể loại của nghệ sĩ này
            artist_genres = artist_genre_map[artist_id]
            
            # Kiểm tra xem nghệ sĩ này có thuộc thể loại V-Pop không
            is_vietnamese = any(vg in artist_genres for vg in VIETNAMESE_GENRES)
            
            if is_vietnamese:
                # Nếu đúng là V-Pop, thêm vào danh sách kết quả
                vietnamese_new_releases.append({
                    'release_name': item['name'],
                    'artist_name': item['artists'][0]['name'],
                    'release_date': item['release_date'],
                    'release_type': item['album_type'],
                    'track_id_or_album_id': item['id'],
                    'artist_genres': ", ".join(artist_genres) # Thêm genre để kiểm tra
                })

    # --- BƯỚC 5: HIỂN THỊ KẾT QUẢ ---
    print(f"\n--- TÌM THẤY {len(vietnamese_new_releases)} BẢN PHÁT HÀNH MỚI CỦA VIỆT NAM ---")
    
    if vietnamese_new_releases:
        df_vpop = pd.DataFrame(vietnamese_new_releases)
        
        # In 10 kết quả đầu tiên
        print(df_vpop.head(10).to_markdown(index=False, numalign="left", stralign="left"))
        
        # Lưu lại file CSV chỉ chứa nhạc Việt
        output_file = 'new_releases_VPOP_only.csv'
        df_vpop.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\nĐã lưu file kết quả vào: {output_file}")
    else:
        print("Không tìm thấy bản phát hành V-Pop mới nào trong 50 kết quả hàng đầu.")

except Exception as e:
    print(f"Lỗi: {e}")