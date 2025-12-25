import pandas as pd
import os

# Đọc dữ liệu
song_df = pd.read_csv('data/song_list_info.csv')

# Lấy danh sách artist từ cả 'artists' và 'featured_artists'
artist_rows = []
for idx, row in song_df.iterrows():
    # Lấy artist chính
    main_artist = str(row['artists']).strip()
    if main_artist and main_artist.lower() != 'nan':
        artist_rows.append({'artist': main_artist, 'type': 'main', 'song_title': row['title']})
    # Lấy featured artist (có thể nhiều, phân tách bằng dấu phẩy)
    feat = str(row['featured_artists']).strip()
    if feat and feat.lower() != 'nan':
        for f in feat.split(','):
            f = f.strip()
            if f:
                artist_rows.append({'artist': f, 'type': 'featured', 'song_title': row['title']})

# Đọc artist_id từ các file khác nếu có
artist_id_map = {}
# Ví dụ: nếu có file data/artist_id_map.csv với cột 'artist','artist_id'
artist_id_file = 'data/artist_id_map.csv'
if os.path.exists(artist_id_file):
    id_df = pd.read_csv(artist_id_file)
    for _, r in id_df.iterrows():
        artist_id_map[str(r['artist']).strip()] = r['artist_id']

# Gán artist_id nếu có
for row in artist_rows:
    row['artist_id'] = artist_id_map.get(row['artist'], '')

# Xuất file
artist_df = pd.DataFrame(artist_rows)
artist_df = artist_df.drop_duplicates()
artist_df.to_csv('data/artist_info_extracted.csv', index=False, encoding='utf-8-sig')
print('Đã xuất file data/artist_info_extracted.csv!')
