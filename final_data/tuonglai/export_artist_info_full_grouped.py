import pandas as pd
import os
from collections import defaultdict, Counter

# Đọc dữ liệu
song_df = pd.read_csv('data/song_list_info.csv')

# Gom thông tin nghệ sĩ
artist_info = defaultdict(lambda: {'main_count': 0, 'featured_count': 0, 'songs': set(), 'genres': set(), 'release_dates': set()})

for idx, row in song_df.iterrows():
    title = str(row['title']).strip()
    genres = [g.strip() for g in str(row.get('spotify_genres', '')).split(',') if g.strip()]
    release_date = str(row.get('spotify_release_date', '')).strip()
    # Main artist
    main_artist = str(row['artists']).strip()
    if main_artist and main_artist.lower() != 'nan':
        info = artist_info[main_artist]
        info['main_count'] += 1
        info['songs'].add(title)
        info['genres'].update(genres)
        if release_date:
            info['release_dates'].add(release_date)
    # Featured artists
    feat = str(row['featured_artists']).strip()
    if feat and feat.lower() != 'nan':
        for f in feat.split(','):
            f = f.strip()
            if f:
                info = artist_info[f]
                info['featured_count'] += 1
                info['songs'].add(title)
                info['genres'].update(genres)
                if release_date:
                    info['release_dates'].add(release_date)

# Đọc artist_id nếu có
artist_id_map = {}
artist_id_file = 'data/artist_id_map.csv'
if os.path.exists(artist_id_file):
    id_df = pd.read_csv(artist_id_file)
    for _, r in id_df.iterrows():
        artist_id_map[str(r['artist']).strip()] = r['artist_id']

# Tạo DataFrame
rows = []
for artist, info in artist_info.items():
    all_dates = sorted(info['release_dates'])
    rows.append({
        'artist': artist,
        'artist_id': artist_id_map.get(artist, ''),
        'main_song_count': info['main_count'],
        'featured_song_count': info['featured_count'],
        'total_song_count': info['main_count'] + info['featured_count'],
        'all_songs': '; '.join(sorted(info['songs'])),
        'all_genres': ', '.join(sorted(info['genres'])),
        'first_release': all_dates[0] if all_dates else '',
        'last_release': all_dates[-1] if all_dates else ''
    })

artist_df = pd.DataFrame(rows)
artist_df = artist_df.sort_values('total_song_count', ascending=False)
artist_df.to_csv('data/artist_info_full_grouped.csv', index=False, encoding='utf-8-sig')
print('Đã xuất file data/artist_info_full_grouped.csv!')
