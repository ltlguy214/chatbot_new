import pandas as pd
from collections import Counter

# Đọc dữ liệu
song_df = pd.read_csv('data/song_list_info.csv')

# Xử lý nghệ sĩ (cột 'artists')
artist_rows = []
for artist in song_df['artists'].dropna().unique():
    artist_songs = song_df[song_df['artists'] == artist]
    num_songs = len(artist_songs)
    genres = []
    for g in artist_songs['spotify_genres'].dropna():
        genres.extend([x.strip() for x in str(g).split(',')])
    genre_counts = Counter(genres)
    all_genres = ', '.join(sorted(set(genres)))
    first_release = artist_songs['spotify_release_date'].min() if 'spotify_release_date' in artist_songs else ''
    last_release = artist_songs['spotify_release_date'].max() if 'spotify_release_date' in artist_songs else ''
    artist_rows.append({
        'artist': artist,
        'num_songs': num_songs,
        'all_genres': all_genres,
        'top_genre': genre_counts.most_common(1)[0][0] if genre_counts else '',
        'first_release': first_release,
        'last_release': last_release
    })

artist_df = pd.DataFrame(artist_rows)
artist_df = artist_df.sort_values('num_songs', ascending=False)

# Xuất file
artist_df.to_csv('data/artist_info_full.csv', index=False, encoding='utf-8-sig')
print('Đã xuất file data/artist_info_full.csv!')
