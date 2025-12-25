import pandas as pd

# Đọc file phân tích và file gốc

# Đọc các file
features = pd.read_csv('Audio_lyric/librosa_analysis.csv')
info = pd.read_csv('final_data/balanced_500_500.csv')
nlp = pd.read_csv('Audio_lyric/nlp_analysis.csv')

# Tách các trường cần lấy thêm từ nlp_analysis.csv
nlp_extra = nlp[['file_name','lyric_total_words','lexical_diversity','noun_count','verb_count','adj_count']]
lyrics_sentiment = nlp[['file_name','sentiment','sentiment_score','sentiment_confidence']]

# Merge features + info
merged = pd.merge(features, info, on='file_name', how='inner')
# Merge thêm các trường ngôn ngữ học từ nlp_analysis.csv
merged = pd.merge(merged, nlp_extra, on='file_name', how='left')

# Merge sentiment mới từ lyrics_extracted.csv
merged = pd.merge(merged, lyrics_sentiment, on='file_name', how='left', suffixes=('', '_new'))

# Nếu có cột sentiment_new (do trùng tên), thay thế cột sentiment bằng sentiment_new
for col in ['sentiment','sentiment_score','sentiment_confidence']:
    if f'{col}_new' in merged.columns:
        merged[col] = merged[f'{col}_new']
        merged = merged.drop(columns=[f'{col}_new'])

# Bỏ cột label, chỉ giữ is_hit
if 'label' in merged.columns:
    merged = merged.drop(columns=['label'])

# Đếm số lượng hit và non_hit
hit_count = (merged['is_hit'] == 1).sum()
non_hit_count = (merged['is_hit'] == 0).sum()
print(f'Số lượng hit: {hit_count}')
print(f'Số lượng non_hit: {non_hit_count}')

# Loại bỏ các cột không cần thiết trước khi xuất

# Giữ lại file_name, chỉ loại các cột không cần thiết khác
cols_to_drop = [
    'source', 'match_key', '__key__', 'featured_artists', 'spotify_track_id'
]
merged = merged.drop(columns=[col for col in cols_to_drop if col in merged.columns])

# Đưa các cột quan trọng lên đầu

# Đưa file_name lên đầu cùng các cột quan trọng
first_cols = [
    'file_name', 'title', 'artists', 'spotify_popularity', 'spotify_release_date',
    'spotify_genres', 'is_hit', 'total_plays', 'spotify_streams'
]
other_cols = [col for col in merged.columns if col not in first_cols]
merged = merged[first_cols + other_cols]

merged.to_csv('final_data/mergerd_balanced_and_features.csv', index=False)
