import pandas as pd

df = pd.read_csv('data/vpop_master_metadata.csv')

print('=== COLUMN NAMES ===')
cols = list(df.columns)
for i, col in enumerate(cols, 1):
    print(f"{i}. {col}")

print('\n=== SUMMARY ===')
print(f'Total songs: {len(df)}')
print(f'Songs on 4 platforms: {(df["total_platforms"] == 4).sum()}')
print(f'Songs on 3 platforms: {(df["total_platforms"] == 3).sum()}')
print(f'Songs on 2 platforms: {(df["total_platforms"] == 2).sum()}')
print(f'Songs on 1 platform: {(df["total_platforms"] == 1).sum()}')
print(f'Songs on 0 platforms: {(df["total_platforms"] == 0).sum()}')

print('\n=== HIT SCORE DISTRIBUTION ===')
print(df['hit_score'].describe())

print('\n=== SAMPLE: Songs with 4 platforms ===')
print(df[df['total_platforms'] == 4][['title', 'artists', 'hit_score', 'best_peak_rank', 'total_appearances']].head(10))

print('\n=== SAMPLE: Songs with 1 platform ===')
print(df[df['total_platforms'] == 1][['title', 'artists', 'source', 'hit_score']].head(10))
