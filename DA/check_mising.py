import pandas as pd

# Đường dẫn file của bạn
FILE_AUDIO = r'final_data\mergerd_balanced_and_features.csv'
FILE_NLP = r'Audio_lyric\nlp_analysis.csv'

# 1. Load dữ liệu
df_audio = pd.read_csv(FILE_AUDIO)
df_nlp = pd.read_csv(FILE_NLP)

# 2. Merge thử nghiệm (Inner join)
df_merged = pd.merge(df_audio, df_nlp[['file_name', 'lyric']], on='file_name', how='inner')

# 3. Tìm các bài bị mất sau merge (do lệch file_name)
missing_merge = df_audio[~df_audio['file_name'].isin(df_merged['file_name'])]

# 4. Tìm các bài bị loại do Lyric bị trống (NaN)
missing_lyric = df_merged[df_merged['lyric'].isna()]

# 5. Tìm các bài bị loại do không thuộc 4 dòng nhạc (Hip-Hop, Indie, V-Pop, Vinahouse)
# Copy hàm extract từ file P4 của bạn
def extract_all_genres(genre_raw):
    try:
        cleaned = str(genre_raw).lower()
        mapped = []
        if 'vinahouse' in cleaned: mapped.append('Vinahouse')
        if 'hip hop' in cleaned or 'rap' in cleaned: mapped.append('Hip-Hop')
        if 'indie' in cleaned: mapped.append('Indie')
        if 'v-pop' in cleaned or 'pop' in cleaned: mapped.append('V-Pop')
        return mapped
    except: return []

df_merged['genres_list'] = df_merged['spotify_genres'].apply(extract_all_genres)
missing_genre = df_merged[df_merged['genres_list'].apply(len) == 0]

# 6. TỔNG HỢP KẾT QUẢ
print("="*60)
print(f"🔍 PHÂN TÍCH LÝ DO MẤT 6 BÀI HÁT")
print("="*60)
print(f"1. Do không khớp file_name khi merge:  {len(missing_merge)} bài")
print(f"2. Do nội dung Lyric bị trống (NaN):    {len(missing_lyric)} bài")
print(f"3. Do không thuộc 4 dòng nhạc mục tiêu: {len(missing_genre)} bài")
print("-"*60)

if len(missing_genre) > 0:
    print("📋 DANH SÁCH BÀI BỊ LOẠI DO SAI THỂ LOẠI (Có thể là Bolero/Lo-fi):")
    print(missing_genre[['file_name', 'spotify_genres']].head(10))