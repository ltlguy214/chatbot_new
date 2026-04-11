import pandas as pd
import os
from sklearn.preprocessing import MultiLabelBinarizer

# --- 0. TẠO THƯ MỤC LƯU TRỮ ---
if not os.path.exists('final_data'):
    os.makedirs('final_data')

# --- 1. ĐỌC VÀ GHÉP DỮ LIỆU ---
print("⏳ Đang đọc dữ liệu âm thanh và lyric...")
df_audio = pd.read_csv('final_data/merged_inner_data_final.csv', low_memory=False)
df_lyrics_topics = pd.read_csv('final_data/lyrics_topic_features.csv') # File vừa tải từ Colab

# Ghép 2 bảng lại dựa trên spotify_track_id
df = pd.merge(df_audio, df_lyrics_topics, on='spotify_track_id', how='inner')
print(f"✅ Đã ghép thành công. Tổng số bài hát sau khi ghép: {len(df)}")

# Dọn dẹp cột lỗi
df = df.drop(columns=[c for c in ['Audio_Error', 'Lyrics_Error'] if c in df.columns])

# --- 2. XỬ LÝ MULTI-LABEL GENRES ---
df['genres_list'] = df['genres'].fillna('Unknown').apply(lambda x: [i.strip() for i in str(x).split(',')])
mlb = MultiLabelBinarizer()
genres_encoded = mlb.fit_transform(df['genres_list'])
genres_df = pd.DataFrame(genres_encoded, columns=[f"genre_{c}" for c in mlb.classes_], index=df.index)

# --- 3. XỬ LÝ MUSICAL KEY (ONE-HOT CHO "D Min", "C# Maj"...) ---
keys_ohe = pd.get_dummies(df['musical_key'], prefix='key', dtype=int)

# --- 4. PHÂN LOẠI CÁC NHÓM BIẾN ---
# Cột định danh (Giữ lại Date để chia Train/Test)
identity_cols = ['spotify_track_id', 'title', 'artists', 'spotify_release_date', 'genres']

# Target
target_cols = ['is_hit', 'spotify_popularity', 'final_sentiment']

# Metadata cần loại bỏ
metadata_to_drop = [
    'featured_artists', 'spotify_genres', 'genres_list',
    'main_artist_id', 'file_name', 'musical_key', 'album_type'
]

# Tự động xác định các cột số (Bao gồm cả Features Âm thanh và Features Topics)
ignore_all = identity_cols + target_cols + metadata_to_drop + list(genres_df.columns) + list(keys_ohe.columns)
num_cols = [col for col in df.columns if col not in ignore_all and df[col].dtype in ['float64', 'int64']]

print(f"📊 Số lượng đặc trưng số sẽ được lưu THÔ (không scale): {len(num_cols)} (Gồm Audio và Lyric Topics)")

# --- 5. XỬ LÝ MISSING & SCALING ---
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

print("ℹ️  BỎ QUA SCALING trong master data để tránh Data Leakage (TimeSeries).")
print("    Scaling sẽ được thực hiện bên trong Pipeline/CV của từng task.")

df_master = pd.concat([
    df[identity_cols],
    df[target_cols],
    df[num_cols],
    keys_ohe,
    genres_df
], axis=1)

# Hàm xử lý ngày tháng thông minh
def fix_spotify_date(date_val):
    if pd.isna(date_val): 
        return "1900-01-01" # Gán một ngày rất cũ cho các bài không rõ ngày để đẩy lên đầu
    
    date_str = str(date_val).strip()
    
    # Trường hợp chỉ có năm (ví dụ: "2020")
    if len(date_str) == 4:
        return f"{date_str}-01-01"
    
    # Trường hợp chỉ có năm và tháng (ví dụ: "2020-05")
    if len(date_str) == 7:
        return f"{date_str}-01"
        
    return date_str

print("📅 Đang chuẩn hóa ngày phát hành...")
# Áp dụng sửa ngày trước khi chuyển sang định dạng datetime
df_master['spotify_release_date'] = df_master['spotify_release_date'].apply(fix_spotify_date)

# Chuyển hẳn sang kiểu datetime để sắp xếp
df_master['spotify_release_date'] = pd.to_datetime(df_master['spotify_release_date'], errors='coerce')

# Sắp xếp từ cũ đến mới (quan trọng cho TimeSeriesSplit)
df_master = df_master.sort_values('spotify_release_date').reset_index(drop=True)

print(f"Đã xử lý xong. Ngày cũ nhất: {df_master['spotify_release_date'].min().date()}")
print(f"Ngày mới nhất: {df_master['spotify_release_date'].max().date()}")

# --- 7. LƯU FILE KẾT QUẢ ---
output_path = 'final_data/data_prepared_for_ML.csv'
df_master.to_csv(output_path, index=False, encoding='utf-8-sig')

print("\n" + "="*40)
print("MASTER DATA READY 100% FOR ALL 5 TASKS")
print("="*40)
print(f"Tổng số đặc trưng đầu vào: {len(num_cols) + len(keys_ohe.columns) + len(genres_df.columns)}")
print(f"Trong đó có {df_lyrics_topics.shape[1]-1} đặc trưng từ Chủ đề Lyric.")
print(f"Dữ liệu đã được sắp xếp từ {df_master['spotify_release_date'].min().date()} đến {df_master['spotify_release_date'].max().date()}")