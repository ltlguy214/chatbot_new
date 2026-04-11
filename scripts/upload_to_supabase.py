import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from supabase import create_client, Client
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# --- CẤU HÌNH SUPABASE ---
SUPABASE_URL = str(os.getenv('SUPABASE_URL', '')).strip()
SUPABASE_KEY = str(os.getenv('SUPABASE_KEY', '') or os.getenv('SUPABASE_SERVICE_ROLE_KEY', '')).strip()  # Nên dùng Service Role Key để có quyền insert


def _is_missing_or_placeholder(value: str) -> bool:
    text = str(value or '').strip()
    if not text:
        return True
    lowered = text.lower()
    return lowered in {'none', 'null', 'changeme', 'your_key_here', 'your_url_here', 'your_supabase_key'}

if _is_missing_or_placeholder(SUPABASE_URL) or _is_missing_or_placeholder(SUPABASE_KEY):
    raise SystemExit(
        "Missing SUPABASE_URL/SUPABASE_KEY. Create a .env (see .env.example) or set env vars before running."
    )

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def upload_full_track_features():
    try:
        # 1. Đọc cả 2 file
        df_inner = pd.read_csv('merged_inner_data_final.csv')
        df_ml = pd.read_csv('data_prepared_for_ML.csv')
        
        # 2. Xử lý Vector Chủ đề (Từ 16 cột Topic)
        topic_cols = [f'topic_prob_{i}' for i in range(16)]
        df_ml[topic_cols] = df_ml[topic_cols].fillna(0)
        df_ml['topic_embedding'] = df_ml[topic_cols].values.tolist()
        df_topics_only = df_ml[['spotify_track_id', 'topic_embedding']]
        
        # Merge vector chủ đề vào file dữ liệu chính
        df = pd.merge(df_inner, df_topics_only, on='spotify_track_id', how='left')

        # 3. Tạo Vector Âm thanh (30 chiều)
        audio_cols = [
            'tempo_bpm', 'rms_energy', 'spectral_centroid_mean', 
            'zero_crossing_rate', 'spectral_rolloff', 'beat_strength_mean'
        ] 
        audio_cols += [f'chroma{i}_mean' for i in range(1, 13)]
        audio_cols += [f'mfcc{i}_mean' for i in range(1, 13)]

        features_data = df[audio_cols].fillna(0)
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features_data)
        df['audio_feature_embedding'] = scaled_features.tolist()

        # 4. BÍ QUYẾT: Xóa bỏ các cột đã tồn tại trong bảng 'songs'
        cols_already_in_songs = [
            'title', 'artists', 'featured_artists', 'album_type', 
            'spotify_popularity', 'spotify_release_date', 'spotify_genres', 
            'genres', 'main_artist_id', 'is_hit', 'file_name', 
            'Audio_Error', 'Lyrics_Error'
        ]
        # Xóa các cột đó đi, giữ lại TẤT CẢ những thứ còn lại (kèm 2 cột vector vừa tạo)
        df_upload = df.drop(columns=[col for col in cols_already_in_songs if col in df.columns])
        
        # 5. Xử lý chuẩn hóa để đưa lên SQL
        # Thay thế NaN, Inf thành None (để tương thích chuẩn SQL NULL)
        df_upload = df_upload.replace({np.nan: None, np.inf: None, -np.inf: None})
        
        data = df_upload.to_dict(orient='records')
        print(f"Bắt đầu tải {len(data)} bài hát với FULL ĐẶC TRƯNG lên Supabase...")

        # 6. Tải lên theo lô (batch)
        batch_size = 300 # Giảm size xuống một chút vì dữ liệu giờ nhiều cột hơn
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            
            # Format mảng thành chuỗi JSON "[0.1, 0.2]" cho cột Vector của Supabase
            for row in batch:
                if row.get('audio_feature_embedding'):
                    row['audio_feature_embedding'] = f"[{','.join(map(str, row['audio_feature_embedding']))}]"
                if row.get('topic_embedding'):
                    row['topic_embedding'] = f"[{','.join(map(str, row['topic_embedding']))}]"

            supabase.table("track_features").upsert(batch).execute()
            print(f"Đã tải xong: {min(i + batch_size, len(data))}/{len(data)}")
            
        print("--- THÀNH CÔNG RỰC RỠ! DATABASE ĐÃ HOÀN HẢO! ---")

    except Exception as e:
        print(f"Lỗi rồi: {e}")

if __name__ == "__main__":
    upload_full_track_features()


# def upload_lyrics():
#     try:
#         # 1. Đọc file
#         df = pd.read_csv('data/lyrics_data.csv')
        
#         # 2. Lọc cột (Giữ lại spotify_track_id, file_name, lyric)
#         expected_columns = ['spotify_track_id', 'file_name', 'lyric']
#         columns_to_keep = [col for col in expected_columns if col in df.columns]
#         df = df[columns_to_keep]
        
#         # 3. Xử lý triệt để lỗi dữ liệu trống
#         df = df.replace({np.nan: None})
        
#         data = df.to_dict(orient='records')
#         print(f"Bắt đầu tải {len(data)} lời bài hát lên Supabase...")

#         # 4. Đẩy lên Supabase theo batch bằng UPSERT
#         batch_size = 500
#         for i in range(0, len(data), batch_size):
#             batch = data[i:i + batch_size]
            
#             # Dùng upsert để tránh lỗi trùng lặp key
#             supabase.table("lyrics").upsert(batch).execute()
            
#             print(f"Đã tải xong: {min(i + batch_size, len(data))}/{len(data)}")
            
#         print("--- TẢI LỜI BÀI HÁT THÀNH CÔNG! ---")

#     except Exception as e:
#         print(f"Lỗi rồi: {e}")

# if __name__ == "__main__":
#     upload_lyrics()