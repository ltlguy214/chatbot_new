import pandas as pd
import numpy as np
from supabase import create_client, Client
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# --- CẤU HÌNH SUPABASE ---
SUPABASE_URL = str(os.getenv('SUPABASE_URL', '')).strip()
SUPABASE_KEY = str(os.getenv('SUPABASE_KEY', '') or os.getenv('SUPABASE_SERVICE_ROLE_KEY', '')).strip()


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

def upload_lyrics_safe():
    try:
        print("1. Đang đọc và đối chiếu dữ liệu...")
        
        # 1. Đọc danh sách ID bài hát CHUẨN từ file đã làm sạch
        # (Sử dụng đường dẫn file giống như file Python lúc nãy của bạn)
        df_valid_songs = pd.read_csv('final_data/merged_inner_data_final.csv')
        valid_ids = set(df_valid_songs['spotify_track_id'].dropna())

        # 2. Đọc file lyrics
        df_lyrics = pd.read_csv('data/lyrics_data.csv') # Sửa lại đường dẫn nếu file này nằm trong thư mục khác
        
        # 3. BƯỚC QUAN TRỌNG: Lọc bỏ các bài hát mồ côi (không có trong valid_ids)
        df_lyrics = df_lyrics[df_lyrics['spotify_track_id'].isin(valid_ids)]
        df_lyrics = df_lyrics.dropna(subset=['spotify_track_id', 'lyric'])
        df_lyrics = df_lyrics.replace({np.nan: None})

        cols_to_keep = ['spotify_track_id', 'file_name', 'lyric']
        records = df_lyrics[cols_to_keep].to_dict(orient='records')

        print(f"2. Bắt đầu đẩy {len(records)} lời bài hát (đã lọc sạch lỗi khóa ngoại) lên Supabase...")

        batch_size = 500
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            supabase.table("lyrics").upsert(batch).execute()
            print(f"Tiến độ: {min(i + batch_size, len(records))}/{len(records)}")

        print("--- THÀNH CÔNG RỰC RỠ! DATABASE ĐÃ HOÀN TẤT 100%! ---")

    except Exception as e:
        print(f"Lỗi: {e}")

if __name__ == "__main__":
    upload_lyrics_safe()