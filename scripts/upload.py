import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from supabase import create_client, Client
import json
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# --- TỪ ĐIỂN TOPIC ---
TOPIC_MAPPING = {
    "topic_prob_-1": "Nhiễu / Không phân loại", "topic_prob_0": "Ballad Thất tình / Chia tay",
    "topic_prob_1": "Tình yêu đôi lứa / Lãng mạn", "topic_prob_2": "Rap Hiphop Gai góc",
    "topic_prob_3": "Rap Đời sống / Hustle", "topic_prob_4": "Nhạc Trữ tình / Hoài cổ",
    "topic_prob_5": "Pop / R&B Âu Mỹ (English)", "topic_prob_6": "Tình cảm Gia đình / Cha mẹ",
    "topic_prob_7": "Nhạc Tết / Xuân", "topic_prob_8": "Hoài niệm / Kỷ niệm",
    "topic_prob_9": "Lòng yêu nước / Tự hào", "topic_prob_10": "Pop hiện đại / Thả thính",
    "topic_prob_11": "Cấu trúc Anh ngữ bổ trợ", "topic_prob_12": "Ad-libs / Biểu cảm",
    "topic_prob_13": "Tâm sự / Tỏ tình trực tiếp", "topic_prob_14": "Ngôn ngữ ngoại lai (Latin/Pháp)"
}

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

def upload_100_percent_full():
    try:
        # 1. Đọc dữ liệu
        print("1. Đang đọc 3 file dữ liệu...")
        df_inner = pd.read_csv('final_data/merged_inner_data_final.csv')
        df_ml = pd.read_csv('final_data/data_prepared_for_ML.csv')
        df_vibes = pd.read_csv('DA\\final_data\\VPop_5_Vibes_Final.csv')

        # Xử lý Topic (15 cột)
        topic_cols = [c for c in df_ml.columns if c.startswith('topic_prob_')]
        df_ml['main_topic'] = df_ml[topic_cols].apply(lambda r: TOPIC_MAPPING.get(r.idxmax(), "Khác") if r.max()>0 else "Không xác định", axis=1)
        df_ml['topic_embedding'] = df_ml[topic_cols].fillna(0).values.tolist()

        # 2. Gộp dữ liệu
        df = pd.merge(df_inner, df_vibes[['spotify_track_id', 'cluster_main', 'vibe']], on='spotify_track_id', how='left')
        
        # GIỮ LẠI CÁC CỘT TOPIC_PROB ĐỂ VẼ BIỂU ĐỒ SAU NÀY
        cols_to_keep_from_ml = ['spotify_track_id', 'topic_embedding', 'main_topic'] + topic_cols
        df = pd.merge(df, df_ml[cols_to_keep_from_ml], on='spotify_track_id', how='left')

        # 3. Ép kiểu số nguyên tuyệt đối cho Cluster
        if 'cluster_main' in df.columns:
            df['cluster_main'] = pd.to_numeric(df['cluster_main'], errors='coerce').astype('Int64')

        # 4. Tạo Audio Vector (30 chiều)
        audio_cols = ['tempo_bpm', 'rms_energy', 'spectral_centroid_mean', 'zero_crossing_rate', 'spectral_rolloff', 'beat_strength_mean'] 
        audio_cols += [f'chroma{i}_mean' for i in range(1, 13)] + [f'mfcc{i}_mean' for i in range(1, 13)]
        scaler = StandardScaler()
        for col in audio_cols:
            if col not in df.columns: df[col] = 0.0
        df['audio_feature_embedding'] = scaler.fit_transform(df[audio_cols].fillna(0)).tolist()

        # 5. Loại bỏ cột thừa đã có ở bảng 'songs'
        cols_drop = ['title', 'artists', 'featured_artists', 'album_type', 'spotify_popularity', 'spotify_release_date', 'spotify_genres', 'genres', 'main_artist_id', 'is_hit', 'file_name', 'Audio_Error', 'Lyrics_Error']
        df_upload = df.drop(columns=[c for c in cols_drop if c in df.columns])

        # 6. Chuyển sang Dict và làm sạch
        records = df_upload.to_dict(orient='records')
        final_payload = []
        
        for row in records:
            clean_row = {}
            for k, v in row.items():
                # BẢO VỆ: Nếu là list (Vector) thì giữ nguyên, không dùng isna kiểm tra
                if isinstance(v, list):
                    clean_row[k] = v
                # Xử lý NaN / NaT / Khuyết cho các biến số, chuỗi thông thường
                elif pd.isna(v):
                    clean_row[k] = None
                # Xử lý ép kiểu Integer cho cluster
                elif k == 'cluster_main':
                    clean_row[k] = int(v)
                else:
                    clean_row[k] = v
            
            # Format 2 Vector thành String cho Supabase pgvector
            if isinstance(clean_row.get('audio_feature_embedding'), list):
                clean_row['audio_feature_embedding'] = f"[{','.join(map(str, clean_row['audio_feature_embedding']))}]"
            else:
                clean_row['audio_feature_embedding'] = None
                
            if isinstance(clean_row.get('topic_embedding'), list):
                clean_row['topic_embedding'] = f"[{','.join(map(str, clean_row['topic_embedding']))}]"
            else:
                clean_row['topic_embedding'] = None
                
            final_payload.append(clean_row)

        print(f"2. Bắt đầu đẩy {len(final_payload)} bản ghi FULL DATA (Không thiếu một cột nào)...")
        
        # 7. Tải lên
        batch_size = 200
        for i in range(0, len(final_payload), batch_size):
            batch = final_payload[i:i + batch_size]
            supabase.table("track_features").upsert(batch).execute()
            print(f"Tiến độ: {min(i + batch_size, len(final_payload))}/{len(final_payload)}")

        print("--- THÀNH CÔNG RỰC RỠ! DATABASE BÂY GIỜ LÀ HOÀN HẢO! ---")

    except Exception as e:
        print(f"Lỗi: {e}")

if __name__ == "__main__":
    upload_100_percent_full()