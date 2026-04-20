import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# --- CHỈ ĐƯỜNG CHO PYTHON ---
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from chatbot.supabase import get_supabase_client 
from chatbot.env import load_env

def update_vectors_scientific():
    load_env()
    supabase = get_supabase_client()
    
    print("1. Đang đọc dữ liệu gốc...")
    # Sử dụng raw string hoặc dấu gạch chéo xuôi để tránh lỗi đường dẫn Windows
    csv_path = os.path.join(ROOT_DIR, 'final_data', 'merged_inner_data_final.csv')
    df = pd.read_csv(csv_path)
    
    feature_cols = [f'mfcc{i}_mean' for i in range(1, 14)] + \
                   [f'chroma{i}_mean' for i in range(1, 13)] + \
                   [f'spectral_contrast_band{i}_mean' for i in range(1, 8)] + \
                   ['tempo_bpm', 'rms_energy', 'spectral_centroid_mean', 'zero_crossing_rate', 
                    'spectral_rolloff', 'spectral_flatness_mean', 'beat_strength_mean', 'onset_rate']
    
    # =========================================================
    # BƯỚC QUAN TRỌNG: ÉP KIỂU SỐ (FIX LỖI STRING ' ')
    # =========================================================
    print("⏳ Đang dọn dẹp dữ liệu (Chuyển đổi các ô chứa chữ/khoảng trắng sang số)...")
    for col in feature_cols:
        # errors='coerce' sẽ biến các ô chứa ' ' hoặc chữ thành NaN
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Sau khi biến lỗi thành NaN, bây giờ mới dùng fillna(0) được
    df_features = df[feature_cols].fillna(0)
    
    print("2. Đang huấn luyện StandardScaler...")
    scaler = StandardScaler()
    z_matrix = scaler.fit_transform(df_features)
    
    # Đảm bảo thư mục models tồn tại
    os.makedirs(os.path.join(ROOT_DIR, 'models'), exist_ok=True)
    joblib.dump(scaler, os.path.join(ROOT_DIR, 'models', 'audio_scaler.joblib'))
    print("✅ Đã lưu file thước đo chuẩn vào 'models/audio_scaler.joblib'")
    
    print("3. Đang đẩy Vector chuẩn lên Supabase...")
    success = 0
    # Chuyển sang dùng bulk update hoặc xử lý theo từng dòng
    for idx, row in df.iterrows():
        track_id = row['spotify_track_id']
        vector_40d = [float(x) for x in z_matrix[idx]] 
        
        try:
            res = supabase.table("track_features").update({
                "audio_feature_embedding": vector_40d
            }).eq("spotify_track_id", track_id).execute()
            
            if res.data:
                success += 1
                if success % 100 == 0:
                    print(f"  -> Đã đẩy {success} bài...")
        except Exception as e:
            print(f"❌ Lỗi tại bài {track_id}: {e}")

    print(f"🎉 HOÀN THÀNH! Cập nhật thành công {success} vector lên Database.")

if __name__ == "__main__":
    update_vectors_scientific()