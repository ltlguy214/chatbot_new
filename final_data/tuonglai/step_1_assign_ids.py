import pandas as pd
import os

# =============================================================================
# CẤU HÌNH ĐƯỜNG DẪN
# =============================================================================
BRIDGE_FILE = os.path.join("final_data", "balanced_500_500.csv")
LIBROSA_FILE = os.path.join("Audio_lyric", "librosa_analysis.csv")
NLP_FILE = os.path.join("Audio_lyric", "nlp_analysis.csv")

# =============================================================================
# HÀM XỬ LÝ
# =============================================================================

def assign_track_id():
    print("🚀 BƯỚC 1: GÁN TRACK ID (CHỈ TRACK_ID & UNIQUE COUNT)...")

    # 1. Đọc file cầu nối
    if not os.path.exists(BRIDGE_FILE):
        print(f"❌ Lỗi: Không tìm thấy file {BRIDGE_FILE}")
        return

    print(f"📂 Đang đọc cầu nối: {BRIDGE_FILE}")
    df_bridge = pd.read_csv(BRIDGE_FILE)
    
    # --- THAY ĐỔI: CHỈ LẤY ĐÚNG 2 CỘT CẦN THIẾT ---
    map_df = df_bridge[['file_name', 'spotify_track_id']]
    
    # Đổi tên 'spotify_track_id' -> 'track_id'
    map_df = map_df.rename(columns={'spotify_track_id': 'track_id'})
    
    # Loại bỏ trùng lặp file_name trong file map
    map_df = map_df.drop_duplicates(subset=['file_name'])

    # ---------------------------------------------------------
    # 2. Xử lý File Librosa
    # ---------------------------------------------------------
    if os.path.exists(LIBROSA_FILE):
        print(f"\n🛠️ Đang xử lý: {LIBROSA_FILE}")
        df_librosa = pd.read_csv(LIBROSA_FILE)
        original_len = len(df_librosa)
        
        # Xóa track_id cũ nếu đã tồn tại để tránh lỗi duplicate cột
        if 'track_id' in df_librosa.columns:
            df_librosa = df_librosa.drop(columns=['track_id'])

        # Merge
        df_librosa_mapped = pd.merge(df_librosa, map_df, on='file_name', how='left')

        # Thống kê
        matched = df_librosa_mapped['track_id'].notnull().sum()
        unique_ids = df_librosa_mapped['track_id'].nunique() # <--- ĐẾM SỐ UNIQUE
        
        print(f"   - Tổng số dòng (files): {original_len}")
        print(f"   - Đã gán được track_id: {matched}")
        print(f"   - Số lượng Unique track_id: {unique_ids}") # <--- IN RA Ở ĐÂY
        
        if unique_ids < matched:
            print(f"   ⚠️ Lưu ý: Có {matched - unique_ids} file trùng ID (Duplicate tracks).")

        # Lưu đè
        df_librosa_mapped.to_csv(LIBROSA_FILE, index=False, encoding='utf-8-sig')
        print("   ✅ Đã lưu xong file Librosa.")

    # ---------------------------------------------------------
    # 3. Xử lý File NLP
    # ---------------------------------------------------------
    if os.path.exists(NLP_FILE):
        print(f"\n🛠️ Đang xử lý: {NLP_FILE}")
        df_nlp = pd.read_csv(NLP_FILE)
        
        if 'track_id' in df_nlp.columns:
            df_nlp = df_nlp.drop(columns=['track_id'])
            
        # Merge
        df_nlp_mapped = pd.merge(df_nlp, map_df, on='file_name', how='left')
        
        unique_ids_nlp = df_nlp_mapped['track_id'].nunique() # <--- ĐẾM SỐ UNIQUE

        # Lưu đè
        df_nlp_mapped.to_csv(NLP_FILE, index=False, encoding='utf-8-sig')
        print(f"   ✅ Đã lưu xong file NLP (Unique IDs: {unique_ids_nlp}).")

    print("\n🏁 HOÀN TẤT BƯỚC 1.")

if __name__ == "__main__":
    assign_track_id()