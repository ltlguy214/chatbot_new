import os
import pandas as pd
from analysis_backend import VPopAnalysisBackend

# ==========================================
# CẤU HÌNH FILE TEST
# ==========================================
# 1. Tên file nhạc MP3 thực tế của bạn (Bài 1000 Ánh Mắt)
TEST_AUDIO_FILE = r"chatbot\Test_app\bay không cần cánh - rhyder.mp3" 

# 2. Đường dẫn tới file CSV chứa data gốc (Đường dẫn tương đối hoặc tuyệt đối)
CSV_FILE = "final_data/merged_inner_data_final.csv"

# 3. ID bài hát muốn so sánh (1000 Ánh Mắt - Shiki)
TRACK_ID = "4yVjUXGvBqnHhoP322fSZr"

def run_full_comparison():
    print("🚀 ĐANG KHỞI TẠO BACKEND...")
    backend = VPopAnalysisBackend(supabase_client=None)

    # Tạm tắt NLP/Topic để test nhanh Librosa
    backend.analyze_lyric_topics = lambda text: {f"topic_prob_{i}": 0.0 for i in range(16)}

    # =======================================================
    # 1. PHÂN TÍCH FILE ÂM THANH BẰNG CODE BACKEND
    # =======================================================
    print(f"\n🎵 Đang phân tích file: {TEST_AUDIO_FILE}...")
    if not os.path.exists(TEST_AUDIO_FILE):
        print(f"❌ LỖI: Không tìm thấy file '{TEST_AUDIO_FILE}'")
        return
        
    computed_data = backend.analyze_full_audio(TEST_AUDIO_FILE)

    # =======================================================
    # 2. ĐỌC DỮ LIỆU TỪ DATABASE (CSV)
    # =======================================================
    print(f"📂 Đang đọc dữ liệu gốc từ: {CSV_FILE}...")
    if not os.path.exists(CSV_FILE):
        print(f"❌ LỖI: Không tìm thấy file CSV tại '{CSV_FILE}'")
        return
        
    df = pd.read_csv(CSV_FILE, low_memory=False)
    row = df[df['spotify_track_id'] == TRACK_ID]
    
    if row.empty:
        print(f"❌ LỖI: Không tìm thấy ID {TRACK_ID} trong file CSV.")
        return
        
    truth_data = row.iloc[0].to_dict()

    # =======================================================
    # 3. SO SÁNH ĐỐI XỨNG 40D + PHYSICAL FEATURES
    # =======================================================
    print("\n" + "="*90)
    print(f"📊 BẢNG SO SÁNH CHI TIẾT: CODE TÍNH TOÁN vs DỮ LIỆU GỐC (CSV)")
    print("="*90)
    print(f"{'Thông số (Features)':<30} | {'Tính toán (Backend)':<20} | {'Gốc (Database CSV)':<20} | Độ Lệch")
    print("-" * 90)

    # Gom các đặc trưng theo nhóm để dễ nhìn
    phys_keys = ['tempo_bpm', 'rms_energy', 'spectral_centroid_mean', 'zero_crossing_rate',
                 'spectral_rolloff', 'spectral_flatness_mean', 'beat_strength_mean', 'onset_rate']
    mfcc_keys = [f'mfcc{i}_mean' for i in range(1, 14)]
    chroma_keys = [f'chroma{i}_mean' for i in range(1, 13)]
    contrast_keys = [f'spectral_contrast_band{i}_mean' for i in range(1, 8)]

    all_keys = phys_keys + mfcc_keys + chroma_keys + contrast_keys

    for key in all_keys:
        val_comp = computed_data.get(key, None)
        val_truth = truth_data.get(key, None)
        
        # Bỏ qua nếu dữ liệu gốc bị NaN
        if val_comp is not None and val_truth is not None and not pd.isna(val_truth):
            try:
                # Ép kiểu về Float để tránh lỗi trừ chuỗi (String)
                val_comp_f = float(val_comp)
                val_truth_f = float(val_truth)
                
                diff = abs(val_comp_f - val_truth_f)
                
                # Cảnh báo lệch (Lệch trên 5% và chênh lệch tuyệt đối > 0.1)
                alert = "⚠️ Có lệch (Do MP3 khác âm lượng/chất lượng)" if (diff > 0.05 * abs(val_truth_f) and diff > 0.1) else ""
                
                print(f"{key:<30} | {val_comp_f:<20.4f} | {val_truth_f:<20.4f} | {diff:.4f} {alert}")
            except ValueError:
                # Nếu không thể ép sang số (Dữ liệu trong CSV bị lỗi chữ)
                print(f"{key:<30} | {str(val_comp):<20} | {str(val_truth):<20} | ❌ LỖI DATA (Chữ)")
        else:
            print(f"{key:<30} | {str(val_comp):<20} | {str(val_truth):<20} | ---")
            
    print("="*90)
    print("💡 LƯU Ý: Nếu bạn thấy cảnh báo '⚠️ Có lệch' ở nhiều dòng, có thể do:")
    print("   1. File MP3 bạn đang test có chất lượng bitrate/độ dài khác với file lúc tải về đợt crawl data.")
    print("   2. Phiên bản thư viện Librosa hiện tại khác với phiên bản lúc crawl.")
    print("   Nếu lệch cực nhỏ (ví dụ 0.001) thì hoàn toàn bình thường do sai số làm tròn thập phân.")

if __name__ == "__main__":
    run_full_comparison()