import joblib
import os
import sys

# --- BÍ KÍP TRỊ LỖI "No module named 'DA'" ---
# Ép Python phải nhìn bao quát toàn bộ thư mục gốc D:\Hit_songs_DA
thu_muc_goc = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if thu_muc_goc not in sys.path:
    sys.path.append(thu_muc_goc)
# ---------------------------------------------

def ep_can_model(ten_file_cu, ten_file_moi):
    try:
        print(f"Đang đọc file khổng lồ: {ten_file_cu} ...")
        # 1. Mở file model to
        model = joblib.load(ten_file_cu)
        
        print(f"Đang ép cân và lưu lại thành: {ten_file_moi} ...")
        # 2. Lưu lại với tham số compress=3 (Nén cực mạnh, file siêu nhỏ)
        joblib.dump(model, ten_file_moi, compress=3)
        
        # 3. Báo cáo dung lượng
        size_cu = os.path.getsize(ten_file_cu) / (1024 * 1024)
        size_moi = os.path.getsize(ten_file_moi) / (1024 * 1024)
        print(f"✅ Xong! Dung lượng giảm từ {size_cu:.2f} MB xuống còn {size_moi:.2f} MB!")
        
    except Exception as e:
        print(f"Lỗi rồi: {e}")

if __name__ == "__main__":
    # Nhớ dùng dấu / để tránh lỗi đường dẫn nhé
    
    # Ép cân P1
    ep_can_model('DA/models/best_model_p1.pkl', 'DA/models/best_model_p1_compressed.pkl')
    
    # Ép cân P4
    ep_can_model('DA/models/best_model_p4.pkl', 'DA/models/best_model_p4_compressed.pkl')