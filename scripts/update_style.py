import pandas as pd
import sys
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# 1. Đảm bảo load biến môi trường
from chatbot.env import load_env
load_env()

from chatbot.supabase import get_supabase_client

# 2. Đọc dữ liệu từ file CSV
csv_path = os.path.join("DA", "final_data", "VPop_5_Vibes_Final.csv")
if not os.path.exists(csv_path):
    print(f"❌ Không tìm thấy file CSV tại: {csv_path}")
    sys.exit(1)

df = pd.read_csv(csv_path)

data_to_update = df[['spotify_track_id', 'cluster_main', 'vibe']].to_dict(orient='records')

supabase = get_supabase_client()
if supabase is None:
    print("❌ Lỗi: Không thể kết nối Supabase. Kiểm tra lại file .env")
    sys.exit(1)

print(f"🔄 Đang chuẩn bị cập nhật {len(data_to_update)} bài hát (Chỉ update vibe và cluster_main)...")

# Hàm cập nhật với cơ chế Retry (Thử lại)
def update_single_row(record, retries=3):
    track_id = record.pop('spotify_track_id')
    for attempt in range(retries):
        try:
            # Lệnh update: Chỉ cập nhật những trường có trong record (vibe, cluster_main)
            supabase.table('songs').update(record).eq('spotify_track_id', track_id).execute()
            return True, track_id
        except Exception as e:
            err_msg = str(e).lower()
            if "disconnected" in err_msg or "timeout" in err_msg or "502" in err_msg:
                # Nếu là lỗi mạng/server, ngủ 2 giây rồi thử lại
                time.sleep(2)
                continue
            else:
                # Nếu là lỗi logic (vd: sai tên cột), báo lỗi ngay
                return False, f"{track_id} - Lỗi: {str(e)}"
    
    return False, f"{track_id} - Thất bại sau {retries} lần thử (Server disconnected)"

# 3. Chạy đa luồng (Multi-threading) an toàn
success_count = 0
error_count = 0
# GIẢM SỐ LUỒNG XUỐNG 5 ĐỂ TRÁNH QUÁ TẢI SERVER SUPABASE
max_workers = 5  

print(f"🚀 Bắt đầu bắn dữ liệu lên Supabase với {max_workers} luồng song song (Có cơ chế Retry)...")

with ThreadPoolExecutor(max_workers=max_workers) as executor:
    # Gửi task vào queue kèm theo thời gian nghỉ siêu nhỏ (rate limiting)
    futures = []
    for rec in data_to_update:
        futures.append(executor.submit(update_single_row, dict(rec))) # Truyền bản sao của dict
        time.sleep(0.01) # Ngắt nhịp nhẹ để phân tán request
    
    # Theo dõi tiến độ
    for idx, future in enumerate(as_completed(futures), 1):
        is_success, msg = future.result()
        if is_success:
            success_count += 1
        else:
            error_count += 1
            print(f"❌ {msg}")
            
        # In log mỗi 100 bài
        if idx % 100 == 0:
            print(f"⚡ Đã xử lý {idx}/{len(data_to_update)} bài hát... (Thành công: {success_count}, Lỗi: {error_count})")

print(f"\n✨ HOÀN TẤT: Cập nhật thành công {success_count} bài, thất bại {error_count} bài.")