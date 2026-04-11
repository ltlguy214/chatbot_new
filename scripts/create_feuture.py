import pandas as pd
import numpy as np

# 1. Đọc file dữ liệu
file_path = 'data/merged_inner_data_final.csv'
df = pd.read_csv(file_path, low_memory=False)

print(f"📊 Đang xử lý {len(df)} bài hát...")

# 2. Tính toán wps (Words Per Second)
# Công thức: Tổng số từ / Thời lượng (giây)
# Sử dụng np.where để tránh lỗi chia cho 0 nếu bài hát có duration = 0
df['lyrical_density'] = np.where(
    df['duration_sec'] > 0, 
    (df['lyric_total_words'] / df['duration_sec']).round(4), 
    0
)

# 3. Lưu đè trực tiếp vào file cũ
df.to_csv(file_path, index=False, encoding='utf-8-sig')

print(f"✅ Đã thêm cột 'lyrical_density' vào cuối file {file_path} thành công!")