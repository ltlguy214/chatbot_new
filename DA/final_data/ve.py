import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# 1. Đường dẫn tới file CSV kết quả của P3
csv_path = 'DA/final_data/model_comparison_results_p3.csv'
output_path = 'DA/tasks/Sentiment/p3_model_comparison_f1_fixed.png'

# Đọc dữ liệu
try:
    df = pd.read_csv(csv_path)
except FileNotFoundError:
    print(f"❌ Không tìm thấy file {csv_path}.")
    exit()

# 2. Xử lý dữ liệu
# Chọn cột F1 để vẽ. (Ưu tiên CV để chọn best model, nếu không có thì dùng Test)
if 'CV_F1_Macro' in df.columns:
    metric_col = 'CV_F1_Macro'
    label_metric = 'CV F1-Macro (Train, TimeSeriesSplit)'
elif 'CV_Score' in df.columns:
    metric_col = 'CV_Score'
    label_metric = 'CV F1-Macro (Train, TimeSeriesSplit)'
else:
    metric_col = 'Test_F1_Macro'
    label_metric = 'Test F1-Macro (Report only)'

# SẮP XẾP GIẢM DẦN: F1 cao nhất ở vị trí số 0
df_plot = df.sort_values(by=metric_col, ascending=False).reset_index(drop=True)

# 3. Khởi tạo biểu đồ
fig, ax = plt.subplots(figsize=(16, 10))

# Tạo dải màu (đậm nhất cho top 1 ở trên cùng)
colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(df_plot)))

# Vẽ thanh ngang
bars = ax.barh(df_plot['Model'], df_plot[metric_col], color=colors, edgecolor='white', linewidth=1.5)

# 4. Gắn text giá trị ở cuối mỗi thanh
for i, (idx, row) in enumerate(df_plot.iterrows()):
    ax.text(row[metric_col] + 0.005, i, f"{row[metric_col]:.4f}",
            va='center', fontsize=11, fontweight='bold', color='black')

# 5. Định dạng nhãn và tiêu đề
ax.set_xlabel(label_metric, fontsize=13, fontweight='bold')
ax.set_ylabel('Model', fontsize=13, fontweight='bold')
ax.set_title(
    f"So sánh hiệu năng Phân loại Cảm xúc — F1-Macro",
    fontsize=16,
    fontweight='bold',
    pad=20,
)

# LẬT TRỤC Y: Ép mô hình ở vị trí số 0 (F1 cao nhất) trồi lên trên cùng
ax.invert_yaxis()

# 6. Chỉnh trục X và lưới
# Cộng thêm một khoảng chừa chỗ cho text hiển thị số
ax.set_xlim(0.0, max(df_plot[metric_col]) + 0.06) 
ax.grid(axis='x', alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Lưu file
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.tight_layout()
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"✅ Đã vẽ và lưu biểu đồ thành công tại: {output_path}")