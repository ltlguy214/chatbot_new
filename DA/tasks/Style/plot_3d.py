import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import os

print("⏳ Đang tải dữ liệu từ VPop_5_Vibes_Final.csv...")
# Đọc dữ liệu cuối cùng đã được gán nhãn
file_path = 'DA/final_data/VPop_5_Vibes_Final.csv'
df = pd.read_csv(file_path)

# Tự động tìm cột chứa tên Vibe
vibe_col = 'Vibe' if 'Vibe' in df.columns else df.columns[-1]

# Bảng màu chuẩn đồng bộ với Radar Chart
color_map = {
    'Bùng nổ / Sôi động': '#1f77b4',  # Xanh dương
    'Tươi mới / Yêu đời': '#ff7f0e',  # Cam
    'Kịch tính / Da diết': '#2ca02c', # Xanh lá
    'Sâu lắng / Thấu cảm': '#d62728', # Đỏ
    'Bình yên / Chữa lành': '#9467bd' # Tím
}

# Kiểm tra các cột thông tin bài hát để hiển thị khi di chuột
hover_cols = []
if 'artists' in df.columns: hover_cols.append('artists')
hover_name = 'title' if 'title' in df.columns else None

# ====================================================================
# 1. VẼ BIỂU ĐỒ 3D TƯƠNG TÁC (ĐỂ DEMO)
# ====================================================================
print("🧮 Đang tạo không gian 3D tương tác với Đặc trưng thật...")
fig_html = px.scatter_3d(
    df, 
    x='tempo_bpm',          # Trục X: Tốc độ
    y='rms_energy',         # Trục Y: Năng lượng
    z='beat_strength_mean', # Trục Z: Lực nhịp
    color=vibe_col, 
    color_discrete_map=color_map,
    hover_name=hover_name, 
    hover_data=hover_cols,
    title='Phân bố 5 Vibes theo Không gian Âm thanh Thực tế',
    opacity=0.8
)

# Thu nhỏ các điểm dữ liệu để không bị dính vào nhau và thêm viền trắng cho sắc nét
fig_html.update_traces(marker=dict(size=3.5, line=dict(width=0.5, color='white')))
fig_html.update_layout(
    scene=dict(
        xaxis_title='Nhịp điệu (Tempo - BPM)',
        yaxis_title='Năng lượng (RMS Energy)',
        zaxis_title='Lực đánh của nhịp (Beat Strength)'
    ),
    margin=dict(l=0, r=0, b=0, t=40)
)

out_html = 'DA/tasks/Style/5vibes_Real_3D.html'
fig_html.write_html(out_html)
print(f"✅ Đã tạo biểu đồ 3D tương tác: {out_html}")


# ====================================================================
# 2. VẼ BIỂU ĐỒ TĨNH (ĐỂ CHỤP VÀO SLIDE / BÁO CÁO)
# ====================================================================
print("📸 Đang tạo ảnh 3D tĩnh...")
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

for vibe in df[vibe_col].unique():
    subset = df[df[vibe_col] == vibe]
    color = color_map.get(vibe, 'gray')
    ax.scatter(
        subset['tempo_bpm'], 
        subset['rms_energy'], 
        subset['beat_strength_mean'], 
        label=vibe, c=color, s=20, alpha=0.7, edgecolors='w', linewidth=0.3
    )

ax.set_title('Không gian 3D phân cụm 5 Vibes', fontsize=14, fontweight='bold')
ax.set_xlabel('Tempo (BPM)')
ax.set_ylabel('Energy')
ax.set_zlabel('Beat Strength')

# Xoay góc tối ưu để thấy rõ sự phân tách
ax.view_init(elev=25, azim=45) 
plt.legend(bbox_to_anchor=(1.15, 1), loc='upper left')

# Lưu ảnh với dpi vừa phải, KHÔNG dùng bbox_inches='tight' để tránh lỗi MemoryError
out_png = 'DA/tasks/Style/5vibes_Real_3D_static.png'
plt.savefig(out_png, dpi=150) 
print(f"✅ Đã tạo ảnh 3D tĩnh: {out_png}")
print("\n🎉 HOÀN TẤT! Hãy mở file HTML trên trình duyệt để xem thành quả!")