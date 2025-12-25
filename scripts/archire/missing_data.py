import pandas as pd
import os

# 1. Xác định thư mục hiện tại của file script này (thư mục final_data)
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Cấu hình đường dẫn tệp - Trỏ thẳng vào file trong cùng thư mục
target_file = os.path.join(current_dir, 'balanced_500_500.csv')
report_file = os.path.join(current_dir, 'imputed_rows.csv')

# 3. Kiểm tra file có tồn tại không trước khi đọc
if not os.path.exists(target_file):
    print(f"LỖI: Không tìm thấy file {target_file}")
    print("Hãy đảm bảo file CSV nằm cùng thư mục với file script .py này.")
else:
    # 4. Đọc dữ liệu gốc
    df = pd.read_csv(target_file)

    # Đánh dấu các dòng bị thiếu để theo dõi
    mask_plays = df['total_plays'].isna()
    mask_streams = df['spotify_streams'].isna()
    any_missing = mask_plays | mask_streams

    print(f"Bắt đầu xử lý...")
    print(f"- Thiếu total_plays: {mask_plays.sum()} dòng")
    print(f"- Thiếu spotify_streams: {mask_streams.sum()} dòng")

    # 5. XỬ LÝ TOTAL_PLAYS (Grouped Median)
    # Ưu tiên 1: Theo Ca sĩ và trạng thái Hit
    df['total_plays'] = df['total_plays'].fillna(
        df.groupby(['artists', 'is_hit'])['total_plays'].transform('median')
    )
    # Ưu tiên 2: Theo trạng thái Hit (cho các ca sĩ không có mẫu khác)
    df['total_plays'] = df['total_plays'].fillna(
        df.groupby('is_hit')['total_plays'].transform('median')
    )

    # 6. XỬ LÝ SPOTIFY_STREAMS (Grouped Median)
    # Điền theo trạng thái Hit
    df['spotify_streams'] = df['spotify_streams'].fillna(
        df.groupby('is_hit')['spotify_streams'].transform('median')
    )

    # 7. LƯU DỮ LIỆU (Ghi đè trực tiếp vào file gốc)
    df.to_csv(target_file, index=False)

    # 8. Lưu riêng file báo cáo các dòng đã được điền
    imputed_rows = df[any_missing]
    imputed_rows.to_csv(report_file, index=False)

    print("-" * 30)
    print("KẾT QUẢ:")
    print(f"1. Đã CẬP NHẬT TRỰC TIẾP file tại: {target_file}")
    print(f"2. Đã lưu file báo cáo dòng đã điền tại: {report_file}")
    print(f"3. Tổng số dòng đã xử lý thành công: {any_missing.sum()}")
    print("Dữ liệu hiện tại đã sẵn sàng để huấn luyện mô hình (0 dòng missing).")