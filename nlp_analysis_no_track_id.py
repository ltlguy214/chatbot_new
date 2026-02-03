import pandas as pd

# 1. Đọc file dữ liệu
file_path = 'Audio_lyric/nlp_analysis.csv'
df = pd.read_csv(file_path)

# 2. Xóa cột 'track_id' nếu nó tồn tại
if 'track_id' in df.columns:
    df.drop(columns=['track_id'], inplace=True)
    print("Đã xóa cột 'track_id'.")
else:
    print("Cột 'track_id' không tồn tại trong file.")

# 3. Lưu lại file sau khi xóa (lưu ra file mới hoặc ghi đè tùy bạn)
# Ở đây mình lưu ra file mới để an toàn
output_path = 'nlp_analysis_no_track_id.csv'
df.to_csv(output_path, index=False)

print(f"File đã được lưu tại: {output_path}")
print("Các cột còn lại:", df.columns.tolist())