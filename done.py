import pandas as pd

# 1. Tải tệp CSV
df = pd.read_csv('testcase_final.csv')

# 2. Xóa chữ 'DONE' (thay thế bằng chuỗi rỗng) trên toàn bộ bảng dữ liệu
df_cleaned = df.replace('DONE', '')

# Hoặc nếu bạn chỉ muốn xóa ở một cột cụ thể (ví dụ cột Run_Status):
# df['Run_Status'] = df['Run_Status'].replace('DONE', '')

# 3. Lưu kết quả ra tệp mới
df_cleaned.to_csv('testcase_final.csv', index=False)

print("Đã xóa xong và lưu vào tệp testcase_final_cleaned.csv")