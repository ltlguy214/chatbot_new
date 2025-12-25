
import pandas as pd

# Đường dẫn file cần kiểm tra
FILE = 'final_data\\mergerd_balanced_and_features.csv'

# Đọc dữ liệu
try:
    df = pd.read_csv(FILE, encoding='utf-8-sig')
except Exception:
    df = pd.read_csv(FILE)

# Kiểm tra tổng số dòng và số dòng thiếu dữ liệu ở mỗi cột
missing_report = []
for col in df.columns:
    missing = df[col].isna().sum()
    missing_report.append((col, missing))

print(f'Tổng số dòng: {len(df)}')
print('Số dòng thiếu dữ liệu theo từng cột:')
for col, missing in missing_report:
    print(f'- {col}: {missing}')

# Xuất chi tiết các dòng có thiếu dữ liệu ra file
missing_rows = df[df.isna().any(axis=1)]
missing_rows.to_csv('final_data/mergerd_balanced_and_features_missing_rows.csv', index=False, encoding='utf-8-sig')
print(f'Đã xuất chi tiết các dòng thiếu dữ liệu ra final_data/mergerd_balanced_and_features_missing_rows.csv')