import pandas as pd

# Đường dẫn file
file_main = 'features_for_model.csv'
file_add = 'features_for_model_1.csv'

# Đọc file chính, bỏ dòng lỗi định dạng nếu có
main = pd.read_csv(file_main, on_bad_lines='skip')
# Đọc file bổ sung, bỏ dòng lỗi định dạng nếu có
add = pd.read_csv(file_add, on_bad_lines='skip')

# Loại trùng file_name
merged = pd.concat([main, add[~add['file_name'].isin(main['file_name'])]], ignore_index=True)

# Ghi đè vào file chính
merged.to_csv(file_main, index=False)
print(f'Done! Đã merge trực tiếp vào {file_main} (bỏ qua dòng lỗi định dạng nếu có)')
