import pandas as pd

# Đường dẫn file
input_csv = 'final_data\\balanced_500_500.csv'
output_csv = 'Audio_lyric\\librosa_analysis.csv'

# Đọc danh sách file_name từ input và output
input_df = pd.read_csv(input_csv)
output_df = pd.read_csv(output_csv)

input_files = set(input_df['file_name'].dropna())
output_files = set(output_df['file_name'].dropna())

missing = input_files - output_files
extra = output_files - input_files

print(f'Tổng số bài hát trong input: {len(input_files)}')
print(f'Tổng số bài đã phân tích (output): {len(output_files)}')
print(f'Số bài còn thiếu: {len(missing)}')
print(f'Số bài dư (không có trong input): {len(extra)}')

if missing:
    with open('missing_files.txt', 'w', encoding='utf-8-sig') as f:
        for fn in sorted(missing):
            f.write(fn + '\n')
    print('Đã xuất danh sách thiếu: missing_files.txt')
else:
    print('Đã phân tích đủ tất cả các bài hát!')
