import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, accuracy_score

# 1. ĐỌC VÀ TIỀN XỬ LÝ DỮ LIỆU
df = pd.read_csv('final_thesis_evaluation_report.csv')

# Gộp category bị trùng tên
df['Category'] = df['Category'].replace('Edge Case', 'Edge Cases')

# Chuẩn hóa cột Evaluation: Chuyển các giá trị số (lỗi log) thành 'ERROR'
df['Evaluation'] = df['Evaluation'].apply(lambda x: 'PASS' if x == 'PASS' else ('FAIL' if x == 'FAIL' else 'ERROR'))
df['Is_Pass'] = df['Evaluation'].apply(lambda x: 1 if x == 'PASS' else 0)

# Chuẩn hóa cột thời gian (chuyển về dạng số, bỏ qua các giá trị lỗi)
df['Total Response Time (ms)'] = pd.to_numeric(df['Total Response Time (ms)'], errors='coerce')

# Xử lý các Intent bị trống hoặc lỗi để chạy classification_report
valid_intents = df['Expected Intent'].dropna().unique()
df_intent = df[df['Predicted Intent'].isin(valid_intents) & df['Expected Intent'].isin(valid_intents)]


print("="*70)
print("BÁO CÁO ĐÁNH GIÁ CHẤT LƯỢNG HỆ THỐNG (EVALUATION REPORT - CHƯƠNG 4)")
print("="*70)

# =====================================================================
# 1. ĐỘ ĐO THÀNH CÔNG TỔNG THỂ (PASS RATE)
# =====================================================================
total_cases = len(df)
pass_cases = df['Is_Pass'].sum()
overall_pass_rate = (pass_cases / total_cases) * 100

print("\n1. TỶ LỆ THÀNH CÔNG TỔNG THỂ (OVERALL PASS RATE):")
print(f"- Tổng số Test Case: {total_cases}")
print(f"- Số case PASS: {pass_cases}")
print(f"- Số case FAIL/ERROR: {total_cases - pass_cases}")
print(f"- Tỷ lệ thành công: {overall_pass_rate:.2f}%\n")

print("=> Phân tích theo Category:")
category_pass = df.groupby('Category')['Is_Pass'].agg(['count', 'sum'])
category_pass['pass_rate'] = (category_pass['sum'] / category_pass['count']) * 100
for index, row in category_pass.iterrows():
    print(f"   + {index}: {row['pass_rate']:.2f}% ({int(row['sum'])}/{int(row['count'])})")

# =====================================================================
# 2. ĐỘ ĐO PHÂN LOẠI Ý ĐỊNH (INTENT CLASSIFICATION METRICS)
# =====================================================================
print("\n" + "="*70)
print("2. ĐỘ CHÍNH XÁC NHẬN DIỆN Ý ĐỊNH (INTENT CLASSIFICATION METRICS):")
print("="*70)

# Tính Accuracy tổng
intent_accuracy = accuracy_score(df_intent['Expected Intent'], df_intent['Predicted Intent'])
print(f"- Độ chính xác tổng thể (Accuracy): {intent_accuracy * 100:.2f}%\n")

# Báo cáo chi tiết: Precision, Recall, F1-Score
# Chỉ lấy top các Intent chính để báo cáo không bị quá dài
report = classification_report(df_intent['Expected Intent'], df_intent['Predicted Intent'], zero_division=0)
print(report)

# =====================================================================
# 3. ĐỘ ĐO HIỆU NĂNG VÀ ĐỘ TRỄ (LATENCY METRICS)
# =====================================================================
print("="*70)
print("3. ĐỘ ĐO HIỆU NĂNG THỜI GIAN PHẢN HỒI (LATENCY METRICS):")
print("="*70)

latency = df['Total Response Time (ms)'].dropna()

print(f"- Thời gian phản hồi trung bình (Mean): {latency.mean():.2f} ms")
print(f"- Thời gian phản hồi trung vị (Median / P50): {latency.median():.2f} ms")
print(f"- Phân vị thứ 95 (P95 Latency - Quan trọng): {np.percentile(latency, 95):.2f} ms")
print(f"- Thời gian phản hồi chậm nhất (Max): {latency.max():.2f} ms")

print("\n=> Các Intent có thời gian phản hồi chậm nhất (Bottlenecks):")
avg_latency_by_intent = df.groupby('Expected Intent')['Total Response Time (ms)'].mean().sort_values(ascending=False).head(3)
for intent, time in avg_latency_by_intent.items():
    print(f"   + {intent}: {time:.2f} ms")

print("\n" + "="*70)
print("HOÀN TẤT ĐÁNH GIÁ!")