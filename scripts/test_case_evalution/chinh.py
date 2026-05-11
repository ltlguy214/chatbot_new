import pandas as pd
import os

def reset_vibe_testcases():
    # Cập nhật đường dẫn mới
    input_file = r"scripts\test_case_evalution\testcase_final.csv"
    report_file = r"scripts\test_case_evalution\final_thesis_evaluation_report.csv"

    # 1. Kiểm tra tồn tại TRƯỚC KHI đọc file
    if not os.path.exists(input_file):
        print(f"❌ Không tìm thấy file testcase tại: {input_file}")
        return

    df = pd.read_csv(input_file)
    reset_count = 0
    reset_ids = [] # Lưu lại ID của các câu bị reset

    # 2. Quét file testcase và đổi trạng thái
    for idx, row in df.iterrows():
        # Chỉ tập trung quét cột Extracted Entities
        entities = str(row.get('Extracted Entities', '')).strip().lower()
        
        # Nhận diện: Nếu Entities có chứa "mood=" hoặc "vibe="
        if 'mood=' in entities or 'vibe=' in entities:
            df.at[idx, 'Run_Status'] = 'PENDING'
            reset_ids.append(row.get('Test Case ID'))
            reset_count += 1

    # Lưu lại file testcase
    df.to_csv(input_file, index=False, encoding='utf-8-sig')
    print(f"✨ HOÀN TẤT: Đã tìm thấy và Reset {reset_count} Test Cases liên quan đến Vibe/Mood về 'PENDING'.")

    # 3. Dọn dẹp kết quả cũ trong file Report (Nếu file tồn tại)
    if os.path.exists(report_file) and len(reset_ids) > 0:
        try:
            df_report = pd.read_csv(report_file)
            # Lọc bỏ các dòng có Test Case ID trùng với những câu vừa bị reset
            df_report_cleaned = df_report[~df_report['Test Case ID'].isin(reset_ids)]
            df_report_cleaned.to_csv(report_file, index=False, encoding='utf-8-sig')
            print(f"🧹 Đã xóa {len(reset_ids)} kết quả cũ trong file report để chuẩn bị ghi kết quả mới.")
        except Exception as e:
            print(f"⚠️ Không thể dọn dẹp file report: {e}")

    print("👉 Hãy chạy lệnh `python test_e2e.py` để hệ thống tự động test lại các câu này!")

if __name__ == "__main__":
    reset_vibe_testcases()