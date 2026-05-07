import pandas as pd
from playwright.sync_api import sync_playwright
import csv
from datetime import datetime
import json
import re
import numpy as np
import sys
import os

def _read_template_csv_tolerant(path: str) -> pd.DataFrame:
    """Read TC_TEMPLATES CSVs that may contain unquoted commas."""
    encodings = ["utf-8", "utf-8-sig"]
    last_err: Exception | None = None
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc, newline="") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if not header:
                    return pd.DataFrame()

                header = [str(h).strip() for h in header]
                expected_cols = len(header)
                rows: list[dict] = []
                for raw_row in reader:
                    if not raw_row or not any(str(c).strip() for c in raw_row):
                        continue
                    row = list(raw_row)
                    if len(row) > expected_cols:
                        row = row[: expected_cols - 1] + [",".join(row[expected_cols - 1 :])]
                    if len(row) < expected_cols:
                        row = row + [""] * (expected_cols - len(row))
                    rows.append({header[i]: row[i] for i in range(expected_cols)})

                return pd.DataFrame(rows)
        except Exception as e:
            last_err = e
            continue

    raise last_err or RuntimeError("Failed to read template CSV")

def _get_col_val(row, possible_names):
    """Hàm trích xuất an toàn chống KeyError."""
    for name in possible_names:
        if name in row:
            return str(row[name])
    for k, v in row.items():
        if str(k).lower().strip() in [n.lower() for n in possible_names]:
            return str(v)
    return ""

def _get_actual_col_name(df, possible_names):
    """Tìm tên cột thực tế trong DataFrame để ghi đè."""
    for name in possible_names:
        if name in df.columns: return name
    for c in df.columns:
        if str(c).lower().strip() in [n.lower() for n in possible_names]: return c
    return None

def get_files_for_intent(intent, query):
    """Quét và lấy file tự động dựa vào Intent."""
    base_dir = ""
    paths = []
    
    intent = str(intent).strip().upper()
    if intent == "SEARCH_AUDIO":
        base_dir = os.path.join("chatbot", "Test_app", "SEARCH_AUDIO")
    elif intent in ["ANALYZE_READY", "ANALYSIS_READY"]:
        base_dir = os.path.join("chatbot", "Test_app", "ANALYSIS_READY")
        if not os.path.exists(base_dir):
            base_dir = os.path.join("chatbot", "Test_app", "ANALYZE_READY")
    else:
        return []

    if not os.path.exists(base_dir):
        print(f"[WARN] Thư mục chứa file không tồn tại: {base_dir}")
        return []
        
    mp3_files = [f for f in os.listdir(base_dir) if f.lower().endswith(('.mp3', '.wav'))]
    if not mp3_files:
        print(f"[WARN] Không có file âm thanh (.mp3, .wav) nào trong {base_dir}")
        return []
        
    chosen_mp3 = mp3_files[0]
    for f in mp3_files:
        name_no_ext = os.path.splitext(f)[0]
        if f.lower() in query.lower() or name_no_ext.lower() in query.lower():
            chosen_mp3 = f
            break
            
    paths.append(os.path.join(base_dir, chosen_mp3))
    
    if intent in ["ANALYZE_READY", "ANALYSIS_READY"]:
        base_name = os.path.splitext(chosen_mp3)[0]
        txt_path = os.path.join(base_dir, f"{base_name}.txt")
        if os.path.exists(txt_path):
            paths.append(txt_path)
        else:
            print(f"[WARN] Lệnh Analyze nhưng không tìm thấy file lyrics '{base_name}.txt' đi kèm.")
            
    return paths

def _wait_for_streamlit_ready(page) -> None:
    for _ in range(6):
        try:
            page.wait_for_selector('textarea', timeout=30000)
            return
        except Exception:
            try:
                page.wait_for_timeout(1500)
                page.reload(wait_until='domcontentloaded')
            except Exception:
                pass
    page.wait_for_selector('textarea', timeout=60000)

def run_and_evaluate_test_cases():
    input_file = r'testcase_final_top10.csv'
    if len(sys.argv) >= 2 and str(sys.argv[1]).strip():
        input_file = str(sys.argv[1]).strip()
    
    df = _read_template_csv_tolerant(input_file)
    
    STATUS_COL = "Run_Status"
    if STATUS_COL not in df.columns:
        df[STATUS_COL] = ""
        df.to_csv(input_file, index=False, encoding='utf-8-sig')

    output_file = 'final_thesis_evaluation_report.csv'
    if len(sys.argv) >= 3 and str(sys.argv[2]).strip():
        output_file = str(sys.argv[2]).strip()
    
    # Ghi header ĐÃ ĐƯỢC LÀM SẠCH và đẩy Evaluation xuống cuối
    if not os.path.exists(output_file):
        with open(output_file, mode='w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Test Case ID', 'No.', 'User Query', 
                'Expected Intent', 'Predicted Intent', 'Extracted Entities', 'Search Strategy',
                'Expected Result (ID)', 'Actual Result (ID)', 
                'Intent Processing Time (ms)', 'Backend Processing Time (ms)', 'Total Response Time (ms)', 'Notes & Error Analysis',
                'Evaluation'
            ])
    
    print(f"Bắt đầu quá trình kiểm thử bán tự động (Semi-Auto) + Auto File Upload + Resume.")
    print(f"Nguồn: {input_file} | Kết quả lưu tại: {output_file}")

    note_mapping = {
        r'\[Exact Match\]': 'Truy vấn chính xác',
        r'\[Fuzzy\]': 'Kiểm thử sai chính tả',
        r'\[Regex Noise\]': 'Nhiễu cú pháp',
        r'\[Lớp 2 Fallback Test\]': 'Kiểm thử cơ chế dự phòng',
        r'\[Cross-Intent Noise\]': 'Nhiễu ý định phân loại',
        r'\[No accents\]': 'Văn bản không dấu',
        r'\[Foreign chars\]': 'Ký tự ngoại ngữ',
        r'\[Partial Match\]': 'Truy vấn một phần',
        r'\[Number/Text Convert\]': 'Chuyển đổi định dạng số',
        r'\[Number/Text\]': 'Chuyển đổi định dạng số',
        r'\[English title\]': 'Truy vấn ngoại ngữ',
        r'\[Punctuation\]': 'Nhiễu dấu câu',
        r'\[Short Title\]': 'Truy vấn quá ngắn',
        r'\[Noise Suffix\]': 'Nhiễu hậu tố',
        r'\[Entity Ambiguity\]': 'Thực thể mơ hồ',
        r'\[Special Chars\]': 'Ký tự đặc biệt',
        r'\[Ambiguous\]': 'Ngữ nghĩa mơ hồ',
        r'\[Artist Suffix\]': 'Nhiễu hậu tố nghệ sĩ'
    }

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)

        def _new_page():
            page = browser.new_page()
            last_err: Exception | None = None
            for _ in range(12):
                try:
                    page.goto("http://localhost:8501", wait_until='domcontentloaded', timeout=30000)
                    last_err = None
                    break
                except Exception as e:
                    last_err = e
                    try:
                        page.wait_for_timeout(1500)
                    except Exception:
                        pass
            if last_err is not None:
                raise last_err
            _wait_for_streamlit_ready(page)
            return page

        print("Đang khởi động Streamlit... Vui lòng đợi...")
        page = _new_page()
        print("Khởi động xong! Bắt đầu xả đạn test case 🚀\n")

        for index, row in df.iterrows():
            stt = _get_col_val(row, ['STT', 'No.', 'id']) or str(index + 1)
            test_id = f"TC{int(stt):02d}"
            
            if str(row.get(STATUS_COL, "")).strip().upper() == "DONE":
                print(f"⏭️ Bỏ qua [{test_id}] vì đã chạy trước đó.")
                continue

            query = _get_col_val(row, ['Query', 'User Query', 'Câu hỏi'])
            expected_intent = _get_col_val(row, ['Expected_Action', 'Expected Intent', 'Expected Action']).strip()
            expected_result_id = _get_col_val(row, ['Expected_Result_ID', 'Expected Result (ID)', 'Expected ID']).strip()
            raw_note = _get_col_val(row, ['Mục đích test', 'Note', 'Ghi chú', 'Mô tả'])

            academic_note = raw_note
            for pat, rep in note_mapping.items():
                academic_note = re.sub(pat, rep, academic_note, flags=re.IGNORECASE)

            count_before = page.locator('div[data-testid="stChatMessage"]').count()

            files_to_upload = get_files_for_intent(expected_intent, query)
            if files_to_upload:
                try:
                    page.locator('input[type="file"]').set_input_files(files_to_upload)
                    print(f"📎 [Upload] Đã tự động đính kèm: {[os.path.basename(f) for f in files_to_upload]}")
                    page.wait_for_timeout(500)
                except Exception as e:
                    print(f"⚠️ [Upload Error] Không thể tải file: {e}")

            chat_input = page.locator('textarea')
            chat_input.fill(query)
            chat_input.press("Enter")

            try:
                page.wait_for_function(f"""
                    () => {{
                        const bubbles = document.querySelectorAll('div[data-testid="stChatMessage"]');
                        if (bubbles.length < {count_before + 2}) return false;

                        const last = bubbles[bubbles.length - 1];
                        const noSpinner = !last.querySelector('[data-testid="stSpinner"]');
                        const hasMeta = document.querySelector('#test-metadata');

                        return noSpinner && hasMeta;
                    }}
                """, timeout=60000)
            except:
                print(f"[WARN] Timeout nhẹ ở: {query}")
                page.wait_for_timeout(2000)

            actual_result = "N/A"
            evaluation = "FAIL"
            predicted_intent = "N/A"
            extracted_entities = "None"
            search_strategy = "N/A"

            intent_lat = backend_lat = total_lat = 0.0
            actual_top = ""

            try:
                if page.locator('#test-metadata').count() > 0:
                    meta_text = page.locator('#test-metadata').last.inner_text()
                    meta_data = json.loads(meta_text)

                    predicted_intent = meta_data.get("action", "N/A")
                    search_strategy = meta_data.get("search_strategy", "N/A")

                    params = meta_data.get("params", {})
                    extracted_entities = "; ".join([f"{k}={v}" for k, v in params.items() if v]) or "None"

                    intent_lat = round(meta_data.get("intent_ms", 0), 2)
                    backend_lat = round(meta_data.get("backend_ms", 0), 2)
                    total_lat = round(meta_data.get("total_ms", 0), 2)
                    
                    actual_top = str(meta_data.get('top_track_id') or '').strip()
            except Exception as e:
                print(f"[META ERROR] {e}")

            try:
                last_message_html = page.locator('div[data-testid="stChatMessage"]').last.inner_html()
            except:
                last_message_html = ""

            exp_id_clean = "" if expected_result_id in ['nan', 'N/A', 'None'] else expected_result_id
            is_valid_id = bool(re.match(r'^[A-Za-z0-9]{22}$', actual_top))
            
            if is_valid_id:
                actual_result = actual_top
            else:
                actual_result = "Không trả về ID" if predicted_intent in ["GENERAL_CHAT", "MISSING_FILE"] else "Không tìm thấy"

            if is_valid_id and actual_result != exp_id_clean:
                print(f"\n⚠️ PHÁT HIỆN ID MỚI CHO TEST CASE [{test_id}]:")
                print(f"   - Query: {query}")
                print(f"   - Hành động dự đoán: {predicted_intent}")
                print(f"   - ID mong đợi (cũ): {exp_id_clean or '[Trống]'}")
                print(f"   - ID thực tế (Top 1 mới): {actual_result}")
                
                try:
                    choice = input("👉 Cập nhật ID này vào CSV? (y/n - enter để bỏ qua): ").strip().lower()
                except KeyboardInterrupt:
                    print("\n🛑 Đã dừng tiến trình kiểm thử bằng Ctrl+C.")
                    browser.close()
                    sys.exit(0)
                    
                if choice == 'y':
                    exp_col_name = _get_actual_col_name(df, ['Expected_Result_ID', 'Expected Result (ID)', 'Expected ID'])
                    if exp_col_name:
                        df.at[index, exp_col_name] = actual_result
                        df.to_csv(input_file, index=False, encoding='utf-8-sig')
                        print("✅ Đã ghi nhận ID mới!")
                        
                        expected_result_id = actual_result
                        exp_id_clean = actual_result
                    else:
                        print("❌ Không tìm thấy cột ID trong CSV để ghi.")

            found = False
            if exp_id_clean:
                try:
                    if 'meta_data' in locals() and isinstance(meta_data, dict):
                        top_ids = meta_data.get('top_track_ids', [])
                        if isinstance(top_ids, list) and exp_id_clean in [str(x).strip() for x in top_ids]:
                            found = True
                        elif actual_top == exp_id_clean:
                            found = True
                except:
                    pass

                if not found and exp_id_clean in (last_message_html or ''):
                    found = True

                if found:
                    actual_result = exp_id_clean
                    if predicted_intent == expected_intent:
                        evaluation = "PASS"
                else:
                    evaluation = "FAIL"
            else:
                if predicted_intent == expected_intent:
                    evaluation = "PASS"
                    actual_result = "Đúng hành động"
                else:
                    actual_result = "Sai hành động"

            if total_lat > 8000:
                academic_note += " | [Outlier] Latency cao bất thường"

            if evaluation == "FAIL":
                if predicted_intent != expected_intent:
                    academic_note += " | [Intent Error]"
                elif search_strategy == "N/A":
                    academic_note += " | [Search Failed]"
                else:
                    academic_note += " | [Ranking Error]"

            print(f"[{evaluation}] {test_id} ({total_lat}ms) - {search_strategy}")

            # Đã đẩy evaluation xuống cuối cùng
            with open(output_file, mode='a', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow([
                    test_id, stt, query,
                    expected_intent, predicted_intent, extracted_entities, search_strategy,
                    expected_result_id, actual_result,
                    intent_lat, backend_lat, total_lat, academic_note,
                    evaluation
                ])

            df.at[index, STATUS_COL] = "DONE"
            df.to_csv(input_file, index=False, encoding='utf-8-sig')

        try:
            page.close()
        except Exception:
            pass

        browser.close()
    
    print("\n" + "="*50)
    print(" 📊 BÁO CÁO ĐÁNH GIÁ HIỆU NĂNG HỆ THỐNG (SYSTEM KPI)")
    print("="*50)
    
    df_report = pd.read_csv(output_file)
    total_cases = len(df_report)
    
    if total_cases > 0:
        pass_cases = len(df_report[df_report['Evaluation'] == 'PASS'])
        fail_cases = total_cases - pass_cases
        accuracy = (pass_cases / total_cases) * 100
        
        avg_intent = df_report['Intent Processing Time (ms)'].mean()
        avg_backend = df_report['Backend Processing Time (ms)'].mean()
        avg_total = df_report['Total Response Time (ms)'].mean()
        
        lats = df_report['Total Response Time (ms)'].dropna().tolist()
        p95_latency = np.percentile(lats, 95) if lats else 0.0
        
        outliers = df_report[df_report['Total Response Time (ms)'] > 8000]['Test Case ID'].tolist()
        outliers_str = ", ".join(outliers) if outliers else "None"

        print(f"Total Test Cases: {total_cases}")
        print(f"Pass: {pass_cases} ({accuracy:.0f}%)")
        print(f"Fail: {fail_cases} ({(fail_cases/total_cases)*100:.0f}%)\n")
        
        print(f"Average Intent Latency: ~{avg_intent:.1f} ms")
        print(f"Average Backend Latency: ~{avg_backend:.0f} ms")
        print(f"Average Total Latency: ~{avg_total:.0f} ms\n")
        
        print(f"P95 Total Latency: ~{p95_latency:.0f} ms")
        print(f"Outliers (>8s): {outliers_str}")
        
    else:
        print("Không có dữ liệu test.")

    fail_ids = df_report[df_report['Evaluation'] == 'FAIL']['Test Case ID'].tolist()
    if fail_ids:
        print("\nTest Case FAIL IDs for further analysis:")
        print(", ".join(fail_ids))
        
    print(f"\nBáo cáo CSV chuẩn học thuật: {output_file}")

if __name__ == "__main__":
    run_and_evaluate_test_cases()