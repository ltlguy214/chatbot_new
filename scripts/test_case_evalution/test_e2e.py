import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
from playwright.sync_api import sync_playwright
import csv
import json
import re
import numpy as np
import sys
import os

def _read_template_csv_tolerant(path: str) -> pd.DataFrame:
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
    for name in possible_names:
        if name in row:
            return str(row[name])
    for k, v in row.items():
        if str(k).lower().strip() in [n.lower() for n in possible_names]:
            return str(v)
    return ""

def _get_actual_col_name(df, possible_names):
    for name in possible_names:
        if name in df.columns: return name
    for c in df.columns:
        if str(c).lower().strip() in [n.lower() for n in possible_names]: return c
    return None

def get_files_for_intent(intent, query):
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
        return []
        
    mp3_files = [f for f in os.listdir(base_dir) if f.lower().endswith(('.mp3', '.wav'))]
    if not mp3_files:
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
    input_file = r'scripts\test_case_evalution\testcase_final.csv'
    if len(sys.argv) >= 2 and str(sys.argv[1]).strip():
        input_file = str(sys.argv[1]).strip()
    
    df = _read_template_csv_tolerant(input_file)
    
    STATUS_COL = "Run_Status"
    if STATUS_COL not in df.columns:
        df[STATUS_COL] = ""
        df.to_csv(input_file, index=False, encoding='utf-8-sig')

    output_file = r'scripts\test_case_evalution\final_thesis_evaluation_report.csv'
    if len(sys.argv) >= 3 and str(sys.argv[2]).strip():
        output_file = str(sys.argv[2]).strip()
    
    # Ép chặt CỘT CHUẨN theo đúng yêu cầu của bạn
    columns_out = [
        'Test Case ID', 'User Query', 'Category', 
        'Expected Intent', 'Predicted Intent', 'Extracted Entities', 'Search Strategy',
        'Expected Result (ID)', 'Actual Result (ID)', 'Evaluation',
        'Intent Processing Time (ms)', 'Backend Processing Time (ms)', 'Total Response Time (ms)',
        'Run_Status'
    ]
    
    # Đọc file cũ và TỰ ĐỘNG CHẶT BỎ CÁC CỘT RÁC (No., Type, Notes, etc.)
    if os.path.exists(output_file):
        try:
            df_out = pd.read_csv(output_file, dtype=str)
            # Thêm cột nếu thiếu
            for col in columns_out:
                if col not in df_out.columns:
                    df_out[col] = ""
            # Ép lại đúng thứ tự và loại bỏ cột thừa
            df_out = df_out[columns_out]
        except pd.errors.EmptyDataError:
            df_out = pd.DataFrame(columns=columns_out)
    else:
        df_out = pd.DataFrame(columns=columns_out)
        
    df_out = df_out.astype(object)

    print(f"Bắt đầu quá trình kiểm thử có kiểm duyệt thủ công (Manual Review).")
    print(f"Nguồn: {input_file} | Kết quả lưu tại: {output_file}\n")

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

        page = _new_page()

        for index, row in df.iterrows():
            stt = _get_col_val(row, ['STT', 'No.', 'id']) or str(index + 1)
            test_id = f"TC{int(stt):02d}"
            
            if str(row.get(STATUS_COL, "")).strip().upper() == "DONE":
                print(f"⏭️ Bỏ qua [{test_id}] vì đã chạy trước đó.")
                continue

            query = _get_col_val(row, ['Query', 'User Query', 'Câu hỏi'])
            expected_intent = _get_col_val(row, ['Expected_Action', 'Expected Intent', 'Expected Action']).strip()
            # Đọc thêm cột Category từ file input để lưu sang báo cáo
            category = _get_col_val(row, ['Category', 'Phân loại', 'Loại', 'Nhóm'])
            
            need_to_type = True
            
            while True:
                expected_result_id = _get_col_val(df.loc[index], ['Expected_Result_ID', 'Expected Result (ID)', 'Expected ID']).strip()

                if need_to_type:
                    count_before = page.locator('div[data-testid="stChatMessage"]').count()

                    files_to_upload = get_files_for_intent(expected_intent, query)
                    if files_to_upload:
                        try:
                            page.locator('input[type="file"]').set_input_files(files_to_upload)
                            page.wait_for_timeout(500)
                        except Exception:
                            pass

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
                        page.wait_for_timeout(2000)
                else:
                    page.wait_for_timeout(500)

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
                except Exception:
                    pass

                exp_id_clean = "" if expected_result_id in ['nan', 'N/A', 'None'] else expected_result_id
                is_valid_id = bool(re.match(r'^[A-Za-z0-9]{22}$', actual_top))
                
                if is_valid_id:
                    actual_result = actual_top
                elif predicted_intent == "UI_BLOCKED":  # <-- UI đã chặn nên intent thực tế là UI_BLOCKED
                    actual_result = "UI Blocked" 
                elif predicted_intent == "GENERAL_CHAT":
                    actual_result = "No ID Needed"
                else:
                    actual_result = "Không tìm thấy"

                print("\n" + "-"*50)
                print(f"📌 KIỂM TRA & ĐÁNH GIÁ - [{test_id}]")
                print(f"   - Query: {query}")
                print(f"   - Intent dự đoán: {predicted_intent} (Mong muốn: {expected_intent})")
                print(f"   - ID thực tế: {actual_result} (ID cũ trong CSV: {exp_id_clean or '[Trống]'})")
                
                # --- [2] AUTO-PASS BỎ QUA KIỂM DUYỆT THỦ CÔNG ---
                is_auto_pass = False
                
                # NẾU CSV mong chờ MISSING_FILE VÀ Giao diện thực tế đã trả về UI_BLOCKED -> Tự động cho PASS
                if expected_intent == "MISSING_FILE" and predicted_intent == "UI_BLOCKED":
                    print(f"   ✅ AUTO-PASS: Hệ thống UI Guardrail đã đánh chặn thành công lỗi thiếu file!")
                    evaluation = "PASS"
                    is_auto_pass = True
                if is_auto_pass:
                    choice = 'y'
                else:
                    try:
                        choice = input("\n👉 QUYẾT ĐỊNH CỦA BẠN:\n"
                                       "   [y] ĐÁNH GIÁ PASS (Lưu kết quả & đi tiếp)\n"
                                       "   [n] ĐÁNH GIÁ FAIL (Lưu kết quả Fail & đi tiếp)\n"
                                       "   [r] LẤY KẾT QUẢ MỚI (Tôi vừa tự bấm Rerun trên app web)\n"
                                       "   [t] CHẠY LẠI TỪ ĐẦU (Script sẽ tự gõ lại query này)\n"
                                       "Lựa chọn của bạn: ").strip().lower()
                    except KeyboardInterrupt:
                        print("\n🛑 Dừng tiến trình.")
                        sys.exit(0)

                if choice == 'y':
                    if is_valid_id and actual_result != exp_id_clean:
                        exp_col_name = _get_actual_col_name(df, ['Expected_Result_ID', 'Expected Result (ID)', 'Expected ID'])
                        if exp_col_name:
                            df.at[index, exp_col_name] = actual_result
                            df.to_csv(input_file, index=False, encoding='utf-8-sig')
                            expected_result_id = actual_result
                            print(f"   ✅ Đã tự động cập nhật ID mới ({actual_result}) vào file testcase gốc!")
                    evaluation = "PASS"
                    break
                
                elif choice == 'n':
                    evaluation = "FAIL"
                    break
                
                elif choice == 'r':
                    print("🔄 Đang lấy lại kết quả mới nhất từ app web...")
                    need_to_type = False
                    continue
                    
                elif choice == 't':
                    print("🔄 Đang bắt đầu gõ lại query...")
                    need_to_type = True
                    continue
                    
                else:
                    print("⚠️ Lựa chọn không hợp lệ. Đánh dấu FAIL mặc định.")
                    evaluation = "FAIL"
                    break

            print(f"=> KẾT LUẬN: [{evaluation}] {test_id} ({total_lat}ms)")

            # GHÉP ĐÚNG THEO CỘT CHUẨN ĐÃ KHAI BÁO
            new_row_data = {
                'Test Case ID': test_id,
                'User Query': query,
                'Category': category,
                'Expected Intent': expected_intent,
                'Predicted Intent': predicted_intent,
                'Extracted Entities': extracted_entities,
                'Search Strategy': search_strategy,
                'Expected Result (ID)': expected_result_id,
                'Actual Result (ID)': actual_result,
                'Evaluation': evaluation,
                'Intent Processing Time (ms)': intent_lat,
                'Backend Processing Time (ms)': backend_lat,
                'Total Response Time (ms)': total_lat,
                'Run_Status': 'DONE'
            }

            if 'Test Case ID' in df_out.columns:
                df_out['Test Case ID'] = df_out['Test Case ID'].astype(str).str.strip()
            
            # Xóa các bản ghi cũ của test_case này để thay thế bằng cái mới
            df_out = df_out[df_out['Test Case ID'] != test_id.strip()]

            # Nối dữ liệu vào và gán lại dataframe
            df_out = pd.concat([df_out, pd.DataFrame([new_row_data])], ignore_index=True)

            # Khôi phục thứ tự chuẩn theo "TC01, TC02, ..."
            df_out['sort_key'] = df_out['Test Case ID'].str.extract(r'(\d+)').astype(float)
            df_out = df_out.sort_values('sort_key').drop(columns=['sort_key'])

            # Xuất ra file an toàn
            df_out.to_csv(output_file, index=False, encoding='utf-8-sig')

            df.at[index, STATUS_COL] = "DONE"
            df.to_csv(input_file, index=False, encoding='utf-8-sig')

        try:
            page.close()
        except Exception:
            pass

        browser.close()
    
    print("\n" + "="*50)
    print(" 📊 BÁO CÁO ĐÁNH GIÁ HIỆU NĂNG")
    print("="*50)
    
    df_report = pd.read_csv(output_file, dtype=str)
    total_cases = len(df_report)
    
    if total_cases > 0:
        pass_cases = len(df_report[df_report['Evaluation'] == 'PASS'])
        fail_cases = total_cases - pass_cases
        accuracy = (pass_cases / total_cases) * 100
        
        df_report['Total Response Time (ms)'] = pd.to_numeric(df_report['Total Response Time (ms)'], errors='coerce')
        lats = df_report['Total Response Time (ms)'].dropna().tolist()
        p95_latency = np.percentile(lats, 95) if lats else 0.0

        print(f"Total Test Cases: {total_cases}")
        print(f"Pass: {pass_cases} ({accuracy:.0f}%)")
        print(f"Fail: {fail_cases} ({(fail_cases/total_cases)*100:.0f}%)\n")
        print(f"P95 Total Latency: ~{p95_latency:.0f} ms")
    else:
        print("Không có dữ liệu test.")

if __name__ == "__main__":
    run_and_evaluate_test_cases()