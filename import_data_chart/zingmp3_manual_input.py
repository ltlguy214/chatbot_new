import time
import sys
import os
import pandas as pd
from urllib.parse import quote_plus

# --- Imports giữ nguyên ---
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium_stealth import stealth

# ==============================================================================
# --- SETUP DRIVER ---
# ==============================================================================
def setup_driver():
    print("🚀 Khởi động Chế độ Nhập Tay (Manual Input)...")
    chrome_options = Options()
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    # Tắt dòng này để hiện thanh thông báo "Chrome is being controlled..." cho dễ nhìn
    # chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_argument("--start-maximized")
    
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)

    stealth(driver,
            languages=["en-US", "en"],
            vendor="Google Inc.",
            platform="Win32",
            webgl_vendor="Intel Inc.",
            renderer="Intel Iris OpenGL Engine",
            fix_hairline=True,
            )
    return driver

# ==============================================================================
# --- MAIN (CHẾ ĐỘ NHẬP TAY) ---
# ==============================================================================

def main():
    path_file = "bsides_non_hit.csv" # Kiểm tra lại đường dẫn file của bạn
    
    if not os.path.exists(path_file):
        # Check thư mục data/ nếu cần
        if os.path.exists("data/bsides_non_hit.csv"):
            path_file = "data/bsides_non_hit.csv"
        else:
            print(f"❌ Không tìm thấy file {path_file}")
            return

    print(f"📂 Đang đọc file: {path_file}")
    df = pd.read_csv(path_file)
    
    # Đảm bảo cột tồn tại
    if 'total_plays' not in df.columns:
        df['total_plays'] = None

    # Đếm số bài thiếu
    missing_df = df[df['total_plays'].isna() | (df['total_plays'] == '')]
    total_missing = len(missing_df)
    print(f"📉 Có {total_missing} bài còn thiếu dữ liệu.")
    print("="*60)
    print("HƯỚNG DẪN:")
    print("1. Trình duyệt sẽ mở trang tìm kiếm bài hát.")
    print("2. Bạn nhìn số liệu trên web.")
    print("3. Quay lại đây nhập con số đó vào và bấm Enter.")
    print("   - Gõ 'skip' để bỏ qua.")
    print("   - Gõ 'exit' để dừng và lưu file.")
    print("="*60)

    if total_missing == 0:
        print("✅ Tất cả đã đủ dữ liệu! Không cần nhập thêm.")
        return

    driver = setup_driver()
    
    try:
        for index, row in df.iterrows():
            # Chỉ xử lý những dòng thiếu dữ liệu
            current_val = row.get('total_plays')
            if pd.notna(current_val) and str(current_val).strip() != "":
                continue
            
            title = row['track_name']
            artist = row['artist_name']
            
            # 1. Mở trình duyệt tìm bài hát
            query = f"{title} {artist}"
            encoded_query = quote_plus(query)
            url = f"https://zingmp3.vn/tim-kiem/tat-ca?q={encoded_query}"
            
            driver.get(url)
            
            # 2. Hỏi người dùng nhập liệu
            print(f"\n🎵 Đang mở: {title} - {artist}")
            user_input = input("👉 Nhập số plays (hoặc Enter để bỏ qua): ").strip()
            
            # Xử lý lệnh thoát
            if user_input.lower() == 'exit':
                print("Đang dừng và lưu file...")
                break
            
            # Xử lý nhập liệu
            if user_input and user_input.lower() != 'skip':
                # Xóa dấu chấm/phẩy nếu bạn lỡ tay nhập (VD: 1.000 -> 1000)
                clean_input = user_input.replace('.', '').replace(',', '')
                
                df.at[index, 'total_plays'] = clean_input
                df.to_csv(path_file, index=False) # Lưu ngay lập tức
                print("✅ Đã lưu!")
            else:
                print("⏭️ Đã bỏ qua.")

    except Exception as e:
        print(f"❌ Lỗi: {e}")
    finally:
        print("💾 Đang lưu file lần cuối...")
        df.to_csv(path_file, index=False)
        driver.quit()
        print("✅ Hoàn tất!")

if __name__ == "__main__":
    main()