import time
import sys
import os
import pandas as pd
import re
from urllib.parse import quote_plus

# --- Imports Selenium ---
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium_stealth import stealth

# ==============================================================================
# --- SETUP DRIVER ---
# ==============================================================================
def setup_driver():
    print("🚀 Khởi động trình duyệt để nhập liệu NCT...")
    chrome_options = Options()
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
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

# Hàm hỗ trợ parse số nếu bạn lỡ nhập "1.5K"
def parse_compact_input(text: str) -> int:
    if not text: return 0
    s = text.strip().replace(',', '.').upper()
    # Giữ số và KMB
    s = re.sub(r"[^0-9KMB\.]", "", s)
    m = re.search(r"(\d+(?:\.\d+)?)\s*([KMB]?)", s)
    if not m:
        digits = re.findall(r"\d+", s)
        return int(digits[0]) if digits else 0
    val = float(m.group(1))
    suf = m.group(2)
    mult = 1
    if suf == 'K': mult = 1_000
    elif suf == 'M': mult = 1_000_000
    elif suf == 'B': mult = 1_000_000_000
    return int(val * mult)

# ==============================================================================
# --- MAIN (CHẾ ĐỘ NHẬP TAY CHO NCT) ---
# ==============================================================================

def main():
    # Đường dẫn file
    path_file = "bsides_non_hit.csv"
    
    if not os.path.exists(path_file):
        if os.path.exists("data/bsides_non_hit.csv"):
            path_file = "data/bsides_non_hit.csv"
        else:
            print(f"❌ Không tìm thấy file {path_file}")
            return

    print(f"📂 Đang đọc file: {path_file}")
    df = pd.read_csv(path_file)
    
    # Đảm bảo cột total_likes_nct tồn tại
    if 'total_likes_nct' not in df.columns:
        df['total_likes_nct'] = None

    # Lọc các bài thiếu dữ liệu
    missing_mask = df['total_likes_nct'].isna() | (df['total_likes_nct'] == '') | (df['total_likes_nct'].astype(str) == '-1')
    missing_df = df[missing_mask]
    
    total_missing = len(missing_df)
    print(f"📉 Tìm thấy {total_missing} bài NCT cần nhập tay.")
    
    if total_missing == 0:
        print("✅ Dữ liệu NCT đã đầy đủ!")
        return

    print("="*60)
    print("📖 HƯỚNG DẪN (NCT MODE):")
    print("1. Web sẽ mở trang tìm kiếm.")
    print("2. Nhìn số liệu trên web.")
    print("3. Quay lại đây NHẬP SỐ.")
    print("   - Bấm ENTER (không cần gõ gì) để BỎ QUA.")
    print("   - Gõ 'e' để LƯU và THOÁT.")
    print("="*60)
    
    driver = setup_driver()
    
    try:
        for index, row in df.iterrows():
            # Chỉ xử lý dòng thiếu
            current_val = row.get('total_likes_nct')
            if pd.notna(current_val) and str(current_val).strip() != "" and str(current_val) != "-1":
                continue
            
            title = row['track_name']
            artist = row['artist_name']
            
            # URL tìm kiếm NCT (Tổng hợp)
            query = f"{title} {artist}"
            encoded_query = quote_plus(query)
            url = f"https://www.nhaccuatui.com/tim-kiem?q={encoded_query}"
            
            print(f"\n🎵 [{index}] Đang mở: {title} - {artist}")
            driver.get(url)
            
            # Vòng lặp nhập liệu
            while True:
                # Nhắc nhở: Enter=Skip
                user_input = input("👉 Nhập Likes (Enter=skip, e=exit): ").strip()
                
                # 1. Xử lý thoát
                if user_input.lower() in ['e', 'exit']:
                    print("💾 Đang lưu file...")
                    df.to_csv(path_file, index=False)
                    driver.quit()
                    print("✅ Đã thoát.")
                    return

                # 2. Xử lý Skip (Bấm Enter hoặc gõ s)
                if user_input == "" or user_input.lower() in ['s', 'skip']:
                    print("⏭️  Bỏ qua.")
                    break
                
                # 3. Xử lý nhập số
                try:
                    clean_val = parse_compact_input(user_input)
                    df.at[index, 'total_likes_nct'] = clean_val
                    df.to_csv(path_file, index=False) # Lưu ngay
                    print(f"✅ Đã lưu: {clean_val}")
                    break # Sang bài tiếp theo
                except Exception:
                    print("⚠️ Lỗi định dạng số, vui lòng nhập lại!")

    except Exception as e:
        print(f"❌ Lỗi: {e}")
    finally:
        print("💾 Lưu lần cuối...")
        df.to_csv(path_file, index=False)
        driver.quit()
        print("✅ Hoàn tất!")

if __name__ == "__main__":
    main()