import time
import sys
import os
import re
import pandas as pd
from urllib.parse import quote_plus

# --- Imports từ script gốc ---
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By 
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager
from selenium_stealth import stealth

# ==============================================================================
# --- CẤU HÌNH SELECTORS (Giữ nguyên của bạn) ---
# ==============================================================================
SELECTORS = {
    "cookie_button": "//button[normalize-space()='Đồng ý']",
    "ad_modal_close": "//div[contains(@class, 'zm-modal-content')]//button[contains(@class, 'btn-close')]",
    "portal_overlay": "div.zm-portal",
    "context_menu": "div.zm-portal-menu, div.zm-context-menu",
    "search_result_section": "//h3[contains(text(), 'Bài Hát')]/ancestor::div[contains(@class, 'section')]",
    "media_item": "div.media-item",
    
    "more_button_paths": [
        ".//i[contains(@class,'ic-more')]/ancestor::button[1]",
        ".//button[contains(@aria-label,'Khác') or contains(@title,'Khác')]",
        ".//button[contains(@class,'more') or contains(@class,'btn-more') or contains(@class,'zm-actions-more')]",
        ".//span[contains(@class,'ic-more') or contains(@class,'icon-more') or contains(@class,'bi-three-dots')]/ancestor::button[1]",
    ]
}

# ==============================================================================
# --- CÁC HÀM HỖ TRỢ ---
# ==============================================================================

def setup_driver():
    """Thiết lập Chrome Driver với Stealth mode."""
    print("Đang khởi động Selenium (Stealth Mode)...")
    chrome_options = Options()
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    chrome_options.add_argument('--disable-blink-features=AutomationControlled')
    chrome_options.add_argument("--start-maximized")
    chrome_options.add_argument("--disable-notifications")

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

def handle_popups(driver):
    """Xử lý banner Cookie và modal Quảng cáo ban đầu."""
    try:
        WebDriverWait(driver, 5).until(
            EC.element_to_be_clickable((By.XPATH, SELECTORS["cookie_button"]))
        ).click()
        print("Đã đồng ý Cookie.")
    except Exception:
        pass

    try:
        WebDriverWait(driver, 5).until(
            EC.element_to_be_clickable((By.XPATH, SELECTORS["ad_modal_close"]))
        ).click()
        print("Đã đóng modal quảng cáo.")
    except Exception:
        pass

def parse_compact_number(text: str) -> int:
    """Chuyển đổi 1.2M -> 1200000 và cả số thường 500 -> 500."""
    if not text: return 0
    s = text.strip().replace(',', '.').upper()
    # [SỬA 1]: Làm sạch chuỗi, chỉ giữ số và KMB để tránh lỗi
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

def find_more_button(item):
    """Tìm nút 3 chấm trong item."""
    try:
        candidates = item.find_elements(By.XPATH, SELECTORS["more_button_paths"][0])
        visible = [b for b in candidates if 'is-hidden' not in (b.get_attribute('class') or '')]
        if visible: return visible[0]
        if candidates: return candidates[-1]
    except Exception: pass
        
    for xp in SELECTORS["more_button_paths"][1:]:
        try:
            btn = item.find_element(By.XPATH, xp)
            if btn: return btn
        except Exception: pass
    return None

def get_overlay_text_and_html(driver):
    """Lấy text từ popup overlay."""
    try:
        portal = WebDriverWait(driver, 3).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, SELECTORS["portal_overlay"]))
        )
        return portal.text
    except Exception: pass
    try:
        menu = driver.find_element(By.CSS_SELECTOR, SELECTORS["context_menu"])
        return menu.text
    except Exception: return ""

def get_likes_plays(driver, item_element):
    """Click 'More' -> Parse Likes/Plays."""
    more_btn = find_more_button(item_element)
    if not more_btn: return None, None
        
    panel_text = ""
    try:
        # Thử click bằng ActionChains hoặc JS (Giữ nguyên logic của bạn)
        try:
            ActionChains(driver).move_to_element(more_btn).pause(0.1).click(more_btn).perform()
        except Exception:
            driver.execute_script("arguments[0].click();", more_btn)
        
        time.sleep(1.0) # Đợi popup hiện
        panel_text = get_overlay_text_and_html(driver)
        
        # Đóng menu bằng ESC
        ActionChains(driver).send_keys(Keys.ESCAPE).perform()
        time.sleep(0.2)
            
    except Exception as e:
        print(f"  [Lỗi Menu]: {type(e).__name__}")

    if not panel_text: return None, None

    # [SỬA 2]: Thay đổi Regex để bắt được cả số không có K/M/B (thêm dấu ?)
    # Cũ: r"\b\d+(?:[\.,]\d+)?\s*[KMB]\b"
    # Mới: r"\b\d+(?:[\.,]\d+)?\s*[KMB]?\b"
    lines = panel_text.split('\n')
    candidates = []
    
    for line in lines:
        matches = re.findall(r"\b\d+(?:[\.,]\d+)?\s*[KMB]?\b", line, flags=re.IGNORECASE)
        for m in matches:
            val = parse_compact_number(m)
            # Lọc nhiễu: Nếu số < 20 và không có KMB thì bỏ qua (tránh bắt nhầm số thứ tự bài hát)
            has_suffix = re.search(r"[KMB]", m, re.IGNORECASE)
            if has_suffix or val > 20:
                candidates.append(val)
    
    # Sắp xếp giảm dần, số lớn nhất là Total Plays
    candidates.sort(reverse=True)
    
    total_likes, total_plays = None, None
    
    if candidates:
        total_plays = str(candidates[0]) # Lấy số to nhất làm Plays
        # Nếu có số thứ 2 thì gán làm Likes (không quan trọng lắm nhưng giữ format cũ)
        if len(candidates) > 1:
            total_likes = str(candidates[1])
            
    return total_likes, total_plays

# ==============================================================================
# --- HÀM MỚI: TÌM KIẾM VÀ LẤY SỐ LIỆU ---
# ==============================================================================

def scrape_song_plays(driver, title, artist):
    query = f"{title} {artist}"
    print(f"Đang tìm: {query}")
    
    encoded_query = quote_plus(query)
    url = f"https://zingmp3.vn/tim-kiem/tat-ca?q={encoded_query}"
    
    try:
        driver.get(url)
        # Đợi trang load
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div.zm-section, div.container"))
        )
        
        # Tìm section "Bài Hát"
        song_items = []
        try:
            section = driver.find_element(By.XPATH, SELECTORS["search_result_section"])
            song_items = section.find_elements(By.CSS_SELECTOR, SELECTORS["media_item"])
        except Exception:
            # Fallback: Lấy các media-item đầu tiên tìm thấy
            song_items = driver.find_elements(By.CSS_SELECTOR, SELECTORS["media_item"])
        
        if not song_items:
            print("  -> Không tìm thấy bài hát nào.")
            return None

        # Thử tối đa 3 kết quả đầu tiên
        for i, item in enumerate(song_items[:3]):
            try:
                # Scroll và Hover để nút More hiện ra (quan trọng)
                driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", item)
                ActionChains(driver).move_to_element(item).perform()
                time.sleep(0.2)
                
                _, plays = get_likes_plays(driver, item)
                
                if plays:
                    print(f"  -> Tìm thấy Plays: {plays}")
                    return plays
            except Exception as e:
                continue
        
        print("  -> Không lấy được plays từ các kết quả đầu (Có thể số quá nhỏ hoặc lỗi hiển thị).")
        return None

    except Exception as e:
        print(f"  -> Lỗi tìm kiếm: {e}")
        return None

# ==============================================================================
# --- MAIN ---
# ==============================================================================

def main():
    # [LƯU Ý] Đường dẫn file của bạn
    path_file = "data/bsides_non_hit.csv"
    
    if not os.path.exists(path_file):
        # Thử tìm ở thư mục hiện tại nếu không thấy trong data/
        if os.path.exists("bsides_non_hit.csv"):
            path_file = "bsides_non_hit.csv"
        else:
            print(f"Không tìm thấy file {path_file}")
            return

    print(f"Đang đọc file {path_file}...")
    df = pd.read_csv(path_file)
    
    # Thêm cột total_plays nếu chưa có
    if 'total_plays' not in df.columns:
        df['total_plays'] = None

    driver = setup_driver()
    
    try:
        # Vào trang chủ 1 lần để handle popup cookie/ads
        driver.get("https://zingmp3.vn")
        handle_popups(driver)
        
        for index, row in df.iterrows():
            # Nếu đã có dữ liệu thì bỏ qua (hữu ích khi chạy lại)
            if pd.notna(row.get('total_plays')) and str(row['total_plays']).strip() != "":
                continue
                
            title = row['track_name']
            artist = row['artist_name']
            
            plays = scrape_song_plays(driver, title, artist)
            
            if plays:
                df.at[index, 'total_plays'] = plays
                # Lưu liên tục đề phòng crash
                if index % 5 == 0:
                    df.to_csv(path_file, index=False)
                    print("💾 Checkpoint saved.")
            
            time.sleep(1) # Nghỉ nhẹ tránh block

    except Exception as e:
        print(f"Lỗi nghiêm trọng: {e}")
    finally:
        print("Đang lưu file kết quả...")
        df.to_csv(path_file, index=False)
        print("Đóng trình duyệt.")
        driver.quit()
        print(f"Hoàn tất! File lưu tại: {path_file}")

if __name__ == "__main__":
    main()