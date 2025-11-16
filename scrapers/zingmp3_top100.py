import time
import sys
import os
import csv
import re
from datetime import datetime 

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.common.by import By 
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.common.action_chains import ActionChains
    from selenium.webdriver.common.keys import Keys
    from bs4 import BeautifulSoup
    from webdriver_manager.chrome import ChromeDriverManager
    from selenium_stealth import stealth
except Exception as e:
    print("Missing required package or import failed:", e)
    print("Install required packages with:\n    pip install selenium webdriver-manager beautifulsoup4 selenium-stealth")
    sys.exit(1)

# ==============================================================================
# --- CÀI ĐẶT VÀ HẰNG SỐ ---
# ==============================================================================

# Ghi đè nghệ sĩ (dùng khi Zing MP3 hiển thị sai)
OVERRIDES = {
    "không đau nữa rồi": ("EM XINH SAY HI", "Orange, Châu Bùi, Mỹ Mỹ, Pháp Kiều, 52Hz"),
    "cứ đổ tại cơn mưa": ("EM XINH SAY HI", "Phương Ly, Orange, Châu Bùi, 52Hz, Vũ Thảo My"),
    "gã săn cá": ("EM XINH SAY HI", "Lâm Bảo Ngọc, Quỳnh Anh Shyn, MAIQUINN, Saabirose, Quang Hùng MasterD"),
    "đoạn kịch câm": ("ANH TRAI SAY HI", "CONGB, Negav, B Ray, CODYNAMVO, Phạm Đình Thái Ngân, Mỹ Mỹ"),
    "người như anh xứng đáng cô đơn": ("ANH TRAI SAY HI", "Vũ Cát Tường, Karik, Negav, Ngô Kiến Huy, Jey B"),
    "hơn là bạn": ("ANH TRAI SAY HI", "Ngô Kiến Huy, Vũ Cát Tường, Karik, VƯƠNG BÌNH, Sơn.K, MIN"),
    "hermosa": ("ANH TRAI SAY HI", "Sơn.K, buitruonglinh, Tez, Mason Nguyen, CONGB"),
    "sớm muộn thì": ("ANH TRAI SAY HI", "Hustlang Robber, Nhâm Phương Nam, Mason nguyen, Jaysonlei, Khoi Vu, LAMOON"),
    "đa nghi": ("ANH TRAI SAY HI", "Negav, Hải Nam, CODYNAMVO, Dillan Hoàng Phan"),
    "người yêu chưa sinh ra": ("ANH TRAI SAY HI", "Ogenus, BigDaddy, Phúc Du, HUSTLANG Robber, Dillan Hoàng Phan"),
    "make up": ("ANH TRAI SAY HI", "Dillan Hoàng Phan, buitruonglinh, Đỗ Nam Sơn, Lohan, Ryn Lee, Bảo Thy"),
    "dẫu có đến đâu": ("ANH TRAI SAY HI", "CONGB, VƯƠNG BÌNH, Lohan, Nhâm Phương Nam"),
    "rơi tự do": ("EM XINH SAY HI", "LyHan"),
}

# Các mẫu tiêu đề dùng để LỌC BỎ khỏi kết quả
SKIP_PATTERNS = [
    r'\bremix\b', r'\bcover\b', r'\(AI Version\)', r'\(AI\)', r'\blive\b',
    r'\blive version\b', r'\bacoustic version\b', r'\binstrumental\b',
    r'\bkaraoke\b', r'\bremaster\b', r'\bRAP VERSION\b', r'\bEDIT\b',
    r'\bVERSION\b', r'\bca sĩ giấu mặt\b',
]

# Gom các selector (CSS/XPath) vào một chỗ để dễ bảo trì
SELECTORS = {
    "cookie_button": "//button[normalize-space()='Đồng ý']",
    "ad_modal_close": "//div[contains(@class, 'zm-modal-content')]//button[contains(@class, 'btn-close')]",
    "chart_item": "div.chart-song-item",
    "view_top_100": "//button[normalize-space()='Xem top 100']",
    "last_song_rank": "//span[@class='number' and text()='100']",
    "portal_overlay": "div.zm-portal",
    "context_menu": "div.zm-portal-menu, div.zm-context-menu",
    "item_rank": "span.number",
    "item_title": "a.item-title",
    "item_duration": "div.duration",
    "item_subtitle": "h3.subtitle",
    # Các XPaths khác nhau để tìm nút "More" (3 chấm)
    "more_button_paths": [
        ".//i[contains(@class,'ic-more')]/ancestor::button[1]",
        ".//button[contains(@aria-label,'Khác') or contains(@title,'Khác')]",
        ".//button[contains(@class,'more') or contains(@class,'btn-more') or contains(@class,'zm-actions-more')]",
        ".//span[contains(@class,'ic-more') or contains(@class,'icon-more') or contains(@class,'bi-three-dots')]/ancestor::button[1]",
    ]
}

# --- Định nghĩa Schema Header ---
# Schema 10 cột MỚI (đây là schema mục tiêu)
NEW_HEADER = [
    'Date', 'Rank', 'Title', 'Artist', 'Featured_Artists', 'Source', 
    'Duration', 'URL', 'Total_Likes', 'Total_Plays'
]

# --- Các schema CŨ cần kiểm tra để nâng cấp ---
# Định nghĩa phần cột cơ sở (giống nhau ở tất cả các schema cũ)
_BASE_OLD_COLS = ['Date', 'Rank', 'Title', 'Artists', 'Duration', 'URL']

OLD_HEADER_7 = _BASE_OLD_COLS + ['Total_Likes']
OLD_HEADER_8 = _BASE_OLD_COLS + ['Total_Likes', 'Total_Plays']
OLD_HEADER_10 = _BASE_OLD_COLS + ['Total_Likes', 'Total_Plays', 'Genre', 'Album']
OLD_HEADER_11 = _BASE_OLD_COLS + ['Total_Likes', 'Total_Plays', 'Release_Date', 'Genre', 'Album']

# Đóng gói các header cũ lại để dễ kiểm tra
OLD_HEADERS = {
    tuple(OLD_HEADER_7): 7,
    tuple(OLD_HEADER_8): 8,
    tuple(OLD_HEADER_10): 10,
    tuple(OLD_HEADER_11): 11,
}

# ==============================================================================
# --- HÀM TIỆN ÍCH (Utilities) ---
# ==============================================================================

def parse_compact_number(text: str) -> int:
    """Chuyển đổi chuỗi dạng 210K, 9M, 1.5B -> số nguyên."""
    if not text:
        return 0
    s = text.strip().replace(',', '.').upper()
    m = re.search(r"(\d+(?:\.\d+)?)\s*([KMB]?)", s)
    if not m:
        digits = re.findall(r"\d+", s)
        return int(digits[0]) if digits else 0
    val = float(m.group(1))
    suf = m.group(2)
    mult = 1
    if suf == 'K':
        mult = 1_000
    elif suf == 'M':
        mult = 1_000_000
    elif suf == 'B':
        mult = 1_000_000_000
    return int(val * mult)

def check_date_exists(filepath, date_to_check):
    """Kiểm tra xem một ngày cụ thể đã tồn tại trong cột 'Date' của file CSV chưa."""
    try:
        dir_name = os.path.dirname(filepath)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        with open(filepath, 'r', encoding='utf-8-sig') as f:
            for i, line in enumerate(f):
                if i == 0: continue
                if line.startswith(f"{date_to_check},"):
                    return True
    except FileNotFoundError:
        print(f"File {filepath} chưa tồn tại. Sẽ tạo file mới.")
        return False
    except Exception as e:
        print(f"Lỗi khi kiểm tra file CSV ({type(e).__name__}). Bỏ qua kiểm tra.")
        return False
    return False

def get_csv_header(filepath):
    """Đọc và trả về header (dạng list) từ file CSV."""
    try:
        with open(filepath, 'r', encoding='utf-8-sig') as rf:
            first_line = rf.readline()
        return [c.strip() for c in first_line.strip().split(',')]
    except Exception:
        return None

# ==============================================================================
# --- HÀM XỬ LÝ CSV (Migration & Saving) ---
# ==============================================================================

def migrate_csv_schema(filepath, header_cols):
    """
    Nâng cấp file CSV từ schema cũ sang schema 10 cột (NEW_HEADER).
    Đây là logic phức tạp bạn đã viết, được đóng gói lại.
    """
    header_tuple = tuple(header_cols)
    if header_tuple not in OLD_HEADERS:
        print(f"Phát hiện header lạ, không rõ cách nâng cấp. Sẽ tạo backup và ghi đè.")
        # Logic dự phòng: backup và ghi đè
    
    schema_version = OLD_HEADERS.get(header_tuple, 0)
    print(f"Phát hiện schema {schema_version} cột. Đang nâng cấp lên schema 10 cột...")

    backup_file = filepath + '.bak'
    os.replace(filepath, backup_file)
    rows_old = []

    try:
        with open(backup_file, 'r', encoding='utf-8-sig') as rfb:
            reader = csv.reader(rfb)
            next(reader, None)  # Bỏ qua header cũ
            
            for r in reader:
                if not r: continue # Bỏ qua dòng trống
                
                # Logic chuyển đổi dựa trên schema cũ
                # Tất cả đều phải chuyển đổi sang NEW_HEADER
                # [Date, Rank, Title, Artist, Featured_Artists, Source, Duration, URL, Total_Likes, Total_Plays]
                
                artists_combined = r[3] if len(r) > 3 else ''
                artist_list = [a.strip() for a in artists_combined.split(',') if a.strip()]
                main_artist = artist_list[0] if artist_list else ''
                featured = ', '.join(artist_list[1:]) if len(artist_list) > 1 else ''

                if schema_version == 7:
                    # r = [Date, Rank, Title, Artists, Duration, URL, Total_Likes]
                    new_row = [
                        r[0], r[1], r[2], main_artist, featured, 'ZINGMP3',
                        r[4] if len(r) > 4 else '', r[5] if len(r) > 5 else '',
                        r[6] if len(r) > 6 else '', "" # Thêm Total_Plays (trống)
                    ]
                elif schema_version == 8:
                    # r = [Date, Rank, Title, Artists, Duration, URL, Total_Likes, Total_Plays]
                    new_row = [
                        r[0], r[1], r[2], main_artist, featured, 'ZINGMP3',
                        r[4] if len(r) > 4 else '', r[5] if len(r) > 5 else '',
                        r[6] if len(r) > 6 else '', r[7] if len(r) > 7 else ''
                    ]
                elif schema_version == 10:
                    # r = [Date, Rank, Title, Artists, Duration, URL, Total_Likes, Total_Plays, Genre, Album]
                    # Bỏ 2 cột cuối
                    new_row = [
                        r[0], r[1], r[2], main_artist, featured, 'ZINGMP3',
                        r[4] if len(r) > 4 else '', r[5] if len(r) > 5 else '',
                        r[6] if len(r) > 6 else '', r[7] if len(r) > 7 else ''
                    ]
                elif schema_version == 11:
                    # r = [Date, Rank, Title, Artists, Duration, URL, Total_Likes, Total_Plays, Release_Date, Genre, Album]
                    # Bỏ 3 cột cuối
                    new_row = [
                        r[0], r[1], r[2], main_artist, featured, 'ZINGMP3',
                        r[4] if len(r) > 4 else '', r[5] if len(r) > 5 else '',
                        r[6] if len(r) > 6 else '', r[7] if len(r) > 7 else ''
                    ]
                else:
                    # Trường hợp header lạ, cố gắng giữ 8 cột đầu
                    new_row = [
                        r[0], r[1], r[2], main_artist, featured, 'ZINGMP3',
                        r[4] if len(r) > 4 else '', r[5] if len(r) > 5 else '',
                        r[6] if len(r) > 6 else '', r[7] if len(r) > 7 else ''
                    ]
                rows_old.append(new_row)

        # Ghi lại file với header MỚI và dữ liệu ĐÃ NÂNG CẤP
        with open(filepath, 'w', newline='', encoding='utf-8-sig') as wf:
            w = csv.writer(wf)
            w.writerow(NEW_HEADER)
            w.writerows(rows_old)
        print(f"Đã nâng cấp file thành công. Bản sao lưu: {backup_file}")
        
    except Exception as e:
        print(f"Lỗi nghiêm trọng khi nâng cấp schema: {type(e).__name__}. Đang khôi phục backup.")
        # Khôi phục backup nếu lỗi
        if os.path.exists(backup_file):
            os.replace(backup_file, filepath)
        sys.exit(1) # Dừng lại để tránh làm hỏng dữ liệu

def save_data_to_csv(filepath, data_rows, header):
    """Lưu dữ liệu vào CSV, tự động kiểm tra và nâng cấp header nếu cần."""
    try:
        dir_name = os.path.dirname(filepath)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
            
        file_exists = os.path.exists(filepath)
        
        if file_exists:
            current_header = get_csv_header(filepath)
            if current_header != header:
                print("Header không khớp! Đang bắt đầu quá trình nâng cấp...")
                migrate_csv_schema(filepath, current_header)
        
        # Ghi dữ liệu
        mode = 'a' if file_exists else 'w'
        with open(filepath, mode, newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(header) # Ghi header nếu là file mới
            writer.writerows(data_rows)
            
        if not file_exists:
            print(f"\n--- ĐÃ LƯU THÀNH CÔNG {len(data_rows)} BẢN GHI VÀO FILE MỚI '{filepath}' ---")
        else:
            print(f"\n--- ĐÃ THÊM {len(data_rows)} BẢN GHI VÀO '{filepath}' ---")
            
    except Exception as e:
        print(f"\n--- CÓ LỖI KHI LƯU FILE CSV: {e} ---")

# ==============================================================================
# --- HÀM XỬ LÝ SELENIUM (Driver & Navigation) ---
# ==============================================================================

def setup_driver():
    """Thiết lập và trả về một instance của Chrome Driver đã kích hoạt Stealth."""
    print("Đang khởi động Selenium (Stealth Mode)...")
    chrome_options = Options()
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    chrome_options.add_argument('--disable-blink-features=AutomationControlled')
    chrome_options.add_argument("--start-maximized") 
    chrome_options.add_argument("--disable-infobars")
    chrome_options.add_argument("--disable-extensions")
    # chrome_options.add_argument("--headless") 

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
    """Xử lý banner Cookie và modal Quảng cáo."""
    # --- Xử lý Cookie Banner ---
    try:
        WebDriverWait(driver, 15).until(
            EC.element_to_be_clickable((By.XPATH, SELECTORS["cookie_button"]))
        ).click()
        print("Đã đồng ý Cookie.")
        time.sleep(1)
    except Exception:
        print("Không tìm thấy banner Cookie (bỏ qua).")

    # --- Xử lý Modal Quảng cáo ---
    try:
        WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, SELECTORS["ad_modal_close"]))
        ).click()
        print("Đã đóng modal quảng cáo.")
        time.sleep(2)
    except Exception:
        print("Không tìm thấy modal quảng cáo (bỏ qua).")

def expand_to_top_100(driver):
    """Chờ 10 bài đầu, click 'Xem Top 100' và chờ 100 bài tải đủ."""
    try:
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, SELECTORS["chart_item"]))
        )
        print("10 bài hát đầu đã tải.")
    except Exception as e:
        print(f"Không thể tải 10 bài hát đầu: {type(e).__name__}")
        return False # Báo hiệu thất bại

    try:
        view_all_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, SELECTORS["view_top_100"]))
        )
        driver.execute_script("arguments[0].scrollIntoView(true);", view_all_button)
        time.sleep(0.5)
        driver.execute_script("arguments[0].click();", view_all_button)
        print("Đã click 'Xem top 100'.")
        
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.XPATH, SELECTORS["last_song_rank"]))
        )
        print("Đã tải đủ 100 bài hát.")
        time.sleep(2)
        return True # Báo hiệu thành công
    except Exception as e:
        print(f"Không click được 'Xem top 100' hoặc không tải đủ 100 bài: {type(e).__name__}")
        print("Script sẽ tiếp tục và chỉ lấy 10 bài.")
        return True # Vẫn tiếp tục dù chỉ có 10 bài

# ==============================================================================
# --- HÀM XỬ LÝ CÀO DỮ LIỆU (Scraping Logic) ---
# ==============================================================================

def find_more_button(item):
    """Tìm nút 3 chấm 'Khác' trong một item bài hát."""
    try:
        candidates = item.find_elements(By.XPATH, SELECTORS["more_button_paths"][0])
        visible = [b for b in candidates if 'is-hidden' not in (b.get_attribute('class') or '')]
        if visible: return visible[0]
        if candidates: return candidates[-1]
    except Exception:
        pass
        
    for xp in SELECTORS["more_button_paths"][1:]:
        try:
            btn = item.find_element(By.XPATH, xp)
            if btn: return btn
        except Exception:
            pass
    return None

def get_overlay_text_and_html(driver):
    """Lấy text và HTML của portal overlay sau khi click nút 'Khác'."""
    try:
        portal = WebDriverWait(driver, 3).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, SELECTORS["portal_overlay"]))
        )
        return portal.text, portal.get_attribute('innerHTML'), portal
    except Exception:
        pass
    try:
        menu = driver.find_element(By.CSS_SELECTOR, SELECTORS["context_menu"])
        return menu.text, menu.get_attribute('innerHTML'), menu
    except Exception:
        pass
    return "", "", None

def extract_artists(subtitle_element):
    """Tách Nghệ sĩ chính và Nghệ sĩ nổi bật từ element h3.subtitle."""
    try:
        atags = subtitle_element.find_elements(By.TAG_NAME, 'a')
        if atags:
            all_artists = [a.text.strip() for a in atags if a.text.strip()]
            if all_artists:
                main_artist = all_artists[0]
                featured_artists = all_artists[1:]
                return main_artist, ", ".join(featured_artists)
        # Fallback: không có tag <a>
        return subtitle_element.text.strip(), ""
    except Exception:
        return "", ""

def get_likes_plays(driver, item_element):
    """Click nút 'Khác', lấy text và parse ra
    Total_Likes, Total_Plays."""
    more_btn = find_more_button(item_element)
    if not more_btn:
        return "", ""
        
    panel_text = ""
    try:
        try:
            ActionChains(driver).move_to_element(more_btn).pause(0.1).click(more_btn).perform()
        except Exception:
            driver.execute_script("arguments[0].click();", more_btn)
        time.sleep(1.5)
        
        panel_text, _, _ = get_overlay_text_and_html(driver)
        
        # Đóng menu
        ActionChains(driver).send_keys(Keys.ESCAPE).perform()
        time.sleep(0.2)
            
    except Exception as e:
        if os.environ.get('ZING_DEBUG') == '1':
            print(f"Menu error: {type(e).__name__}")

    if not panel_text:
        return "", ""

    # Parse
    total_likes, total_plays = "", ""
    tokens_kmb = re.findall(r"\b\d+(?:[\.,]\d+)?\s*[KMB]\b", panel_text, flags=re.IGNORECASE)
    if tokens_kmb:
        if len(tokens_kmb) >= 1:
            total_likes = str(parse_compact_number(tokens_kmb[0]))
        if len(tokens_kmb) >= 2:
            total_plays = str(parse_compact_number(tokens_kmb[1]))
    return total_likes, total_plays

def scrape_song_item(el, driver, current_date):
    """
    Trích xuất toàn bộ dữ liệu từ một Selenium element (item bài hát).
    Trả về một list đã được định dạng để ghi vào CSV, hoặc None nếu bị lọc.
    """
    try:
        # Scroll và hover
        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", el)
        time.sleep(0.2)
        try:
            ActionChains(driver).move_to_element(el).perform()
            time.sleep(0.1)
        except Exception: pass

        # Trích xuất thông tin cơ bản
        rank = el.find_element(By.CSS_SELECTOR, SELECTORS["item_rank"]).text.strip()
        title_el = el.find_element(By.CSS_SELECTOR, SELECTORS["item_title"])
        title = title_el.text.strip()
        
        href = title_el.get_attribute('href') or ""
        zing_url = f"https://zingmp3.vn{href}" if href.startswith('/') else href
        
        duration = el.find_element(By.CSS_SELECTOR, SELECTORS["item_duration"]).text.strip()
        
        subtitle = el.find_element(By.CSS_SELECTOR, SELECTORS["item_subtitle"])
        artist_text, featured_text = extract_artists(subtitle)

        # Áp dụng OVERRIDES
        title_normalized = title.lower().strip()
        if title_normalized in OVERRIDES:
            artist_text, featured_text = OVERRIDES[title_normalized]
            print(f"  🔧 Override applied: {artist_text} ft. {featured_text}")

        # Lọc (Skip)
        for pattern in SKIP_PATTERNS:
            if re.search(pattern, title, flags=re.IGNORECASE):
                print(f"#{rank} - {title}")
                print(f"  ⏭️ Bỏ qua (matched: {pattern})")
                return None
        
        # Lấy thông tin chi tiết (Likes/Plays) - phần chậm nhất
        total_likes, total_plays = get_likes_plays(driver, el)
        
        print(f"#{rank} - {title} | Likes={total_likes} Plays={total_plays}")

        # Làm sạch nghệ sĩ
        clean_artist = re.sub(r'["""″‟]', '', artist_text or '').strip()
        clean_artist = re.sub(r'\s+', ' ', clean_artist)
        clean_featured = re.sub(r'["""″‟]', '', featured_text or '').strip()
        clean_featured = re.sub(r'\s+', ' ', clean_featured)

        # Trả về list theo đúng thứ tự của NEW_HEADER
        return [
            current_date, rank, title, clean_artist, clean_featured, 
            'ZINGMP3', duration, zing_url, total_likes, total_plays
        ]

    except Exception as e:
        print(f"Lỗi xử lý 1 item: {type(e).__name__}")
        return None

# ==============================================================================
# --- HÀM CHÍNH (MAIN) ---
# ==============================================================================

def main():    
    current_date = os.environ.get('ZING_DATE') or datetime.now().strftime('%Y-%m-%d')
    output_file = os.environ.get('ZING_OUTPUT_FILE', 'data/zingmp3_top100.csv')
    
    if check_date_exists(output_file, current_date):
        print(f"❌ Dữ liệu ngày {current_date} đã tồn tại. → Bỏ qua, không scrape lại.")   

        sys.exit(0)
    else:
        print(f"Chưa có dữ liệu cho ngày {current_date}. Bắt đầu thu thập...")

    # --- Bước 2: Khởi động Selenium ---
    driver = setup_driver()
    song_data = [] # List để chứa các hàng (dạng list)

    try:
        # --- Bước 3: Điều hướng và xử lý Popups ---
        driver.get("https://zingmp3.vn/zing-chart")
        handle_popups(driver)
        
        if not expand_to_top_100(driver):
            # Nếu không tải được 10 bài đầu -> thoát
            raise Exception("Không tải được 10 bài đầu tiên.")

        # --- Bước 4: Lấy danh sách items và cào dữ liệu ---
        print("Đang lấy danh sách bài hát và chuẩn bị trích xuất chi tiết...")
        items = driver.find_elements(By.CSS_SELECTOR, SELECTORS["chart_item"])
        
        if not items:
            print("Không tìm thấy item bài hát nào với class 'chart-song-item'.")
            return # Thoát sớm

        print(f"Tìm thấy {len(items)} bài hát trên trang.")
        max_items_env = os.environ.get('ZING_MAX_ITEMS')
        limit = int(max_items_env) if max_items_env and max_items_env.isdigit() else len(items)
        
        for idx, el in enumerate(items[:limit], start=1):
            print(f"--- Đang xử lý item {idx}/{limit} ---")
            song_row = scrape_song_item(el, driver, current_date)
            if song_row:
                song_data.append(song_row)

    except Exception as e:
        print(f"Lỗi trong quá trình chạy Selenium: {e}")
    finally:
        print("Đang đóng trình duyệt...")
        try:
            driver.quit()
        except Exception:
            pass

    # --- Bước 5: Lưu dữ liệu vào CSV ---
    if song_data:
        save_data_to_csv(output_file, song_data, NEW_HEADER)
    else:
        print("Không thu thập được dữ liệu nào để lưu.")
if __name__ == "__main__":
    main()