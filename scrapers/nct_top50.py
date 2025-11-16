import time
import sys
import os
import csv
import re
from urllib.parse import urljoin
from datetime import datetime

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.common.by import By 
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from bs4 import BeautifulSoup
    from webdriver_manager.chrome import ChromeDriverManager
    from selenium_stealth import stealth
    import pandas as pd # Thêm pandas
except Exception as e:
    print("Missing required package or import failed:", e)
    print("Install required packages with:\n    pip install selenium webdriver-manager beautifulsoup4 selenium-stealth pandas")
    sys.exit(1)

# [SỬA 1] BẢNG QUY TẮC SỬA LỖI DỮ LIỆU
NCT_OVERRIDES = {
    # Key: title chuẩn hóa (lowercase, đã strip)
    "không đau nữa rồi": (
        "EM XINH SAY HI", 
        "Orange, Châu Bùi, Mỹ Mỹ, Pháp Kiều, 52Hz"
    ),
    "cứ đổ tại cơn mưa": (
        "EM XINH SAY HI",
        "Phương Ly, Orange, Châu Bùi, 52Hz, Vũ Thảo My"
    ),
    "gã săn cá": (
        "EM XINH SAY HI",
        "Lâm Bảo Ngọc, Quỳnh Anh Shyn, MAIQUINN, Saabirose, Quang Hùng MasterD"
    ),
    "đoạn kịch câm": (
        "ANH TRAI SAY HI",
        "CONGB, Negav, B Ray, CODYNAMVO, Phạm Đình Thái Ngân, Mỹ Mỹ"
    ),
    "người như anh xứng đáng cô đơn": (
        "ANH TRAI SAY HI",
        "Vũ Cát Tường, Karik, Negav, Ngô Kiến Huy, Jey B"
    ),
    "hơn là bạn": (
        "ANH TRAI SAY HI",
        "Ngô Kiến Huy, Vũ Cát Tường, Karik, VƯƠNG BÌNH, Sơn.K, MIN"
    ),
    "hermosa": (
        "ANH TRAI SAY HI",
        "Sơn.K, buitruonglinh, Tez, Mason Nguyen, CONGB"
    ),
    "sớm muộn thì": (
        "ANH TRAI SAY HI",
        "Hustlang Robber, Nhâm Phương Nam, Mason nguyen, Jaysonlei, Khoi Vu, LAMOON"
    ),
    "đa nghi": (
        "ANH TRAI SAY HI",
        "Negav, Hải Nam, CODYNAMVO, Dillan Hoàng Phan"
    ),
    "người yêu chưa sinh ra": (
        "ANH TRAI SAY HI",
        "Ogenus, BigDaddy, Phúc Du, HUSTLANG Robber, Dillan Hoàng Phan"
    ),
    "make up": (
        "ANH TRAI SAY HI",
        "Dillan Hoàng Phan, buitruonglinh, Đỗ Nam Sơn, Lohan, Ryn Lee, Bảo Thy"
    ),
    "dẫu có đến đâu": (
        "ANH TRAI SAY HI",
        "CONGB, VƯƠNG BÌNH, Lohan, Nhâm Phương Nam"
    )
}

# --- Cấu hình Selenium (Giữ nguyên) ---
chrome_options = Options()
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
chrome_options.add_experimental_option('useAutomationExtension', False)
chrome_options.add_argument('--disable-blink-features=AutomationControlled')
chrome_options.add_argument("--start-maximized") 

try:
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
except Exception as e:
    print(f"Lỗi khi khởi tạo Chrome Driver: {e}")
    print("Vui lòng đảm bảo Chrome đã được cài đặt và cập nhật.")
    sys.exit(1)


# Kích hoạt Stealth (Giữ nguyên)
stealth(driver,
        languages=["en-US", "en"],
        vendor="Google Inc.",
        platform="Win32",
        webgl_vendor="Intel Inc.",
        renderer="Intel Iris OpenGL Engine",
        fix_hairline=True,
        )
# ---------------------------------------------

# --- Hàm tự động tìm URL chart (Giữ nguyên) ---
def get_chart_url(target_date=None):
    """
    Tạo URL chart cho ngày cụ thể.
    Format: https://www.nhaccuatui.com/chart/1-1-d{day_of_year}-{year}
    """
    if target_date is None:
        target_date = datetime.now()
    
    day_of_year = target_date.timetuple().tm_yday
    year = target_date.year
    
    chart_url = f"https://www.nhaccuatui.com/chart/1-1-d{day_of_year}-{year}"
    date_str = target_date.strftime('%Y-%m-%d')
    
    return chart_url, date_str

# Output luôn là data\nct_top50.csv (append mode)
OUTPUT_CSV = os.path.join('data', 'nct_top50.csv')
APPEND_MODE = True
ERROR_SCREENSHOT = 'nct_error_v2.png'

def check_date_exists(csv_path, target_date):
    """Kiểm tra xem ngày đã tồn tại trong CSV chưa"""
    if not os.path.exists(csv_path):
        return False
    try:
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        if 'Date' in df.columns:
            existing_dates = df['Date'].unique()
            return target_date in existing_dates
    except pd.errors.EmptyDataError:
        print(f"File {csv_path} rỗng.")
        return False
    except Exception as e:
        print(f"Lỗi khi đọc CSV: {e}")
        pass
    return False

# Hàm thử load chart và phát hiện ngày thực tế
def try_load_chart(target_date):
    """Thử load chart cho một ngày cụ thể và trả về (success, actual_date)"""
    NCT_URL, initial_date = get_chart_url(target_date)
    print(f"\nThử truy cập chart ngày: {initial_date}")
    print(f"URL: {NCT_URL}")
    
    try:
        driver.get(NCT_URL)
        time.sleep(3)  # Chờ trang load
        
        page_html = driver.page_source
        import re as _re
        m_updated = _re.search(r'Updated on\s+(\d{2}/\d{2}/\d{4})', page_html)
        
        if m_updated:
            raw_dmy = m_updated.group(1)
            parsed = datetime.strptime(raw_dmy, '%d/%m/%Y')
            actual_date = parsed.strftime('%Y-%m-%d')
            print(f"✓ Tìm thấy chart! Ngày cập nhật thực tế: {actual_date}")
            return True, actual_date, NCT_URL
        else:
            print(f"✗ Không tìm thấy 'Updated on' trên trang")
            return False, initial_date, NCT_URL
    except Exception as e:
        print(f"✗ Lỗi khi load trang: {e}")
        return False, initial_date, NCT_URL

# Thử load chart hôm nay trước
print("="*60)
print("BẮT ĐẦU: Đang mở NhacCuaTui (Stealth Mode)...")
target_date = datetime.now()
success, chart_date, NCT_URL = try_load_chart(target_date)

# Nếu không thành công hoặc chart chưa được cập nhật (ngày chart < hôm nay), thử hôm qua
from datetime import timedelta
if not success or chart_date < target_date.strftime('%Y-%m-%d'):
    print(f"\n⚠ Chart ngày {target_date.strftime('%Y-%m-%d')} chưa có hoặc chưa cập nhật")
    print("→ Thử lại với ngày hôm qua...")
    yesterday = target_date - timedelta(days=1)
    success, chart_date, NCT_URL = try_load_chart(yesterday)
    
    if not success:
        print("\n✗ Không thể tải chart. Dừng lại.")
        driver.quit()
        sys.exit(1)

# Kiểm tra xem ngày chart này đã có trong CSV chưa
if check_date_exists(OUTPUT_CSV, chart_date):
    print(f"\n❌ Dữ liệu ngày {chart_date} đã tồn tại. → Bỏ qua, không scrape lại.")
    driver.quit()
    sys.exit(0)

print(f"\n{'='*60}")
print(f"Chart date (final): {chart_date}")
print(f"Output file (final): {OUTPUT_CSV}")
print(f"{'='*60}\n")
# --- BƯỚC 1: Chờ BXH tải (với retry nhẹ) ---
print("Đang chờ danh sách BXH tải...")
chart_item_selector = "div.song-item"
loaded = False
for attempt in range(3):
    try:
        WebDriverWait(driver, 20 + attempt*10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, chart_item_selector))
        )
        loaded = True
        break
    except Exception:
        try:
            driver.refresh()
            time.sleep(2)
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(1)
            driver.execute_script("window.scrollTo(0, 0);")
        except Exception:
            pass

if not loaded:
    print("Không thể tải danh sách bài hát: TimeoutException")
    driver.get_screenshot_as_file(ERROR_SCREENSHOT)
    driver.quit()
    sys.exit(1)
else:
    print("Danh sách bài hát đã tải (div.song-item detected).")

# Click "Show more"
print("Kiểm tra nút 'Show more' để load tất cả bài hát...")
try:
    show_more = WebDriverWait(driver, 5).until(
        EC.presence_of_element_located((By.XPATH, "//span[contains(text(), 'Show more')]"))
    )
    driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", show_more)
    time.sleep(0.5)
    driver.execute_script("arguments[0].click();", show_more)
    print("Đã click 'Show more' thành công!")
    time.sleep(2)
except Exception as e:
    print(f"Không tìm thấy hoặc click được 'Show more': {e}")
    try:
        for _ in range(3):
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(1.5)
    except Exception:
        pass

time.sleep(2)

# --- Lấy dữ liệu và phân tích (Giữ nguyên logic V12 của bạn) ---
print("Đang thu thập danh sách bài hát từ DOM...")
items = driver.find_elements(By.CSS_SELECTOR, "div.song-item")
if not items:
    print("Không tìm thấy item bài hát nào trong DOM.")
    driver.quit()
    sys.exit(1)

print(f"Tìm thấy {len(items)} bài hát (DOM).")

items_with_rank = 0
for item in items:
    try:
        rank = item.find_element(By.CSS_SELECTOR, 'span.idx').text.strip()
        if rank:
            items_with_rank += 1
    except Exception:
        pass
print(f"Số bài có Rank: {items_with_rank}")

print("Bắt đầu deep-scrape...")

try:
    MAX_ITEMS = int(os.environ.get('NCT_MAX_ITEMS', '0') or '0')
except Exception:
    MAX_ITEMS = 0

# [SỬA 1] Thêm Helper: split_artists_for_csv
def split_artists_for_csv(raw_artist_string):
    """
    Tách chuỗi nghệ sĩ thành (main_artist, featured_artists)
    Logic: tách bằng dấu phẩy (,)
    """
    if pd.isna(raw_artist_string) or raw_artist_string == '':
        return 'Unknown', ''
    clean_artist = re.sub(r'[\[\]\(\)]', '', str(raw_artist_string)).strip()
    parts = re.split(r'\s*,\s*', clean_artist)
    main_artist = ''
    featured = ''
    if parts:
        main_artist = parts[0].strip()
        if len(parts) > 1:
            featured_list = [p.strip() for p in parts[1:] if p.strip()]
            if featured_list:
                featured = ', '.join(featured_list)
    return main_artist, featured

def get_song_url_by_click(rank_text: str):
    try:
        if 'chart' not in driver.current_url:
            driver.get(NCT_URL)
            WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, "div.song-item")))
        xpath_item = f"//div[contains(@class,'song-item')]//span[@class='idx' and normalize-space(text())='{rank_text}']/ancestor::div[contains(@class,'song-item')]"
        item_el = WebDriverWait(driver, 5).until(
            EC.presence_of_element_located((By.XPATH, xpath_item))
        )
        try:
            name_el = item_el.find_element(By.CSS_SELECTOR, "span.name")
        except Exception:
            name_el = None
        if not name_el:
            return ''
        driver.execute_script("arguments[0].scrollIntoView({block:'center'});", name_el)
        time.sleep(0.2)
        current = driver.current_url
        driver.execute_script("arguments[0].click();", name_el)
        WebDriverWait(driver, 10).until(lambda d: d.current_url != current and '/song/' in d.current_url)
        song_url = driver.current_url
        driver.back()
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, "div.song-item")))
        time.sleep(0.2)
        return song_url
    except Exception:
        return ''

def regex_first(pattern, text, flags=0):
    m = re.search(pattern, text, flags)
    return m.group(1) if m else None

def normalize_int(num_text):
    if not num_text: return ''
    s = re.sub(r"[^0-9]", "", num_text)
    if not s: return ''
    try: return str(int(s))
    except Exception: return s

def extract_json_data(html_source, song_key: str = None):
    import json
    result = {'total_likes': ''}
    try:
        m = re.search(r'<script[^>]*id="__NUXT_DATA__"[^>]*>(.*?)</script>', html_source, re.DOTALL)
        if not m: return result
        json_str = m.group(1)
        data = json.loads(json_str)
        if not isinstance(data, list) or len(data) < 2: return result
        main_obj = data[1]
        
        def fill_from_song_object(song_obj):
            def get_val(obj, key):
                v = obj.get(key)
                v = resolve(v)
                if isinstance(v, str) and re.fullmatch(r"[0-9\.]+", v):
                    try: return int(v.replace('.', ''))
                    except Exception: return v
                return v
            total_likes = get_val(song_obj, 'totalLiked')
            if isinstance(total_likes, (int, float, str)):
                result['total_likes'] = str(total_likes)

        def resolve(val):
            if isinstance(val, int) and 0 <= val < len(data):
                return data[val]
            return val

        def find_field(root_idx_or_obj, field_name, max_depth=5):
            visited = set()
            def _walk(node, depth):
                if depth > max_depth: return None
                if isinstance(node, int):
                    if node in visited: return None
                    visited.add(node)
                    node = resolve(node)
                if isinstance(node, dict):
                    if field_name in node:
                        val = resolve(node[field_name])
                        if isinstance(val, (int, float)): return val
                        if isinstance(val, str):
                            s = re.sub(r"[^0-9]", "", val)
                            if s.isdigit(): return int(s)
                    for v in node.values():
                        res = _walk(v, depth+1)
                        if res is not None: return res
                elif isinstance(node, list):
                    for v in node:
                        res = _walk(v, depth+1)
                        if res is not None: return res
                return None
            return _walk(root_idx_or_obj, 0)

        state_ref = main_obj.get('state')
        root_idx = None
        if isinstance(state_ref, int) and state_ref < len(data):
            state_obj = data[state_ref]
            if isinstance(state_obj, dict):
                detail_key = next((k for k in state_obj.keys() if k.startswith('dataDetail:')), None)
                if detail_key:
                    root_idx = state_obj.get(detail_key)

        if root_idx is None:
            for idx, elem in enumerate(data):
                if isinstance(elem, dict):
                    detail_key = next((k for k in elem.keys() if k.startswith('dataDetail:')), None)
                    if detail_key:
                        root_idx = elem.get(detail_key)
                        break

        def find_node_by_key(root_idx_or_obj, key_field='key', key_value=None, max_depth=6):
            if not key_value: return None
            visited = set()
            def _walk(node, depth):
                if depth > max_depth: return None
                if isinstance(node, int):
                    if node in visited: return None
                    visited.add(node)
                    node = resolve(node)
                if isinstance(node, dict):
                    kv = resolve(node.get(key_field))
                    if kv == key_value: return node
                    for v in node.values():
                        r = _walk(v, depth+1)
                        if r is not None: return r
                elif isinstance(node, list):
                    for v in node:
                        r = _walk(v, depth+1)
                        if r is not None: return r
                return None
            return _walk(root_idx_or_obj, 0)

        if root_idx is not None and song_key:
            anchor = find_node_by_key(root_idx, key_value=song_key)
            if isinstance(anchor, dict):
                fill_from_song_object(anchor)
        
        try:
            _likes_int_chk = int(result['total_likes']) if str(result['total_likes']).isdigit() else 0
        except Exception: _likes_int_chk = 0
        if _likes_int_chk < 30 and root_idx is not None:
            tl = find_field(root_idx, 'totalLiked')
            if tl is not None:
                result['total_likes'] = str(int(tl))
        
        try:
            likes_int = int(result['total_likes']) if str(result['total_likes']).isdigit() else 0
        except Exception: likes_int = 0
        if not result['total_likes'] or likes_int < 30:
            m2 = re.search(r'#icon-detail_like[\s\S]{0,200}?<span class="count"[^>]*>([0-9\.]+)</span>', html_source, re.IGNORECASE)
            if m2:
                try: result['total_likes'] = str(int(m2.group(1).replace('.', '')))
                except Exception: pass
        
        return result
    except Exception as e:
        return result

base = "https://www.nhaccuatui.com"

def clean_group_name(s: str) -> str:
    if s is None: return ''
    s = str(s)
    s = re.sub(r'["“”″‟]', '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

# Bước 1: Thu thập dữ liệu cơ bản
song_list = []
for el in items:
    info = { 'rank':'', 'title':'', 'artist':'', 'duration':'', 'url':'' }
    try:
        try:
            rank_text = el.find_element(By.CSS_SELECTOR, 'span.idx').text.strip()
            info['rank'] = rank_text
        except Exception: pass
        try:
            title_text = el.find_element(By.CSS_SELECTOR, 'span.name').text.strip()
            info['title'] = title_text
        except Exception: pass
        try:
            artist_elems = el.find_elements(By.CSS_SELECTOR, 'div.artist-wrap a, div.artist-wrap span')
            names = [a.text.strip() for a in artist_elems if a.text and a.text.strip()]
            info['artist'] = ', '.join(names) if names else ''
        except Exception: pass
        try:
            outer = el.get_attribute('outerHTML') or ''
            d = regex_first(r'(\b\d{1,2}:\d{2}\b)', outer)
            info['duration'] = d or ''
        except Exception: pass
    finally:
        if info['rank'] or info['title']:
            song_list.append(info)

print(f"Đã thu thập thông tin cơ bản của {len(song_list)} bài hát từ chart.")

# Bước 2: Truy cập từng URL (Giữ nguyên logic V12 của bạn)
rows = []
processed = 0
total = len(song_list)
print(f"\n=== BẮT ĐẦU SCRAPE {total} BÀI HÁT ===\n")

for idx, info in enumerate(song_list, 1):
    print(f"[{idx}/{total}] Đang xử lý: Rank {info.get('rank','')} - {info.get('title','')[:40]}...")
    
    url = info.get('url') or ''
    total_likes = ''
    
    def visit_and_extract(u: str):
        try:
            driver.get(u)
            WebDriverWait(driver, 10).until(lambda d: d.execute_script('return document.readyState') == 'complete')
            time.sleep(0.8)
            html = driver.page_source
            song_key = ''
            try:
                part = u.split('/song/')[-1]
                song_key = part.split('?')[0]
            except Exception: song_key = ''
            return extract_json_data(html, song_key)
        except Exception:
            return {}

    if url:
        s = visit_and_extract(url)
        if s:
            total_likes = s.get('total_likes','')
            print(f"  → Total_Likes: {total_likes}")
    else:
        r = info.get('rank','').strip()
        if r:
            song_url = get_song_url_by_click(r)
            if song_url:
                url = song_url
                print(f"  → URL: {url}")
                s = visit_and_extract(url)
                if s:
                    total_likes = s.get('total_likes','')
                    print(f"  → Total_Likes: {total_likes}")

    # ==========================================================
    # [SỬA 1] TÁCH ARTIST VÀ ÁP DỤNG OVERRIDE
    # ==========================================================
    title = info.get('title','')
    title_lower = title.lower().strip() # Thêm strip() cho chắc
    raw_artist_string = clean_group_name(info.get('artist',''))
    
    main_artist = ''
    featured_artists = ''

    if title_lower in NCT_OVERRIDES:
        # 1. Áp dụng override (ghi đè)
        main_artist, featured_artists = NCT_OVERRIDES[title_lower]
        print(f"  → Áp dụng override cho '{title}'")
    else:
        # 2. Tách tự động
        main_artist, featured_artists = split_artists_for_csv(raw_artist_string)

    # ==========================================================
    # [SỬA 2] THAY ĐỔI THỨ TỰ CỘT KHI GHI
    # ==========================================================
    rows.append([
        chart_date,         # Date
        info.get('rank',''),  # Rank
        title,              # Title
        main_artist,        # Artists (Đã sửa)
        featured_artists,   # Featured_Artists (Đã sửa)
        'NCT',              # Source (Thêm mới)
        info.get('duration',''), # Duration
        total_likes,        # Total_Likes (Trước URL)
        url                 # URL (Sau Likes)
    ])

    processed += 1
    if MAX_ITEMS and processed >= MAX_ITEMS:
        print(f"\nĐạt giới hạn {MAX_ITEMS} bài, dừng lại.")
        break

# Đóng trình duyệt
driver.quit()

# [SỬA 2] Cập nhật Header cho đúng thứ tự mới
header = ['Date', 'Rank', 'Title', 'Artists', 'Featured_Artists', 'Source', 'Duration', 'Total_Likes', 'URL']
try:
    file_exists = os.path.exists(OUTPUT_CSV)
    mode = 'a' if APPEND_MODE and file_exists else 'w'
    
    # Đảm bảo thư mục 'data' tồn tại
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    
    with open(OUTPUT_CSV, mode, newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        if not file_exists or not APPEND_MODE:
            writer.writerow(header) # Ghi header mới
            
        for row in rows:
            # Lọc remix trước khi ghi
            title_lower = str(row[2]).lower()
            if 'remix' in title_lower or '#1' in title_lower:
                continue
            # Clean Title quotes
            row[2] = re.sub(r'["“”″‟]', '', str(row[2]))
            writer.writerow(row)
            
    print(f"\n--- ĐÃ LƯU THÀNH CÔNG DỮ LIỆU VÀO FILE '{OUTPUT_CSV}' ---")
except Exception as e:
    print(f"\n--- CÓ LỖI KHI LƯU FILE CSV: {e} ---")