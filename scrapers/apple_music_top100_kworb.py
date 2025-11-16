import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import os
from datetime import datetime, timedelta

OVERRIDES = {
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
    "dẫu có đến đâu": ("ANH TRAI SAY HI", "CONGB, VƯƠNG BÌNH, Lohan, Nhâm Phương Nam"),
    "chuyện đôi ta": ("Emcee L", "Muộii (Starry Night)"),
}

URL = "https://kworb.net/charts/apple_s/vn.html"
OUTPUT_FILE = os.environ.get('APPLE_OUTPUT_FILE', 'data/apple_music_top100_kworb_vn.csv')

# === Helper: tách nghệ sĩ chính và featured giống split_artists.py ===
def clean_group_name(s: str) -> str:
    """Loại bỏ dấu ngoặc kép trong tên nhóm nghệ sĩ như EM XINH "SAY HI" -> EM XINH SAY HI.
    Gộp khoảng trắng dư và strip hai đầu.
    """
    if s is None:
        return ''
    s = str(s)
    s = re.sub(r'["“”″‟]', '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s
def split_artists_for_csv(raw_artist_string):
    """
    Tách chuỗi nghệ sĩ thành (main_artist, featured_artists)
    Logic giống split_artists.py: tách bằng dấu phẩy (,) hoặc dấu và (&)
    """
    if pd.isna(raw_artist_string) or raw_artist_string == '':
        return 'Unknown', ''

    # Làm sạch ngoặc vuông/tròn
    clean_artist = re.sub(r'[\[\]\(\)]', '', str(raw_artist_string)).strip()

    # Tách theo dấu phẩy hoặc dấu &
    parts = re.split(r'\s*[,&]\s*', clean_artist)

    main_artist = ''
    featured = ''
    if parts:
        main_artist = parts[0].strip()
        if len(parts) > 1:
            featured_list = [p.strip() for p in parts[1:] if p.strip()]
            if featured_list:
                featured = ', '.join(featured_list)

    return main_artist, featured

print(f"Đang lấy dữ liệu từ {URL}...")

try:
    response = requests.get(URL)
    response.raise_for_status()
    response.encoding = 'utf-8'
    
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Lấy ngày từ title của trang
    chart_date = None
    try:
        # Tìm title hoặc header chứa ngày
        page_text = soup.get_text()
        # Tìm pattern "Apple Music Daily Chart - Vietnam - YYYY/MM/DD"
        for line in page_text.split('\n'):
            if 'Apple Music' in line and 'Vietnam' in line:
                m = re.search(r'(\d{4}/\d{2}/\d{2})', line)
                if m:
                    chart_date = m.group(1).replace('/', '-')
                    print(f"DEBUG: Tìm thấy ngày từ title: {chart_date}")
                    break
        
        # Fallback: tìm bất kỳ YYYY/MM/DD nào
        if not chart_date:
            m2 = re.search(r'(\d{4}/\d{2}/\d{2})', page_text)
            if m2:
                chart_date = m2.group(1).replace('/', '-')
                print(f"DEBUG: Tìm thấy ngày từ fallback: {chart_date}")
    except Exception as e:
        print(f"Không tìm thấy ngày trên trang: {e}")
    
    # Nếu không tìm thấy ngày, dùng ngày hôm qua (vì Kworb cập nhật dữ liệu của ngày trước)
    if not chart_date:
        yesterday = datetime.now() - timedelta(days=1)
        chart_date = yesterday.strftime('%Y-%m-%d')
        print(f"DEBUG: Dùng ngày hôm qua: {chart_date}")
    
    collection_date = chart_date
    print(f"Ngày của bảng xếp hạng: {collection_date}")
    
    # Tìm bảng chứa dữ liệu
    table = soup.find('table')
    if not table:
        # Thử tìm tất cả tables
        tables = soup.find_all('table')
        if tables:
            table = tables[0]
        else:
            print("Không tìm thấy bảng dữ liệu!")
            exit(1)
    
    rows = table.find_all('tr')[1:]  # Bỏ header
    
    print(f"Tìm thấy {len(rows)} dòng dữ liệu")
    
    data = []
    
    for row in rows[:100]:  # Chỉ lấy 100 bài đầu
        try:
            cols = row.find_all('td')
            if len(cols) < 2:
                continue
            
            # Rank (cột 0)
            rank_text = cols[0].get_text(strip=True)
            try:
                rank = int(rank_text)
            except:
                continue
            
            # Position change (cột 1) - bỏ qua
            
            # Artist - Title (cột 2)
            artist_title_elem = cols[2] if len(cols) > 2 else cols[1]
            artist_title = artist_title_elem.get_text(strip=True)
            
            # Tách artist và title
            # Format: "Artist - Title" hoặc "Artist1 & Artist2 - Title"
            artist = ''
            title = artist_title
            
            if ' - ' in artist_title:
                parts = artist_title.split(' - ', 1)
                artist = parts[0].strip()
                title = parts[1].strip()
            
            # Trích xuất featured artist từ title nếu có (feat. X) hoặc (with X)
            title_featured = ''
            feat_match = re.search(r'\((?:feat\.|ft\.|featuring|with)\s+([^)]+)\)', title, re.IGNORECASE)
            if feat_match:
                title_featured = feat_match.group(1).strip()
                # Xóa phần (feat. X) khỏi title
                title = re.sub(r'\s*\((?:feat\.|ft\.|featuring|with)\s+[^)]+\)', '', title, flags=re.IGNORECASE).strip()
            
            # Chuẩn hóa artist + featured theo split_artists.py
            main_artist, featured = split_artists_for_csv(artist)
            
            # Kết hợp featured từ artist string và từ title
            if title_featured:
                if featured:
                    featured = featured + ', ' + title_featured
                else:
                    featured = title_featured
            
            # Áp dụng OVERRIDES nếu có
            title_normalized = title.lower().strip()
            if title_normalized in OVERRIDES:
                main_artist, featured = OVERRIDES[title_normalized]
                print(f"  🔧 Override applied for '{title}': {main_artist} ft. {featured}")
            
            data.append({
                'Date': collection_date,
                'Rank': rank,
                'Title': title,
                'Artists': main_artist,
                'Featured_Artists': featured,
                'Source': 'APPLE_MUSIC'
            })
            
            if rank <= 10:
                print(f"{rank:3d}. {title[:40]:40s} - {artist[:30]}")
        
        except Exception as e:
            print(f"Lỗi khi xử lý dòng: {e}")
            continue
    
    if not data:
        print("Không thu thập được dữ liệu!")
        exit(1)
    
    # Tạo DataFrame
    df = pd.DataFrame(data)
    
    # Sắp xếp theo rank
    df = df.sort_values('Rank').head(100).reset_index(drop=True)

    # ===== CLEAN GROUP NAMES & REMOVE REMIX =====
    remix_mask = df['Title'].str.contains(r'(?i)remix', na=False)
    if remix_mask.any():
        removed = remix_mask.sum()
        print(f"🔁 Loại bỏ {removed} bài có từ 'Remix' trong tiêu đề")
        df = df[~remix_mask].reset_index(drop=True)

    df['Artists'] = df['Artists'].apply(clean_group_name)
    if 'Featured_Artists' in df.columns:
        df['Featured_Artists'] = df['Featured_Artists'].apply(clean_group_name)
    
    print(f"\n=== ĐÃ THU THẬP {len(df)} BẢN GHI ===")
    print(f"Ngày: {collection_date}")
    
    # ===== FILTER VPOP VÀ DEDUPLICATE =====
    # 1. FILTER VPOP - Logic từ filter_vietnam_songs.py
    # Danh sách nghệ sĩ quốc tế cần loại bỏ (phải match chính xác toàn bộ tên)
    international_artists_exact = [
        'Jung Kook', 'Jungkook', 'BTS', 'Jimin', 'RM', 'Suga', 'Jin', 'J-Hope',
        'BLACKPINK', 'Rosé', 'Jennie','JENNIE', 'Lisa', 'Jisoo',
        'NewJeans', 'IVE', 'aespa', 'TWICE', 'ITZY', 'Red Velvet',
        'Stray Kids', 'NCT', 'EXO', 'SEVENTEEN', 'TXT',
        'Taylor Swift', 'Bruno Mars', 'Ed Sheeran', 'The Weeknd',
        'Ariana Grande', 'Justin Bieber', 'Billie Eilish',
        'Dua Lipa', 'Olivia Rodrigo', 'Sabrina Carpenter',
        'Jack Harlow', 'Latto', 'Drake', 'Travis Scott',
        'Post Malone', 'SZA', 'Doja Cat', 'Metro Boomin',
        'Charlie Puth', 'Shawn Mendes', 'Harry Styles',
        'Adele', 'Beyoncé', 'Rihanna', 'Lady Gaga',
        'Park Hyo Shin', 'IU', 'Taeyeon', 'Lee Mujin',
        'LE SSERAFIM', 'CORTIS', 'Madison Beer', 'HUNTR/X', 'G-DRAGON'
    ]
    
    # Danh sách nghệ sĩ quốc tế (match nếu xuất hiện trong tên)
    international_keywords = [
        'Jung Kook', 'Jungkook', 'Jack Harlow', 'Latto',
        'Park Hyo Shin'
    ]
    
    def is_international_artist(artist, featured):
        """Kiểm tra xem có phải nghệ sĩ quốc tế không"""
        artist_str = str(artist).strip()
        featured_str = str(featured if pd.notna(featured) else '').strip()
        full_artist = artist_str + ' ' + featured_str
        
        # Check exact match với tên chính
        if artist_str in international_artists_exact:
            return True
        
        # Check featured artists
        if featured_str:
            for intl in international_artists_exact:
                if intl in featured_str:
                    return True
        
        # Check keywords trong full artist string
        full_lower = full_artist.lower()
        for keyword in international_keywords:
            if keyword.lower() in full_lower:
                return True
        
        # Special case: "V" chỉ khi nó là nghệ sĩ chính và đứng một mình
        if artist_str == 'V':
            return True
        
        return False
    
    # Filter ra non-VPOP
    # Check xem có cột featured_artists không
    def filter_vpop_and_deduplicate(df):
        """
        Filter chỉ lấy VPOP và xóa duplicate.
        Trả về: (df_filtered, vpop_removed_count, dup_removed_count)
        """
        if df.empty:
            return df, 0, 0

        # 1. FILTER VPOP
        initial_vpop_count = len(df)
        
        # [SỬA CHO APPLE MUSIC]
        # Kiểm tra xem df có cột 'featured_artists' không (Apple Music không có)
        if 'Featured_Artists' in df.columns:
            df_vpop = df[~df.apply(lambda row: is_international_artist(row['Artists'], row['Featured_Artists']), axis=1)].copy()
        else:
            # Nếu không có Featured_Artists, chỉ check Artists
            df_vpop = df[~df.apply(lambda row: is_international_artist(row['Artists'], None), axis=1)].copy()
        
        vpop_removed_count = initial_vpop_count - len(df_vpop)
        
        # 2. DEDUPLICATE (trong batch)
        if 'Date' in df_vpop.columns:
                    duplicate_cols = ['Date', 'Title', 'Artists']
        else:
                    duplicate_cols = ['Title', 'Artists']
        
        initial_dedupe_count = len(df_vpop)
        df_vpop = df_vpop.drop_duplicates(subset=duplicate_cols, keep='first')
        
        dup_removed_count = initial_dedupe_count - len(df_vpop)
        
        return df_vpop, vpop_removed_count, dup_removed_count

    # ===== [THAY ĐỔI] KHỐI MAIN VÀ LOGIC LƯU FILE =====
    print("\n🔍 Đang filter VPOP và xóa duplicate (trong batch mới)...")
    
    # Gọi hàm filter mới
    df_filtered, vpop_removed_count, dup_removed_count = filter_vpop_and_deduplicate(df)
    
    # In số lượng đã xóa (nếu có)
    if vpop_removed_count > 0:
            print(f"  ✓ Đã loại bỏ {vpop_removed_count} bài hát non-VPOP")
    if dup_removed_count > 0:
            print(f"  ✓ Đã xóa {dup_removed_count} bản ghi duplicate (trong batch mới)")

    if df_filtered.empty:
        print("❌ Không còn dữ liệu VPOP mới sau khi filter!")
    else:
        final_new_count = len(df_filtered)
        print(f"✅ Còn lại {final_new_count} bản ghi VPOP mới sau khi filter.\n")
        
        # 3. Lấy ngày của dữ liệu (đã có biến 'collection_date')
        current_data_date = collection_date
        
        file_exists = os.path.exists(OUTPUT_FILE)
        date_already_exists = False

        # 4. Kiểm tra xem file có tồn tại và ngày này đã được ghi chưa
        if file_exists:
            try:
                # Tải NHANH CỘT 'Date' để kiểm tra
                df_old_dates = pd.read_csv(OUTPUT_FILE, usecols=['Date'])
                if not df_old_dates.empty and current_data_date in df_old_dates['Date'].values:
                    date_already_exists = True
            except pd.errors.EmptyDataError:
                print(f"   File {OUTPUT_FILE} tồn tại nhưng bị rỗng.")
                file_exists = False
            except ValueError:
                    print(f"   Cột 'Date' không tìm thấy trong file cũ. Sẽ ghi mới.")
                    file_exists = False
            except Exception as e:
                print(f"⚠️ Lỗi khi đọc file cũ ({e}). Coi như file không tồn tại.")
                file_exists = False 

        # 5. Quyết định ghi hay không
        if date_already_exists:
            print(f"❌ Dữ liệu cho ngày {current_data_date} đã tồn tại. → Bỏ qua, không scrape lại.")
        else:
            # Đảm bảo thư mục tồn tại
            os.makedirs(os.path.dirname(OUTPUT_FILE) if os.path.dirname(OUTPUT_FILE) else '.', exist_ok=True)
            
            try:
                df_filtered.to_csv(OUTPUT_FILE, 
                                    mode='a',  # Chế độ Nối tiếp
                                    header=(not file_exists), # Chỉ ghi header nếu file mới
                                    index=False, 
                                    encoding='utf-8-sig')
                
                # In thông báo thành công
                print(f"✅ Đã thêm {final_new_count} bản ghi vào {OUTPUT_FILE}.")
                    
            except Exception as e:
                print(f"Lỗi khi lưu file: {e}")
                import traceback
                traceback.print_exc()

    # 6. Đọc lại file để lấy tổng số bản ghi (luôn chạy)
    print() # Thêm một dòng trống
    try:
        if os.path.exists(OUTPUT_FILE):
            total_df = pd.read_csv(OUTPUT_FILE)
            print(f"Tổng bản ghi của file {OUTPUT_FILE}: {len(total_df)}")
        else:
            # Nếu file không tồn tại VÀ chúng ta không ghi gì (vì date_already_exists=True)
            if 'date_already_exists' not in locals() or not date_already_exists:
                 print("File chưa tồn tại.")
    except Exception as e:
        print(f"Không thể đọc tổng số bản ghi: {e}")


except requests.exceptions.RequestException as e:
    print(f"Lỗi kết nối: {e}")
except Exception as e:
    print(f"Lỗi: {e}")
    import traceback
    traceback.print_exc()