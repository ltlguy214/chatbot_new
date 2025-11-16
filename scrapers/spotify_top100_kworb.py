import re
import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import os

URL = 'https://kworb.net/spotify/country/vn_daily.html'
FILE_NAME = 'data/spotify_top100_kworb_vn.csv'

def clean_group_name(s: str) -> str:
    """Remove embedded quotes in composite artist group names.
    Example: EM XINH "SAY HI" -> EM XINH SAY HI; ANH TRAI "SAY HI" -> ANH TRAI SAY HI.
    Collapse whitespace and strip edges.
    """
    if s is None:
        return ''
    s = str(s)
    s = re.sub(r'["“”″‟]', '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def _split_featured_list(txt: str) -> list:
    """Split a featured artists text segment into individual names, preserving order.
    Supports separators: comma, 'and', '/', 'x', '×', '·', '•'.
    NOTE: Ampersand (&) is NOT treated as separator - preserved as part of artist name.
    Example: "Minh Tốc & Lam" stays as one artist.
    """
    if not txt:
        return []
    t = str(txt)
    # Normalize different separators to comma (but NOT ampersand)
    seps = [r'\s+and\s+', r'\s+x\s+', r'\s*×\s*', r'\s*/\s*', r'\s*·\s*', r'\s*•\s*']
    for pat in seps:
        t = re.sub(pat, ',', t, flags=re.IGNORECASE)
    # Now split by comma
    parts = [p.strip() for p in t.split(',') if p and p.strip()]
    # Deduplicate while preserving order
    seen = set()
    out = []
    for p in parts:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out

def split_title_artist(text):
    """Tách và chuẩn hóa title, artist và featured artists từ chuỗi cols[2].
    Trả về (title, artist, featured)
    Hàm này dùng chung cho toàn module.
    """
    if not text or not isinstance(text, str):
        return ('', 'Unknown', '')

    raw = text.strip()
    # Remove control chars and stray quotes (keep commas for featured parsing)
    raw = re.sub(r"[\n\t\"]", ' ', raw)
    # Remove bracketed annotations like [Clean], [Explicit]
    raw = re.sub(r"\[.*?\]", "", raw)

    # Extract collaborations / featured artists in parentheses and inline (ft., feat., featuring)
    featured_names = []
    
    # Parenthetical markers: (w/ ...), (with ...), (ft. ...), (feat. ...)
    # Handle nested parentheses by finding matching close paren
    keywords = [r'w/', r'with', r'ft\.?', r'feat\.?', r'featuring']
    keyword_pattern = r'\((?:' + '|'.join(keywords) + r')\s*'
    
    for m in re.finditer(keyword_pattern, raw, flags=re.I):
        start = m.end()  # Position after opening '(' and keyword
        # Find matching closing parenthesis by counting
        depth = 1
        i = start
        while i < len(raw) and depth > 0:
            if raw[i] == '(':
                depth += 1
            elif raw[i] == ')':
                depth -= 1
            i += 1
        
        if depth == 0:  # Found matching close paren
            content = raw[start:i-1].strip()
            if content:
                featured_names.extend(_split_featured_list(content))
    
    # Inline ft/feat outside parentheses, e.g. "Title ft. A, B" or "Artist - Title feat C & D"
    for m in re.finditer(r"(?:\bft\.?\b|\bfeat\.?\b|\bfeaturing\b)\s+([^,()\-]+)", raw, flags=re.I):
        seg = m.group(1).strip()
        featured_names.extend(_split_featured_list(seg))
    
    # Dedup preserve order
    if featured_names:
        tmp = []
        seen = set()
        for n in featured_names:
            if n and n not in seen:
                seen.add(n)
                tmp.append(n)
        featured_names = tmp

    # Remove those markers from a working string used for title/artist parsing
    # Use same logic to find and remove complete parenthetical expressions
    s = raw
    for m in re.finditer(keyword_pattern, s, flags=re.I):
        start_pos = m.start()
        end_keyword = m.end()
        depth = 1
        i = end_keyword
        while i < len(s) and depth > 0:
            if s[i] == '(':
                depth += 1
            elif s[i] == ')':
                depth -= 1
            i += 1
        if depth == 0:
            # Remove the entire parenthetical from start_pos to i
            s = s[:start_pos] + s[i:]
    
    s = re.sub(r"(?:\bft\.?\b|\bfeat\.?\b|\bfeaturing\b)\s+[^,()\-]+", "", s, flags=re.I)

    # Normalize separators (dash, en-dash, em-dash) and split on the FIRST dash
    parts = re.split(r"\s[-–—]\s|\s-\s|\s–\s", s, maxsplit=1)
    if len(parts) == 2:
        left_raw, right_raw = parts[0].strip(), parts[1].strip()
        # Kworb usually: Artist - Title
        artist_raw = left_raw
        title_raw = right_raw
    else:
        # No dash found: fallback assume whole is title
        artist_raw = 'Unknown'
        title_raw = parts[0].strip()

    # Remove any remaining parenthetical annotations from title and artist
    artist_clean = re.sub(r"\s*\(.*?\)\s*", "", artist_raw).strip()
    title_clean = re.sub(r"\s*\(.*?\)\s*", "", title_raw).strip()

    # Collapse whitespace
    artist_clean = re.sub(r"\s+", " ", artist_clean)
    title_clean = re.sub(r"\s+", " ", title_clean)

    if not artist_clean:
        artist_clean = 'Unknown'

    # Final normalization of whitespace; KEEP commas out of title/artist to be safe
    title_clean = re.sub(r"[\n\t\"]", ' ', title_clean).strip()
    artist_clean = re.sub(r"[\n\t\"]", ' ', artist_clean).strip()
    title_clean = re.sub(r"\s+", ' ', title_clean)
    artist_clean = re.sub(r"\s+", ' ', artist_clean)
    # Remove commas from title and artist only
    title_clean = title_clean.replace(',', '')
    artist_clean = artist_clean.replace(',', '')

    if not title_clean:
        title_clean = ''
    if not artist_clean:
        artist_clean = 'Unknown'

    featured_str = ', '.join(featured_names)
    return title_clean, artist_clean, featured_str

def parse_number(s):
    """Convert a string to int removing commas, return None on failure."""
    try:
        if s is None:
            return None
        ss = str(s).replace(',', '').strip()
        if ss == '':
            return None
        return int(re.sub(r"[^0-9-]", "", ss))
    except Exception:
        return None

def parse_peak_data(text):
    """Parse peak position and count from text.
    Returns (position, count) tuple, both None if parsing fails.
    Format examples:
    - "1" -> position=1, count=0 (no x marker means count=0)
    - "1 (x565)" -> position=1, count=565
    - "1(x565)" -> position=1, count=565
    - "Pk(x5)" -> position=0, count=5
    """
    if not text or not isinstance(text, str):
        return None, None
    
    text_strip = text.strip()
    
    # Case 1: "Pk(xN)" format → position=0, count=N
    pk_match = re.search(r"Pk\s*\(x(\d+)\)", text_strip, re.IGNORECASE)
    if pk_match:
        try:
            count = int(pk_match.group(1))
            return 0, count
        except:
            return None, None
    
    # Case 2: Normal "N" or "N (xM)" format
    pos_match = re.search(r"(\d+)", text_strip)
    if not pos_match:
        return None, None
        
    try:
        position = int(pos_match.group(1))
        
        # Look for count in (xN) format; if absent, count=0
        count_match = re.search(r"\(x(\d+)\)", text_strip)
        count = int(count_match.group(1)) if count_match else 0
        
        return position, count
    except (ValueError, TypeError, IndexError):
        return None, None

def scrape_kworb_chart():
    """Tải và phân tích bảng xếp hạng Spotify Daily Chart Vietnam từ Kworb."""
    try:
        # 1. Tải nội dung HTML
        response = requests.get(URL, timeout=10)
        response.raise_for_status() # Kiểm tra lỗi HTTP (4xx hoặc 5xx)
        
        soup = BeautifulSoup(response.content, 'html.parser')

        # 2. Tìm Bảng (ưu tiên HTML <table> nếu có), nếu không, dò <pre> hoặc văn bản dạng bảng
        # Note: Kworb often renders the chart as plain text lines starting with '|'.
        text_all = soup.get_text('\n')
        pipe_lines = [ln for ln in text_all.splitlines() if ln.strip().startswith('|')]

        # If there are many pipe-delimited lines, prefer parsing them instead of any HTML <table>
        if len(pipe_lines) > 10:
            chart_table = None
        else:
            chart_table = soup.find('table', class_='sortable') or soup.find('table')

        data = []

        if chart_table:
            rows = chart_table.find_all('tr')
        else:
            # Fallback: dùng các dòng bắt đầu bằng '|' (Kworb thường xuất bản bảng dạng văn bản)
            lines = pipe_lines
            if not lines:
                print("Không tìm thấy bảng xếp hạng (không có <table> hoặc dòng '|' trên trang).")
                return pd.DataFrame()

            # Tạo giả danh sách hàng giống cấu trúc tr -> td
            rows = []
            for ln in lines:
                # giữ nguyên dòng trong một thuộc tính để xử lý giống 'tr'
                tr = BeautifulSoup(f"<tr><td>{ln}</td></tr>", 'html.parser')
                rows.append(tr)
        
    # Gán Date cho lần thu thập này
    # Hỗ trợ override ngày từ biến môi trường SPOTIFY_DATE
        spotify_date = os.environ.get('SPOTIFY_DATE')
        # Try to parse the chart date from the page header (e.g. "Spotify Daily Chart - Vietnam - 2025/11/04 | Totals")
        chart_date = None
        try:
            for ln in text_all.splitlines():
                if 'Spotify Daily Chart' in ln or 'Spotify Daily' in ln:
                    m = re.search(r"(\d{4}/\d{2}/\d{2})", ln)
                    if m:
                        chart_date = m.group(1).replace('/', '-')
                        break
            # fallback: any yyyy/mm/dd anywhere on page
            if not chart_date:
                m2 = re.search(r"(\d{4}/\d{2}/\d{2})", text_all)
                if m2:
                    chart_date = m2.group(1).replace('/', '-')
        except Exception:
            chart_date = None

        if spotify_date:
            collection_date = spotify_date
        elif chart_date:
            collection_date = chart_date
        else:
            collection_date = datetime.now().strftime('%Y-%m-%d')

        # 3. Lặp qua các hàng, bỏ qua hàng tiêu đề nếu dạng table HTML có header
        # Tuỳ theo dạng (HTML table hoặc dòng '|' text) sẽ khác cách truy xuất

    # If rows are actual <tr> with <td>s (from an HTML table)
        if chart_table:
            # skip header if present
            start_idx = 1 if rows and rows[0].find_all(['th', 'td']) and any(th.name == 'th' for th in rows[0].find_all()) else 1
            for row in rows[1:]:
                cols = row.find_all('td')
                if len(cols) < 3:
                    # sometimes our fallback created a single-td with the whole line
                    text_line = row.get_text('\n').strip()
                    # attempt to parse like the text fallback below
                if len(cols) >= 3:
                    rank = cols[0].get_text(separator=' ', strip=True)
                    # normalize rank to integer when possible
                    rank_val = parse_number(rank)
                    # (debug prints removed) — do not print raw columns during normal runs
                    title_artist = cols[2].get_text(separator=' ', strip=True)
                    # try to find a numeric field after title (search remaining columns)
                    streams_text = ''
                    numeric_candidates = []
                    cols_text = [c.get_text(separator=' ', strip=True) for c in cols[3:]]
                    for c in cols_text:
                        if re.search(r'\d', c):
                            numeric_candidates.append(c)

                    # Prefer the numeric candidate that is followed by a signed change (+/-)
                    chosen = None
                    chosen_idx = -1
                    for i, cand in enumerate(cols_text):
                        if not re.search(r'\d', cand):
                            continue
                        # check next token for signed change
                        if i + 1 < len(cols_text) and re.match(r'^[+-][\d,]+$', cols_text[i+1].strip()):
                            chosen = cand
                            chosen_idx = i
                            break

                    # Otherwise prefer candidate that contains comma or numeric value >= 1000
                    if not chosen:
                        for cand in numeric_candidates:
                            raw = cand.replace('+', '').replace('-', '').replace(',', '').strip()
                            if raw.isdigit():
                                try:
                                    val = int(raw)
                                except:
                                    val = 0
                                if ',' in cand or val >= 1000:
                                    chosen = cand
                                    break

                    if not chosen and numeric_candidates:
                        chosen = numeric_candidates[0]
                        # locate its index in cols_text (first match)
                        for i, cand in enumerate(cols_text):
                            if cand == chosen:
                                chosen_idx = i
                                break
                    streams_text = chosen or ''
                    # Parse title and artist from cols[2]
                    title, artist, featured = split_title_artist(title_artist)

                    # days_on_chart from cols[3]
                    days_on_chart = None
                    if len(cols) > 3:
                        days_txt = cols[3].get_text(separator=' ', strip=True)
                        m = re.search(r"(\d+)", days_txt)
                        if m:
                            try:
                                days_on_chart = int(m.group(1))
                            except:
                                days_on_chart = None
                    
                    # peak_position and peak_count handling (robust)
                    peak_position = None
                    peak_count = None
                    if len(cols) > 4:
                        peak_txt = cols[4].get_text(separator=' ', strip=True)
                        if peak_txt:
                            # First try new flexible parser
                            peak_position, peak_count = parse_peak_data(peak_txt)

                    # If cols[5] contains an explicit (xN) marker, prefer that value for peak_count
                    if len(cols) > 5:
                        next_txt = cols[5].get_text(separator=' ', strip=True)
                        m = re.search(r"\(x(\d+)\)", next_txt)
                        if m:
                            peak_count = int(m.group(1))

                    # Determine daily_idx based on if cols[5] was the (xN) marker
                    if len(cols) > 5 and re.search(r"\(x\d+\)", cols[5].get_text(separator=' ', strip=True)):
                        daily_idx = 6
                    else:
                        daily_idx = 5

                    # daily_streams
                    daily_streams = None
                    if len(cols) > daily_idx:
                        streams_txt = cols[daily_idx].get_text(separator=' ', strip=True)
                        try:
                            if re.search(r"\d", streams_txt):
                                cleaned = streams_txt.replace(',', '').replace('+', '').replace('-', '')
                                cleaned = re.sub(r"[^0-9]", "", cleaned)
                                if cleaned:
                                    daily_streams = int(cleaned)
                        except:
                            daily_streams = None

                    # streams_change
                    streams_change = None
                    if len(cols) > daily_idx + 1:
                        change_txt = cols[daily_idx + 1].get_text(separator=' ', strip=True)
                        # Only accept if it looks like a signed number
                        if re.match(r'^\s*[+-][\d,]+\s*$', change_txt):
                            try:
                                cleaned = change_txt.replace(',', '')
                                # preserve negative sign
                                streams_change = int(re.sub(r"[^0-9-]", "", cleaned))
                            except:
                                streams_change = None
                        else:
                            streams_change = None

                    # Fallback: if daily_streams missing/zero, use 'chosen' token heuristic
                    if (daily_streams is None or daily_streams == 0) and chosen:
                        try:
                            c = re.sub(r"[^0-9]", "", chosen)
                            if c:
                                daily_streams = int(c)
                        except:
                            pass
                        # Attempt to get streams_change from the token right after chosen (if signed)
                        if streams_change is None and chosen_idx >= 0 and chosen_idx + 1 < len(cols_text):
                            nxt = cols_text[chosen_idx + 1].strip()
                            if re.match(r'^[+-][\d,]+$', nxt):
                                try:
                                    cleaned = nxt.replace(',', '')
                                    # preserve negative sign
                                    val = int(re.sub(r"[^0-9-]", "", cleaned))
                                    streams_change = val
                                except:
                                    pass

                    # seven_day_streams
                    seven_day_streams = None
                    if len(cols) > daily_idx + 2:
                        seven_txt = cols[daily_idx + 2].get_text(separator=' ', strip=True)
                        try:
                            cleaned = seven_txt.replace(',', '').replace('+', '').replace('-', '')
                            cleaned = re.sub(r"[^0-9]", "", cleaned)
                            if cleaned:
                                seven_day_streams = int(cleaned)
                        except:
                            seven_day_streams = None

                    # Absolute-position mapping (Kworb standard): 7-day streams @ col[8], 7-day change @ col[9], total @ col[10]
                    # Prefer absolute mapping if parseable
                    seven_abs = None
                    seven_chg_abs = None
                    total_abs = None
                    try:
                        if len(cols) > 8:
                            sev_txt = cols[8].get_text(separator=' ', strip=True)
                            if re.search(r"\d", sev_txt):
                                sev_clean = re.sub(r"[^0-9]", "", sev_txt.replace(',', ''))
                                if sev_clean:
                                    seven_abs = int(sev_clean)
                        if len(cols) > 9:
                            sev_chg_txt = cols[9].get_text(separator=' ', strip=True)
                            # Capture first signed number even with extra text like "+3,005 (7Day+)"
                            m = re.search(r'([+-]\s*[\d,]+)', sev_chg_txt)
                            if m:
                                sev_chg_clean = int(re.sub(r"[^0-9-]", "", m.group(1)))
                                seven_chg_abs = sev_chg_clean
                        if len(cols) > 10:
                            tot_txt = cols[10].get_text(separator=' ', strip=True)
                            if re.search(r"\d", tot_txt):
                                tot_clean = re.sub(r"[^0-9]", "", tot_txt.replace(',', ''))
                                if tot_clean:
                                    total_abs = int(tot_clean)
                    except:
                        pass

                    if seven_abs is not None:
                        seven_day_streams = seven_abs
                    if seven_chg_abs is not None:
                        seven_day_change = seven_chg_abs
                    if total_abs is not None:
                        total_streams = total_abs

                    # Fallbacks based on chosen index sequence if needed (only if not already set by absolute mapping)
                    if seven_day_change is None and len(cols) > daily_idx + 3:
                        seven_change_txt = cols[daily_idx + 3].get_text(separator=' ', strip=True)
                        if re.match(r'^\s*[+-][\d,]+\s*$', seven_change_txt):
                            try:
                                cleaned = seven_change_txt.replace(',', '')
                                if cleaned:
                                    seven_day_change = int(re.sub(r"[^0-9-]", "", cleaned))
                            except:
                                seven_day_change = None
                        else:
                            seven_day_change = None

                    # total_streams (only if not already set)
                    if total_streams is None and len(cols) > daily_idx + 4:
                        total_txt = cols[daily_idx + 4].get_text(separator=' ', strip=True)
                        try:
                            cleaned = total_txt.replace(',', '').replace('+', '').replace('-', '')
                            cleaned = re.sub(r"[^0-9]", "", cleaned)
                            if cleaned:
                                total_streams = int(cleaned)
                        except:
                            total_streams = None
                        # try cols[6] or cols[7] for 7-day streams (page variations)
                        cand7 = cols[6].get_text(separator=' ', strip=True)
                        if re.search(r'\d', cand7):
                            try:
                                seven_day_streams = int(re.sub(r'[^0-9]', '', cand7))
                            except:
                                seven_day_streams = None
                            # check next column for 7-day change
                            if len(cols) > 7:
                                try:
                                    seven_day_change = cols[7].get_text(separator=' ', strip=True)
                                    if re.search(r'[+-]?\d', seven_day_change):
                                        seven_day_change = int(re.sub(r'[^0-9-]', '', seven_day_change))
                                    else:
                                        seven_day_change = None
                                except:
                                    seven_day_change = None
                        elif len(cols) > 7:
                            cand7b = cols[7].get_text(separator=' ', strip=True)
                            if re.search(r'\d', cand7b):
                                    try:
                                        c = cand7b.replace(',', '').replace('+', '').replace('-', '')
                                        c = re.sub(r'[^0-9]', '', c)
                                        if c:
                                            seven_day_streams = int(c)
                                            # check next column for 7-day change
                                            if len(cols) > 8:
                                                try:
                                                    seven_day_change = cols[8].get_text(separator=' ', strip=True)
                                                    if re.search(r'[+-]?\d', seven_day_change):
                                                        seven_day_change = int(re.sub(r'[^0-9-]', '', seven_day_change))
                                                    else:
                                                        seven_day_change = None
                                                except:
                                                    seven_day_change = None
                                    except:
                                        seven_day_streams = None
                    # Fallbacks based on chosen index sequence if needed
                    if chosen_idx >= 0:
                        # Determine seven_day_streams as next numeric after (chosen_idx+1)
                        seven_idx_found = None
                        if seven_day_streams is None or seven_day_streams == 0 or (daily_streams is not None and seven_day_streams == daily_streams):
                            for j in range(chosen_idx + 2, len(cols_text)):
                                tok = cols_text[j]
                                if re.search(r"\d", tok) and not re.match(r'^[+-]', tok.strip()):
                                    try:
                                        seven_day_streams = int(re.sub(r"[^0-9]", "", tok))
                                        seven_idx_found = j
                                        break
                                    except:
                                        pass
                        # Determine seven_day_change using signed tokens list
                        if seven_day_change is None:
                            signed_positions = [(k, cols_text[k].strip()) for k in range(0, len(cols_text)) if re.match(r'^\s*[+-][\d,]+\s*$', cols_text[k])]
                            # Prefer signed appearing AFTER seven_idx_found
                            cand_vals = []
                            if seven_idx_found is not None:
                                for k, tok in signed_positions:
                                    if k > seven_idx_found:
                                        try:
                                            cand_vals.append((k, int(re.sub(r"[^0-9-]", "", tok))))
                                        except:
                                            pass
                            # Fallback: any signed token (prefer the last one), but different from streams_change
                            if not cand_vals and signed_positions:
                                for k, tok in signed_positions:
                                    try:
                                        val = int(re.sub(r"[^0-9-]", "", tok))
                                        if streams_change is None or val != streams_change:
                                            cand_vals.append((k, val))
                                    except:
                                        pass
                            if cand_vals:
                                # take the last candidate by index to likely match 7-day change at row end
                                cand_vals.sort(key=lambda x: x[0])
                                seven_day_change = cand_vals[-1][1]
                        # If still identical to streams_change, nullify to 0 to avoid duplication
                        if seven_day_change is not None and streams_change is not None and seven_day_change == streams_change:
                            seven_day_change = 0
                        # Determine total_streams as the last numeric token
                        if total_streams is None or total_streams == 0:
                            for j in range(len(cols_text)-1, -1, -1):
                                tok = cols_text[j]
                                if re.search(r"\d", tok) and not re.match(r'^[+-]', tok.strip()):
                                    try:
                                        total_streams = int(re.sub(r"[^0-9]", "", tok))
                                        break
                                    except:
                                        pass

                    # total_streams is ALWAYS in cols[10]
                    total_streams = None
                    if len(cols) > 10:
                        total_txt = cols[10].get_text(separator=' ', strip=True)
                        try:
                            if re.search(r"\d", total_txt):
                                cleaned = total_txt.replace(',', '').replace('+', '').replace('-', '')
                                cleaned = re.sub(r"[^0-9]", "", cleaned)
                                if cleaned:
                                    total_streams = int(cleaned)
                        except:
                            total_streams = None

                    data.append({
                        'Date': collection_date,
                        'Rank': rank_val,
                        'Title': title,
                        'Artists': artist,
                        'Featured_Artists': featured,
                        'days_on_chart': days_on_chart,
                        'peak_position': peak_position,
                        'peak_count': peak_count,
                        'daily_streams': daily_streams,
                        'streams_change': streams_change,
                        'seven_day_streams': seven_day_streams,
                        'seven_day_change': seven_day_change,
                        'total_streams': total_streams
                    })
        else:
            # rows are soup objects each containing a single td with the whole '|' line
            for row in rows:
                line = row.get_text().strip()
                # break into parts by '|'
                parts = [p.strip() for p in line.split('|') if p.strip() != '']
                if len(parts) < 3:
                    continue
                # common pattern: [rank, change, title, ... , daily_streams, ...]
                rank = parts[0]
                rank_val = parse_number(rank)
                title_artist = parts[2]
                # Robust extraction for pipe-format
                tail = parts[3:]
                # days_on_chart from first tail token numeric
                days_on_chart = None
                if tail:
                    m = re.search(r"(\d+)", tail[0])
                    if m:
                        try:
                            days_on_chart = int(m.group(1))
                        except:
                            days_on_chart = None

                # Find first numeric token followed by signed change => daily and its change
                daily_streams = None
                streams_change = None
                chosen_idx = -1
                for i, tok in enumerate(tail):
                    if not re.search(r"\d", tok):
                        continue
                    next_tok = tail[i+1].strip() if i+1 < len(tail) else ''
                    if re.match(r'^[+-][\d,]+$', next_tok):
                        try:
                            daily_streams = int(re.sub(r"[^0-9]", "", tok))
                        except:
                            daily_streams = None
                        try:
                            streams_change = int(re.sub(r"[^0-9-]", "", next_tok))
                        except:
                            streams_change = None
                        chosen_idx = i
                        break

                # If not found, fallback: first big numeric token as daily
                if daily_streams is None:
                    for i, tok in enumerate(tail):
                        if re.search(r"\d", tok) and not re.match(r'^[+-]', tok.strip()):
                            try:
                                val = int(re.sub(r"[^0-9]", "", tok))
                            except:
                                val = 0
                            if val >= 1000:
                                daily_streams = val
                                chosen_idx = i
                                # try get signed change next
                                if i+1 < len(tail):
                                    nxt = tail[i+1].strip()
                                    if re.match(r'^[+-][\d,]+$', nxt):
                                        try:
                                            streams_change = int(re.sub(r"[^0-9-]", "", nxt))
                                        except:
                                            pass
                                break

                # Seven-day streams: next numeric after chosen_idx+1
                seven_day_streams = None
                if chosen_idx >= 0:
                    for j in range(chosen_idx+2, len(tail)):
                        tok = tail[j]
                        if re.search(r"\d", tok) and not re.match(r'^[+-]', tok.strip()):
                            try:
                                seven_day_streams = int(re.sub(r"[^0-9]", "", tok))
                                break
                            except:
                                pass

                # Seven-day change: prefer a signed different from streams_change; pick the last signed after seven_day_streams
                seven_day_change = None
                if chosen_idx >= 0:
                    signed_positions = [(j, tail[j].strip()) for j in range(0, len(tail)) if re.match(r'^\s*[+-][\d,]+\s*$', tail[j])]
                    seven_idx_found = None
                    # find index of seven_day_streams token again
                    for j in range(chosen_idx+2, len(tail)):
                        tok = tail[j]
                        if re.search(r"\d", tok) and not re.match(r'^[+-]', tok.strip()):
                            seven_idx_found = j
                            break
                    cand_vals = []
                    if seven_idx_found is not None:
                        for k, tok in signed_positions:
                            if k > seven_idx_found:
                                try:
                                    val = int(re.sub(r"[^0-9-]", "", tok))
                                    if streams_change is None or val != streams_change:
                                        cand_vals.append((k, val))
                                except:
                                    pass
                    if not cand_vals and signed_positions:
                        for k, tok in signed_positions:
                            try:
                                val = int(re.sub(r"[^0-9-]", "", tok))
                                if streams_change is None or val != streams_change:
                                    cand_vals.append((k, val))
                            except:
                                pass
                    if cand_vals:
                        cand_vals.sort(key=lambda x: x[0])
                        seven_day_change = cand_vals[-1][1]

                # Total streams: last numeric token
                total_streams = None
                for j in range(len(tail)-1, -1, -1):
                    tok = tail[j]
                    if re.search(r"\d", tok) and not re.match(r'^[+-]', tok.strip()):
                        try:
                            total_streams = int(re.sub(r"[^0-9]", "", tok))
                            break
                        except:
                            pass

                title, artist, featured = split_title_artist(title_artist)
                # Create row data with all fields, ensuring peak data is properly initialized
                row_data = {
                    'Date': collection_date,
                    'Rank': rank_val if rank_val is not None else 0,
                    'Title': title,
                    'Artists': artist,
                    'Featured_Artists': featured,
                    'days_on_chart': days_on_chart if days_on_chart is not None else 0,
                    'peak_position': peak_position if peak_position is not None else 0,
                    'peak_count': peak_count if peak_count is not None else 0,
                    'daily_streams': daily_streams if daily_streams is not None else 0,
                    'streams_change': streams_change if streams_change is not None else 0,
                    'seven_day_streams': seven_day_streams if seven_day_streams is not None else 0,
                    'seven_day_change': seven_day_change if seven_day_change is not None else 0,
                    'total_streams': total_streams if total_streams is not None else 0
                }
                
                # Replace None with 0 for numeric fields
                numeric_cols = ['Rank', 'days_on_chart', 'peak_position', 'peak_count',
                              'daily_streams', 'streams_change', 'seven_day_streams',
                              'seven_day_change', 'total_streams']
                for col in numeric_cols:
                    if row_data[col] is None:
                        row_data[col] = 0
                
                # Add row to data list
                data.append(row_data)

            # Build DataFrame with specified data types
        df = pd.DataFrame(data, columns=[
            'Date', 'Rank', 'Title', 'Artists', 'Featured_Artists',
            'days_on_chart', 'peak_position', 'peak_count', 'daily_streams',
            'streams_change', 'seven_day_streams', 'seven_day_change', 'total_streams'
        ])
        
        # Ensure numeric columns are properly typed
        num_cols = ['Rank', 'days_on_chart', 'peak_position', 'peak_count', 
                       'daily_streams', 'streams_change', 'seven_day_streams', 
                       'seven_day_change', 'total_streams']
        
        # Convert numeric columns to int with 0 for NaN values
        for col in num_cols:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
            except Exception as e:
                print(f"Error converting {col}: {e}")
                df[col] = df[col].fillna(0).astype(int)
                   
        # Convert numeric columns silently (use 0 for NaN)
        for col in num_cols:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
            except Exception:
                # Fallback to numeric coercion without forcing int
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Limit to top 100 by rank to satisfy requirement
        try:
            if 'Rank' in df.columns:
                # Keep rows with positive rank and take top 100
                df = df[df['Rank'] > 0].sort_values('Rank').head(100).reset_index(drop=True)
        except Exception:
            # if anything goes wrong, return the original df
            pass

        return df

    except requests.exceptions.HTTPError as errh:
        print(f"Lỗi HTTP: {errh}")
    except requests.exceptions.RequestException as erre:
        print(f"Lỗi Kết nối: {erre}")
    except Exception as e:
        print(f"Lỗi không xác định: {e}")
    return pd.DataFrame()


def repair_malformed_csv(path):
    """Repair a malformed kworb CSV by parsing lines and rebuilding a clean CSV with the current schema."""
    import csv
    repaired = []
    header = ['Date','Rank','Title','Artists','Featured_Artists','Days_on_chart','Peak_position','Peak_count','Daily_streams','Streams_change','Seven_day_streams','Seven_day_change','Total_streams']
    try:
        with open(path, 'r', encoding='utf-8-sig', newline='') as f:
            reader = csv.reader(f)
            try:
                first = next(reader)
            except StopIteration:
                return False

            for row in reader:
                if not row:
                    continue
                # basic mapping
                ts = row[0] if len(row) > 0 else ''
                rank = row[1] if len(row) > 1 else ''
                title_artist = row[2] if len(row) > 2 else ''
                tail = row[3:]

        # Use split helper from this module
        title, artist, featured = split_title_artist(title_artist)

        # Initialize peak fields
        peak_position = None
        peak_count = None

        # Parse peak info if available using helper function
        peak_txt = row[4] if len(row) > 4 else None
        if peak_txt:
            peak_position, peak_count = parse_peak_data(peak_txt)

        # days_on_chart from first numeric field
        days_on_chart = None
        if len(tail) > 0:
            m = re.search(r"(\d+)", tail[0])
            if m:
                try:
                    days_on_chart = int(m.group(1))
                except:
                    days_on_chart = None
                    daily_streams = None
                if len(tail) > 1:
                    try:
                        cleaned = tail[1].replace(',', '').replace('+', '').replace('-', '')
                        cleaned = re.sub(r"[^0-9]", "", cleaned)
                        if cleaned:
                            daily_streams = int(cleaned)
                    except:
                        daily_streams = None

                streams_change = None
                for t in tail:
                    if re.search(r"\d", t):
                        sc_clean = t.replace(',', '').replace('+', '').replace('-', '')
                        sc_clean = re.sub(r"[^0-9]", "", sc_clean)
                        try:
                            if sc_clean:
                                streams_change = int(sc_clean)
                        except:
                            streams_change = None
                        break

                nums_after = []
                for t in tail:
                    if re.search(r"\d", t):
                        try:
                            c = t.replace(',', '').replace('+', '').replace('-', '')
                            c = re.sub(r"[^0-9]", "", c)
                            if c:
                                nums_after.append(int(c))
                        except:
                            pass
                seven_day_streams = None
                seven_day_change = None
                total_streams = None
                
                # Parse numeric tokens for 7-day streams and change
                if len(nums_after) >= 1:
                    seven_day_streams = nums_after[0]
                    if len(nums_after) >= 2:
                        # Look for signed number for 7-day change
                        for t in tail:
                            if re.search(r'[+-]?\d', t):
                                try:
                                    seven_day_change = int(re.sub(r'[^0-9-]', '', t))
                                    break
                                except:
                                    pass
                        # Last number is usually total streams
                        total_streams = nums_after[-1]

                daily_streams_text = str(daily_streams) if daily_streams is not None else ''

                repaired.append({
                    'Date': ts,
                    'Rank': parse_number(rank),
                    'Title': title,
                    'Artists': artist,
                    'Featured_Artists': featured,
                    'days_on_chart': days_on_chart,
                    'daily_streams': daily_streams,
                    'streams_change': streams_change,
                    'seven_day_streams': seven_day_streams,
                    'seven_day_change': seven_day_change,
                    'total_streams': total_streams,
                    'daily_streams_text': daily_streams_text
                })
                # Date,Rank,Title,Artists,Duration,NCT_URL,Total_Likes
        # write repaired file
        df_rep = pd.DataFrame(repaired)
        # coerce numeric cols
        for col in ['days_on_chart','daily_streams','streams_change','seven_day_streams','total_streams','Rank']:
            if col in df_rep.columns:
                df_rep[col] = pd.to_numeric(df_rep[col], errors='coerce').astype('Int64')

        df_rep.to_csv(path, index=False, encoding='utf-8-sig')
        return True
    except Exception as e:
        print(f"Không thể repair CSV: {e}")
        return False

# =======================================================
# 3. LƯU TRỮ DỮ LIỆU VÀ LỌC SẠCH
# =======================================================
# ===== VPOP FILTER LOGIC (từ filter_vietnam_songs.py) =====
# Danh sách nghệ sĩ quốc tế cần loại bỏ (phải match chính xác toàn bộ tên)
INTERNATIONAL_ARTISTS_EXACT = [
    'Jung Kook', 'Jungkook', 'BTS', 'Jimin', 'RM', 'Suga', 'Jin', 'J-Hope',
    'BLACKPINK', 'Rosé', 'Jennie', 'Lisa', 'Jisoo',
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
    'LE SSERAFIM','JISOO','ZAYN'
]

# Danh sách nghệ sĩ quốc tế (match nếu xuất hiện trong tên)
INTERNATIONAL_KEYWORDS = [
    'Jung Kook', 'Jungkook', 'Jack Harlow', 'Latto',
    'Park Hyo Shin'
]

def is_international_artist(artist, featured):
    """Kiểm tra xem có phải nghệ sĩ quốc tế không"""
    import pandas as pd
    artist_str = str(artist).strip()
    featured_str = str(featured if pd.notna(featured) else '').strip()
    full_artist = artist_str + ' ' + featured_str
    
    # Check exact match với tên chính
    if artist_str in INTERNATIONAL_ARTISTS_EXACT:
        return True
    
    # Check featured artists
    if featured_str:
        for intl in INTERNATIONAL_ARTISTS_EXACT:
            if intl in featured_str:
                return True
    
    # Check keywords trong full artist string
    full_lower = full_artist.lower()
    for keyword in INTERNATIONAL_KEYWORDS:
        if keyword.lower() in full_lower:
            return True
    
    # Special case: "V" chỉ khi nó là nghệ sĩ chính và đứng một mình
    if artist_str == 'V':
        return True
    
    return False

def filter_vpop_and_deduplicate(df):
    """
    Filter chỉ lấy VPOP và xóa duplicate
    - Loại bỏ các nghệ sĩ quốc tế (exact match và keyword)
    - Xóa các bản ghi trùng lặp (cùng Date, Title, Artist)
    """
    if df.empty:
        return df
    
    # 1. FILTER VPOP - Logic từ filter_vietnam_songs.py
    # Filter ra non-VPOP
    df_vpop = df[~df.apply(lambda row: is_international_artist(row['Artists'], row['Featured_Artists']), axis=1)].copy()
    
    removed_count = len(df) - len(df_vpop)
    if removed_count > 0:
        print(f"  ✓ Đã loại bỏ {removed_count} bài hát non-VPOP")
    
    # 2. DEDUPLICATE
    # Xác định key để check duplicate
    if 'Date' in df_vpop.columns:
        duplicate_cols = ['Date', 'Title', 'Artists']
    else:
        duplicate_cols = ['Title', 'Artists']
    
    # Check duplicate
    initial_count = len(df_vpop)
    df_vpop = df_vpop.drop_duplicates(subset=duplicate_cols, keep='first')
    
    dup_removed = initial_count - len(df_vpop)
    if dup_removed > 0:
        print(f"  ✓ Đã xóa {dup_removed} bản ghi duplicate")
    
    return df_vpop

if __name__ == '__main__':
    # 1. Cào dữ liệu mới
    df_new = scrape_kworb_chart()

    if not df_new.empty:
        # Remove remix titles (case-insensitive)
        remix_mask = df_new['Title'].str.contains(r'(?i)remix', na=False)
        if remix_mask.any():
            print(f"🔁 Loại bỏ {remix_mask.sum()} bài có 'Remix' trong tiêu đề")
            df_new = df_new[~remix_mask].reset_index(drop=True)
        # Clean group names quotes
        df_new['Artists'] = df_new['Artists'].apply(clean_group_name)
        df_new['Featured_Artists'] = df_new['Featured_Artists'].apply(clean_group_name)
    
    if df_new.empty:
        print("❌ Không tìm thấy dữ liệu mới từ Kworb.")
    else:
        print(f"✅ Tải xong {len(df_new)} bản ghi từ Kworb.")
        
        # 2. Lọc VPOP và xóa duplicate (trong batch mới)
        print("\n🔍 Đang filter VPOP và xóa duplicate (trong batch mới)...")
        df_new = filter_vpop_and_deduplicate(df_new)
        
        if df_new.empty:
            print("❌ Không còn dữ liệu mới sau khi filter VPOP!")
        else:
            print(f"✅ Còn lại {len(df_new)} bản ghi VPOP mới sau khi filter.\n")
            
            OUTPUT_FILE = os.environ.get('SPOTIFY_OUTPUT_FILE', FILE_NAME)
            
            # 3. Lấy ngày của dữ liệu vừa cào
            # Giả định tất cả dữ liệu mới đều cùng một ngày
            current_data_date = df_new['Date'].iloc[0]
            
            file_exists = os.path.exists(OUTPUT_FILE)
            date_already_exists = False

            # 4. Kiểm tra xem file có tồn tại và ngày này đã được ghi chưa
            if file_exists:
                try:
                    # Tải NHANH CỘT 'Date' để kiểm tra trùng lặp
                    df_old_dates = pd.read_csv(OUTPUT_FILE, usecols=['Date'])
                    if not df_old_dates.empty and current_data_date in df_old_dates['Date'].values:
                        date_already_exists = True
                except Exception as e:
                    print(f"⚠️ Lỗi khi đọc file cũ ({e}). Coi như file không tồn tại.")
                    file_exists = False # Coi như file lỗi, sẽ ghi mới

            # 5. Quyết định ghi hay không (hỗ trợ overwrite qua ENV SPOTIFY_OVERWRITE=1)
            overwrite_flag = os.environ.get('SPOTIFY_OVERWRITE', '0') == '1'
            if date_already_exists and not overwrite_flag:
                print(f"❌ Dữ liệu cho ngày {current_data_date} đã tồn tại. → Bỏ qua, không scrape lại..")
            else:
                # If overwriting, remove existing rows for that date
                if overwrite_flag and file_exists:
                    try:
                        df_old = pd.read_csv(OUTPUT_FILE, encoding='utf-8-sig')
                        before_len = len(df_old)
                        df_old = df_old[df_old['Date'] != current_data_date]
                        df_old.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
                        print(f"🧹 Đã xóa {before_len - len(df_old)} bản ghi cũ cho ngày {current_data_date}.")
                        file_exists = os.path.exists(OUTPUT_FILE) and len(df_old) > 0
                    except Exception as e:
                        print(f"⚠️ Không thể xóa dữ liệu cũ cho ngày {current_data_date}: {e}")
                # Nếu file chưa tồn tại (file_exists=False), 'header' sẽ là True.
                # Nếu file đã tồn tại (file_exists=True), 'header' sẽ là False.
                try:
                    df_new.to_csv(OUTPUT_FILE, 
                                  mode='a',  # 'a' = (APPEND)
                                  header=(not file_exists), # Chỉ ghi header nếu file CHƯA TỒN TẠI
                                  index=False, 
                                  encoding='utf-8-sig')
                    
                    if file_exists and not overwrite_flag:
                        print(f"✅ Đã thêm {len(df_new)} bản ghi mới vào {OUTPUT_FILE}.")
                    else:
                        print(f"✅ Đã TẠO MỚI file {OUTPUT_FILE} với {len(df_new)} bản ghi.")
                        
                except Exception as e:
                    print(f"Lỗi khi lưu file: {e}")
                    import traceback
                    traceback.print_exc()
            try:
                if os.path.exists(OUTPUT_FILE):
                    total_df = pd.read_csv(OUTPUT_FILE)
                    print(f"Tổng bản ghi của file {OUTPUT_FILE}: {len(total_df)}")
                else:
                    print("File chưa tồn tại.")
            except Exception as e:
                print(f"Không thể đọc tổng số bản ghi: {e}")