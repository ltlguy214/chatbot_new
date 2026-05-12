import json
import re
import os
import time
import hashlib
import difflib
from pydantic import BaseModel, Field

try:
    from chatbot.env import load_env
    from chatbot.dictionaries import MOOD_MAP, GENRE_KEYWORDS, POPULARITY_KEYWORDS, ATTRIBUTE_KEYWORDS, KNOWLEDGE_KEYWORDS
    from chatbot.text_processing import normalize_text_nfd_strip_accents, normalize_teencode
except ModuleNotFoundError:
    from env import load_env
    from dictionaries import MOOD_MAP, GENRE_KEYWORDS, POPULARITY_KEYWORDS, ATTRIBUTE_KEYWORDS, KNOWLEDGE_KEYWORDS
    from text_processing import normalize_text_nfd_strip_accents, normalize_teencode

load_env()

_DEFAULT_INTENT_MODEL_CANDIDATES: list[str] = ['gemini-2.0-flash']
ALLOWED_ACTIONS: set[str] = {"SEARCH_TRACK", "SEARCH_AUDIO", "DISCOVER_MUSIC", "ANALYZE_READY", "GENERAL_CHAT"}
_GEMINI_KEY_COOLDOWN_UNTIL: dict[str, float] = {}

class IntentParams(BaseModel):
    song_title: str = Field(default="")
    artist: str = Field(default="")
    mood: str = Field(default="")
    genre: str = Field(default="")
    lyric_snippet: str = Field(default="")
    seed_name: str = Field(default="")
    attributes: str = Field(default="")
    popularity_flag: bool = Field(default=False)

def _empty_params() -> dict:
    return {"song_title": "", "artist": "", "mood": "", "genre": "", "lyric_snippet": "", "seed_name": "", "attributes": "", "popularity_flag": False}

# --- CÁC HÀM TIỆN ÍCH LLM ---
def _key_fingerprint(api_key: str) -> str:
    try: return hashlib.sha256(str(api_key).encode('utf-8')).hexdigest()[:16]
    except Exception: return "unknown"

def _parse_gemini_api_keys() -> list[str]:
    keys = []
    raw = str(os.getenv("GEMINI_API_KEYS") or "").strip()
    if raw:
        for part in re.split(r"[\n,;]+", raw):
            if part.strip() and part.strip() not in keys: keys.append(part.strip())
    single = str(os.getenv("GEMINI_API_KEY") or "").strip()
    if single and single not in keys: keys.append(single)
    return keys

def _normalize_model_name(name: str) -> str:
    name = str(name or '').strip()
    if name.startswith('models/'): name = name[len('models/'):]
    return name

def _intent_model_candidates_from_env() -> list[str]:
    raw = str(os.getenv('GEMINI_INTENT_MODEL') or '').strip()
    raw2 = str(os.getenv('GEMINI_MODEL') or '').strip()
    candidates = []
    for v in [raw, raw2]:
        n = _normalize_model_name(v)
        if n and n not in candidates: candidates.append(n)
    for m in _DEFAULT_INTENT_MODEL_CANDIDATES:
        if m not in candidates: candidates.append(m)
    return candidates

def _looks_like_quota_error(err: Exception) -> bool:
    msg = str(err).lower()
    return any(k in msg for k in ["resource_exhausted", "quota", "429", "rate limit"])

def resolve_coreference(user_input: str, history_context: list, has_file: bool = False) -> str:
    import re
    lower_input = user_input.lower().strip()
    if has_file:
        return user_input
    if re.search(r'\b(là bài gì|bài gì|nhạc gì|tên gì|nhận diện|shazam|của ai|ai hát)\b', lower_input):
        return user_input
    
    pronouns_list = [
        'anh ấy', 'anh ay', 'cô ấy', 'co ay', 'người đó', 'nguoi do', 
        'ca sĩ đó', 'ca si do', 'chú ấy', 'chu ay', 'nhóm đó', 'nhom do', 
        'ông ấy', 'ong ay', 'anh y', 'co y', 'ổng', 'bả', 'chả', 
        'bài này', 'bai nay'
    ]
    lower_input = user_input.lower().strip()
    
    # --- SỬA TẠI ĐÂY: Dùng Regex \b để bắt chính xác nguyên từ ---
    # pattern này sẽ đảm bảo 'bả' không khớp với 'bản', 'chả' không khớp với 'chẳng'
    pattern = r'\b(' + '|'.join(re.escape(p) for p in pronouns_list) + r')\b'
    
    if not re.search(pattern, lower_input):
        return user_input
    # -----------------------------------------------------------

    # Nếu có đại từ thực sự, lục lại lịch sử RAM
    for msg in reversed(history_context):
        if msg.get('role') == 'assistant' and msg.get('track_previews'):
            # Lấy nghệ sĩ từ bài hát được recommend gần nhất
            last_artist = msg['track_previews'][0].get('artist')
            if last_artist:
                # Kiểm tra thêm: Nếu trong câu user gõ đã có tên nghệ sĩ khác thì không đè vào
                # (Tránh trường hợp: "Tìm nhạc Low G cho bả" -> bị biến thành Low G của Low G)
                if last_artist.lower() in lower_input:
                    return user_input
                
                print(f"🔄 [RAM CACHE] Dịch đại từ trong: '{user_input}' -> '{last_artist}'")
                return user_input + f" của {last_artist}"
                
    return user_input

def call_gemini_intent_api(user_input: str, has_file: bool) -> dict | None:
    file_context = "ĐÃ TẢI LÊN" if has_file else "CHƯA TẢI LÊN"
    prompt = f"""Bạn là AI điều phối cho hệ thống âm nhạc V-Pop. Dưới đây là thông tin hội thoại (có thể bao gồm lịch sử trò chuyện để bạn nắm ngữ cảnh): Câu hỏi: "{user_input}"

    Nhiệm vụ: Dựa vào ngữ cảnh (nếu có) và câu cuối cùng người dùng nói, hãy phân tích ý định hiện tại. 
    Ngữ cảnh: Người dùng {file_context} file nhạc.
    HÃY CHỌN 1 ACTION DUY NHẤT TRONG CÁC TRƯỜNG HỢP SAU. ĐỌC KỸ ĐIỀU KIỆN ÁP DỤNG:  
    1. SEARCH_TRACK: Tìm bài hát bằng văn bản.
       - Nếu user đưa tên bài (có thể kèm ca sĩ): trích xuất vào `song_title` và `artist`.
       - Nếu user đưa một đoạn lời/lyrics/câu hát: trích xuất vào `lyric_snippet` (có thể kèm `artist`).
       - Nếu không chắc là tên hay lời: ưu tiên điền `song_title`.
    2. SEARCH_AUDIO: Tìm bài hát bằng giai điệu/âm thanh (Shazam-like).
       - LLM KHÔNG cần và KHÔNG được đoán trạng thái file.
       - Nếu user hỏi "Bài này là bài gì?" / "Nhạc gì đây?" thì chọn action này.
    3. DISCOVER_MUSIC: Khám phá & gợi ý nhạc (trích xuất TẤT CẢ thông tin vào params tương ứng):
       - NẾU người dùng hỏi dạng nghi vấn (VD: "Có bài nào của A không?", "Nhạc gì buồn vậy?"): ĐÂY VẪN LÀ LỆNH TÌM NHẠC. Bắt buộc chọn DISCOVER_MUSIC, TUYỆT ĐỐI KHÔNG chọn GENERAL_CHAT.
       - Bài mẫu: `seed_name` (VD: "Tìm bài giống bài Suýt Nữa Thì").
       - Thuộc tính nhạc lý: `attributes` (VD: "nhịp 120 bpm", "nhạc nhanh", "năng lượng thấp", "vừa phải").
       - Top hit / trending: `popularity_flag=true` (có thể kèm `artist`, `genre`).
       - Mood/Genre/Artist (một hoặc nhiều tiêu chí): `mood`, `genre`, `artist`. mood: Trích xuất cảm giác/mục đích (VD: "để ngủ" -> "chill", "buồn").
       - Nếu người dùng yêu cầu phủ định (VD: "Đừng mở nhạc buồn, đổi bài vui đi"), hãy HIỂU Ý HỌ VÀ BỎ QUA yếu tố phủ định ("buồn"), chỉ trích xuất yếu tố mong muốn ("vui") vào biến `mood`. Tuyệt đối không chọn GENERAL_CHAT cho các câu đổi nhạc.
    4. ANALYZE_READY: Phân tích file audio để dự đoán Hit/Viral.
       - LLM KHÔNG cần và KHÔNG được đoán trạng thái file.
       - Nếu user bảo "phân tích bài này", chọn action này.
    5. GENERAL_CHAT: Mọi câu hỏi không thuộc các nhóm trên (chào hỏi, kiến thức, lạc đề, mơ hồ...).

    QUY TẮC TRÍCH XUẤT PARAMS:
        - Điền đúng field tương ứng. Nếu không có, để rỗng "" (hoặc false cho boolean).
        - LƯU Ý KẾT HỢP NGỮ CẢNH: Nếu hội thoại đang diễn ra (VD: trước đó nhắc đến "tempo 100", giờ nói "rap đi"), BẠN PHẢI GỘP cả điều kiện cũ (attributes: "tempo 100") và mới (genre: "rap") vào params. Tuyệt đối không được "rơi rụng" yêu cầu cũ.
        - Không tự bịa param.
    
    RULE QUAN TRỌNG:
    CHỈ CHỌN 1 ACTION PHÙ HỢP NHẤT. KHÔNG ĐƯỢC CHỌN NHIỀU HƠN 1 ACTION. NẾU KHÔNG CHẮC, HÃY CHỌN GENERAL_CHAT.
    CHỈ TRẢ VỀ JSON:
    {{"action": "TÊN_ACTION", "params": {{"song_title": "", "artist": "", "mood": "", "genre": "", "lyric_snippet": "", "seed_name": "", "attributes": "", "popularity_flag": false}}, "thought": ""}}"""
    
    keys = _parse_gemini_api_keys()
    if not keys: return None
    now = time.time()
    for api_key in keys:
        fp = _key_fingerprint(api_key)
        if _GEMINI_KEY_COOLDOWN_UNTIL.get(fp, 0) > now: continue
        try:
            from google import genai
            client = genai.Client(api_key=api_key)
            for model_name in _intent_model_candidates_from_env():
                try:
                    response = client.models.generate_content(model=model_name, contents=prompt)
                    text = getattr(response, "text", "")
                    match = re.search(r"\{.*\}", text, re.DOTALL)
                    if match:
                        data = json.loads(match.group())
                        data['method'] = "LLM"
                        return data
                except Exception as ex:
                    if _looks_like_quota_error(ex):
                        _GEMINI_KEY_COOLDOWN_UNTIL[fp] = time.time() + 600
                        break
        except Exception:
            continue
    return None

# ====================================================================
# TÁI TẠO HOÀN TOÀN LOGIC CŨ Ở ĐÂY
# ====================================================================
def analyze_user_intent(user_input: str, history_context: list, has_file: bool = False, known_artists: list = None) -> dict:
    if known_artists is None: known_artists = []
    import re
    import unicodedata
    user_input = re.sub(r'["\'“”‘’]', '', user_input)
    user_input = re.sub(r'^(helo|hello|hi|chào|chao|ê|e|bot|ad|admin)\b\s*', '', user_input, flags=re.IGNORECASE).strip()
    user_input = re.sub(r'(?i)\b(\d+|mười|một|hai|ba|bốn|năm|sáu|bảy|tám|chín|chục|trăm)\s*k\b', r'\1 ngàn', user_input)
    user_input = re.sub(r'(?i)\b(nhac|bai|ca|loi|hat|list|quay|bungno|chill|buon|vui|remix)\b(?=[a-z])', r'\1 ', user_input)
    user_input = re.sub(r'(?i)(?<=[a-z])(nhac|bai|ca|loi|hat|quay|party|remix|hit|hot)\b', r' \1', user_input)
    user_input = user_input.replace('nhacquay', 'nhạc quẩy').replace('quayparty', 'quẩy party')
    # --- [LÁ CHẮN MỚI]: Bảo vệ nghệ sĩ có 1 chữ cái đứng đầu khỏi bị dịch thành Teencode ---
    user_input = re.sub(r'(?i)\b(b)\s+(ray)\b', r'bray', user_input)
    user_input = re.sub(r'(?i)\b(a)\s+(mee)\b', r'amee', user_input)
    user_input = re.sub(r'(?i)\b(g)\s+(ducky)\b', r'gducky', user_input)
    user_input = re.sub(r'(?i)\b(t)\s+(linh)\b', r'tlinh', user_input)
    user_input = re.sub(r'(?i)\b(k)\s+(icm)\b', r'kicm', user_input)
    # -----------------------------------------------------------------------------------

    user_input = normalize_teencode(user_input)
    resolved_input = resolve_coreference(user_input, history_context)
    lower_prompt = resolved_input.lower()
    prompt_norm = normalize_text_nfd_strip_accents(resolved_input)

    artist_val = ""
    mood_val = ""
    genre_val = ""
    found_moods = []
    found_genres = []
    active_criteria = []
    is_strong_artist = False 
    intent_data = {"action": "GENERAL_CHAT", "params": _empty_params(), "thought": "", "method": "Rule"}

    # HÀM _verify_artist CHÍNH XÁC NHƯ BẢN CŨ CỦA BẠN
    def _verify_artist(a_name: str, is_strong_intent: bool = False) -> str:
        if not a_name or not known_artists: return ""
        titles_to_strip = [
            'chị đẹp', 'chi dep', 'anh trai', 'ca sĩ', 'ca si', 'nghệ sĩ', 'nghe si',
            'idol', 'thần tượng', 'than tuong', 'diva', 'rapper', 'nhóm nhạc', 'nhom nhac',
            'ban nhạc', 'ban nhac', 'ông hoàng', 'ong hoang', 'nữ hoàng', 'nu hoang',
            'nhạc sĩ', 'nhac si', 'đội', 'team',
            'bản mp3', 'ban mp3', 'mp3', 'audio', 'official', 'lyrics', 'video', 'bản',
            'cực gắt', 'cuc gat', 'cực mạnh', 'cuc manh', 'cực', 'cuc', 'gắt', 'gat', 
            'cháy', 'chay', 'siêu', 'sieu', 'rất', 'rat', 'quá', 'lắm', 'lam'
        ]
        temp_name = a_name.lower().strip()
        for title in titles_to_strip:
            temp_name = re.sub(rf'\b{title}\b', '', temp_name, flags=re.IGNORECASE).strip()
        if not temp_name: return ""

        forbidden_words = {
            'nhưng', 'và', 'thì', 'mà', 'là', 'rằng', 'tuy', 'cho', 'của', 'để', 'những', 
            'các', 'một', 'cái', 'lắm', 'quá', 'thật', 'nào', 'gì', 'chi', 'nhỉ', 'hả', 
            'nữa', 'thôi', 'luôn', 'chứ', 'đi', 'nhé', 'nha', 'thử', 'trong', 'ngoài', 
            'nay', 'này', 'đó', 'kia', 'vậy', 'vừa', 'vừa phải', 'vua', 'vua phai',
            'năng lượng', 'nang luong', 'tempo', 'nhịp', 'nhip', 'luong',
            'tầm', 'khoảng', 'mức', 'nhiều', 'ít', 'chậm', 'nhanh', 'mạnh', 'nhẹ', 'uy luc', 'luc', 'lực',
            'cao', 'thấp', 'thap', 'vừa', 'nhiều', 'style',
            'cực kỳ', 'cuc ky', 'cực kì', 'cuc ki', 'rất', 'rat', 'siêu', 'sieu', 
            'quá', 'hơi', 'hoi', 'khá', 'kha', 'nhất', 'nhat', 'nhì', 'nhi', 'đỉnh', 'dinh', 'đám', 'dam', 'lắm', 'lam', 'quá', 'qua',
            'có', 'co', 'nghe', 'nhạc', 'nhac', 'bài', 'bai', 'và', 'va', 'với', 'voi', 
            'ổn định', 'on dinh', 'tâm hồn', 'tam hon',
            'thật', 'cạn', 'thật cạn',
            'đang', 'dang', 'thịnh hành', 'thinh hanh', 'hot', 'top', 'viral', 'trending',
            'em', 'anh', 'tôi', 'toi', 'mình', 'minh', 'bạn', 'ban', 'nó', 'no', 'họ', 'ho',
            'hiện tại', 'hien tai', 'giống', 'tựa', 'kiểu', 'tương tự', 'như', 'giong', 'tua', 'kieu', 'tuong tu',
            'ngày hôm qua', 'ngay hom qua', 'quá khứ', 'qua khu', 'tương lai', 'tuong lai'
            'bot', 'ad', 'admin', 'chatbot', 'ai', 'vmusic'
        }
        no_accent_forbidden = {normalize_text_nfd_strip_accents(w) for w in forbidden_words}
        forbidden_words.update(no_accent_forbidden)
        if temp_name in forbidden_words: return ""

        a_norm = normalize_text_nfd_strip_accents(temp_name)
        if a_norm in forbidden_words: return ""

        a_norm = re.sub(r'[.?!,\-&_()\[\]]', '', a_norm).strip()
        if len(a_norm) < 2: return "" 
        
        a_nospace = a_norm.replace(" ", "")
        a_words = a_norm.split()
        
        prefix_match = ""
        prefix_score = 0
        contains_match = ""
        contains_score = 9999
        
        # ==========================================
        # QUÉT VÒNG 1: KHỚP ĐÍCH DANH 100% 
        # ==========================================
        for db_a in known_artists:
            db_a_norm = normalize_text_nfd_strip_accents(db_a)
            db_a_norm = re.sub(r'[.?!,\-&_()\[\]]', '', db_a_norm).strip()
            
            if a_norm == db_a_norm: 
                return db_a # Tìm thấy người giống hệt thì thoát luôn!
        # ==========================================
        # QUÉT VÒNG 2: KHỚP DÍNH CHỮ & NẰM TRONG (Chỉ chạy khi Vòng 1 trượt)
        # ==========================================
        prefix_match = ""
        prefix_score = 0
        contains_match = ""
        contains_score = 9999

        for db_a in known_artists:
            db_a_norm = normalize_text_nfd_strip_accents(db_a)
            db_a_norm = re.sub(r'[.?!,\-&_()\[\]]', '', db_a_norm).strip()
            if not db_a_norm: continue

            db_a_nospace = db_a_norm.replace(" ", "")

            # B. KHỚP DÍNH CHỮ
            if a_nospace and db_a_nospace:
                if a_nospace in db_a_nospace and len(a_nospace) >= len(db_a_nospace) * 0.6: return db_a
                elif db_a_nospace in a_nospace and len(db_a_nospace) >= len(a_nospace) * 0.6: return db_a

            # C. NGHỆ SĨ NẰM ĐẦU CÂU
            if a_norm.startswith(db_a_norm + " "):
                if len(db_a_nospace) > 3 or len(a_words) <= 4 or is_strong_intent:
                    if len(db_a_norm) > prefix_score:
                        prefix_score = len(db_a_norm)
                        prefix_match = db_a
                    
            # D. NGHỆ DANH BỊ KẸP GIỮA
            elif f" {a_norm} " in f" {db_a_norm} ":
                if len(db_a_norm) < contains_score:
                    contains_score = len(db_a_norm)
                    contains_match = db_a
                    
        if prefix_match: return prefix_match
        if contains_match: return contains_match

        # QUÉT VÒNG 2: XỬ LÝ LỖI TYPO
        best_match = ""
        best_score = 0.0 
        
        for db_a in known_artists:
            db_a_norm = normalize_text_nfd_strip_accents(db_a)
            db_a_norm = re.sub(r'[.?!,\-&_()\[\]]', '', db_a_norm).strip()
            if not db_a_norm: continue
            
            db_nospace = db_a_norm.replace(" ", "")
            db_words = db_a_norm.split()

            # Tối ưu: Bỏ qua sớm nếu độ dài chênh lệch quá vô lý (lớn hơn 40% độ dài tên gốc)
            if len(db_nospace) <= 3 and abs(len(a_nospace) - len(db_nospace)) >= 1:
                continue

            if abs(len(a_nospace) - len(db_nospace)) > max(3, len(db_nospace) * 0.4):
                continue

            # Chiến lược 1: Điểm giống nhau trên chuỗi dính liền (Trị lỗi "seachian" -> "seachains")
            ratio_nospace = difflib.SequenceMatcher(None, a_nospace, db_nospace).ratio()
            
            # Chiến lược 2: Điểm giống nhau trên chuỗi nguyên bản có dấu cách
            ratio_norm = difflib.SequenceMatcher(None, a_norm, db_a_norm).ratio()
            
            # Chiến lược 3: Điểm giống nhau khi bị đảo từ (Trị lỗi "mtp sơn tùng" -> "sơn tùng mtp")
            a_sort = " ".join(sorted(a_words))
            db_sort = " ".join(sorted(db_words))
            ratio_sort = difflib.SequenceMatcher(None, a_sort, db_sort).ratio()
            
            # [MỚI] Chiến lược 4: Fuzzy Prefix (Dành cho nghệ danh dài có hậu tố như Sơn Tùng M-TP)
            ratio_prefix = 0.0
            if len(db_nospace) > len(a_nospace):
                # Cắt phần đầu của tên chuẩn sao cho dài tương đương tên user gõ (chỉ cho phép sai số 1 ký tự)
                prefix_db = db_nospace[:len(a_nospace) + 1]
                ratio_prefix = difflib.SequenceMatcher(None, a_nospace, prefix_db).ratio()

            # Lấy điểm cao nhất trong 4 chiến lược
            score = max(ratio_nospace, ratio_norm, ratio_sort, ratio_prefix)
            
            # NGƯỠNG ĐỘNG (Dynamic Threshold):
            # - Tên ngắn (<= 4 ký tự) rất dễ nhận vơ (VD: "Vũ" dễ nhầm thành "Vy"), cần ngưỡng cao
            # - Tên dài (VD: "seachains" - 9 ký tự) cho phép sai số nhiều hơn.
            threshold = 0.92 if len(a_nospace) <= 4 else 0.75
            
            if score >= threshold and score > best_score:
                best_score = score
                best_match = db_a

        if best_match: return best_match
        
        # Nếu vẫn không tìm thấy và là Intent mạnh (VD: "của ..."), trả về tên user gõ (viết hoa chữ đầu)
        if is_strong_intent and len(a_norm) >= 2 and len(a_words) <= 5:
            return " ".join([w.capitalize() for w in temp_name.split()])
            
        return ""

    # --- REGEX NHẬN DIỆN CƠ BẢN CŨ ---
    if re.search(r'\b(?:bài\s+nào|bài\s+gì|nhạc\s+gì|bài\s+chi|nhạc\s+nào)\b', lower_prompt):
        quick_match_name = None
    else:
        quick_match_name = re.search(r'(?:(?:tìm|tim|mở|mo|mơ|bật|bat|nghe|phát|phat|gợi ý|goi y)(?:\s+(?:cho|giúp|giup|mình|minh|tôi|toi|em|anh|chị|chi|các|cac|vài|vai|mấy|may|một|mot|những|nhung))*\s+|(?:cho(?:\s+(?:tôi|toi|mình|minh|em|anh))?\s+nghe)\s+)?(?:bài hát|bai hat|bài|bai|ca khúc|ca khuc|bản nhạc|ban nhac|đoạn nhạc|doan nhac|nhạc|nhac|playlist)\s+(.+)', lower_prompt)
    
    # [NÂNG CẤP] Bắt mọi biến thể tìm lời: "có bài có câu mà", "tìm bài có câu hát là", "có đoạn lyric như"...
    quick_match_lyric = re.search(r'(?:tìm\s+|tim\s+|cho\s+hỏi\s+)?(?:có\s+|co\s+)?(?:bài\s+|bai\s+)?(?:có\s+|co\s+)?(?:lời|loi|câu|cau|đoạn|doan|chữ|chu|lyrics?)\s+(?:(?:bài\s+hát|bai hat|hát|hat|nhạc|nhac)\s+)?(?:là\s+|như\s+|mà\s+|hát\s+là\s+)?(.+)', lower_prompt)    
    is_analyze = has_file and bool(re.search(r'(phân tích|demo|đánh giá|chấm điểm|hit|viral)', lower_prompt))
    is_audio = has_file and (bool(re.search(r'(bài này|giai điệu|nhận diện|đây là|bài gì|tìm)', lower_prompt)) or len(lower_prompt.split()) <= 8)

    found_pops = [p for p in POPULARITY_KEYWORDS if p in prompt_norm]
    
    clean_attr_prompt = re.sub(r'\b(nhe|nha|nhi|di|nua|thoi|ha|luon)\b', '', prompt_norm)
    found_attrs = []
    sorted_keywords = sorted(ATTRIBUTE_KEYWORDS, key=len, reverse=True)
    temp_prompt = clean_attr_prompt

    art_norm = normalize_text_nfd_strip_accents(artist_val if artist_val else "")

    for k in sorted_keywords:
        if re.search(rf'\b{k}\b', temp_prompt):
            # CHỐT CHẶN: Nếu từ khóa (manh) nằm trong tên ca sĩ không dấu (phan manh quynh) -> BỎ QUA
            if art_norm and k in art_norm:
                continue
            found_attrs.append(k)
            temp_prompt = temp_prompt.replace(k, "")

    is_negated_seed = bool(re.search(r'\b(không|chẳng|chả|đừng)\s+(giống|tựa|kiểu|như|style|tương tự)\b', lower_prompt))
    match_seed = None
    if not is_negated_seed:
        # Tách riêng 'kiểu', buộc phải đi kèm 'bài/ca khúc' để không ăn nhầm vào tính từ (kiểu dằn vặt, kiểu buồn...)
        match_seed = re.search(r'\b(?:giống|tựa|style|tương tự)+(?:\s+(?:như|giống|tựa|với))*\s+(?:bài hát|bài|track|ca khúc)?\s*(.+)', lower_prompt)
        if not match_seed:
            match_seed = re.search(r'\b(?:kiểu)\s+(?:như\s+)?(?:bài hát|bài|track|ca khúc)\s+(.+)', lower_prompt)
    
    
    match_cua = re.search(
        r"\b(?:của|cua|of|do|ca\s+sĩ|ca\s+si|ca\s+sỹ|ca\s+sy|nghệ\s+sĩ|nghe\s+si|nghệ\s+sỹ|nghe\s+sy)\s+(.*?)(?=\s+(?:cực|cuc|gắt|gat|cháy|chay|siêu|sieu|rất|rat|buồn|buon|vui|có|co|hát|hat|ra|mới|moi|với|nhạc|bài|đoạn|lời|câu|chữ|thể\s+loại|the\s+loai|vibe|chủ\s+đề|chu\s+de|topic|về|ve|đang|dang|hot|top|hit|bxh|viral|trending|đình|dinh|làm|lam|nào|nao|hiện\s+tại|hien\s+tai|mà|ma|không|khong|chưa|chua)\b|\s*$|[\.\?!,])",
        lower_prompt, 
    )
    if match_cua:
        artist_val = match_cua.group(1).strip()
        is_strong_artist = True
    else:
        match_nhac = re.search(r'\b(?:nhạc|playlist|nghe)\s+(?!bài|ca khúc|giống|tựa|kiểu|tương tự|style)(.+)', lower_prompt)
        if match_nhac:
            cand = match_nhac.group(1).strip()
            
            # 1. Ép toàn bộ về không dấu ngay từ đầu để quét rác triệt để
            cand_norm = normalize_text_nfd_strip_accents(cand)
            
            cleanup_words = list(MOOD_MAP.keys()) + GENRE_KEYWORDS + POPULARITY_KEYWORDS + ATTRIBUTE_KEYWORDS + ['trẻ', 'tre', 'hiện đại', 'hien dai', 'đi', 'nhé', 'nha', 'với', 'cho tôi', 'thử', 'nhạc', 'bản', 'bài', 'acoustic', 'lofi', 'hát', 'hat']
            
            # 2. Đảm bảo từ khóa quét rác cũng không có dấu để match 100%
            cleanup_words = [normalize_text_nfd_strip_accents(w) for w in cleanup_words]
            cleanup_words = list(dict.fromkeys(cleanup_words)) # Xóa trùng lặp
            
            cleanup_words.sort(key=len, reverse=True)
            for kw in cleanup_words:
                cand_norm = re.sub(rf'\b{kw}\b', '', cand_norm, flags=re.IGNORECASE).strip()
                
            artist_val = re.sub(r'\s+', ' ', cand_norm).strip()
    
    tail_regex = r'(?:\s|^)(cho\s+tôi|nhé|đi|nha|với|luôn|chứ|hiện\s+nay|nhỉ|gì\s+nhiều\s+nhất|đang\s+thịnh\s+hành|quốc\s+dân|cực\s+mạnh|tâm\s+hồn|nhạc\s+trẻ|mới\s+nhất|hay\s+nhất|hay\s+nhat|người\s+ta|nguoi\s+ta|nhiều\s+nhất|nhieu\s+nhat|dạo\s+này|dao\s+nay|quốc\s+dân|quoc\s+dan|hiện\s+nay|hien\s+nay|có\s+bài|co\s+bai|có\s+ca\s+khúc|co\s+ca\s+khuc|hát\s+bài|hat\s+bai|ra\s+bài|ra\s+bai|đang\s+leo\s+chart|dang\s+leo\s+chart).*$'
    junk_list = [
        'nhưng', 'và', 'thì', 'mà', 'là', 'rằng', 'tuy', 'cho', 'của', 'để', 'những', 'các', 'một', 'cái', 
        'lắm', 'quá', 'thật', 'nào', 'gì', 'chi', 'nhỉ', 'hả', 'nữa', 'thôi', 'luôn', 'chứ', 'đi', 'nhé', 
        'nha', 'thử', 'cực mạnh', 'cực gắt', 'cực', 'gắt', 'cháy', 'siêu', 'tâm hồn', 'hiện nay', 'quốc dân','quoc dan', 'hien nay',
        'nhạc trẻ', 'hôm nay','dạo này', 'cuối tuần','xả','lúc','khi','đang','buổi','sáng',
        'người ta', 'nguoi ta', 'người', 'nguoi', 'ai', 'gi', 'nhi', 'nhieu'
        ]
    for junk in junk_list:
        artist_val = re.sub(rf'(?:\s|^){junk}(?:\s|$)', ' ', artist_val, flags=re.IGNORECASE).strip()
        
    artist_val = re.sub(r'\s+', ' ', artist_val).strip()
    artist_val = re.sub(r'[.?!,]', '', artist_val).strip()
    artist_val = _verify_artist(artist_val, is_strong_intent=is_strong_artist)
    
    def _looks_out_of_scope() -> bool:
        pn = str(prompt_norm or "")
        pn = pn.replace('đ', 'd').replace('Đ', 'd')
        hw_device_pat = r"\b(airpods|bluetooth|marshall|micro\b|mic\b|condenser|tai\s+nghe|loa\b|card\s+am\s+thanh|sound\s*card)\b"
        hw_intent_pat = r"\b(mua|tu\s+van|tam\s+\d|trieu|tot\s+nhat|khac\s+phuc|loi\b|bi\s+re|re\s+mot\s+ben|driver\b|tai\s+o\s+dau|mat\s+driver)\b"
        if re.search(hw_device_pat, pn, flags=re.IGNORECASE) and re.search(hw_intent_pat, pn, flags=re.IGNORECASE): return True

        hard_patterns = [r"\b(thoi\s+tiet|nhiet\s+do|mua\s+khong|nang\s+khong)\b", r"\b(nau\s+pho|huong\s+dan\s+.*nau|cong\s+thuc|mon\s+an)\b", r"\b(meo\s+de\s+ngu\s+ngon|ngu\s+ngon\b)\b", r"\b(bot\s+telegram|telegram\b|python\b|lap\s+trinh|code\s+giup|viet\s+code)\b", r"\b(driver\b|card\s+am\s+thanh|sound\s*card)\b", r"\b(chinh\s+sua\s+video|ghep\s+nhac\s+nen|phan\s+mem|ung\s+dung|app\b|dien\s+thoai)\b", r"\b(review\b|bo\s+phim|pitch\s+perfect|lich\s+chieu|chieu\s+rap|rap\s+phim)\b", r"\b(chuong\s+trinh|gameshow|tap\s+toi\s+qua|bi\s+loai)\b", r"\b(tin\s+don|ly\s+do\s+thuc\s+su|scandal|drama|phot\b|ai\s+la\s+nguoi\s+co\s+loi|da\s+co\s+gia\s+dinh|co\s+gia\s+dinh\s+chua)\b", r"\b(nhac\s+phu)\b", r"\b(diep\s+khuc\s+tru\s+luong|tru\s+luong|sep\s+minh)\b", r"\b(than\s+ngheo|ke\s+kho|hang\s+xom)\b"]
        for pat in hard_patterns:
            if re.search(pat, pn, flags=re.IGNORECASE): return True

        if bool(re.search(r"\b(bai\s+hat|ca\s+khuc|playlist|list\s+nhac|loi\s+bai\s+hat|lyrics|vpop|nhac\s*ly|hop\s+am)\b", pn)): return False
        soft_patterns = [r"\b(shop\s+nao\s+ban|ban\s+quan\s+ao|quan\s+ao\s+dep)\b", r"\b(thu\s+do\s+cua|nuoc\s+uc|thanh\s+pho\s+nao)\b", r"\b(poodle|nuoi\s+cho|thu\s+cung)\b", r"\b(bitcoin|ethereum|dau\s+tu)\b", r"\b(trang\s+diem|make\s*up|di\s+tiec)\b", r"\b(lich\s+am|ngay\s+bao\s+nhieu)\b", r"\b(co\s+vua|danh\s+voi\s+minh\s+mot\s+van)\b", r"\b(ke\s+cho\s+minh\s+mot\s+cau\s+chuyen|cau\s+chuyen\s+ma|kinh\s+di)\b", r"\b(tin\s+tuc|tinh\s+hinh\s+the\s+gioi)\b", r"\b(tai\s+sao\s+con\s+nguoi|can\s+phai\s+lam\s+viec)\b"]
        for pat in soft_patterns:
            if re.search(pat, pn, flags=re.IGNORECASE): return True
        return False

    def _has_knowledge_token() -> bool:
        for tok_pattern in KNOWLEDGE_KEYWORDS:
            if re.search(tok_pattern, prompt_norm): return True
        return False

    is_explicit_play_request = bool(
        re.search(r"\b(tim\s+(bai|nhac|doan|ban)|kiem\s+(bai|nhac)|ten\s+bai|co\s+(bai|nhac)\s+nao|bai\s+nao|nhac\s+nao)\b", prompt_norm, flags=re.IGNORECASE)
        or re.search(r"\b(mo|bat|phat)\s+(nhac|bai(\s+hat)?|playlist|list\s+nhac|doan|ban)\b", prompt_norm, flags=re.IGNORECASE)
        or re.search(r"\bnghe\s+(nhac|bai(\s+hat)?|playlist|list\s+nhac|doan|ban)\b", prompt_norm, flags=re.IGNORECASE)
        or re.search(r"\bgoi\s*y(?:\s+(?:cho|giup|minh|toi|em|anh|chi|cac|vai|mot|nhung))*\s+(nhac|bai(\s+hat)?|playlist|list\s+nhac|doan|ban)\b", prompt_norm, flags=re.IGNORECASE)
    )
    if re.search(r"\b(tim\s+hieu|tìm\s+hiểu)\b", lower_prompt, flags=re.IGNORECASE): is_explicit_play_request = False

    is_list_like_knowledge = bool(re.search(r"^(nhung|cac)\s+", prompt_norm) and re.search(r"\b(nghe\s*si|nhac\s*si|nhac\s*cu|dong\s+nhac)\b", prompt_norm, flags=re.IGNORECASE))
    if re.search(r"\b(bai\s+hat|bài\s+hát|playlist|list\s+nhac|list\s+nhạc)\b", lower_prompt, flags=re.IGNORECASE): is_list_like_knowledge = False

    looks_like_knowledge = (_has_knowledge_token() or is_list_like_knowledge) and (not is_explicit_play_request)
    is_negative_action = bool(re.search(r'\b(đừng|không|kh|ko|thôi|trừ|bỏ|chẳng|chả)\s+(mở|bật|nghe|muốn|tìm|thích)\b', lower_prompt))
    # Tìm vị trí bắt đầu của vế yêu cầu mới (Positive Intent)
    # Danh sách các từ khóa đánh dấu sự bắt đầu của yêu cầu mới
    positive_triggers = [
        r'\bbật\b', r'\bbat\b', 
        r'\bmở\b', r'\bmo\b', 
        r'\bphát\b', r'\bphat\b', 
        r'\bnghe\b', 
        r'\btìm\b', r'\btim\b', 
        r'\bcho\b'
    ]

    clean_for_kw = lower_prompt

    if is_negative_action:
        # Tìm xem trong câu có động từ hành động nào xuất hiện sau từ phủ định không
        # Ví dụ: "đừng mở nhạc buồn nữa BẬT bài vui đi"
        matches = list(re.finditer('|'.join(positive_triggers), lower_prompt))
        
        if len(matches) > 0:
            # Lấy vị trí của động từ cuối cùng (thường là vế yêu cầu mới)
            last_match_start = matches[-1].start()
            # Chỉ lấy phần văn bản từ động từ đó trở đi để quét mood
            clean_for_kw = lower_prompt[last_match_start:]
        else:
            # Nếu không có động từ mới, ta xóa vế phủ định đi theo cách cũ
            clean_for_kw = re.sub(r'\b(đừng|không|thôi|bỏ)\s+(mở|bật|nghe|hát).*?(nữa|đi|$)', '', lower_prompt)

    # Sau đó mới quét found_moods trên clean_for_kw
    prefix_pattern = r'(?:\b|nhạc|bài|bật|mở|nghe|tìm|list|cho)'
    suffix_pattern = r'(?:\b|đi|nhé|nha|luôn|chứ|nữa|rồi)'

    found_moods = list(set([
        MOOD_MAP[k] for k in MOOD_MAP 
        if re.search(rf'{prefix_pattern}{k}{suffix_pattern}', clean_for_kw)
    ]))

    found_genres = list(set([
        g for g in GENRE_KEYWORDS 
        if re.search(rf'{prefix_pattern}{g}{suffix_pattern}', clean_for_kw)
    ]))
    
    # --- CẬP NHẬT GIÁ TRỊ TẠI ĐÂY ---
    mood_val = ", ".join(found_moods) if found_moods else ""
    genre_val = ", ".join(found_genres) if found_genres else ""
    active_criteria = [x for x in [mood_val, genre_val, artist_val] if x]
    needs_ai = bool(re.search(r'\b(hợp âm|nhạc lý|khác nhau|nghĩa là|phân biệt|ai hát)\b', lower_prompt)) or looks_like_knowledge
    
    if len(history_context) >= 2 and history_context[-2].get('role') == 'assistant':
        last_bot_msg = str(history_context[-2].get('content', ''))
        if "?" in last_bot_msg or "gợi ý" in last_bot_msg.lower() or "nhé" in last_bot_msg.lower():
            if not is_explicit_play_request and not quick_match_name and not quick_match_lyric and len(active_criteria) == 0: 
                needs_ai = True

    if found_pops and not _has_knowledge_token(): needs_ai = False
        
    is_short_search = False
    if not has_file and not needs_ai and not found_moods and not found_genres and not found_pops:
        words = lower_prompt.split()
        if 2 <= len(words) <= 10 and not re.search(r'\b(chào|hello|hi|tại sao|là gì|thế nào|bao nhiêu|khi nào|ai là|hướng dẫn|cách để|làm sao|đặc điểm|tiểu sử|nguồn gốc|sự nghiệp|phong cách|tầm quan trọng|phân tích|đánh giá|ý nghĩa)\b', lower_prompt):
            is_short_search = True

    is_strict_analyze_intent = (
        bool(re.search(r'\b(tiem nang|kha nang thanh hit|ti le viral|ty le viral|du doan hit|cham diem|xac suat hit|xac suat thanh hit|ban thu|feedback|len top trending|review|danh gia demo|demo|kiem tra ban phoi|kiem tra.*tiem nang|phan tich.*thanh hit|phan tich.*viral|phan tich.*tiem nang|nhac.*phan tich|phan tich.*ban phoi|danh gia.*ban phoi|check.*hit|check.*ban phoi)\b', prompt_norm)) or
        bool(re.search(r'\b(phan tich|danh gia|cham diem|kiem tra|nhan xet|du doan|check|gop y|nghe thu|xem giup)\b.*\b(bai|ban nhac|ca khuc|file|am thanh|doan nhac|doan beat|giai dieu|bai hat|ban phoi|ban thu|track|mix|master)\s+(nay|sau|day|duoi day|cho minh|giup minh|moi)\b', prompt_norm)) or
        bool(re.search(r'\b(bai|ban nhac|ca khuc|file|am thanh|doan nhac|giai dieu|bai hat|ban phoi|ban thu|track|mix|master)\s+(nay|moi|cua minh|cua toi|cua e|cua em)\b.*\b(phan tich|danh gia|cham diem|kiem tra|nhan xet|du doan|check|gop y)\b', prompt_norm)) or
        bool(re.search(r'\b(phan tich|danh gia|cham diem|kiem tra|nhan xet|du doan|check|gop y|nghe thu)\s+(giup|ho|cho minh|gium|nhe|di|thu|voi|xem|dum)\b', prompt_norm)) or
        bool(re.search(r'\b(nho|xin)\s+(ad|admin|bot|vmusic|ai)?\s*(phan tich|danh gia|cham diem|kiem tra|nhan xet|du doan|check|gop y)\b', prompt_norm))
    )

    is_strict_audio_search = (
        bool(re.search(r'\b(bai|nhac|doan|doan beat|doan nhac|giai dieu|beat|file|audio|ca khuc|day)\s+(nay|dang phat|vua roi|day|duoi day)\s*(la\s+)?(bai|nhac|bai hat|ca khuc)?\s*(gi|ten gi|gi vay|gi z|gi v|gi ta|cua ai|nao)\b', prompt_norm)) or
        bool(re.search(r'\b(nhan dien|tim ten|cho hoi ten|cho biet ten|shazam)\s+(bai|nhac|doan|beat|giai dieu|file|am thanh)\s+(nay|dang phat|day|duoi day)\b', prompt_norm)) or
        bool(re.search(r'\b(bai gi day|bai nay ten gi|nhac gi day|day la bai gi|ten bai nay|bai nay la bai gi|bai nay bai gi|tim giup bai nay|bai nay a|bai nay gi vay|bai gi vay|bai gi z|bai gi ta|bai gi the)\b', prompt_norm)) or
        bool(re.search(r'\b(tim.*bang am thanh|tim.*qua am thanh|nhan dien.*am thanh)\b', prompt_norm))
    )
    if found_attrs: is_strict_audio_search = False

    is_greeting = bool(re.match(r'^(chao|hello|hi|alo|helo|hey|e|xin chao|yo)\b', prompt_norm)) and len(prompt_norm.split()) <= 5
    if is_greeting and re.search(r'\b(tim|mo|bat|phat|nghe|nhac|bai|goi y|rcm)\b', prompt_norm): is_greeting = False

    start_pattern = r'^(bat dau|let\'s go|lets go)(?:\s+(thoi|nao|di|luon|ngay))?\s*$'
    is_start = bool(re.fullmatch(start_pattern, prompt_norm))

    is_exact_track_command = bool(re.match(r'^(bat|mo|phat|nghe|tim|play|search)(?:\s+(?:cho|giup|minh|toi|em|anh|chi|a|e|c|cac|vai|mot|nhung))*\s+(bai|nhac|ca khuc|track|doan|ban)\b', prompt_norm))
    if (found_moods or found_attrs or found_genres or found_pops) and not has_file and not is_strict_analyze_intent and not is_strict_audio_search and not is_exact_track_command and not quick_match_lyric and not match_seed and not _looks_out_of_scope() and not needs_ai:

        raw_text = normalize_text_nfd_strip_accents(resolved_input)
        num_match = re.search(r'\b(\d{2,3})\b', raw_text)

        # CHỐT CHẶN 2: Lọc lại attributes một lần nữa cho chắc chắn
        art_norm = normalize_text_nfd_strip_accents(artist_val)
        final_attrs = [a for a in found_attrs if not (art_norm and a.lower() in art_norm)]
        attr_str = ", ".join(final_attrs)

        if num_match:
            val = int(num_match.group(1))
            has_music_kw = bool(re.search(r'\b(bpm|tempo)\b', raw_text, re.IGNORECASE))
            
            if val >= 40 or has_music_kw:
                attr_str += f" {val}"
        intent_data["action"] = "DISCOVER_MUSIC"
        intent_data["params"] = {
            "mood": mood_val,
            "genre": genre_val,
            "artist": artist_val,
            "attributes": attr_str.strip(),
            "popularity_flag": bool(found_pops)
        }
        return intent_data
    
    # Chat chào hỏi / Kiến thức / Ngoài phạm vi
    junk_filter = r'\b(tên|tên\s+là|tìm|tim|mở|mo|bật|bat|nghe|phát|phat|gợi ý|goi y|cho|tôi|toi|mình|minh|xin|một|mot|vài|vai|những|nhung|bạn|ban|có|co|thể|the|không|khong|ko|cần|can|muốn|muon|giúp|giup|hộ|ho|này|nay|kia|đó|do|của|cua|ca sĩ|ca si|nhạc sĩ|nhac si|nghệ sĩ|nghe si|bài hát|bai hat|bài|bai|ca khúc|ca khuc|nhạc|nhac|playlist|list|đi|nhé|nha|với|voi|luôn|luon|chứ|chu|nữa|nua|thử|thu|nào|nao|nhỉ|nhi|hả|ha|vậy|vay|giùm|gium|được|duoc|chưa|chua|thì|thi|vào|vao|đây|day|trong|ngoài|ngoai|do|làm|lam|để|de|ngay|luôn|thịnh\s+hành|thinh\s+hanh|hot|trending|viral|đang|dang)\b'

    if is_greeting or is_start or _looks_out_of_scope() or (looks_like_knowledge and not has_file):
        intent_data["action"] = "GENERAL_CHAT"
    
    elif has_file:
        if is_strict_analyze_intent or bool(re.search(r'\b(phân tích|đánh giá|chấm điểm)\b', lower_prompt)): intent_data["action"] = "ANALYZE_READY"
        else: intent_data["action"] = "SEARCH_AUDIO"
    
    elif is_strict_analyze_intent or is_strict_audio_search:
        if is_strict_analyze_intent: intent_data["action"] = "ANALYZE_READY"
        else: intent_data["action"] = "SEARCH_AUDIO"
    
    elif match_seed and not has_file:
        raw_seed = match_seed.group(1).strip()
        # 1. Gọt sạch rác phần đuôi (nhỉ, nhé, đi...)
        seed_content = re.sub(r'\s+(không|nhỉ|vậy|đi|nha|với|nhất|chứ|nhé|nữa|xem|chatbot).*$', '', raw_seed).strip()
        
        # 2. Gọt sạch rác phần đầu (nhạc của, bài của...) giúp xóa chữ "nhạc của" lọt vào seed_name
        seed_content = re.sub(r'^(nhạc\s+của|nhac\s+cua|bài\s+của|bai\s+cua|ca\s+khúc\s+của|ca\s+khuc\s+cua|bài|bai|nhạc|nhac|ca\s+khúc|ca\s+khuc)\s+', '', seed_content, flags=re.IGNORECASE).strip()
        
        final_seed_name = seed_content
        final_seed_artist = artist_val 

        # 3. [MỚI] Xử lý bóc tách Artist thông minh bên trong Seed
        if " của " in f" {seed_content.lower()} " or " cua " in f" {seed_content.lower()} ":
            parts = re.split(r'\s+(?:của|cua)\s+', seed_content, flags=re.IGNORECASE)
            if len(parts) >= 2:
                potential_artist = parts[-1].strip()
                # CHỐT CHẶN: Nếu vế sau là "ngày hôm qua" hoặc "hiện tại", tuyệt đối KHÔNG coi là ca sĩ
                if not re.search(r'\b(ngay\s+hom\s+qua|hien\s+tai)\b', normalize_text_nfd_strip_accents(potential_artist)):
                    verified_art = _verify_artist(potential_artist, is_strong_intent=True)
                    if verified_art:
                        final_seed_name = " ".join(parts[:-1]).strip()
                        final_seed_artist = verified_art
                    else:
                        # Nếu ko phải ca sĩ thật, trả lại cả cụm cho seed_name và xóa artist rác
                        final_seed_name = seed_content
                        final_seed_artist = ""
                else:
                    # Là từ chỉ thời gian -> Giữ nguyên cả cụm làm tên bài
                    final_seed_name = seed_content
                    final_seed_artist = ""
        
        intent_data["action"] = "DISCOVER_MUSIC"
        intent_data["params"] = {
            "seed_name": final_seed_name, 
            "artist": final_seed_artist,
            "mood": mood_val,
            "genre": genre_val,
            "popularity_flag": bool(found_pops),
            "attributes": ", ".join(found_attrs) if found_attrs else ""
        }
        
    elif quick_match_lyric and not has_file and not needs_ai:
        snippet = quick_match_lyric.group(1).strip()
        snippet = re.sub(r'^(là\s+|chữ\s+|có\s+|co\s+)', '', snippet).strip('"\'')
        extracted_artist = artist_val 
        match_artist = re.search(r'\s+(của|do|feat|ft\.?)\s+(.+)$', snippet, re.IGNORECASE)
        if match_artist and not extracted_artist: extracted_artist = _verify_artist(match_artist.group(2).strip(), True)
        snippet = re.sub(r'\s+(của|do|feat|ft\.?)\s+.*$', '', snippet, flags=re.IGNORECASE).strip()
        intent_data["action"] = "SEARCH_TRACK"
        intent_data["params"] = {"lyric_snippet": snippet, "artist": extracted_artist}
    
    elif quick_match_name and not has_file and not needs_ai:
        raw_query = quick_match_name.group(1).strip()
        raw_query = re.sub(r'^(nhạc\s+buồn|nhạc\s+phim|nhạc\s+chill|bài\s+hát|ca\s+khúc|nhạc)\s+', '', raw_query, flags=re.IGNORECASE).strip()
        raw_query = re.sub(r'\s+(nhạc\s+phim.*|ost.*)$', '', raw_query, flags=re.IGNORECASE).strip()
        raw_query = re.sub(r'[.?!,]+$', '', raw_query).strip()
        raw_query = re.sub(r'\s+(đi|nhé|nha|với|luôn|nào|thử)$', '', raw_query, flags=re.IGNORECASE).strip()
        raw_query = re.sub(r'[.?!,]+$', '', raw_query).strip()
        
        if re.match(r'^(?:của|do|for)\s+', raw_query, re.IGNORECASE):
            s_title = ""
            s_artist = _verify_artist(re.sub(r'^(?:của|do|for)\s+', '', raw_query, flags=re.IGNORECASE).strip(), True)
        else:
            # [SỬA]: Dùng tham lam (.*) để lấy chữ "của" CUỐI CÙNG
            split_match = re.search(r'(.*)\s+(?:của|do|for)\s+(.+)', raw_query, re.IGNORECASE)
            if split_match:
                pot_title = split_match.group(1).strip()
                pot_artist = split_match.group(2).strip()
                
                # CHỐT CHẶN: Không bóc ca sĩ nếu vế sau là đại từ hoặc từ chỉ thời gian
                if pot_artist.lower() in ['em', 'anh', 'tôi', 'toi', 'mình', 'minh', 'bạn', 'ban', 'nó', 'no'] or \
                   re.search(r'\b(ngay\s+hom\s+qua|hien\s+tai|qua\s+khu|tuong\s+lai)\b', normalize_text_nfd_strip_accents(pot_artist)):
                    s_title = raw_query
                    s_artist = ""
                else:
                    verified = _verify_artist(pot_artist, is_strong_intent=True)
                    if verified:
                        s_title = pot_title
                        s_artist = verified
                    else:
                        s_title = raw_query
                        s_artist = ""
            else:
                s_title = raw_query
                s_artist = ""
        check_intent_title = re.sub(r'\b(đi|nhé|nha|với|luôn|nào|thử)\b', '', s_title, flags=re.IGNORECASE)
        check_intent_title = re.sub(r'[.?!,]', '', check_intent_title).strip()
        
        title_core = check_intent_title.lower()
        
        all_trigger_words = list(MOOD_MAP.keys()) + GENRE_KEYWORDS + ATTRIBUTE_KEYWORDS + POPULARITY_KEYWORDS + junk_list
        all_trigger_words.sort(key=len, reverse=True)
        for kw in all_trigger_words:
            title_core = re.sub(rf'\b{kw}\b', '', title_core, flags=re.IGNORECASE).strip()
            
        title_core = re.sub(r'[\W_]+', '', title_core).strip()
        
        check_empty_name = re.sub(junk_filter, '', check_intent_title, flags=re.IGNORECASE)
        check_empty_name = re.sub(r'[\W_]+', '', check_empty_name).strip()
        
        # Bơm đầy đủ tham số để không bị rơi rớt "cực gắt" hay "tâm trạng"
        final_artist = s_artist if s_artist else artist_val
        # Lọc sạch rác: Chỉ lấy attribute nếu nó KHÔNG nằm trong tên ca sĩ (so sánh KHÔNG DẤU)
        art_norm = normalize_text_nfd_strip_accents(final_artist)
        valid_attrs = [a for a in found_attrs if not (art_norm and a.lower() in art_norm)]
        attr_str = ", ".join(valid_attrs) if valid_attrs else ""
        
        params_dm = {
            "artist": final_artist, 
            "mood": mood_val, 
            "genre": genre_val, 
            "attributes": attr_str,
            "popularity_flag": bool(found_pops)
        }

        # Phân luồng
        if not check_empty_name:
            intent_data["action"] = "DISCOVER_MUSIC" if (final_artist or attr_str or len(active_criteria) > 0 or found_pops) else "GENERAL_CHAT"
            intent_data["params"] = params_dm
        elif len(title_core) <= 2 and (len(active_criteria) >= 1 or len(found_attrs) >= 1 or found_pops):
            # Nếu gọt sạch từ khóa (vui vẻ, yêu đời...) mà không còn chữ gì -> Chắc chắn là Gợi ý nhạc
            intent_data["action"] = "DISCOVER_MUSIC"
            intent_data["params"] = params_dm
        elif check_intent_title in GENRE_KEYWORDS or check_intent_title in MOOD_MAP:
            intent_data["action"] = "DISCOVER_MUSIC"
            intent_data["params"] = params_dm
        elif final_artist and not s_title: 
            intent_data["action"] = "DISCOVER_MUSIC"
            intent_data["params"] = params_dm
            
        # --- [CHỐT CHẶN BẢO VỆ MOOD/GENRE] ---
        elif len(active_criteria) >= 1 and not re.search(r'\b(bài|bài hát|ca khúc|bai|bai hat|ca khuc|track)\b', prompt_norm):
            # Nếu KHÔNG gọi đích danh "bài hát" cụ thể, nhưng có Mood/Genre -> Chắc chắn là Gợi ý nhạc
            intent_data["action"] = "DISCOVER_MUSIC"
            intent_data["params"] = params_dm
        # ------------------------------------
        
        else:
            intent_data["action"] = "SEARCH_TRACK"
            intent_data["params"] = {"song_title": s_title, "artist": final_artist}
    
    elif is_short_search:
        split_match = re.search(r'(.*)\s+(?:của|do)\s+(.+)', lower_prompt, re.IGNORECASE)
        if split_match:
            pot_title = split_match.group(1).strip()
            pot_artist = split_match.group(2).strip()
            
            if re.search(r'\b(ngay\s+hom\s+qua|hien\s+tai|qua\s+khu|tuong\s+lai)\b', normalize_text_nfd_strip_accents(pot_artist)):
                s_title = lower_prompt
                s_artist = ""
            else:
                verified = _verify_artist(pot_artist, True)
                if verified:
                    s_title = pot_title
                    s_artist = verified
                else:
                    s_title = lower_prompt
                    s_artist = ""
        else:
            s_title = lower_prompt
            s_artist = ""
        check_title = re.sub(junk_filter, '', s_title, flags=re.IGNORECASE)
        check_empty_short = re.sub(r'[\W_]+', '', check_title).strip()
        if not check_empty_short:
            intent_data["action"] = "DISCOVER_MUSIC" if s_artist else "GENERAL_CHAT"
            intent_data["params"] = {"artist": s_artist} if s_artist else {}
        else:
            clean_s_title = re.sub(r'^(tìm|tim|mở|mo|bật|bat|nghe|phát|phat|gợi ý|goi y|hát|hat)\s+(?:cho\s+tôi\s+|cho\s+mình\s+|tôi\s+|mình\s+)?(?:một\s+|vài\s+|những\s+)?(?:bài hát|bai hat|bài|bai|ca khúc|ca khuc|nhạc|nhac)?\s+', '', s_title, flags=re.IGNORECASE).strip()
            clean_s_title = re.sub(r'\s+(đi|nhé|nha|với|luôn|chứ|nữa|thử|nào|nao|nhỉ|nhihả|ha)$', '', clean_s_title, flags=re.IGNORECASE).strip()
            if not clean_s_title: clean_s_title = s_title
            intent_data["action"] = "SEARCH_TRACK"
            intent_data["params"] = {"song_title": clean_s_title, "artist": s_artist}

    elif len(active_criteria) >= 1 and not needs_ai:
        intent_data["action"] = "DISCOVER_MUSIC"
        intent_data["params"] = {"mood": mood_val, "genre": genre_val, "artist": artist_val}
    else:
        lines = []
        # Lấy 3 tin nhắn gần nhất để làm ngữ cảnh nối tiếp
        for msg in history_context[-3:]:
            c = str(msg.get('content') or '').strip()
            if c: lines.append(f"{msg.get('role', 'user')}: {c[:500] + '…' if len(c) > 500 else c}")
        
        ctx_prompt = "Ngữ cảnh hội thoại gần đây:\n" + "\n".join(lines) + f"\n\nNgười dùng vừa nói: {user_input}" if lines else user_input
        llm_res = call_gemini_intent_api(ctx_prompt, has_file=has_file)
        if isinstance(llm_res, dict): return llm_res

    # Định dạng lại Params trả về
    final_params = _empty_params()
    final_params.update(intent_data.get("params", {}))
    final_params["raw_query"] = user_input
    intent_data["params"] = final_params

    return intent_data