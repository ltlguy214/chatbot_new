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
ALLOWED_ACTIONS: set[str] = {"SEARCH_TRACK", "SEARCH_AUDIO", "DISCOVER_MUSIC", "ANALYZE_READY", "MISSING_FILE", "GENERAL_CHAT"}
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

def resolve_coreference(user_input: str, history_context: list) -> str:
    """Dịch đại từ (ví dụ: anh ấy, bài này) dựa trên ngữ cảnh lịch sử."""
    
    # Bê lại ĐỦ 100% list cũ của bạn + thêm "bài này", "bai nay"
    pronouns_list = [
        'anh ấy', 'anh ay', 'cô ấy', 'co ay', 'người đó', 'nguoi do', 
        'ca sĩ đó', 'ca si do', 'chú ấy', 'chu ay', 'nhóm đó', 'nhom do', 
        'ông ấy', 'ong ay', 'anh y', 'co y', 'ổng', 'bả', 'chả', 
        'bài này', 'bai nay'
    ]
    lower_input = user_input.lower().strip()
    
    # Bắt trúng nếu trong câu có đại từ HOẶC user chỉ gõ đúng 1 từ (ví dụ: "ổng")
    if not any(lower_input.startswith(p + " ") or p in lower_input or lower_input == p for p in pronouns_list):
        return user_input

    # Nếu có đại từ, lục lại lịch sử RAM
    for msg in reversed(history_context):
        if msg.get('role') == 'assistant' and msg.get('track_previews'):
            # Lấy nghệ sĩ từ bài hát được recommend gần nhất
            last_artist = msg['track_previews'][0].get('artist')
            if last_artist:
                print(f"🔄 [RAM CACHE] Dịch đại từ trong: '{user_input}' -> '{last_artist}'")
                # Gắn thêm chữ " của " vào đuôi để kích hoạt cờ is_strong_artist ở Regex bên dưới
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
       - Bài mẫu: `seed_name` (VD: "Tìm bài giống bài Suýt Nữa Thì").
       - Thuộc tính nhạc lý: `attributes` (VD: "nhịp 120 bpm", "nhạc nhanh", "năng lượng thấp", "vừa phải").
       - Top hit / trending: `popularity_flag=true` (có thể kèm `artist`, `genre`).
       - Mood/Genre/Artist (một hoặc nhiều tiêu chí): `mood`, `genre`, `artist`. mood: Trích xuất cảm giác/mục đích (VD: "để ngủ" -> "chill", "buồn").
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
    user_input = normalize_teencode(user_input)
    resolved_input = resolve_coreference(user_input, history_context)
    lower_prompt = resolved_input.lower()
    prompt_norm = normalize_text_nfd_strip_accents(resolved_input)

    intent_data = {"action": "GENERAL_CHAT", "params": _empty_params(), "thought": "", "method": "Rule"}

    # HÀM _verify_artist CHÍNH XÁC NHƯ BẢN CŨ CỦA BẠN
    def _verify_artist(a_name: str, is_strong_intent: bool = False) -> str:
        if not a_name or not known_artists: return ""
        titles_to_strip = [
            'chị đẹp', 'chi dep', 'anh trai', 'ca sĩ', 'ca si', 'nghệ sĩ', 'nghe si',
            'idol', 'thần tượng', 'than tuong', 'diva', 'rapper', 'nhóm nhạc', 'nhom nhac',
            'ban nhạc', 'ban nhac', 'ông hoàng', 'ong hoang', 'nữ hoàng', 'nu hoang',
            'nhạc sĩ', 'nhac si', 'đội', 'team',
            'bản mp3', 'ban mp3', 'mp3', 'audio', 'official', 'lyrics', 'video', 'bản'
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
            'cao', 'thấp', 'thap', 'vừa', 'nhiều'
        }
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
        
        for db_a in known_artists:
            db_a_norm = normalize_text_nfd_strip_accents(db_a)
            db_a_norm = re.sub(r'[.?!,\-&_()\[\]]', '', db_a_norm).strip()
            if not db_a_norm: continue

            db_a_nospace = db_a_norm.replace(" ", "")

            # A. KHỚP CHÍNH XÁC 100% -> Chốt đơn ngay lập tức! (Bảo vệ ca sĩ "Vũ")
            if a_norm == db_a_norm: return db_a
            # B. [MỚI]: KHỚP DÍNH CHỮ (Vd: "sontung" khớp "Sơn Tùng M-TP")
            if a_nospace and db_a_nospace:
                if a_nospace in db_a_nospace and len(a_nospace) >= len(db_a_nospace) * 0.6: return db_a
                elif db_a_nospace in a_nospace and len(db_a_nospace) >= len(a_nospace) * 0.6: return db_a

            # C. NGHỆ SĨ NẰM ĐẦU CÂU (Vd: gõ "Sơn Tùng m-tp remix" -> Bắt "Sơn Tùng M-TP")
            # Chọn tên DB DÀI NHẤT để tránh lấy "Sơn" thay vì "Sơn Tùng"
            if a_norm.startswith(db_a_norm + " "):
                if len(db_a_norm) > prefix_score:
                    prefix_score = len(db_a_norm)
                    prefix_match = db_a
            # D. NGHỆ DANH BỊ KẸP GIỮA (Vd: gõ "MCK" -> Bắt "RPT MCK")
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
            
            # Lấy điểm cao nhất trong 3 chiến lược
            score = max(ratio_nospace, ratio_norm, ratio_sort)
            
            # NGƯỠNG ĐỘNG (Dynamic Threshold):
            # - Tên ngắn (<= 4 ký tự) rất dễ nhận vơ (VD: "Vũ" dễ nhầm thành "Vy"), cần ngưỡng cao
            # - Tên dài (VD: "seachains" - 9 ký tự) cho phép sai số nhiều hơn.
            threshold = 0.85 if len(a_nospace) <= 4 else 0.75
            
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
        quick_match_name = re.match(r'^(?:(?:tìm|tim|mở|mo|bật|bat|nghe|phát|phat|cho(?: tôi)? nghe)\s+)?(?:những\s+)?(?:bài hát|bai hat|bài|bai|ca khúc|ca khuc)\s+(.+)', lower_prompt)
    
    quick_match_lyric = re.search(r'(?:có\s+|co\s+)?(?:lời|loi|câu|cau|đoạn|doan|chữ|chu|lyrics?)\s+(?:(?:bài\s+hát|bai hat|hát|hat|nhạc|nhac)\s+)?(?:là\s+|như\s+)?(.+)', lower_prompt)
    
    is_analyze = has_file and bool(re.search(r'(phân tích|demo|đánh giá|chấm điểm|hit|viral)', lower_prompt))
    is_audio = has_file and (bool(re.search(r'(bài này|giai điệu|nhận diện|đây là|bài gì|tìm)', lower_prompt)) or len(lower_prompt.split()) <= 8)

    found_moods = list(dict.fromkeys([MOOD_MAP[k] for k in MOOD_MAP if re.search(rf'\b{k}\b', lower_prompt)]))
    found_genres = list(dict.fromkeys([g for g in GENRE_KEYWORDS if re.search(rf'\b{g}\b', lower_prompt)]))
    found_pops = [p for p in POPULARITY_KEYWORDS if p in prompt_norm]
    
    found_attrs = []
    sorted_keywords = sorted(ATTRIBUTE_KEYWORDS, key=len, reverse=True)
    temp_prompt = prompt_norm
    for k in sorted_keywords:
        if re.search(rf'\b{k}\b', temp_prompt):
            found_attrs.append(k)
            temp_prompt = temp_prompt.replace(k, "")
    found_attrs = [k for k in ATTRIBUTE_KEYWORDS if re.search(rf'\b{k}\b', prompt_norm)]

    safe_no_bound = ['quẩy', 'party', 'chill', 'tết', 'xuân', 'lofi']
    for kw in safe_no_bound:
        if kw in lower_prompt:
            if kw in MOOD_MAP and MOOD_MAP[kw] not in found_moods: found_moods.append(MOOD_MAP[kw])
            if kw in GENRE_KEYWORDS and kw not in found_genres: found_genres.append(kw)

    is_negated_seed = bool(re.search(r'\b(không|chẳng|chả|đừng)\s+(giống|tựa|kiểu|như|style|tương tự)\b', lower_prompt))
    match_seed = None
    if not is_negated_seed:
        match_seed = re.search(r'\b(?:giống|tựa|style|kiểu|tương tự)+(?:\s+(?:như|giống|tựa|với))*\s+(?:bài hát|bài|track|ca khúc)?\s*(.+)', lower_prompt)
    
    artist_val = ""
    is_strong_artist = False 
    
    match_cua = re.search(
        r"\b(?:của|of|do|ca sĩ|ca si|nghệ sĩ|nghe si)\s+(.*?)(?=\s+(?:có|co|với|nhạc|bài|đoạn|lời|câu|chữ|thể\s+loại|the\s+loai|vibe|chủ\s+đề|chu\s+de|topic|về|ve|đang|dang|hot|top|hit|bxh|viral|trending|đình|dinh|làm|lam|nào|nao)\b|\s*$|[\.\?!,])",
        lower_prompt, 
    )
    if match_cua:
        artist_val = match_cua.group(1).strip()
        is_strong_artist = True
    else:
        match_nhac = re.search(r'\b(?:nhạc|playlist|nghe)\s+(?!bài|ca khúc)(.+)', lower_prompt)
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
    
    artist_val = re.sub(r'\s+(cho\s+tôi|nhé|đi|nha|với|luôn|chứ|hiện\s+nay|nhỉ|gì\s+nhiều\s+nhất|đang\s+thịnh\s+hành|quốc\s+dân|cực\s+mạnh|tâm\s+hồn|nhạc\s+trẻ|mới\s+nhất|hay\s+nhất).*$', '', artist_val, flags=re.IGNORECASE).strip()
    junk_list = ['nhưng', 'và', 'thì', 'mà', 'là', 'rằng', 'tuy', 'cho', 'của', 'để', 'những', 'các', 'một', 'cái', 'lắm', 'quá', 'thật', 'nào', 'gì', 'chi', 'nhỉ', 'hả', 'nữa', 'thôi', 'luôn', 'chứ', 'đi', 'nhé', 'nha', 'thử', 'cực mạnh', 'tâm hồn', 'hiện nay', 'quốc dân', 'nhạc trẻ', 'hôm nay', 'dạo này', 'cuối tuần', 'xả', 'lúc', 'khi', 'đang', 'buổi', 'sáng', 'trưa', 'tối']
    for junk in junk_list:
        artist_val = re.sub(rf'(?:\s|^){junk}(?:\s|$)', ' ', artist_val, flags=re.IGNORECASE).strip()
        
    artist_val = re.sub(r'\s+', ' ', artist_val).strip()
    artist_val = re.sub(r'[.?!,]', '', artist_val).strip()
    artist_val = _verify_artist(artist_val, is_strong_intent=is_strong_artist)
    
    mood_val = ", ".join(found_moods) if found_moods else ""
    genre_val = ", ".join(found_genres) if found_genres else ""
    active_criteria = [x for x in [mood_val, genre_val, artist_val] if x]

    # [PHẦN MỚI CHÈN LẠI]: Kiểm tra bot vừa hỏi lại người dùng (Dấu hiệu follow-up)
    needs_ai = False
    if len(history_context) >= 2 and history_context[-2].get('role') == 'assistant':
        last_bot_msg = str(history_context[-2].get('content', ''))
        if "?" in last_bot_msg or "gợi ý" in last_bot_msg.lower() or "nhé" in last_bot_msg.lower():
            is_explicit = bool(re.search(r"\b(tim|mo|bat|phat|nghe)\s+(nhac|bai|playlist)\b", prompt_norm))
            if not is_explicit: needs_ai = True

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
        re.search(r"\b(tim\s+(bai|nhac)|ten\s+bai)\b", prompt_norm, flags=re.IGNORECASE)
        or re.search(r"\b(mo|bat|phat)\s+(nhac|bai(\s+hat)?|playlist|list\s+nhac)\b", prompt_norm, flags=re.IGNORECASE)
        or re.search(r"\bnghe\s+(nhac|bai(\s+hat)?|playlist|list\s+nhac)\b", prompt_norm, flags=re.IGNORECASE)
        or re.search(r"\bgoi\s*y(?:\s+(?:cho|giup|minh|cac|vai|mot|nhung))*\s+(nhac|bai(\s+hat)?|playlist|list\s+nhac)\b", prompt_norm, flags=re.IGNORECASE)
    )
    if re.search(r"\b(tim\s+hieu|tìm\s+hiểu)\b", lower_prompt, flags=re.IGNORECASE): is_explicit_play_request = False

    is_list_like_knowledge = bool(re.search(r"^(nhung|cac)\s+", prompt_norm) and re.search(r"\b(nghe\s*si|nhac\s*si|nhac\s*cu|dong\s+nhac)\b", prompt_norm, flags=re.IGNORECASE))
    if re.search(r"\b(bai\s+hat|bài\s+hát|playlist|list\s+nhac|list\s+nhạc)\b", lower_prompt, flags=re.IGNORECASE): is_list_like_knowledge = False

    looks_like_knowledge = (_has_knowledge_token() or is_list_like_knowledge) and (not is_explicit_play_request)
    is_negative_action = bool(re.search(r'\b(đừng|không|kh|ko|thôi|trừ|bỏ|chẳng|chả)\s+(mở|bật|nghe|muốn|tìm|thích)\b', lower_prompt))
    needs_ai = bool(re.search(r'\b(hợp âm|nhạc lý|khác nhau|nghĩa là|phân biệt|ai hát)\b', lower_prompt)) or looks_like_knowledge or is_negative_action
    
    if len(history_context) >= 2 and history_context[-2].get('role') == 'assistant':
        last_bot_msg = str(history_context[-2].get('content', ''))
        if "?" in last_bot_msg or "gợi ý" in last_bot_msg.lower() or "nhé" in last_bot_msg.lower():
            if not is_explicit_play_request: needs_ai = True

    if found_pops and not _has_knowledge_token(): needs_ai = False
        
    is_short_search = False
    if not has_file and not needs_ai and not found_moods and not found_genres and not found_pops:
        words = lower_prompt.split()
        if 2 <= len(words) <= 10 and not re.search(r'\b(chào|hello|hi|tại sao|là gì|thế nào|bao nhiêu|khi nào|ai là|hướng dẫn|cách để|làm sao|đặc điểm|tiểu sử|nguồn gốc|sự nghiệp|phong cách|tầm quan trọng|phân tích|đánh giá|ý nghĩa)\b', lower_prompt):
            is_short_search = True

    is_strict_analyze_intent = (
        bool(re.search(r'\b(tiem nang|kha nang thanh hit|ti le viral|ty le viral|du doan hit|cham diem|xác suất hit|xác suất thanh hit|ban thu|feedback|len top trending|review|danh gia demo|demo|kiem tra ban phoi|kiem tra.*tiem nang|phan tich.*thanh hit|phan tich.*viral|phan tich.*tiem nang|nhac.*phan tich|phan tich.*ban phoi)\b', prompt_norm)) or
        bool(re.search(r'\b(phan tich|danh gia|cham diem|kiem tra|nhan xet|du doan)\b.*\b(bai|ban nhac|ca khuc|file|am thanh|doan nhac|doan beat|giai dieu|bai hat)\s+(nay|sau|day|duoi day)\b', prompt_norm))
    )
    is_strict_audio_search = (
        bool(re.search(r'^(bai|nhac|doan|doan beat|doan nhac|giai dieu|beat|file|audio|ca khuc|day)\s+(nay|dang phat|vua roi|day|duoi day)?\s*(la\s+)?(bai|nhac|bai hat|ca khuc)?\s*(gi|ten gi|gi vay|gi z|gi v|gi ta|cua ai|nao)\b', prompt_norm)) or
        bool(re.search(r'\b(nhan dien|tim ten|cho hoi ten|cho biet ten|shazam)\s+(bai|nhac|doan|beat|giai dieu|file|am thanh)\s+(nay|dang phat|day|duoi day)\b', prompt_norm)) or
        bool(re.search(r'^(bai gi day|bai nay ten gi|nhac gi day|day la bai gi|ten bai nay|bai nay la bai gi|bai nay bai gi|tim giup bai nay|bai nay a|bai nay gi vay)\b', prompt_norm)) or
        bool(re.search(r'\b(tim.*bang am thanh|tim.*qua am thanh|nhan dien.*am thanh)\b', prompt_norm))
    )
    if found_attrs: is_strict_audio_search = False

    is_greeting = bool(re.match(r'^(chao|hello|hi|alo|helo|hey|e|xin chao|yo)\b', prompt_norm)) and len(prompt_norm.split()) <= 5
    if is_greeting and re.search(r'\b(tim|mo|bat|phat|nghe|nhac|bai|goi y)\b', prompt_norm): is_greeting = False

    start_pattern = r'^(bat dau|let\'s go|lets go)(?:\s+(thoi|nao|di|luon|ngay))?\s*$'
    is_start = bool(re.fullmatch(start_pattern, prompt_norm))

    if (found_moods or found_attrs or found_genres) and not has_file:
    
        raw_text = normalize_text_nfd_strip_accents(resolved_input)
        num_match = re.search(r'\b(\d{2,3})\b', raw_text)

        attr_str = ", ".join(found_attrs)

        if num_match:
            attr_str += f" {num_match.group(1)}"

        intent_data["action"] = "DISCOVER_MUSIC"
        intent_data["params"] = {
            "mood": mood_val,
            "genre": genre_val,
            "artist": artist_val,
            "attributes": attr_str.strip()
        }
        return intent_data

    junk_filter = r'\b(tìm|tim|mở|mo|bật|bat|nghe|phát|phat|gợi ý|goi y|cho|tôi|toi|mình|minh|xin|một|mot|vài|vai|những|nhung|bạn|ban|có|co|thể|the|không|khong|ko|cần|can|muốn|muon|giúp|giup|hộ|ho|này|nay|kia|đó|do|của|cua|ca sĩ|ca si|nhạc sĩ|nhac si|nghệ sĩ|nghe si|bài hát|bai hat|bài|bai|ca khúc|ca khuc|nhạc|nhac|playlist|list|đi|nhé|nha|với|voi|luôn|luon|chứ|chu|nữa|nua|thử|thu|nào|nao|nhỉ|nhi|hả|ha|vậy|vay|giùm|gium|được|duoc|chưa|chua|thì|thi|vào|vao|đây|day|trong|ngoài|ngoai|do|làm|lam|để|de|ngay|luôn)\b'


    if is_greeting or is_start or _looks_out_of_scope() or (looks_like_knowledge and not has_file):
        intent_data["action"] = "GENERAL_CHAT"
    
    elif has_file:
        if is_strict_analyze_intent or bool(re.search(r'\b(phân tích|đánh giá|chấm điểm)\b', lower_prompt)): intent_data["action"] = "ANALYZE_READY"
        else: intent_data["action"] = "SEARCH_AUDIO"
    
    elif is_strict_analyze_intent or is_strict_audio_search:
        if is_strict_analyze_intent: intent_data["action"] = "ANALYZE_READY"
        else: intent_data["action"] = "SEARCH_AUDIO"
    
    elif match_seed and not has_file:
        seed_name = re.sub(r'\s+(không|nhỉ|vậy|đi|nha|với|nhất|chứ|nhé|nữa|xem|chatbot).*$', '', match_seed.group(1).strip()).strip()
        intent_data["action"] = "DISCOVER_MUSIC"
        intent_data["params"] = {"seed_name": seed_name, "artist": artist_val}
    
    elif len(found_attrs) >= 1 and not has_file and not needs_ai and not (
        quick_match_name and len(
            re.sub(
                r'\d+', '',
                re.sub(
                    r'\b(' + '|'.join(
                        found_attrs + found_moods + found_genres + found_pops + [
                            'có','co','không','khong','bài','bai','nhạc','nhac',
                            'và','va','nào','nao','cho','tôi','toi','những','nhung',
                            'một','mot','các','cac','cái','cai'
                        ]
                    ) + r')\b',
                    '',
                    normalize_text_nfd_strip_accents(quick_match_name.group(1))
                )
            ).replace(" ", "")
        ) > 5
    ):

        raw_text = normalize_text_nfd_strip_accents(resolved_input)
        num_match = re.search(r'\b(\d{2,3})\b', raw_text)

        attr_str = ", ".join(found_attrs)

        if num_match:
            attr_str += f" {num_match.group(1)}"

        params_dm = {
            "attributes": attr_str.strip(),
            "song_title": "",
            "artist": artist_val,
            "mood": mood_val,
            "genre": genre_val
        }

        if found_pops:
            params_dm["popularity_flag"] = True

        intent_data["action"] = "DISCOVER_MUSIC"
        intent_data["params"] = params_dm

    elif found_pops and not has_file and not needs_ai:
        is_safe_to_route = True
        if quick_match_name:
            title_check = normalize_text_nfd_strip_accents(quick_match_name.group(1))
            for k in found_pops: title_check = re.sub(rf'\b{k}\b', '', title_check).strip()
            title_check = re.sub(r'\b(vay|nhi|da|do|nha|nhe|di|thoi|a|nao|the|vay\?|nhi\?|the\?|nao\?|dang|ca|khuc)\b\s*\??', '', title_check).strip()
            title_check = re.sub(r'[.?!,]+$', '', title_check).strip()
            if len(title_check) > 5 and not artist_val: is_safe_to_route = False
        if is_safe_to_route:
            intent_data["action"] = "DISCOVER_MUSIC"
            intent_data["params"] = {"artist": artist_val, "genre": genre_val, "mood": mood_val, "popularity_flag": True}
        else:
            s_title = quick_match_name.group(1).strip() if quick_match_name else str(user_input)
            intent_data["action"] = "SEARCH_TRACK"
            intent_data["params"] = {"song_title": s_title, "artist": artist_val}
    
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
            split_match = re.search(r'(.*?)\s+(?:của|do|for)\s+(.+)', raw_query)
            if split_match:
                s_title = split_match.group(1).strip()
                s_artist = _verify_artist(split_match.group(2).strip(), True)
            else:
                s_title = raw_query
                s_artist = ""
        check_intent_title = re.sub(r'\b(đi|nhé|nha|với|luôn|nào|thử)\b', '', s_title, flags=re.IGNORECASE)
        check_intent_title = re.sub(r'[.?!,]', '', check_intent_title).strip()
        title_core = check_intent_title
        for kw in found_moods + found_genres: title_core = re.sub(rf'\b{kw}\b', '', title_core, flags=re.IGNORECASE).strip()
        check_empty_name = re.sub(junk_filter, '', check_intent_title, flags=re.IGNORECASE)
        check_empty_name = re.sub(r'[\W_]+', '', check_empty_name).strip()
        if not check_empty_name:
            intent_data["action"] = "DISCOVER_MUSIC" if s_artist else "GENERAL_CHAT"
            intent_data["params"] = {"artist": s_artist} if s_artist else {}
        elif len(title_core) < 2 and len(active_criteria) >= 1:
            intent_data["action"] = "DISCOVER_MUSIC"
            intent_data["params"] = {"mood": mood_val, "genre": genre_val, "artist": artist_val}
        elif check_intent_title in GENRE_KEYWORDS:
            intent_data["action"] = "DISCOVER_MUSIC"
            intent_data["params"] = {"genre": check_intent_title}
        elif check_intent_title in MOOD_MAP:
            intent_data["action"] = "DISCOVER_MUSIC"
            intent_data["params"] = {"mood": MOOD_MAP[check_intent_title]}
        elif s_artist and not s_title: 
            intent_data["action"] = "DISCOVER_MUSIC"
            intent_data["params"] = {"artist": s_artist}
        else:
            intent_data["action"] = "SEARCH_TRACK"
            intent_data["params"] = {"song_title": s_title, "artist": s_artist}
    elif is_short_search:
        split_match = re.search(r'(.*?)\s+(?:của|do)\s+(.+)', lower_prompt)
        if split_match:
            s_title = split_match.group(1).strip()
            s_artist = _verify_artist(split_match.group(2).strip(), True)
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
    intent_data["params"] = final_params

    return intent_data