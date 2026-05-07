import re
import unicodedata

import difflib
try:
    from chatbot.dictionaries import TEENCODE_MAP
except ModuleNotFoundError:
    from dictionaries import TEENCODE_MAP

# Chuẩn hóa tất cả các key trong từ điển về NFC 1 lần duy nhất lúc load file
NORMALIZED_TEENCODE_MAP = {
    unicodedata.normalize('NFC', k): v 
    for k, v in TEENCODE_MAP.items()
}

# 2. SẮP XẾP keys theo độ dài giảm dần. 
# Việc này cực kỳ quan trọng để thay thế các cụm từ dài (vd: "top top") 
# trước khi nó bị thay thế nhầm bởi từ ngắn ("top").
SORTED_TEENCODE_KEYS = sorted(NORMALIZED_TEENCODE_MAP.keys(), key=len, reverse=True)


def normalize_teencode(text: str) -> str:
    """Chuẩn hóa teencode, từ viết tắt thành tiếng Việt chuẩn."""
    if not text:
        return ""
    
    # 1. Ép chuẩn NFC đầu vào để trị dứt điểm đứt gãy Unicode
    text = unicodedata.normalize('NFC', text)
    
    # 2. Dọn dẹp khoảng trắng thừa
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 3. Quét qua từ điển đã sắp xếp để đè regex
    for k in SORTED_TEENCODE_KEYS:
        v = NORMALIZED_TEENCODE_MAP[k]
        
        # Hàm con để cố gắng giữ nguyên Viết Hoa/Viết Thường của user
        def repl(match):
            word = match.group(0)
            if word.isupper(): return v.upper()
            if word.istitle(): return v.capitalize()
            return v

        # Dùng (?<!\w) và (?!\w) thay vì \w+ hay \b.
        # Cách này bắt chính xác 100% các cụm từ có dấu cách ("top top") 
        # hoặc ký tự đặc biệt ("gii?") mà không bị dính vào giữa từ khác.
        pattern = rf'(?<!\w){re.escape(k)}(?!\w)'
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)

    return text
def fuzzy_phrase(text: str, vocab: list[str], threshold: float = 0.85) -> str:
    STOP_WORDS = {"va", "la", "di", "nhe", "nha", "roi"}

    tokens = text.split()
    new_tokens = []

    for t in tokens:
        if t in STOP_WORDS:
            new_tokens.append(t)
            continue

        match = difflib.get_close_matches(t, vocab, n=1, cutoff=threshold)
        new_tokens.append(match[0] if match else t)

    return " ".join(new_tokens)

def normalize_text_nfc(text: str) -> str:
    """Chuẩn hóa NFC để hiển thị và lưu trữ chính xác tiếng Việt có dấu."""
    if not text:
        return ""
    text = unicodedata.normalize('NFC', text.lower())
    return re.sub(r'\s+', ' ', text).strip()

def normalize_text_nfd_strip_accents(text: str, apply_fuzzy: bool = False, vocab: list[str] = None) -> str:
    if not text:
        return ""
    
    text = str(text)
    text = unicodedata.normalize('NFKD', text)
    text = ''.join(ch for ch in text if not unicodedata.combining(ch))
    text = text.replace('đ', 'd').replace('Đ', 'D')
    
    text = text.lower().strip()
    
    # 🔥 APPLY FUZZY Ở ĐÂY
    if apply_fuzzy and vocab:
        text = fuzzy_phrase(text, vocab)
    
    return text


def map_emotion_to_mood(emotion_text: str) -> str:
    """Map nhan cam xuc sang mood noi bo de query."""
    if not emotion_text:
        return 'neutral'
    normalized = normalize_text_nfd_strip_accents(emotion_text)
    if normalized in {'tich cuc', 'positive'}:
        return 'energetic'
    if normalized in {'tieu cuc', 'negative'}:
        return 'sad'
    return 'neutral'

def normalize_mood_token(raw_mood: str) -> str:
    """Normalize Vietnamese/English mood strings into internal query labels."""
    mood = normalize_text_nfd_strip_accents(raw_mood)
    if not mood:
        return 'neutral'
    sad_tokens = {'buon', 'sad', 'suy', 'tieu cuc', 'negative', 'melancholy', 'depressed', 'that tinh', 'broken', 'heartbreak'}
    energetic_tokens = {'vui', 'happy', 'quay', 'quay tung', 'quay het minh', 'energetic', 'dance', 'party', 'soi dong', 'tich cuc', 'positive'}
    chill_tokens = {'chill', 'thu gian', 'relax', 'binh yen', 'healing', 'acoustic', 'lofi', 'lo-fi'}

    if any(t in mood for t in sad_tokens): return 'sad'
    if any(t in mood for t in energetic_tokens): return 'energetic'
    if any(t in mood for t in chill_tokens): return 'neutral'
    return 'neutral'