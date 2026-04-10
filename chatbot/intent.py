import json
import re
import os
import unicodedata
import time

try:
    from chatbot.env import load_env
except ModuleNotFoundError:
    from env import load_env

# Load .env robustly (works across different CWDs).
load_env()

_INTENT_MODEL = "gemini-2.0-flash"

# 11 ACTIONS:
ALLOWED_ACTIONS: set[str] = {
    "SEARCH_NAME",
    "SEARCH_LYRIC",
    "SEARCH_AUDIO",
    "RECOMMEND_MOOD",
    "RECOMMEND_ARTIST",
    "RECOMMEND_GENRE",
    "ANALYZE_READY",
    "MISSING_FILE",
    "CLARIFY",
    "MUSIC_KNOWLEDGE",
    "OUT_OF_SCOPE",
}


_GEMINI_QUOTA_COOLDOWN_UNTIL: float = 0.0


def _parse_gemini_api_keys() -> list[str]:
    keys: list[str] = []

    raw = str(os.getenv("GEMINI_API_KEYS") or "").strip()
    if raw:
        for part in re.split(r"[\n,;]+", raw):
            part = part.strip()
            if part and part not in keys:
                keys.append(part)

    single = str(os.getenv("GEMINI_API_KEY") or "").strip()
    if single and single not in keys:
        keys.append(single)

    # Optional numbered keys for rotation.
    for i in range(1, 21):
        v = str(os.getenv(f"GEMINI_API_KEY_{i}") or "").strip()
        if v and v not in keys:
            keys.append(v)

    return keys


def _intent_model_from_env() -> str:
    # Enforce a single model version as requested.
    return _INTENT_MODEL


def _looks_like_quota_error(err: Exception) -> bool:
    msg = str(err).lower()
    return (
        "resource_exhausted" in msg
        or "quota" in msg
        or "429" in msg
        or "rate limit" in msg
        or "rate-limit" in msg
    )


def _short_error_for_ui(err: Exception | None) -> str:
    if err is None:
        return ""
    try:
        msg = str(err)
        lowered = msg.lower()
        if "cannot import name 'genai' from 'google'" in lowered or "cannot import name \"genai\" from \"google\"" in lowered:
            return "Thiếu google-genai hoặc đang chạy nhầm Python env (hãy chạy Streamlit bằng .venv312)"
        # Redact anything that looks like a Google API key.
        msg = re.sub(r"AIza[0-9A-Za-z\-_]{20,}", "[REDACTED]", msg)
        msg = re.sub(r"\s+", " ", msg).strip()
        return msg[:180]
    except Exception:
        return ""


def _empty_params() -> dict:
    return {
        "song_title": "",
        "artist": "",
        "mood": "",
        "genre": "",
        "lyric_snippet": "",
    }

def normalize(text):
    text = text.lower()
    text = unicodedata.normalize('NFC', text)
    return text

def parse_intent_llm(user_input, has_file=False):
    """
    Phân tích ý định người dùng thành 11 Action chuẩn.
    """
    file_context = "ĐÃ TẢI LÊN" if has_file else "CHƯA TẢI LÊN"
    
    prompt = f"""
    Bạn là AI điều phối cho hệ thống âm nhạc V-Pop. Hãy phân tích câu lệnh sau: "{user_input}"
    Ngữ cảnh: Người dùng {file_context} file nhạc.

    HÃY CHỌN 1 ACTION DUY NHẤT TRONG 11 TRƯỜNG HỢP SAU:

    --- NHÓM LISTENER (NGƯỜI NGHE) ---
    1. SEARCH_NAME: Tìm đích danh tên bài hát/ca sĩ.
    2. SEARCH_LYRIC: Tìm bài hát qua một đoạn lời nhạc.
    3. SEARCH_AUDIO: Tìm bài hát tương đồng bằng giai điệu (yêu cầu có file).
    4. RECOMMEND_MOOD: Gợi ý nhạc theo tâm trạng (vui, buồn, chill, quẩy).
    5. RECOMMEND_ARTIST: Gợi ý nhạc tương tự một ca sĩ nào đó.
    6. RECOMMEND_GENRE: Gợi ý theo thể loại (Rap, Pop, Ballad).

    --- NHÓM PRODUCER (NHÀ SẢN XUẤT) ---
    7. ANALYZE_READY: Phân tích bài hát, dự đoán hit (KHI ĐÃ CÓ FILE).
    8. MISSING_FILE: Muốn phân tích bài hát NHƯNG CHƯA TẢI FILE.

    --- NHÓM HỆ THỐNG ---
    9. CLARIFY: Câu hỏi mập mờ, cần hỏi lại để rõ ý.
    10. MUSIC_KNOWLEDGE: Hỏi kiến thức nhạc lý, lịch sử âm nhạc chung.
    11. OUT_OF_SCOPE: Hỏi chuyện không liên quan đến âm nhạc.

    CHỈ TRẢ VỀ JSON:
    {{
        "action": "TÊN_ACTION",
        "params": {{
            "song_title": "",
            "artist": "",
            "mood": "",
            "genre": "",
            "lyric_snippet": ""
        }},
        "thought": ""
    }}
    """
    
    try:
        global _GEMINI_QUOTA_COOLDOWN_UNTIL
        now = time.time()
        if _GEMINI_QUOTA_COOLDOWN_UNTIL and now < _GEMINI_QUOTA_COOLDOWN_UNTIL:
            remaining = int(_GEMINI_QUOTA_COOLDOWN_UNTIL - now)
            return _heuristic_intent(
                user_input,
                has_file=has_file,
                thought=f"Gemini đang quá quota (cooldown {remaining}s), mình dùng phân loại nhanh.",
            )

        keys = _parse_gemini_api_keys()
        if not keys:
            return {
                "action": "CLARIFY",
                "params": _empty_params(),
                "thought": "Thiếu cấu hình Gemini (GEMINI_API_KEY / GEMINI_API_KEYS).",
            }

        model_name = _intent_model_from_env()
        last_err: Exception | None = None
        response_text: str | None = None

        for api_key in keys:
            try:
                from google import genai

                client = genai.Client(api_key=api_key)
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                )
                response_text = getattr(response, "text", None)
                if response_text:
                    break
            except Exception as e:
                last_err = e
                if _looks_like_quota_error(e):
                    # Rotate key on quota/rate-limit.
                    _GEMINI_QUOTA_COOLDOWN_UNTIL = time.time() + 10 * 60
                    continue
                # Non-quota errors: still try next key (can be invalid key / region).
                continue

        if not response_text:
            # Avoid leaking raw quota/stack traces to the UI.
            if last_err and _looks_like_quota_error(last_err):
                _GEMINI_QUOTA_COOLDOWN_UNTIL = time.time() + 10 * 60
                detail = _short_error_for_ui(last_err)
                suffix = f" ({detail})" if detail else ""
                return _heuristic_intent(user_input, has_file=has_file, thought=f"Gemini đang quá tải/quá quota{suffix}, mình dùng phân loại nhanh.")
            detail = _short_error_for_ui(last_err)
            suffix = f" ({type(last_err).__name__}: {detail})" if detail else f" ({type(last_err).__name__})" if last_err else ""
            return _heuristic_intent(user_input, has_file=has_file, thought=f"Không gọi được Gemini{suffix}, mình dùng phân loại nhanh.")
        
        # Lọc lấy JSON an toàn
        match = re.search(r"\{.*\}", response_text, re.DOTALL)
        
        if match:
            clean_json_text = match.group() 
            intent_data = json.loads(clean_json_text)

            # ===== VALIDATE PARAMS =====
            required_fields = ["song_title", "artist", "mood", "genre", "lyric_snippet"]

            if "params" not in intent_data:
                intent_data["params"] = {}

            for field in required_fields:
                if field not in intent_data["params"]:
                    intent_data["params"][field] = ""

            # normalize action
            intent_data["action"] = str(intent_data.get("action") or "").upper().strip()
            if intent_data["action"] not in ALLOWED_ACTIONS:
                intent_data["action"] = "CLARIFY"
                intent_data["thought"] = str(intent_data.get("thought") or "") or "Action không hợp lệ"
            # ===== END VALIDATE =====
            
            # ===== HANDLE FILE LOGIC =====
            if intent_data["action"] == "SEARCH_AUDIO" and not has_file:
                return {
                    "action": "MISSING_FILE",
                    "params": _empty_params(),
                    "thought": "Bạn chưa tải file âm thanh lên!"
                }

            if intent_data["action"] == "ANALYZE_READY" and not has_file:
                return {
                    "action": "MISSING_FILE",
                    "params": _empty_params(),
                    "thought": "Bạn chưa tải file để phân tích!"
                }
            # ===== END =====
            return intent_data
        else:
            return {
                    "action": "CLARIFY",
                    "params": _empty_params(),
                    "thought": "AI không trả về JSON chuẩn"
                }
            
    except Exception as e:
        return _heuristic_intent(user_input, has_file=has_file, thought="Có lỗi khi phân tích ý định, mình dùng phân loại nhanh.")


def _heuristic_intent(user_input: str, *, has_file: bool, thought: str) -> dict:
    """Fallback intent parser when Gemini is unavailable."""

    text = normalize(str(user_input or ""))

    if any(k in text for k in ["phân tích", "phan tich", "dự đoán", "du doan", "hit"]):
        return {
            "action": "ANALYZE_READY" if has_file else "MISSING_FILE",
            "params": _empty_params(),
            "thought": thought,
        }

    if any(k in text for k in ["tìm", "tim", "giống", "giong", "tương tự", "tuong tu"]) and has_file:
        return {
            "action": "SEARCH_AUDIO",
            "params": _empty_params(),
            "thought": thought,
        }

    if any(k in text for k in ["lời", "loi", "lyrics", "câu", "cau"]):
        return {
            "action": "SEARCH_LYRIC",
            "params": _empty_params(),
            "thought": thought,
        }

    mood_map = {
        "buồn": "buồn",
        "sad": "buồn",
        "suy": "buồn",
        "vui": "vui",
        "happy": "vui",
        "chill": "chill",
        "thư giãn": "chill",
        "thu gian": "chill",
        "quẩy": "quẩy",
        "quay": "quẩy",
        "dance": "quẩy",
    }
    for k, v in mood_map.items():
        if k in text:
            params = _empty_params()
            params["mood"] = v
            return {
                "action": "RECOMMEND_MOOD",
                "params": params,
                "thought": thought,
            }

    genre_map = {
        "rap": "rap",
        "hiphop": "rap",
        "hip-hop": "rap",
        "ballad": "ballad",
        "pop": "pop",
        "rock": "rock",
    }
    for k, v in genre_map.items():
        if k in text:
            params = _empty_params()
            params["genre"] = v
            return {
                "action": "RECOMMEND_GENRE",
                "params": params,
                "thought": thought,
            }

    if any(k in text for k in ["ca sĩ", "ca si", "nghệ sĩ", "nghe si", "artist"]):
        return {
            "action": "RECOMMEND_ARTIST",
            "params": _empty_params(),
            "thought": thought,
        }

    if any(k in text for k in ["hợp âm", "hop am", "nhạc lý", "nhac ly", "theory"]):
        return {
            "action": "MUSIC_KNOWLEDGE",
            "params": _empty_params(),
            "thought": thought,
        }

    return {
        "action": "CLARIFY",
        "params": _empty_params(),
        "thought": thought,
    }

# Chạy test lại
if __name__ == "__main__":
    print("\nTest 2: Phân tích xem bài này có thành hit được không?")
    print(parse_intent_llm("Phân tích xem bài này có thành hit được không?", has_file=False))