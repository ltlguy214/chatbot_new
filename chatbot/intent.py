import json
import re
import os
import unicodedata
import time
import hashlib

try:
    from chatbot.env import load_env
except ModuleNotFoundError:
    from env import load_env

# Load .env robustly (works across different CWDs).
load_env()

_DEFAULT_INTENT_MODEL_CANDIDATES: list[str] = [
    # Try newer models first; fall back to older ones if needed.
    'gemini-2.5-flash',
    'gemini-2.0-flash',
    'gemini-1.5-flash',
    'gemini-1.5-pro',
]

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


_GEMINI_KEY_COOLDOWN_UNTIL: dict[str, float] = {}
_GEMINI_MODEL_CANDIDATE_CACHE: dict[str, tuple[float, list[str]]] = {}


def _key_fingerprint(api_key: str) -> str:
    """Return a non-sensitive id for cooldown bookkeeping."""

    try:
        return hashlib.sha256(str(api_key).encode('utf-8')).hexdigest()[:16]
    except Exception:
        return "unknown"


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


def _normalize_model_name(name: str) -> str:
    name = str(name or '').strip()
    if name.startswith('models/'):
        name = name[len('models/') :]
    return name


def _intent_model_candidates_from_env() -> list[str]:
    """Return model candidates for intent parsing.

    Supports env overrides but remains robust to model deprecations (404).
    """

    raw = str(os.getenv('GEMINI_INTENT_MODEL') or '').strip()
    raw2 = str(os.getenv('GEMINI_MODEL') or '').strip()
    candidates: list[str] = []
    for v in [raw, raw2]:
        n = _normalize_model_name(v)
        if n and n not in candidates:
            candidates.append(n)

    for m in _DEFAULT_INTENT_MODEL_CANDIDATES:
        if m and m not in candidates:
            candidates.append(m)

    return candidates


def _iter_model_names(list_result: object) -> list[str]:
    names: list[str] = []
    try:
        items = list_result
        if isinstance(items, dict):
            items = items.get('models') or items.get('data') or []
        for item in items or []:
            name = None
            if isinstance(item, dict):
                name = item.get('name') or item.get('model')
            else:
                name = getattr(item, 'name', None) or getattr(item, 'model', None)
            if name:
                n = _normalize_model_name(str(name))
                if n and n not in names:
                    names.append(n)
    except Exception:
        return names
    return names


def _rank_model_name(name: str) -> tuple[int, int, str]:
    n = str(name or '').lower()
    tier = 2
    if 'flash' in n:
        tier = 0
    elif 'pro' in n:
        tier = 1

    ver = 99
    if '2.5' in n:
        ver = 0
    elif '2.0' in n:
        ver = 1
    elif '1.5' in n:
        ver = 2

    return tier, ver, n


def _discover_intent_models(client: object, *, cache_key: str) -> list[str]:
    """Discover available models from the API (best-effort).

    Cached for 30 minutes per key fingerprint.
    """

    try:
        now = float(time.time())
    except Exception:
        now = 0.0

    cached = _GEMINI_MODEL_CANDIDATE_CACHE.get(cache_key)
    if cached:
        ts, models = cached
        if now and ts and now - ts < 30 * 60 and models:
            return models

    try:
        models_api = getattr(client, 'models', None)
        if models_api is None or not hasattr(models_api, 'list'):
            return []
        list_result = models_api.list()
        names = _iter_model_names(list_result)
        names = [n for n in names if 'gemini' in n.lower() and 'embedding' not in n.lower()]
        names = sorted(names, key=_rank_model_name)
        picked = names[:8]
        if picked:
            _GEMINI_MODEL_CANDIDATE_CACHE[cache_key] = (now, picked)
        return picked
    except Exception:
        return []


def _looks_like_model_not_found_error(err: Exception) -> bool:
    msg = str(err).lower()
    return (
        '404' in msg
        or 'not_found' in msg
        or 'no longer available' in msg
        or ('this model' in msg and 'available' in msg)
    )


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
        now = time.time()

        keys = _parse_gemini_api_keys()
        if not keys:
            return _heuristic_intent(
                user_input,
                has_file=has_file,
                thought="Gemini chưa được cấu hình (GEMINI_API_KEY / GEMINI_API_KEYS), mình dùng phân loại nhanh.",
            )

        model_candidates = _intent_model_candidates_from_env()
        last_err: Exception | None = None
        last_quota_err: Exception | None = None
        response_text: str | None = None

        # Only use keys that are not cooling down.
        eligible_keys: list[str] = []
        min_remaining: int | None = None
        for api_key in keys:
            fp = _key_fingerprint(api_key)
            until = float(_GEMINI_KEY_COOLDOWN_UNTIL.get(fp, 0.0) or 0.0)
            if until and now < until:
                remaining = int(until - now)
                if min_remaining is None or remaining < min_remaining:
                    min_remaining = remaining
                continue
            eligible_keys.append(api_key)

        if not eligible_keys:
            remaining = int(min_remaining or 0)
            suffix = f" (cooldown {remaining}s)" if remaining > 0 else ""
            return _heuristic_intent(
                user_input,
                has_file=has_file,
                thought=f"Gemini đang quá quota/quá tải{suffix}, mình dùng phân loại nhanh.",
            )

        for api_key in eligible_keys:
            try:
                from google import genai

                client = genai.Client(api_key=api_key)
                saw_model_not_found = False
                for model_name in model_candidates:
                    try:
                        response = client.models.generate_content(
                            model=model_name,
                            contents=prompt,
                        )
                        response_text = getattr(response, "text", None)
                        if response_text:
                            break
                    except Exception as ex:
                        last_err = ex
                        if _looks_like_quota_error(ex):
                            last_quota_err = ex
                            fp = _key_fingerprint(api_key)
                            _GEMINI_KEY_COOLDOWN_UNTIL[fp] = time.time() + 10 * 60
                            response_text = None
                            break
                        if _looks_like_model_not_found_error(ex):
                            # Try next model candidate with the same key.
                            saw_model_not_found = True
                            continue
                        # Other errors: try next key.
                        break

                # If configured candidates are deprecated (404), discover models and retry once.
                if not response_text and saw_model_not_found:
                    fp = _key_fingerprint(api_key)
                    discovered = _discover_intent_models(client, cache_key=fp)
                    for model_name in discovered:
                        try:
                            response = client.models.generate_content(
                                model=model_name,
                                contents=prompt,
                            )
                            response_text = getattr(response, 'text', None)
                            if response_text:
                                break
                        except Exception as ex:
                            last_err = ex
                            if _looks_like_quota_error(ex):
                                last_quota_err = ex
                                _GEMINI_KEY_COOLDOWN_UNTIL[fp] = time.time() + 10 * 60
                                response_text = None
                                break
                            if _looks_like_model_not_found_error(ex):
                                continue
                            break

                if response_text:
                    break
            except Exception as e:
                last_err = e
                if _looks_like_quota_error(e):
                    # Rotate key on quota/rate-limit.
                    last_quota_err = e
                    fp = _key_fingerprint(api_key)
                    _GEMINI_KEY_COOLDOWN_UNTIL[fp] = time.time() + 10 * 60
                    continue
                # Non-quota errors: still try next key (can be invalid key / region).
                continue

        if not response_text:
            # Avoid leaking raw quota/stack traces to the UI.
            if last_quota_err is not None and _looks_like_quota_error(last_quota_err):
                detail = _short_error_for_ui(last_quota_err)
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

            intent_data = _postprocess_intent(intent_data, user_input)
            
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
            return _heuristic_intent(user_input, has_file=has_file, thought="Gemini không trả về JSON chuẩn, mình dùng phân loại nhanh.")
            
    except Exception as e:
        return _heuristic_intent(user_input, has_file=has_file, thought="Có lỗi khi phân tích ý định, mình dùng phân loại nhanh.")


def _cleanup_entity_text(value: str) -> str:
    text = str(value or '').strip()
    if not text:
        return ''
    # Remove wrapping quotes.
    text = text.strip('"\'“”‘’`')
    # Trim common trailing fillers.
    text = re.sub(r"\s+(nhe|nhé|di|đi|voi|với|nha|nhá|giup|giúp|please|pls)\b.*$", "", text, flags=re.IGNORECASE)
    # Trim trailing punctuation.
    text = re.sub(r"[\s\.,;:!\?]+$", "", text).strip()
    return text


def _extract_artist_from_text(user_input: str) -> str:
    raw = unicodedata.normalize('NFC', str(user_input or '')).strip()
    if not raw:
        return ''

    def _reject_ambiguous(cand: str) -> bool:
        c = normalize(str(cand or ''))
        if not c:
            return True
        # Avoid misclassifying mood/genre tokens as artist names.
        bad = {
            'vui', 'buon', 'buồn', 'sad', 'suy', 'chill', 'thu gian', 'thư giãn', 'quay', 'quẩy', 'dance',
            'rap', 'hiphop', 'hip-hop', 'pop', 'rock', 'ballad',
            'toi', 'tôi', 'minh', 'mình', 'ban', 'bạn', 'tao', 't',
        }
        if c in bad:
            return True
        # Too short to be a meaningful artist name.
        if len(c) < 3:
            return True
        return False

    patterns = [
        # "tìm nhạc của Phùng Khánh Linh"
        r"(?:nhac|nhạc|bai hat|bài hát|bai|bài|song|songs|track|tracks)\s+(?:cua|của)\s+(.+)$",
        # "bài của Phùng Khánh Linh"
        r"(?:bai|bài)\s+(?:cua|của)\s+(.+)$",
        # "ca sĩ Phùng Khánh Linh"
        r"(?:ca\s*si|ca\s*sĩ|nghe\s*si|nghệ\s*sĩ|artist)\s+(.+)$",
    ]

    for pat in patterns:
        m = re.search(pat, raw, flags=re.IGNORECASE)
        if not m:
            continue
        cand = _cleanup_entity_text(m.group(1))
        if cand and not _reject_ambiguous(cand):
            return cand

    # Generic: if contains "của/cua" and looks like a person name after it.
    m = re.search(r"\b(?:cua|của)\b\s+(.+)$", raw, flags=re.IGNORECASE)
    if m:
        cand = _cleanup_entity_text(m.group(1))
        if cand and len(cand) >= 3 and not _reject_ambiguous(cand):
            return cand

    return ''


def _extract_lyric_snippet_from_text(user_input: str) -> str:
    raw = unicodedata.normalize('NFC', str(user_input or '')).strip()
    if not raw:
        return ''

    # Prefer quoted snippet.
    m = re.search(r"[\"'“”‘’]([^\"'“”‘’]{5,})[\"'“”‘’]", raw)
    if m:
        return _cleanup_entity_text(m.group(1))

    m = re.search(r"(?:loi|lời|lyrics|doan|đoạn)\s*(?:nhac|nhạc)?\s*[:\-]?\s*(.+)$", raw, flags=re.IGNORECASE)
    if m:
        return _cleanup_entity_text(m.group(1))

    # Fallback: use the full user input.
    return raw


def _extract_song_title_from_text(user_input: str) -> tuple[str, str]:
    """Return (song_title, artist) best-effort for SEARCH_NAME."""

    raw = unicodedata.normalize('NFC', str(user_input or '')).strip()
    if not raw:
        return '', ''

    # 1) Quoted title: "See Tình" / 'See Tinh'
    m = re.search(r"[\"'“”‘’`]([^\"'“”‘’`]{2,})[\"'“”‘’`]", raw)
    if m:
        title = _cleanup_entity_text(m.group(1))
        artist = _extract_artist_from_text(raw)
        return title, artist

    # 2) "tên bài (hát) ..." / "bài hát tên ..."
    m = re.search(r"(?:ten\s+bai\s+hat|ten\s+bai|tên\s+bài\s+hát|tên\s+bài)\s*[:\-]?\s*(.+)$", raw, flags=re.IGNORECASE)
    if m:
        tail = _cleanup_entity_text(m.group(1))
        if tail:
            # Split "<title> của <artist>" if present.
            m2 = re.search(r"^(.+?)\s+(?:cua|của)\s+(.+)$", tail, flags=re.IGNORECASE)
            if m2:
                return _cleanup_entity_text(m2.group(1)), _cleanup_entity_text(m2.group(2))
            return tail, ''

    return '', ''


def _postprocess_intent(intent_data: dict, user_input: str) -> dict:
    """Fill missing params when Gemini returns action but empty entity."""

    if not isinstance(intent_data, dict):
        return intent_data
    params = intent_data.get('params')
    if not isinstance(params, dict):
        params = {}
        intent_data['params'] = params

    action = str(intent_data.get('action') or '').upper().strip()

    if action == 'RECOMMEND_ARTIST':
        if not str(params.get('artist') or '').strip():
            artist = _extract_artist_from_text(user_input)
            if artist:
                params['artist'] = artist

    if action == 'SEARCH_LYRIC':
        if not str(params.get('lyric_snippet') or '').strip():
            snippet = _extract_lyric_snippet_from_text(user_input)
            if snippet:
                params['lyric_snippet'] = snippet

    # If Gemini is unsure (CLARIFY) but the pattern is clearly artist-based, upgrade.
    if action == 'CLARIFY':
        artist = _extract_artist_from_text(user_input)
        if artist:
            intent_data['action'] = 'RECOMMEND_ARTIST'
            params['artist'] = artist

    return intent_data


def _heuristic_intent(user_input: str, *, has_file: bool, thought: str) -> dict:
    """Fallback intent parser when Gemini is unavailable."""

    text = normalize(str(user_input or ""))

    # Strong artist pattern: "nhạc/bài của <artist>".
    artist = _extract_artist_from_text(str(user_input or ''))
    if artist:
        params = _empty_params()
        params['artist'] = artist
        return {
            'action': 'RECOMMEND_ARTIST',
            'params': params,
            'thought': thought,
        }

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
        params = _empty_params()
        params['lyric_snippet'] = _extract_lyric_snippet_from_text(str(user_input or ''))
        return {
            "action": "SEARCH_LYRIC",
            "params": params,
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
        params = _empty_params()
        # Best-effort extraction after the keyword.
        extracted = _extract_artist_from_text(str(user_input or ''))
        if extracted:
            params['artist'] = extracted
        return {
            "action": "RECOMMEND_ARTIST",
            "params": params,
            "thought": thought,
        }

    # SEARCH_NAME best-effort when user explicitly provides a title.
    # (Avoid stealing mood/genre queries like "tìm nhạc vui".)
    mood_tokens = ["buồn", "sad", "suy", "vui", "happy", "chill", "thư giãn", "thu gian", "quẩy", "quay", "dance"]
    genre_tokens = ["rap", "hiphop", "hip-hop", "ballad", "pop", "rock"]
    if not any(tok in text for tok in (mood_tokens + genre_tokens)):
        title, artist2 = _extract_song_title_from_text(str(user_input or ''))
        if title:
            params = _empty_params()
            params['song_title'] = title
            if artist2:
                params['artist'] = artist2
            return {
                'action': 'SEARCH_NAME',
                'params': params,
                'thought': thought,
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