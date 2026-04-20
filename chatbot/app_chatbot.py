import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
import warnings
import re
import time
import tempfile
import traceback
import json
import urllib.parse
import urllib.request
import logging
import unicodedata
import hashlib
import uuid
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

try:
    # When running from a parent folder that contains the `chatbot/` package.
    from chatbot.env import load_env
    from chatbot.render import render_spotify_artist_payload
    from chatbot.spotify import (
        spotify_access_token,
        spotify_api_get_json,
        spotify_get_artist_top_tracks as spotify_get_artist_top_tracks_api,
        spotify_get_track_metadata as spotify_get_track_metadata_api,
        spotify_get_tracks_metadata as spotify_get_tracks_metadata_api,
        spotify_get_tracks_metadata,
        spotify_pick_image_url,
        spotify_search_artist as spotify_search_artist_api,
    )
    from chatbot.supabase import (
        get_supabase_client as _supabase_get_client,
        query_supabase_lyrics as _supabase_query_lyrics,
        query_supabase_vector as _supabase_query_vector,
        append_chat_history as _supabase_append_chat_history,
        fetch_chat_history as _supabase_fetch_chat_history,
        fetch_recent_chat_history as _supabase_fetch_recent_chat_history,
    )
    from chatbot.model_store import resolve_model_paths
except ModuleNotFoundError:
    # Streamlit Cloud often executes with repo-root being this folder itself.
    # In that case, imports must be local (no `chatbot.` prefix).
    from env import load_env
    from render import render_spotify_artist_payload
    from spotify import (
        spotify_access_token,
        spotify_api_get_json,
        spotify_get_artist_top_tracks as spotify_get_artist_top_tracks_api,
        spotify_get_track_metadata as spotify_get_track_metadata_api,
        spotify_get_tracks_metadata as spotify_get_tracks_metadata_api,
        spotify_get_tracks_metadata,
        spotify_pick_image_url,
        spotify_search_artist as spotify_search_artist_api,
    )
    from supabase import (
        get_supabase_client as _supabase_get_client,
        query_supabase_lyrics as _supabase_query_lyrics,
        query_supabase_vector as _supabase_query_vector,
        append_chat_history as _supabase_append_chat_history,
        fetch_chat_history as _supabase_fetch_chat_history,
        fetch_recent_chat_history as _supabase_fetch_recent_chat_history,
    )
    from model_store import resolve_model_paths

# LẤY HOẶC TẠO SESSION
def _get_or_create_session_id() -> str:
    if 'chat_session_id' in st.session_state and str(st.session_state.chat_session_id).strip():
        return str(st.session_state.chat_session_id).strip()

    sid = ''
    try:
        sid = str(st.query_params.get('sid', '') or '').strip()
    except Exception:
        try:
            sid = str((st.experimental_get_query_params() or {}).get('sid', [''])[0] or '').strip()
        except Exception:
            sid = ''

    if not sid:
        sid = uuid.uuid4().hex
        try:
            st.query_params['sid'] = sid
        except Exception:
            try:
                st.experimental_set_query_params(sid=sid)
            except Exception:
                pass

    st.session_state.chat_session_id = sid
    return sid


def _persist_chat_message(*, module: str, role: str, content: str) -> None:
    content = str(content or '').strip()
    if not content:
        return
    try:
        _supabase_append_chat_history(
            session_id=_get_or_create_session_id(),
            role=str(role or 'user'),
            content=content,
            module=str(module or 'default'),
        )
    except Exception:
        return


def _load_chat_history_into_state(*, module: str, state_key: str, greeting: list[dict]) -> None:
    if state_key in st.session_state and isinstance(st.session_state.get(state_key), list):
        return

    rows: list[dict] = []
    try:
        rows = _supabase_fetch_chat_history(session_id=_get_or_create_session_id(), module=module, limit=500)
    except Exception:
        rows = []

    messages: list[dict] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        role = str(row.get('role') or 'assistant')
        content = str(row.get('content') or '')
        if not content:
            continue
        messages.append({'role': role, 'content': content})

    st.session_state[state_key] = messages if messages else list(greeting)

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    # Nếu thiếu python-dotenv, app vẫn chạy với biến môi trường hệ thống.
    pass

# Robust load: find workspace/root .env even if Streamlit runs from a subdir.
load_env()


def _load_dotenv_fallback() -> None:
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        candidate_paths = [
            os.path.join(here, '.env'),
            os.path.abspath(os.path.join(here, '..', '.env')),
            os.path.abspath(os.path.join(here, '..', '..', '.env')),
        ]
        env_path = next((p for p in candidate_paths if os.path.exists(p)), None)
        if not env_path:
            return

        with open(env_path, 'r', encoding='utf-8') as f:
            for raw in f.read().splitlines():
                line = raw.strip()
                if not line or line.startswith('#') or '=' not in line:
                    continue
                key, val = line.split('=', 1)
                key = key.strip()
                if not key or key in os.environ:
                    continue
                val = val.strip().strip('"').strip("'")
                os.environ[key] = val
    except Exception:
        return


_load_dotenv_fallback()

# =============================================================================
# 1. CẤU HÌNH & IMPORT MODULE BÊN NGOÀI (Giữ nguyên từ code cũ)
# =============================================================================
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_AUDIO_LYRIC_DIR = os.path.join(_REPO_ROOT, 'Audio_lyric')

for _p in (_REPO_ROOT, _AUDIO_LYRIC_DIR):
    try:
        if _p and _p not in sys.path:
            sys.path.append(_p)
    except Exception:
        pass

def safe_joblib_load(file_path, *, mmap_mode: str | None = None):
    try:
        if mmap_mode:
            return joblib.load(file_path, mmap_mode=mmap_mode)
        return joblib.load(file_path)
    except Exception as ex:
        # If mmap_mode failed, retry without it for compatibility.
        if mmap_mode:
            try:
                return joblib.load(file_path)
            except Exception as ex2:
                msg = str(ex2) or str(ex)
                detail = f"{type(ex2).__name__}: {msg}" if msg else f"{type(ex2).__name__}"
                raise Exception(f"Failed to load {file_path}: {detail}") from ex2

        msg = str(ex)
        detail = f"{type(ex).__name__}: {msg}" if msg else f"{type(ex).__name__}"
        raise Exception(f"Failed to load {file_path}: {detail}") from ex


@st.cache_resource(show_spinner=False)
def _cached_joblib_load(path: str):
    return safe_joblib_load(path)

def safe_remove(file_path):
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception:
        pass

try:
    # Prefer the maintained backend inside the `chatbot/` package.
    try:
        from chatbot.analysis_backend import AudioAnalyzer, NLPAnalyzer
    except ModuleNotFoundError:
        from analysis_backend import AudioAnalyzer, NLPAnalyzer

    audio_analyzer = AudioAnalyzer()
    nlp_analyzer = NLPAnalyzer()

    def build_intent_parsing_prompt(user_input):
        return (
            'Bạn là hệ thống Intent Parser cho chatbot âm nhạc V-Pop. '
            'Chỉ trả về JSON gọn, không kèm giải thích. '
            f'User input: {user_input}'
        )

    def build_recommendation_generation_prompt(top_5_songs_metadata, user_input):
        return (
            'Hãy viết một đoạn gợi ý thân thiện dựa trên top_5_songs_metadata '
            f'và user_input={user_input}. Dữ liệu: {top_5_songs_metadata}'
        )

    def build_producer_advice_prompt(hit_probability, shap_values, technical_meta):
        try:
            shap_json = json.dumps(shap_values, ensure_ascii=False)
        except Exception:
            shap_json = str(shap_values)

        # Lấy các thông số kỹ thuật ra để AI đọc
        tempo = technical_meta.get('tempo', 'N/A')
        energy = technical_meta.get('energy', 'N/A')
        duration = technical_meta.get('duration', 'N/A')
        lexical = technical_meta.get('lexical', 'N/A')
        style = technical_meta.get('style', 'N/A')
        emotion = technical_meta.get('emotion', 'N/A')
        genres = technical_meta.get('genres', 'N/A')

        return (
            "Bạn là một Music Producer chuyên nghiệp, dày dạn kinh nghiệm tại thị trường V-Pop. "
            "Nhiệm vụ của bạn là đưa ra nhận xét chuyên môn dựa trên dữ liệu định lượng nhưng phải diễn đạt bằng ngôn ngữ âm nhạc tự nhiên.\n\n"
        
            "⚠️ QUY TẮC CỐT LÕI:\n"
            "1. TUYỆT ĐỐI KHÔNG liệt kê tên biến kỹ thuật (như mfcc, chroma, centroid...). Hãy dịch chúng sang thuật ngữ âm nhạc (ví dụ: độ sáng, độ dày, âm sắc, sự bùng nổ).\n"
            "2. KHÔNG đưa các con số phần trăm (%) lẻ tẻ từ SHAP vào bài viết. Chỉ dùng con số Hit Probability tổng quát.\n"
            "3. Văn phong: Chuyên nghiệp, truyền cảm hứng, như một người đàn anh đang hướng dẫn đàn em trong phòng thu.\n\n"

            "NHIỆM VỤ CỦA BẠN:\n"
            "1. LỜI CHÀO & TỔNG QUAN: Nhận xét ngắn gọn về bản demo dựa trên: "
            f"Tempo {technical_meta['tempo']} BPM, Năng lượng {technical_meta['energy']}, Thời lượng {technical_meta['duration']}s, "
            f"Độ đa dạng từ {technical_meta['lexical']}, Phong cách {technical_meta['style']}, Cảm xúc {technical_meta['emotion']}, Thể loại {technical_meta['genres']}.\n"
            f"2. ĐÁNH GIÁ TIỀM NĂNG: Nhận xét về con số {hit_probability} xác suất thành Hit một cách thực tế.\n"
            "3. PHÂN TÍCH ƯU/NHƯỢC ĐIỂM: Dựa vào SHAP_JSON, hãy nói về các yếu tố giúp bài hát bắt tai hoặc các phần làm bài hát bị rời rạc (Dùng ngôn ngữ producer: mix, arrangement, vocal).\n"
            "4. HÀNH ĐỘNG CẢI THIỆN: Đề xuất 3 thay đổi cụ thể để đẩy bài hát lên tầm cao mới.\n\n"
            
            f"DỮ LIỆU ĐẦU VÀO (DÙNG ĐỂ PHÂN TÍCH, KHÔNG ĐƯỢC CHÉP NGUYÊN XI): {shap_json}"
        )
except Exception:
    # Fallback lightweight analyzers so app never breaks.
    class DummyAnalyzer:
        def process_audio_file(self, path):
            return {}

        def analyze_full_lyrics(self, text):
            return {}

        def extract_keywords(self, text):
            if not text:
                return []
            words = re.findall(r"\w+", text.lower())
            # Fallback tokenizer đơn giản để tab chat không lỗi khi thiếu src NLP module.
            return [w for w in words if len(w) > 2][:8]

    audio_analyzer = DummyAnalyzer()
    nlp_analyzer = DummyAnalyzer()

    def build_intent_parsing_prompt(user_input):
        return (
            'Ban la he thong Intent Parser cho chatbot am nhac V-Pop. '
            'Chi tra ve JSON voi ba truong emotion, style, genre. '
            f'User input: {user_input}'
        )

    def build_recommendation_generation_prompt(top_5_songs_metadata, user_input):
        return (
            'Hay viet mot doan goi y than thien dua tren top_5_songs_metadata '
            f'va user_input={user_input}. Du lieu: {top_5_songs_metadata}'
        )

    def build_producer_advice_prompt(hit_probability, shap_values, technical_meta):
        try:
            shap_json = json.dumps(shap_values, ensure_ascii=False)
        except Exception:
            shap_json = str(shap_values)
        
        # Lấy thông số kỹ thuật ra biến
        tempo = technical_meta.get('tempo', 'N/A')
        energy = technical_meta.get('energy', 'N/A')
        style = technical_meta.get('style', 'N/A')
        emotion = technical_meta.get('emotion', 'N/A')
        genres = technical_meta.get('genres', 'N/A')
        duration = technical_meta.get('duration', 'N/A')
        lexical = technical_meta.get('lexical', 'N/A')

        return (
            "Bạn là Music Producer chuyên V-Pop. Phân tích dựa trên dữ liệu định lượng. "
            "CHỈ được dùng số liệu có trong SHAP_JSON; không được bịa thêm feature hay phần trăm. "
            "Nếu thiếu dữ liệu thì nói rõ 'không có dữ liệu'.\n\n"
            "Hãy bắt đầu bằng một câu chào thân thiện và liệt kê các thông số bạn vừa phân tích được từ bản demo của người dùng "
            f"(bao gồm: Tempo {tempo} BPM, RMS Energy {energy}, Phong cách {style}, Cảm xúc {emotion}, Thể loại {genres}, Thời lượng {duration} giây, Độ đa dạng từ vựng {lexical}).\n\n"
            f"Sau đó, nhận xét về khả năng trở thành Hit ({hit_probability}) và dựa vào dữ liệu SHAP dưới đây để đưa ra lời khuyên chuyên sâu.\n\n"
            f"SHAP_JSON={shap_json}\n\n"
            "Yêu cầu output:\n"
            "1) 1-2 câu nhận xét tổng quan về demo.\n"
            "2) 1-2 câu tóm tắt khả năng hit.\n"
            "3) Liệt kê tối đa 5 yếu tố 'đẩy lên' (shap_value dương) và tối đa 5 yếu tố 'kéo xuống' (shap_value âm).\n"
            "4) Đề xuất 3 hành động cải thiện thực tế (mix/arrangement/lyrics) dựa đúng các yếu tố đó.\n"
        )

try:
    from mutagen.mp3 import MP3
    from mutagen.id3 import ID3, USLT
    MUTAGEN_AVAILABLE = True
except ImportError:
    MUTAGEN_AVAILABLE = False

# =============================================================================
# 2. COMPATIBILITY SHIMS CHO MODEL PICKLE CŨ
# =============================================================================
try:
    import sklearn._loss as _sklearn_loss

    # Một số model cũ serialize tham chiếu module nội bộ `_loss`.
    sys.modules.setdefault('_loss', _sklearn_loss)

    # Bridge các class Cy*Loss sang namespace sklearn._loss cho pickle cũ.
    try:
        import sklearn._loss._loss as _sklearn_loss_impl

        for _loss_name in dir(_sklearn_loss_impl):
            if _loss_name.startswith('Cy') and _loss_name.endswith('Loss') and not hasattr(_sklearn_loss, _loss_name):
                setattr(_sklearn_loss, _loss_name, getattr(_sklearn_loss_impl, _loss_name))
    except Exception:
        pass
except Exception:
    pass

try:
    from sklearn.compose._column_transformer import _RemainderColsList
except ImportError:
    class _RemainderColsList(list):
        pass

    import sklearn.compose._column_transformer

    sklearn.compose._column_transformer._RemainderColsList = _RemainderColsList

from sklearn.impute import SimpleImputer

if not hasattr(SimpleImputer, '_fill_dtype'):
    SimpleImputer._fill_dtype = property(lambda self: getattr(self, '_fit_dtype', None))

if not hasattr(np.random, 'bit_generator'):
    np.random.bit_generator = np.random._bit_generator

try:
    from numpy.random import MT19937

    if not hasattr(np.random, 'MT19937'):
        np.random.MT19937 = MT19937
except Exception:
    pass

try:
    import numpy.random._pickle as _np_random_pickle

    _orig_bitgen_ctor = _np_random_pickle.__bit_generator_ctor

    def _compat_bit_generator_ctor(bit_generator_name='MT19937'):
        # Pickle cũ đôi khi lưu class object thay vì tên string.
        if isinstance(bit_generator_name, type):
            bit_generator_name = bit_generator_name.__name__
        return _orig_bitgen_ctor(bit_generator_name)

    _np_random_pickle.__bit_generator_ctor = _compat_bit_generator_ctor
except Exception:
    pass

# =============================================================================
# 3. CẤU HÌNH TRANG & CUSTOM CSS (Style VlogMusic)
# =============================================================================
warnings.filterwarnings('ignore')
st.set_page_config(page_title="VMusic AI", page_icon="🎵", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    * { font-family: 'Inter', sans-serif; }
    .stApp { background: #0b1020 !important; background-attachment: fixed; }
    /* Leave room so the fixed chat input doesn't cover the last messages */
    .main .block-container { padding-top: 1rem !important; padding-bottom: 8rem !important; max-width: 980px !important; }
    header[data-testid="stHeader"] { background: transparent; height: 0rem; }
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%); border-right: 1px solid rgba(255, 255, 255, 0.1); }
    [data-testid="stSidebar"] .element-container { color: #ffffff; }

    h1, h2, h3, p, span, div, label { color: #ffffff; font-weight: 500; }
    h1 { background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 3rem !important; margin-bottom: 0.5rem !important; font-weight: 800;}

    .stTabs [data-baseweb="tab-list"] { gap: 8px; background: rgba(0,0,0,0.2); padding: 8px; border-radius: 16px; }
    .stTabs [data-baseweb="tab"] { height: 50px; background: transparent; border-radius: 12px; color: rgba(255, 255, 255, 0.6); font-weight: 600; font-size: 15px; padding: 0 24px; transition: all 0.3s ease; border: none; }
    .stTabs [data-baseweb="tab"]:hover { background: rgba(102, 126, 234, 0.2); color: #ffffff; }
    .stTabs [aria-selected="true"] { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important; color: #ffffff !important; box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4); }

    .stButton>button { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border: none; border-radius: 12px; padding: 12px 32px; font-weight: 600; transition: all 0.3s ease; box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4); }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6); }

    .stTextInput>div>div>input, .stTextArea>div>div>textarea { background: rgba(48, 43, 99, 0.5) !important; border: 2px dashed rgba(102, 126, 234, 0.6) !important; border-radius: 16px; padding: 16px; transition: all 0.3s ease; }
    .stTextInput>div>div>input:focus, .stTextArea>div>div>textarea:focus { border-color: #667eea !important; background: rgba(48, 43, 99, 0.7) !important; }

    /* Main composer: place send icon inside the input (right side) */
    div[data-testid="stVerticalBlock"]:has(#main-composer-text-anchor) { position: relative; }
    div[data-testid="stVerticalBlock"]:has(#main-composer-text-anchor) [data-testid="stTextInput"] input {
        padding-right: 3.25rem !important;
    }
    div[data-testid="stVerticalBlock"]:has(#main-composer-text-anchor) [data-testid="stButton"] {
        position: absolute;
        right: 0.75rem;
        top: 50%;
        transform: translateY(-50%);
        z-index: 30;
    }
    div[data-testid="stVerticalBlock"]:has(#main-composer-text-anchor) [data-testid="stButton"] button {
        width: 44px !important;
        height: 44px !important;
        min-width: 44px !important;
        min-height: 44px !important;
        padding: 0 !important;
        border-radius: 999px !important;
        background: transparent !important;
        box-shadow: none !important;
        border: 1px solid rgba(102, 126, 234, 0.28) !important;
    }
    div[data-testid="stVerticalBlock"]:has(#main-composer-text-anchor) [data-testid="stButton"] button:hover {
        transform: none !important;
        box-shadow: none !important;
        background: rgba(48, 43, 99, 0.45) !important;
    }

    /* Compact uploader (inside + popover), avoid big dark dropzone */
    [data-testid="stFileUploader"] {
        padding: 0 !important;
        margin: 0 !important;
        background: transparent !important;
        border: none !important;
    }

    [data-testid="stFileUploaderDropzone"] {
        min-height: unset !important;
        padding: 0 !important;
        border: none !important;
        border-radius: 0 !important;
        background: transparent !important;
        box-shadow: none !important;
    }

    /* Hide the drag/drop instruction box so only the browse button remains */
    [data-testid="stFileUploaderDropzoneInstructions"] {
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    /* Make browse button compact like a menu item */
    [data-testid="stFileUploaderDropzone"] button {
        width: 100% !important;
        border-radius: 12px !important;
        background: rgba(48, 43, 99, 0.55) !important;
        border: 1px solid rgba(102, 126, 234, 0.28) !important;
        color: rgba(235, 240, 255, 0.95) !important;
        box-shadow: 0 8px 22px rgba(8, 12, 30, 0.25) !important;
    }

    [data-testid="stFileUploaderDropzone"] button:hover {
        background: rgba(48, 43, 99, 0.68) !important;
        transform: none !important;
    }

    /* Popover trigger as a pure round '+' button (hide the caret icon) */
    button[data-testid="stPopoverButton"] > div {
        width: 100% !important;
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    button[data-testid="stPopoverButton"] p {
        margin: 0 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }

    /* Hide the popover caret (expand_more) while keeping the '+' icon */
    button[data-testid="stPopoverButton"] > div > div:last-child {
        display: none !important;
    }

    /* Tighten the gap between the input row and the '+ / Flash' row */
    div[data-testid="stHorizontalBlock"]:has(div[data-testid="stPopover"]):has(div[data-testid="stSelectbox"]:has(input[aria-label*="Gemini model"])) {
        margin-top: 0px !important;
    }

    /* Reduce the default widget bottom spacing for the composer input so row-2 sits closer */
    div[data-testid="stTextInput"]:has(input[aria-label="Main composer"]),
    div[data-testid="stTextInput"]:has(input[aria-label="Composer"]) {
        margin-bottom: 0px !important;
    }

    /* Reduce top spacing for row-2 widgets */
    div[data-testid="stPopover"],
    div[data-testid="stSelectbox"]:has(input[aria-label*="Gemini model"]) {
        margin-top: 0 !important;
    }

    /* Gemini model select (Flash/Pro) as a compact pill, matching the composer style */
    div[data-testid="stSelectbox"]:has(input[aria-label*="Gemini model"]) [data-baseweb="select"] {
        background: rgba(48, 43, 99, 0.5) !important;
        border: 1px solid rgba(102, 126, 234, 0.3) !important;
        border-radius: 999px !important;
        box-shadow: 0 6px 18px rgba(8, 12, 30, 0.18) !important;
        display: inline-flex !important;
        flex: 0 0 auto !important;
        width: fit-content !important;
        min-width: 110px !important;
        max-width: none !important;
        margin-right: 0px !important;
        cursor: pointer !important;
    }

    /* Force hand cursor even when hovering the text/input inside the select */
    div[data-testid="stSelectbox"]:has(input[aria-label*="Gemini model"]) [data-baseweb="select"] input,
    div[data-testid="stSelectbox"]:has(input[aria-label*="Gemini model"]) [data-baseweb="select"] span,
    div[data-testid="stSelectbox"]:has(input[aria-label*="Gemini model"]) [data-baseweb="select"] div {
        cursor: pointer !important;
    }

    div[data-testid="stSelectbox"]:has(input[aria-label*="Gemini model"]) [data-baseweb="select"] > div {
        background: transparent !important;
        border: none !important;
        border-radius: 999px !important;
        min-height: 46px !important;
        padding-left: 8px !important;
        padding-right: 4px !important;
        white-space: nowrap !important;
    }

    /* Make the Flash text sit closer to the caret icon */
    div[data-testid="stSelectbox"]:has(input[aria-label*="Gemini model"]) [data-baseweb="select"] [data-baseweb="icon"] {
        margin-left: 1px !important;
    }

    /* Ensure the caret icon stays visible (some themes hide it on hover) */
    div[data-testid="stSelectbox"]:has(input[aria-label*="Gemini model"]) [data-baseweb="select"] img,
    div[data-testid="stSelectbox"]:has(input[aria-label*="Gemini model"]) [data-baseweb="select"] svg {
        opacity: 1 !important;
        visibility: visible !important;
        filter: none !important;
    }

    /* Blur/translucent panels (avoid black) for '+' popover content */
    div[data-testid="stPopoverBody"][data-baseweb="popover"] {
        background: rgba(48, 43, 99, 0.72) !important;
        border: 1px solid rgba(102, 126, 234, 0.28) !important;
        box-shadow: 0 10px 30px rgba(8, 12, 30, 0.35) !important;
        backdrop-filter: blur(14px) !important;
    }

    /* Streamlit wraps popover content in a dark div; neutralize it */
    div[data-testid="stPopoverBody"][data-baseweb="popover"] > div {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }

    /* Blur/translucent panels (avoid black) for select dropdown menu */
    ul[data-testid="stSelectboxVirtualDropdown"] {
        background: rgba(48, 43, 99, 0.82) !important;
        border: 1px solid rgba(102, 126, 234, 0.28) !important;
        backdrop-filter: blur(14px) !important;
    }

    div[data-baseweb="popover"]:has(ul[data-testid="stSelectboxVirtualDropdown"]) {
        background: rgba(48, 43, 99, 0.80) !important;
        border: 1px solid rgba(102, 126, 234, 0.28) !important;
        box-shadow: 0 10px 30px rgba(8, 12, 30, 0.35) !important;
        backdrop-filter: blur(14px) !important;
    }

    /* Send button: more vivid (gradient + glow) */
    .st-key-main_compose_send .stButton > button,
    .st-key-listener_compose_send .stButton > button,
    .st-key-producer_compose_send .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border: none !important;
        box-shadow: 0 10px 28px rgba(102, 126, 234, 0.35) !important;
    }

    .st-key-main_compose_send .stButton > button [data-testid="stIconMaterial"],
    .st-key-listener_compose_send .stButton > button [data-testid="stIconMaterial"],
    .st-key-producer_compose_send .stButton > button [data-testid="stIconMaterial"] {
        color: rgba(235, 240, 255, 0.95) !important;
    }

    /* Keep row-2 controls from overlapping the chat input */
    div[data-testid="stSelectbox"]:has(input[aria-label*="Gemini model"]) {
        margin-top: 0 !important;
        display: flex !important;
        justify-content: flex-end !important;
        padding-right: 0px !important;
    }

    /* Streamlit sometimes renders the caret as a sibling <img alt="open">; keep it visible */
    div[data-testid="stSelectbox"]:has(input[aria-label*="Gemini model"]) img[alt="open"] {
        opacity: 1 !important;
        visibility: visible !important;
        filter: none !important;
        display: block !important;
        width: 18px !important;
        height: 18px !important;
        pointer-events: none !important;
    }

    /* Subtle hover glow (light only) */
    button[data-testid="stPopoverButton"]:hover {
        background: rgba(48, 43, 99, 0.62) !important;
        border-color: rgba(102, 126, 234, 0.42) !important;
        box-shadow: 0 10px 26px rgba(102, 126, 234, 0.18) !important;
        transform: none !important;
    }

    div[data-testid="stSelectbox"]:has(input[aria-label*="Gemini model"]) [data-baseweb="select"]:hover {
        background: rgba(48, 43, 99, 0.62) !important;
        border-color: rgba(102, 126, 234, 0.42) !important;
        box-shadow: 0 10px 26px rgba(102, 126, 234, 0.18) !important;
    }

    .st-key-main_compose_send .stButton > button:hover,
    .st-key-listener_compose_send .stButton > button:hover,
    .st-key-producer_compose_send .stButton > button:hover {
        box-shadow: 0 12px 34px rgba(102, 126, 234, 0.45) !important;
        filter: brightness(1.03) !important;
        transform: none !important;
    }

    div[data-testid="stSelectbox"]:has(input[aria-label*="Gemini model"]) [data-baseweb="select"] * {
        color: rgba(235, 240, 255, 0.95) !important;
        cursor: pointer !important;
    }

    /* Compact composer (pill input + round icon buttons) */
    .st-key-producer_compose_plus button,
    .st-key-producer_compose_send button,
    .st-key-listener_compose_plus button,
    .st-key-listener_compose_send button,
    .st-key-main_compose_send button {
        width: 46px !important;
        height: 46px !important;
        min-width: 46px !important;
        min-height: 46px !important;
        padding: 0 !important;
        border-radius: 999px !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        font-size: 18px !important;
        line-height: 1 !important;
    }

    /* Chat bubbles spacing */
    div[data-testid="stChatMessage"] {
        margin-top: 0.35rem !important;
        margin-bottom: 0.35rem !important;
    }
    
    div[data-testid="stChatMessageContent"] {
        border-radius: 14px !important;
        padding: 0.85rem 1rem !important;
        line-height: 1.45 !important;
    }
    /* --- LÀM HỘP THOẠI CO DÃN VỪA VẶN THEO CHỮ --- */
    div[data-testid="stChatMessage"] {
        width: fit-content !important;
        max-width: 100% !important;
        min-width: 100px !important;
    }

    /* --- ÉP CỘT CỦA USER SANG LỀ PHẢI (BẮT CẢ BẢN MỚI LẪN CŨ) --- */
    div[data-testid="column"]:nth-child(2),
    div[data-testid="stColumn"]:nth-child(2) {
        display: flex !important;
        flex-direction: column !important;
        align-items: flex-end !important;
    }

    /* --- ĐẢO AVATAR VÀ ĐỔI MÀU NỀN CHO TIN NHẮN USER --- */
    div[data-testid="column"]:nth-child(2) div[data-testid="stChatMessage"],
    div[data-testid="stColumn"]:nth-child(2) div[data-testid="stChatMessage"] {
        flex-direction: row-reverse !important;
        margin-left: auto !important;
        margin-right: 0 !important;
    }

    /* Chỉnh lại lề cho Avatar User sau khi lật sang phải */
    div[data-testid="column"]:nth-child(2) div[data-testid="stChatMessage"] > div:first-child,
    div[data-testid="stColumn"]:nth-child(2) div[data-testid="stChatMessage"] > div:first-child {
        margin-left: 15px !important;
        margin-right: 0 !important;
    }
            
    .st-key-producer_compose_plus svg,
    .st-key-producer_compose_send svg,
    .st-key-listener_compose_plus svg,
    .st-key-listener_compose_send svg,
    .st-key-main_compose_send svg {
        width: 20px !important;
        height: 20px !important;
    }

    .st-key-producer_compose_plus button:hover,
    .st-key-producer_compose_send button:hover,
    .st-key-listener_compose_plus button:hover,
    .st-key-listener_compose_send button:hover,
    .st-key-main_compose_send button:hover {
        transform: none !important;
    }

    .st-key-producer_compose_text [data-baseweb="input"],
    .st-key-listener_compose_text [data-baseweb="input"],
    .st-key-main_compose_text [data-baseweb="input"] {
        background: rgba(48, 43, 99, 0.5) !important;
        border: none !important;
        border-radius: 999px !important;
        min-height: 46px !important;
        padding: 6px 14px !important;
        box-shadow: 0 6px 18px rgba(8, 12, 30, 0.18) !important;
    }

    .st-key-producer_compose_text input,
    .st-key-listener_compose_text input,
    .st-key-main_compose_text input {
        background: transparent !important;
        border: none !important;
        outline: none !important;
        box-shadow: none !important;
        color: rgba(235, 240, 255, 0.95) !important;
        min-height: 30px !important;
    }

    /* Remove dashed focus rings/borders that can appear on some browsers */
    .st-key-producer_compose_text input:focus,
    .st-key-producer_compose_text input:focus-visible,
    .st-key-listener_compose_text input:focus,
    .st-key-listener_compose_text input:focus-visible,
    .st-key-main_compose_text input:focus,
    .st-key-main_compose_text input:focus-visible {
        border: none !important;
        outline: none !important;
        box-shadow: none !important;
    }

    .st-key-producer_compose_text [data-baseweb="input"],
    .st-key-listener_compose_text [data-baseweb="input"],
    .st-key-main_compose_text [data-baseweb="input"] {
        border: none !important;
        outline: none !important;
    }

    /* Chat composer inputs (Home: aria-label="Main composer", Listener/Producer: aria-label="Composer") */
    div[data-testid="stTextInput"]:has(input[aria-label="Main composer"]) [data-baseweb="input"],
    div[data-testid="stTextInput"]:has(input[aria-label="Composer"]) [data-baseweb="input"] {
        background: rgba(48, 43, 99, 0.5) !important;
        border: 1px solid rgba(102, 126, 234, 0.28) !important;
        border-radius: 999px !important;
        box-shadow: 0 6px 18px rgba(8, 12, 30, 0.18) !important;
        padding: 6px 14px !important;
    }

    div[data-testid="stTextInput"]:has(input[aria-label="Main composer"]) input,
    div[data-testid="stTextInput"]:has(input[aria-label="Composer"]) input {
        background: transparent !important;
        border: none !important;
        outline: none !important;
        box-shadow: none !important;
        padding: 0 !important;
        min-height: 30px !important;
        color: rgba(235, 240, 255, 0.95) !important;
    }

    div[data-testid="stTextInput"]:has(input[aria-label="Main composer"]) input:focus,
    div[data-testid="stTextInput"]:has(input[aria-label="Main composer"]) input:focus-visible,
    div[data-testid="stTextInput"]:has(input[aria-label="Composer"]) input:focus,
    div[data-testid="stTextInput"]:has(input[aria-label="Composer"]) input:focus-visible {
        border: none !important;
        outline: none !important;
        box-shadow: none !important;
    }

    .st-key-producer_compose_text [data-baseweb="base-input"],
    .st-key-listener_compose_text [data-baseweb="base-input"],
    .st-key-main_compose_text [data-baseweb="base-input"] {
        border: none !important;
        outline: none !important;
        box-shadow: none !important;
        background: transparent !important;
    }



    .stChatMessage { background: rgba(48, 43, 99, 0.5) !important; border: 1px solid rgba(102, 126, 234, 0.3) !important; border-radius: 16px; padding: 15px; margin-bottom: 10px; backdrop-filter: blur(10px); }
    .stChatInputContainer { padding-bottom: 20px !important; }

    [data-testid="stStatusWidget"] { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important; border-radius: 16px !important; border: none !important; box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important; }
    [data-testid="stStatusWidget"] * { background: transparent !important; color: white !important;}

    [data-testid="stMetricValue"] { font-size: 28px !important; font-weight: 700 !important; }
    [data-testid="stMetricLabel"] { color: rgba(255, 255, 255, 0.7) !important; font-size: 14px !important; }

    /* Dark, soft containers for charts/tables to blend with page gradient */
    [data-testid="stDataFrame"] {
        background: rgba(12, 18, 42, 0.78) !important;
        border: 1px solid rgba(125, 143, 255, 0.22) !important;
        border-radius: 12px !important;
        box-shadow: 0 6px 18px rgba(8, 12, 30, 0.25) !important;
        overflow: hidden;
    }

    [data-testid="stDataFrame"] * {
        color: rgba(229, 235, 255, 0.92) !important;
    }

    [data-testid="stDataFrame"] [role="columnheader"] {
        background: rgba(20, 30, 64, 0.95) !important;
    }

    [data-testid="stDataFrame"] [role="gridcell"] {
        background: rgba(11, 16, 36, 0.9) !important;
        border-color: rgba(125, 143, 255, 0.12) !important;
    }

    /* Composer chat buttons: same compact size, aligned with input height */
    .st-key-chat_upload_toggle button,
    .st-key-chat_send_button button {
        width: 100% !important;
        min-width: 0 !important;
        height: 30px !important;
        min-height: 30px !important;
        padding: 0 !important;
        border-radius: 8px !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        font-size: 14px !important;
        line-height: 1 !important;
    }

    .st-key-chat_upload_toggle,
    .st-key-chat_send_button {
        display: flex;
        align-items: center;
        justify-content: flex-end;
    }

    /* Chat composer: one shared frame for input + action buttons */
    .st-key-chat_prompt {
        margin-bottom: 0 !important;
    }

    .st-key-chat_prompt [data-baseweb="input"] {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }

    .st-key-chat_prompt input {
        border: none !important;
        outline: none !important;
        background: transparent !important;
        box-shadow: none !important;
        min-height: 30px !important;
        padding: 0 6px !important;
    }

    .st-key-chat_prompt input:focus,
    .st-key-chat_prompt input:focus-visible {
        border: none !important;
        outline: none !important;
        box-shadow: none !important;
    }

    .st-key-chat_upload_toggle,
    .st-key-chat_send_button {
        margin-top: 0 !important;
    }

    /* Hide Streamlit helper text like "Press Enter to apply" for inputs (chat-like UX). */
    [data-testid="InputInstructions"] {
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    /* Hide sidebar entirely (user requested no sidebar UI). */
    [data-testid="stSidebar"],
    [data-testid="collapsedControl"],
    [data-testid="stSidebarCollapsedControl"] {
        display: none !important;
        visibility: hidden !important;
        width: 0 !important;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 4. SIDEBAR - LOAD MODELS (Cho cả 5 bài toán)
# =============================================================================
P2_NUMERIC_FEATS = []
pipeline_task1 = None
selected_model_name = ""
p1_data = None
p2_data = None
p3_data = None
p4_data = None

if 'app_mode' not in st.session_state:
    st.session_state.app_mode = None

with st.sidebar:
    st.title("Models")

    with st.expander('Trạng thái models (P0–P4)', expanded=False):
        # Help debug the common issue: running Streamlit with the wrong Python env.
        try:
            st.caption(f"Python: {sys.executable}")
            venv = str(os.getenv('VIRTUAL_ENV') or '').strip()
            if venv:
                st.caption(f"Venv: {venv}")
        except Exception:
            pass

        # Quick dependency checks (no secrets) for the 2 most frequent failures.
        try:
            try:
                from google import genai as _genai  # type: ignore

                _ = _genai  # silence linters
                st.caption("✅ Gemini SDK: google-genai OK")
            except Exception as ex:
                st.warning(f"⚠️ Gemini SDK chưa sẵn sàng ({type(ex).__name__}: {str(ex)[:120]})")
        except Exception:
            pass

        try:
            try:
                import sentence_transformers as _stx  # type: ignore

                _ = _stx
                st.caption("✅ Embedding: sentence-transformers OK")
            except Exception as ex:
                st.warning(f"⚠️ Embedding chưa sẵn sàng ({type(ex).__name__}: {str(ex)[:120]})")
        except Exception:
            pass

        try:
            expected_py = os.path.join(_REPO_ROOT, '.venv312', 'Scripts', 'python.exe')
            if os.path.exists(expected_py):
                st.code(f"{expected_py} -m streamlit run chatbot\\app_chatbot.py", language='text')
        except Exception:
            pass

        resolved = {}
        resolve_report: dict[str, str] = {}
        try:
            resolved = resolve_model_paths(
                allow_download=True,
                report=resolve_report,
                tasks=["P0", "P2", "P3", "P4"],
            )
        except Exception:
            resolved = {}

        # Quick visibility into env resolution (no secrets).
        dotenv_used = str(os.getenv('DOTENV_PATH_USED') or '').strip()
        if dotenv_used:
            st.caption(f".env: {dotenv_used}")

        # If a model is missing, show why (download/config errors).
        for key in ['P0', 'P2', 'P3', 'P4']:
            if key not in resolved:
                note = str(resolve_report.get(key) or 'missing').strip()
                st.warning(f"⚠️ {key} chưa sẵn sàng ({note})")

        # P1 is intentionally disabled in this build.
        st.caption("ℹ️ P1 (Popularity) đang tắt: SHAP chỉ chạy P0, không dự đoán P1.")

        # Load P0 (Hit)
        p0_path = str(resolved.get('P0') or '')
        if p0_path and os.path.exists(p0_path):
            try:
                data = _cached_joblib_load(p0_path)
                pipeline_task1 = data['pipeline']
                selected_model_name = data.get('model_name', 'Best Model')
                st.markdown("✅ **P0 (Hit):** Sẵn sàng")
            except Exception as ex:
                st.error(f"❌ Lỗi load P0: {ex}")

        # Load P1 (Popularity)
        p1_path = str(resolved.get('P1') or '')
        if p1_path and os.path.exists(p1_path):
            try:
                # Avoid caching P1: it can be large and caching may increase RAM usage.
                p1_candidates = [
                    p1_path,
                ]
                p1_loaded_from = ''
                last_ex = None
                for cand in p1_candidates:
                    cand = str(cand or '').strip()
                    if not cand or not os.path.exists(cand):
                        continue
                    try:
                        # Try normal load first; mmap_mode is not supported for some compressed joblib files.
                        p1_data = joblib.load(cand)
                        p1_loaded_from = cand
                        last_ex = None
                        break
                    except Exception as ex:
                        last_ex = ex
                        # Retry with mmap_mode for large numpy payloads (best-effort).
                        try:
                            p1_data = joblib.load(cand, mmap_mode='r')
                            p1_loaded_from = cand
                            last_ex = None
                            break
                        except Exception as ex2:
                            last_ex = ex2
                            continue

                if p1_data is None:
                    raise last_ex or Exception('Unknown error')
                st.markdown("✅ **P1 (Phổ biến):** Sẵn sàng")
                if p1_loaded_from and p1_loaded_from != p1_path:
                    st.caption(f"P1 loaded from: {p1_loaded_from}")
            except Exception as ex:
                st.error(f"❌ Lỗi load P1: {type(ex).__name__}: {ex}")

        # Load P2 (Clustering)
        p2_path = str(resolved.get('P2') or '')
        if p2_path and os.path.exists(p2_path):
            try:
                p2_data = _cached_joblib_load(p2_path)
                P2_NUMERIC_FEATS = p2_data.get('numeric_features', [])
                st.markdown("✅ **P2 (Cụm):** Sẵn sàng")
            except Exception as ex:
                st.error(f"❌ Lỗi load P2: {ex}")

        # Load P3 (Emotion)
        p3_path = str(resolved.get('P3') or '')
        if p3_path and os.path.exists(p3_path):
            try:
                p3_data = _cached_joblib_load(p3_path)
                st.markdown("✅ **P3 (Cảm xúc):** Sẵn sàng")
            except Exception as ex:
                st.error(f"❌ Lỗi load P3: {ex}")

        # Load P4 (Genre)
        p4_path = str(resolved.get('P4') or '')
        if p4_path and os.path.exists(p4_path):
            try:
                p4_data = _cached_joblib_load(p4_path)
                st.markdown("✅ **P4 (Thể loại):** Sẵn sàng")
            except Exception as ex:
                st.error(f"❌ Lỗi load P4: {ex}")

# =============================================================================
# 5. CÁC HÀM BỔ TRỢ XỬ LÝ DỮ LIỆU
# =============================================================================
def extract_lyrics_from_metadata(audio_path):
    if not MUTAGEN_AVAILABLE:
        return None
    try:
        audio = MP3(audio_path, ID3=ID3)
        for tag in audio.tags.values():
            if isinstance(tag, USLT):
                return tag.text
        if 'USLT::XXX' in audio.tags:
            return str(audio.tags['USLT::XXX'])
    except Exception:
        pass
    return None


def _is_truthy_env(name: str, default: str = '0') -> bool:
    value = os.getenv(name, default)
    return str(value).strip().lower() in {'1', 'true', 'yes', 'on'}


def _asr_transcribe_lyrics_from_audio(audio_path: str, audio_bytes: bytes | None = None):
    """Best-effort ASR: returns (lyrics_text, meta_dict). Never raises."""
    if not _is_truthy_env('ASR_ENABLED', default='0'):
        return '', {'source': 'asr-disabled', 'error': None}

    try:
        from faster_whisper import WhisperModel  # pyright: ignore[reportMissingImports]
    except Exception as ex:
        return '', {'source': 'asr-missing-dependency', 'error': str(ex)}

    try:
        model_size = os.getenv('ASR_MODEL_SIZE', 'small')
        language = os.getenv('ASR_LANGUAGE', 'vi')
        compute_type = os.getenv('ASR_COMPUTE_TYPE', 'int8')
        device = os.getenv('ASR_DEVICE', 'cpu')

        # Cache by audio hash within the session to avoid re-transcribing on reruns.
        cache = st.session_state.get('asr_cache')
        if cache is None:
            cache = {}
            st.session_state.asr_cache = cache

        cache_key = None
        if audio_bytes:
            cache_key = hashlib.sha256(audio_bytes).hexdigest()
            cached = cache.get(cache_key)
            if isinstance(cached, str) and cached.strip():
                return cached, {'source': 'asr-cache', 'error': None, 'model': model_size}

        # Load model (kept in a lightweight session cache to reduce repeated init).
        model_key = f'asr_model::{device}::{compute_type}::{model_size}'
        model = st.session_state.get(model_key)
        if model is None:
            model = WhisperModel(model_size, device=device, compute_type=compute_type)
            st.session_state[model_key] = model

        segments, info = model.transcribe(
            audio_path,
            language=language or None,
            vad_filter=True,
            beam_size=5,
        )

        parts = []
        for seg in segments:
            text = getattr(seg, 'text', '')
            if text:
                parts.append(str(text).strip())

        lyrics = re.sub(r"\s+", " ", " ".join(parts)).strip()
        if cache_key and lyrics:
            cache[cache_key] = lyrics

        detected_lang = getattr(info, 'language', None) if info is not None else None
        return lyrics, {
            'source': 'asr-faster-whisper',
            'error': None,
            'model': model_size,
            'language': language or detected_lang,
        }
    except Exception as ex:
        return '', {'source': 'asr-error', 'error': str(ex)}


def create_input_df(features_dict, task='P0'):
    """Tạo DataFrame với đầy đủ các cột (kể cả dummy features) để model không báo lỗi thiếu biến"""
    df = pd.DataFrame([features_dict])

    # 1. Đảm bảo có đủ 2 cột text cho TF-IDF
    if 'lyric' not in df.columns:
        df['lyric'] = df.get('clean_lyric', "")
    if 'clean_lyric' not in df.columns:
        df['clean_lyric'] = df.get('lyric', "")

    # 2. Xử lý Sentiment (Bắt buộc phải có để OneHotEncoder chạy được)
    if 'final_sentiment' not in df.columns:
        df['final_sentiment'] = 'neutral'

    # 3. Trám các biến NLP (Nếu người dùng không nhập lời)
    for col in ['lyric_total_words', 'lexical_diversity', 'noun_count', 'verb_count', 'adj_count']:
        if col not in df.columns:
            df[col] = 0.0

    # 4. Tính toán các Engineered Features bắt buộc của P1
    duration = max(df.get('duration_sec', pd.Series([1.0])).iloc[0], 1.0)
    tempo = df.get('tempo_bpm', pd.Series([0.0])).iloc[0]
    beat_strength = df.get('beat_strength_mean', pd.Series([0.5])).iloc[0]

    if 'words_per_second' not in df.columns:
        df['words_per_second'] = df.get('lyric_total_words', pd.Series([0])).iloc[0] / duration
    if 'rhythmic_impact' not in df.columns:
        df['rhythmic_impact'] = tempo * beat_strength

    # contrast_range đã có từ Librosa, nếu thiếu thì mặc định 0
    if 'contrast_range' not in df.columns:
        df['contrast_range'] = 0.0
    if 'sentiment_intensity' not in df.columns:
        df['sentiment_intensity'] = 1.0
    if 'score_lexicon' not in df.columns:
        df['score_lexicon'] = 0.0

    return df


# =============================================================================
# 6. HÀM KHUNG TÍCH HỢP NGOẠI VI (PLACEHOLDER)
# =============================================================================
def save_uploaded_file(uploaded_file, suffix=".wav"):
    """Lưu file upload tạm thời và trả về đường dẫn để các analyzer/model có thể đọc."""
    if uploaded_file is None:
        raise ValueError('uploaded_file is None')

    file_bytes = None
    if isinstance(uploaded_file, (bytes, bytearray)):
        file_bytes = bytes(uploaded_file)
    elif hasattr(uploaded_file, 'getbuffer'):
        file_bytes = bytes(uploaded_file.getbuffer())
    elif hasattr(uploaded_file, 'getvalue'):
        file_bytes = bytes(uploaded_file.getvalue())
    elif hasattr(uploaded_file, 'read'):
        file_bytes = uploaded_file.read()
    else:
        raise TypeError(f'Unsupported uploaded_file type: {type(uploaded_file)}')

    temp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp.write(file_bytes)
    temp.close()
    return temp.name


def _audio_mime_from_name(file_name):
    ext = os.path.splitext(str(file_name or ''))[-1].lower()
    if ext == '.mp3':
        return 'audio/mpeg'
    if ext == '.wav':
        return 'audio/wav'
    return 'audio/wav'


def _clean_optional_text(value):
    """Chuan hoa text optional tu LLM, tra ve None neu gia tri khong hop le."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.lower() in {'null', 'none', 'khong ro', 'không rõ', 'unknown', 'n/a'}:
        return None
    return text


def _map_emotion_to_mood(emotion_text):
    """Map nhan cam xuc (VN) sang mood noi bo de query Supabase."""
    if not emotion_text:
        return 'neutral'
    normalized = _normalize_text(emotion_text)
    if normalized in {'tich cuc', 'positive'}:
        return 'energetic'
    if normalized in {'tieu cuc', 'negative'}:
        return 'sad'
    return 'neutral'


def _normalize_mood_token(raw_mood: str) -> str:
    """Normalize Vietnamese/English mood strings into internal query labels."""

    mood = _normalize_text(raw_mood)
    if not mood:
        return 'neutral'

    sad_tokens = {
        'buon', 'sad', 'suy', 'tieu cuc', 'negative', 'melancholy', 'depressed', 'that tinh', 'broken', 'heartbreak'
    }
    energetic_tokens = {
        'vui', 'happy', 'quay', 'quay tung', 'quay het minh', 'energetic', 'dance', 'party', 'soi dong', 'tich cuc', 'positive'
    }
    chill_tokens = {'chill', 'thu gian', 'relax', 'binh yen', 'healing', 'acoustic', 'lofi', 'lo-fi'}

    if any(t in mood for t in sad_tokens):
        return 'sad'
    if any(t in mood for t in energetic_tokens):
        return 'energetic'
    if any(t in mood for t in chill_tokens):
        return 'neutral'
    return 'neutral'


def _enrich_intent_for_query(intent_json):
    """Bo sung cac truong query_mood/query_keywords tu intent (emotion/style/genre/tempo/context)."""
    emotion = _clean_optional_text(intent_json.get('emotion'))
    style = _clean_optional_text(intent_json.get('style'))
    genre = _clean_optional_text(intent_json.get('genre'))
    tempo = _clean_optional_text(intent_json.get('tempo'))
    context = _clean_optional_text(intent_json.get('context'))

    query_mood = str(intent_json.get('mood', '')).strip().lower()
    if not query_mood:
        query_mood = _map_emotion_to_mood(emotion)
    query_mood = _normalize_mood_token(query_mood)

    keyword_bucket = []
    for k in intent_json.get('keywords', []) or []:
        k_text = _clean_optional_text(k)
        if k_text:
            keyword_bucket.append(k_text)
    if style:
        keyword_bucket.append(style)
    if genre:
        keyword_bucket.append(genre)
    if tempo:
        keyword_bucket.append(tempo)
    if context:
        keyword_bucket.append(context)

    # Loai trung lap keyword theo compare khong dau/khong hoa thuong.
    dedup = []
    seen = set()
    for token in keyword_bucket:
        key = _normalize_text(token)
        if key and key not in seen:
            dedup.append(token)
            seen.add(key)

    enriched = dict(intent_json)
    enriched['emotion'] = emotion
    enriched['style'] = style
    enriched['genre'] = genre
    enriched['tempo'] = tempo
    enriched['context'] = context
    enriched['mood'] = query_mood or 'neutral'
    enriched['keywords'] = dedup[:8]

    # Provide a default `query_text` for RPCs that expect a text query.
    # (Many Supabase RPCs compute/lookup embeddings from this string.)
    existing_query_text = str(intent_json.get('query_text', '') or '').strip()
    if existing_query_text:
        enriched['query_text'] = existing_query_text
    else:
        enriched['query_text'] = ' '.join([enriched['mood']] + enriched['keywords']).strip()
    return enriched

def _build_supabase_fallback(intent_json):
    """Tạo dữ liệu gợi ý mẫu khi chưa kết nối DB thật hoặc truy vấn thất bại."""
    enriched_intent = _enrich_intent_for_query(intent_json)
    mood = enriched_intent.get('mood', 'neutral')
    sample_db = [
        {'title': 'Bai Nay Chill Phet', 'artist': 'DEN ft. MIN', 'score': 0.93, 'spotify_id': '3nQNiWdeP6z4xj6Vxk7I9d'},
        {'title': 'See Tinh', 'artist': 'Hoang Thuy Linh', 'score': 0.91, 'spotify_id': '5A8q8kW5Bm3f6fQ4W9D8nF'},
        {'title': 'Hai Phut Hon', 'artist': 'Phao', 'score': 0.89, 'spotify_id': '2H7PHVdQ3mXqEHXcvclTB0'},
        {'title': 'Mascara', 'artist': 'Chillies', 'score': 0.86, 'spotify_id': '1o4wo2mqkhfILou3NFGnke'},
        {'title': 'Ngay Mai Em Di', 'artist': 'Leloi x Soobin', 'score': 0.84, 'spotify_id': '3B4X8v6Gf0yIh5o4xQ7TRS'},
    ]
    if mood == 'sad':
        sample_db[0]['title'] = 'Co Chang Trai Viet Len Cay'
    if mood == 'energetic':
        sample_db[1]['title'] = 'De Mi Noi Cho Ma Nghe'
    return sample_db[: intent_json.get('top_k', 5)]


def _normalize_supabase_rows(rows):
    """
    Chuẩn hóa dữ liệu trả về từ RPC match_vpop_tracks.
    Map chính xác các cột từ database sang object dùng trong App.

    Output:
      - spotify_id: str
      - title: str
      - artist: str
      - score: float (0..1)  -> dùng nội bộ để tương thích code cũ
      - similarity: float (0..100) -> hiển thị % cho người dùng
    """

    normalized_data = []
    if not rows:
        return normalized_data

    for idx, item in enumerate(rows):
        if not isinstance(item, dict):
            continue

        spotify_id = item.get('spotify_id')
        if not spotify_id:
            spotify_id = item.get('spotify_track_id') or item.get('track_id') or ''

        title = item.get('title') or item.get('song_title') or item.get('track_name') or f'Không rõ tên bài #{idx + 1}'
        artist = item.get('artist') or item.get('artist_name') or 'Không rõ ca sĩ'

        raw_similarity = item.get('similarity', None)
        if raw_similarity is None:
            raw_similarity = item.get('score', None)
        if raw_similarity is None:
            raw_similarity = 0.0

        try:
            raw_similarity = float(raw_similarity)
        except Exception:
            raw_similarity = 0.0

        # Chuẩn hóa: score luôn là 0..1; similarity luôn là % 0..100
        if 0.0 <= raw_similarity <= 1.0:
            score = raw_similarity
            similarity_pct = raw_similarity * 100.0
        elif 0.0 <= raw_similarity <= 100.0:
            similarity_pct = raw_similarity
            score = raw_similarity / 100.0
        else:
            score = 0.0
            similarity_pct = 0.0

        normalized_data.append(
            {
                'spotify_id': str(spotify_id),
                'title': str(title),
                'artist': str(artist),
                'score': float(score),
                'similarity': round(float(similarity_pct), 2),
                # THÊM 2 DÒNG NÀY:
                'vibe': item.get('vibe') or '',
                'main_topic': item.get('main_topic') or ''
            }
        )
    return normalized_data


def _get_supabase_client():
    """Compatibility wrapper (client cached inside `chatbot.supabase`)."""

    return _supabase_get_client()


def query_supabase_lyrics(user_input: str, match_threshold: float = 0.4, match_count: int = 5):
    """Compatibility wrapper (implementation in `chatbot.supabase`)."""

    return _supabase_query_lyrics(user_input, match_threshold=match_threshold, match_count=match_count)


def query_supabase_vector(intent_json):
    """Compatibility wrapper (implementation in `chatbot.supabase`)."""

    try:
        return _supabase_query_vector(
            intent_json,
            build_fallback=_build_supabase_fallback,
            normalize_rows=_normalize_supabase_rows,
            enrich_intent=_enrich_intent_for_query,
        )
    except Exception as ex:
        return {
            'tracks': _build_supabase_fallback(intent_json),
            'source': 'fallback-query-error',
            'error': str(ex),
        }


def _spotify_access_token():
    return spotify_access_token()


def _spotify_search_link(title='', artist=''):
    query = ' '.join([str(title or '').strip(), str(artist or '').strip()]).strip()
    if not query:
        return ''
    return f"https://open.spotify.com/search/{urllib.parse.quote(query)}"


def _spotify_api_get_json(url: str, *, timeout: int = 12) -> dict | None:
    return spotify_api_get_json(url, timeout=timeout)


def _spotify_pick_image_url(images: object, *, prefer_width: int = 320) -> str:
    return spotify_pick_image_url(images, prefer_width=prefer_width)


def _format_int_vi(value: int | None) -> str:
    if value is None:
        return ''
    try:
        return f"{int(value):,}".replace(',', '.')
    except Exception:
        return str(value)


def _format_duration_ms(duration_ms: int | None) -> str:
    if duration_ms is None:
        return ''
    try:
        total_seconds = int(duration_ms) // 1000
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        return f"{minutes:d}:{seconds:02d}"
    except Exception:
        return ''


def _render_spotify_artist_payload(payload: dict) -> None:
    return render_spotify_artist_payload(payload)


def spotify_search_artist(artist_name: str, *, market: str = 'VN') -> dict | None:
    return spotify_search_artist_api(artist_name, market=market)


def spotify_get_artist_top_tracks(artist_id: str, *, market: str = 'VN', limit: int = 5) -> list[dict]:
    return spotify_get_artist_top_tracks_api(artist_id, market=market, limit=limit)


def spotify_get_track_metadata(track_id: str) -> dict | None:
    return spotify_get_track_metadata_api(track_id)


def spotify_get_tracks_metadata(track_ids: list[str], *, batch_size: int = 5) -> dict[str, dict]:
    return spotify_get_tracks_metadata_api(track_ids, batch_size=batch_size)


def _build_track_previews_from_spotify_batch(top_tracks: list[dict], *, batch_size: int = 5) -> tuple[list[dict], dict[str, int]]:
    """Build track previews using 1 Spotify call per `batch_size` tracks."""
    # --- THÊM HÀM CACHE NÀY NGAY TRÊN HÀM BÊN DƯỚI ---
@st.cache_data(ttl=3600, show_spinner=False)  # Cache lưu kết quả trong 1 tiếng (3600s)
def _fetch_spotify_payloads_cached(valid_ids_tuple: tuple) -> dict:
    try:
        return spotify_get_tracks_metadata(list(valid_ids_tuple), batch_size=len(valid_ids_tuple)) or {}
    except Exception:
        return {}

def _build_track_previews_from_spotify_batch(top_tracks: list[dict], *, batch_size: int = 5) -> tuple[list[dict], dict[str, int]]:
    """Build track previews using 1 Spotify call per `batch_size` tracks."""

    if not isinstance(top_tracks, list) or not top_tracks:
        return [], {}

    valid_ids: list[str] = []
    for t in top_tracks[:batch_size]:
        tid = str((t or {}).get('spotify_id') or '').strip()
        if tid and re.fullmatch(r'[A-Za-z0-9]{22}', tid):
            valid_ids.append(tid)

    payloads: dict[str, dict] = {}
    if valid_ids:
        # Gọi hàm Cache thay vì gọi thẳng API
        payloads = _fetch_spotify_payloads_cached(tuple(valid_ids))

    popularity_by_id: dict[str, int] = {}
    for tid, p in payloads.items():
        try:
            popularity_by_id[str(tid)] = int((p or {}).get('popularity', -1))
        except Exception:
            popularity_by_id[str(tid)] = -1

    previews: list[dict] = []
    # (Khối xử lý vòng lặp bên dưới giữ nguyên y xì của bạn nhé...)
    for t in top_tracks[:batch_size]:
        title = str((t or {}).get('title') or '')
        artist = str((t or {}).get('artist') or '')
        track_id = str((t or {}).get('spotify_id') or '').strip()

    if not isinstance(top_tracks, list) or not top_tracks:
        return [], {}

    valid_ids: list[str] = []
    for t in top_tracks[:batch_size]:
        tid = str((t or {}).get('spotify_id') or '').strip()
        if tid and re.fullmatch(r'[A-Za-z0-9]{22}', tid):
            valid_ids.append(tid)

    payloads: dict[str, dict] = {}
    try:
        if valid_ids:
            payloads = spotify_get_tracks_metadata(valid_ids, batch_size=batch_size) or {}
    except Exception:
        payloads = {}

    popularity_by_id: dict[str, int] = {}
    for tid, p in payloads.items():
        try:
            popularity_by_id[str(tid)] = int((p or {}).get('popularity', -1))
        except Exception:
            popularity_by_id[str(tid)] = -1

    previews: list[dict] = []
    for t in top_tracks[:batch_size]:
        title = str((t or {}).get('title') or '')
        artist = str((t or {}).get('artist') or '')
        track_id = str((t or {}).get('spotify_id') or '').strip()

        search_link = _spotify_search_link(title, artist)
        base_link = f'https://open.spotify.com/track/{track_id}' if re.fullmatch(r'[A-Za-z0-9]{22}', track_id or '') else search_link

        sp = payloads.get(track_id) if track_id else None
        if isinstance(sp, dict) and sp.get('id'):
            external = (((sp.get('external_urls') or {}) if isinstance(sp.get('external_urls'), dict) else {}) or {}).get('spotify') or base_link
            preview = sp.get('preview_url') or external
            previews.append(
                {
                    'title': title,
                    'artist': artist,
                    'spotify_id': track_id,
                    'preview_url': str(preview or ''),
                    'external_url': str(external or ''),
                    'preview_source': 'spotify-tracks-batch',
                    'popularity': popularity_by_id.get(track_id, -1),
                }
            )
        else:
            previews.append(
                {
                    'title': title,
                    'artist': artist,
                    'spotify_id': track_id,
                    'preview_url': str(base_link or search_link),
                    'external_url': str(base_link or search_link),
                    'preview_source': 'fallback-open-or-search',
                    'popularity': -1,
                }
            )

    return previews, popularity_by_id


def _aligned_chat_col(role: str):
    role = str(role or 'assistant')
    if role == 'user':
        cols = st.columns([0.15, 0.85], gap='small')
        return cols[1]
    cols = st.columns([0.85, 0.15], gap='small')
    return cols[0]


def get_spotify_preview(spotify_track_id, title='', artist=''):
    """
    Trả link nghe thử Spotify.
    Ưu tiên gọi API thật nếu có credentials; fallback về open link nếu không có token/id.
    """
    track_id = str(spotify_track_id or '').strip()
    is_track_id_valid = bool(re.fullmatch(r'[A-Za-z0-9]{22}', track_id))
    search_link = _spotify_search_link(title, artist)
    base_link = f'https://open.spotify.com/track/{track_id}' if is_track_id_valid else search_link

    if not track_id:
        return {
            'preview_url': search_link,
            'external_url': search_link,
            'source': 'fallback-search-no-track-id',
        }

    if not is_track_id_valid:
        return {
            'preview_url': search_link,
            'external_url': search_link,
            'source': 'fallback-search-invalid-track-id',
        }

    token = None
    try:
        token = _spotify_access_token()
    except Exception:
        token = None

    if not token:
        return {
            'preview_url': base_link,
            'external_url': base_link,
            'source': 'fallback-no-token',
        }

    try:
        req = urllib.request.Request(
            url=f'https://api.spotify.com/v1/tracks/{track_id}',
            headers={'Authorization': f'Bearer {token}'},
            method='GET',
        )
        with urllib.request.urlopen(req, timeout=12) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            external = data.get('external_urls', {}).get('spotify', base_link)
            preview = data.get('preview_url') or external
            return {
                'preview_url': preview,
                'external_url': external,
                'source': 'live-spotify-api',
            }
    except Exception:
        return {
            'preview_url': base_link or search_link,
            'external_url': base_link or search_link,
            'source': 'fallback-track-open-or-search-url',
        }


def _is_direct_audio_url(url: str) -> bool:
    url = str(url or '').strip().lower()
    if not url.startswith('http'):
        return False
    if url.endswith('.mp3'):
        return True
    if url.endswith('.m4a') or url.endswith('.mp4'):
        return True
    # Spotify preview URLs often contain "mp3-preview".
    if 'mp3-preview' in url:
        return True
    return False


def _mime_from_audio_url(url: str) -> str:
    url = str(url or '').strip().lower()
    if url.endswith('.mp3') or 'mp3-preview' in url:
        return 'audio/mpeg'
    if url.endswith('.m4a') or url.endswith('.mp4'):
        return 'audio/mp4'
    return 'audio/mpeg'


def _norm_preview_query(text: str) -> str:
    text = unicodedata.normalize('NFKC', str(text or '')).strip().lower()
    text = re.sub(r'\s+', ' ', text)
    return text


def _extract_spotify_track_id(url: str) -> str | None:
    u = str(url or '').strip()
    if not u:
        return None
    m = re.search(r"(?:open\.spotify\.com/track/|spotify:track:)([A-Za-z0-9]{22})", u)
    return m.group(1) if m else None


def _render_spotify_embed(track_id: str) -> None:
    tid = str(track_id or '').strip()
    if not tid:
        return
    # Spotify embed player (matches the card UI in the screenshot).
    try:
        # CÁCH MỚI: Dùng thẳng st.iframe hoặc st.components.v1 tùy phiên bản để tắt warning
        if hasattr(st, 'iframe'):
            st.iframe(
                f"https://open.spotify.com/embed/track/{tid}",
                height=152,
                scrolling=False,
            )
        else:
            import streamlit.components.v1 as components
            components.iframe(
                f"https://open.spotify.com/embed/track/{tid}",
                height=152,
                scrolling=False,
            )
        return
    except Exception:
        # Fallback an toàn tuyệt đối bằng HTML thuần (Không bao giờ bị warning)
        st.markdown(
            (
                '<iframe '
                f'src="https://open.spotify.com/embed/track/{tid}" '
                'width="100%" height="152" frameborder="0" '
                'allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture" '
                'loading="lazy"></iframe>'
            ),
            unsafe_allow_html=True,
        )


@lru_cache(maxsize=2048)
def _deezer_find_preview_url_cached(*, title: str, artist: str) -> str | None:
    """Return a Deezer 30s preview URL if available (public API, no auth)."""
    try:
        q = f'artist:"{artist}" track:"{title}"'
        url = 'https://api.deezer.com/search?q=' + urllib.parse.quote(q) + '&limit=1'
        req = urllib.request.Request(url=url, headers={'User-Agent': 'Mozilla/5.0'}, method='GET')
        with urllib.request.urlopen(req, timeout=4) as resp:
            data = json.loads(resp.read().decode('utf-8'))
        rows = data.get('data') or []
        if not rows:
            return None
        preview = (rows[0] or {}).get('preview')
        return str(preview).strip() if preview else None
    except Exception:
        return None


def _deezer_find_preview_url(*, title: str, artist: str) -> str | None:
    return _deezer_find_preview_url_cached(
        title=_norm_preview_query(title),
        artist=_norm_preview_query(artist),
    )


@lru_cache(maxsize=2048)
def _itunes_find_preview_url_cached(*, title: str, artist: str) -> str | None:
    """Return an iTunes 30s previewUrl if available (public API, no auth)."""
    try:
        term = f'{title} {artist}'.strip()
        url = (
            'https://itunes.apple.com/search?term='
            + urllib.parse.quote(term)
            + '&media=music&entity=song&limit=1'
        )
        req = urllib.request.Request(url=url, headers={'User-Agent': 'Mozilla/5.0'}, method='GET')
        with urllib.request.urlopen(req, timeout=4) as resp:
            data = json.loads(resp.read().decode('utf-8'))
        results = data.get('results') or []
        if not results:
            return None
        preview = (results[0] or {}).get('previewUrl')
        return str(preview).strip() if preview else None
    except Exception:
        return None


def _itunes_find_preview_url(*, title: str, artist: str) -> str | None:
    return _itunes_find_preview_url_cached(
        title=_norm_preview_query(title),
        artist=_norm_preview_query(artist),
    )


def _fetch_preview_audio_bytes(*, url: str, max_bytes: int = 3_000_000) -> bytes | None:
    """Download remote preview audio into bytes, with session cache and size cap."""
    url = str(url or '').strip()
    if not url:
        return None

    cache = st.session_state.get('preview_audio_cache')
    if not isinstance(cache, dict):
        cache = {}
        st.session_state.preview_audio_cache = cache

    cached = cache.get(url)
    if isinstance(cached, (bytes, bytearray)) and len(cached) > 0:
        return bytes(cached)

    try:
        req = urllib.request.Request(url=url, headers={'User-Agent': 'Mozilla/5.0'}, method='GET')
        with urllib.request.urlopen(req, timeout=12) as resp:
            payload = resp.read(max_bytes + 1)
        if not payload or len(payload) > max_bytes:
            return None
        cache[url] = payload
        st.session_state.preview_audio_cache = cache
        return payload
    except Exception:
        return None


def _resolve_playable_preview(
    *,
    spotify_id: str,
    title: str,
    artist: str,
    spotify_preview_url: str | None = None,
    spotify_external_url: str | None = None,
    spotify_source: str | None = None,
) -> dict:
    """Resolve a playable ~30s preview URL for Streamlit audio.

    Priority:
    1) Spotify API preview_url (direct mp3) if available
    2) Deezer 30s preview (direct mp3)
    3) iTunes previewUrl (usually m4a)
    """
    if spotify_preview_url is None or spotify_external_url is None:
        spotify = get_spotify_preview(spotify_id, title=title, artist=artist)
        spotify_preview = spotify.get('preview_url')
        external_url = spotify.get('external_url') or spotify_preview
        spotify_source = spotify.get('source')
    else:
        spotify_preview = spotify_preview_url
        external_url = spotify_external_url or spotify_preview

    if _is_direct_audio_url(str(spotify_preview)):
        return {
            'preview_url': str(spotify_preview),
            'preview_source': f"spotify::{spotify_source or 'unknown'}",
            'external_url': str(external_url or ''),
            'mime': _mime_from_audio_url(str(spotify_preview)),
        }

    deezer_url = _deezer_find_preview_url(title=title, artist=artist)
    if _is_direct_audio_url(str(deezer_url)):
        return {
            'preview_url': str(deezer_url),
            'preview_source': 'deezer-search',
            'external_url': str(external_url or ''),
            'mime': _mime_from_audio_url(str(deezer_url)),
        }

    itunes_url = _itunes_find_preview_url(title=title, artist=artist)
    if _is_direct_audio_url(str(itunes_url)):
        return {
            'preview_url': str(itunes_url),
            'preview_source': 'itunes-search',
            'external_url': str(external_url or ''),
            'mime': _mime_from_audio_url(str(itunes_url)),
        }

    return {
        'preview_url': '',
        'preview_source': 'none',
        'external_url': str(external_url or ''),
        'mime': 'audio/mpeg',
    }


def _render_track_previews(track_previews: list[dict]) -> None:
    """Render playable previews (30s) inside a chat message."""
    if not track_previews:
        return

    for idx, item in enumerate(track_previews, start=1):
        title = str(item.get('title') or '')
        artist = str(item.get('artist') or '')
        preview_url = str(item.get('preview_url') or '')
        external_url = str(item.get('external_url') or '')
        mime = str(item.get('mime') or 'audio/mpeg')
        source = str(item.get('preview_source') or '')
        # --- LẤY THÊM THÔNG TIN MỚI ---
        vibe = str(item.get('vibe') or '').strip()
        topic = str(item.get('main_topic') or '').strip()

        # Card-like layout to match the premium feel of Spotify embed cards.
        with st.container(border=True):
            header = f"{idx}. {title}".strip()
            if header.endswith('.'):
                header = header[:-1]
            st.markdown(f"**{header}**")
            meta_parts = []
            if artist:
                meta_parts.append(artist)
            if vibe:
                meta_parts.append(f"{vibe}")
            if topic:
                meta_parts.append(f"Chủ đề: {topic}")
            
            if meta_parts:
                st.caption(" • ".join(meta_parts))

            if preview_url and _is_direct_audio_url(preview_url):
                # Prefer URL playback (no server-side download) to keep the app responsive.
                st.audio(preview_url, format=mime)
                if source and source != 'none':
                    st.caption(f"Preview source: {source}")
            else:
                # Fallback: Spotify embed player (playable even when preview_url is missing).
                spotify_id = str(item.get('spotify_id') or '').strip()
                track_id = spotify_id if re.fullmatch(r'[A-Za-z0-9]{22}', spotify_id) else _extract_spotify_track_id(external_url)
                if track_id:
                    _render_spotify_embed(track_id)
                else:
                    if external_url:
                        st.caption(f"Không có preview 30s. Mở link: {external_url}")
                    else:
                        st.caption('Không có preview 30s cho bài này.')


def _autoscroll_to_latest_chat() -> None:
    """Best-effort auto scroll to the latest chat bubble."""
    try:
        # CÁCH MỚI: Dùng st.markdown thay vì st.components.v1.html để không bị Warning
        st.markdown(
            """
            <script>
                (function() {
                    try {
                        const doc = window.parent.document;
                        const nodes = doc.querySelectorAll('div[data-testid="stChatMessage"]');
                        if (nodes && nodes.length) {
                            nodes[nodes.length - 1].scrollIntoView({block: 'end'});
                        } else {
                            window.parent.scrollTo(0, doc.body.scrollHeight);
                        }
                    } catch (e) {
                    }
                })();
            </script>
            """,
            unsafe_allow_html=True,
        )
    except Exception:
        return


def intent_json_to_markdown(intent_json):
    """Hiển thị JSON intent đẹp mắt trong khung chat."""
    return json.dumps(intent_json, ensure_ascii=False, indent=2)


def _format_llm_advice_output(text):
    """Chuẩn hóa output text của LLM trước khi render markdown."""
    if not text:
        return 'Chưa có lời khuyên từ LLM.'
    return text.strip()


def _format_recent_history_for_llm(*, module: str, limit: int = 5) -> str:
    module = str(module or '').strip()
    if not module:
        return ''

    rows: list[dict] = []
    try:
        rows = _supabase_fetch_recent_chat_history(
            session_id=_get_or_create_session_id(),
            module=module,
            limit=max(1, int(limit)),
        )
    except Exception:
        rows = []

    lines: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        role = str(row.get('role') or '').strip() or 'user'
        content = str(row.get('content') or '').strip()
        if not content:
            continue
        if len(content) > 500:
            content = content[:500] + '…'
        lines.append(f"{role}: {content}")

    if not lines:
        return ''
    return "Recent chat history (last 5 messages):\n" + "\n".join(lines)


def call_gemini_engine(prompt, *, module: str | None = None, system_prefix: str | None = None):
    """Khung gọi Gemini free API (nếu đã cấu hình GEMINI_API_KEY)."""
    def _parse_gemini_api_keys() -> list[str]:
        keys: list[str] = []
        raw = str(os.getenv('GEMINI_API_KEYS') or '').strip()
        if raw:
            for part in re.split(r"[\n,;]+", raw):
                part = part.strip()
                if part and part not in keys:
                    keys.append(part)

        single = str(os.getenv('GEMINI_API_KEY') or '').strip()
        if single and single not in keys:
            keys.append(single)

        for i in range(1, 21):
            v = str(os.getenv(f'GEMINI_API_KEY_{i}') or '').strip()
            if v and v not in keys:
                keys.append(v)
        return keys

    def _looks_like_quota_error(err: Exception) -> bool:
        msg = str(err).lower()
        return (
            'resource_exhausted' in msg
            or 'quota' in msg
            or '429' in msg
            or 'rate limit' in msg
            or 'rate-limit' in msg
        )

    def _looks_like_model_not_found_error(err: Exception) -> bool:
        msg = str(err).lower()
        return (
            '404' in msg
            or 'not_found' in msg
            or 'no longer available' in msg
            or ('this model' in msg and 'available' in msg)
        )

    def _normalize_model_name(name: str) -> str:
        name = str(name or '').strip()
        if name.startswith('models/'):
            name = name[len('models/') :]
        return name

    api_keys = _parse_gemini_api_keys()
    # Ưu tiên model override từ UI.
    override_model = _normalize_model_name(str(st.session_state.get('gemini_model_override') or '').strip())
    model_name = 'gemini-2.5-flash'

    if not api_keys:
        st.session_state['gemini_last_error'] = 'GEMINI_API_KEY / GEMINI_API_KEYS is empty'
        return None

    # Per-key cooldown: rotate keys when one hits 429/quota.
    try:
        now = float(time.time())
    except Exception:
        now = 0.0
    try:
        cooldowns = st.session_state.get('gemini_key_cooldown_until')
        if not isinstance(cooldowns, dict):
            cooldowns = {}
    except Exception:
        cooldowns = {}

    def _fp(key: str) -> str:
        try:
            return hashlib.sha256(str(key).encode('utf-8')).hexdigest()[:16]
        except Exception:
            return 'unknown'

    eligible_keys: list[str] = []
    min_remaining: int | None = None
    for k in api_keys:
        until = float(cooldowns.get(_fp(k), 0.0) or 0.0)
        if until and now and now < until:
            remaining = int(until - now)
            if min_remaining is None or remaining < min_remaining:
                min_remaining = remaining
            continue
        eligible_keys.append(k)

    if not eligible_keys:
        st.session_state['gemini_key_cooldown_until'] = cooldowns
        st.session_state['gemini_last_error'] = f'quota-cooldown:{int(min_remaining or 0)}s'
        return None

    # Model candidates: allow overrides, but remain robust to deprecations (404).
    env_model = _normalize_model_name(str(os.getenv('GEMINI_MODEL') or '').strip())
    model_candidates: list[str] = []
    for m in [
        override_model,
        env_model,
        model_name,
        'gemini-2.0-flash',

    ]:
        m = _normalize_model_name(m)
        if m and m not in model_candidates:
            model_candidates.append(m)

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

    def _discover_models(client: object, fp: str) -> list[str]:
        # Cache in session_state for 30 minutes to avoid repeated API calls.
        try:
            cache = st.session_state.get('gemini_model_candidate_cache')
            if not isinstance(cache, dict):
                cache = {}
        except Exception:
            cache = {}

        try:
            now2 = float(time.time())
        except Exception:
            now2 = 0.0

        cached = cache.get(fp)
        if isinstance(cached, dict):
            ts = float(cached.get('ts') or 0.0)
            models = cached.get('models')
            if now2 and ts and now2 - ts < 30 * 60 and isinstance(models, list) and models:
                return [str(m) for m in models if str(m).strip()]

        try:
            models_api = getattr(client, 'models', None)
            if models_api is None or not hasattr(models_api, 'list'):
                return []
            list_result = models_api.list()
            names = _iter_model_names(list_result)
            names = [n for n in names if 'gemini' in n.lower() and 'embedding' not in n.lower()]
            names = sorted(names, key=_rank_model_name)
            picked = names[:8]
            cache[fp] = {'ts': now2, 'models': picked}
            st.session_state['gemini_model_candidate_cache'] = cache
            return picked
        except Exception:
            st.session_state['gemini_model_candidate_cache'] = cache
            return []

    # Add lightweight conversational context (optional) so responses stay consistent.
    prompt_text = str(prompt or '')
    if module:
        history_block = _format_recent_history_for_llm(module=str(module), limit=5)
        if history_block:
            prompt_text = f"{history_block}\n\n---\n\n{prompt_text}"

    # System prefix (persona). Default keeps existing behavior.
    prefix = system_prefix
    if prefix is None:
        prefix = (
            "Bạn là trợ lý âm nhạc V-Pop. Luôn trả lời bằng tiếng Việt, ngắn gọn, lịch sự. "
            "Nếu thiếu dữ liệu thì nói rõ và đưa gợi ý tiếp theo."
        )
    prefix = str(prefix or '').strip()
    if prefix:
        prompt_text = f"{prefix}\n\n{prompt_text}"

    last_error = None
    try:
        from google import genai  # type: ignore

        for api_key in eligible_keys:
            try:
                client = genai.Client(api_key=api_key)
            except Exception as ex:
                last_error = str(ex)
                continue

            saw_model_not_found = False
            saw_quota = False
            for candidate in model_candidates:
                try:
                    response = client.models.generate_content(
                        model=candidate,
                        contents=prompt_text,
                    )
                    text = getattr(response, 'text', None)
                    text = str(text or '').strip()
                    if text:
                        st.session_state['gemini_model_used'] = candidate
                        st.session_state['gemini_last_error'] = None
                        return text
                except Exception as ex:
                    last_error = f"{candidate}: {ex}"
                    if _looks_like_quota_error(ex):
                        saw_quota = True
                        try:
                            cooldowns[_fp(api_key)] = time.time() + 10 * 60
                            st.session_state['gemini_key_cooldown_until'] = cooldowns
                        except Exception:
                            pass
                        break
                    if _looks_like_model_not_found_error(ex):
                        # Try the next model candidate for the same key.
                        saw_model_not_found = True
                        continue

            # If configured models are deprecated (404), discover available models and retry once.
            if saw_model_not_found and not saw_quota:
                fp = _fp(api_key)
                discovered = _discover_models(client, fp)
                for candidate in discovered:
                    try:
                        response = client.models.generate_content(
                            model=candidate,
                            contents=prompt_text,
                        )
                        text = getattr(response, 'text', None)
                        text = str(text or '').strip()
                        if text:
                            st.session_state['gemini_model_used'] = candidate
                            st.session_state['gemini_last_error'] = None
                            return text
                    except Exception as ex:
                        last_error = f"{candidate}: {ex}"
                        if _looks_like_quota_error(ex):
                            try:
                                cooldowns[_fp(api_key)] = time.time() + 10 * 60
                                st.session_state['gemini_key_cooldown_until'] = cooldowns
                            except Exception:
                                pass
                            break
                        if _looks_like_model_not_found_error(ex):
                            continue
    except Exception as ex:
        last_error = str(ex)

    if last_error:
        st.session_state['gemini_last_error'] = last_error

    return None


def _normalize_handle_action_rows(rows: list[dict]) -> list[dict]:
    """Normalize rows from handle_action (table SELECT or RPC) into a stable schema."""

    out: list[dict] = []
    for r in rows or []:
        if not isinstance(r, dict):
            continue

        spotify_id = r.get('spotify_id') or r.get('spotify_track_id') or r.get('track_id')
        title = r.get('title') or r.get('song_title') or r.get('track_name')
        artist = r.get('artist') or r.get('artists') or r.get('artist_name')

        raw_similarity = r.get('similarity', None)
        if raw_similarity is None:
            raw_similarity = r.get('score', None)
        if raw_similarity is None:
            raw_similarity = 0.0

        try:
            raw_similarity = float(raw_similarity)
        except Exception:
            raw_similarity = 0.0

        # Normalize: score is always 0..1; similarity is always 0..100 for display/prompting.
        if 0.0 <= raw_similarity <= 1.0:
            score = raw_similarity
            similarity_pct = raw_similarity * 100.0
        elif 0.0 <= raw_similarity <= 100.0:
            similarity_pct = raw_similarity
            score = raw_similarity / 100.0
        else:
            score = 0.0
            similarity_pct = 0.0

        item = {
            'spotify_id': str(spotify_id or '').strip(),
            'title': str(title or '').strip(),
            'artist': str(artist or '').strip(),
            'score': float(score),
            'similarity': round(float(similarity_pct), 2),
            'vibe': r.get('vibe') or '',
            'main_topic': r.get('main_topic') or '',
        }

        # Carry technical metadata when available (used by LLM & UI).
        for key in ['tempo_bpm', 'rms_energy', 'genres', 'popularity', 'spotify_popularity']:
            if key in r and r[key] is not None:
                item[key] = r[key]

        out.append(item)
    return out


def generate_arrangement_advice_llm(result_bundle):
    """Sinh lời khuyên cải thiện bản phối từ kết quả định lượng; fallback nếu LLM chưa sẵn sàng."""

    # HÀM LỒNG (NESTED): Đã xóa bỏ contribution_percent để AI không bịa số liệu
    def _compact_shap_for_prompt(payload: object, top_each: int = 6) -> dict:
        if not isinstance(payload, dict):
            return {}
        tasks = payload.get('tasks')
        if not isinstance(tasks, dict):
            return {}
        p0 = tasks.get('p0') or {}
        if not isinstance(p0, dict):
            return {}

        contribs_all = p0.get('contributions') or p0.get('contributions_top') or []
        if not isinstance(contribs_all, list) or not contribs_all:
            return {}

        def _safe_float(v, default=0.0):
            try: return float(v)
            except Exception: return float(default)

        cleaned = []
        for r in contribs_all:
            if not isinstance(r, dict): continue
            cleaned.append({
                'feature': r.get('feature'),
                'shap_value': _safe_float(r.get('shap_value'), 0.0),
                # KHÔNG gửi contribution_percent sang AI
            })

        pos = [x for x in cleaned if x.get('shap_value', 0.0) > 0]
        neg = [x for x in cleaned if x.get('shap_value', 0.0) < 0]

        # Sửa sort theo shap_value (vì percent đã bị xóa)
        pos.sort(key=lambda x: abs(x['shap_value']), reverse=True)
        neg.sort(key=lambda x: abs(x['shap_value']), reverse=True)

        k = max(1, int(top_each))
        return {
            'p0_title': p0.get('title', 'P0'),
            'p0_top_positive': pos[:k],
            'p0_top_negative': neg[:k],
        }

    # --- BẮT ĐẦU LOGIC CHÍNH CỦA HÀM NGOÀI ---
    shap_payload = result_bundle.get('shap_values')
    shap_values = _compact_shap_for_prompt(shap_payload)
    
    # Gói thông số kỹ thuật để các biến "sáng" lên và AI có dữ liệu
    technical_meta = {
        'tempo': f"{float(result_bundle['raw_features'].get('tempo_bpm', 0)):.1f}",
        'energy': f"{float(result_bundle['raw_features'].get('rms_energy', 0)):.4f}",
        'duration': f"{float(result_bundle['raw_features'].get('duration_sec', 0)):.1f}",
        'lexical': f"{float(result_bundle['raw_features'].get('lexical_diversity', 0)):.3f}",
        'style': result_bundle['p2'].get('cluster_name', 'Unknown'),
        'emotion': result_bundle['p3'].get('emotion_label', 'Unknown'),
        'genres': ', '.join(result_bundle['p4']['genres'])
    }

    # Gọi hàm prompt với đúng 3 tham số
    prompt = build_producer_advice_prompt(
        hit_probability=f"{result_bundle['p0']['hit_prob']:.1f}%",
        shap_values=shap_values,
        technical_meta=technical_meta
    )

    # Ưu tiên Gemini Engine
    gemini_answer = call_gemini_engine(prompt, module='producer')
    if gemini_answer:
        return _format_llm_advice_output(gemini_answer)

    # Fallback khi Gemini lỗi
    return "Hệ thống đang bận phân tích, vui lòng thử lại sau giây lát."

def generate_listener_recommendation_text(user_query, top_tracks):
    """Sinh doan chat than thien cho Top 5 bai hat, fallback neu LLM khong san sang."""
    metadata_lines = []
    for idx, track in enumerate(top_tracks[:5], start=1):
        metadata_lines.append(
            f"{idx}. Ten: {track.get('title', '')}; Nghe si: {track.get('artist', '')}; "
            f"Ly do phu hop: do tuong dong {float(track.get('score', 0.0)):.2f}"
        )
    top_5_songs_metadata = '\n'.join(metadata_lines)
    prompt = build_recommendation_generation_prompt(top_5_songs_metadata, user_query)
    llm_text = call_gemini_engine(prompt, module='listener')
    if llm_text:
        return _format_llm_advice_output(llm_text)
    return 'Mình đã chọn 5 bài phù hợp nhất theo tâm trạng và phong cách bạn đang tìm.'


# =============================================================================
# 7. HÀM SUY LUẬN CHO 5 BÀI TOÁN (P0 -> P4)
# =============================================================================
P2_STYLE_LABELS = {
    'EXPLOSIVE': 'Bùng nổ/Sôi động',
    'DRAMATIC': 'Kịch tính/Da diết',
    'HEALING': 'Bình yên/Chữa lành',
    'EMPATHETIC': 'Sâu lắng/Thấu cảm',
    'FRESH': 'Tươi mới/Yêu đời',
}


def _normalize_text(text):
    """Chuẩn hóa text để so khớp keyword ổn định hơn."""
    text = '' if text is None else str(text)
    text = unicodedata.normalize('NFKD', text)
    text = ''.join(ch for ch in text if not unicodedata.combining(ch))
    return text.lower().strip()


def _label_from_features_for_p2(full_feats):
    """Gán nhãn phong cách theo heuristic từ tempo + energy."""
    tempo = float(full_feats.get('tempo_bpm', 100))
    energy = float(full_feats.get('rms_energy', 0.05))

    if tempo >= 122 and energy >= 0.09:
        return P2_STYLE_LABELS['EXPLOSIVE']
    if tempo >= 108 and energy < 0.07:
        return P2_STYLE_LABELS['DRAMATIC']
    if tempo < 88 and energy < 0.05:
        return P2_STYLE_LABELS['HEALING']
    if tempo < 102 and energy < 0.075:
        return P2_STYLE_LABELS['EMPATHETIC']
    return P2_STYLE_LABELS['FRESH']


def _map_raw_cluster_name_to_p2_style(raw_name, full_feats):
    """Ánh xạ tên cụm thô từ model sang bộ nhãn phong cách chuẩn P2."""
    normalized = _normalize_text(raw_name)

    keyword_map = [
        (('upbeat', 'dance', 'energetic', 'soi dong', 'bung no'), P2_STYLE_LABELS['EXPLOSIVE']),
        (('dramatic', 'intense', 'kich tinh', 'da diet'), P2_STYLE_LABELS['DRAMATIC']),
        (('chill', 'acoustic', 'healing', 'ballad', 'binh yen', 'chua lanh'), P2_STYLE_LABELS['HEALING']),
        (('emotional', 'melancholy', 'deep', 'lofi', 'lo-fi', 'sau lang', 'thau cam'), P2_STYLE_LABELS['EMPATHETIC']),
        (('fresh', 'happy', 'bright', 'tuoi moi', 'yeu doi'), P2_STYLE_LABELS['FRESH']),
    ]

    for keywords, style_label in keyword_map:
        if any(k in normalized for k in keywords):
            return style_label

    return _label_from_features_for_p2(full_feats)


def predict_p0(df_input, full_feats):
    """P0 - Xác suất hit."""
    if pipeline_task1 is not None:
        pred = pipeline_task1.predict(df_input)[0]
        if hasattr(pipeline_task1, 'predict_proba'):
            proba = pipeline_task1.predict_proba(df_input)[0]
            hit_prob = float(proba[1] * 100 if len(proba) > 1 else proba[0] * 100)
        else:
            hit_prob = float(100.0 if pred == 1 else 0.0)
        return {
            'label': 'HIT' if int(pred) == 1 else 'NON-HIT',
            'hit_prob': max(0.0, min(100.0, hit_prob)),
            'source': 'model',
        }

    # Placeholder khi chưa có model thực
    tempo = float(full_feats.get('tempo_bpm', 0))
    energy = float(full_feats.get('rms_energy', 0))
    score = ((tempo / 180.0) * 0.45 + min(energy * 8.0, 1.0) * 0.55) * 100
    score = max(0.0, min(100.0, score))
    return {
        'label': 'HIT' if score >= 50 else 'NON-HIT',
        'hit_prob': score,
        'source': 'placeholder',
    }


def predict_p1(df_input, full_feats):
    """P1 - Điểm phổ biến."""
    if p1_data and isinstance(p1_data, dict) and 'pipeline' in p1_data:
        p1_model = p1_data['pipeline']
        score = float(p1_model.predict(df_input)[0])
        return {
            'popularity_score': max(0.0, min(100.0, score)),
            'source': 'model',
        }

    tempo = float(full_feats.get('tempo_bpm', 100))
    energy = float(full_feats.get('rms_energy', 0.05))
    lexical_div = float(full_feats.get('lexical_diversity', 0.5))
    score = 30 + tempo * 0.15 + energy * 90 + lexical_div * 12
    return {
        'popularity_score': max(0.0, min(100.0, score)),
        'source': 'placeholder',
    }


def predict_p2(df_input, full_feats):
    """P2 - Cụm phong cách."""
    if p2_data and isinstance(p2_data, dict):
        try:
            numeric_cols = p2_data.get('numeric_features', [])
            if not numeric_cols:
                raise ValueError('Thiếu numeric_features trong p2_data')

            for col in numeric_cols:
                if col not in df_input.columns:
                    if col.startswith('sentiment_'):
                        df_input[col] = 1.0 if 'neutral' in col else 0.0
                    else:
                        df_input[col] = 0.0

            x_raw = df_input[numeric_cols].values
            x_imputed = p2_data['imputer'].transform(x_raw)
            x_scaled = p2_data['scaler'].transform(x_imputed)

            # Keep inference path consistent with training artifacts:
            # imputer -> scaler -> pca -> clusterer (no extra runtime weighting).
            x_pca = p2_data['pca'].transform(x_scaled)
            cluster_id = int(p2_data['clusterer'].predict(x_pca)[0])

            cluster_name_map = p2_data.get('cluster_names', {})
            c_name = cluster_name_map.get(cluster_id)
            if c_name is None:
                c_name = cluster_name_map.get(str(cluster_id), f'Cụm {cluster_id}')
            c_name = _map_raw_cluster_name_to_p2_style(c_name, full_feats)
            return {
                'cluster_id': cluster_id,
                'cluster_name': c_name,
                'source': 'model',
            }
        except Exception as ex:
            logging.warning('P2 clustering inference failed, fallback to heuristic: %s', ex)

    cname = _label_from_features_for_p2(full_feats)
    return {
        'cluster_id': -1,
        'cluster_name': cname,
        'source': 'placeholder',
    }


def predict_p3(df_input, full_feats):
    """P3 - Cảm xúc."""
    if p3_data and isinstance(p3_data, dict) and 'pipeline' in p3_data:
        p3_model = p3_data['pipeline']
        pred_sent = p3_model.predict(df_input)[0]
        sentiment_map = {0: 'Tiêu cực', 1: 'Trung tính', 2: 'Tích cực'}
        return {
            'emotion_label': sentiment_map.get(int(pred_sent), 'Trung tính'),
            'source': 'model',
        }

    energy = float(full_feats.get('rms_energy', 0))
    label = 'Tích cực' if energy > 0.08 else 'Trung tính'
    return {
        'emotion_label': label,
        'source': 'placeholder',
    }


def predict_p4(df_input, full_feats):
    """P4 - Thể loại."""
    if p4_data and isinstance(p4_data, dict) and 'pipeline' in p4_data:
        p4_model = p4_data['pipeline']
        pred_raw = p4_model.predict(df_input)
        if hasattr(pred_raw, 'toarray'):
            pred_raw = pred_raw.toarray()
        pred_row = pred_raw[0] if len(pred_raw.shape) > 1 else pred_raw
        genre_labels = p4_data.get('classes', ['Hip-Hop', 'Indie', 'V-Pop', 'Vinahouse'])
        detected = [genre_labels[i] for i, val in enumerate(pred_row) if val == 1 and i < len(genre_labels)]
        if not detected:
            detected = ['V-Pop']
        return {
            'genres': detected,
            'source': 'model',
        }

    tempo = float(full_feats.get('tempo_bpm', 100))
    energy = float(full_feats.get('rms_energy', 0.05))
    if tempo > 120 and energy > 0.08:
        genres = ['Dance Pop', 'Vinahouse']
    elif tempo < 95:
        genres = ['Ballad', 'V-Pop']
    else:
        genres = ['Modern V-Pop']
    return {
        'genres': genres,
        'source': 'placeholder',
    }


def run_parallel_models(df_input, full_feats):
    """Chạy đồng thời 5 bài toán để rút ngắn thời gian phản hồi UI."""
    with ThreadPoolExecutor(max_workers=5) as executor:
        f0 = executor.submit(predict_p0, df_input.copy(), full_feats)
        f1 = executor.submit(predict_p1, df_input.copy(), full_feats)
        f2 = executor.submit(predict_p2, df_input.copy(), full_feats)
        f3 = executor.submit(predict_p3, df_input.copy(), full_feats)
        f4 = executor.submit(predict_p4, df_input.copy(), full_feats)
        return {
            'p0': f0.result(),
            'p1': f1.result(),
            'p2': f2.result(),
            'p3': f3.result(),
            'p4': f4.result(),
        }


# =============================================================================
# 8. SHAP ANALYSIS (P0 / P1)
# =============================================================================
try:
    import shap
    import matplotlib.pyplot as plt

    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False


def render_shap_analysis(df_input, full_feats):
    """Hiển thị SHAP cho P0/P1 theo đúng pipeline và gom về đặc trưng nghiệp vụ dễ đọc."""
    st.subheader('Bước 4 - SHAP Analysis: Minh bạch hóa quyết định')

    focus_features = ['tempo_bpm', 'rms_energy', 'beat_strength_mean', 'lyric_total_words', 'lexical_diversity']
    row = []
    for feat in focus_features:
        row.append(float(df_input.iloc[0][feat]) if feat in df_input.columns else float(full_feats.get(feat, 0.0)))

    feature_df = pd.DataFrame([row], columns=focus_features)

    def _build_demo_contrib(task_name):
        values = np.abs(feature_df.iloc[0].values.astype(float))
        total = float(values.sum()) if values.sum() > 0 else 1.0
        percent = values / total * 100
        return pd.DataFrame(
            {
                'feature': focus_features,
                'contribution_percent': percent,
                'task': task_name,
            }
        ).sort_values('contribution_percent', ascending=False)

    def _extract_pipeline_parts(model):
        if model is None or not hasattr(model, 'named_steps'):
            return None, None, None, None

        preprocessor = model.named_steps.get('preprocessor')
        selector = model.named_steps.get('feature_selection')
        estimator = model.steps[-1][1] if getattr(model, 'steps', None) else None

        if preprocessor is None or estimator is None:
            return None, None, None, None
        return preprocessor, selector, estimator, model.steps[-1][0]

    def _is_tree_estimator(estimator):
        tree_tokens = (
            'xgb', 'randomforest', 'extratrees', 'gradientboosting',
            'decisiontree', 'histgradientboosting', 'lightgbm', 'catboost'
        )
        cls_name = estimator.__class__.__name__.lower()
        mod_name = estimator.__class__.__module__.lower()
        return any(tok in cls_name or tok in mod_name for tok in tree_tokens)

    def _build_local_background(base_df):
        # Tạo neighborhood nhỏ quanh mẫu hiện tại để SHAP ổn định hơn với 1 sample inference.
        rows = [base_df.iloc[0].copy()]
        deltas = {
            'tempo_bpm': max(abs(float(base_df.iloc[0].get('tempo_bpm', 100.0))) * 0.08, 2.0),
            'rms_energy': max(abs(float(base_df.iloc[0].get('rms_energy', 0.05))) * 0.08, 0.003),
            'beat_strength_mean': max(abs(float(base_df.iloc[0].get('beat_strength_mean', 0.5))) * 0.08, 0.01),
            'lyric_total_words': max(abs(float(base_df.iloc[0].get('lyric_total_words', 0.0))) * 0.08, 5.0),
            'lexical_diversity': max(abs(float(base_df.iloc[0].get('lexical_diversity', 0.5))) * 0.08, 0.01),
        }
        for feat, delta in deltas.items():
            if feat not in base_df.columns:
                continue
            plus_row = base_df.iloc[0].copy()
            minus_row = base_df.iloc[0].copy()
            plus_row[feat] = float(plus_row[feat]) + delta
            minus_row[feat] = float(minus_row[feat]) - delta
            rows.append(plus_row)
            rows.append(minus_row)
        return pd.DataFrame(rows).reset_index(drop=True)

    def _ensure_feature_names(preprocessor, transformed_count):
        try:
            names = preprocessor.get_feature_names_out()
            return [str(n) for n in names]
        except Exception:
            return [f'feature_{i}' for i in range(transformed_count)]

    def _apply_selector_if_any(x_matrix, feature_names, selector):
        if selector is None:
            return x_matrix, feature_names

        try:
            x_sel = selector.transform(x_matrix)
            if hasattr(selector, 'get_support'):
                mask = selector.get_support()
                if len(mask) == len(feature_names):
                    selected_names = [name for name, keep in zip(feature_names, mask) if keep]
                else:
                    selected_names = [f'feature_{i}' for i in range(x_sel.shape[1])]
            else:
                selected_names = [f'feature_{i}' for i in range(x_sel.shape[1])]
            return x_sel, selected_names
        except Exception:
            return x_matrix, feature_names

    def _normalize_shap_vector(shap_values, expected_len):
        vals = getattr(shap_values, 'values', shap_values)
        arr = np.array(vals)

        if arr.ndim == 3:
            # Binary classifier có thể trả (n_samples, n_features, n_classes)
            arr = arr[0, :, 1 if arr.shape[2] > 1 else 0]
        elif arr.ndim == 2:
            arr = arr[0]
        elif arr.ndim == 1:
            arr = arr
        else:
            arr = np.array(arr).flatten()

        if arr.shape[0] != expected_len:
            arr = arr.flatten()[:expected_len]
            if arr.shape[0] < expected_len:
                arr = np.pad(arr, (0, expected_len - arr.shape[0]))

        return arr.astype(float)

    def _aggregate_to_focus_features(shap_vals, shap_feature_names, task_name):
        rows = []
        for feat in focus_features:
            feat_l = feat.lower()
            matched_idx = [
                idx for idx, name in enumerate(shap_feature_names)
                if feat_l in str(name).lower()
            ]
            if not matched_idx:
                signed = 0.0
                abs_val = 0.0
            else:
                sub = np.array([shap_vals[i] for i in matched_idx], dtype=float)
                signed = float(sub.sum())
                abs_val = float(np.abs(sub).sum())

            rows.append(
                {
                    'feature': feat,
                    'shap_value': signed,
                    'abs_shap': abs_val,
                    'task': task_name,
                }
            )

        df_rows = pd.DataFrame(rows)
        total = float(df_rows['abs_shap'].sum()) if float(df_rows['abs_shap'].sum()) > 0 else 1.0
        df_rows['contribution_percent'] = df_rows['abs_shap'] / total * 100.0
        return df_rows[['feature', 'shap_value', 'contribution_percent', 'task']].sort_values(
            'contribution_percent', ascending=False
        )

    def _predict_scalar(model, sample_df, use_proba):
        if use_proba and hasattr(model, 'predict_proba'):
            return float(model.predict_proba(sample_df)[:, 1][0])
        pred = model.predict(sample_df)
        return float(pred[0]) if np.ndim(pred) > 0 else float(pred)

    def _build_sensitivity_contrib(task_name, model, use_proba):
        base_df = df_input.iloc[[0]].copy()
        base_score = _predict_scalar(model, base_df, use_proba)

        min_steps = {
            'tempo_bpm': 3.0,
            'rms_energy': 0.005,
            'beat_strength_mean': 0.01,
            'lyric_total_words': 5.0,
            'lexical_diversity': 0.01,
        }

        impacts = []
        for feat in focus_features:
            if feat not in base_df.columns:
                impacts.append(0.0)
                continue

            try:
                base_val = float(base_df.iloc[0][feat])
            except Exception:
                impacts.append(0.0)
                continue

            step = max(abs(base_val) * 0.1, min_steps.get(feat, 0.01))

            plus_df = base_df.copy()
            minus_df = base_df.copy()
            plus_df.loc[plus_df.index[0], feat] = base_val + step
            minus_df.loc[minus_df.index[0], feat] = base_val - step

            try:
                plus_score = _predict_scalar(model, plus_df, use_proba)
                minus_score = _predict_scalar(model, minus_df, use_proba)
                impact = abs(plus_score - base_score) + abs(minus_score - base_score)
            except Exception:
                impact = 0.0

            impacts.append(float(impact))

        impacts = np.array(impacts, dtype=float)
        total = float(impacts.sum()) if impacts.sum() > 0 else 1.0
        percent = impacts / total * 100

        return pd.DataFrame(
            {
                'feature': focus_features,
                'shap_value': impacts,
                'contribution_percent': percent,
                'task': task_name,
            }
        ).sort_values('contribution_percent', ascending=False)

    def _plot_contrib_dark(df_contrib, title):
        fig, ax = plt.subplots(figsize=(7, 3.6))
        fig.patch.set_facecolor('#171e44')
        ax.set_facecolor('#111733')
        ax.barh(
            df_contrib['feature'][::-1],
            df_contrib['contribution_percent'][::-1],
            color='#7b8ef0',
            alpha=0.92,
        )
        ax.set_xlabel('Contribution (%)', color='#d7e0ff')
        ax.set_title(f'Top feature contribution - {title}', color='#eef2ff')
        ax.tick_params(axis='x', colors='#c8d3ff')
        ax.tick_params(axis='y', colors='#dbe4ff')
        for spine in ax.spines.values():
            spine.set_color('#3b477f')
        ax.grid(axis='x', color='#3a456f', alpha=0.35)
        return fig

    def _build_true_shap_contrib(task_name, model, use_proba):
        preprocessor, selector, estimator, _ = _extract_pipeline_parts(model)
        if preprocessor is None or estimator is None:
            raise ValueError('Pipeline không có preprocessor/estimator phù hợp để tính SHAP.')

        base_df = df_input.iloc[[0]].copy()
        bg_df = _build_local_background(base_df)

        x_bg = preprocessor.transform(bg_df)
        x_one = preprocessor.transform(base_df)

        if hasattr(x_bg, 'toarray'):
            x_bg = x_bg.toarray()
        if hasattr(x_one, 'toarray'):
            x_one = x_one.toarray()

        feature_names = _ensure_feature_names(preprocessor, int(x_bg.shape[1]))
        x_bg, feature_names = _apply_selector_if_any(x_bg, feature_names, selector)
        x_one, _ = _apply_selector_if_any(x_one, feature_names, selector)

        if _is_tree_estimator(estimator):
            if use_proba and hasattr(estimator, 'predict_proba'):
                explainer = shap.TreeExplainer(estimator, data=x_bg, model_output='probability')
            else:
                explainer = shap.TreeExplainer(estimator, data=x_bg)
            shap_values = explainer(x_one)
        else:
            # Fallback generic explainer cho estimator không phải tree.
            if use_proba and hasattr(estimator, 'predict_proba'):
                predict_fn = lambda x: estimator.predict_proba(x)[:, 1]
            else:
                predict_fn = estimator.predict
            explainer = shap.Explainer(predict_fn, x_bg, feature_names=feature_names)
            shap_values = explainer(x_one)

        shap_vec = _normalize_shap_vector(shap_values, expected_len=len(feature_names))
        df_contrib = _aggregate_to_focus_features(shap_vec, feature_names, task_name)
        return df_contrib

    p1_model = p1_data.get('pipeline') if isinstance(p1_data, dict) and 'pipeline' in p1_data else None
    model_specs = [
        ('P0 - Hit Probability', pipeline_task1, True),
        ('P1 - Popularity Score', p1_model, False),
    ]

    contrib_blocks = []

    if SHAP_AVAILABLE:
        for title, model, use_proba in model_specs:
            if model is None:
                contrib_blocks.append((title, _build_demo_contrib(title), False, 'model-missing', 'Model chưa được load'))
                continue
            try:
                df_contrib = _build_true_shap_contrib(title, model, use_proba)
                if float(np.abs(df_contrib['shap_value']).sum()) <= 1e-12:
                    sens_df = _build_sensitivity_contrib(title, model, use_proba)
                    contrib_blocks.append((title, sens_df, False, 'shap-zero', 'SHAP trả toàn 0 cho mẫu hiện tại.'))
                else:
                    contrib_blocks.append((title, df_contrib, True, 'ok', ''))
            except Exception as ex:
                err_msg = str(ex).strip() or ex.__class__.__name__
                logging.warning('SHAP failed for %s: %s', title, err_msg)
                contrib_blocks.append((title, _build_sensitivity_contrib(title, model, use_proba), False, 'shap-failed', err_msg))
    else:
        contrib_blocks = [
            ('P0 - Hit Probability', _build_demo_contrib('P0 - Hit Probability'), False, 'shap-missing', 'Thư viện SHAP chưa cài hoặc import lỗi'),
            ('P1 - Popularity Score', _build_demo_contrib('P1 - Popularity Score'), False, 'shap-missing', 'Thư viện SHAP chưa cài hoặc import lỗi'),
        ]

    def _df_to_prompt_records(df_contrib):
        records = []
        if df_contrib is None or df_contrib.empty:
            return records
        for row in df_contrib.to_dict(orient='records'):
            records.append(
                {
                    'feature': str(row.get('feature', '')),
                    'shap_value': float(row.get('shap_value', 0.0)),
                    'contribution_percent': float(row.get('contribution_percent', 0.0)),
                    'task': str(row.get('task', '')),
                }
            )
        return records

    shap_payload = {
        'focus_features': list(focus_features),
        'tasks': {},
    }

    for title, df_contrib, is_real_shap, status, detail in contrib_blocks:
        key = title.split('-')[0].strip().lower()  # p0 / p1
        shap_payload['tasks'][key] = {
            'title': title,
            'status': status,
            'detail': detail,
            'is_real_shap': bool(is_real_shap),
            'contributions': _df_to_prompt_records(df_contrib),
        }

        st.markdown(f'#### {title}')
        c_shap1, c_shap2 = st.columns([3, 2])
        with c_shap1:
            if is_real_shap:
                fig = _plot_contrib_dark(df_contrib, title)
                st.pyplot(fig, clear_figure=True)
            else:
                fig = _plot_contrib_dark(df_contrib, title)
                st.pyplot(fig, clear_figure=True)
                if status == 'model-missing':
                    st.caption('Đang hiển thị placeholder vì model chưa được load.')
                elif status == 'shap-failed':
                    st.caption('SHAP lỗi ở model hiện tại, đang hiển thị sensitivity contribution.')
                    if detail:
                        st.caption(f'Chi tiết lỗi SHAP: {detail}')
                elif status == 'shap-zero':
                    st.caption('SHAP cho toàn bộ giá trị 0, đang hiển thị sensitivity contribution để dễ đọc kết quả.')
                    if detail:
                        st.caption(f'Chi tiết lỗi SHAP: {detail}')
                else:
                    st.caption('Thư viện SHAP chưa khả dụng, đang hiển thị biểu đồ placeholder.')
        with c_shap2:
            st.dataframe(df_contrib, use_container_width=True)

    return shap_payload


def render_shap_payload_cached(shap_payload: dict | None):
    """Render SHAP payload đã cache (không tính SHAP lại)."""
    if not shap_payload or not isinstance(shap_payload, dict):
        st.caption('Chưa có SHAP payload để hiển thị.')
        return

    tasks = shap_payload.get('tasks') or {}
    if not isinstance(tasks, dict) or not tasks:
        st.caption('SHAP payload rỗng.')
        return

    # st.markdown('### SHAP (cached)')
    for task_key, task in tasks.items():
        task = task or {}
        title = str(task.get('title') or task_key).strip()
        contributions = task.get('contributions') or []
        if not isinstance(contributions, list) or not contributions:
            continue

        df = pd.DataFrame(contributions)
        if 'contribution_percent' in df.columns:
            df = df.sort_values('contribution_percent', ascending=False)

        # st.markdown(f'#### {title}')
        # st.dataframe(df, use_container_width=True)


# =============================================================================
# 9. HÀM RENDER DASHBOARD
# =============================================================================
def render_dashboard(bundle):
    """Hiển thị dashboard tổng hợp P0..P4 trên một màn hình."""
    st.subheader('Bước 3 - Dashboard Tổng hợp 5 bài toán')

    def _render_kpi(col, label, value, badge=''):
        badge_html = (
            "<div style='margin-top: 10px; display: inline-block; padding: 4px 10px; "
            "border-radius: 999px; background: rgba(63, 201, 142, 0.2); color: #66f2b5; "
            "font-weight: 700; font-size: 0.9rem;'>"
            f"{badge}"
            "</div>"
        ) if badge else ""

        col.markdown(
            (
                "<div style='min-height: 120px;'>"
                f"<div style='font-size: 1.45rem; font-weight: 700; line-height: 1.25; margin-bottom: 4px;'>"
                f"{label}"
                "</div>"
                "<div style='font-size: 1.45rem; font-weight: 800; line-height: 1.25; "
                "color: rgba(255, 255, 255, 0.95); white-space: normal; word-break: break-word;'>"
                f"{value}"
                "</div>"
                f"{badge_html}"
                "</div>"
            ),
            unsafe_allow_html=True,
        )

    c1, c2, c3, c4, c5 = st.columns(5)
    _render_kpi(c1, 'P0 - Xác suất Hit', f"{bundle['p0']['hit_prob']:.1f}%", f"↑ {bundle['p0']['label']}")
    if isinstance(bundle.get('p1'), dict) and bundle['p1'].get('popularity_score') is not None:
        _render_kpi(c2, 'P1 - Điểm phổ biến', f"{float(bundle['p1']['popularity_score']):.1f}/100")
    else:
        _render_kpi(c2, 'P1 - Điểm phổ biến', 'N/A', 'P1 disabled')
    _render_kpi(c3, 'P2 - Cụm phong cách', bundle['p2']['cluster_name'])
    _render_kpi(c4, 'P3 - Cảm xúc', bundle['p3']['emotion_label'])
    _render_kpi(c5, 'P4 - Thể loại', ', '.join(bundle['p4']['genres'][:2]))

    st.markdown('### Chỉ số audio/text chính')
    f = bundle['raw_features']
    k1, k2, k3, k4 = st.columns(4)
    k1.metric('Tempo', f"{float(f.get('tempo_bpm', 0)):.1f} BPM")
    k2.metric('RMS Energy', f"{float(f.get('rms_energy', 0)):.4f}")
    k3.metric('Duration', f"{float(f.get('duration_sec', 0)):.1f} sec")
    k4.metric('Lexical Diversity', f"{float(f.get('lexical_diversity', 0)):.3f}")

    # Bảng tóm tắt nguồn dự đoán để dễ debug demo
    summary_df = pd.DataFrame(
        [
            {'task': 'P0', 'result': bundle['p0']['label'], 'source': bundle['p0']['source']},
            {
                'task': 'P1',
                'result': (
                    f"{float(bundle['p1']['popularity_score']):.1f}"
                    if isinstance(bundle.get('p1'), dict) and bundle['p1'].get('popularity_score') is not None
                    else 'N/A'
                ),
                'source': (bundle.get('p1') or {}).get('source', 'disabled') if isinstance(bundle.get('p1'), dict) else 'disabled',
            },
            {'task': 'P2', 'result': bundle['p2']['cluster_name'], 'source': bundle['p2']['source']},
            {'task': 'P3', 'result': bundle['p3']['emotion_label'], 'source': bundle['p3']['source']},
            {'task': 'P4', 'result': ', '.join(bundle['p4']['genres']), 'source': bundle['p4']['source']},
        ]
    )
    # st.dataframe(summary_df, use_container_width=True)
# =============================================================================
# 10. GIAO DIỆN SIÊU TRỢ LÝ (SINGLE-PAGE AGENT) & BỘ ĐIỀU PHỐI (ORCHESTRATOR)
# =============================================================================
try:
    from chatbot.intent import parse_intent_llm
except ModuleNotFoundError:
    from chatbot.intent import parse_intent_llm

try:
    from chatbot.action_handler import handle_action as _handle_action
except ModuleNotFoundError:
    from action_handler import handle_action as _handle_action

import io


_cache_data = getattr(st, 'cache_data', st.cache_resource)


@_cache_data(show_spinner=False)
def _load_artist_list() -> list[str]:
    """Load artist names for fuzzy matching (RECOMMEND_ARTIST).

    Prefer Supabase table `artists` (paged). Fall back to local CSV if needed.
    """

    # 1) Supabase table first
    try:
        client = _get_supabase_client()
        if client is not None:
            candidates = ['artist_name', 'name', 'artist']
            page_size = 1000
            max_rows = 50000
            for col in candidates:
                try:
                    out: list[str] = []
                    start = 0
                    while start < max_rows:
                        end = start + page_size - 1
                        resp = client.table('artists').select(col).range(start, end).execute()
                        rows = getattr(resp, 'data', None) or []
                        if not rows:
                            break
                        for r in rows:
                            if isinstance(r, dict):
                                name = str(r.get(col) or '').strip()
                                if name:
                                    out.append(name)
                        if len(rows) < page_size:
                            break
                        start += page_size
                    if out:
                        # dedupe while preserving order
                        return list(dict.fromkeys(out))
                except Exception:
                    continue
    except Exception:
        pass

    # 2) CSV fallback
    try:
        csv_path = os.path.join(_REPO_ROOT, 'data', 'artists_vietnam.csv')
        if not os.path.exists(csv_path):
            return []
        df = pd.read_csv(csv_path)
        col = 'artist_name' if 'artist_name' in df.columns else None
        if col is None:
            return []
        artists = [str(x).strip() for x in df[col].dropna().tolist()]
        artists = [a for a in artists if a]
        return list(dict.fromkeys(artists))
    except Exception:
        return []

# --- 10.1 ĐÍNH KÈM FILE (trên màn hình chính, không dùng sidebar) ---
if 'global_audio_bytes' not in st.session_state:
    st.session_state.global_audio_bytes = None
    st.session_state.global_audio_name = None
if 'global_lyric_text' not in st.session_state:
    st.session_state.global_lyric_text = None
    st.session_state.global_lyric_name = None

def _embed_query_text(text: str) -> list[float] | None:
    """Embedding helper for vector RPCs.

    Uses the same provider as `chatbot.supabase` to keep dimensions consistent.
    """
    # Attach error info to the function object for downstream UI.
    try:
        _embed_query_text.last_error = None  # type: ignore[attr-defined]
    except Exception:
        pass

    try:
        from chatbot.supabase import encode_lyrics_embedding_debug

        vec, err = encode_lyrics_embedding_debug(str(text or '').strip())
        try:
            _embed_query_text.last_error = err  # type: ignore[attr-defined]
        except Exception:
            pass
        return vec
    except Exception as ex:
        try:
            _embed_query_text.last_error = f"{type(ex).__name__}: {ex}"  # type: ignore[attr-defined]
        except Exception:
            pass
        return None


def _answer_music_knowledge_local(user_question: str) -> str | None:
    """Fast, deterministic answers for common MUSIC_KNOWLEDGE questions."""

    q_raw = str(user_question or '').strip()
    if not q_raw:
        return None
    q = _normalize_text(q_raw)

    # --- Chord: e.g., "hợp âm C thứ", "Cm", "C7", "Cm7" ---
    if 'hop am' in q or 'hợp âm' in q_raw.lower():
        import re

        # Try to capture root note.
        m = re.search(r"\b([a-g])\s*([#b])?\b", q)
        if m:
            root = (m.group(1) or '').upper()
            accidental = (m.group(2) or '')
            note = root + accidental

            is_minor = (' thu' in q) or ('minor' in q) or (re.search(r"\bcm\b", q) is not None)
            is_major = (' truong' in q) or ('major' in q)

            # 7th detection
            has_maj7 = 'maj7' in q
            has_m7 = ('m7' in q and 'maj7' not in q)
            has_7 = ('7' in q and not has_maj7 and not has_m7)

            # Only implement C-family precisely (covers the reported bug); others fall back to LLM.
            if note == 'C':
                if is_minor and not (has_maj7 or has_m7 or has_7):
                    return "Hợp âm C thứ (Cm) là hợp âm 3 nốt gồm: C – Eb – G. Công thức: 1 – b3 – 5 (so với gam C trưởng)."
                if is_major and not (has_maj7 or has_m7 or has_7):
                    return "Hợp âm C trưởng (C) là hợp âm 3 nốt gồm: C – E – G. Công thức: 1 – 3 – 5."
                if has_7 and not is_minor:
                    return "Hợp âm C7 (dominant 7) gồm: C – E – G – Bb. Công thức: 1 – 3 – 5 – b7."
                if has_m7 or (is_minor and has_7):
                    return "Hợp âm Cm7 gồm: C – Eb – G – Bb. Công thức: 1 – b3 – 5 – b7."
                if has_maj7:
                    return "Hợp âm Cmaj7 gồm: C – E – G – B. Công thức: 1 – 3 – 5 – 7."

    # --- Indie vs Pop ---
    if ('indie' in q) and ('pop' in q) and any(tok in q for tok in ['khac', 'khác', 'difference', 'phan biet', 'phân biệt', 'la gi']):
        return (
            "Indie (independent) thường chỉ cách làm nhạc/định hướng phát hành: nghệ sĩ/label nhỏ tự chủ hơn về sáng tạo, "
            "sound có thể ‘lạ’ và kén người nghe hơn. Pop là dòng nhạc hướng đại chúng: cấu trúc dễ nghe (verse–chorus), hook rõ, "
            "mix/master và thông điệp thường tối ưu cho số đông. Một bài ‘indie pop’ có thể vừa indie về cách làm, vừa pop về giai điệu."
        )

    return None


def _dynamic_intro_text(*, user_prompt: str, action: str, tracks: list[dict], params: dict = None) -> str:
    """Generate a flexible intro (LLM when possible; otherwise a varied fallback)."""
    params = params or {}

    # Lấy map điểm live từ Spotify (nếu có truyền vào)
    live_pop_map = params.get('live_pop_map', {})

    seed_tempo = None
    seed_energy = None
    try:
        if params.get('seed_tempo_bpm') is not None:
            seed_tempo = float(params.get('seed_tempo_bpm'))
    except Exception:
        seed_tempo = None
    try:
        if params.get('seed_rms_energy') is not None:
            seed_energy = float(params.get('seed_rms_energy'))
    except Exception:
        seed_energy = None
    
    try:
        compact = []
        max_pop = -1

        for t in (tracks or [])[:5]:
            if not isinstance(t, dict):
                continue

            # --- 1. KHAI BÁO spotify_id Ở ĐÂY ĐỂ TRÁNH LỖI NAMEERROR ---
            spotify_id = str(t.get('spotify_id') or '').strip()

            title = str(t.get('title') or '').strip()
            artist = str(t.get('artist') or 'Chưa rõ ca sĩ').strip()
            
            # Trích xuất Vibe, Topic, Similarity
            vibe = str(t.get('vibe') or '').strip()
            topic = str(t.get('main_topic') or '').strip()
            similarity = t.get('similarity')
            
            # Trích xuất thông số kỹ thuật (DNA)
            tempo = t.get('tempo_bpm')
            energy = t.get('rms_energy')
            genres = str(t.get('genres') or '').strip()
            
            # 4. [QUAN TRỌNG] Gọi spotify_id SAU KHI nó đã được khởi tạo
            live_pop = live_pop_map.get(spotify_id, -1)
            popularity = live_pop if live_pop >= 0 else t.get('spotify_popularity') or t.get('popularity')

            # --- CẬP NHẬT ĐIỂM MAX VÀO BIẾN ---
            if popularity is not None:
                try:
                    if int(popularity) > max_pop:
                        max_pop = int(popularity)
                except Exception:
                    pass

            # Gói thông tin thành chuỗi
            track_info = f"- Tên bài: {title} | Ca sĩ: {artist}"
            
            extra_info = []
            if genres:
                extra_info.append(f"Thể loại: {genres}")
            if tempo is not None and energy is not None:
                extra_info.append(f"Tempo: {float(tempo):.0f} BPM, Năng lượng: {float(energy):.2f}")
                if action == 'RECOMMEND_SEED' and seed_tempo is not None and seed_energy is not None:
                    try:
                        d_tempo = abs(float(tempo) - float(seed_tempo))
                        d_energy = abs(float(energy) - float(seed_energy))
                        extra_info.append(f"Chênh lệch so với bài mẫu: {d_tempo:.0f} BPM, {d_energy:.2f} năng lượng")
                    except Exception:
                        pass
            if vibe or topic:
                extra_info.append(f"Cảm nhận: {vibe}, {topic}")
            if similarity is not None:
                extra_info.append(f"Độ tương đồng: {similarity}%")
            if popularity is not None:
                extra_info.append(f"Độ hot Spotify: {popularity}")
                
            if extra_info:
                track_info += f" ({' - '.join(extra_info)})"
                
            if title:
                compact.append(track_info)
                
        track_block = "\n".join(compact)

        # Compact request context (so LLM can speak naturally and stay on-topic).
        ctx_bits = []
        for k, label in [
            ('song_title', 'Bài hát'),
            ('artist', 'Nghệ sĩ'),
            ('mood', 'Tâm trạng'),
            ('genre', 'Thể loại'),
            ('lyric_snippet', 'Đoạn lời'),
            ('seed_name', 'Bài mẫu'),
            ('seed_vibe', 'Vibe bài mẫu'),
            ('seed_genres', 'Thể loại bài mẫu'),
            ('seed_title', 'Tên bài mẫu (khớp DB)'),
            ('seed_artist', 'Ca sĩ bài mẫu (khớp DB)'),
            ('seed_tempo_bpm', 'Tempo bài mẫu'),
            ('seed_rms_energy', 'Năng lượng bài mẫu'),
            ('attributes', 'Thuộc tính nhạc lý'),
        ]:
            v = params.get(k)
            v = '' if v is None else str(v).strip()
            if v:
                ctx_bits.append(f"{label}: {v}")
        request_ctx = " | ".join(ctx_bits)
        
        # --- [QUAN TRỌNG] PHÂN NHÁNH PROMPT CHO AI THEO ACTION ---
        
        if action == "RECOMMEND_SEED" and params.get("seed_name"):
            seed = params.get("seed_name")
            llm_prompt = (
                f"Bạn là Music Producer và chuyên gia V-Pop. Người dùng muốn tìm các bài hát có DNA âm thanh giống với bài gốc '{seed}'.\n"
                f"Hệ thống đã dò 'DNA âm thanh' dựa trên vector audio 40D (đặc trưng nhịp/energy/timbre) trong kho nhạc và tìm được các bài tương tự.\n"
                f"Ngữ cảnh yêu cầu: {request_ctx or 'N/A'}\n\n"
                f"Nhiệm vụ: Viết 2-3 câu. (1) Nêu *bài #1* là giống nhất và vì sao; (2) viện dẫn số liệu cụ thể: Tempo, Năng lượng, Độ tương đồng (%), và nếu có thì 'Chênh lệch so với bài mẫu'; (3) có thể nhắc nhanh là các bài còn lại cũng nằm trong dải tương đồng gần nhau.\n\n"
                f"⚠️ QUY TẮC: KHÔNG bịa tên ca sĩ. KHÔNG gạch đầu dòng. KHÔNG thêm bài ngoài danh sách.\n\n"
                f"Danh sách kết quả:\n{track_block}"
            )
            
        elif action == "SEARCH_AUDIO":
            llm_prompt = (
                f"Bạn là hệ thống nhận diện nhạc AI. Người dùng vừa tải lên một file âm thanh và hệ thống đã nhận diện được ĐÚNG MỘT KẾT QUẢ dưới đây.\n"
                f"Nhiệm vụ: Thông báo kết quả nhận diện cho người dùng bằng 1 câu ngắn gọn, thân thiện.\n\n"
                f"⚠️ QUY TẮC: BẠN PHẢI CHÉP CHÍNH XÁC 'Tên bài' VÀ 'Ca sĩ' TỪ DANH SÁCH DƯỚI ĐÂY. TUYỆT ĐỐI KHÔNG tự động sửa hay đổi tên ca sĩ thành người khác.\n\n"
                f"Kết quả nhận diện:\n{track_block}"
            )
        
        # 3. HÀNH ĐỘNG: GỢI Ý BÀI HÁT HOT/TOP HIT
        elif action == "RECOMMEND_POPULARITY":
            # Thêm sẵn chữ "/100 điểm" vào biến để AI copy nguyên xi
            pop_str = f" Cụ thể, hãy khoe rằng danh sách này có bài đạt điểm 'Độ hot Spotify' lên tới {max_pop}/100." if max_pop > 0 else ""
            
            llm_prompt = (
                f"Người dùng muốn xem các bài hát hot nhất hoặc top hit hiện tại.\n"
                f"Nhiệm vụ: Dựa vào danh sách dưới đây, hãy viết 2-3 câu giới thiệu cực kỳ hào hứng và bắt trend (kiểu như: 'Đây là những track đang làm mưa làm gió...').\n"
                f"Bạn HÃY ĐIỂM TÊN 2 đến 3 bài hát nổi bật nhất (Rút gọn tên cho tự nhiên) để dẫn dắt.{pop_str}\n\n"
                f"⚠️ QUY TẮC CỐT LÕI (PHẢI TUÂN THỦ NGHIÊM NGẶT):\n"
                f"- TUYỆT ĐỐI KHÔNG BỊA ĐIỂM SỐ. Bắt buộc dùng đúng con số đã cung cấp ở trên.\n"
                f"- Khi nhắc đến điểm số, BẮT BUỘC phải thêm thang điểm vào để câu văn rõ nghĩa (Ví dụ: '...đạt tới 71/100 điểm trên Spotify').\n"
                f"- Trả lời tự nhiên như một đoạn hội thoại. TUYỆT ĐỐI KHÔNG dùng gạch đầu dòng hay đánh số thứ tự 1, 2, 3.\n"
                f"- Không được nhắc lại toàn bộ 5 bài, vì giao diện đã hiển thị chi tiết rồi.\n"
                f"- Chỉ xưng là 'mình' hoặc 'VMusic AI'.\n"
                f"- KHÔNG dùng đại từ số nhiều (ví dụ: cấm dùng 'các bạn').\n\n"
                f"Danh sách bài hát Hot:\n{track_block}"
            )
        # 4. HÀNH ĐỘNG: GỢI Ý THEO TÂM TRẠNG/CẢM XÚC
        elif action == "RECOMMEND_MOOD":
            mood_req = params.get("mood", "tâm trạng này")
            llm_prompt = (
                f"Bạn là một người bạn tâm giao tinh tế. Người dùng đang muốn nghe nhạc với tâm trạng: '{mood_req}'.\n"
                f"Nhiệm vụ: Dựa vào danh sách dưới đây, hãy viết 2-3 câu an ủi, đồng cảm hoặc cổ vũ tinh thần, sau đó giới thiệu khéo léo.\n"
                f"(Ví dụ buồn: 'Mình hiểu cảm giác này. Thử nghe vài bản nhạc sâu lắng này để vơi đi nhé...' | Ví dụ vui: 'Tuyệt vời! Bật ngay list này để quẩy bung nóc nào...').\n"
                f"Bạn CÓ THỂ điểm tên nhẹ nhàng 1 hoặc tối đa 2 bài hát để dẫn dắt (Nhớ rút gọn tên cho tự nhiên).\n\n"
                f"⚠️ QUY TẮC CỐT LÕI (PHẢI TUÂN THỦ NGHIÊM NGẶT):\n"
                f"- TUYỆT ĐỐI KHÔNG liệt kê toàn bộ danh sách bài hát hay ca sĩ ra (vì giao diện đã tự vẽ thẻ nhạc rồi, bạn kể lể lại sẽ bị lủng củng).\n"
                f"- TUYỆT ĐỐI KHÔNG dùng gạch đầu dòng hay đánh số thứ tự 1, 2, 3.\n"
                f"- Chỉ xưng là 'mình' hoặc 'VMusic AI'.\n"
                f"- KHÔNG dùng đại từ số nhiều (ví dụ: cấm dùng 'các bạn').\n\n"
                f"Danh sách gợi ý:\n{track_block}"
            )
        
        # 5. HÀNH ĐỘNG: TÌM BÀI HÁT CỤ THỂ (SEARCH_NAME)
        elif action == "SEARCH_NAME":
            llm_prompt = (
                f"Người dùng đang tìm kiếm một bài hát cụ thể.\n"
                f"Nhiệm vụ: Dựa vào danh sách dưới đây, hãy viết 1 câu ngắn gọn, tự nhiên để xác nhận đã tìm thấy bài hát theo yêu cầu.\n"
                f"(Ví dụ: 'Mình tìm thấy ca khúc Nơi Này Có Anh của Sơn Tùng M-TP cho bạn rồi đây.' hoặc 'Đây là bản nhạc bạn đang tìm nhé.')\n\n"
                f"⚠️ QUY TẮC CỐT LÕI (PHẢI TUÂN THỦ NGHIÊM NGẶT):\n"
                f"- Nếu danh sách có nhiều kết quả (ví dụ bản gốc kèm bản lofi/remix), có thể nhắc khéo là hệ thống tìm được vài phiên bản bên dưới.\n"
                f"- Chỉ xưng 'mình'.\n\n"
                f"Ngữ cảnh yêu cầu: {request_ctx or 'N/A'}\n"
                f"Danh sách tìm được:\n{track_block}"
            )

        # 6. HÀNH ĐỘNG: CÁC TRƯỜNG HỢP CÒN LẠI (TÌM CA SĨ, TÊN BÀI, THỂ LOẠI...)
        else:
            llm_prompt = (
                f"Bạn là trợ lý âm nhạc V-Pop siêu nhiệt tình. Hãy viết 1-2 câu giới thiệu danh sách bài hát dưới đây, nói chuyện tự nhiên như một người bạn.\n"
                f"Hãy dựa vào 'Ngữ cảnh trích xuất' để biết người dùng đang tìm gì (Tìm nghệ sĩ, tìm thể loại, hay tìm lời bài hát) và điều chỉnh câu nói cho phù hợp.\n\n"
                f"User request: {user_prompt}\nAction: {action}\nNgữ cảnh trích xuất: {request_ctx or 'N/A'}\n\n"
                f"⚠️ QUY TẮC: Không gạch đầu dòng, không đánh số. GIỮ NGUYÊN tên bài và ca sĩ.\n"
                f"Danh sách gợi ý:\n{track_block}"
            )
            
        text = call_gemini_engine(llm_prompt, module='listener')
        text = str(text or '').strip()
        if text:
            return text + "\n\n"
    except Exception as e:
        print(f"Lỗi LLM Intro: {e}")

    # Fallback khi LLM gặp lỗi
    candidates = [
        "Mình chọn ra vài bài hợp với yêu cầu của bạn đây:\n\n",
        "Dưới đây là một vài gợi ý mình nghĩ bạn sẽ thích:\n\n",
        "Ok, mình gợi ý bạn những bài này nhé:\n\n",
        "Mình tìm được một danh sách khá ổn cho bạn đây:\n\n",
    ]
    try:
        h = hashlib.sha256((user_prompt + action).encode('utf-8')).hexdigest()
        idx = int(h[:2], 16) % len(candidates)
        return candidates[idx]
    except Exception:
        return candidates[0]
    
# --- 10.2 HEADER KHUNG CHAT ---

_TIPS_MD = """
💡 **Mẹo sử dụng để tìm nhạc nhanh và chính xác hơn**

*Để hệ thống hiểu đúng ý bạn ngay từ giây đầu tiên, bạn có thể tham khảo một số cách diễn đạt phổ biến dưới đây nhé!*\


━━━━━━━━━━━━━━━━━━
### 🔍 Tìm bài hát cụ thể
━━━━━━━━━━━━━━━━━━
- Nhập trực tiếp tên bài hát:
  
  → "Bạn hãy tìm bào hát Nơi này có anh cho tôi nhé."

  → "Bật cho tôi bài Chạy Ngay Đi của Sơn Tùng"

- Nếu nhớ một đoạn lời:
  
  → "Bài hát nào có đoạn mang tiền về cho mẹ?"
  
  → "Tìm bài có lời: em của ngày hôm qua"

- Nếu có file nhạc:
  
  → Tải lên file MP3/WAV và hỏi: "Bài này là bài gì thế?"

━━━━━━━━━━━━━━━━━━
### 🎧 Khám phá & gợi ý nhạc
━━━━━━━━━━━━━━━━━━
- Theo tâm trạng:
  
  → "Gợi ý vài bản nhạc buồn / sâu lắng / chữa lành / vui / cho tôi đi.", 
  
  → "Gợi ý nhạc chill, thư giãn sau ngày làm việc nhé", 
  
  → "Gợi ý cho tôi bài hát nào bùng nổ tí đi."

- Theo nghệ sĩ:
  
  → "Gợi ý cho tôi vài bài hát của Đen Vâu", "Playlist Bích Phương"

- Theo thể loại:
  
  → "Gợi ý nhạc thể loại rap"
  
  → "Gợi ý nhạc thể loại ballad"
  
  → "Gợi ý nhạc thể loại Indie"
  
  → "Gợi ý nhạc thể loại Pop cho tôi nhé"

- Tìm bài có phong cách tương tự:
  
  → "Có bài nào giống See Tình không?"

- Theo nhịp điệu / năng lượng:
  
  → "Nhạc nhanh để tập gym", "Nhạc nhẹ nhàng dễ ngủ"
  
  → "Tìm cho tôi vài bài có năng lượng mạnh đi"
  
  → "Tìm cho tôi bài có nhịp 160 bpm"

- Xem bài hát nổi bật:
  
  → "Top 5 bài hot nhất",
  
  → "List nhạc đang trending"
  
  → "List Nhạc đang viral"

━━━━━━━━━━━━━━━━━━
### 🧩 Tìm kiếm nâng cao
━━━━━━━━━━━━━━━━━━

- Bạn có thể kết hợp nhiều yếu tố:

  → "Nhạc rap buồn."  

  → "Nhạc buồn của Sơn Tùng m-tp"

  → "Hãy gợi ý nhạc buồn thể loại Ballad."

━━━━━━━━━━━━━━━━━━
### 🎼 Phân tích bài hát
━━━━━━━━━━━━━━━━━━
- Tải file nhạc và yêu cầu:
  
  → "Phân tích bài này có tiềm năng hit không?"
"""
# Chia 3 cột: Cột trái (trống để ép title ra giữa), Cột giữa (Title), Cột phải (Nút i)
# Dùng vertical_alignment="center" để mọi thứ căn giữa theo chiều dọc
col_spacer, col_title, col_info = st.columns([1.5, 7, 1.5], vertical_alignment="center")

with col_title:
    st.markdown("""
    <div style="text-align: center; padding: 0;">
        <h1 style="margin: 0; font-size: 2.35rem;">🎧 VMusic AI</h1>
    </div>
    """, unsafe_allow_html=True)

with col_info:
    # Dùng Material Icon thay vì Emoji Windows để nút trông "pro" hơn
    if hasattr(st, 'popover'):
        with st.popover(":material/info:"):
            st.markdown(_TIPS_MD)
    else:
        with st.expander(":material/info:", expanded=False):
            st.markdown(_TIPS_MD)

show_debug = os.getenv('CHATBOT_SHOW_ACTION_DEBUG') == '1'

# --- 10.3 KẾT NỐI LẠI LỊCH SỬ SUPABASE (DÙNG MODULE 'home') ---
_GREETING_MSG = {
    'role': 'assistant',
    'content': (
        "Chào bạn, mình là **VMusic AI**!\n\n"
        "Để mình giúp bạn tìm nhạc nhanh nhất, bạn có thể thử gõ:\n\n"
        "- \"Mở bài Nơi Này Có Anh\" *(Tìm tên bài)*\n"
        "- \"Gợi ý nhạc buồn\" *(Tìm theo cảm xúc)*\n"
        "- Hoặc mở mục **📎 Tải file** để tìm theo giai điệu / phân tích.\n\n"
        "Bạn có thể bấm nút **ℹ️** để xem thêm ví dụ."
    )
}

_load_chat_history_into_state(
    module='home',
    state_key='main_messages', 
    greeting=[_GREETING_MSG]
)

# Ép câu chào luôn nằm ở vị trí đầu tiên của session_state ngay cả khi load từ DB
if st.session_state.main_messages and st.session_state.main_messages[0].get('content') != _GREETING_MSG['content']:
    st.session_state.main_messages.insert(0, _GREETING_MSG)


def _md_preserve_newlines(text: str) -> str:
    """Keep single newlines visible when rendering with st.markdown()."""

    raw = str(text or '')
    # Markdown requires two spaces before newline for a line break.
    return raw.replace('\r\n', '\n').replace('\r', '\n').replace('\n', '  \n')

# Hiển thị lại toàn bộ lịch sử
for message in (st.session_state.main_messages or []):
    role = str(message.get('role', 'assistant'))
    with _aligned_chat_col(role):
        with st.chat_message(role):

            # Xử lý xuống dòng cho Markdown
            content = str(message.get('content', ''))
            if content:
                st.markdown(_md_preserve_newlines(content))

            # RENDER LẠI CARD SPOTIFY, BẢNG SHAP, DASHBOARD TỪ LỊCH SỬ
            audio_bytes = message.get('audio_bytes')
            if audio_bytes:
                st.audio(bytes(audio_bytes), format=_audio_mime_from_name(message.get('audio_name', 'audio.wav')))
            
            if message.get('spotify_artist_payload'):
                _render_spotify_artist_payload(message.get('spotify_artist_payload'))
            
            if message.get('track_previews'):
                _render_track_previews(message.get('track_previews'))
                
            if message.get('analysis_bundle'):
                bundle = message['analysis_bundle']
                render_dashboard(bundle)
                render_shap_payload_cached(bundle.get('shap_values'))

# --- 10.4 COMPOSER (Đưa nút đính kèm vào trong, giữ nguyên logic cũ) ---
if "chat_input_key" not in st.session_state:
    st.session_state.chat_input_key = 0

if 'pending_prompt' not in st.session_state:
    st.session_state.pending_prompt = None

# Cập nhật: Thêm dấu chấm (.) và các MIME types tiêu chuẩn để không bị lỗi đỏ
chat_result = st.chat_input(
    "Hỏi VMusic AI (hỗ trợ đính kèm MP3, WAV, TXT)", 
    accept_file="multiple", 
    file_type=[".mp3", ".wav", ".txt", "audio/mpeg", "audio/wav", "text/plain"]
)

typed = None
if chat_result:
    # Lấy text
    typed = chat_result.text if hasattr(chat_result, 'text') else chat_result.get('text')
    
    # Logic xử lý file giữ nguyên
    files = chat_result.files if hasattr(chat_result, 'files') else chat_result.get('files', [])
    for file in files:
        ext = os.path.splitext(file.name)[-1].lower()
        if ext in ['.mp3', '.wav']:
            st.session_state.global_audio_bytes = file.getvalue()
            st.session_state.global_audio_name = file.name
        elif ext == '.txt':
            try:
                st.session_state.global_lyric_text = file.getvalue().decode('utf-8', errors='ignore')
            except Exception:
                st.session_state.global_lyric_text = file.getvalue().decode('latin-1', errors='ignore')
            st.session_state.global_lyric_name = file.name

pending = st.session_state.pop('pending_prompt', None)
prompt = str(pending) if pending else typed

# Mọi biến logic bên dưới KHÔNG THAY ĐỔI
has_file = st.session_state.get('global_audio_bytes') is not None
has_lyric = bool(str(st.session_state.get('global_lyric_text') or '').strip())


# --- 10.5 XỬ LÝ YÊU CẦU MỚI (BẢN CHUẨN) ---
if prompt:
    # 1. Tạo chuỗi nội dung để hiển thị
    user_msg_to_save = prompt
    if has_file:
        user_msg_to_save += f"\n\n*(Có đính kèm file: {st.session_state.global_audio_name})*"
        if has_lyric:
            user_msg_to_save += f"\n*(Có đính kèm lyrics: {st.session_state.global_lyric_name})*"
    
    # [FIX LẶP] Chỉ append 1 lần duy nhất
    st.session_state.main_messages.append({'role': 'user', 'content': user_msg_to_save})
    
    # Lưu cờ trạng thái
    st.session_state.processing_prompt = prompt
    st.session_state.processing_has_file = has_file
    st.session_state.processing_has_lyric = has_lyric
    st.session_state.msg_to_persist = user_msg_to_save 

    st.rerun()
# ==========================================
# BẮT ĐẦU LUỒNG XỬ LÝ AI SAU KHI RERUN (GIAO DIỆN ĐÃ ĐƯỢC LÀM SẠCH)
# ==========================================
if st.session_state.get('processing_prompt'):
    p_prompt = st.session_state.processing_prompt
    p_has_file = st.session_state.processing_has_file
    p_has_lyric = st.session_state.get('processing_has_lyric', False)
    p_msg_save = st.session_state.pop('msg_to_persist', None)

    # ⏱️ [BƯỚC 1] BẮT ĐẦU ĐO TỔNG THỜI GIAN
    start_total = time.perf_counter()

    # MỞ HỘP THOẠI CỦA AI TRƯỚC TIÊN
    with _aligned_chat_col("assistant"):
        with st.chat_message("assistant"):

            # ⏳ BAO TRỌN TẤT CẢ VÀO 1 VÒNG XOAY TRONG HỘP THOẠI
            with st.spinner("⏳ Đang phân tích dữ liệu và tìm kiếm kết quả phù hợp nhất..."):
                
                # ⏱️ [BƯỚC 2] ĐO THỜI GIAN AI PHÂN TÍCH Ý ĐỊNH (INTENT)
                start_intent = time.perf_counter()
                intent_data = parse_intent_llm(p_prompt, has_file=p_has_file)
                intent_ms = (time.perf_counter() - start_intent) * 1000
            
                action = intent_data.get("action", "CLARIFY")
                params = intent_data.get("params", {})

                # ⏱️ [BƯỚC 3] BẮT ĐẦU ĐO THỜI GIAN BACKEND
                start_backend = time.perf_counter()

                if show_debug:
                    st.caption(f"*(AI Action: **{action}**)*")

                # --- LUỒNG 1: TÌM NHẠC ---
                if action in ["SEARCH_NAME", "SEARCH_LYRIC", "SEARCH_AUDIO", 
                            "RECOMMEND_MOOD", "RECOMMEND_ARTIST", "RECOMMEND_GENRE",
                            "ADVANCED_SEARCH", "RECOMMEND_SEED", "RECOMMEND_ATTRIBUTES", "RECOMMEND_POPULARITY"]:
                    
                    supabase_client = _get_supabase_client()
                    artist_list = _load_artist_list() if action in ['RECOMMEND_ARTIST', 'ADVANCED_SEARCH'] else None

                    temp_audio = None
                    params_to_use = params
                    try:
                        # [FIX SEARCH_AUDIO] Dùng p_has_file và check bytes trong session
                        if action == "SEARCH_AUDIO" and p_has_file and st.session_state.get('global_audio_bytes'):
                            audio_obj = io.BytesIO(st.session_state.global_audio_bytes)
                            audio_obj.name = st.session_state.global_audio_name
                            suffix = os.path.splitext(str(st.session_state.global_audio_name or ''))[-1].lower() or '.wav'
                            temp_audio = save_uploaded_file(audio_obj, suffix=suffix)
                            params_to_use = dict(params or {})
                            params_to_use['audio_path'] = temp_audio

                        result = _handle_action(
                            action, params_to_use, supabase_client,
                            embed_fn=_embed_query_text, has_file=p_has_file, artist_list=artist_list
                        )
                    finally:
                        if temp_audio: safe_remove(temp_audio)

                    # Normalize result into the same shape the renderer expects.
                    top_tracks: list[dict] = []
                    vector_result = {'source': None, 'error': None}
                    if isinstance(result, str):
                        vector_result['source'] = 'fallback-handle-action'
                        vector_result['error'] = result
                        top_tracks = []
                    elif isinstance(result, list):
                        top_tracks = _normalize_handle_action_rows([r for r in result if isinstance(r, dict)])
                        vector_result['source'] = 'live-supabase'
                    elif isinstance(result, dict):
                        # allow future extension: {'tracks': [...], 'source':..., 'error':...}
                        top_tracks = _normalize_handle_action_rows([r for r in (result.get('tracks') or []) if isinstance(r, dict)])
                        vector_result['source'] = result.get('source')
                        vector_result['error'] = result.get('error')
                        #LẤY ĐOẠN LỜI NHẠC:
                        vector_result['snippet'] = result.get('snippet', '')
                    else:
                        top_tracks = []
                        vector_result['source'] = 'fallback-handle-action'
                        vector_result['error'] = 'Kết quả truy vấn không hợp lệ'

                    # Show whether we're using live Supabase or a fallback (and why).
                    try:
                        src = str(vector_result.get('source') or '').strip()
                        err = str(vector_result.get('error') or '').strip()
                        if src or err:
                            err_one_line = re.sub(r"\s+", " ", err)[:220]
                            suffix = f" | Lỗi: {err_one_line}" if err_one_line else ""
                    except Exception:
                        pass

                    # SEARCH: keep top 5 by similarity (score desc).
                    if isinstance(top_tracks, list) and str(action).startswith('SEARCH_'):
                        try:
                            top_tracks = sorted(top_tracks, key=lambda t: float((t or {}).get('score', 0.0)), reverse=True)
                        except Exception:
                            pass
                    
                    if top_tracks:
                        # 1. Gọi câu chào mặc định của hệ thống
                        # Pass intent params (plus any backend meta) so LLM can speak convincingly.
                        intro_params = dict(params or {})
                        try:
                            if isinstance(result, dict):
                                seed_meta = result.get('seed_meta')
                                if isinstance(seed_meta, dict):
                                    intro_params.update({
                                        'seed_name': seed_meta.get('seed_name') or intro_params.get('seed_name'),
                                        'seed_vibe': seed_meta.get('seed_vibe'),
                                        'seed_genres': seed_meta.get('seed_genres'),
                                        'seed_title': seed_meta.get('seed_title'),
                                        'seed_artist': seed_meta.get('seed_artist'),
                                        'seed_tempo_bpm': seed_meta.get('seed_tempo_bpm'),
                                        'seed_rms_energy': seed_meta.get('seed_rms_energy'),
                                    })
                        except Exception:
                            pass

                        intro_text = _dynamic_intro_text(user_prompt=prompt, action=action, tracks=top_tracks, params=intro_params)

                        # Nếu là SEARCH_LYRIC và có snippet, nhờ Gemini viết lại câu chào cho "ngầu"
                        snippet = vector_result.get('snippet', '')
                        if action == "SEARCH_LYRIC" and snippet:
                            song_title = top_tracks[0].get('title', 'bài này')
                            artist_name = top_tracks[0].get('artist', '')
                            
                            # Prompt này giúp Gemini tạo ra câu nói tự nhiên như chúng ta đã bàn
                            ai_prompt = f"""
                            Bạn là chatbot V-Pop. Người dùng tìm từ khóa: '{params.get("lyric_snippet")}'. 
                            Hệ thống tìm thấy bài {song_title} của {artist_name} có đoạn lời: "{snippet}".
                            Hãy viết 1 câu giới thiệu ngắn gọn, thân thiện (có trích dẫn đoạn lời đó) để hỏi xem có đúng bài người dùng tìm không.
                            """
                            # Gọi Gemini (hàm call_gemini_engine đã có sẵn trong file của bạn)
                            llm_text = call_gemini_engine(ai_prompt, module='home')
                            if llm_text:
                                intro_text = llm_text

                        # 3. Xử lý preview Spotify
                        top5 = list(top_tracks[:5])
                        track_previews, popularity_by_id = _build_track_previews_from_spotify_batch(top5, batch_size=5)

                        # RECOMMEND: sort by Spotify popularity (desc) when available.
                        if str(action).startswith('RECOMMEND_') and popularity_by_id and str(action) != 'RECOMMEND_SEED':
                            def _pop_key(item: dict) -> int:
                                tid = str((item or {}).get('spotify_id') or '').strip()
                                return int(popularity_by_id.get(tid, -1))

                            top5_sorted = sorted(top5, key=_pop_key, reverse=True)
                            track_previews, _ = _build_track_previews_from_spotify_batch(top5_sorted, batch_size=5)
                        
                        # Sau đó mới hiển thị và lưu lịch sử như bình thường
                        st.markdown(intro_text)
                        _render_track_previews(track_previews) # Giữ nguyên luồng hiển thị card nhạc

                        # Lưu vào session để không bị mất khi F5
                        st.session_state.main_messages.append({
                            'role': 'assistant', 
                            'content': intro_text, 
                            'track_previews': track_previews
                        })
                        if p_msg_save:
                            _persist_chat_message(module='home', role='user', content=p_msg_save)
                        _persist_chat_message(module='home', role='assistant', content=intro_text)
                    else:
                        # Keep error/empty messages in the same chat style (avoid Streamlit warning boxes).
                        msg = str(vector_result.get('error') or '').strip() or "Không tìm thấy bài hát phù hợp."
                        st.markdown(_md_preserve_newlines(msg))
                        st.session_state.main_messages.append({'role': 'assistant', 'content': msg})
                        
                        # LƯU DB KHI KHÔNG TÌM THẤY BÀI HÁT
                        if p_msg_save:
                            _persist_chat_message(module='home', role='user', content=p_msg_save)
                        _persist_chat_message(module='home', role='assistant', content=msg)

                # --- LUỒNG 2: PHÂN TÍCH ---
                elif action == "ANALYZE_READY":
                    if not has_file or not has_lyric:
                        msg = "⚠️ Bạn hãy đính kèm file nhạc và lời bài hát để mình phân tích nhé."
                        st.warning(msg)
                        st.session_state.main_messages.append({'role': 'assistant', 'content': msg})
                        _persist_chat_message(module='home', role='assistant', content=msg)
                        st.stop()

                    audio_obj = io.BytesIO(st.session_state.global_audio_bytes)
                    audio_obj.name = st.session_state.global_audio_name
                    
                    try:
                        # Dùng spinner thay cho status để icon quay tròn hiện tự nhiên trong khung chat
                        with st.spinner('🔍 Đang đánh giá file nhạc của bạn...'):
                            temp_audio = save_uploaded_file(audio_obj, suffix='.wav')
                            from chatbot.analyze_ready_action import run_analyze_ready
                            supabase_client = _get_supabase_client()
                            
                            try:
                                bundle = run_analyze_ready(
                                    audio_path=temp_audio,
                                    lyric_text=st.session_state.get('global_lyric_text'),
                                    supabase_client=supabase_client,
                                    allow_download=False,
                                    compute_shap=True, # Giữ True để AI có dữ liệu đọc
                                    force_storage=True,
                                    skip_p1=True,
                                )
                            finally:
                                safe_remove(temp_audio)
                        
                        # Lấy lời khuyên từ AI (Hàm này Huy đã thêm thông số Tempo, Duration... ở bước trước)
                        advice = generate_arrangement_advice_llm(bundle)
                        
                        # Hiển thị trực tiếp lời khuyên ra khung chat
                        st.markdown(advice)
                        
                        # Lưu vào lịch sử (Chỉ lưu text lời khuyên để load lại không hiện bảng dashboard)
                        st.session_state.main_messages.append({
                            'role': 'assistant', 
                            'content': advice
                        })
                        _persist_chat_message(module='home', role='assistant', content=f"Đã phân tích xong bài hát.")
                    
                    except Exception as ex:
                        st.error(f'Lỗi hệ thống: {ex}')

                elif action == "GREETING":
                    msg = "Xin chào! Mình là VMusic AI, trợ lý âm nhạc V-Pop của bạn. Bạn có thể hỏi mình về bài hát, tìm nhạc theo tâm trạng, hoặc đính kèm file để mình phân tích nhé!"
                    st.markdown(msg)
                    st.session_state.main_messages.append({'role': 'assistant', 'content': msg})
                    
                    if p_msg_save:
                        _persist_chat_message(module='home', role='user', content=p_msg_save)
                    _persist_chat_message(module='home', role='assistant', content=msg)
                    
                elif action == "MISSING_FILE":
                    st.warning("Vui lòng đính kèm tệp âm thanh để hệ thống có thể thực hiện phân tích hoặc tìm kiếm dựa trên nội dung âm thanh.")
                    msg = "Vui lòng đính kèm file."
                    st.session_state.main_messages.append({'role': 'assistant', 'content': msg})
                    
                    if p_msg_save:
                        _persist_chat_message(module='home', role='user', content=p_msg_save)
                    _persist_chat_message(module='home', role='assistant', content=msg)

                elif action == "OUT_OF_SCOPE":
                    ans = call_gemini_engine(
                        f"Người dùng đang hỏi ngoài lề hoặc hỏi về bản thân bạn/người dùng. Hãy trả lời thân thiện dựa vào lịch sử trò chuyện. Câu hỏi: {prompt}", 
                        module='home'
                    )
                    msg = "Xin lỗi, mình là Trợ lý AI chuyên về âm nhạc V-Pop. Mình chỉ có thể giúp bạn tìm nhạc, phân tích bài hát hoặc trả lời các kiến thức về âm nhạc thôi!"
                    st.markdown(msg)
                    st.session_state.main_messages.append({'role': 'assistant', 'content': msg})
                    
                    if p_msg_save:
                        _persist_chat_message(module='home', role='user', content=p_msg_save)
                    _persist_chat_message(module='home', role='assistant', content=msg)
                    
                elif action == "CLARIFY":
                    msg = "Xin lỗi, mình chưa hiểu rõ ý bạn lắm. Bạn có thể nói rõ hơn là bạn muốn tìm bài hát, nghe nhạc theo tâm trạng, hay muốn mình phân tích file âm thanh không?"
                    st.markdown(msg)
                    st.session_state.main_messages.append({'role': 'assistant', 'content': msg})
                    
                    if p_msg_save:
                        _persist_chat_message(module='home', role='user', content=p_msg_save)
                    _persist_chat_message(module='home', role='assistant', content=msg)

                elif action == "MUSIC_KNOWLEDGE":
                    # Dùng p_prompt (chứa câu hỏi hiện tại) thay vì prompt
                    user_q = str(p_prompt or '').strip()
                    
                    # 1. Thử dùng hàm trả lời nhanh (Fast-path) đã định nghĩa ở trên
                    fast_ans = _answer_music_knowledge_local(user_q)
                    
                    if fast_ans:
                        msg = fast_ans
                    else:
                        # 2. Nếu không có sẵn, gọi AI với prompt sạch sẽ, không vòng lặp
                        knowledge_prompt = (
                            f"Bạn là chuyên gia âm nhạc và nhạc lý. Hãy trả lời câu hỏi sau một cách chính xác, dễ hiểu và ngắn gọn.\n"
                            f"Quy tắc: Đi thẳng vào câu trả lời, KHÔNG chào hỏi, KHÔNG gợi ý bài hát.\n\n"
                            f"Câu hỏi: {user_q}"
                        )
                        ans = call_gemini_engine(knowledge_prompt, module='home')
                        msg = ans if ans else "Hệ thống đang bận, bạn vui lòng thử lại sau nhé."
                    
                    # 3. Hiển thị và lưu lịch sử
                    st.markdown(msg)
                    st.session_state.main_messages.append({'role': 'assistant', 'content': msg})
                    
                    if p_msg_save:
                        _persist_chat_message(module='home', role='user', content=p_msg_save)
                    _persist_chat_message(module='home', role='assistant', content=msg)

                else:
                    ans = call_gemini_engine(f"Trả lời như chatbot: {prompt}", module='home')
                    msg = ans if ans else "Hệ thống hỏi đáp đang nâng cấp."
                    st.markdown(msg)
                    st.session_state.main_messages.append({'role': 'assistant', 'content': msg})
                    
                    if p_msg_save:
                        _persist_chat_message(module='home', role='user', content=p_msg_save)
                    _persist_chat_message(module='home', role='assistant', content=msg)
    
    # ⏱️ [KẾT THÚC] TÍNH TOÁN VÀ IN BÁO CÁO HIỆU SUẤT
    backend_ms = (time.perf_counter() - start_backend) * 1000
    total_ms = (time.perf_counter() - start_total) * 1000

    print(f"\n--- ⚡ [BÁO CÁO HIỆU SUẤT] ---")
    print(f"📝 Prompt: {p_prompt}")
    print(f"🎯 Action: {action} | Params: {json.dumps(params, ensure_ascii=False)}")
    print(f"🧠 Intent Latency:  {intent_ms:>8.2f} ms")
    print(f"⚙️ Backend Latency: {backend_ms:>8.2f} ms")
    print(f"🚀 Total Latency:   {total_ms:>8.2f} ms")
    print(f"📂 Audio: {p_has_file} | Lyric: {p_has_lyric}")
    
    # --- [MỚI] TRÍCH XUẤT DANH SÁCH BÀI HÁT CHO VÀO BÁO CÁO ---
    print(f"🎵 Danh sách bài hát xuất ra web:")
    try:
        # Dùng trực tiếp biến track_previews thay vì assistant_msg
        if 'track_previews' in locals() and track_previews:
            for idx, track in enumerate(track_previews, 1):
                t_title = track.get('title', 'Unknown')
                t_artist = track.get('artist', 'Unknown')
                t_pop = track.get('popularity', 'N/A')
                print(f"   {idx}. {t_title} - {t_artist} (Độ hot: {t_pop})")
        else:
            print("   (Không có bài hát nào được gợi ý hoặc đây là luồng hỏi đáp/phân tích)")
    except Exception as e:
        print(f"   (Lỗi ghi log bài hát: {e})")
        
    print(f"---------------------------\n")

    # --- [QUAN TRỌNG] CHỈ DỌN DẸP Ở CUỐI CÙNG ---
    st.session_state.global_audio_bytes = None
    st.session_state.global_audio_name = None
    st.session_state.global_lyric_text = None
    st.session_state.global_lyric_name = None
    st.session_state.chat_input_key += 1 # Reset thanh chat

    # Ensure the view stays at the newest message after rerun.
    _autoscroll_to_latest_chat()