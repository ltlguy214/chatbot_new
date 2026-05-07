from __future__ import annotations
from typing import Any, Callable
import os
import re
import unicodedata
import difflib

try:
    from rapidfuzz import process as _rf_process  # type: ignore
except Exception:  # pragma: no cover
    _rf_process = None

_GLOBAL_SONGS_CACHE = []

def _get_all_songs_cached(supabase_client):
    """
    Cache toàn bộ Metadata bài hát xuống RAM (chỉ tốn vài MB). 
    Phục vụ cho Local Fuzzy Search không dấu siêu tốc.
    """
    global _GLOBAL_SONGS_CACHE
    if _GLOBAL_SONGS_CACHE:
        return _GLOBAL_SONGS_CACHE
    if supabase_client is None:
        return []
    try:
        out = []
        start = 0
        while True:
            # Lấy đủ Data cho Ranker
            res = supabase_client.table('songs').select(
                'spotify_track_id, title, artists, vibe, main_topic, final_sentiment, spotify_popularity, is_hit, genres'
            ).range(start, start + 999).execute()
            rows = getattr(res, 'data', None) or []
            if not rows: break
            out.extend(rows)
            if len(rows) < 1000: break
            start += 1000
        _GLOBAL_SONGS_CACHE = out
        return out
    except Exception as e:
        print(f"[Cache Error] {e}")
        return []

def _extract_one(query: str, choices: list[str]) -> tuple[str, float, int] | None:
    """Return best match as (match, score, index).

    Uses rapidfuzz when available; falls back to stdlib difflib otherwise.
    Score is on a 0-100 scale.
    """

    if not query or not choices:
        return None

    if _rf_process is not None:
        try:
            match = _rf_process.extractOne(query, choices)
            if not match:
                return None
            m, score, idx = match
            return str(m), float(score), int(idx)
        except Exception:
            # Fall through to difflib.
            pass

    try:
        best = difflib.get_close_matches(query, choices, n=1, cutoff=0)
        if not best:
            return None
        m = best[0]
        try:
            idx = next(i for i, c in enumerate(choices) if c == m)
        except StopIteration:
            idx = 0
        score = difflib.SequenceMatcher(a=query, b=m).ratio() * 100.0
        return str(m), float(score), int(idx)
    except Exception:
        return None

def _normalize_text(text: str) -> str:
    if not text: return ""
    # Lowercase & Bỏ dấu chuẩn Unicode
    text = text.lower()
    text = unicodedata.normalize('NFD', text)
    text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
    # Xử lý chữ đ và ký tự đặc biệt
    text = text.replace('đ', 'd').replace('Đ', 'd')
    text = re.sub(r'[^a-z0-9\s]', '', text)
    # Trim khoảng trắng
    return re.sub(r'\s+', ' ', text).strip()

def _safe_embed(embed_fn: Callable[[str], Any] | None, text: str) -> list[float] | None:
    if embed_fn is None:
        return None
    try:
        vec = embed_fn(str(text or '').strip())
        if vec is None:
            return None
        if hasattr(vec, 'tolist'):
            vec = vec.tolist()
        if isinstance(vec, (list, tuple)) and len(vec) > 0:
            return [float(x) for x in vec]
        return None
    except Exception:
        return None


def _embed_error(embed_fn: Callable[[str], Any] | None) -> str:
    if embed_fn is None:
        return ''
    try:
        err = getattr(embed_fn, 'last_error', None)
        return str(err or '').strip()
    except Exception:
        return ''


def _normalize_track_rows(rows: Any) -> list[dict]:
    if not isinstance(rows, list):
        return []
    out: list[dict] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        spotify_id = r.get('spotify_id') or r.get('spotify_track_id') or r.get('track_id')
        title = r.get('title') or r.get('song_title') or r.get('track_name')
        artist = r.get('artist') or r.get('artists') or r.get('artist_name')
        if not spotify_id and not title:
            continue

        item = {
            'spotify_id': str(spotify_id or '').strip(),
            'title': str(title or '').strip(),
            'artist': str(artist or '').strip(),
            'vibe': r.get('vibe') or '',
            'main_topic': r.get('main_topic') or '',
            'final_sentiment': r.get('final_sentiment') or '',
            # Include these for UI display if needed
            'genres': r.get('genres') or r.get('genres') or '',
            'is_hit': r.get('is_hit') or 0
        }

        if 'similarity' in r:
            item['similarity'] = r.get('similarity')
        if 'score' in r:
            item['score'] = r.get('score')
        for key in [
            'tempo_bpm', 'rms_energy', 'beat_strength_mean', 'lexical_diversity',
            'delta_tempo_bpm', 'delta_rms_energy', 'delta_beat_strength_mean', 'delta_lexical_diversity',
            'spotify_popularity', 'tempo', 'energy', 'popularity'
        ]:
            if key in r and r[key] is not None:
                item[key] = r[key]
        out.append(item)
    return out


import math
from collections import defaultdict
from datetime import datetime

def rank_and_normalize_tracks(raw_rows: list[dict], limit: int = 5, boosts: dict = None) -> list[dict]:
    if not raw_rows: return []
    boosts = boosts or {}
    ranked = []
    dropped = [] 
    current_year = datetime.now().year

    # Pre-computation
    target_vibes = [_normalize_text(v) for v in (boosts.get('vibe') if isinstance(boosts.get('vibe'), list) else [boosts.get('vibe')] if boosts.get('vibe') else [])]
    target_topics = [_normalize_text(t) for t in (boosts.get('topics') if isinstance(boosts.get('topics'), list) else [boosts.get('topics')] if boosts.get('topics') else [])]
    
    target_sent = str(boosts.get('sentiment') or '').lower()
    target_title = _normalize_text(boosts.get('title') or '')
    target_artist = _normalize_text(boosts.get('artist') or '')
    
    # [FIX 2] Xử lý an toàn nếu genre truyền vào là List (Đa thể loại)
    raw_boost_genre = boosts.get('genre')
    if isinstance(raw_boost_genre, list):
        target_genres = [_normalize_text(g) for g in raw_boost_genre]
    else:
        target_genres = [_normalize_text(raw_boost_genre)] if raw_boost_genre else []

    target_seed_vibe = _normalize_text(boosts.get('seed_vibe') or '')
    
    target_seed_genre = _normalize_text(boosts.get('seed_genre') or '')
    action_mode = boosts.get('action_mode', 'search')

    # 1st PASS: MAIN SCORING
    for r in raw_rows:
        raw_sim = float(r.get('similarity') or r.get('score') or 0.0)
        sim = max(0.0, min(raw_sim, 1.0))
        is_vector = sim > 0

        # Drop hard vector rác
        if is_vector and sim < 0.3:
            dropped.append(r)
            continue

        pop_raw = float(r.get('spotify_popularity') or 50)
        pop = math.log1p(pop_raw) / math.log1p(100) 
        hit = min(1.0, float(r.get('is_hit') or 0))

        base_score = 0.0
        boost_score = 0.0
        penalty_score = 0.0

        raw_vibe = r.get('vibe') or ''
        db_vibe_list = [_normalize_text(v.strip()) for v in raw_vibe.replace('/', ',').split(',') if v.strip()]
        raw_topic = r.get('main_topic') or ''
        db_topic_padded = f" {_normalize_text(raw_topic)} " 
        db_title = _normalize_text(r.get('title') or r.get('song_title') or '')
        raw_artists = r.get('artists') or ''
        db_artist_list = [_normalize_text(a.strip()) for a in raw_artists.split(',') if a.strip()]
        raw_genre = r.get('genres') or ''
        db_genre_list = [_normalize_text(g.strip()) for g in raw_genre.replace('/', ',').split(',') if g.strip()]

        # BASE SCORE
        if action_mode == 'SEARCH_LYRIC': 
             # Trọng số 85% cho độ khớp lời, dẹp Popularity sang một bên
             base_score = (0.85 * sim) + (0.10 * pop) + (0.05 * hit)
        elif action_mode == 'mood':
            base_score = (0.5 * sim) + (0.2 * pop) + (0.3 * hit) if is_vector else (0.6 * pop) + (0.4 * hit)
        elif is_vector:
            base_score = (0.6 * sim) + (0.25 * pop) + (0.15 * hit)
        else:
            base_score = (0.7 * pop) + (0.3 * hit)

        if not raw_vibe and not raw_genre: penalty_score -= 0.05

        # DYNAMIC BOOSTS & PENALTIES
        if target_vibes:
            matched_vibe = False
            for v_norm in target_vibes:
                if v_norm in db_vibe_list: 
                    boost_score += 0.25; matched_vibe = True; break
                elif any(v_norm in db_v for db_v in db_vibe_list): 
                    boost_score += 0.15; matched_vibe = True; break
            if not matched_vibe: penalty_score -= 0.05

        if target_topics:
            match_count = sum(1 for t in target_topics if f" {t} " in db_topic_padded)
            if match_count > 0: boost_score += min(0.3, 0.1 * match_count)

        if target_sent:
            db_sent = str(r.get('final_sentiment') or '').lower()
            if db_sent:
                if target_sent == db_sent: boost_score += 0.15  
                else: penalty_score -= 0.1   

        if target_title:
            if target_title == db_title: boost_score += 0.3
            elif target_title in db_title or db_title in target_title: boost_score += 0.15

        if target_artist:
            if target_artist in db_artist_list: boost_score += 0.25
            elif any(target_artist in a for a in db_artist_list): boost_score += 0.15

        if target_genres:
            for tg in target_genres:
                if tg in db_genre_list: boost_score += 0.25
                elif any(tg in g for g in db_genre_list): boost_score += 0.15

        if boosts.get('target_tempo') and r.get('tempo_bpm'):
            diff = abs(float(boosts['target_tempo']) - float(r['tempo_bpm']))
            if diff <= 5: boost_score += 0.25
            elif diff <= 15: boost_score += 0.1

        if boosts.get('target_energy') and r.get('rms_energy'):
            diff = abs(float(boosts['target_energy']) - float(r['rms_energy']))
            if diff <= 0.1: boost_score += 0.25
            elif diff <= 0.2: boost_score += 0.1

        if target_seed_vibe and target_seed_vibe in db_vibe_list: boost_score += 0.1
        if target_seed_genre and target_seed_genre in db_genre_list: boost_score += 0.1

        release_date = r.get('spotify_release_date')
        if release_date:
            try:
                year = int(str(release_date)[:4])
                age = current_year - year
                if age <= 1: boost_score += 0.1
                elif age <= 3: boost_score += 0.05
            except: pass
        
        # =========================================================
        # TỔNG HỢP (PASS 1: Tính Raw Score chưa Scale)
        # =========================================================
        raw_score = base_score + boost_score + penalty_score
        
        r['score_breakdown'] = {
            'base': round(base_score, 4),
            'boosts': round(boost_score, 4),
            'penalties': round(penalty_score, 4)
        }
        r['raw_score'] = raw_score
        ranked.append(r)

    # Khôi phục dự phòng nếu thiếu bài
    if len(ranked) < limit and dropped:
        needed = limit - len(ranked)
        dropped.sort(key=lambda x: float(x.get('similarity') or 0), reverse=True)
        for r in dropped[:needed]:
            # Vector rác bị đày xuống đáy xã hội
            raw_score = float(r.get('similarity') or 0) - 2.0 
            r['score_breakdown'] = {'base': round(raw_score, 4), 'boosts': 0, 'penalties': 0}
            r['raw_score'] = raw_score
            ranked.append(r)

    # Sắp xếp lần 1: TUYỆT ĐỐI ƯU TIÊN ĐỘ TƯƠNG ĐỒNG (SIMILARITY)
    # Nếu tương đồng bằng nhau -> Mới dùng Popularity để xếp trên/dưới
    ranked.sort(
        key=lambda x: (
            float(x.get('similarity') or x.get('score') or 0.0), 
            float(x.get('spotify_popularity') or 0.0)
        ), 
        reverse=True
    )

    # =========================================================
    # PASS 2: DIVERSITY -> CLAMP -> NON-LINEAR SCALE
    # =========================================================
    final_list = []
    artist_count = defaultdict(int)

    for r in ranked:
        raw_artists = r.get('artists') or ''
        artist_main = _normalize_text(raw_artists.split(',')[0]) if raw_artists else ''

        # 1. Tính mức phạt trùng lặp
        if artist_main:
            div_penalty = 0.05 * (artist_count[artist_main] ** 1.2)
        else:
            div_penalty = 0.0
        r['score_breakdown']['diversity_penalty'] = round(-div_penalty, 4)

        # 2. Áp dụng phạt vào Raw Score
        current_raw = r['raw_score'] - div_penalty
        
        # 3. FIX CHUẨN: Clamp chặn đáy (Chống nổ math.exp)
        current_raw = min(3.0, max(-2.0, current_raw))

        # 4. Scale Non-linear
        final_score = 2.0 * (1 - math.exp(-0.8 * current_raw))
        r['final_mix_score'] = final_score

        final_list.append(r)
        if artist_main:
            artist_count[artist_main] += 1

    # Sắp xếp lần 2: Thứ tự hiển thị cuối cùng
    # Tương tự, lấy Similarity làm "Vua", Popularity làm "Á quân"
    final_list.sort(
        key=lambda x: (
            float(x.get('similarity') or x.get('score') or 0.0), 
            float(x.get('spotify_popularity') or 0.0)
        ), 
        reverse=True
    )

    # =========================================================
    # PASS 3: GLOBAL SOFTMAX (FIX: Tính trên toàn bộ Pool)
    # =========================================================
    if final_list:
        scores = [r['final_mix_score'] for r in final_list]
        max_score = max(scores) if scores else 0
        exp_scores = [math.exp(s - max_score) for s in scores]
        sum_exp = sum(exp_scores)
        
        for i, r in enumerate(final_list):
            r['prob'] = exp_scores[i] / sum_exp if sum_exp > 0 else 0.0

    # Trả về Top K (Cắt sau khi đã Softmax)
    return _normalize_track_rows(final_list[:limit])

def _artist_index(artist_list: Any) -> tuple[list[str], list[str]]:
    """Return (original_artists, normalized_artists) for rapidfuzz."""
    if not artist_list:
        return [], []
    if isinstance(artist_list, dict):
        a = artist_list.get('artists')
        n = artist_list.get('normalized')
        if isinstance(a, list) and isinstance(n, list) and len(a) == len(n):
            return [str(x) for x in a], [str(x) for x in n]
    if isinstance(artist_list, list):
        artists = [str(x or '').strip() for x in artist_list if str(x or '').strip()]
        normalized = [_normalize_text(a) for a in artists]
        return artists, normalized
    return [], []


def _find_best_artist(user_input: str, artist_list: Any) -> str | None:
    artists, normalized_artists = _artist_index(artist_list)
    if not artists or not normalized_artists:
        return None
    query = _normalize_text(user_input)
    if not query:
        return None
    match = _extract_one(query, normalized_artists)
    if not match:
        return None
    _m, score, idx = match
    try:
        if float(score) > 70:
            return artists[int(idx)]
    except Exception:
        return None
    return None


def _load_artist_list_from_supabase(supabase: Any, *, max_rows: int = 20000) -> list[str]:
    """Best-effort artist-name loader from Supabase table `artists`.

    This is a fallback for cases where the UI doesn't pass `artist_list`.
    Keep it bounded to avoid heavy work per request.
    """

    if supabase is None:
        return []

    page_size = 1000
    max_rows = max(0, int(max_rows))
    if max_rows == 0:
        return []

    try:
        max_rows_env = str(os.getenv('CHATBOT_ARTISTS_MAX_ROWS', '') or '').strip()
        if max_rows_env.isdigit():
            max_rows = max(1000, min(200000, int(max_rows_env)))
    except Exception:
        pass

    for col in ['artist_name', 'name', 'artist']:
        out: list[str] = []
        start = 0
        try:
            while start < max_rows:
                end = start + page_size - 1
                resp = supabase.table('artists').select(col).range(start, end).execute()
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
        except Exception:
            continue

        if out:
            return list(dict.fromkeys(out))

    return []


def _mood_maps(mood_text: str) -> tuple[list[str], list[str], str]:
    """
    BÓC TÁCH ĐỘC LẬP: Chỉ gán nhãn nếu người dùng thực sự nhắc đến.
    Các nhãn (labels) trả về BẮT BUỘC phải là chuỗi con (substring) khớp với Database.
    Return: (target_vibes_list, target_topics_list, sentiment)
    """
    if not mood_text:
        return [], [], ""
        
    m = _normalize_text(mood_text)
    
    target_vibes = []
    target_topics = []
    target_sentiment = ""
    
    # ==========================================
    # 1. BẮT TÍN HIỆU SENTIMENT (Cảm xúc cốt lõi)
    # ==========================================
    if any(k in m for k in ['buon', 'suy','sau', 'sad', 'tieu cuc', 'khoc', 'nang long']):
        target_sentiment = "negative"
    elif any(k in m for k in ['vui', 'happy', 'tich cuc', 'yeu doi']):
        target_sentiment = "positive"
        
    # ==========================================
    # 2. BẮT TÍN HIỆU VIBE (Phải khớp chuỗi con trong DB)
    # Các nhãn có trong DB: "Kịch tính / Da diết", "Sâu lắng / Thấu cảm", 
    # "Bình yên / Chữa lành", "Tươi mới / Yêu đời", "Bùng nổ / Sôi động"
    # ==========================================
    if any(k in m for k in ['da diet', 'kich tinh', 'cao trao', 'dằn vặt', 'dan vat', 'não nề', 'nao ne']):
        target_vibes.append("Kịch tính") # Sẽ khớp "Kịch tính / Da diết"
        
    if any(k in m for k in ['sau lang', 'tham', 'sau tham', 'thau cam']):
        target_vibes.append("Sâu lắng") # Sẽ khớp "Sâu lắng / Thấu cảm"

    # "Hoài cổ" thường được người dùng mô tả theo cảm giác hoài niệm + sâu lắng.
    # Giữ nó như một tín hiệu riêng để không làm lệch các query chỉ có "kỷ niệm".
    if any(k in m for k in ['hoai co', 'co xua', 'xua cu', 'xua']):
        if "Kỷ niệm" not in target_topics:
            target_topics.append("Kỷ niệm")
        if "Sâu lắng" not in target_vibes:
            target_vibes.append("Sâu lắng")
        
    if any(k in m for k in ['ru ngủ','ru ngu', 'an ui', 'an ủi', 'stress', 'chill', 'thu gian', 'binh yen', 'lofi', 'nhe nhang', 'healing', 'chua lanh']):
        target_vibes.append("Bình yên") # Sẽ khớp "Bình yên / Chữa lành"
        
    if any(k in m for k in ['tuoi moi', 'yeu doi', 'động lực', 'dong luc']):
        target_vibes.append("Tươi mới") # Sẽ khớp "Tươi mới / Yêu đời"
        
    if any(k in m for k in ['gym', 'quay','quẩy', 'party', 'sung', 'chay', 'soi dong', 'bung no', 'nang luong', 'xập xình', 'xap xinh', 'cháy máy', 'chay may']):
        target_vibes.append("Bùng nổ") # Sẽ khớp "Bùng nổ / Sôi động"
        
    # ==========================================
    # 3. BẮT TÍN HIỆU TOPIC (Phải khớp chuỗi con trong DB)
    # Các nhãn có trong DB: "Ballad Thất tình / Chia tay", "Hoài niệm / Kỷ niệm", 
    # "Tình yêu đôi lứa / Lãng mạn", "Tình cảm Gia đình / Cha mẹ", "Nhạc Tết / Xuân",...
    # ==========================================
    if any(k in m for k in ['that tinh', 'chia tay', 'dau long', 'phan boi', 'co don', 'thui ruot']):
        target_topics.append("Thất tình") # Khớp "Ballad Thất tình / Chia tay"
        
    if any(k in m for k in ['tinh yeu', 'ngot ngao', 'lang man']):
        target_topics.append("Tình yêu") # Khớp "Tình yêu đôi lứa / Lãng mạn"
        
    if any(k in m for k in ['ky niem', 'qua khu', 'hoai niem', 'nho nhung']):
        target_topics.append("Kỷ niệm") # Khớp "Hoài niệm / Kỷ niệm"
        
    if any(k in m for k in ['gia dinh', 'me', 'cha', 'que huong']):
        target_topics.append("Gia đình") # Khớp "Tình cảm Gia đình / Cha mẹ"
        
    if any(k in m for k in ['tha thinh', 'tan tinh', 'crush']):
        target_topics.append("Thả thính") # Khớp "Pop hiện đại / Thả thính"
        
    if any(k in m for k in ['tet', 'xuan', 'nam moi']):
        target_topics.append("Xuân") # Khớp "Nhạc Tết / Xuân"
        
    if any(k in m for k in ['tu hao', 'yeu nuoc', 'que huong']):
        target_topics.append("Tự hào") # Khớp "Lòng yêu nước / Tự hào"
    
    # Loại bỏ các giá trị trùng lặp nếu có
    return list(set(target_vibes)), list(set(target_topics)), target_sentiment


def get_genre_target(genre_text: str) -> str:
    if not genre_text:
        return ""
        
    text = _normalize_text(genre_text)
    raw = genre_text.lower()
    
    # Bắt chuẩn cả chữ có dấu và không dấu
    if any(k in text or k in raw for k in ['rap', 'hip hop', 'hiphop', 'underground', 'trap']): 
        return "Rap/Hip-hop"
        
    if any(k in text or k in raw for k in ['ballad']): 
        return "Ballad"
        
    if any(k in text or k in raw for k in ['indie']): 
        return "Indie"
        
    if any(k in text or k in raw for k in ['pop', 'nhac tre', 'nhạc trẻ', 'vpop', 'mainstream', 'hien dai', 'hiện đại']): 
        return "Pop"
    
    return genre_text

def get_genre_targets(genre_text: str) -> list[str]:
    """Băm chuỗi đa thể loại (ngăn cách bởi dấu phẩy) và trả về list chuẩn"""
    if not genre_text:
        return []
    
    # Cắt chuỗi dựa trên dấu phẩy
    raw_genres = [g.strip() for g in genre_text.split(',')]
    mapped = []
    
    for g in raw_genres:
        target = get_genre_target(g) # Dùng lại hàm dịch (số ít) đã có
        if target and target not in mapped:
            mapped.append(target)
            
    return mapped


def handle_action(
    action: str,
    params: dict,
    supabase: Any,
    embed_fn: Callable[[str], Any] | None = None,
    has_file: bool = False,
    artist_list=None,
    *,
    match_threshold: float | None = None,
    match_count: int = 5,
) -> Any:
    """Route action -> Supabase query.

    Notes:
    - Vector RPC must be called as: match_vpop_tracks(query_embedding, match_threshold, match_count)
    - `embed_fn` should return a 1-D vector (list/np.ndarray)
    """

    action = str(action or "").strip().upper()
    params = params if isinstance(params, dict) else {}

    if supabase is None:
        return {
            'tracks': [],
            'source': 'fallback-no-supabase-client',
            'error': 'Chưa kết nối được Supabase client',
            'path': ["Level 0: Precheck", "Error: No Supabase"],
        }

    # =========================
    # 1. SEARCH_NAME (Tìm Tên Bài Hát + Nghệ sĩ)
    # =========================
    elif action == "SEARCH_NAME":
        execution_path = ["Level 1: SQL Exact"]
        song_title = str(params.get("song_title") or "").strip()
        artist = str(params.get("artist", "") or "").strip()

        if not song_title:
            return {
                'tracks': [],
                'source': 'fallback-missing-param',
                'error': 'Bạn muốn tìm bài hát tên gì nhỉ?',
                'path': execution_path,
            }

        print(f"[SEARCH_NAME] Đầu vào: Title='{song_title}', Artist='{artist}'")

        try:
            import re
            rows = []
            source_label = ""
            original_song_title = song_title

            # Hàm con dùng để search SQL tái sử dụng nhiều lần
            def try_sql_search(title_query, artist_query):
                import unicodedata
                import re
                
                q = supabase.table('songs').select(
                    'spotify_track_id, title, artists, vibe, main_topic, final_sentiment, spotify_popularity, is_hit, genres'
                )
                
                # 1. BẮT BUỘC CHUẨN HÓA NFC: Trị triệt để bệnh lệch bảng mã từ bàn phím Apple (NFD) so với DB (NFC)
                clean_title = unicodedata.normalize('NFC', title_query.strip())
                
                # 2. Xử lý dính chữ thông minh
                if " " not in clean_title and len(clean_title) >= 4:
                    # Ráp `%` trực tiếp vào chuỗi NFC sẽ không làm đứt gãy các dấu câu tiếng Việt
                    spaced_title = "%".join(clean_title)
                    q = q.ilike('title', f'%{spaced_title}%')
                else:
                    q = q.ilike('title', f'%{clean_title}%')
                    
                # 3. Xử lý triệt để tên Nghệ sĩ (Bao quát mọi trường hợp có/không gạch nối)
                if artist_query:
                    clean_artist = unicodedata.normalize('NFC', artist_query.strip())
                    # Chuyển đổi mọi dấu cách, dấu gạch ngang thành wildcard (Bắt được cả 'Sơn Tùng M-TP' và 'Sơn Tùng MTP')
                    artist_wildcard = re.sub(r'[-\s]+', '%', clean_artist)
                    q = q.ilike('artists', f'%{artist_wildcard}%')
                    
                res = q.limit(int(match_count) * 4).execute()
                return getattr(res, 'data', None) or []

            # --- LỚP 0.1: TÌM BẢN GỐC ---
            rows = try_sql_search(song_title, artist)
            if rows:
                source_label = 'text-search:exact'
                for r in rows: r['similarity'] = 1.0

            if not rows:
                print(f"[SEARCH_NAME] Không tìm thấy bản gốc. Bắt đầu phẫu thuật chuỗi: '{song_title}'")
                
                # --- LỚP 1: Xử Lý Số Đếm Mở Rộng ---
                # FIX: Đảo thứ tự Map ưu tiên chuỗi dài trước, XÓA bỏ "năm" -> "5" để tránh hỏng chữ "10k năm"
                num_map = {
                    " mười k ": " 10k ", " mười ": " 10 ", " một ": " 1 ", " hai ": " 2 ", " ba ": " 3 ", 
                    " bốn ": " 4 ", " sáu ": " 6 ", " bảy ": " 7 ", " tám ": " 8 ", " chín ": " 9 ", 
                    " ngàn ": "k ", " phần ": "/"
                }
                padded_title = f" {song_title.lower()} "
                for word, digit in num_map.items():
                    padded_title = padded_title.replace(word, digit)
                padded_title = re.sub(r'(\d)\s*/\s*(\d)', r'\1/\2', padded_title)
                numeric_title = padded_title.strip()
                
                if numeric_title != song_title:
                    execution_path.append("Level 1.5: Numeric Mapping")
                    rows = try_sql_search(numeric_title, artist)
                    if rows:
                        source_label = 'text-search:numeric'
                        song_title = numeric_title 
                        for r in rows: r['similarity'] = 0.95

                # --- LỚP 1.5: LOCAL SMART FUZZY (Cứu cánh không dấu, sai chính tả, tiền tố nhiễu) ---
                # --- LỚP 1.5: PRODUCTION-GRADE FUZZY STICKY MATCH ---
                if not rows:
                    all_songs = _get_all_songs_cached(supabase)
                    from rapidfuzz import fuzz # Đảm bảo đã cài rapidfuzz
                    
                    if all_songs:
                        execution_path.append("Level 2: RapidFuzz Sticky Engine")
                        # Chuẩn hóa dính liền để trị 'chayngaydi'
                        q_t_sticky = _normalize_text(song_title).replace(" ", "")
                        q_a_norm = _normalize_text(artist)
                        
                        scored = []
                        for s in all_songs:
                            db_t_norm = _normalize_text(s.get('title') or '')
                            db_t_sticky = db_t_norm.replace(" ", "")
                            db_a_norm = _normalize_text(s.get('artists') or '')
                            
                            # 1. Tính điểm khớp Tên bài (Fuzzy Ratio)
                            title_score = fuzz.ratio(q_t_sticky, db_t_sticky)
                            
                            # Chấp nhận nếu khớp trên 85% hoặc là substring của nhau
                            if title_score > 85 or q_t_sticky in db_t_sticky or db_t_sticky in q_t_sticky:
                                sim = max(0.85, title_score / 100.0)
                                
                                # 2. Tính điểm khớp Nghệ sĩ (Token Set Ratio để né 'Official', 'M-TP')
                                if q_a_norm:
                                    artist_score = fuzz.token_set_ratio(q_a_norm, db_a_norm)
                                    if artist_score > 75:
                                        sim += 0.05 # Thưởng đúng nghệ sĩ
                                    else:
                                        sim -= 0.35 # Phạt nặng nếu sai nghệ sĩ
                                
                                if sim > 0.6:
                                    s_copy = dict(s)
                                    s_copy['similarity'] = round(sim * 100, 2)
                                    scored.append(s_copy)
                        
                        if scored:
                            scored.sort(key=lambda x: x['similarity'], reverse=True)
                            rows = scored[:int(match_count)]
                            source_label = 'production-fuzzy-sticky'
                
                # --- 2. Bóc Tách Ca Sĩ Bằng Fuzzy Token ---
                if not rows and not artist:
                    execution_path.append("Level 2: Entity Disambiguation")
                    words = numeric_title.split()
                    if len(words) >= 3:
                        found_artist = False
                        for i in range(min(4, len(words) - 1), 0, -1):
                            potential_artist_str = " ".join(words[-i:])
                            if len(potential_artist_str) < 3: continue
                            
                            ar_res = supabase.table('artists').select('artist_name').ilike('artist_name', f'%{potential_artist_str}%').limit(10).execute()
                            ar_rows = getattr(ar_res, 'data', None) or []
                            
                            for row in ar_rows:
                                db_artist = row.get('artist_name', '')
                                if len(db_artist) >= 3 and _normalize_text(potential_artist_str) in _normalize_text(db_artist):
                                    artist = db_artist
                                    song_title = " ".join(words[:-i]).strip()
                                    print(f"[SEARCH_NAME] AI Bóc Tách Dính Chữ -> Title: '{song_title}', Artist: '{artist}'")
                                    found_artist = True
                                    
                                    rows = try_sql_search(song_title, artist)
                                    if rows:
                                        source_label = 'text-search:split-artist'
                                        for r in rows: r['similarity'] = 0.90
                                        break
                            if found_artist: break

                # --- LỚP 2: Fallback Bỏ tên ca sĩ ---
                if not rows and artist:
                    execution_path.append("Level 2: Attribute Fallback")
                    print(f"[Fallback] Bỏ qua ca sĩ '{artist}'...")
                    rows = try_sql_search(song_title, "")
                    if rows:
                        source_label = 'text-search:name-only-fallback'
                        for r in rows: r['similarity'] = 0.9

                # --- LỚP 3: MÀNG LỌC FUZZY + VECTOR SIÊU CẤP ---
                if not rows:
                    execution_path.append("Level 2: Non-accent Trigram")
                    full_query = f"{song_title} {artist}".strip()
                    clean_query = _normalize_text(full_query) # Loại bỏ HOÀN TOÀN dấu tiếng Việt
                    print(f"[Fallback] Kích hoạt Non-Accent Vector & Fuzzy cho: '{clean_query}'...")
                    
                    try:
                        # 1. Thử Vét Cạn Bằng Trigram Fuzzy (Tốt nhất cho chữ không dấu)
                        fuzzy_res = supabase.rpc('match_lyrics_fuzzy', {
                            'query_text': clean_query,
                            'match_threshold': 0.15, # Vét cạn đáy
                            'match_count': int(match_count) * 4
                        }).execute()
                        
                        f_rows = getattr(fuzzy_res, 'data', None) or []
                        if f_rows:
                            # Fuzzy trả về ID, cần map lại lấy Metadata
                            ids = [r['spotify_track_id'] for r in f_rows]
                            meta_res = supabase.table('songs').select('*').in_('spotify_track_id', ids).execute()
                            meta_map = {m['spotify_track_id']: m for m in (meta_res.data or [])}
                            
                            for f in f_rows:
                                if f['spotify_track_id'] in meta_map:
                                    row = meta_map[f['spotify_track_id']]
                                    # Bơm điểm để Ranker ưu tiên Fuzzy
                                    row['similarity'] = min(0.96, f.get('similarity', 0.5) + 0.3)
                                    rows.append(row)
                            
                            if rows:
                                source_label = 'fuzzy-trigram:non-accent'
                                
                    except Exception as e:
                        print(f"[Lỗi Fuzzy Fallback] {e}")
            
            # --- KẾT THÚC ---
            if not rows:
                return {
                    'tracks': [],
                    'source': 'search-name-empty',
                    'error': f"Tiếc quá, hệ thống hiện chưa có bài '{original_song_title}'.",
                    'path': execution_path,
                }
            # =======================================================
            # GLOBAL RANKER
            # =======================================================
            ranked_tracks = rank_and_normalize_tracks(
                raw_rows=rows,
                limit=int(match_count),
                boosts={'title': song_title, 'artist': artist}
            )

            return {
                'tracks': ranked_tracks, 
                'source': source_label, 
                'error': None,
                'path': execution_path,
            }
        
        except Exception as ex:
            return {
                'tracks': [],
                'source': 'search-name-error',
                'error': f"Lỗi tìm kiếm: {ex}",
                'path': execution_path,
            }


    # =========================
    # 2. SEARCH_LYRIC (Optimized: Title -> Fuzzy Trigram -> Substring Chunking)
    # =========================
    elif action == "SEARCH_LYRIC":
        execution_path = ["Level 1: Title Match"]
        lyric = str(params.get("lyric_snippet", "") or "").strip().strip("'").strip('"')
        if not lyric:
            return {
                'tracks': [],
                'source': 'fallback-missing-param',
                'error': 'Bạn muốn tìm theo đoạn lời nào?',
                'path': execution_path,
            }

        print(f"[SEARCH_LYRIC] Bắt đầu tìm kiếm thuần chuỗi: '{lyric}'")
        
        # --- 1. CHUẨN HÓA VĂN BẢN ---
        def deep_clean(t):
            import re 
            t = re.sub(r'[\n\r\,\.\?\!\-]', ' ', t).lower()
            return re.sub(r'\s+', ' ', t).strip()

        clean_query = deep_clean(lyric)
        is_unaccented = (clean_query == _normalize_text(clean_query))
        pool = []

        try:
            # --- TẦNG 1: TÌM TIÊU ĐỀ (ĐIỂM TUYỆT ĐỐI) ---
            t_res = supabase.table('songs').select(
                'spotify_track_id, title, artists, vibe, main_topic, spotify_popularity, is_hit, genres'
            ).ilike('title', f'%{clean_query}%').limit(3).execute()
            
            if t_res.data:
                for r in t_res.data:
                    row = dict(r)
                    row['similarity'] = 1.0
                    pool.append(row)

            # --- TẦNG 2: FUZZY TRIGRAM TOÀN CÂU ---
            execution_path.append("Level 2: Fuzzy Trigram")
            fuzzy_thresh = 0.04 if is_unaccented else 0.10
            l_res = supabase.rpc('match_lyrics_fuzzy', {
                'query_text': clean_query,
                'match_threshold': fuzzy_thresh,
                'match_count': 15
            }).execute()

            if l_res.data:
                c_ids = [r['spotify_track_id'] for r in l_res.data]
                c_meta = supabase.table('songs').select('*').in_('spotify_track_id', c_ids).execute()
                c_map = {s['spotify_track_id']: s for s in (c_meta.data or [])}
                
                for match in l_res.data:
                    tid = match['spotify_track_id']
                    if tid in c_map:
                        row = dict(c_map[tid])
                        row['similarity'] = float(match['similarity'])
                        pool.append(row)

            # --- TẦNG 3: SUBSTRING CHUNKING (HẠ SÁCH CUỐI CÙNG) ---
            max_sim = max([r['similarity'] for r in pool]) if pool else 0.0
            words = clean_query.split()
            
            # Khởi động Chunking nếu Fuzzy toàn câu không tìm ra kết quả tốt (>85%)
            if max_sim < 0.85 and len(words) >= 5:
                execution_path.append("Level 3: Substring Chunking")
                print(f" -> Level 3: Điểm max = {max_sim:.2f}. Bắt đầu Chunking cứu viện...")
                
                # Cắt 4 chữ đầu và 4 chữ cuối
                chunks = [" ".join(words[:4]), " ".join(words[-4:])]
                
                for chunk in chunks:
                    chunk_res = supabase.rpc('match_lyrics_fuzzy', {
                        'query_text': chunk,
                        'match_threshold': 0.10,
                        'match_count': 3
                    }).execute()
                    
                    if chunk_res.data:
                        c_ids = [r['spotify_track_id'] for r in chunk_res.data]
                        c_meta = supabase.table('songs').select('*').in_('spotify_track_id', c_ids).execute()
                        c_map = {s['spotify_track_id']: s for s in (c_meta.data or [])}
                        
                        for match in chunk_res.data:
                            tid = match['spotify_track_id']
                            if tid in c_map:
                                row = dict(c_map[tid])
                                row['similarity'] = float(match['similarity'])
                                pool.append(row)

            # --- RANKING & DEDUP ---
            if pool:
                unique = {}
                for r in pool:
                    tid = r.get('spotify_track_id') or r.get('spotify_id')
                    # Giữ lại bản ghi có điểm cao nhất nếu bị trùng
                    if tid and (tid not in unique or r['similarity'] > unique[tid]['similarity']):
                        unique[tid] = r
                
                artist_query = str(params.get("artist", "")).strip()
                
                ranked_tracks = rank_and_normalize_tracks(
                    raw_rows=list(unique.values()),
                    limit=int(match_count),
                    boosts={'action_mode': 'SEARCH_LYRIC', 'title': clean_query, 'artist': artist_query}
                )        
                return {'tracks': ranked_tracks, 'source': 'smart-fuzzy-logic', 'error': None, 'path': execution_path}
            
            return {'tracks': [], 'error': f"Không tìm thấy bài hát nào khớp với: '{lyric}'.", 'path': execution_path}

        except Exception as e:
            return {'tracks': [], 'error': f"Lỗi logic: {e}", 'path': execution_path}
        

    # =========================
    # 3. SEARCH_AUDIO (Nâng cấp: Dùng Backend Scaler + Multi-segment)
    # =========================
    elif action == "SEARCH_AUDIO":
        execution_path = ["Level 1: Audio Fingerprint"]
        # params lúc này chứa 'audio_path' do Streamlit gửi qua
        audio_path = params.get("audio_path")
        
        if not audio_path or not os.path.exists(audio_path):
            return {
                'tracks': [], 
                'source': 'error', 
                'error': 'Mình không tìm thấy file âm thanh của bạn. Vui lòng tải file lên nhé!',
                'path': execution_path,
            }

        try:
            # Import Backend tại chỗ để tránh lỗi vòng lặp (circular import)
            try:
                from chatbot.analysis_backend import VPopAnalysisBackend
            except ModuleNotFoundError:
                from analysis_backend import VPopAnalysisBackend
            
            # Khởi tạo Backend (Nó sẽ tự động load models/audio_scaler.joblib)
            backend = VPopAnalysisBackend(supabase_client=supabase)
            
            print(f"[SEARCH_AUDIO] Đang phân tích dấu vân tay âm thanh chuẩn hóa cho: {os.path.basename(audio_path)}")
            
            # Gọi hàm search đã được tối ưu (Lấy 3 đoạn 30s-60s-90s và ép Z-score)
            res = backend.search_similar_tracks(
                audio_path=audio_path,
                match_count= 1
            )

            if res.get("error"):
                return {'tracks': [], 'source': 'audio-search-error', 'error': res["error"], 'path': execution_path}

            tracks_data = res.get("tracks", [])

            return {
                'tracks': _normalize_track_rows(tracks_data)[:1], 
                'source': 'audio-similarity-scientific-40d',
                'error': None,
                'path': execution_path,
            }
            
        except Exception as e:
            return {
                'tracks': [], 
                'source': 'error', 
                'error': f'Lỗi quy trình xử lý âm thanh AI: {e}',
                'path': execution_path,
            }


    # =========================
    # 4. RECOMMEND_MOOD
    # =========================
    elif action == "RECOMMEND_MOOD":
        execution_path = ["Level 1: Mood Mapping"]
        mood_query = str(params.get("mood") or "").strip()
        if not mood_query:
            return {
                'tracks': [],
                'source': 'error',
                'error': 'Bạn muốn nghe nhạc theo tâm trạng như thế nào? Hãy nói cho mình biết nhé.',
                'path': execution_path,
            }

        # LỚP 1: LẤY ĐÚNG TÍN HIỆU NGƯỜI DÙNG NHẮC TỚI
        target_vibes, target_topics, target_sentiment = _mood_maps(mood_query)
        print(f"[RECOMMEND_MOOD] vibes={target_vibes}, topics={target_topics}, sentiment={target_sentiment}")

        try:
            q = supabase.table('songs').select(
                'spotify_track_id, title, artists, vibe, main_topic, final_sentiment, spotify_popularity, is_hit, genres'
            )

            # LỚP 2: LẮP RÁP SQL THÔNG MINH
            if target_sentiment:
                q = q.eq('final_sentiment', target_sentiment)
                
            if target_vibes:
                vibe_cond = ",".join([f"vibe.ilike.%{v}%" for v in target_vibes])
                q = q.or_(vibe_cond)
                
            if target_topics:
                topic_cond = ",".join([f"main_topic.ilike.%{t}%" for t in target_topics])
                q = q.or_(topic_cond)

            # [FIX LỖI 1]: ÉP SẮP XẾP THEO ĐỘ HOT ĐỂ TRÁNH LẤY RANDOM THEO BẢNG CHỮ CÁI (A-Z)
            q = q.order('spotify_popularity', desc=True)

            # Lấy dư x4 số lượng để Ranker chọn lọc
            res = q.limit(int(match_count) * 4).execute()
            rows = getattr(res, 'data', None) or []
            source_label = 'exact-mood-ranked'

            # LỚP 3: FALLBACK NẾU QUÁ KHẮT KHE (Vector Search)
            if not rows and (target_vibes or target_topics or target_sentiment):
                execution_path.append("Level 2: Vector Fallback")
                print(f"[Fallback] Điều kiện quá gắt, nới lỏng sang Vector Search cho: {mood_query}")
                query_embedding = _safe_embed(embed_fn, mood_query)
                if query_embedding:
                    res_vec = supabase.rpc("match_vpop_tracks", {
                        "query_embedding": query_embedding,
                        "match_threshold": float(match_threshold or 0.35),
                        "match_count": int(match_count) * 4
                    }).execute()
                    rows = getattr(res_vec, 'data', None) or []
                    source_label = 'vector-fallback:mood-ranked'

            if not rows:
                return {'tracks': [], 'source': 'empty', 'error': f"Chưa tìm thấy nhạc phù hợp với tâm trạng '{mood_query}'.", 'path': execution_path}

            # [FIX LỖI 2]: GÁN ĐIỂM SIMILARITY NẾU TÌM BẰNG SQL ĐỂ UI KHÔNG BỊ 0.0%
            if 'vector-fallback' not in source_label:
                for r in rows:
                    match_score = 0.65 # Điểm gốc
                    # Khớp cái nào cộng điểm cái đó
                    if target_sentiment and r.get('final_sentiment') == target_sentiment:
                        match_score += 0.15
                    if target_vibes and any(v in (r.get('vibe') or '') for v in target_vibes):
                        match_score += 0.10
                    if target_topics and any(t in (r.get('main_topic') or '') for t in target_topics):
                        match_score += 0.10
                    r['similarity'] = min(0.98, match_score)

            # GLOBAL RANKER
            ranked_tracks = rank_and_normalize_tracks(
                raw_rows=rows,
                limit=int(match_count),
                boosts={
                    'vibe': target_vibes,
                    'topics': target_topics,
                    'sentiment': target_sentiment
                }
            )

            return {'tracks': ranked_tracks, 'source': source_label, 'error': None, 'path': execution_path}

        except Exception as e:
            return {'tracks': [], 'source': 'error', 'error': str(e), 'path': execution_path}


    # =========================
    # 5. RECOMMEND_ARTIST (Kiến trúc Fallback Tiêu chuẩn)
    # =========================
    elif action == "RECOMMEND_ARTIST":
        execution_path = ["Level 1: SQL Artist ILIKE"]
        artist = str(params.get("artist", "") or "").strip()
        if not artist:
            return {'tracks': [], 'source': 'fallback-missing-param', 'error': 'Bạn muốn nghe nhạc của nghệ sĩ nào?', 'path': execution_path}

        print(f"[RECOMMEND_ARTIST] Đang tìm nghệ sĩ: '{artist}'")

        try:
            rows = []
            source_label = ""
            artist_query = artist # Biến lưu tên ca sĩ dùng để Ranking

            # --- LỚP 1: TÌM KIẾM CHUẨN (SQL ILIKE) ---
            # Xử lý hoàn hảo các trường hợp gõ đúng, hoặc gõ thiếu họ/tên lót (VD: "Sơn Tùng" -> "Sơn Tùng M-TP")
            q1 = supabase.table('songs').select(
                'title, artists, vibe, main_topic, spotify_track_id, spotify_popularity, is_hit, genres, final_sentiment'
            ).ilike('artists', f'%{artist}%')
            
            # [FIX]: Tăng limit lên gấp 10 lần để lấy dư dả dữ liệu, bù cho việc tí nữa sẽ lọc bớt Sơn Tùng ra
            res1 = q1.order('spotify_popularity', desc=True).limit(int(match_count) * 10).execute()
            raw_rows = getattr(res1, 'data', None) or []
            
            rows = []
            if raw_rows:
                exact_rows = []
                target_artist = artist.lower().strip()
                
                # Quét qua 50 bài hát hot nhất có chứa chữ "Tùng"
                for r in raw_rows:
                    # Chẻ cột artists trong DB ra (Vd: "Tùng, Trang" -> ["tùng", "trang"])
                    db_artists_list = [a.strip().lower() for a in str(r.get('artists', '')).split(',')]
                    
                    # NẾU "tùng" ĐỨNG ĐỘC LẬP TỪNG CHỮ (thì giữ lại), TỪ CHỐI "sơn tùng m-tp"
                    if target_artist in db_artists_list:
                        exact_rows.append(r)
                
                # Nếu lọc ra được nhạc của đúng anh "Tùng", thì lấy danh sách đó
                if exact_rows:
                    rows = exact_rows[:int(match_count) * 4]
                else:
                    # Nếu user gõ "sơn tùng" (không độc lập trong DB), thì xài lại list ILIKE bình thường
                    rows = raw_rows[:int(match_count) * 4]
                    
                source_label = 'text-search:artist-exact'

            # --- LỚP 2: BẢO VỆ GÕ DÍNH CHỮ (Wildcard SQL) ---
            # Kích hoạt khi gõ: "sơntùng"
            if not rows:
                execution_path.append("Level 1.5: Wildcard Whitespace")
                artist_wildcard = artist.replace(" ", "%")
                if artist_wildcard != artist:
                    print(f"[Fallback] Nới lỏng khoảng trắng: '{artist_wildcard}'")
                    q2 = supabase.table('songs').select(
                        'title, artists, vibe, main_topic, spotify_track_id, spotify_popularity, is_hit, genres , final_sentiment'
                    ).ilike('artists', f'%{artist_wildcard}%')
                
                    res2 = q2.order('spotify_popularity', desc=True).limit(int(match_count) * 4).execute()
                    rows = getattr(res2, 'data', None) or []
                    if rows:
                        source_label = 'text-search:artist-wildcard'
                        artist_query = artist_wildcard # Cập nhật query để Ranker biết

            ## --- LỚP 3: AI SỬA LỖI CHÍNH TẢ (Fuzzy Matching) ---
            # CHỈ kích hoạt khi gõ sai chính tả nhẹ. Tuyệt đối không chạy trước để tránh phá data.
            if not rows:
                execution_path.append("Level 2: Fuzzy Spelling")
                try:
                    current_artist_list = _load_artist_list_from_supabase(supabase)
                except Exception:
                    current_artist_list = []

                if current_artist_list:
                    # Lấy tên không dấu để kiểm tra
                    query_norm = _normalize_text(artist)
                    match = _extract_one(query_norm, [_normalize_text(a) for a in current_artist_list])
                    
                    # SIẾT CHẶT: Ngưỡng an toàn >= 88% VÀ không được chênh lệch quá 4 ký tự
                    # Để chặn vụ "son tug" (7 chữ) biến thành "Cao Thái Sơn" (12 chữ)
                    if match and match[1] >= 88:
                        best_artist_norm = match[0]
                        best_artist = current_artist_list[match[2]]
                        
                        # Điều kiện chặn độ dài
                        if abs(len(query_norm) - len(best_artist_norm)) <= 7:
                            if best_artist.lower() != artist.lower():
                                print(f"[Fallback] AI dò lỗi chính tả: Đã nắn '{artist}' thành '{best_artist}'")
                                
                                # Tìm lại với tên đã sửa
                                q3 = supabase.table('songs').select(
                                    'title, artists, vibe, main_topic, spotify_track_id, spotify_popularity, is_hit, genres, final_sentiment'
                                ).ilike('artists', f'%{best_artist}%')
                                
                                res3 = q3.order('spotify_popularity', desc=True).limit(int(match_count) * 4).execute()
                                rows = getattr(res3, 'data', None) or []
                                if rows:
                                    source_label = 'text-search:artist-fuzzy'
                                    artist_query = best_artist

            # KẾT THÚC CHUỖI TÌM KIẾM: Nếu vẫn trống
            if not rows:
                return {'tracks': [], 'source': 'search-artist-empty', 'error': f"Tiếc quá, hiện tại mình chưa có bài nào của '{artist}'.", 'path': execution_path}

            # BƠM ĐIỂM SIMILARITY (Fix lỗi 0.0%)
            if 'vector-fallback' not in source_label:
                for r in rows:
                    db_artists = _normalize_text(r.get('artists') or '')
                    clean_query = _normalize_text(artist_query).replace("%", " ")
                    
                    if clean_query in db_artists:
                        r['similarity'] = 0.95
                    else:
                        r['similarity'] = 0.70

            # =======================================================
            # GLOBAL RANKER
            # =======================================================
            ranked_tracks = rank_and_normalize_tracks(
                raw_rows=rows,
                limit=int(match_count),
                boosts={'artist': artist_query.replace("%", " ")}
            )
            return {'tracks': ranked_tracks, 'source': source_label, 'error': None, 'path': execution_path}
        except Exception as ex:
            return {'tracks': [], 'source': 'search-artist-error', 'error': f"Lỗi hệ thống: {ex}", 'path': execution_path}
        
    # =========================
    # 6. RECOMMEND_GENRE (Lọc Thể loại chuẩn xác + Bắt nhiều nhãn)
    # =========================
    elif action == "RECOMMEND_GENRE":
        execution_path = ["Level 1: SQL Genre Exact/ILIKE"]
        genre_query = str(params.get("genre") or "").strip()
        if not genre_query:
            return {'tracks': [], 'source': 'fallback-missing-param', 'error': 'Bạn muốn nghe thể loại nhạc gì?', 'path': execution_path}

        # Gọi hàm mới để trả về mảng các target (vd: ['Pop', 'Rap/Hip-hop'])
        mapped_genres = get_genre_targets(genre_query)
        # Update lại params để ghi log cho chuẩn
        params["mapped_genres"] = mapped_genres
        print(f"[RECOMMEND_GENRE] query='{genre_query}' -> mapped={mapped_genres}")

        try:
            # [FIX 2]: Dùng hàm để khởi tạo truy vấn mới 100% mỗi lần gọi, chống dính bộ lọc cũ
            def get_base_q():
                return supabase.table('songs').select(
                    'spotify_track_id, title, artists, vibe, main_topic, final_sentiment, spotify_popularity, is_hit, genres'
                ).order('spotify_popularity', desc=True)

            rows = []
            source_label = 'text-search:genre-ranked'

            # NẾU USER TÌM 1 THỂ LOẠI (Vd: "Indie")
            if len(mapped_genres) == 1:
                target = mapped_genres[0]
                
                # BƯỚC 1: Tìm Nhạc THUẦN (Khởi tạo get_base_q() mới)
                res_exact = get_base_q().eq('genres', target).limit(int(match_count) * 2).execute()
                rows = getattr(res_exact, 'data', None) or []
                for r in rows: r['similarity'] = 1.0 
                
                # BƯỚC 2: Tìm Nhạc LAI (Khởi tạo get_base_q() mới)
                if len(rows) < int(match_count) * 2:
                    res_like = get_base_q().ilike('genres', f'%{target}%').limit(int(match_count) * 3).execute()
                    likes = getattr(res_like, 'data', None) or []
                    existing_ids = {r['spotify_track_id'] for r in rows}
                    for r in likes:
                        if r['spotify_track_id'] not in existing_ids:
                            r['similarity'] = 0.85 
                            rows.append(r)

            # NẾU USER TÌM NHIỀU THỂ LOẠI (Vd: "Pop và Ballad")
            elif len(mapped_genres) > 1:
                query_and = get_base_q() # <-- Khởi tạo mới
                for target in mapped_genres:
                    query_and = query_and.ilike('genres', f'%{target}%')
                res_multi = query_and.limit(max(20, int(match_count) * 4)).execute()
                rows = getattr(res_multi, 'data', None) or []
                for r in rows: r['similarity'] = 0.95
                
                if not rows:
                    execution_path.append("Level 1.5: SQL Genre OR Fallback")
                    query_or = get_base_q() # <-- Khởi tạo mới
                    or_conds = ",".join([f"genres.ilike.%{t}%" for t in mapped_genres])
                    query_or = query_or.or_(or_conds)
                    res_or = query_or.limit(max(20, int(match_count) * 4)).execute()
                    rows = getattr(res_or, 'data', None) or []
                    for r in rows: r['similarity'] = 0.85 

            # =========================
            # VECTOR FALLBACK (Dành cho thể loại mập mờ)
            # =========================
            if not rows:
                execution_path.append("Level 2: Semantic Vector")
                query_embedding = _safe_embed(embed_fn, genre_query)
                if query_embedding:
                    res_vec = supabase.rpc("match_vpop_tracks", {
                        "query_embedding": query_embedding,
                        "match_threshold": float(match_threshold or 0.35),
                        "match_count": int(match_count) * 4
                    }).execute()
                    rows = getattr(res_vec, 'data', None) or []
                    for r in rows: r['similarity'] = max(0.7, float(r.get('similarity', 0.7)))
                    source_label = 'vector-fallback:genre-ranked'

            if not rows:
                return {'tracks': [], 'source': 'empty', 'error': f"Chưa tìm thấy nhạc thuộc thể loại '{genre_query}'.", 'path': execution_path}

            # =======================================================
            # GLOBAL RANKER
            # =======================================================
            ranked_tracks = rank_and_normalize_tracks(
                raw_rows=rows,
                limit=int(match_count),
                boosts={
                    'genre': mapped_genres # Đưa mảng genre vào để Ranker thưởng điểm phụ
                }
            )

            return {'tracks': ranked_tracks, 'source': source_label, 'error': None, 'path': execution_path}

        except Exception as e:
            return {'tracks': [], 'source': 'error', 'error': str(e), 'path': execution_path}


    # =========================
    # 7. ANALYZE_READY (Phân tích chuyên sâu: Librosa + NLP + SHAP)
    # =========================
    elif action == "ANALYZE_READY":
        execution_path = ["Level 1: Analyze Ready"]
        audio_path = params.get("audio_path")
        if not audio_path or not os.path.exists(audio_path):
            return {'error': "Không tìm thấy file âm thanh để phân tích.", 'source': 'analyze-error', 'path': execution_path}

        lyric_text = params.get('lyric_text')
        lyric_path = params.get('lyric_path')
        if not lyric_text and not lyric_path:
            return {
                'error': "Bạn cần cung cấp lời nhạc (.txt) để phân tích (hiện không dùng Speech-to-Text).",
                'source': 'analyze-error',
                'path': execution_path,
            }

        try:
            from chatbot.analyze_ready_action import run_analyze_ready

            bundle = run_analyze_ready(
                audio_path=str(audio_path),
                lyric_text=str(lyric_text) if lyric_text else None,
                lyric_path=str(lyric_path) if lyric_path else None,
                supabase_client=supabase,
                allow_download=True,
                compute_shap=True,
                force_storage=True,
                skip_p1=True,
            )

            return {
                'action': 'DISPLAY_ANALYSIS',
                'bundle': bundle,
                'source': 'analyze:ready',
                'error': None,
                'path': execution_path,
            }
        except Exception as e:
            return {'error': f"Lỗi phân tích chuyên sâu: {str(e)}", 'source': 'analyze-error', 'path': execution_path}

    # =========================
    # 8. MISSING_FILE
    # =========================
    elif action == "MISSING_FILE":
        return {
            'tracks': [], 
            'source': 'fallback-missing-file', 
            'error': 'Bạn quên đính kèm file âm thanh (MP3/WAV) ở thanh bên trái (Sidebar) rồi kìa! Hãy tải file lên để mình phân tích nhé.',
            'path': ["Level 0: Static Response"],
        }
    # =========================
    # 9. CLARIFY
    # =========================
    elif action == "CLARIFY":
        return {
            'tracks': [], 
            'source': 'action-clarify', 
            'error': 'Xin lỗi, mình chưa hiểu rõ ý bạn lắm. Bạn có thể nói rõ hơn là bạn muốn tìm bài hát, nghe nhạc theo tâm trạng, hay muốn mình phân tích file âm thanh không?',
            'path': ["Level 0: Static Response"],
        }
    

    # =========================
    # 10. MUSIC_KNOWLEDGE
    # =========================
    elif action == "MUSIC_KNOWLEDGE":
        # Kiến thức âm nhạc (tiểu sử ca sĩ, nhạc lý) không nằm trong bảng Songs.
        # Ta trả về cờ "TRIGGER_LLM" để giao diện biết mà tự động gọi AI Gemini trả lời.
        return {
            'tracks': [], 
            'source': 'action-music-knowledge', 
            'error': 'TRIGGER_LLM_ANSWER',
            'path': ["Level 0: Trigger LLM"],
        }


    # =========================
    # 11. OUT_OF_SCOPE
    # =========================
    elif action == "OUT_OF_SCOPE":
        return {
            'tracks': [], 
            'source': 'action-out-of-scope', 
            'error': 'Xin lỗi, mình là Trợ lý AI chuyên về âm nhạc V-Pop. Mình chỉ có thể giúp bạn tìm nhạc, phân tích bài hát hoặc trả lời các kiến thức về âm nhạc thôi nhé!',
            'path': ["Level 0: Static Response"],
        }

    # =========================
    # 12. ADVANCED_SEARCH (Tìm kiếm Kết hợp - Đã tối ưu bằng Helper & Ranker)
    # =========================
    elif action == "ADVANCED_SEARCH":
        execution_path = ["Level 1: SQL Filter Join"]
        mood = str(params.get("mood", "")).strip()
        genre = str(params.get("genre", "")).strip()
        artist = str(params.get("artist", "")).strip()

        if not any([mood, genre, artist]):
            return {'tracks': [], 'source': 'error', 'error': 'Thiếu thông số tìm kiếm nâng cao.', 'path': execution_path}

        print(f"[ADVANCED_SEARCH] Đang lọc chéo: Mood='{mood}', Genre='{genre}', Artist='{artist}'")

        try:
            q = supabase.table('songs').select(
                'spotify_track_id, title, artists, vibe, main_topic, final_sentiment, spotify_popularity, is_hit, genres'
            )
            
            boosts = {} # Cuốn sổ ghi chép điểm thưởng cho Ranker

            # 1. Ráp mảnh ghép Thể loại (Bắt buộc - AND)
            if genre:
                mapped_genre = get_genre_target(genre)
                q = q.ilike('genres', f'%{mapped_genre}%')
                boosts['genre'] = mapped_genre

            # 2. Ráp mảnh ghép Nghệ sĩ (Bắt buộc - AND)
            if artist:
                q = q.ilike('artists', f'%{artist}%')
                boosts['artist'] = artist

            # 3. Ráp mảnh ghép Tâm trạng (Linh hoạt - Gom 3 tín hiệu)
            if mood:
                target_vibes, target_topics, target_sentiment = _mood_maps(mood)
                boosts.update({'vibe': target_vibes, 'topics': target_topics, 'sentiment': target_sentiment})
                
                mood_filters = []
                if target_sentiment:
                    mood_filters.append(f"final_sentiment.eq.{target_sentiment}")
                if target_vibes:
                    mood_filters.extend([f"vibe.ilike.%{v}%" for v in target_vibes])
                if target_topics:
                    mood_filters.extend([f"main_topic.ilike.%{t}%" for t in target_topics])
                
                # Ép DB lọc bằng OR cho các tín hiệu tâm trạng
                if mood_filters:
                    q = q.or_(",".join(mood_filters))

            # [FIX QUAN TRỌNG]: Ép sắp xếp theo Popularity để tránh kéo Random
            q = q.order('spotify_popularity', desc=True)

            # 4. CHẠY TRUY VẤN SQL (Lấy dư ra x4 để Ranker làm việc)
            res = q.limit(int(match_count) * 4).execute()
            rows = getattr(res, 'data', None) or []
            source_label = 'advanced-search-sql'

            # 5. VECTOR FALLBACK (Cứu cánh nếu lọc chéo quá gắt không ra bài nào)
            if not rows:
                execution_path.append("Level 2: Semantic Vector")
                combo_text = f"{mood} {genre} {artist}".strip()
                print(f"[Fallback] Lọc chéo không ra, đưa vào Vector Search: '{combo_text}'")
                query_embedding = _safe_embed(embed_fn, combo_text)
                if query_embedding:
                    thr = float(match_threshold) if match_threshold is not None else 0.35 # Nới lỏng điểm
                    res_vec = supabase.rpc(
                        "match_vpop_tracks", 
                        {"query_embedding": query_embedding, "match_threshold": thr, "match_count": int(match_count) * 4}
                    ).execute()
                    rows = getattr(res_vec, 'data', None) or []
                    source_label = 'vector-fallback:advanced'

            if not rows:
                return {'tracks': [], 'source': 'search-advanced-empty', 'error': "Khẩu vị của bạn mặn quá, hệ thống lọc mãi không ra bài nào khớp hết các điều kiện này!", 'path': execution_path}

            # [FIX LỖI 0.0%]: Bơm điểm Similarity giả lập cho các bài chui qua màng lọc SQL
            if 'vector-fallback' not in source_label:
                for r in rows:
                    match_score = 0.65
                    if artist and _normalize_text(artist) in _normalize_text(r.get('artists') or ''):
                        match_score += 0.15
                    if genre and _normalize_text(mapped_genre) in _normalize_text(r.get('genres') or ''):
                        match_score += 0.10
                    # Tăng điểm nếu khớp mood (sentiment/vibe/topic) để kết quả ổn định hơn.
                    if mood:
                        try:
                            db_vibe = _normalize_text(r.get('vibe') or '')
                            db_topic = _normalize_text(r.get('main_topic') or '')
                            vibe_targets = [_normalize_text(v) for v in (target_vibes or [])]
                            topic_targets = [_normalize_text(t) for t in (target_topics or [])]

                            if vibe_targets and any(v and v in db_vibe for v in vibe_targets):
                                match_score += 0.12
                            if topic_targets and any(t and t in db_topic for t in topic_targets):
                                match_score += 0.12
                            if target_sentiment and r.get('final_sentiment') == target_sentiment:
                                match_score += 0.08

                            # Prefer non-sad tracks for certain vibes when user didn't ask for "buồn".
                            # This helps avoid popularity-only dominance among many same-vibe candidates.
                            if not target_sentiment and vibe_targets and any(v == 'kich tinh' for v in vibe_targets):
                                db_sent = str(r.get('final_sentiment') or '').lower().strip()
                                if db_sent == 'positive':
                                    match_score += 0.08
                                elif db_sent == 'negative':
                                    match_score -= 0.05
                        except Exception:
                            # Best-effort only; don't break ADVANCED_SEARCH.
                            pass
                    r['similarity'] = min(0.98, match_score)

            # =======================================================
            # GLOBAL RANKER
            # =======================================================
            ranked = rank_and_normalize_tracks(
                raw_rows=rows,
                limit=int(match_count),
                boosts=boosts
            )

            return {'tracks': ranked, 'source': source_label, 'error': None, 'path': execution_path}

        except Exception as e:
            return {'tracks': [], 'source': 'search-advanced-error', 'error': f"Lỗi truy vấn đa luồng: {e}", 'path': execution_path}
        

    # =========================
    # 13. RECOMMEND_SEED (Toán Học Tuyến Tính Tuyệt Đối & Flat Queries)
    # =========================
    elif action == "RECOMMEND_SEED":
        execution_path = ["Level 1: Seed Lookup"]
        seed_name = str(params.get("seed_name", "")).strip()
        seed_artist_query = str(params.get("artist", "") or "").strip()

        if not seed_name:
            return {
                'tracks': [],
                'source': 'fallback-missing-param',
                'error': 'Bạn muốn nghe nhạc giống bài nào? (Thiếu seed_name)',
                'path': execution_path,
            }

        try:
            # --- BƯỚC 1: TIỀN XỬ LÝ LỌC RÁC TỪ CHUỖI USER ---
            import re
            # Gọt bỏ các từ thừa mà LLM có thể lỡ trích xuất nhầm vào seed_name
            clean_seed_name = re.sub(r'^(bài hát|bài|nhạc của|nhạc|ca khúc)\s+', '', seed_name, flags=re.IGNORECASE).strip()
            if not clean_seed_name:
                clean_seed_name = seed_name
                
            seed_track = None
            
            # --- BƯỚC 2: LỚP 1 - TÌM SQL (Ưu tiên bài Hot nhất nếu trùng tên) ---
            song_q = supabase.table('songs').select(
                'spotify_track_id, title, artists, vibe, genres, final_sentiment, spotify_popularity'
            ).ilike('title', f'%{clean_seed_name}%')
            
            if seed_artist_query:
                # Nới lỏng khoảng trắng để bắt ca sĩ gõ thiếu/dính chữ
                aw = seed_artist_query.replace(' ', '%')
                song_q = song_q.ilike('artists', f'%{aw}%')
                execution_path.append('Level 1.1: Seed Artist Filter')

            # Lấy 5 bài khớp, SAU ĐÓ xếp hạng bằng Độ Hot để né bản Remix/Cover (Fix vụ The Masked Singer)
            song_f = song_q.order('spotify_popularity', desc=True).limit(5).execute()
            
            if song_f.data:
                seed_track = song_f.data[0]
            else:
                # --- BƯỚC 3: LỚP 2 - LOCAL SMART FUZZY (Cứu cánh không dấu, dính chữ) ---
                execution_path.append("Level 2: Fuzzy Seed Lookup")
                all_songs = _get_all_songs_cached(supabase)
                if all_songs:
                    import difflib
                    query_t = _normalize_text(clean_seed_name)
                    query_a = _normalize_text(seed_artist_query) if seed_artist_query else ""
                    query_full = f"{query_t} {query_a}".strip()
                    
                    if len(query_full) >= 3:
                        scored = []
                        for s in all_songs:
                            db_t = _normalize_text(s.get('title') or s.get('song_title') or '')
                            db_a = _normalize_text(s.get('artists') or '')
                            db_full = f"{db_t} {db_a}".strip()
                            
                            if not db_full: continue
                            
                            score = 0.0
                            if query_full == db_full:
                                score = 100.0
                            elif query_full in db_full and len(query_full) >= 5:
                                penalty = (len(db_full) - len(query_full)) * 0.2
                                score = max(75.0, 95.0 - penalty)
                            elif db_full in query_full and len(db_full) >= 5:
                                score = 90.0 * (len(db_full) / len(query_full))
                            else:
                                # Tính ratio chéo cứu gõ dính chữ "yeumotnguoicole"
                                seq_ratio = difflib.SequenceMatcher(None, query_full, db_full).ratio() * 100.0
                                qw = set(query_full.split())
                                dw = set(db_full.split())
                                jaccard = 0.0
                                if qw and dw:
                                    jaccard = (len(qw.intersection(dw)) / len(qw.union(dw))) * 100.0
                                score = max(seq_ratio, jaccard)
                            
                            if score >= 75.0:
                                s_copy = dict(s)
                                s_copy['seed_sim_score'] = score
                                scored.append(s_copy)
                                
                        if scored:
                            # Ranking kép: Ưu tiên 1 - Độ khớp chuỗi. Ưu tiên 2 - Độ phổ biến (né rác)
                            scored.sort(key=lambda x: (x['seed_sim_score'], float(x.get('spotify_popularity') or 0)), reverse=True)
                            seed_track = scored[0]

            # Kiểm tra chốt chặn cuối cùng
            if not seed_track:
                return {'tracks': [], 'source': 'rec-seed-empty', 'error': f"Không thấy bài '{seed_name}' trong hệ thống.", 'path': execution_path}

            t_id = seed_track['spotify_track_id']
            seed_title = seed_track.get('title')
            seed_artist = seed_track.get('artists')
            seed_vibe_raw = str(seed_track.get('vibe') or '').strip()
            seed_genre_raw = str(seed_track.get('genres') or '').strip()
            seed_sent_raw = str(seed_track.get('final_sentiment') or '').strip()

            # Bản normalized dùng cho scoring
            seed_vibe = seed_vibe_raw.lower()
            seed_genre = seed_genre_raw.lower()
            seed_sent = seed_sent_raw.lower()

            # 4. LẤY THÔNG SỐ VẬT LÝ BÀI MẪU (1 Query đơn)
            # ... (Giữ nguyên toàn bộ logic tính toán tuyến tính vật lý ở dưới của bạn) ...
            feat_res = supabase.table('track_features').select('tempo_bpm, rms_energy, beat_strength_mean, lexical_diversity').eq('spotify_track_id', t_id).limit(1).execute()
            if not feat_res.data:
                return {'tracks': [], 'source': 'rec-seed-no-vec', 'error': "Bài mẫu chưa có dữ liệu âm thanh.", 'path': execution_path}

            seed_tempo = float(feat_res.data[0].get('tempo_bpm', 120.0))
            seed_energy = float(feat_res.data[0].get('rms_energy', 0.5))
            seed_beat = float(feat_res.data[0].get('beat_strength_mean', 0.5))
            seed_lex = float(feat_res.data[0].get('lexical_diversity', 0.5))

            # 3. LỌC NHANH TRÊN DATABASE BẰNG BIÊN ĐỘ VẬT LÝ (Chặn đứng Full Table Scan)
            # Chỉ lấy các bài chênh lệch Tempo tối đa ±15 và Energy ±0.15
            range_res = supabase.table('track_features').select(
                'spotify_track_id, tempo_bpm, rms_energy, beat_strength_mean, lexical_diversity'
            ).gte('tempo_bpm', seed_tempo - 15).lte('tempo_bpm', seed_tempo + 15) \
             .gte('rms_energy', seed_energy - 0.15).lte('rms_energy', seed_energy + 0.15) \
             .limit(200).execute()
            
            candidate_feats = range_res.data or []
            ids = [f['spotify_track_id'] for f in candidate_feats if f['spotify_track_id'] != t_id]

            if not ids:
                return {'tracks': [], 'source': 'rec-seed-none', 'error': "Không tìm thấy bài hát nào có cùng nhịp điệu cốt lõi.", 'path': execution_path}

            # 4. KÉO METADATA BẰNG ID (Siêu tốc độ)
            songs_data = supabase.table('songs').select(
                'spotify_track_id, title, artists, vibe, main_topic, spotify_popularity, is_hit, genres, final_sentiment'
            ).in_('spotify_track_id', ids).execute().data or []
            
            song_map = {s['spotify_track_id']: s for s in songs_data}

            # 5. TÁCH THỂ LOẠI THÔNG MINH
            import re
            def parse_genres(g_str):
                return set([g.strip() for g in re.split(r'[,/]', g_str) if g.strip()])
            
            seed_g_set = parse_genres(seed_genre)
            conflict_genres = {'bolero', 'vinahouse', 'rap', 'hip-hop', 'r&b', 'indie', 'ballad', 'rock'}

            # 6. CHẤM ĐIỂM TUYẾN TÍNH (LINEAR PENALTY) - KHÔNG LẠM PHÁT
            rows = []
            for f in candidate_feats:
                c_id = f['spotify_track_id']
                if c_id not in song_map: continue
                s = song_map[c_id]
                
                t_bpm = float(f.get('tempo_bpm', 0))
                t_nrg = float(f.get('rms_energy', 0))
                t_beat = float(f.get('beat_strength_mean', 0.5))
                t_lex = float(f.get('lexical_diversity', 0.5))
                
                score = 100.0
                
                # Trừ điểm vật lý tàn nhẫn:
                # Lệch 1 BPM = -1.5 điểm. Lệch 10 BPM mất 15 điểm.
                score -= abs(t_bpm - seed_tempo) * 1.5 
                # Lệch 0.1 Energy = -10 điểm
                score -= abs(t_nrg - seed_energy) * 100.0
                # Lệch 0.1 Beat = -5 điểm
                score -= abs(t_beat - seed_beat) * 50.0
                # Lệch 0.1 Lexical = -5 điểm
                score -= abs(t_lex - seed_lex) * 50.0
                
                # Sàng lọc Thể loại
                track_g_set = parse_genres(str(s.get('genres') or '').lower())
                
                # Bắt buộc phải có ít nhất 1 thể loại chung
                if seed_g_set and not seed_g_set.intersection(track_g_set):
                    score -= 40.0 # Không có điểm chung -> Giết
                else:
                    # Kiểm tra thể loại lai tạp
                    forbidden = track_g_set - seed_g_set
                    for fg in forbidden:
                        if fg in conflict_genres:
                            score -= 25.0 # Mang gen ngoại lai (như Rap lai Bolero) -> Trừ nặng
                
                # Thưởng nhẹ cảm xúc (+2đ)
                if seed_vibe and seed_vibe in str(s.get('vibe') or '').lower():
                    score += 2.0
                if seed_sent and seed_sent == str(s.get('final_sentiment') or '').lower():
                    score += 2.0
                    
                score = max(0.0, score)
                
                # Điểm phải > 65% mới cho hiển thị
                if score >= 65.0:
                    rows.append({
                        'spotify_id': c_id,
                        'title': s.get('title'),
                        'artist': s.get('artists'),
                        'vibe': s.get('vibe'),
                        'main_topic': s.get('main_topic'),
                        'tempo_bpm': t_bpm,
                        'rms_energy': t_nrg,
                        'beat_strength_mean': t_beat,
                        'lexical_diversity': t_lex,
                        'delta_tempo_bpm': abs(t_bpm - seed_tempo),
                        'delta_rms_energy': abs(t_nrg - seed_energy),
                        'delta_beat_strength_mean': abs(t_beat - seed_beat),
                        'delta_lexical_diversity': abs(t_lex - seed_lex),
                        'spotify_popularity': s.get('spotify_popularity') or 0,
                        'is_hit': s.get('is_hit'),
                        'genres': s.get('genres'),
                        'final_sentiment': s.get('final_sentiment'),
                        'similarity': round(score, 2),
                        'score': score
                    })

            # 7. SẮP XẾP VÀ KHỬ TRÙNG (100% Thuần Điểm Score)
            rows.sort(key=lambda x: x['score'], reverse=True)

            final_list = []
            seen_artists = set()
            for r in rows:
                main_artist = str(r.get('artist', '')).split(',')[0].strip().lower()
                if main_artist not in seen_artists:
                    final_list.append(r)
                    seen_artists.add(main_artist)
                if len(final_list) >= int(match_count):
                    break

            if len(final_list) < int(match_count):
                for r in rows:
                    if r not in final_list:
                        final_list.append(r)
                    if len(final_list) >= int(match_count):
                        break

            return {
                'tracks': _normalize_track_rows(final_list),
                'source': 'recommendation:seed-linear-penalty',
                'error': None,
                'request_params': {
                    'seed_name': seed_name,
                    'artist': seed_artist_query,
                },
                'seed_meta': {
                    'seed_name': seed_name,
                    'seed_spotify_track_id': t_id,
                    'seed_title': seed_title,
                    'seed_artist': seed_artist,
                    'seed_vibe': seed_vibe_raw,
                    'seed_genres': seed_genre_raw,
                    'seed_final_sentiment': seed_sent_raw,
                    'seed_tempo_bpm': seed_tempo,
                    'seed_rms_energy': seed_energy,
                    'seed_beat_strength_mean': seed_beat,
                    'seed_lexical_diversity': seed_lex,
                },
                'path': execution_path,
            }
        except Exception as e:
            return {'tracks': [], 'source': 'error', 'error': str(e), 'path': execution_path}
        

    # =========================
    # 14. RECOMMEND_ATTRIBUTES (Chỉ lọc theo Nhạc lý: Tempo & Energy)
    # =========================
    elif action == "RECOMMEND_ATTRIBUTES":
        execution_path = ["Level 1: Attribute Rule Map"]
        min_t, max_t = 0, 250
        min_e, max_e = 0.0, 1.0
        
        # Lấy text từ attributes + song_title để fallback
        raw_text = _normalize_text(str(params.get("attributes", "") + params.get("song_title", "")))
        
        # --- Ánh xạ Tempo (Nhịp điệu) ---
        import re
        
        # PRIORITY 1: Explicit BPM number passed separately from router
        bpm_val = params.get("bpm")
        if bpm_val is not None:
            try:
                bpm_val = int(bpm_val)
                if 40 <= bpm_val <= 250:
                    min_t, max_t = max(0, bpm_val - 10), min(250, bpm_val + 10)
                    params["target_tempo"] = float(bpm_val)
            except (ValueError, TypeError):
                bpm_val = None
        
        # PRIORITY 2: Implicit tempo keywords
        if bpm_val is None:
            if any(k in raw_text for k in ['cham', 'slow']):
                min_t, max_t = 60, 90
            elif any(k in raw_text for k in ['on dinh', 'on dink', 'nhip nhang', 'nhip ngang', 'vua', 'binh thuong', 'thuong']):
                # "Nhịp nhàng" (steady rhythm) = mid-tempo (90-120 BPM)
                min_t, max_t = 90, 120
            elif any(k in raw_text for k in ['nhanh', 'fast', 'nhip nhanh']):
                min_t, max_t = 120, 160
            elif any(k in raw_text for k in ['rat nhanh', 'don dap', 'speed up', 'cuc nhanh']):
                min_t, max_t = 160, 220

        # --- Ánh xạ Energy (Năng lượng) ---
        if any(k in raw_text for k in ['thap', 'yeu', 'nhe', 'em diu', 'mong manh']):
            min_e, max_e = 0.0, 0.15
        elif any(k in raw_text for k in ['cao', 'manh', 'uy luc', 'cang', 'day']):
            min_e, max_e = 0.3, 1.0

        try:
            # Lấy đủ Data cho Ranker
            res = supabase.table('track_features').select(
                "spotify_track_id, tempo_bpm, rms_energy, songs(title, artists, vibe, main_topic, spotify_popularity, is_hit, genres, final_sentiment)"
            ).gte('tempo_bpm', min_t).lte('tempo_bpm', max_t) \
             .gte('rms_energy', min_e).lte('rms_energy', max_e) \
             .limit(int(match_count) * 4).execute()
            
            rows = []
            for r in (res.data or []):
                s = r.get('songs', {}) or {}
                # Ép phẳng JSON từ bảng liên kết (Join)
                row_flat = {
                    'spotify_id': r['spotify_track_id'], 
                    'title': s.get('title'), 
                    'artists': s.get('artists'),
                    'vibe': s.get('vibe'),           
                    'main_topic': s.get('main_topic'), 
                    'tempo_bpm': float(r.get('tempo_bpm', 0)), 
                    'rms_energy': float(r.get('rms_energy', 0)),
                    'spotify_popularity': s.get('spotify_popularity'),
                    'is_hit': s.get('is_hit'),
                    'genres': s.get('genres'),
                    'final_sentiment': s.get('final_sentiment')
                }
                rows.append(row_flat)

            # =======================================================
            # GLOBAL RANKER
            # =======================================================
            # Trích xuất target từ params (Giả sử bạn có target_tempo, target_energy từ LLM)
            target_t = float(params.get("target_tempo")) if params.get("target_tempo") else None
            target_e = float(params.get("target_energy")) if params.get("target_energy") else None

            ranked = rank_and_normalize_tracks(
                raw_rows=rows, limit=int(match_count), 
                boosts={'target_tempo': target_t, 'target_energy': target_e}
            )

            return {'tracks': ranked, 'source': 'recommendation:attributes-ranked', 'error': None, 'path': execution_path}
        except Exception as e:
            return {'tracks': [], 'source': 'error', 'error': str(e), 'path': execution_path}


    # =========================
    # 16. RECOMMEND_POPULARITY (Gợi ý Playlist Top Hit - BXH)
    # =========================
    elif action == "RECOMMEND_POPULARITY":
        execution_path = ["Level 1: Popularity Top-N"]
        # Lấy tên nghệ sĩ từ params (nếu có)
        artist_filter = str(params.get("artist", "") or "").strip()
        
        msg_log = f"[RECOMMEND_POPULARITY] Đang tổng hợp Top 5 bài hát Hot nhất"
        if artist_filter:
            msg_log += f" của nghệ sĩ: '{artist_filter}'"
        print(msg_log)

        try:
            # 1. Khởi tạo truy vấn gốc
            q = supabase.table('songs').select(
                'spotify_track_id, title, artists, spotify_popularity'
            )
            
            # 2. Nếu có tên ca sĩ -> Ép lệnh lọc ILIKE
            if artist_filter:
                # Dùng cơ chế fuzzy/wildcard để nới lỏng tìm kiếm tên ca sĩ
                artist_wildcard = artist_filter.replace(" ", "%")
                q = q.ilike('artists', f'%{artist_wildcard}%')

            # 3. Lấy Top 5 bài Hot nhất (Sắp xếp DESC)
            res = q.order('spotify_popularity', desc=True).limit(5).execute()
            
            rows = []
            for r in (res.data or []):
                rows.append({
                    'spotify_id': r['spotify_track_id'], 
                    'title': r.get('title'), 
                    'artist': r.get('artists'),
                    'popularity': r.get('spotify_popularity')
                })

            # 4. Trả về cho Chatbot
            return {
                'tracks': _normalize_track_rows(rows), 
                'source': 'recommendation:popularity-playlist', 
                'error': None,
                'path': execution_path,
            }
        except Exception as e:
            return {'tracks': [], 'source': 'error', 'error': str(e), 'path': execution_path}