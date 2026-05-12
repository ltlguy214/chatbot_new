from __future__ import annotations
from typing import Any, Callable
import os
import re
import unicodedata
import difflib
import math
from collections import defaultdict
from datetime import datetime
from rapidfuzz import fuzz 

try:
    from rapidfuzz import process as _rf_process
except Exception:
    _rf_process = None

_GLOBAL_SONGS_CACHE = []

DERIVATIVE_ARTISTS = [
    'forest studio', 'lofi', 'remix', 'cover', 'live', 'acoustic', 
    'orinn', 'freak d', 'mee media', 'nguyenn', 'instrumental'
]
JUNK_FILTER = r'\b(tìm|tim|mở|mo|bật|bat|nghe|phát|phat|gợi ý|goi y|cho|tôi|toi|mình|minh|xin|một|mot|vài|vai|những|nhung|bạn|ban|có|co|thể|the|không|khong|ko|cần|can|muốn|muon|giúp|giup|hộ|ho|này|nay|kia|đó|do|của|cua|ca sĩ|ca si|nhạc sĩ|nhac si|nghệ sĩ|nghe si|bài hát|bai hat|bài|bai|ca khúc|ca khuc|nhạc|nhac|playlist|list|đi|nhé|nha|với|voi|luôn|luon|chứ|chu|nữa|nua|thử|thu|nào|nao|nhỉ|nhi|hả|ha|vậy|vay|giùm|gium|được|duoc|chưa|chua|thì|thi|vào|vao|đây|day|trong|ngoài|ngoai|do|làm|lam|để|de|ngay|luôn|thịnh\s+hành|thinh\s+hanh|hot|trending|viral|đang|dang)\b'

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
            if boosts.get('is_deep_semantic'):
                base_score = sim
            else:
                # Luồng từ khóa thông thường (Level 4 gốc) giữ nguyên tỉ lệ
                base_score = (0.6 * sim) + (0.25 * pop) + (0.15 * hit)
        else:
            base_score = (0.7 * pop) + (0.3 * hit)

        if not raw_vibe and not raw_genre: penalty_score -= 0.05

        # --- [FIX LỖI PHẠT QUÁ TAY TẠI ĐÂY] ---
        if not target_artist: 
            lower_artists = raw_artists.lower()
            if any(kw in lower_artists for kw in DERIVATIVE_ARTISTS):
                # ĐÃ SỬA THÀNH -0.3 ĐỂ KHÔNG BỊ RỚT XUỐNG 0.0%
                penalty_score -= 0.3 
            if 'remix' in db_title or 'cover' in db_title or 'live' in db_title:
                # ĐÃ SỬA THÀNH -0.2
                penalty_score -= 0.2
        # -----------------------------------------------------------------

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
            if diff == 0: 
                boost_score += 0.8  # Ưu tiên tuyệt đối bài khớp chuẩn 130
            elif diff <= 2: 
                boost_score += 0.4  # Thưởng cao cho sai số nhỏ
            elif diff <= 5: 
                boost_score += 0.1

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

    # Sắp xếp lần 1: ưu tiên điểm tổng hợp (raw_score).
    # `similarity` ở nhiều luồng SQL/Rule có thể không tồn tại hoặc = 0.0,
    # nên nếu sort theo similarity sẽ vô tình biến ranker thành "popularity-only".
    ranked.sort(
        key=lambda x: (
            float(x.get('raw_score') or 0.0),
            float(x.get('spotify_popularity') or 0.0),
        ),
        reverse=True,
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

    # Sắp xếp lần 2: dùng điểm sau non-linear scale.
    final_list.sort(
        key=lambda x: (
            float(x.get('final_mix_score') or x.get('raw_score') or 0.0), 
            float(x.get('spotify_popularity') or 0.0),
        ),
        reverse=True,
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

    # =========================================================
    # PASS 3: CHỐT ĐIỂM CHUẨN XÁC THEO YÊU CẦU
    # =========================================================
    for r in final_list:
        try:
            # Lấy độ khớp chuỗi (%) thật sự từ bộ lọc SQL
            original_sim = float(r.get('similarity') or 0.0)
            score = float(r.get('final_mix_score') or r.get('raw_score') or 0.0)
            
            # Nếu người dùng đang TÌM LỜI, hãy hiển thị Độ khớp lời thật sự ra Web!
            if action_mode == 'SEARCH_LYRIC' and original_sim > 0:
                r['similarity'] = min(0.99, original_sim)
            else:
                # Các luồng khác vẫn dùng điểm Ranker (nhưng cấm không cho rớt xuống âm)
                r['similarity'] = min(0.99, max(0.0, score / 2.0))
        except Exception:
            r['similarity'] = 0.0

    # SẮP XẾP BẰNG ĐIỂM TOÁN HỌC NGUYÊN BẢN (Tuyệt đối không dùng similarity bị giới hạn)
    final_list.sort(
        key=lambda x: (
            float(x.get('final_mix_score') or x.get('raw_score') or 0.0),
            float(x.get('spotify_popularity') or 0.0),
        ),
        reverse=True,
    )
    # --- [MỚI] DYNAMIC CUTOFF CHO TÌM LỜI BÀI HÁT ---
    if action_mode == 'SEARCH_LYRIC' and final_list:
        max_sim = max([r.get('similarity', 0.0) for r in final_list])
        
        # Nếu có bất kỳ bài nào quá chuẩn xác (>= 85%), dọn sạch rác dưới 75%
        if max_sim >= 0.85:
            final_list = [r for r in final_list if r.get('similarity', 0.0) >= 0.75]

    # Trả về Top K
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
    if any(k in m for k in ['da diet', 'kich tinh']):
        target_vibes.append("Kịch tính") # Sẽ khớp "Kịch tính / Da diết"
        
    if any(k in m for k in ['sâu lắng', 'tham', 'sau tham', 'thau cam']):
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
    if not genre_text: return ""
    text = _normalize_text(genre_text)
    if any(k in text for k in ['rap', 'hip hop', 'hiphop', 'trap', 'underground']): return "Rap/Hip-hop"
    if any(k in text for k in ['ballad']): return "Ballad"
    if any(k in text for k in ['indie']): return "Indie"
    if any(k in text for k in ['pop', 'nhac tre', 'nhạc trẻ', 'vpop', 'mainstream', 'hien dai', 'hiện đại']): return "Pop"
    return genre_text

def get_genre_targets(genre_text: str) -> list[str]:
    """Băm chuỗi đa thể loại (ngăn cách bởi dấu phẩy) và trả về list chuẩn"""
    if not genre_text: return []
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
    # 1. SEARCH_TRACK (Super Engine: Tìm Tên Bài Hát -> Ca Sĩ -> Tự động Fallback Lời Bài Hát)
    # =========================
    elif action == "SEARCH_TRACK":
        execution_path = ["Level 1: Track Search Router"]

        song_title = str(params.get("song_title") or "").strip()
        lyric_snippet = str(params.get("lyric_snippet") or "").strip().strip("'").strip('"')
        artist = str(params.get("artist", "") or "").strip()

        # Rút gọn các Query chung nếu có
        if not song_title and not lyric_snippet:
            q = str(params.get("query") or params.get("text") or "").strip()
            if q:
                song_title = q

        if not song_title and not lyric_snippet:
            return {
                'tracks': [],
                'source': 'fallback-missing-param',
                'error': 'Bạn muốn tìm bài hát theo tên hoặc theo một đoạn lời nhạc nào nhỉ?',
                'path': execution_path,
            }

        # ========================================================
        # CHIẾN LƯỢC 1: TÌM THEO TÊN BÀI HÁT (SEARCH_NAME Tích hợp)
        # ========================================================
        if song_title:
            execution_path.append("Level 1.1: Name Search Engine")
            print(f"[SEARCH_TRACK -> NAME] Đầu vào: Title='{song_title}', Artist='{artist}'")
            try:
                rows = []
                source_label = ""
                original_song_title = song_title

                def try_sql_search(title_query, artist_query):
                    
                    q = supabase.table('songs').select(
                        'spotify_track_id, title, artists, vibe, main_topic, final_sentiment, spotify_popularity, is_hit, genres'
                    )
                    
                    clean_title = unicodedata.normalize('NFC', title_query.strip())
                    if " " not in clean_title and len(clean_title) >= 4:
                        spaced_title = "%".join(clean_title)
                        q = q.ilike('title', f'%{spaced_title}%')
                    else:
                        q = q.ilike('title', f'%{clean_title}%')
                        
                    if artist_query:
                        clean_artist = unicodedata.normalize('NFC', artist_query.strip())
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
                    print(f"[SEARCH_TRACK] Không tìm thấy bản gốc. Bắt đầu phẫu thuật chuỗi: '{song_title}'")
                    
                    # --- LỚP 1: Xử Lý Số Đếm Mở Rộng ---
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
                        execution_path.append("Level 1.1.5: Numeric Mapping")
                        rows = try_sql_search(numeric_title, artist)
                        if rows:
                            source_label = 'text-search:numeric'
                            song_title = numeric_title 
                            for r in rows: r['similarity'] = 0.95

                    # --- LỚP 1.2: BÓC TÁCH CA SĨ BẰNG RAM CACHE (ĐƯA LÊN TRƯỚC ĐỂ TRÁNH BỊ CHẶN) ---
                    if not rows and not artist:
                        execution_path.append("Level 1.2: Entity Disambiguation")
                        words = numeric_title.split()
                        if len(words) >= 3:
                            all_songs = _get_all_songs_cached(supabase)
                            if all_songs:
                                # Tạo từ điển ca sĩ từ cache (Siêu tốc độ, không cần gọi SQL)
                                known_artists_map = {}
                                for s in all_songs:
                                    db_arts = str(s.get('artists') or '').split(',')
                                    for a in db_arts:
                                        a = a.strip()
                                        if a and len(a) >= 3:
                                            norm_a = _normalize_text(a)
                                            known_artists_map[norm_a.replace(" ", "")] = a
                                            known_artists_map[norm_a] = a
                                
                                for i in range(min(5, len(words) - 1), 0, -1):
                                    potential_artist_str = " ".join(words[-i:])
                                    if len(potential_artist_str) < 3: continue
                                    
                                    norm_pot = _normalize_text(potential_artist_str)
                                    norm_pot_nospace = norm_pot.replace(" ", "")
                                    
                                    best_match = ""
                                    
                                    # [FIX TẠI ĐÂY]: Luật bắt ca sĩ khắt khe hơn để tránh cắt nhầm tên bài
                                    if len(norm_pot_nospace) <= 4:
                                        # Tên quá ngắn (như BAN, MIN) phải khớp 100%
                                        if norm_pot_nospace in known_artists_map:
                                            best_match = known_artists_map[norm_pot_nospace]
                                    else:
                                        # Tên dài cho phép sai số 15% (threshold 0.85)
                                        for norm_db, real_db in known_artists_map.items():
                                            if norm_pot_nospace in norm_db and len(norm_pot_nospace) >= len(norm_db) * 0.85:
                                                best_match = real_db
                                                break
                                            elif norm_db in norm_pot_nospace and len(norm_db) >= len(norm_pot_nospace) * 0.85:
                                                best_match = real_db
                                                break
                                    
                                    if not best_match:
                                        # Nâng cutoff từ 0.85 lên 0.88 để cực kỳ an toàn
                                        matches = difflib.get_close_matches(norm_pot_nospace, [k for k in known_artists_map.keys() if " " not in k], n=1, cutoff=0.88)
                                        if matches: best_match = known_artists_map[matches[0]]

                                    if best_match:
                                        artist = best_match
                                        song_title = " ".join(words[:-i]).strip()
                                        print(f"[SEARCH_TRACK] AI Bóc Tách Dính Chữ -> Title: '{song_title}', Artist: '{artist}'")
                                        
                                        rows = try_sql_search(song_title, artist)
                                        if rows:
                                            source_label = 'text-search:split-artist'
                                            for r in rows: r['similarity'] = 0.90
                                            
                                        # Dời lệnh break ra ngoài if rows:
                                        # Mục đích: Chốt giữ song_title và artist chuẩn xác để
                                        # các tầng RapidFuzz/Vector phía dưới có "đạn" chuẩn để bắn
                                        break

                    # --- LỚP 1.5: PRODUCTION-GRADE FUZZY STICKY MATCH (ĐÃ CHẶN ĂN HÔI TÊN NGẮN) ---
                    if not rows:
                        all_songs = _get_all_songs_cached(supabase)
                        if all_songs:
                            execution_path.append("Level 1.3: RapidFuzz Sticky Engine")
                            q_t_sticky = _normalize_text(song_title).replace(" ", "")
                            q_a_norm = _normalize_text(artist)
                            
                            scored = []
                            for s in all_songs:
                                db_t_norm = _normalize_text(s.get('title') or '')
                                db_t_sticky = db_t_norm.replace(" ", "")
                                db_a_norm = _normalize_text(s.get('artists') or '')
                                
                                title_score = fuzz.ratio(q_t_sticky, db_t_sticky)
                                
                                # FIX CHÍNH LÀ ĐÂY: Chặn các bài có tên quá ngắn (<5 ký tự) ăn hôi Substring Match
                                is_substring = False
                                if len(db_t_sticky) >= 5 and db_t_sticky in q_t_sticky: 
                                    is_substring = True
                                elif len(q_t_sticky) >= 5 and q_t_sticky in db_t_sticky: 
                                    is_substring = True

                                if title_score > 85 or is_substring:
                                    sim = max(0.85, title_score / 100.0)
                                    
                                    if q_a_norm:
                                        artist_score = fuzz.token_set_ratio(q_a_norm, db_a_norm)
                                        if artist_score > 75: sim += 0.05
                                        else: sim -= 0.35 
                                    
                                    if sim > 0.6:
                                        s_copy = dict(s)
                                        s_copy['similarity'] = round(sim * 100, 2)
                                        scored.append(s_copy)
                            
                            if scored:
                                scored.sort(key=lambda x: x['similarity'], reverse=True)
                                rows = scored[:int(match_count)]
                                source_label = 'production-fuzzy-sticky'

                    # --- LỚP 2: Fallback Bỏ tên ca sĩ ---
                    if not rows and artist:
                        execution_path.append("Level 1.4: Attribute Fallback")
                        print(f"[Fallback] Bỏ qua ca sĩ '{artist}'...")
                        rows = try_sql_search(song_title, "")
                        if rows:
                            source_label = 'text-search:name-only-fallback'
                            for r in rows: r['similarity'] = 0.9

                    # --- LỚP 3: MÀNG LỌC FUZZY + VECTOR SIÊU CẤP ---
                    if not rows:
                        execution_path.append("Level 1.5: Non-accent Trigram")
                        full_query = f"{song_title} {artist}".strip()
                        clean_query = _normalize_text(full_query) 
                        print(f"[Fallback] Kích hoạt Non-Accent Vector & Fuzzy cho: '{clean_query}'...")
                        
                        try:
                            fuzzy_res = supabase.rpc('match_lyrics_fuzzy', {
                                'query_text': clean_query,
                                'match_threshold': 0.15,
                                'match_count': int(match_count) * 4
                            }).execute()
                            
                            f_rows = getattr(fuzzy_res, 'data', None) or []
                            if f_rows:
                                ids = [r['spotify_track_id'] for r in f_rows]
                                meta_res = supabase.table('songs').select('*').in_('spotify_track_id', ids).execute()
                                meta_map = {m['spotify_track_id']: m for m in (meta_res.data or [])}
                                
                                for f in f_rows:
                                    if f['spotify_track_id'] in meta_map:
                                        row = meta_map[f['spotify_track_id']]
                                        row['similarity'] = min(0.96, f.get('similarity', 0.5) + 0.3)
                                        rows.append(row)
                                
                                if rows:
                                    source_label = 'fuzzy-trigram:non-accent'
                                    
                        except Exception as e:
                            print(f"[Lỗi Fuzzy Fallback Name] {e}")

                # NẾU CHIẾN LƯỢC 1 TÌM THẤY -> RANK & RETURN NGAY
                if rows:
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
                print(f"[SEARCH_TRACK] Cảnh báo tại luồng Name Search: {ex}")
                # Không return, chủ động trượt xuống nhánh Lyric cứu viện

        # ========================================================
        # CHIẾN LƯỢC 2: FALLBACK TÌM THEO LỜI BÀI HÁT (SEARCH_LYRIC Tích hợp)
        # Được kích hoạt khi Strategy 1 tìm không ra bài, hoặc không có song_title
        # ========================================================
        lyric_query = lyric_snippet or song_title
        if not lyric_query:
            return {
                'tracks': [],
                'source': 'fallback-missing-param',
                'error': 'Bạn muốn tìm theo đoạn lời nào?',
                'path': execution_path,
            }

        execution_path.append("Level 2: Lyric Match Fallback")
        print(f"[SEARCH_TRACK -> LYRIC] Bắt đầu tìm kiếm thuần chuỗi: '{lyric_query}'")

        clean_query = _normalize_text(lyric_query)
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
            execution_path.append("Level 2.1: Fuzzy Trigram")
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
                execution_path.append("Level 2.2: Substring Chunking")
                print(f" -> Level 2.2: Điểm max = {max_sim:.2f}. Bắt đầu Chunking cứu viện...")
                
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

            # --- RANKING & DEDUP TỪ LYRIC ---
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
            
            return {'tracks': [], 'source': 'search-lyric-empty', 'error': f"Hệ thống không tìm thấy bài hát nào khớp với tên hoặc lời: '{lyric_query}'.", 'path': execution_path}

        except Exception as e:
            return {'tracks': [], 'source': 'search-lyric-error', 'error': f"Lỗi logic tìm kiếm (Lyric Phase): {e}", 'path': execution_path}
        

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
    # DISCOVER_MUSIC (Siêu động cơ hợp nhất: Trích xuất tất cả params)
    # =========================
    elif action == "DISCOVER_MUSIC":
        execution_path = ["Level 1: Unified Discover Engine"]

        mood = str(params.get("mood", "")).strip()
        genre = str(params.get("genre", "")).strip()
        artist = str(params.get("artist", "")).strip()
        seed_name = str(params.get("seed_name", "")).strip()
        attributes = str(params.get("attributes", "")).strip()
        is_popular = params.get("popularity_flag", False)

        # Kiểm tra Input
        if not any([mood, genre, artist, seed_name, attributes, is_popular]):
            return {
                'tracks': [], 
                'source': 'error', 
                'error': 'Bạn hãy cho mình biết một chút về tâm trạng, thể loại hoặc nghệ sĩ bạn muốn nghe nhé!', 
                'path': execution_path
            }
        def _has_kw(text: str, keywords: list[str]) -> bool:
            return any(re.search(rf'\b{re.escape(k)}\b', text) for k in keywords)   
        # -----------------------------------------------------------
        # NHÁNH 1: TÌM THEO BÀI HÁT MẪU (SEED) -> CHUYỂN TỚI RECOMMEND_SEED
        # Do cơ chế chấm điểm chênh lệch tuyến tính (Linear Penalty) 
        # quá đặc thù, nó cần chạy luồng riêng biệt.
        # -----------------------------------------------------------
        if seed_name:
            execution_path.append("Sub-Branch: Seed Recommendation")
            seed_params = {"seed_name": seed_name}
            if artist:
                seed_params["artist"] = artist

            res = handle_action(
                "RECOMMEND_SEED",
                seed_params,
                supabase,
                embed_fn=embed_fn,
                has_file=has_file,
                artist_list=artist_list,
                match_threshold=match_threshold,
                match_count=match_count,
            )
            if isinstance(res, dict):
                child_path = res.get('path') if isinstance(res.get('path'), list) else []
                res['path'] = execution_path + child_path
            return res
        # -----------------------------------------------------------
        # NHÁNH 2: ATTRIBUTES ENGINE (có thể kết hợp với Mood/Genre/Artist/Popularity)
        # - Seed vẫn là nhánh riêng ở trên.
        # - Khi có attributes, ta lọc trước theo tempo/energy (track_features) để có tập ứng viên,
        #   sau đó lọc + rank theo mood/genre/artist ngay trong Python.
        # -----------------------------------------------------------
        elif attributes or _has_kw(_normalize_text(str(params.get("attributes", ""))),['cham', 'slow', 'nhanh', 'fast', 'don dap', 'bpm', 'tempo']):
            execution_path.append("Sub-Branch: Attributes (+Filters)")
            
            raw_attr_check = _normalize_text(f"{params.get('attributes','')} {params.get('song_title','')} {params.get('mood','')}")
            
        
            def _parse_attribute_ranges(raw_text: str):
                min_t, max_t = 0, 250
                min_e, max_e = 0.0, 1.0
                target_tempo = None
                target_energy = None
                # =========================
                # 1. BPM (ƯU TIÊN CAO NHẤT)
                # =========================
                m_bpm = re.search(r'\b(\d{2,3})\b', raw_text)
                bpm_val = int(m_bpm.group(1)) if m_bpm else None

                if bpm_val and 40 <= bpm_val <= 250:
                    min_t = max(0, bpm_val - 5)
                    max_t = min(250, bpm_val + 5)
                    target_tempo = float(bpm_val)
                else:
                    # =========================
                    # 2. TEMPO KEYWORD
                    # =========================
                    if _has_kw(raw_text, ['rat nhanh', 'don dap', 'speed up']):
                        min_t, max_t = 168, 220
                        target_tempo = 180.0

                    elif _has_kw(raw_text, ['nhanh', 'fast', 'nhip nhanh']):
                        min_t, max_t = 120, 168
                        target_tempo = 140.0

                    elif _has_kw(raw_text, ['on dinh', 'nhip nhang', 'binh thuong']):
                        min_t, max_t = 90, 120
                        target_tempo = 105.0

                    elif _has_kw(raw_text, ['cham', 'slow']):
                        min_t, max_t = 40, 95
                        target_tempo = 70.0

                # =========================
                # 3. ENERGY
                # =========================
                if _has_kw(raw_text, ['nhe', 'mong manh', 'em diu']):
                    min_e, max_e = 0.0, 0.4
                    target_energy = 0.2

                elif _has_kw(raw_text, ['manh', 'cang', 'uy luc', 'day']):
                    min_e, max_e = 0.4, 1.0
                    target_energy = 0.6

                return min_t, max_t, min_e, max_e, target_tempo, target_energy

            # 👉 PARSE Ở ĐÂY
            min_t, max_t, min_e, max_e, target_tempo, target_energy = _parse_attribute_ranges(raw_attr_check)

            # 1) Pull attribute candidates với INNER JOIN để đảm bảo data toàn vẹn
            try:
                candidate_limit = max(200, int(match_count) * 60)
                res = (
                    supabase.table('track_features')
                    .select(
                        "spotify_track_id, tempo_bpm, rms_energy, songs!inner(title, artists, vibe, main_topic, spotify_popularity, is_hit, genres, final_sentiment)"
                    )
                    .gte('tempo_bpm', min_t)
                    .lte('tempo_bpm', max_t)
                    .gte('rms_energy', min_e)
                    .lte('rms_energy', max_e)
                    .limit(candidate_limit)
                    .execute()
                )

                rows: list[dict] = []
                for r in (getattr(res, 'data', None) or []):
                    s = r.get('songs', {}) or {}
                    rows.append(
                        {
                            'spotify_track_id': r.get('spotify_track_id'),
                            'spotify_id': r.get('spotify_track_id'),
                            'title': s.get('title'),
                            'artists': s.get('artists'),
                            'artist': s.get('artists'),
                            'vibe': s.get('vibe'),
                            'main_topic': s.get('main_topic'),
                            'final_sentiment': s.get('final_sentiment'),
                            'tempo_bpm': float(r.get('tempo_bpm', 0) or 0),
                            'rms_energy': float(r.get('rms_energy', 0) or 0),
                            'spotify_popularity': s.get('spotify_popularity'),
                            'is_hit': s.get('is_hit'),
                            'genres': s.get('genres'),
                        }
                    )
            except Exception as e:
                return {
                    'tracks': [],
                    'source': 'discover-attributes-error',
                    'error': str(e),
                    'path': execution_path,
                }

            if not rows:
                return {
                    'tracks': [],
                    'source': 'discover-attributes-empty',
                    'error': 'Không tìm thấy track nào khớp tempo/năng lượng bạn mô tả.',
                    'path': execution_path,
                }

            # 2) Build boosts (reuse mapping logic for mood/genre/artist)
            boosts: dict = {}
            mapped_genres: list[str] = []
            target_vibes: list[str] = []
            target_topics: list[str] = []
            target_sentiment: str = ""

            if mood:
                target_vibes, target_topics, target_sentiment = _mood_maps(mood)
                boosts.update({'vibe': target_vibes, 'topics': target_topics, 'sentiment': target_sentiment})
            if genre:
                mapped_genres = get_genre_targets(genre)
                ranker_genres: list[str] = []
                for g in mapped_genres:
                    ranker_genres.extend([x.strip() for x in re.split(r'[,/]', g) if x.strip()])
                boosts['genre'] = ranker_genres
            if artist:
                boosts['artist'] = artist

            if target_tempo is not None:
                boosts['target_tempo'] = target_tempo
            if target_energy is not None:
                boosts['target_energy'] = target_energy

            # 3) Lọc dựa trên Mood/Genre/Artist
            def _matches_mood(r: dict, *, loose: bool = False) -> bool:
                if not mood:
                    return True
                db_sent = str(r.get('final_sentiment') or '').lower().strip()
                db_vibe = _normalize_text(r.get('vibe') or '')
                db_topic = _normalize_text(r.get('main_topic') or '')

                if target_sentiment and db_sent:
                    if db_sent == target_sentiment:
                        return True
                    if not loose:
                        return False

                if target_vibes:
                    for v in target_vibes:
                        v_norm = _normalize_text(v)
                        if v_norm and v_norm in db_vibe:
                            return True
                if target_topics:
                    for t in target_topics:
                        t_norm = _normalize_text(t)
                        if t_norm and t_norm in db_topic:
                            return True
                return not (target_sentiment or target_vibes or target_topics)

            def _matches_genre(r: dict, *, op: str = 'AND') -> bool:
                if not mapped_genres:
                    return True
                db_gen = _normalize_text(r.get('genres') or '')
                if not db_gen:
                    return False
                if op == 'OR':
                    return any(_normalize_text(g) in db_gen for g in mapped_genres)
                return all(_normalize_text(g) in db_gen for g in mapped_genres)
            
            def _matches_artist(r: dict) -> bool:
                if not artist:
                    return True
                db_art = _normalize_text(r.get('artists') or r.get('artist') or '')
                a = _normalize_text(artist)
                return bool(a) and a in db_art

            filtered = [r for r in rows if _matches_mood(r, loose=False) and _matches_genre(r, op='AND') and _matches_artist(r)]
            
            if not filtered and mapped_genres and len(mapped_genres) > 1:
                execution_path.append('Level 1.5: Relaxed Genre (OR)')
                filtered = [r for r in rows if _matches_mood(r, loose=False) and _matches_genre(r, op='OR') and _matches_artist(r)]

            if not filtered and mood:
                execution_path.append('Level 2: Relaxed Mood')
                filtered = [r for r in rows if _matches_mood(r, loose=True) and _matches_genre(r, op='OR' if (mapped_genres and len(mapped_genres) > 1) else 'AND') and _matches_artist(r)]

            if not filtered and (mood or genre or artist):
                execution_path.append('Level 3: Attributes-Only Fallback')
                filtered = rows

            # 4) Xếp hạng
            ranked = rank_and_normalize_tracks(raw_rows=filtered, limit=int(match_count), boosts=boosts)

            return {
                'tracks': ranked,
                'source': 'discover-engine:attributes-combo',
                'error': None,
                'path': execution_path,
            }

        # -----------------------------------------------------------
        # NHÁNH 3: POPULARITY MODE (có thể kết hợp Mood/Genre/Artist)
        # - Khi user yêu cầu "hot/trending/top", ưu tiên Top-N theo popularity trong phạm vi filter.
        # - Nếu có attributes, nhánh trên sẽ xử lý luôn (có thể kèm popularity_flag mà không cần branch này).
        # -----------------------------------------------------------
        elif is_popular:
            execution_path.append("Sub-Branch: Popularity Top-N")
            q = supabase.table('songs').select(
                'spotify_track_id, title, artists, genres, vibe, main_topic, final_sentiment, spotify_popularity, is_hit'
            )

            # Artist filter (wildcard whitespace to match quickly)
            if artist:
                q = q.ilike('artists', f'%{artist.replace(" ", "%")}%')

            # Genre filter
            if genre:
                mapped_genres = get_genre_targets(genre)
                if mapped_genres:
                    q = q.or_(",".join([f"genres.ilike.%{g}%" for g in mapped_genres]))

            # Mood filter (reuse mood map semantics)
            if mood:
                target_vibes, target_topics, target_sentiment = _mood_maps(mood)
                mood_filters: list[str] = []
                if target_sentiment:
                    mood_filters.append(f"final_sentiment.eq.{target_sentiment}")
                if target_vibes:
                    mood_filters.extend([f"vibe.ilike.%{v}%" for v in target_vibes])
                if target_topics:
                    mood_filters.extend([f"main_topic.ilike.%{t}%" for t in target_topics])
                if mood_filters:
                    q = q.or_(",".join(mood_filters))

            res = q.order('spotify_popularity', desc=True).limit(int(match_count)).execute()
            rows = getattr(res, 'data', None) or []
            if not rows:
                return {
                    'tracks': [],
                    'source': 'discover-popularity-empty',
                    'error': 'Chưa tìm thấy track hot/trending theo điều kiện bạn chọn.',
                    'path': execution_path,
                }

            ranked = rank_and_normalize_tracks(
                raw_rows=rows,
                limit=int(match_count),
                boosts={'artist': artist, 'genre': get_genre_targets(genre) if genre else None},
            )

            return {
                'tracks': ranked,
                'source': 'discover-engine:popularity',
                'error': None,
                'path': execution_path,
            }
        print(f"[DISCOVER_MUSIC] Input: Mood='{mood}', Genre='{genre}', Artist='{artist}'")

        try:
            rows = []
            source_label = ""
            boosts = {}
            
            # --- BƯỚC 1: BÓC TÁCH DỮ LIỆU CHUẨN XÁC ---
            target_vibes, target_topics, target_sentiment = [], [], ""
            if mood:
                target_vibes, target_topics, target_sentiment = _mood_maps(mood)
                boosts.update({'vibe': target_vibes, 'topics': target_topics, 'sentiment': target_sentiment})
            
            mapped_genres = []
            if genre:
                mapped_genres = get_genre_targets(genre)
                # Tối ưu cho Ranker: Phải tách "Rap/Hip-hop" ra để Ranker cộng điểm phụ chính xác
                ranker_genres = []
                for g in mapped_genres:
                    ranker_genres.extend([x.strip() for x in re.split(r'[,/]', g) if x.strip()])
                boosts['genre'] = ranker_genres
                
            artist_query = artist
            if artist:
                boosts['artist'] = artist

            # --- HÀM BUILDER TRUY VẤN SQL ĐỘNG ---
            def build_discover_query(use_exact_genre=False, genre_operator='AND', use_wildcard_artist=False, fuzzy_artist_name=None, fetch_limit=None):
                q = supabase.table('songs').select(
                    'spotify_track_id, title, artists, vibe, main_topic, final_sentiment, spotify_popularity, is_hit, genres'
                )
                
                # Ráp Nghệ sĩ
                if fuzzy_artist_name:
                    q = q.ilike('artists', f'%{fuzzy_artist_name}%')
                elif artist_query:
                    if use_wildcard_artist:
                        aw = artist_query.replace(' ', '%')
                        q = q.ilike('artists', f'%{aw}%')
                    else:
                        q = q.ilike('artists', f'%{artist_query}%')
                        
                # Ráp Thể loại
                if mapped_genres:
                    if use_exact_genre and len(mapped_genres) == 1:
                        q = q.eq('genres', mapped_genres[0])
                    else:
                        if genre_operator == 'AND':
                            for g in mapped_genres:
                                q = q.ilike('genres', f'%{g}%')
                        elif genre_operator == 'OR':
                            or_conds = ",".join([f"genres.ilike.%{g}%" for g in mapped_genres])
                            q = q.or_(or_conds)
                        
                # Ráp Tâm trạng
                if mood:
                    mood_filters = []
                    if target_sentiment:
                        mood_filters.append(f"final_sentiment.eq.{target_sentiment}")
                    if target_vibes:
                        mood_filters.extend([f"vibe.ilike.%{v}%" for v in target_vibes])
                    if target_topics:
                        mood_filters.extend([f"main_topic.ilike.%{t}%" for t in target_topics])
                    if mood_filters:
                        q = q.or_(",".join(mood_filters))
                        
                lim = fetch_limit if fetch_limit else max(200, int(match_count) * 10)
                return q.order('spotify_popularity', desc=True).limit(lim)

            def merge_rows(existing_list, new_rows):
                seen = {r['spotify_track_id'] for r in existing_list}
                for r in new_rows:
                    if r['spotify_track_id'] not in seen:
                        existing_list.append(r)
                        seen.add(r['spotify_track_id'])

            # --- SỬA LỖI UNPACK VÀ TỐI ƯU HÓA MÀNG LỌC ---
            def fetch_and_clean(query_obj, current_artist):
                # 1. Gọi execute() ngay trong hàm để tránh lỗi chưa resolve object SQL
                res = query_obj.execute()
                raw_rows = getattr(res, 'data', None) or []
                
                # 2. Đảm bảo LUÔN TRẢ VỀ TUPLE 2 PHẦN TỬ
                if not raw_rows or not current_artist or (' ' in current_artist.strip()):
                    return raw_rows[:int(match_count) * 4], raw_rows[:int(match_count) * 4]
                    
                exact_rows = []
                target_artist = current_artist.lower().strip()
                target_norm = _normalize_text(target_artist)
                
                # 3. Lọc gắt: Phải đứng độc lập (vd: "Tùng" là "Tùng", không lọt "Sơn Tùng")
                for r in raw_rows:
                    db_artists_list = [a.strip().lower() for a in str(r.get('artists', '')).split(',') if a.strip()]
                    db_artists_norm = [_normalize_text(a) for a in db_artists_list]
                    
                    if target_artist in db_artists_list or target_norm in db_artists_norm:
                        exact_rows.append(r)
                        
                if exact_rows:
                    return exact_rows[:int(match_count) * 4], raw_rows[:int(match_count) * 4]
                else:
                    return [], raw_rows[:int(match_count) * 4]

            # --- BƯỚC 2: CHUỖI TRUY VẤN XUYÊN THẤU (GRACEFUL DEGRADATION) ---
            
            is_single_word_artist = bool(artist_query and ' ' not in artist_query.strip())
            lim = 300 if is_single_word_artist else int(match_count) * 4

            # LỚP 1: TÌM CHÍNH XÁC (STRICT MATCH)
            q1_exact = build_discover_query(use_exact_genre=True, genre_operator='AND', fetch_limit=lim)
            clean_1, raw_1 = fetch_and_clean(q1_exact, artist_query)
            merge_rows(rows, clean_1 if is_single_word_artist else raw_1)
            
            if len(rows) < int(match_count) * 2:
                q1_like = build_discover_query(use_exact_genre=False, genre_operator='AND', fetch_limit=lim)
                clean_2, raw_2 = fetch_and_clean(q1_like, artist_query)
                merge_rows(rows, clean_2 if is_single_word_artist else raw_2)
                
            if rows:
                source_label = 'discover-engine:strict-sql'

            # LỚP 1.5: FALLBACK BẢN GỐC 
            # Dành cho trường hợp nghệ sĩ thực sự không có ai tên "Tùng", thì đành trả về "Sơn Tùng"
            if not rows and is_single_word_artist:
                execution_path.append("Level 1.5: Relaxed Artist (False-Positive Fallback)")
                merge_rows(rows, raw_1)
                if len(rows) < int(match_count) * 2:
                    merge_rows(rows, raw_2)
                if rows:
                    source_label = 'discover-engine:relaxed-artist-sql'

            # LỚP 2: AI DÒ CHÍNH TẢ NGHỆ SĨ TỪ TỪ ĐIỂN
            # (Xử lý "sơm tùng" / "sơn tug" -> "Sơn Tùng M-TP" trước khi Nới lỏng Wildcard)
            if not rows and artist:
                execution_path.append("Level 2: AI Spelling Correction")
                try:
                    current_artist_list = _load_artist_list_from_supabase(supabase)
                    if current_artist_list:
                        query_norm = _normalize_text(artist)
                        match = _extract_one(query_norm, [_normalize_text(a) for a in current_artist_list])
                        
                        # Chặn sửa bậy: Ngưỡng 88% và lệch tối đa 7 kí tự (vd: "son tug" -> "sơn tùng m-tp" = hợp lệ)
                        if match and match[1] >= 85 and abs(len(query_norm) - len(match[0])) <= 7:
                            best_artist = current_artist_list[match[2]]
                            if best_artist.lower() != artist.lower():
                                print(f"[DISCOVER_MUSIC] Tự động sửa lỗi chính tả: '{artist}' -> '{best_artist}'")
                                
                                # Tìm lại với tên đã được AI sửa (best_artist)
                                q_spell = build_discover_query(use_exact_genre=False, genre_operator='OR', fuzzy_artist_name=best_artist)
                                res_spell = q_spell.execute()
                                merge_rows(rows, getattr(res_spell, 'data', None) or [])
                                
                                if rows:
                                    source_label = 'discover-engine:artist-fuzzy-sql'
                                    boosts['artist'] = best_artist
                except Exception as e:
                    print(f"[DISCOVER_MUSIC] Cảnh báo dò chính tả: {e}")

            # LỚP 3: NỚI LỎNG THỂ LOẠI (OR) VÀ DÍNH CHỮ NGHỆ SĨ (WILDCARD)
            # (Xử lý "sơntùng", nhưng AI Spelling đã chặn tốt rồi nên phần này là bảo hiểm cuối)
            if not rows and (len(mapped_genres) > 1 or artist):
                execution_path.append("Level 3: Relaxed SQL (Wildcard/OR)")
                res_wildcard = build_discover_query(use_exact_genre=False, genre_operator='OR', use_wildcard_artist=True).execute()
                merge_rows(rows, getattr(res_wildcard, 'data', None) or [])
                if rows:
                    source_label = 'discover-engine:relaxed-sql'
                    if artist:
                        boosts['artist'] = artist.replace(' ', '%') 

            # LỚP 4: VECTOR FALLBACK SIÊU CẤP
            if not rows and (mood or genre or params.get("raw_query")):
                execution_path.append("Level 4: Semantic Vector Space")
                
                # Chiến thuật: Ưu tiên dùng câu gốc (raw_query) nếu mood/genre bị rỗng
                # Nếu có mood/genre thì gộp lại để tăng độ chính xác
                raw_q = params.get("raw_query", "").strip()
                keywords_part = f"{mood} {genre} {artist}".strip()
                
                if not keywords_part:
                    combo_text = raw_q
                else:
                    # Gộp cả 2 để AI hiểu cả từ khóa lẫn ngữ cảnh câu văn
                    combo_text = f"{raw_q} ({keywords_part})".strip()

                print(f"[Fallback] Kích hoạt Vector Search cho: '{combo_text}'")
                query_embedding = _safe_embed(embed_fn, combo_text)
                if query_embedding:
                    thr = float(match_threshold) if match_threshold is not None else 0.35
                    res_vec = supabase.rpc(
                        "match_vpop_tracks", 
                        {"query_embedding": query_embedding, "match_threshold": thr, "match_count": int(match_count) * 4}
                    ).execute()
                    merge_rows(rows, getattr(res_vec, 'data', None) or [])
                    if rows:
                        source_label = 'discover-engine:vector-fallback'
                        if not mood and not genre:
                            boosts['is_deep_semantic'] = True

            # CHỐT CHẶN CUỐI
            if not rows:
                return {
                    'tracks': [], 
                    'source': 'discover-empty', 
                    'error': "Khẩu vị của bạn đặc biệt quá, hệ thống đã quét hết các lớp màng lọc nhưng vẫn chưa ra bài nào khớp hoàn toàn!", 
                    'path': execution_path
                }

            # --- BƯỚC 3: BƠM ĐIỂM SIMILARITY MÔ PHỎNG ---
            if 'vector-fallback' not in source_label:
                for r in rows:
                    match_score = 0.65 
                    if boosts.get('artist') and _normalize_text(boosts['artist']).replace("%", " ") in _normalize_text(r.get('artists') or ''): 
                        match_score += 0.15
                    if mapped_genres and any(_normalize_text(g) in _normalize_text(r.get('genres') or '') for g in mapped_genres):
                        match_score += 0.10
                    if mood:
                        db_vibe, db_topic = _normalize_text(r.get('vibe') or ''), _normalize_text(r.get('main_topic') or '')
                        if target_vibes and any(_normalize_text(v) in db_vibe for v in target_vibes): match_score += 0.12
                        if target_topics and any(_normalize_text(t) in db_topic for t in target_topics): match_score += 0.12
                        if target_sentiment and str(r.get('final_sentiment') or '').lower() == target_sentiment: match_score += 0.08
                    r['similarity'] = min(0.98, match_score)

            # --- BƯỚC 4: GLOBAL RANKER ---
            ranked = rank_and_normalize_tracks(raw_rows=rows, limit=int(match_count), boosts=boosts)

            return {'tracks': ranked, 'source': source_label, 'error': None, 'path': execution_path}

        except Exception as e:
            return {'tracks': [], 'source': 'discover-error', 'error': f"Lỗi siêu truy vấn: {e}", 'path': execution_path}

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
            clean_seed_name = seed_name.strip(' "\'') 
            seed_track = None
            # --- BƯỚC 2: LỚP 1 - TÌM SQL (Ưu tiên bài Hot nhất nếu trùng tên) ---
            song_q = supabase.table('songs').select(
                'spotify_track_id, title, artists, vibe, genres, final_sentiment, spotify_popularity'
            ).ilike('title', f'%{clean_seed_name}%')
            
            if seed_artist_query:
                aw = seed_artist_query.replace(' ', '%')
                song_q = song_q.ilike('artists', f'%{aw}%')
                execution_path.append('Level 1.1: Seed Artist Filter')

            song_f = song_q.order('spotify_popularity', desc=True).limit(15).execute()
            
            if song_f.data:
                best_seed = None
                best_score = -9999
                
                for track in song_f.data:
                    t_title = str(track.get('title', '')).strip().lower()
                    t_artists = str(track.get('artists', '')).strip().lower()
                    pop = float(track.get('spotify_popularity') or 0)
                    score = pop  # Điểm nền tảng là độ hot
                    
                    # 1. Thưởng CỰC ĐẬM cho bài khớp chính xác 100% tên
                    clean_db_title = re.sub(r'\(.*?\)|\[.*?\]', '', t_title).strip()
                    if clean_db_title == clean_seed_name.lower():
                        score += 1000
                    
                    # 2. Phạt NẶNG bản phái sinh ghi trên Title
                    if "remix" not in clean_seed_name.lower() and "cover" not in clean_seed_name.lower():
                        if "remix" in t_title or "mix" in t_title or "dj" in t_title:
                            score -= 500
                        if "cover" in t_title or "live" in t_title or "version" in t_title:
                            score -= 300
                            
                    # 3. Phạt NẶNG nghệ sĩ/kênh phái sinh ghi ở trường Artists
                    if not seed_artist_query:
                        if any(kw in t_artists for kw in DERIVATIVE_ARTISTS):
                            score -= 800  
                            
                    if score > best_score:
                        best_score = score
                        best_seed = track
                        
                seed_track = best_seed
            else:
                # --- BƯỚC 3: LỚP 2 - LOCAL SMART FUZZY (Cứu cánh không dấu, dính chữ) ---
                execution_path.append("Level 2: Fuzzy Seed Lookup")
                all_songs = _get_all_songs_cached(supabase)
                if all_songs:
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
                                
                                # [MỚI] Áp dụng logic BẢO VỆ BẢN GỐC cho cả mảng Fuzzy Search
                                penalty_bonus = 0
                                db_t_lower = db_t.lower()
                                db_a_lower = db_a.lower()
                                
                                if "remix" not in query_t and "cover" not in query_t:
                                    if "remix" in db_t_lower or "mix" in db_t_lower or "dj" in db_t_lower:
                                        penalty_bonus -= 50
                                    if "cover" in db_t_lower or "live" in db_t_lower or "version" in db_t_lower:
                                        penalty_bonus -= 30
                                        
                                if not seed_artist_query:
                                    if any(_normalize_text(kw) in db_a_lower for kw in DERIVATIVE_ARTISTS):
                                        penalty_bonus -= 80
                                        
                                s_copy['penalty_bonus'] = penalty_bonus
                                scored.append(s_copy)
                                
                        if scored:
                            # Ranking: Tổng điểm (Độ khớp + Điểm phạt) là số 1, Độ phổ biến là số 2
                            scored.sort(key=lambda x: (x['seed_sim_score'] + x.get('penalty_bonus', 0), float(x.get('spotify_popularity') or 0)), reverse=True)
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
        