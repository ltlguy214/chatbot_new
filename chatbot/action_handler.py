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
    text = (text or '').lower().strip()
    text = unicodedata.normalize('NFD', text)
    text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
    text = re.sub(r'[^a-z0-9 ]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


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
            # Include these for UI display if needed
            'genres': r.get('genres') or r.get('genres') or '',
            'is_hit': r.get('is_hit') or 0
        }

        if 'similarity' in r:
            item['similarity'] = r.get('similarity')
        if 'score' in r:
            item['score'] = r.get('score')
        for key in ['tempo_bpm', 'rms_energy', 'spotify_popularity', 'tempo', 'energy', 'popularity']:
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
    target_genre = _normalize_text(boosts.get('genre') or '')
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
        if action_mode == 'mood':
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

        if target_genre:
            if target_genre in db_genre_list: boost_score += 0.25
            elif any(target_genre in g for g in db_genre_list): boost_score += 0.15

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

    # Sắp xếp lần 1: Trả về thứ tự chất lượng thực tế (Chưa dính Diversity)
    ranked.sort(key=lambda x: x['raw_score'], reverse=True)

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
    final_list.sort(key=lambda x: x['final_mix_score'], reverse=True)

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
    PHIÊN BẢN NÂNG CẤP (MULTI-SIGNAL)
    Return: (target_vibes_list, target_topics_list, sentiment)
    """
    if not mood_text:
        return [], [], ""
        
    m = _normalize_text(mood_text)
    
    # 1. NHÓM BUỒN / SUY
    if any(
        k in m
        for k in [
            'buon', 'sad', 'deep',
            'that tinh', 'khoc', 'co don',
            'luy', 'suy',
            'tam trang',
            'sau', 'sau lang', 'sau tham',
            'tham', 'tham thia',
            'dau', 'dau long', 'nang long',
            'thui ruot', 'ruot gan',
            'tuyet vong',
        ]
    ):
        return ["Sâu lắng", "Kịch tính"], ["chia tay", "thất tình", "cô đơn", "phản bội"], "negative"
        
    # 2. NHÓM CHILL / CHỮA LÀNH
    if any(k in m for k in ['chill', 'thu gian', 'binh yen', 'lofi', 'nhe nhang', 'healing', 'chua lanh', 'relax']):
        return ["Bình yên"], ["kỷ niệm", "cuộc sống", "yên bình"], "positive"
        
    # 3. NHÓM VUI / TÍCH CỰC
    if any(k in m for k in ['vui', 'happy', 'tuoi moi', 'yeu doi', 'tich cuc']):
        return ["Tươi mới", "Bùng nổ"], ["tình yêu", "hạnh phúc", "thanh xuân"], "positive"
        
    # 4. NHÓM QUẨY / SÔI ĐỘNG
    if any(k in m for k in ['quay', 'party', 'sung', 'chay', 'soi dong', 'bung no']):
        return ["Bùng nổ"], [], "positive"
        
    # Fallback
    return [mood_text], [], ""


def get_genre_target(genre_text: str) -> str:
    """
    Ánh xạ từ lóng thể loại thành Thể loại chuẩn DB (Supabase Exact Match).
    Các Base Genres trên DB: "Rap/Hip-hop", "Pop", "Ballad", "Indie", "EDM", "R&B".
    """
    if not genre_text:
        return ""
        
    text = _normalize_text(genre_text)
    
    # Sửa "Rap" thành "Rap/Hip-hop" cho đúng chuẩn Database
    if any(k in text for k in ['rap', 'hip hop', 'hiphop', 'underground', 'trap']): 
        return "Rap/Hip-hop"
        
    if any(k in text for k in ['ballad', 'nhac buon', 'tru tinh', 'bolero', 'luy']): 
        return "Ballad"
        
    if any(k in text for k in ['indie', 'lofi', 'acoustic', 'chill']): 
        return "Indie"
        
    if any(k in text for k in ['edm', 'remix', 'vinahouse', 'dance', 'nhac san']): 
        return "EDM"
        
    if any(k in text for k in ['r&b', 'rnb', 'soul']): 
        return "R&B"
        
    if any(k in text for k in ['pop', 'nhac tre', 'vpop', 'mainstream', 'hien dai']): 
        return "Pop"
    
    return genre_text # Fallback: Nếu user nhập một thể loại lạ, cứ giữ nguyên để ilike tìm thử


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
        return {'tracks': [], 'source': 'fallback-no-supabase-client', 'error': 'Chưa kết nối được Supabase client'}

    # =========================
    # 1. SEARCH_NAME (Tìm Tên Bài Hát + Nghệ sĩ)
    # =========================
    elif action == "SEARCH_NAME": # Chuyển thành elif cho mượt
        song_title = str(params.get("song_title") or "").strip()
        artist = str(params.get("artist", "") or "").strip()

        if not song_title:
            return {'tracks': [], 'source': 'fallback-missing-param', 'error': 'Bạn muốn tìm bài hát tên gì nhỉ?'}

        print(f"[SEARCH_NAME] Đang tìm: '{song_title}' - Nghệ sĩ: '{artist}'")

        try:
            rows = []
            source_label = ""

            # --- LỚP 1: Tìm kết hợp cả Tên bài + Nghệ sĩ (Chính xác cao nhất) ---
            q1 = supabase.table('songs').select(
                'spotify_track_id, title, artists, vibe, main_topic, final_sentiment, spotify_popularity, is_hit, genres'
            ).ilike('title', f'%{song_title}%')
            
            if artist:
                q1 = q1.ilike('artists', f'%{artist}%')
            
            # Bỏ .order() đi, cứ lấy dư x4 để Ranker tự sắp xếp
            res1 = q1.limit(int(match_count) * 4).execute()
            rows = getattr(res1, 'data', None) or []
            
            if rows:
                source_label = 'text-search:name+artist'

            # --- LỚP 2: Fallback Bỏ tên ca sĩ (Râu ông nọ cắm cằm bà kia) ---
            if not rows and artist:
                print(f"[Fallback] Không thấy '{song_title}' do ca sĩ '{artist}' hát. Bỏ tên ca sĩ, chỉ tìm theo tên bài hát...")
                q2 = supabase.table('songs').select(
                    'spotify_track_id, title, artists, vibe, main_topic, final_sentiment, spotify_popularity, is_hit, genres'
                ).ilike('title', f'%{song_title}%').limit(int(match_count) * 4)
                
                res2 = q2.execute()
                rows = getattr(res2, 'data', None) or []
                
                if rows:
                    source_label = 'text-search:name-only-fallback'

            # --- LỚP 3: Vector Search (Cứu cánh Tiếng Việt không dấu / Sai chính tả) ---
            if not rows:
                print(f"[Fallback] Kích hoạt Vector Search để đoán tên bài hát: '{song_title}'...")
                try:
                    query_embedding = _safe_embed(embed_fn, song_title)
                    if query_embedding:
                        thr = float(match_threshold) if match_threshold is not None else 0.5
                        res3 = supabase.rpc(
                            "match_vpop_tracks",
                            {"query_embedding": query_embedding, "match_threshold": thr, "match_count": int(match_count) * 4},
                        ).execute()
                        
                        rows = getattr(res3, 'data', None) or []
                        if rows:
                            source_label = 'vector-fallback:name'
                except Exception as e:
                    print(f"[Lỗi Vector Fallback Tên bài] {e}")
            
            # --- KẾT THÚC: Nếu cả 3 lớp đều bó tay ---
            if not rows:
                return {'tracks': [], 'source': 'search-name-empty', 'error': f"Tiếc quá, hệ thống hiện chưa có bài '{song_title}'."}

            # =======================================================
            # GLOBAL RANKER
            # =======================================================
            ranked_tracks = rank_and_normalize_tracks(
                raw_rows=rows,
                limit=int(match_count),
                # Ranker sẽ lấy title và artist để thưởng điểm (+0.3 cho bài nào trùng khớp tên hoàn toàn)
                boosts={'title': song_title, 'artist': artist}
            )

            return {
                'tracks': ranked_tracks, 
                'source': source_label, 
                'error': None
            }
        
        except Exception as ex:
            return {'tracks': [], 'source': 'search-name-error', 'error': f"Lỗi tìm kiếm: {ex}"}
        

    # =========================
    # 2. SEARCH_LYRIC
    # =========================
    elif action == "SEARCH_LYRIC":
        lyric = str(params.get("lyric_snippet", "") or "").strip()
        if not lyric:
            return {
                'tracks': [], 
                'source': 'fallback-missing-param', 
                'error': 'Bạn muốn tìm theo đoạn lời nào?'
            }
                    
        # --- LỚP 1: TÌM CHÍNH XÁC (Hybrid Text Search: Title + Lyric) ---
        print(f"[Text Search] Đang tìm chính xác đoạn lời hoặc tiêu đề: '{lyric}'")
        try:
            # 1. Tìm Tiêu đề (Ưu tiên)
            title_res = supabase.table('songs').select('title, artists, vibe, main_topic, spotify_track_id, spotify_popularity').ilike('title', f'%{lyric}%').limit(3).execute()
            title_rows = getattr(title_res, 'data', None) or []
            
            # 2. Tìm Lời bài hát & Lưu toàn bộ Snippet
            lyric_res = supabase.table('lyrics').select('spotify_track_id, lyric').ilike('lyric', f'%{lyric}%').limit(20).execute()
            l_rows = getattr(lyric_res, 'data', None) or []
            
            lyric_ids = []
            snippet_dict = {} # MỚI: Cuốn sổ lưu snippet theo ID bài hát
            
            for r in l_rows:
                tid = str(r.get('spotify_track_id') or '')
                if tid:
                    lyric_ids.append(tid)
                    full_text = r.get('lyric')
                    if full_text:
                        start_idx = full_text.lower().find(lyric.lower())
                        if start_idx != -1:
                            start = max(0, start_idx - 25)
                            end = min(len(full_text), start_idx + len(lyric) + 35)
                            snippet_dict[tid] = full_text[start:end].replace('\n', ' / ')

            # 3. Lấy thông tin bài hát từ ID lời nhạc
            lyric_tracks = []
            if lyric_ids:
                songs_q = supabase.table('songs').select('title, artists, vibe, main_topic, spotify_track_id, spotify_popularity').in_('spotify_track_id', lyric_ids).order('spotify_popularity', desc=True).limit(10)
                songs_res = songs_q.execute()
                lyric_tracks = getattr(songs_res, 'data', None) or []

            # 4. Gộp và lọc trùng
            combined_rows = title_rows + lyric_tracks
            unique_tracks = {r['spotify_track_id']: r for r in combined_rows if r.get('spotify_track_id')}
            final_rows = list(unique_tracks.values())[:int(match_count)]

            # MỚI: Khớp Snippet với bài hát có rank cao nhất (Top 1)
            final_snippet = ""
            for track in final_rows:
                tid = track.get('spotify_track_id')
                if tid in snippet_dict:
                    final_snippet = snippet_dict[tid]
                    break # Tìm thấy snippet của bài cao nhất thì dừng ngay

            if final_rows:
                return {
                    'tracks': _normalize_track_rows(final_rows),    #spotify_id, title, artists, vibe, main_topic
                    'source': 'text-search:title+lyrics',
                    'snippet': f"...{final_snippet.strip()}..." if final_snippet else "", # Trả về snippet để UI hiển thị  
                    'error': None,
                }
        except Exception as e:
            print(f"[Lỗi Text Search Lời nhạc] {e}")

        # --- LỚP 2: FALLBACK VECTOR SEARCH (Cứu cánh khi nhớ sai lời) ---
        print(f"[Fallback] Không tìm thấy mặt chữ '{lyric}'. Đang dùng AI Vector Search đoán ngữ nghĩa...")
        try:
            query_embedding = _safe_embed(embed_fn, lyric)
            if not query_embedding:
                detail = _embed_error(embed_fn)
                suffix = f" ({detail})" if detail else ""
                return {
                    'tracks': [],
                    'source': 'fallback-missing-embedding',
                    'error': 'Chưa có embedding function hoặc không tạo được embedding' + suffix,
                }
                
            thr = float(match_threshold) if match_threshold is not None else 0.5
            res = supabase.rpc(
                "match_vpop_tracks",
                {
                    "query_embedding": query_embedding,
                    "match_threshold": thr,
                    "match_count": int(match_count),
                },
            ).execute()
            
            rows = getattr(res, 'data', None)
            return {
                'tracks': _normalize_track_rows(rows), #spotify_id, title, artists, vibe, main_topic
                'source': 'vector-fallback:lyrics',
                'error': None,
            }
        except Exception as e:
            return {
                'tracks': [],
                'source': 'fallback-lyric-error',
                'error': f"Lỗi trong quá trình tìm lời: {e}"
            }
    # =========================
    # 3. SEARCH_AUDIO (Nâng cấp: Dùng Backend Scaler + Multi-segment)
    # =========================
    elif action == "SEARCH_AUDIO":
        # params lúc này chứa 'audio_path' do Streamlit gửi qua
        audio_path = params.get("audio_path")
        
        if not audio_path or not os.path.exists(audio_path):
            return {
                'tracks': [], 
                'source': 'error', 
                'error': 'Không tìm thấy file âm thanh. Vui lòng kiểm tra lại Sidebar!'
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
                return {'tracks': [], 'source': 'audio-search-error', 'error': res["error"]}

            tracks_data = res.get("tracks", [])

            return {
                'tracks': _normalize_track_rows(tracks_data)[:1], 
                'source': 'audio-similarity-scientific-40d',
                'error': None
            }
            
        except Exception as e:
            return {
                'tracks': [], 
                'source': 'error', 
                'error': f'Lỗi quy trình xử lý âm thanh AI: {e}'
            }


    # =========================
    # 4. RECOMMEND_MOOD (Multi-Signal & Centralized Ranking)
    # =========================
    elif action == "RECOMMEND_MOOD":
        mood_query = str(params.get("mood") or "").strip()
        if not mood_query:
            return {'tracks': [], 'source': 'error', 'error': 'Bạn muốn nghe nhạc theo tâm trạng như thế nào? Hãy nói cho mình biết nhé.'}

        # LỚP 1: GỌI HÀM HỖ TRỢ ĐỂ LẤY TÍN HIỆU
        target_vibes, target_topics, target_sentiment = _mood_maps(mood_query)
        print(f"[RECOMMEND_MOOD] vibes={target_vibes}, topics={target_topics}")

        try:
            # LỚP 2: TRUY VẤN SQL (Ưu tiên Vibe để lọc thô)
            # Dùng OR để quét tất cả các Vibe liên quan (ví dụ: Sâu lắng OR Kịch tính)
            vibe_filters = ",".join([f"vibe.ilike.%{v}%" for v in target_vibes])
            
            res = supabase.table('songs').select(
                'spotify_track_id, title, artists, vibe, main_topic, final_sentiment, spotify_popularity, is_hit, genres'
            ).or_(vibe_filters).limit(int(match_count) * 4).execute()
            
            rows = getattr(res, 'data', None) or []
            source_label = 'multi-signal:mood-ranked'

            # LỚP 3: VECTOR FALLBACK (Nếu SQL không tìm thấy bài nào khớp Vibe)
            if not rows:
                print(f"[Fallback] Kích hoạt Vector Search cho Mood: {mood_query}")
                query_embedding = _safe_embed(embed_fn, mood_query)
                if query_embedding:
                    res_vec = supabase.rpc("match_vpop_tracks", {
                        "query_embedding": query_embedding,
                        "match_threshold": float(match_threshold or 0.4),
                        "match_count": int(match_count) * 4
                    }).execute()
                    rows = getattr(res_vec, 'data', None) or []
                    source_label = 'vector-fallback:mood-ranked'

            if not rows:
                return {'tracks': [], 'source': 'empty', 'error': f"Chưa tìm thấy nhạc phù hợp với tâm trạng '{mood_query}'."}

            # =======================================================
            # GLOBAL RANKER
            # =======================================================
            # Ném tất cả tín hiệu vào hàm Ranker vạn năng
            ranked_tracks = rank_and_normalize_tracks(
                raw_rows=rows,
                limit=int(match_count),
                boosts={
                    'vibe': target_vibes,
                    'topics': target_topics,
                    'sentiment': target_sentiment
                }
            )

            return {
                'tracks': ranked_tracks,
                'source': source_label,
                'error': None
            }

        except Exception as e:
            return {'tracks': [], 'source': 'error', 'error': str(e)}


    # =========================
    # 5. RECOMMEND_ARTIST (4 Lớp: Fuzzy -> Exact -> Wildcard -> Vector -> Ranker)
    # =========================
    elif action == "RECOMMEND_ARTIST":
        artist = str(params.get("artist", "") or "").strip()
        if not artist:
            return {'tracks': [], 'source': 'fallback-missing-param', 'error': 'Bạn muốn nghe nhạc của nghệ sĩ nào?'}

        print(f"[RECOMMEND_ARTIST] Đang tìm nghệ sĩ: '{artist}'")

        try:
            # --- LỚP 1: SỬA LỖI CHÍNH TẢ (Fuzzy Matching) ---
            try:
                current_artist_list = _load_artist_list_from_supabase(supabase)
            except Exception:
                current_artist_list = []

            best_artist = _find_best_artist(artist, current_artist_list)
            artist_query = best_artist if best_artist else artist
            
            if best_artist and best_artist.lower() != artist.lower():
                print(f" -> AI dò lỗi chính tả: Đã nắn '{artist}' thành '{best_artist}'")

            rows = []
            source_label = ""

            # --- LỚP 2: TÌM KIẾM THEO TÊN (SQL ILIKE) ---
            q1 = supabase.table('songs').select(
                'title, artists, vibe, main_topic, spotify_track_id, spotify_popularity, is_hit, genres, final_sentiment'
            ).ilike('artists', f'%{artist_query}%').limit(int(match_count) * 4)
            
            res1 = q1.execute()
            rows = getattr(res1, 'data', None) or []
            
            if rows:
                source_label = 'text-search:artist-fuzzy'

            # --- LỚP 3: LỚP BẢO VỆ DÍNH CHỮ (Wildcard SQL) ---
            if not rows:
                artist_wildcard = artist.replace(" ", "%")
                if artist_wildcard != artist:
                    print(f"[Fallback] Nới lỏng khoảng trắng: '{artist_wildcard}'")
                    # FIX BUG: Đã sửa artist_query thành artist_wildcard ở đây
                    q2 = supabase.table('songs').select(
                        'title, artists, vibe, main_topic, spotify_track_id, spotify_popularity, is_hit, genres , final_sentiment'
                    ).ilike('artists', f'%{artist_wildcard}%').limit(int(match_count) * 4)
                
                    res2 = q2.execute()
                    rows = getattr(res2, 'data', None) or []
                    
                    if rows:
                        source_label = 'text-search:artist-wildcard'

            # --- LỚP 4: VECTOR SEARCH (Cứu cánh cho Tiếng Việt KHÔNG DẤU) ---
            if not rows:
                print(f"[Fallback] Kích hoạt Vector Search đoán nghệ sĩ: '{artist}'...")
                try:
                    query_embedding = _safe_embed(embed_fn, artist)
                    if query_embedding:
                        thr = float(match_threshold) if match_threshold is not None else 0.5
                        res4 = supabase.rpc(
                            "match_vpop_tracks",
                            {"query_embedding": query_embedding, "match_threshold": thr, "match_count": int(match_count) * 4},
                        ).execute()
                        
                        rows = getattr(res4, 'data', None) or []
                        if rows:
                            source_label = 'vector-fallback:artist'
                except Exception as e:
                    print(f"[Lỗi Vector Fallback Nghệ sĩ] {e}")

            # KẾT THÚC CHUỖI TÌM KIẾM: Nếu vẫn trống
            if not rows:
                return {'tracks': [], 'source': 'search-artist-empty', 'error': f"Tiếc quá, hiện tại mình chưa có bài nào của '{artist}'."}

            # =======================================================
            # GLOBAL RANKER
            # =======================================================
            # Dù tìm được bằng Lớp 2, 3 hay 4, tất cả đều phải qua Ranker để:
            # 1. Chấm điểm Pop/Hit.
            # 2. Phạt điểm nếu bị spam trùng nghệ sĩ.
            # 3. Thưởng điểm nếu trùng khớp artist_query.
            ranked_tracks = rank_and_normalize_tracks(
                raw_rows=rows,
                limit=int(match_count),
                boosts={'artist': artist_query}
            )

            return {'tracks': ranked_tracks, 'source': source_label, 'error': None}

        except Exception as ex:
            return {'tracks': [], 'source': 'search-artist-error', 'error': f"Lỗi hệ thống: {ex}"}


    # =========================
    # 6. RECOMMEND_GENRE (Lọc Thể loại + Centralized Ranking)
    # =========================
    elif action == "RECOMMEND_GENRE":
        genre_query = str(params.get("genre") or "").strip()
        if not genre_query:
            return {'tracks': [], 'source': 'fallback-missing-param', 'error': 'Bạn muốn nghe thể loại nhạc gì?'}

        # BƯỚC 1: GỌI HÀM HỖ TRỢ ĐỂ ÁNH XẠ THỂ LOẠI (Chuẩn Supabase)
        mapped_genre = get_genre_target(genre_query)
        print(f"[RECOMMEND_GENRE] query='{genre_query}' -> mapped='{mapped_genre}'")

        try:
            # BƯỚC 2: TRUY VẤN SQL (Lấy dư data x4 để Ranker có không gian chấm điểm)
            # Lưu ý: Truy vấn thẳng vào cột genres
            res = supabase.table('songs').select(
                'spotify_track_id, title, artists, vibe, main_topic, final_sentiment, spotify_popularity, is_hit, genres'
            ).ilike('genres', f'%{mapped_genre}%').limit(max(20, int(match_count) * 4)).execute()
            
            rows = getattr(res, 'data', None) or []
            source_label = 'text-search:genre-ranked'

            # BƯỚC 3: VECTOR FALLBACK (Nếu SQL bó tay với các thể loại lạ hoặc viết sai chính tả nặng)
            if not rows:
                print(f"[Fallback] Kích hoạt Vector Search cho Thể loại: {genre_query}")
                query_embedding = _safe_embed(embed_fn, genre_query)
                if query_embedding:
                    res_vec = supabase.rpc("match_vpop_tracks", {
                        "query_embedding": query_embedding,
                        "match_threshold": float(match_threshold or 0.4),
                        "match_count": int(match_count) * 4
                    }).execute()
                    
                    rows = getattr(res_vec, 'data', None) or []
                    source_label = 'vector-fallback:genre-ranked'

            if not rows:
                return {'tracks': [], 'source': 'empty', 'error': f"Chưa tìm thấy nhạc thuộc thể loại '{genre_query}'."}

            # =======================================================
            # GLOBAL RANKER
            # =======================================================
            ranked_tracks = rank_and_normalize_tracks(
                raw_rows=rows,
                limit=int(match_count),
                boosts={
                    'genre': mapped_genre  # Truyền thể loại vào để Ranker thưởng điểm
                }
            )

            return {
                'tracks': ranked_tracks,
                'source': source_label,
                'error': None
            }

        except Exception as e:
            return {'tracks': [], 'source': 'error', 'error': str(e)}
    

    # =========================
    # 7. ANALYZE_READY (Phân tích chuyên sâu: Librosa + NLP + SHAP)
    # =========================
    elif action == "ANALYZE_READY":
        audio_path = params.get("audio_path")
        if not audio_path or not os.path.exists(audio_path):
            return {'error': "Không tìm thấy file âm thanh để phân tích.", 'source': 'analyze-error'}

        lyric_text = params.get('lyric_text')
        lyric_path = params.get('lyric_path')
        if not lyric_text and not lyric_path:
            return {
                'error': "Bạn cần cung cấp lời nhạc (.txt) để phân tích (hiện không dùng Speech-to-Text).",
                'source': 'analyze-error',
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
            }
        except Exception as e:
            return {'error': f"Lỗi phân tích chuyên sâu: {str(e)}", 'source': 'analyze-error'}

    # =========================
    # 8. MISSING_FILE
    # =========================
    elif action == "MISSING_FILE":
        return {
            'tracks': [], 
            'source': 'fallback-missing-file', 
            'error': 'Bạn quên đính kèm file âm thanh (MP3/WAV) ở thanh bên trái (Sidebar) rồi kìa! Hãy tải file lên để mình phân tích nhé.'
        }
    # =========================
    # 9. CLARIFY
    # =========================
    elif action == "CLARIFY":
        return {
            'tracks': [], 
            'source': 'action-clarify', 
            'error': 'Xin lỗi, mình chưa hiểu rõ ý bạn lắm. Bạn có thể nói rõ hơn là bạn muốn tìm bài hát, nghe nhạc theo tâm trạng, hay muốn mình phân tích file âm thanh không?'
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
            'error': 'TRIGGER_LLM_ANSWER' 
        }


    # =========================
    # 11. OUT_OF_SCOPE
    # =========================
    elif action == "OUT_OF_SCOPE":
        return {
            'tracks': [], 
            'source': 'action-out-of-scope', 
            'error': 'Xin lỗi, mình là Trợ lý AI chuyên về âm nhạc V-Pop. Mình chỉ có thể giúp bạn tìm nhạc, phân tích bài hát hoặc trả lời các kiến thức về âm nhạc thôi nhé!'
        }

    # =========================
    # 12. ADVANCED_SEARCH (Tìm kiếm Kết hợp - Đã tối ưu bằng Helper & Ranker)
    # =========================
    elif action == "ADVANCED_SEARCH":
        mood = str(params.get("mood", "")).strip()
        genre = str(params.get("genre", "")).strip()
        artist = str(params.get("artist", "")).strip()

        if not any([mood, genre, artist]):
            return {'tracks': [], 'source': 'error', 'error': 'Thiếu thông số tìm kiếm nâng cao.'}

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

            # 3. Ráp mảnh ghép Tâm trạng (Linh hoạt - OR)
            if mood:
                target_vibes, target_topics, target_sentiment = _mood_maps(mood)
                boosts.update({'vibe': target_vibes, 'topics': target_topics, 'sentiment': target_sentiment})
                
                # Ép DB lọc Vibe
                if target_vibes:
                    vibe_filters = ",".join([f"vibe.ilike.%{v}%" for v in target_vibes])
                    q = q.or_(vibe_filters)

            # 4. CHẠY TRUY VẤN SQL (Lấy dư ra x4 để Ranker làm việc)
            res = q.limit(int(match_count) * 4).execute()
            rows = getattr(res, 'data', None) or []
            source_label = 'advanced-search-sql'

            # 5. VECTOR FALLBACK (Cứu cánh nếu lọc quá gắt không ra bài nào)
            if not rows:
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
                return {'tracks': [], 'source': 'search-advanced-empty', 'error': "Khẩu vị của bạn mặn quá, hệ thống lọc mãi không ra bài nào khớp hết các điều kiện này!"}

            # =======================================================
            # GLOBAL RANKER
            # =======================================================
            ranked = rank_and_normalize_tracks(
                raw_rows=rows,
                limit=int(match_count),
                boosts=boosts
            )

            return {'tracks': ranked, 'source': source_label, 'error': None}

        except Exception as e:
            return {'tracks': [], 'source': 'search-advanced-error', 'error': f"Lỗi truy vấn đa luồng: {e}"}
        

    # =========================
    # 13. RECOMMEND_SEED (DNA âm thanh - Chỉ lấy Nhạc lý)
    # =========================
    elif action == "RECOMMEND_SEED":
        seed_name = str(params.get("seed_name", "")).strip()
        try:
            # 1. Tìm ID bài mẫu (SỬA: Lấy thêm vibe và genres)
            song_f = supabase.table('songs').select('spotify_track_id, title, artists, vibe, genres').ilike('title', f'%{seed_name}%').limit(1).execute()
            if not song_f.data: 
                return {'tracks': [], 'source': 'rec-seed-empty', 'error': f"Không thấy bài '{seed_name}'"}
            
            t_id = song_f.data[0]['spotify_track_id']
            # Lưu lại vibe và genre của bài gốc để thưởng điểm
            seed_title = song_f.data[0].get('title')
            seed_artist = song_f.data[0].get('artists')
            seed_vibe = song_f.data[0].get('vibe')
            seed_genre = song_f.data[0].get('genres')

            # Optional: seed audio features for explanation (tempo/energy).
            seed_tempo = None
            seed_energy = None
            try:
                seed_feat = supabase.table('track_features').select('tempo_bpm, rms_energy').eq('spotify_track_id', t_id).limit(1).execute()
                if seed_feat.data:
                    seed_tempo = seed_feat.data[0].get('tempo_bpm')
                    seed_energy = seed_feat.data[0].get('rms_energy')
            except Exception:
                pass

            # 2. Lấy Vector bài mẫu
            v_res = supabase.table('track_features').select('audio_feature_embedding').eq('spotify_track_id', t_id).execute()
            if not v_res.data:
                return {'tracks': [], 'source': 'rec-seed-no-vec', 'error': "Bài hát chưa có dữ liệu âm thanh."}

            # 3. Tìm tương đồng bằng RPC (Lấy dư x4)
            sim = supabase.rpc('match_audio_signature', {
                'query_embedding': v_res.data[0]['audio_feature_embedding'], 
                'match_threshold': 0.3, 
                'match_count': int(match_count) * 4 + 1
            }).execute()
            
            # Lọc list ID (bỏ bài gốc)
            ids = [t['spotify_track_id'] for t in (sim.data or []) if t['spotify_track_id'] != t_id][:int(match_count)*4]

            if not ids:
                return {'tracks': [], 'source': 'rec-seed-none', 'error': "Không tìm thấy bài tương đương."}

            # 4. Lấy Data CẦN THIẾT CHO RANKER
            res = supabase.table('track_features').select(
                "spotify_track_id, tempo_bpm, rms_energy, songs(title, artists, vibe, main_topic, spotify_popularity, is_hit, genres, final_sentiment)"
            ).in_('spotify_track_id', ids).execute()
            
            rows = []
            for r in (res.data or []):
                s = r.get('songs', {}) or {}
                # Gắn thêm điểm similarity trả về từ RPC vào rows để Ranker sử dụng làm Base Score
                sim_score = next((item.get('similarity') for item in (sim.data or []) if item.get('spotify_track_id') == r['spotify_track_id']), 0)
                rows.append({
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
                    'final_sentiment': s.get('final_sentiment'),
                    'similarity': sim_score # TRỌNG YẾU: Cần truyền similarity vào cho Ranker
                })
            
            # =======================================================
            # GLOBAL RANKER
            # =======================================================
            # Thưởng điểm nếu bài gợi ý có cùng vibe hoặc genre với bài gốc
            boosts = {}
            if seed_vibe: boosts['seed_vibe'] = seed_vibe
            if seed_genre: boosts['seed_genre'] = seed_genre

            ranked = rank_and_normalize_tracks(
                raw_rows=rows, 
                limit=int(match_count),
                boosts=boosts
            )
            
            return {
                'tracks': ranked, 
                'source': 'recommendation:seed-ranked', 
                'error': None,
                # Extra context for UI/LLM narration.
                'seed_meta': {
                    'seed_name': seed_name,
                    'seed_title': seed_title,
                    'seed_artist': seed_artist,
                    'seed_vibe': seed_vibe,
                    'seed_genres': seed_genre,
                    'seed_tempo_bpm': seed_tempo,
                    'seed_rms_energy': seed_energy,
                },
            }
        except Exception as e:
            return {'tracks': [], 'source': 'error', 'error': str(e)}
        

    # =========================
    # 14. RECOMMEND_ATTRIBUTES (Chỉ lọc theo Nhạc lý: Tempo & Energy)
    # =========================
    elif action == "RECOMMEND_ATTRIBUTES":
        # Mặc định dải rộng nhất
        min_t, max_t = 0, 250
        min_e, max_e = 0.0, 1.0
        
        raw_text = _normalize_text(str(params.get("attributes", "") + params.get("song_title", "")))

        # --- Ánh xạ Tempo (Nhịp điệu) ---
        if any(k in raw_text for k in ['cham', 'slow']):
            min_t, max_t = 60, 90
        elif any(k in raw_text for k in ['vua', 'binh thuong']):
            min_t, max_t = 90, 120
        elif any(k in raw_text for k in ['nhanh', 'fast']):
            min_t, max_t = 120, 160
        elif any(k in raw_text for k in ['rat nhanh', 'don dap', 'speed up']):
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

            return {'tracks': ranked, 'source': 'recommendation:attributes-ranked', 'error': None}
        except Exception as e:
            return {'tracks': [], 'source': 'error', 'error': str(e)}

    # =========================
    # 16. RECOMMEND_POPULARITY (Gợi ý Playlist Top Hit - BXH)
    # =========================
    # =========================
    # 16. RECOMMEND_POPULARITY (Gợi ý Playlist Top Hit - BXH)
    # =========================
    elif action == "RECOMMEND_POPULARITY":
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
                'error': None
            }
        except Exception as e:
            return {'tracks': [], 'source': 'error', 'error': str(e)}