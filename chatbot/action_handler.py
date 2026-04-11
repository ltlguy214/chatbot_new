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


def _is_truthy_env(name: str, default: str = '0') -> bool:
    return str(os.getenv(name, default) or '').strip().lower() in {'1', 'true', 'yes', 'on'}


def _normalize_text(text: str) -> str:
    text = (text or '').lower().strip()
    text = unicodedata.normalize('NFD', text)
    text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
    text = re.sub(r'[^a-z0-9 ]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _safe_text(value: Any) -> str:
    return str(value or '').lower().strip()


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
        }
        if 'similarity' in r:
            item['similarity'] = r.get('similarity')
        if 'score' in r:
            item['score'] = r.get('score')
        out.append(item)
    return out


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


def _mood_maps(mood: str) -> tuple[str | None, str | None]:
    """Return (sentiment, vibe) from mood text."""
    m_raw = str(mood or '').strip()
    m = _normalize_text(m_raw)

    # Defaults
    sentiment: str | None = None
    vibe_prefix: str | None = None

    # sentiment inference
    if any(k in m for k in ['buon', 'sad', 'that tinh', 'thatinh', 'suy', 'tuyet vong', 'khoc']):
        sentiment = 'negative'
    elif any(k in m for k in ['vui', 'happy', 'yeu doi', 'yeudoi', 'tich cuc', 'tichcuc']):
        sentiment = 'positive'
    elif any(k in m for k in ['chill', 'relax', 'thu gian', 'thugian', 'lofi', 'lo fi', 'healing', 'chua lanh', 'chualanh']):
        sentiment = 'neutral'

    # vibe inference (dataset values look like: "Bình yên / Chữa lành", ...)
    if any(k in m for k in ['chill', 'relax', 'thu gian', 'thugian', 'healing', 'chua lanh', 'chualanh']):
        vibe_prefix = 'Bình yên'
    if any(k in m for k in ['quay', 'quay tung', 'quaytung', 'quayhetminh', 'party', 'soi dong', 'soidong', 'bung no', 'bungno']):
        vibe_prefix = 'Bùng nổ'
        sentiment = sentiment or 'positive'
    if any(k in m for k in ['tieu cuc', 'tieucuc', 'da diet', 'dadiet', 'kich tinh', 'kichtinh', 'da diet', 'dai det']):
        vibe_prefix = 'Kịch tính'
    if any(k in m for k in ['buon', 'sad', 'sau lang', 'saulang', 'tam trang', 'tamtrang']):
        vibe_prefix = vibe_prefix or 'Sâu lắng'
    if any(k in m for k in ['vui', 'happy', 'tuoi moi', 'tuoimoi', 'yeu doi', 'yeudoi']):
        vibe_prefix = vibe_prefix or 'Tươi mới'

    return sentiment, vibe_prefix


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


def _build_mood_query_text(mood: str, *, sentiment: str | None = None, vibe_prefix: str | None = None) -> str:
    base = str(mood or '').strip()
    parts: list[str] = []
    if base:
        parts.append(base)
    if sentiment == 'negative':
        parts.append('buồn thất tình tâm trạng')
    elif sentiment == 'positive':
        parts.append('vui sôi động')
    elif sentiment == 'neutral':
        parts.append('chill thư giãn')

    if vibe_prefix:
        parts.append(vibe_prefix)
    text = ' '.join([p for p in parts if p]).strip()
    return text or 'nhạc v-pop'


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
    # 1. SEARCH_NAME
    # =========================
    if action == "SEARCH_NAME":
        title = str(params.get("song_title", "") or "").strip()
        artist = str(params.get("artist", "") or "").strip()
        q = supabase.table("songs").select("title, artists, spotify_track_id")
        if title:
            q = q.ilike("title", f"%{title}%")
        if artist:
            q = q.ilike("artists", f"%{artist}%")
        res = q.limit(max(1, int(match_count))).execute()
        rows = getattr(res, 'data', None)
        return {
            'tracks': _normalize_track_rows(rows),
            'source': 'live-supabase-table:songs',
            'error': None,
        }

    # =========================
    # 2. SEARCH_LYRIC (RAG)
    # =========================
    if action == "SEARCH_LYRIC":
        lyric = str(params.get("lyric_snippet", "") or "").strip()
        if not lyric:
            return "Bạn muốn tìm theo đoạn lời nào?"
        query_embedding = _safe_embed(embed_fn, lyric)
        if not query_embedding:
            detail = _embed_error(embed_fn)
            suffix = f" ({detail})" if detail else ""
            return {
                'tracks': [],
                'source': 'fallback-missing-embedding',
                'error': 'Chưa có embedding function hoặc không tạo được embedding' + suffix,
            }
        thr = float(match_threshold) if match_threshold is not None else 0.6
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
            'tracks': _normalize_track_rows(rows),
            'source': 'live-supabase-rpc:match_vpop_tracks',
            'error': None,
        }

    # =========================
    # 3. SEARCH_AUDIO
    # =========================
    if action == "SEARCH_AUDIO":
        if not has_file:
            return {'tracks': [], 'source': 'fallback-missing-file', 'error': 'Bạn chưa tải file âm thanh lên!'}
        return {'tracks': [], 'source': 'fallback-not-implemented', 'error': 'TODO: SEARCH_AUDIO chưa được nối pipeline'}

    # =========================
    # 4. RECOMMEND_MOOD (vector)
    # =========================
    if action == "RECOMMEND_MOOD":
        mood = str(params.get("mood", "") or "").strip()
        if not mood:
            return {'tracks': [], 'source': 'fallback-missing-param', 'error': 'Bạn muốn nghe nhạc tâm trạng gì?'}

        sentiment, vibe_prefix = _mood_maps(mood)

        # 4a) Prefer structured mood recommendation from track_features when possible.
        try:
            if not sentiment and not vibe_prefix:
                raise RuntimeError('mood-unmapped')
            tf_q = supabase.table('track_features').select('spotify_track_id')
            if sentiment:
                tf_q = tf_q.eq('final_sentiment', sentiment)
            if vibe_prefix:
                tf_q = tf_q.ilike('vibe', f"{vibe_prefix}%")
            tf_res = tf_q.limit(max(1, int(match_count) * 10)).execute()
            tf_rows = getattr(tf_res, 'data', None)
            tf_ids = [str((r or {}).get('spotify_track_id') or '').strip() for r in (tf_rows or []) if isinstance(r, dict)]
            tf_ids = [tid for tid in tf_ids if tid]

            if tf_ids:
                try:
                    songs_q = (
                        supabase.table('songs')
                        .select('title, artists, spotify_track_id, spotify_popularity')
                        .in_('spotify_track_id', tf_ids)
                    )
                    try:
                        songs_q = songs_q.order('spotify_popularity', desc=True)
                    except Exception:
                        pass
                    songs_res = songs_q.limit(max(1, int(match_count))).execute()
                    song_rows = getattr(songs_res, 'data', None)
                except Exception:
                    songs_res = (
                        supabase.table('songs')
                        .select('title, artists, spotify_track_id')
                        .in_('spotify_track_id', tf_ids)
                        .limit(max(1, int(match_count)))
                        .execute()
                    )
                    song_rows = getattr(songs_res, 'data', None)
                tracks = _normalize_track_rows(song_rows)
                if tracks:
                    return {
                        'tracks': tracks,
                        'source': 'live-supabase-table:track_features->songs',
                        'error': None,
                    }
        except Exception as e:
            if _is_truthy_env('CHATBOT_DEBUG_ACTION_HANDLER', '0'):
                return {
                    'tracks': [],
                    'source': 'fallback-mood-track_features-error',
                    'error': f"Lỗi truy vấn mood theo track_features: {e}",
                }

        # 4b) Fallback: vector search with a more descriptive mood query.
        query_text = _build_mood_query_text(mood, sentiment=sentiment, vibe_prefix=vibe_prefix)
        query_embedding = _safe_embed(embed_fn, query_text)
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
            'match_vpop_tracks',
            {
                'query_embedding': query_embedding,
                'match_threshold': thr,
                'match_count': int(match_count),
            },
        ).execute()
        rows = getattr(res, 'data', None)
        return {
            'tracks': _normalize_track_rows(rows),
            'source': 'live-supabase-rpc:match_vpop_tracks',
            'error': None,
        }

    # =========================
    # 5. RECOMMEND_ARTIST
    # =========================
    if action == "RECOMMEND_ARTIST":
        artist = str(params.get("artist", "") or "").strip()
        if not artist:
            return {'tracks': [], 'source': 'fallback-missing-param', 'error': 'Bạn muốn nghe giống nghệ sĩ nào?'}

        # If UI didn't provide an artist_list, try pulling it from Supabase.
        if not artist_list:
            try:
                artist_list = _load_artist_list_from_supabase(supabase)
            except Exception:
                artist_list = None

        best_artist = _find_best_artist(artist, artist_list)
        artist_query = best_artist or artist

        try:
            q = (
                supabase.table('songs')
                .select('title, artists, spotify_track_id, spotify_popularity')
                .ilike('artists', f"%{artist_query}%")
            )
            try:
                q = q.order('spotify_popularity', desc=True)
            except Exception:
                pass
            res = q.limit(max(1, int(match_count))).execute()
            rows = getattr(res, 'data', None)
        except Exception:
            res = (
                supabase.table('songs')
                .select('title, artists, spotify_track_id')
                .ilike('artists', f"%{artist_query}%")
                .limit(max(1, int(match_count)))
                .execute()
            )
            rows = getattr(res, 'data', None)
        return {
            'tracks': _normalize_track_rows(rows),
            'source': 'live-supabase-table:songs:fuzzy-artist' if best_artist else 'live-supabase-table:songs',
            'error': None,
        }

    # =========================
    # 6. RECOMMEND_GENRE
    # =========================
    if action == "RECOMMEND_GENRE":
        genre = str(params.get("genre", "") or "").strip()
        if not genre:
            return {'tracks': [], 'source': 'fallback-missing-param', 'error': 'Bạn muốn nghe thể loại gì?'}
        res = (
            supabase.table("songs")
            .select("title, artists, spotify_track_id, spotify_genres")
            .ilike("spotify_genres", f"%{genre}%")
            .limit(max(1, int(match_count)))
            .execute()
        )
        rows = getattr(res, 'data', None)
        return {
            'tracks': _normalize_track_rows(rows),
            'source': 'live-supabase-table:songs',
            'error': None,
        }

    # =========================
    # 7. ANALYZE_READY
    # =========================
    if action == "ANALYZE_READY":
        if not has_file:
            return {'tracks': [], 'source': 'fallback-missing-file', 'error': 'Bạn chưa tải file âm thanh lên!'}
        return {'tracks': [], 'source': 'fallback-not-implemented', 'error': 'TODO: chạy model .pkl'}

    # =========================
    # 8. MISSING_FILE
    # =========================
    if action == "MISSING_FILE":
        return {'tracks': [], 'source': 'fallback-missing-file', 'error': 'Bạn chưa tải file âm thanh lên!'}

    # =========================
    # 9. CLARIFY
    # =========================
    if action == "CLARIFY":
        return {'tracks': [], 'source': 'fallback-clarify', 'error': 'Bạn muốn tìm theo mood, nghệ sĩ hay tên bài?'}

    # =========================
    # 10. MUSIC_KNOWLEDGE
    # =========================
    if action == "MUSIC_KNOWLEDGE":
        return {'tracks': [], 'source': 'fallback-not-implemented', 'error': 'TODO: gọi LLM giải thích'}

    # =========================
    # 11. OUT_OF_SCOPE
    # =========================
    if action == "OUT_OF_SCOPE":
        return {'tracks': [], 'source': 'fallback-out-of-scope', 'error': 'Xin lỗi, mình chỉ hỗ trợ nhạc V-Pop'}

    return {'tracks': [], 'source': 'fallback-unknown-action', 'error': 'Không hiểu yêu cầu'}
