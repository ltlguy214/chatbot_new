from __future__ import annotations

import os
import sys
import json
import csv


# Ensure repo-root is importable when running as `python scripts/...`.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def main() -> int:
    # Load env (repo has a robust loader)
    try:
        from chatbot.env import load_env

        load_env()
    except Exception as ex:
        print("ENV_LOAD_FAIL", type(ex).__name__, str(ex)[:200])
        return 2

    # Create Supabase client
    try:
        from chatbot.supabase import get_supabase_client

        client = get_supabase_client()
    except Exception as ex:
        print("CLIENT_INIT_FAIL", type(ex).__name__, str(ex)[:200])
        return 3

    if client is None:
        print("CLIENT_NONE: check SUPABASE_URL/SUPABASE_KEY")
        return 4

    # Build embedding
    query_text = "nhạc buồn thất tình ballad"
    try:
        from chatbot.supabase import encode_lyrics_embedding_debug

        def embed_fn(text: str):
            vec, err = encode_lyrics_embedding_debug(text)
            try:
                embed_fn.last_error = err  # type: ignore[attr-defined]
            except Exception:
                pass
            return vec

        vec, err = encode_lyrics_embedding_debug(query_text)
        if err or not vec:
            print("EMBED_FAIL", err)
            return 5
        print("EMBED_OK", "dim=", len(vec))
    except Exception as ex:
        print("EMBED_EX", type(ex).__name__, str(ex)[:200])
        return 6

    status = 0

    # Call RPC
    try:
        rpc = "match_vpop_tracks"
        payload = {
            "query_embedding": vec,
            "match_threshold": 0.5,
            "match_count": 5,
        }
        resp = client.rpc(rpc, payload).execute()
        rows = getattr(resp, "data", None) or []
        print("RPC_OK", "rows=", len(rows))
        if rows:
            # Print a compact view (ASCII-safe)
            first = rows[0]
            if isinstance(first, dict):
                print("FIRST_KEYS", list(first.keys()))
                print("FIRST_ROW", json.dumps(first, ensure_ascii=True)[:500])
        status = 0
    except Exception as ex:
        print("RPC_FAIL", type(ex).__name__, str(ex)[:400])
        return 7

    # (Optional) Smoke test handle_action routing
    try:
        from chatbot.action_handler import handle_action

        # Probe 1 known song row for SEARCH_NAME / GENRE seed.
        probe_title = ''
        probe_artist = ''
        probe_genre = ''
        try:
            probe_resp = client.table('songs').select('title, artists, spotify_genres').limit(1).execute()
            probe_rows = getattr(probe_resp, 'data', None) or []
            if probe_rows and isinstance(probe_rows[0], dict):
                probe_title = str(probe_rows[0].get('title') or '').strip()
                probe_artist = str(probe_rows[0].get('artists') or '').strip()
                probe_genres = str(probe_rows[0].get('spotify_genres') or '').strip()
                if probe_genres:
                    # Try to pick the first genre token.
                    probe_genre = probe_genres.split(',')[0].strip()
        except Exception:
            pass

        # Mood
        mood_result = handle_action(
            'RECOMMEND_MOOD',
            {'mood': 'buồn'},
            client,
            embed_fn=embed_fn,
            match_count=5,
        )
        if isinstance(mood_result, dict):
            print('MOOD_OK', 'source=', mood_result.get('source'), 'tracks=', len(mood_result.get('tracks') or []))
        else:
            print('MOOD_RET', type(mood_result).__name__)

        # SEARCH_LYRIC (via handle_action -> RPC)
        lyric_result = handle_action(
            'SEARCH_LYRIC',
            {'lyric_snippet': 'tình yêu'},
            client,
            embed_fn=embed_fn,
            match_count=5,
        )
        if isinstance(lyric_result, dict):
            print('LYRIC_OK', 'source=', lyric_result.get('source'), 'tracks=', len(lyric_result.get('tracks') or []))
        else:
            print('LYRIC_RET', type(lyric_result).__name__)

        # SEARCH_NAME (seed from probe row if available)
        name_params = {'song_title': probe_title, 'artist': ''}
        if probe_artist:
            # Use a short slice of artist string (some rows contain multi-artist).
            name_params['artist'] = probe_artist.split(',')[0].strip()
        search_name_result = handle_action(
            'SEARCH_NAME',
            name_params,
            client,
            match_count=5,
        )
        if isinstance(search_name_result, dict):
            print('NAME_OK', 'source=', search_name_result.get('source'), 'tracks=', len(search_name_result.get('tracks') or []))
        else:
            print('NAME_RET', type(search_name_result).__name__)

        # RECOMMEND_GENRE (best-effort genre)
        genre = probe_genre or 'pop'
        genre_result = handle_action(
            'RECOMMEND_GENRE',
            {'genre': genre},
            client,
            match_count=5,
        )
        if isinstance(genre_result, dict):
            print('GENRE_OK', 'genre=', genre, 'source=', genre_result.get('source'), 'tracks=', len(genre_result.get('tracks') or []))
        else:
            print('GENRE_RET', type(genre_result).__name__)

        # Artist list (prefer Supabase artists table; fallback to CSV)
        artist_list: list[str] = []
        try:
            page_size = 1000
            max_rows = 20000
            for col in ['artist_name', 'name', 'artist']:
                artist_list = []
                start = 0
                try:
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
                                    artist_list.append(name)
                        if len(rows) < page_size:
                            break
                        start += page_size
                    artist_list = list(dict.fromkeys(artist_list))
                    if artist_list:
                        print('ARTISTS_TABLE_OK', 'col=', col, 'count=', len(artist_list))
                        break
                except Exception:
                    continue
        except Exception as ex:
            print('ARTISTS_TABLE_FAIL', type(ex).__name__, str(ex)[:200])

        if not artist_list:
            csv_path = os.path.join(_ROOT, 'data', 'artists_vietnam.csv')
            if os.path.exists(csv_path):
                with open(csv_path, 'r', encoding='utf-8', newline='') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        name = str((row or {}).get('artist_name') or '').strip()
                        if name:
                            artist_list.append(name)
                artist_list = list(dict.fromkeys(artist_list))
                print('ARTISTS_CSV_OK', 'count=', len(artist_list))

        artist_result = handle_action(
            'RECOMMEND_ARTIST',
            {'artist': 'Noo Phước Thịnh'},
            client,
            embed_fn=embed_fn,
            artist_list=artist_list,
            match_count=5,
        )
        if isinstance(artist_result, dict):
            print('ARTIST_OK', 'source=', artist_result.get('source'), 'tracks=', len(artist_result.get('tracks') or []))
        else:
            print('ARTIST_RET', type(artist_result).__name__)

        artist_result_fallback = handle_action(
            'RECOMMEND_ARTIST',
            {'artist': 'Noo Phước Thịnh'},
            client,
            embed_fn=embed_fn,
            artist_list=None,
            match_count=5,
        )
        if isinstance(artist_result_fallback, dict):
            print('ARTIST_FALLBACK_OK', 'source=', artist_result_fallback.get('source'), 'tracks=', len(artist_result_fallback.get('tracks') or []))
        else:
            print('ARTIST_FALLBACK_RET', type(artist_result_fallback).__name__)

        # Missing-file behaviors
        audio_no_file = handle_action(
            'SEARCH_AUDIO',
            {},
            client,
            embed_fn=embed_fn,
            has_file=False,
            match_count=5,
        )
        if isinstance(audio_no_file, dict):
            print('AUDIO_NOFILE_OK', 'source=', audio_no_file.get('source'), 'error=', str(audio_no_file.get('error') or '')[:120])
        else:
            print('AUDIO_NOFILE_RET', type(audio_no_file).__name__)

        analyze_no_file = handle_action(
            'ANALYZE_READY',
            {},
            client,
            embed_fn=embed_fn,
            has_file=False,
            match_count=5,
        )
        if isinstance(analyze_no_file, dict):
            print('ANALYZE_NOFILE_OK', 'source=', analyze_no_file.get('source'), 'error=', str(analyze_no_file.get('error') or '')[:120])
        else:
            print('ANALYZE_NOFILE_RET', type(analyze_no_file).__name__)

        clarify = handle_action('CLARIFY', {}, client, embed_fn=embed_fn)
        if isinstance(clarify, dict):
            print('CLARIFY_OK', 'source=', clarify.get('source'), 'error=', str(clarify.get('error') or '')[:120])

        oos = handle_action('OUT_OF_SCOPE', {}, client, embed_fn=embed_fn)
        if isinstance(oos, dict):
            print('OOS_OK', 'source=', oos.get('source'), 'error=', str(oos.get('error') or '')[:120])

    except Exception as ex:
        print('HANDLE_ACTION_FAIL', type(ex).__name__, str(ex)[:300])

    return status


if __name__ == "__main__":
    raise SystemExit(main())
