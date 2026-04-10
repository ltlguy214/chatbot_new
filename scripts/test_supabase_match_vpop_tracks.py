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

    except Exception as ex:
        print('HANDLE_ACTION_FAIL', type(ex).__name__, str(ex)[:300])

    return status


if __name__ == "__main__":
    raise SystemExit(main())
