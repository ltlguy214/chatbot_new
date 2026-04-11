from __future__ import annotations

import os
import sys


# Ensure repo-root is on sys.path so `import chatbot.*` works even when
# executing this file directly (sys.path[0] becomes the `chatbot/` folder).
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _force_utf8_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


_force_utf8_stdout()


def main() -> int:
    from chatbot.env import load_env

    load_env()

    from chatbot.supabase import get_supabase_client, encode_lyrics_embedding_debug

    client = get_supabase_client()
    if client is None:
        print("Supabase client = None (check SUPABASE_URL/SUPABASE_KEY)")
        return 2

    # 1) Simple SELECT probe
    try:
        probe = client.table("songs").select("spotify_track_id,title,artists").limit(3).execute()
        rows = getattr(probe, "data", None) or []
        print("songs.select rows=", len(rows))
        if rows:
            print("songs.first=", rows[0])
    except Exception as ex:
        print("songs.select ERROR:", type(ex).__name__, str(ex))

    # 2) Embedding + RPC test
    query_text = "nhạc buồn thất tình ballad"
    vec, err = encode_lyrics_embedding_debug(query_text)
    if err:
        print("embed ERROR:", err)
        return 3

    print("embed dim=", len(vec or []))

    try:
        resp = client.rpc(
            "match_vpop_tracks",
            {
                "query_embedding": vec,
                "match_threshold": 0.3,
                "match_count": 5,
            },
        ).execute()
        data = getattr(resp, "data", None) or []
        print("rpc rows=", len(data))
        if data:
            print("rpc.first=", data[0])
        return 0
    except Exception as ex:
        print("rpc ERROR:", type(ex).__name__, str(ex))
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
