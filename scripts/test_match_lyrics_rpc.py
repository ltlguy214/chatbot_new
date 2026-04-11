"""Smoke test Supabase RPC `match_lyrics` using SentenceTransformer embeddings.

Usage (PowerShell):
    & .\.venv312\Scripts\python.exe .\scripts\test_match_lyrics_rpc.py "em đang ở đâu" --threshold 0.4 --count 5

Requires env vars (via .env or system env):
  SUPABASE_URL
  SUPABASE_KEY (or SUPABASE_SERVICE_ROLE_KEY)
Optional:
  SUPABASE_LYRICS_RPC (default: match_lyrics)
  SUPABASE_LYRICS_EMBEDDING_MODEL (default: paraphrase-multilingual-MiniLM-L12-v2)
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def _load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:
        pass


def _is_missing_or_placeholder(value: str) -> bool:
    text = str(value or "").strip()
    if not text:
        return True
    lowered = text.lower()
    return lowered in {"none", "null", "changeme", "your_key_here", "your_url_here", "your_supabase_key"} or lowered.startswith(
        "your_"
    )


def main(argv: list[str]) -> int:
    _load_dotenv_if_available()

    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="?", default="", help="User input to embed and search")
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--count", type=int, default=5)
    args = parser.parse_args(argv)

    query_text = (args.query or "").strip()
    if not query_text:
        print("Missing query text. Example: python scripts/test_match_lyrics_rpc.py \"em đang ở đâu\"")
        return 2

    supabase_url = os.getenv("SUPABASE_URL", "")
    supabase_key = os.getenv("SUPABASE_KEY", "") or os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
    rpc_name = os.getenv("SUPABASE_LYRICS_RPC", "match_lyrics")
    model_name = os.getenv("SUPABASE_LYRICS_EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")

    if _is_missing_or_placeholder(supabase_url) or _is_missing_or_placeholder(supabase_key):
        print("Missing SUPABASE_URL/SUPABASE_KEY (or SUPABASE_SERVICE_ROLE_KEY).")
        print("Create a .env from .env.example or set env vars.")
        return 2

    from supabase import create_client
    from sentence_transformers import SentenceTransformer

    print(f"Model: {model_name}")
    print(f"RPC: {rpc_name}")

    model = SentenceTransformer(model_name)
    query_vec = model.encode(query_text).tolist()

    payload = {
        "query_embedding": query_vec,
        "match_threshold": float(args.threshold),
        "match_count": int(args.count),
    }

    client = create_client(supabase_url, supabase_key)
    resp = client.rpc(rpc_name, payload).execute()

    data = getattr(resp, "data", None)
    print("Response.data type:", type(data).__name__)
    print("Response.data preview:")
    print(json.dumps(data, ensure_ascii=False, indent=2)[:4000])

    if isinstance(data, list):
        print(f"Returned rows: {len(data)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
