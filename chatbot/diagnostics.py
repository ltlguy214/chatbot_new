from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
import uuid
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


# Ensure repo root is importable even when running `python chatbot/diagnostics.py`
# (where sys.path[0] becomes the chatbot/ directory).
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


@dataclass
class CheckResult:
    name: str
    status: str  # PASS | FAIL | WARN | SKIP
    details: str = ""
    error: str | None = None
    meta: dict[str, Any] | None = None


def _mask(text: str, *, keep_start: int = 6, keep_end: int = 4) -> str:
    s = str(text or "")
    if len(s) <= keep_start + keep_end + 3:
        return "***"
    return f"{s[:keep_start]}…{s[-keep_end:]}"


def _safe_str(obj: Any) -> str:
    try:
        return str(obj)
    except Exception:
        return repr(obj)


def _import_supabase_module():
    try:
        from chatbot import supabase as supa  # type: ignore

        return supa
    except Exception:
        # Allow running from within the chatbot folder as CWD.
        import supabase as supa  # type: ignore

        return supa


def _load_env_best_effort() -> None:
    """Load .env like the Streamlit app does (best-effort).

    This ensures diagnostics sees SUPABASE_* / SPOTIFY_* / GEMINI_* env vars
    when they are stored in a workspace .env file.
    """

    try:
        from chatbot.env import load_env  # type: ignore

        load_env()
        return
    except Exception:
        pass

    try:
        from env import load_env  # type: ignore

        load_env()
        return
    except Exception:
        return


def _parse_gemini_api_keys() -> list[str]:
    import re

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

    for i in range(1, 21):
        v = str(os.getenv(f"GEMINI_API_KEY_{i}") or "").strip()
        if v and v not in keys:
            keys.append(v)

    return keys


def _looks_like_quota_error(err: Exception) -> bool:
    msg = str(err).lower()
    return (
        "resource_exhausted" in msg
        or "quota" in msg
        or "429" in msg
        or "rate limit" in msg
        or "rate-limit" in msg
    )


def _import_spotify_module():
    try:
        from chatbot import spotify as sp  # type: ignore

        return sp
    except Exception:
        try:
            import spotify as sp  # type: ignore

            return sp
        except Exception:
            return None


def check_supabase_env() -> CheckResult:
    url = str(os.getenv("SUPABASE_URL") or "").strip()
    key = str(os.getenv("SUPABASE_KEY") or os.getenv("SUPABASE_SERVICE_ROLE_KEY") or "").strip()
    disabled = str(os.getenv("SUPABASE_DISABLED") or "").strip().lower() in {"1", "true", "yes", "on"}

    if disabled:
        return CheckResult(
            name="Supabase env",
            status="SKIP",
            details="SUPABASE_DISABLED is enabled",
        )

    if not url or not key:
        return CheckResult(
            name="Supabase env",
            status="FAIL",
            details="Missing SUPABASE_URL and/or SUPABASE_KEY (or SUPABASE_SERVICE_ROLE_KEY)",
            meta={"SUPABASE_URL": bool(url), "SUPABASE_KEY": bool(key)},
        )

    return CheckResult(
        name="Supabase env",
        status="PASS",
        details=f"SUPABASE_URL={_mask(url)}; key={_mask(key)}",
    )


def check_supabase_client() -> CheckResult:
    try:
        supa = _import_supabase_module()
        client = supa.get_supabase_client()
        if client is None:
            return CheckResult(
                name="Supabase client",
                status="FAIL",
                details="get_supabase_client() returned None (misconfigured env or create_client failed)",
            )
        return CheckResult(name="Supabase client", status="PASS", details="Client created")
    except Exception as ex:
        return CheckResult(
            name="Supabase client",
            status="FAIL",
            error=_safe_str(ex),
        )


def check_supabase_rpc_env() -> CheckResult:
    lyrics_rpc = str(os.getenv("SUPABASE_LYRICS_RPC") or "").strip() or "match_lyrics"
    vector_rpc = str(os.getenv("SUPABASE_VECTOR_RPC") or "").strip() or "match_vpop_tracks"
    return CheckResult(
        name="Supabase RPC env",
        status="PASS",
        details=f"lyrics_rpc={lyrics_rpc}; vector_rpc={vector_rpc}",
    )


def check_chat_history_probe() -> CheckResult:
    """Probe the chat_history table with a small SELECT.

    This is stricter than fetch_recent_chat_history (which intentionally fails safe).
    """

    table = str(os.getenv("SUPABASE_CHAT_HISTORY_TABLE") or "chat_history").strip() or "chat_history"

    try:
        supa = _import_supabase_module()
        client = supa.get_supabase_client()
        if client is None:
            return CheckResult(
                name="chat_history probe",
                status="FAIL",
                details="No Supabase client; cannot query table",
            )

        t0 = time.time()
        resp = client.table(table).select("created_at, role, module").limit(1).execute()
        ms = int((time.time() - t0) * 1000)
        data = getattr(resp, "data", None)
        row_count = len(data) if isinstance(data, list) else None

        return CheckResult(
            name="chat_history probe",
            status="PASS",
            details=f"SELECT ok ({ms}ms)",
            meta={"table": table, "rows_returned": row_count},
        )
    except Exception as ex:
        return CheckResult(
            name="chat_history probe",
            status="FAIL",
            details="SELECT failed (table missing, RLS, wrong key, or network)",
            error=_safe_str(ex),
            meta={"table": table},
        )


def check_chat_history_read(*, session_id: str, module: str | None, limit: int = 5) -> CheckResult:
    try:
        supa = _import_supabase_module()

        rows = supa.fetch_recent_chat_history(session_id=session_id, module=module, limit=limit)
        if not isinstance(rows, list):
            return CheckResult(
                name="chat_history read",
                status="FAIL",
                details="fetch_recent_chat_history returned non-list",
                meta={"type": str(type(rows))},
            )

        # Do not print full content (privacy). Only counts + roles.
        roles: list[str] = []
        for r in rows:
            if isinstance(r, dict):
                roles.append(str(r.get("role") or ""))

        return CheckResult(
            name="chat_history read",
            status="PASS",
            details=f"Fetched {len(rows)} rows",
            meta={"session_id": session_id, "module": module, "roles": roles[-limit:]},
        )
    except Exception as ex:
        return CheckResult(
            name="chat_history read",
            status="FAIL",
            error=_safe_str(ex),
            meta={"session_id": session_id, "module": module},
        )


def check_chat_history_write_roundtrip(*, session_id: str, module: str, role: str = "assistant") -> CheckResult:
    """Write one diagnostic row then read it back.

    This verifies INSERT permissions + read path. It adds ONE row to the table.
    """

    marker = f"[diagnostics] roundtrip {uuid.uuid4().hex}"

    try:
        supa = _import_supabase_module()
        ok = supa.append_chat_history(session_id=session_id, role=role, content=marker, module=module)
        if not ok:
            return CheckResult(
                name="chat_history write→read",
                status="FAIL",
                details="append_chat_history returned False (no client / insert blocked)",
                meta={"session_id": session_id, "module": module},
            )

        # Read back using strict client query to avoid fail-safe masking.
        client = supa.get_supabase_client()
        table = str(os.getenv("SUPABASE_CHAT_HISTORY_TABLE") or "chat_history").strip() or "chat_history"
        if client is None:
            return CheckResult(
                name="chat_history write→read",
                status="WARN",
                details="Insert returned True but client is None afterwards; cannot verify",
                meta={"session_id": session_id, "module": module},
            )

        resp = (
            client.table(table)
            .select("content, role, module, created_at")
            .eq("module", module)
            .order("created_at", desc=True)
            .limit(10)
            .execute()
        )
        data = getattr(resp, "data", None) or []
        found = False
        if isinstance(data, list):
            for row in data:
                if isinstance(row, dict) and str(row.get("content") or "") == marker:
                    found = True
                    break

        if not found:
            return CheckResult(
                name="chat_history write→read",
                status="WARN",
                details="Inserted but did not find marker in last 10 rows (RLS filter? eventual consistency?)",
                meta={"session_id": session_id, "module": module, "marker": _mask(marker, keep_start=18, keep_end=6)},
            )

        return CheckResult(
            name="chat_history write→read",
            status="PASS",
            details="Insert + read-back ok",
            meta={"session_id": session_id, "module": module},
        )
    except Exception as ex:
        return CheckResult(
            name="chat_history write→read",
            status="FAIL",
            error=_safe_str(ex),
            meta={"session_id": session_id, "module": module},
        )


def check_spotify_env() -> CheckResult:
    cid = str(os.getenv("SPOTIFY_CLIENT_ID") or "").strip()
    secret = str(os.getenv("SPOTIFY_CLIENT_SECRET") or "").strip()

    if not cid or not secret:
        return CheckResult(
            name="Spotify env",
            status="WARN",
            details="Missing SPOTIFY_CLIENT_ID and/or SPOTIFY_CLIENT_SECRET",
            meta={"SPOTIFY_CLIENT_ID": bool(cid), "SPOTIFY_CLIENT_SECRET": bool(secret)},
        )

    return CheckResult(
        name="Spotify env",
        status="PASS",
        details=f"SPOTIFY_CLIENT_ID={_mask(cid)}; secret={_mask(secret)}",
    )


def check_spotify_token(*, network: bool) -> CheckResult:
    if not network:
        return CheckResult(name="Spotify token", status="SKIP", details="Network tests disabled")

    cid = str(os.getenv("SPOTIFY_CLIENT_ID") or "").strip()
    secret = str(os.getenv("SPOTIFY_CLIENT_SECRET") or "").strip()
    if not cid or not secret:
        return CheckResult(
            name="Spotify token",
            status="FAIL",
            details="Missing SPOTIFY_CLIENT_ID and/or SPOTIFY_CLIENT_SECRET",
        )

    try:
        t0 = time.time()
        basic = base64.b64encode(f"{cid}:{secret}".encode("utf-8")).decode("ascii")
        data = urllib.parse.urlencode({"grant_type": "client_credentials"}).encode("utf-8")
        req = urllib.request.Request(
            url="https://accounts.spotify.com/api/token",
            data=data,
            headers={
                "Authorization": f"Basic {basic}",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        token = str(payload.get("access_token") or "")
        ms = int((time.time() - t0) * 1000)
        if not token:
            return CheckResult(name="Spotify token", status="FAIL", details="No access_token in response")
        return CheckResult(name="Spotify token", status="PASS", details=f"Token ok ({ms}ms)")
    except Exception as ex:
        return CheckResult(name="Spotify token", status="FAIL", error=_safe_str(ex))


def check_gemini_env() -> CheckResult:
    keys = _parse_gemini_api_keys()
    if not keys:
        return CheckResult(
            name="Gemini env",
            status="WARN",
            details="Missing GEMINI_API_KEY / GEMINI_API_KEYS",
        )

    # Mask only the first key (avoid leaking list length patterns too much).
    return CheckResult(
        name="Gemini env",
        status="PASS",
        details=f"keys={len(keys)}; first={_mask(keys[0])}",
    )


def check_gemini_model_env() -> CheckResult:
    app_model = str(os.getenv("GEMINI_MODEL") or "").strip() or "models/gemini-2.5-flash"
    intent_model = str(os.getenv("GEMINI_INTENT_MODEL") or "").strip() or "gemini-2.5-flash"

    def _is_gemini2_flash(name: str) -> bool:
        s = name.lower().strip()
        s = s.replace("models/", "")
        return s.startswith("gemini-2") and "flash" in s

    ok_app = _is_gemini2_flash(app_model)
    ok_intent = _is_gemini2_flash(intent_model)

    if ok_app and ok_intent:
        return CheckResult(
            name="Gemini model env",
            status="PASS",
            details=f"GEMINI_MODEL={app_model}; GEMINI_INTENT_MODEL={intent_model}",
        )

    return CheckResult(
        name="Gemini model env",
        status="WARN",
        details=f"Gemini model is auto-fallback capable. GEMINI_MODEL={app_model}; GEMINI_INTENT_MODEL={intent_model}",
    )


def check_gemini_generate(*, network: bool) -> CheckResult:
    if not network:
        return CheckResult(name="Gemini generate", status="SKIP", details="Network tests disabled")

    keys = _parse_gemini_api_keys()
    if not keys:
        return CheckResult(name="Gemini generate", status="SKIP", details="No Gemini API keys")

    model_name = str(os.getenv("GEMINI_MODEL") or "gemini-2.5-flash").strip() or "gemini-2.5-flash"
    if model_name.startswith("models/"):
        model_name = model_name[len("models/") :]

    prompt = "Reply with exactly: OK"
    last_err: Exception | None = None
    try:
        from google import genai  # type: ignore
    except Exception as ex:
        return CheckResult(name="Gemini generate", status="FAIL", details="google-genai not installed", error=_safe_str(ex))

    t0 = time.time()
    for api_key in keys:
        try:
            client = genai.Client(api_key=api_key)
            resp = client.models.generate_content(model=model_name, contents=prompt)
            text = str(getattr(resp, "text", "") or "").strip()
            ms = int((time.time() - t0) * 1000)
            if text:
                return CheckResult(name="Gemini generate", status="PASS", details=f"Generated text ok ({ms}ms)")
        except Exception as ex:
            last_err = ex
            if _looks_like_quota_error(ex):
                continue
            continue

    return CheckResult(
        name="Gemini generate",
        status="FAIL",
        details="No response from Gemini",
        error=_safe_str(last_err) if last_err else None,
    )


def check_intent_llm(*, network: bool) -> CheckResult:
    if not network:
        return CheckResult(name="Intent (LLM)", status="SKIP", details="Network tests disabled")

    try:
        from chatbot.intent import ALLOWED_ACTIONS, parse_intent_llm  # type: ignore
    except Exception:
        try:
            from intent import ALLOWED_ACTIONS, parse_intent_llm  # type: ignore
        except Exception as ex:
            return CheckResult(name="Intent (LLM)", status="FAIL", details="Cannot import intent.py", error=_safe_str(ex))

    required_fields = {"song_title", "artist", "mood", "genre", "lyric_snippet"}
    samples = [
        ("Tìm bài Hơn cả yêu của Đức Phúc", False),
        ("Phân tích bài này giúp mình", True),
    ]

    try:
        t0 = time.time()
        outputs: list[dict[str, Any]] = []
        thoughts: list[str] = []
        for text, has_file in samples:
            out = parse_intent_llm(text, has_file=has_file)
            if not isinstance(out, dict):
                return CheckResult(name="Intent (LLM)", status="FAIL", details="parse_intent_llm returned non-dict")

            action = str(out.get("action") or "").upper().strip()
            params = out.get("params")
            thoughts.append(str(out.get("thought") or "").strip())
            if action not in set(ALLOWED_ACTIONS):
                return CheckResult(name="Intent (LLM)", status="FAIL", details=f"Invalid action: {action}")
            if not isinstance(params, dict):
                return CheckResult(name="Intent (LLM)", status="FAIL", details="params is not a dict")
            if not required_fields.issubset(set(params.keys())):
                return CheckResult(name="Intent (LLM)", status="FAIL", details="params missing required fields")

            outputs.append({"action": action, "has_file": has_file})

        ms = int((time.time() - t0) * 1000)

        # If everything falls back to OUT_OF_SCOPE with system-ish thoughts, surface it.
        actions = [o.get("action") for o in outputs]
        thought_text = " ".join([t for t in thoughts if t])
        systemish = any(k in thought_text.lower() for k in ["thiếu gemini", "lỗi hệ thống", "quota", "resource_exhausted", "429"])
        all_out = all(a == "OUT_OF_SCOPE" for a in actions) if actions else False

        if all_out and systemish:
            return CheckResult(
                name="Intent (LLM)",
                status="WARN",
                details=f"Validated schema but likely quota/key issue ({ms}ms)",
                meta={"samples": outputs, "thoughts": thoughts[:3]},
            )

        return CheckResult(name="Intent (LLM)", status="PASS", details=f"Validated {len(outputs)} samples ({ms}ms)", meta={"samples": outputs})
    except Exception as ex:
        return CheckResult(name="Intent (LLM)", status="FAIL", details="Intent LLM call failed", error=_safe_str(ex))


def check_model_files(*, load: bool) -> list[CheckResult]:
    """Check expected .pkl paths used by app_chatbot.py.

    By default this checks existence only. With --model-load, it attempts joblib.load.
    """

    candidates = {
        # App currently checks these paths first.
        "P0": ["pkl_file/best_model_p0.pkl", "best_model_p0.pkl", "DA/models/best_model_p0.pkl"],
        "P1": [
            "pkl_file/best_model_p1.pkl",
            "best_model_p1.pkl",
            "DA/models/best_model_p1.pkl",
            "DA/models/best_model_p1_compressed.pkl",
        ],
        "P2": [
            "pkl_file/best_clustering_p2_10_algorithms.pkl",
            "best_clustering_p2_10_algorithms.pkl",
            "DA/models/best_model_p2.pkl",
        ],
        "P3": ["pkl_file/best_model_p3.pkl", "best_model_p3.pkl", "DA/models/best_model_p3.pkl"],
        "P4": [
            "pkl_file/best_model_p4.pkl",
            "pkl_file/best_model_p4_genre.pkl",
            "best_model_p4.pkl",
            "DA/models/best_model_p4.pkl",
            "DA/models/best_model_p4_compressed.pkl",
        ],
    }

    results: list[CheckResult] = []
    try:
        import joblib  # local import
    except Exception as ex:
        return [CheckResult(name="Models", status="FAIL", details="joblib missing", error=_safe_str(ex))]

    for name, paths in candidates.items():
        found = next((p for p in paths if os.path.exists(p)), None)
        if not found:
            results.append(CheckResult(name=f"Model file {name}", status="WARN", details=f"Not found in {paths}"))
            continue

        if not load:
            results.append(CheckResult(name=f"Model file {name}", status="PASS", details=f"Found: {found}"))
            continue

        try:
            t0 = time.time()
            obj = joblib.load(found)
            ms = int((time.time() - t0) * 1000)
            # Avoid printing object; just type + keys.
            meta: dict[str, Any] = {"path": found, "type": str(type(obj))}
            if isinstance(obj, dict):
                meta["keys"] = sorted([str(k) for k in obj.keys()])[:30]
            results.append(CheckResult(name=f"Model load {name}", status="PASS", details=f"Loaded ({ms}ms)", meta=meta))
        except Exception as ex:
            results.append(
                CheckResult(
                    name=f"Model load {name}",
                    status="FAIL",
                    details=f"joblib.load failed for {found}",
                    error=_safe_str(ex),
                )
            )

    return results


def check_model_storage_download(*, enabled: bool) -> CheckResult:
    if not enabled:
        return CheckResult(name="Models (Storage)", status="SKIP", details="Storage download test disabled")

    try:
        from chatbot.model_store import ModelStorageConfig, download_from_storage, stitch_parts  # type: ignore
    except Exception:
        try:
            from model_store import ModelStorageConfig, download_from_storage, stitch_parts  # type: ignore
        except Exception as ex:
            return CheckResult(
                name="Models (Storage)",
                status="FAIL",
                details="Cannot import model_store.py",
                error=_safe_str(ex),
            )

    try:
        cfg = ModelStorageConfig.from_env()
        t0 = time.time()
        # Force download from Storage (do not fall back to local DA/models).
        files = {
            "P0": "best_model_p0.pkl",
            "P2": "best_model_p2.pkl",
            "P3": "best_model_p3.pkl",
            "P4": "best_model_p4_compressed.pkl",
        }

        resolved: dict[str, Path] = {}
        for key, filename in files.items():
            p = download_from_storage(filename=filename, cfg=cfg)
            if p is not None:
                resolved[key] = Path(p)

        p1 = stitch_parts(
            output_name="best_model_p1_compressed.pkl",
            part_names=[
                "best_model_p1_compressed.pkl.part1",
                "best_model_p1_compressed.pkl.part2",
                "best_model_p1_compressed.pkl.part3",
            ],
            cfg=cfg,
        )
        if p1 is not None:
            resolved["P1"] = Path(p1)

        ms = int((time.time() - t0) * 1000)

        required = ["P0", "P1", "P2", "P3", "P4"]
        missing = [k for k in required if k not in resolved]
        meta: dict[str, Any] = {
            "bucket": cfg.bucket,
            "prefix": cfg.prefix,
            "cache_dir": str(cfg.cache_dir),
            "resolved": {k: str(v) for k, v in resolved.items()},
        }

        if missing:
            return CheckResult(
                name="Models (Storage)",
                status="FAIL",
                details=f"Downloaded/resolved but missing: {missing} ({ms}ms)",
                meta=meta,
            )

        # Verify all files are in cache dir and non-empty.
        bad: list[str] = []
        for k, p in resolved.items():
            try:
                pp = Path(p)
                if not (pp.exists() and pp.stat().st_size > 0):
                    bad.append(f"{k}:missing_or_empty")
                    continue
                if cfg.cache_dir not in pp.resolve().parents and pp.resolve() != cfg.cache_dir:
                    bad.append(f"{k}:not_in_cache")
            except Exception:
                bad.append(f"{k}:error")

        if bad:
            return CheckResult(
                name="Models (Storage)",
                status="FAIL",
                details=f"Downloaded but validation failed: {bad} ({ms}ms)",
                meta=meta,
            )

        return CheckResult(
            name="Models (Storage)",
            status="PASS",
            details=f"Downloaded + stitched ok ({ms}ms)",
            meta=meta,
        )
    except Exception as ex:
        return CheckResult(
            name="Models (Storage)",
            status="FAIL",
            details="Storage download/stitch failed",
            error=_safe_str(ex),
        )


def run_diagnostics(
    *,
    session_id: str,
    module: str | None,
    write_test: bool,
    network: bool,
    model_load: bool,
    storage_models: bool,
    gemini_test: bool,
    intent_test: bool,
) -> list[CheckResult]:
    results: list[CheckResult] = []

    results.append(check_supabase_env())
    results.append(check_supabase_client())
    results.append(check_supabase_rpc_env())
    results.append(check_chat_history_probe())
    results.append(check_chat_history_read(session_id=session_id, module=module, limit=5))
    if write_test:
        results.append(check_chat_history_write_roundtrip(session_id=session_id, module="diagnostics"))

    results.append(check_spotify_env())
    results.append(check_spotify_token(network=network))

    results.append(check_gemini_env())
    results.append(check_gemini_model_env())
    if gemini_test:
        results.append(check_gemini_generate(network=network))
    if intent_test:
        results.append(check_intent_llm(network=network))

    results.append(check_model_storage_download(enabled=storage_models))
    results.extend(check_model_files(load=model_load))

    return results


def _to_markdown(results: list[CheckResult]) -> str:
    lines: list[str] = []
    lines.append("# Chatbot Diagnostics Report")
    lines.append("")
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"CWD: {os.getcwd()}")
    dotenv_used = str(os.getenv("DOTENV_PATH_USED") or "").strip()
    if dotenv_used:
        lines.append(f"DOTENV_PATH_USED: {dotenv_used}")
    lines.append("")

    counts = {"PASS": 0, "FAIL": 0, "WARN": 0, "SKIP": 0}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1

    lines.append("## Summary")
    lines.append(f"- PASS: {counts['PASS']} | FAIL: {counts['FAIL']} | WARN: {counts['WARN']} | SKIP: {counts['SKIP']}")
    lines.append("")

    lines.append("## Details")
    for r in results:
        base = f"- [{r.status}] {r.name}: {r.details}".rstrip()
        lines.append(base)
        if r.error:
            lines.append(f"  - error: {r.error}")
        if r.meta:
            try:
                meta_json = json.dumps(r.meta, ensure_ascii=False)
            except Exception:
                meta_json = _safe_str(r.meta)
            lines.append(f"  - meta: {meta_json}")

    lines.append("")
    lines.append("## Notes")
    lines.append("- `chat_history read` uses the safe wrapper and may return 0 rows even when the DB is reachable.")
    lines.append("- `chat_history probe` is the strict SELECT check (best indicator for real connectivity).")
    lines.append("- `chat_history write→read` inserts ONE row into module=diagnostics (only when enabled).")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run connectivity diagnostics for the chatbot.")
    parser.add_argument("--session-id", default=os.getenv("DIAG_SESSION_ID", ""), help="Session id to test chat_history read")
    parser.add_argument("--module", default=os.getenv("DIAG_MODULE", "home"), help="Module to filter chat_history read (use 'none' for all)")
    parser.add_argument("--write-test", action="store_true", help="Insert one diagnostics row and read it back")
    parser.add_argument("--network", action="store_true", help="Enable network tests (Spotify token)")
    parser.add_argument("--gemini-test", action="store_true", help="Call Gemini once to verify it responds (uses quota)")
    parser.add_argument("--intent-test", action="store_true", help="Run intent.py end-to-end (uses Gemini quota)")
    parser.add_argument("--model-load", action="store_true", help="Attempt joblib.load() for model files")
    parser.add_argument(
        "--storage-models",
        action="store_true",
        help="Download models from Supabase Storage (bucket ml_models by default) and stitch P1 into .model_cache",
    )
    parser.add_argument("--json", dest="as_json", action="store_true", help="Output JSON instead of Markdown")

    args = parser.parse_args(argv)

    _load_env_best_effort()

    session_id = str(args.session_id or "").strip() or f"diagnostics-{uuid.uuid4().hex}"
    module = None if str(args.module).strip().lower() in {"", "none", "null", "all"} else str(args.module).strip()

    results = run_diagnostics(
        session_id=session_id,
        module=module,
        write_test=bool(args.write_test),
        network=bool(args.network),
        model_load=bool(args.model_load),
        storage_models=bool(args.storage_models),
        gemini_test=bool(args.gemini_test),
        intent_test=bool(args.intent_test),
    )

    if args.as_json:
        payload = [asdict(r) for r in results]
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(_to_markdown(results))

    # Exit code: 0 if no FAILs.
    return 0 if all(r.status != "FAIL" for r in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
