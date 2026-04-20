from __future__ import annotations

import os
import tempfile
import time
import wave
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from chatbot.action_handler import handle_action
from chatbot.intent import ALLOWED_ACTIONS, parse_intent_llm
from chatbot.supabase import LyricsEmbeddingProvider, SupabaseClientFactory


def _ms(seconds: float) -> float:
    return float(seconds) * 1000.0


def _perf_counter() -> float:
    # Local indirection so tests can monkeypatch if needed.
    return time.perf_counter()


def _safe_method_label(intent_dict: dict | None) -> str:
    if not isinstance(intent_dict, dict):
        return "Rule"
    m = str(intent_dict.get("method") or "").strip().lower()
    if m in {"llm", "gemini"}:
        return "LLM"
    if m in {"rule", "heuristic"}:
        return "Rule"
    # Best-effort fallback.
    thought = str(intent_dict.get("thought") or "").lower()
    if "gemini" in thought:
        return "LLM"
    return "Rule"


@dataclass(frozen=True)
class BenchmarkCase:
    """One benchmark case mapped to a target action.

    We always measure intent latency by calling `parse_intent_llm(Input_Text)`.
    For backend latency, we call `handle_action(Action_Name, params, ...)` so you
    can guarantee coverage for all actions, even if intent occasionally misroutes.
    """

    action_name: str
    input_text: str
    params: dict[str, Any] | None = None
    has_file: bool = False


def _write_temp_wav(*, seconds: float = 1.0, sr: int = 22050, freq_hz: float = 440.0) -> str:
    """Create a tiny WAV file for SEARCH_AUDIO / ANALYZE_READY benchmarking.

    Uses stdlib only.
    """

    import math

    n_frames = max(1, int(seconds * sr))
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp_path = tmp.name
    tmp.close()

    with wave.open(tmp_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sr)

        frames = bytearray()
        for i in range(n_frames):
            t = i / sr
            sample = int(0.25 * 32767.0 * math.sin(2.0 * math.pi * freq_hz * t))
            frames.extend(int(sample).to_bytes(2, byteorder="little", signed=True))
        wf.writeframes(bytes(frames))

    return tmp_path


def _write_temp_lyrics(*, text: str = "em ơi hôm nay em như thế nào") -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".txt")
    tmp_path = tmp.name
    try:
        tmp.write(text.encode("utf-8", errors="ignore"))
    finally:
        tmp.close()
    return tmp_path


def default_cases_16_actions() -> list[BenchmarkCase]:
    """Provide a minimal set of representative inputs for the 16 actions.

    Notes:
    - SEARCH_AUDIO and ANALYZE_READY require an audio file. This function creates
      a tiny temporary WAV file for realistic backend measurement.
    - ANALYZE_READY requires lyrics (.txt). This function creates a temporary
      lyrics file.
    - These prompts are *representative*; if you want deterministic routing, you
      can set GEMINI_* env vars off so intent falls back to Rule Engine.
    """

    audio_path = _write_temp_wav(seconds=1.0)
    lyric_path = _write_temp_lyrics()

    return [
        BenchmarkCase("SEARCH_NAME", 'Mở bài "Nơi này có anh"', {"song_title": "Nơi này có anh"}),
        BenchmarkCase("SEARCH_LYRIC", "Bài gì có câu mang tiền về cho mẹ", {"lyric_snippet": "mang tiền về cho mẹ"}),
        BenchmarkCase("SEARCH_AUDIO", "Bài này là bài gì?", {"audio_path": audio_path}, has_file=True),
        BenchmarkCase("RECOMMEND_MOOD", "Gợi ý nhạc buồn sâu lắng", {"mood": "buồn"}),
        BenchmarkCase("RECOMMEND_ARTIST", "Mở nhạc của Đen Vâu", {"artist": "Đen"}),
        BenchmarkCase("RECOMMEND_GENRE", "Tìm nhạc indie", {"genre": "Indie"}),
        BenchmarkCase("ADVANCED_SEARCH", "Tìm nhạc rap buồn của Đen Vâu", {"genre": "Rap/Hip-hop", "mood": "buồn", "artist": "Đen"}),
        BenchmarkCase("RECOMMEND_SEED", "Có bài nào giống See Tình không?", {"seed_name": "See Tình"}),
        BenchmarkCase("RECOMMEND_ATTRIBUTES", "Nhạc nhịp nhanh để tập gym", {"attributes": "tempo nhanh, năng lượng mạnh"}),
        BenchmarkCase("RECOMMEND_POPULARITY", "Top 5 bài hot nhất", {}),
        BenchmarkCase(
            "ANALYZE_READY",
            "Phân tích bài này",
            {"audio_path": audio_path, "lyric_path": lyric_path},
            has_file=True,
        ),
        BenchmarkCase("MISSING_FILE", "Bài này là bài gì?", {}),
        BenchmarkCase("CLARIFY", "hay", {}),
        BenchmarkCase("MUSIC_KNOWLEDGE", "Tempo là gì?", {}),
        BenchmarkCase("OUT_OF_SCOPE", "Thời tiết hôm nay thế nào?", {}),
        BenchmarkCase("GREETING", "Chào bot", {}),
    ]


def benchmark_cases(
    cases: Iterable[BenchmarkCase],
    *,
    runs_per_case: int = 10,
    supabase_client: Any | None = None,
    embed_fn: Callable[[str], Any] | None = None,
    artist_list: list[str] | None = None,
    match_count: int = 5,
) -> list[dict[str, Any]]:
    """Run latency benchmark for each case.

    Returns a list of dicts. Required keys:
    - Action_Name
    - Input_Text
    - Method (Rule/LLM)
    - Latency_ms (total)

    Also includes useful breakdown keys:
    - Intent_ms
    - Backend_ms
    - Parsed_Action
    """

    if runs_per_case <= 0:
        raise ValueError("runs_per_case must be >= 1")

    if supabase_client is None:
        supabase_client = SupabaseClientFactory().get_client()

    if embed_fn is None:
        provider = LyricsEmbeddingProvider()
        embed_fn = provider.encode

    out: list[dict[str, Any]] = []

    for case in cases:
        action_name = str(case.action_name or "").strip().upper()
        if action_name not in ALLOWED_ACTIONS:
            raise ValueError(f"Unknown action_name: {action_name}")

        params = dict(case.params or {})
        input_text = str(case.input_text or "")

        for i in range(int(runs_per_case)):
            intent_t0 = _perf_counter()
            intent = parse_intent_llm(input_text, has_file=bool(case.has_file))
            intent_ms = _ms(_perf_counter() - intent_t0)

            method = _safe_method_label(intent if isinstance(intent, dict) else None)
            parsed_action = str((intent or {}).get("action") or "") if isinstance(intent, dict) else ""

            backend_ms: float | None = None
            backend_error: str | None = None
            backend_t0 = _perf_counter()
            try:
                _ = handle_action(
                    action_name,
                    params,
                    supabase_client,
                    embed_fn=embed_fn,
                    has_file=bool(case.has_file),
                    artist_list=artist_list,
                    match_count=int(match_count),
                )
            except Exception as ex:
                backend_error = f"{type(ex).__name__}: {ex}"
            finally:
                backend_ms = _ms(_perf_counter() - backend_t0)

            total_ms = float(intent_ms) + float(backend_ms or 0.0)

            out.append(
                {
                    "Action_Name": action_name,
                    "Input_Text": input_text,
                    "Method": method,
                    "Latency_ms": total_ms,
                    "Intent_ms": float(intent_ms),
                    "Backend_ms": float(backend_ms or 0.0),
                    "Parsed_Action": parsed_action,
                    "Run": i + 1,
                    "Backend_Error": backend_error,
                }
            )

    return out


def average_latency_by_action(
    records: Iterable[dict[str, Any]],
    *,
    field: str = "Latency_ms",
) -> list[dict[str, Any]]:
    """Compute average latency per action.

    Designed for the output of `benchmark_cases(..., runs_per_case=10)`.
    Returns list-of-dict so it can be exported as JSON/CSV easily.
    """

    by_action: dict[str, list[float]] = {}

    for r in records:
        if not isinstance(r, dict):
            continue
        action = str(r.get("Action_Name") or "").strip().upper()
        if not action:
            continue
        try:
            v = float(r.get(field))
        except Exception:
            continue
        by_action.setdefault(action, []).append(v)

    out: list[dict[str, Any]] = []
    for action, vals in sorted(by_action.items()):
        if not vals:
            continue
        avg = sum(vals) / float(len(vals))
        out.append({"Action_Name": action, "Average_Latency_ms": avg, "N": len(vals), "Field": field})

    return out


def cleanup_temp_files(cases: Iterable[BenchmarkCase]) -> None:
    """Remove temp audio/lyrics files created by `default_cases_16_actions()`."""

    for c in cases:
        if not isinstance(c, BenchmarkCase):
            continue
        p = dict(c.params or {})
        for k in ("audio_path", "lyric_path"):
            path = str(p.get(k) or "").strip()
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except Exception:
                    pass
