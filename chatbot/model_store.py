from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelStorageConfig:
    bucket: str
    prefix: str
    cache_dir: Path

    @staticmethod
    def from_env() -> "ModelStorageConfig":
        bucket = (
            str(os.getenv("SUPABASE_MODELS_BUCKET") or "").strip()
            or str(os.getenv("SUPABASE_STORAGE_BUCKET") or "").strip()
        )
        # Project default (user requested): keep it overridable via env.
        bucket = bucket or "ml_models"
        prefix = str(os.getenv("SUPABASE_MODELS_PREFIX") or os.getenv("SUPABASE_MODELS_FOLDER") or "").strip()

        # Guardrails: prefix is expected to be a *folder*, not a file list.
        # Users sometimes paste a comma-separated filename list here, which would
        # break remote paths like "a.pkl,b.pkl/best_model_p0.pkl".
        prefix_lower = prefix.lower()
        looks_like_file_list = (
            ("," in prefix)
            or prefix_lower.endswith(".pkl")
            or ".pkl.part" in prefix_lower
            or "best_model_" in prefix_lower
        )
        if looks_like_file_list:
            prefix = ""

        cache_dir_raw = str(os.getenv("MODEL_CACHE_DIR") or "").strip()
        if cache_dir_raw:
            cache_dir = Path(cache_dir_raw)
        else:
            cache_dir = Path(__file__).resolve().parents[1] / ".model_cache"

        cache_dir.mkdir(parents=True, exist_ok=True)
        return ModelStorageConfig(bucket=bucket, prefix=prefix, cache_dir=cache_dir)

    def remote_path(self, filename: str) -> str:
        filename = str(filename or "").lstrip("/")
        p = str(self.prefix or "").strip().strip("/")
        return f"{p}/{filename}" if p else filename


def _import_get_supabase_client():
    try:
        from chatbot.supabase import get_supabase_client  # type: ignore

        return get_supabase_client
    except Exception:
        from supabase import get_supabase_client  # type: ignore

        return get_supabase_client


def _write_bytes_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def download_from_storage(
    *,
    filename: str,
    cfg: ModelStorageConfig | None = None,
    errors: list[str] | None = None,
) -> Path | None:
    cfg = cfg or ModelStorageConfig.from_env()
    filename = str(filename or "").strip()
    if not filename:
        if errors is not None:
            errors.append("Empty filename")
        return None

    dest = cfg.cache_dir / filename
    if dest.exists() and dest.stat().st_size > 0:
        return dest

    if not cfg.bucket:
        if errors is not None:
            errors.append("Missing SUPABASE_MODELS_BUCKET / SUPABASE_STORAGE_BUCKET")
        return None

    get_client = _import_get_supabase_client()
    client = get_client()
    if client is None:
        if errors is not None:
            errors.append("Supabase client unavailable (SUPABASE_URL/SUPABASE_KEY missing or invalid)")
        return None

    remote = cfg.remote_path(filename)
    try:
        blob = client.storage.from_(cfg.bucket).download(remote)
        # supabase-py may return bytes or a file-like object depending on version.
        if isinstance(blob, (bytes, bytearray)):
            data = bytes(blob)
        else:
            data = blob  # type: ignore[assignment]
            try:
                data = bytes(data)
            except Exception:
                data = getattr(blob, "data", None)
                if isinstance(data, (bytes, bytearray)):
                    data = bytes(data)
                else:
                    # last resort
                    data = bytes(str(blob), "utf-8")

        if not data:
            if errors is not None:
                errors.append(f"Downloaded empty payload: bucket={cfg.bucket}, remote={remote}")
            return None

        _write_bytes_atomic(dest, data)
        return dest
    except Exception as ex:
        if errors is not None:
            errors.append(f"Storage download failed: bucket={cfg.bucket}, remote={remote} ({type(ex).__name__}: {ex})")
        return None


def stitch_parts(
    *,
    output_name: str,
    part_names: list[str],
    cfg: ModelStorageConfig | None = None,
    errors: list[str] | None = None,
) -> Path | None:
    cfg = cfg or ModelStorageConfig.from_env()
    output_name = str(output_name or "").strip()
    if not output_name:
        return None

    out_path = cfg.cache_dir / output_name
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path

    part_paths: list[Path] = []
    for name in part_names:
        name = str(name or "").strip()
        if not name:
            continue
        # Prefer cache file if already there.
        p = cfg.cache_dir / name
        if not (p.exists() and p.stat().st_size > 0):
            dl = download_from_storage(filename=name, cfg=cfg, errors=errors)
            if dl is None:
                if errors is not None:
                    errors.append(f"Missing part: {name}")
                return None
            p = dl
        part_paths.append(p)

    if len(part_paths) != len(part_names):
        if errors is not None:
            errors.append("Part count mismatch")
        return None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp.open("wb") as w:
        for p in part_paths:
            w.write(p.read_bytes())
    tmp.replace(out_path)
    return out_path


def _first_existing(paths: list[str]) -> Path | None:
    for p in paths:
        try:
            if p and os.path.exists(p) and os.path.getsize(p) > 0:
                return Path(p)
        except Exception:
            continue
    return None


def resolve_model_paths(*, allow_download: bool = True, report: dict[str, str] | None = None) -> dict[str, Path]:
    """Resolve local model paths, downloading from Supabase Storage if needed.

    Returns a mapping with keys: P0, P1, P2, P3, P4 when available.
    """

    cfg = ModelStorageConfig.from_env()

    # Preferred filenames in storage (per user spec).
    files = {
        "P4": "best_model_p4_compressed.pkl",
        "P1": "best_model_p1_compressed.pkl",
        "P2": "best_model_p2.pkl",
        "P3": "best_model_p3.pkl",
        "P0": "best_model_p0.pkl",
    }

    result: dict[str, Path] = {}

    prefer_storage = str(os.getenv("MODELS_PREFER_STORAGE") or "").strip().lower() in {"1", "true", "yes", "on"}

    def _prefer_order(storage_first: list[str], local_fallback: list[str]) -> list[str]:
        return (storage_first + local_fallback) if prefer_storage else (local_fallback + storage_first)

    def _note(key: str, message: str) -> None:
        if report is not None:
            report[key] = str(message)

    # P0
    p0 = _first_existing(
        _prefer_order(
            [str(cfg.cache_dir / files["P0"])],
            [
                "DA/models/best_model_p0.pkl",
                "pkl_file/best_model_p0.pkl",
                "best_model_p0.pkl",
            ],
        )
    )
    if p0 is not None:
        _note("P0", f"local:{p0}")
    if p0 is None and allow_download:
        errs: list[str] = []
        p0 = download_from_storage(filename=files["P0"], cfg=cfg, errors=errs)
        if p0 is not None:
            _note("P0", f"storage:{p0}")
        elif errs:
            _note("P0", "missing:" + " | ".join(errs[:2]))
    if p0 is not None:
        result["P0"] = p0
    elif "P0" not in result:
        _note("P0", report.get("P0") if report is not None and "P0" in report else "missing")

    # P1 (stitched from parts if needed)
    p1 = _first_existing(
        _prefer_order(
            [str(cfg.cache_dir / files["P1"])],
            [
                "DA/models/best_model_p1_compressed.pkl",
                "DA/models/best_model_p1.pkl",
                "pkl_file/best_model_p1.pkl",
                "best_model_p1.pkl",
            ],
        )
    )

    if p1 is not None:
        _note("P1", f"local:{p1}")

    if p1 is None and allow_download:
        errs: list[str] = []
        stitched = stitch_parts(
            output_name=files["P1"],
            part_names=[
                "best_model_p1_compressed.pkl.part1",
                "best_model_p1_compressed.pkl.part2",
                "best_model_p1_compressed.pkl.part3",
            ],
            cfg=cfg,
            errors=errs,
        )
        p1 = stitched

        if p1 is not None:
            _note("P1", f"storage-stitched:{p1}")
        elif errs:
            _note("P1", "missing:" + " | ".join(errs[:2]))

    if p1 is not None:
        result["P1"] = p1
    elif "P1" not in result:
        _note("P1", report.get("P1") if report is not None and "P1" in report else "missing")

    # P2
    p2 = _first_existing(
        _prefer_order(
            [str(cfg.cache_dir / files["P2"])],
            [
                "DA/models/best_model_p2.pkl",
                "pkl_file/best_clustering_p2_10_algorithms.pkl",
                "best_clustering_p2_10_algorithms.pkl",
            ],
        )
    )
    if p2 is not None:
        _note("P2", f"local:{p2}")
    if p2 is None and allow_download:
        errs: list[str] = []
        p2 = download_from_storage(filename=files["P2"], cfg=cfg, errors=errs)
        if p2 is not None:
            _note("P2", f"storage:{p2}")
        elif errs:
            _note("P2", "missing:" + " | ".join(errs[:2]))
    if p2 is not None:
        result["P2"] = p2
    elif "P2" not in result:
        _note("P2", report.get("P2") if report is not None and "P2" in report else "missing")

    # P3
    p3 = _first_existing(
        _prefer_order(
            [str(cfg.cache_dir / files["P3"])],
            [
                "DA/models/best_model_p3.pkl",
                "pkl_file/best_model_p3.pkl",
                "best_model_p3.pkl",
            ],
        )
    )
    if p3 is not None:
        _note("P3", f"local:{p3}")
    if p3 is None and allow_download:
        errs: list[str] = []
        p3 = download_from_storage(filename=files["P3"], cfg=cfg, errors=errs)
        if p3 is not None:
            _note("P3", f"storage:{p3}")
        elif errs:
            _note("P3", "missing:" + " | ".join(errs[:2]))
    if p3 is not None:
        result["P3"] = p3
    elif "P3" not in result:
        _note("P3", report.get("P3") if report is not None and "P3" in report else "missing")

    # P4
    p4 = _first_existing(
        _prefer_order(
            [str(cfg.cache_dir / files["P4"])],
            [
                "DA/models/best_model_p4_compressed.pkl",
                "DA/models/best_model_p4.pkl",
                "pkl_file/best_model_p4.pkl",
                "pkl_file/best_model_p4_genre.pkl",
            ],
        )
    )
    if p4 is not None:
        _note("P4", f"local:{p4}")
    if p4 is None and allow_download:
        errs: list[str] = []
        p4 = download_from_storage(filename=files["P4"], cfg=cfg, errors=errs)
        if p4 is not None:
            _note("P4", f"storage:{p4}")
        elif errs:
            _note("P4", "missing:" + " | ".join(errs[:2]))
    if p4 is not None:
        result["P4"] = p4
    elif "P4" not in result:
        _note("P4", report.get("P4") if report is not None and "P4" in report else "missing")

    return result
