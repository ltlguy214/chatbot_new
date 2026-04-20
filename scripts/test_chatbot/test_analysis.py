from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description="End-to-end ANALYZE_READY test (P0..P4 + SHAP only P0/P1)")
    parser.add_argument("--audio", required=True, help="Path to a local WAV/MP3 file")
    parser.add_argument("--lyric", required=True, help="Path to a lyric .txt file (required; no Speech-to-Text)")
    parser.add_argument("--no-download", action="store_true", help="Do not download models from Supabase Storage")
    parser.add_argument("--local-models", action="store_true", help="Use local DA/models artifacts (do not force Supabase Storage)")
    parser.add_argument("--no-shap", action="store_true", help="Skip SHAP computation")
    parser.add_argument("--skip-p1", action="store_true", help="Skip P1 entirely (no download/load/predict); SHAP runs only for P0")
    parser.add_argument("--shap-nsamples", type=int, default=None, help="Kernel SHAP nsamples (default env SHAP_NSAMPLES_KERNEL or 80)")
    parser.add_argument("--export-features", default=None, help="Optional path to write extracted feature table CSV")
    parser.add_argument("--json", action="store_true", help="Print full bundle as JSON")
    args = parser.parse_args()

    root = _repo_root()
    sys.path.insert(0, str(root))

    try:
        from chatbot.env import load_env

        load_env()
    except Exception:
        pass

    from chatbot.analyze_ready_action import run_analyze_ready

    bundle = run_analyze_ready(
        audio_path=str(args.audio),
        lyric_path=str(args.lyric),
        allow_download=False if bool(args.local_models) else (not bool(args.no_download)),
        compute_shap=not bool(args.no_shap),
        shap_nsamples_kernel=args.shap_nsamples,
        export_features_path=args.export_features,
        force_storage=False if bool(args.local_models) else True,
        skip_p1=bool(args.skip_p1),
    )

    summary = {
        "p0": bundle.get("p0"),
        "p1": bundle.get("p1"),
        "p2": bundle.get("p2"),
        "p3": bundle.get("p3"),
        "p4": bundle.get("p4"),
        "feature_completeness": bundle.get("feature_completeness"),
        "load_errors": bundle.get("load_errors"),
        "model_sources": bundle.get("model_sources"),
    }

    print("=== ANALYZE_READY SUMMARY ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))

    shap_payload = bundle.get("shap_values") or {}
    tasks = shap_payload.get('tasks') if isinstance(shap_payload, dict) else None
    p0_contrib = (tasks.get('p0') or {}).get('contributions') if isinstance(tasks, dict) else None
    if isinstance(p0_contrib, list) and p0_contrib:
        print("\n=== SHAP P0 (Hit) ===")
        for row in p0_contrib[:10]:
            feat = row.get("feature")
            pct = float(row.get("contribution_percent", 0.0))
            sval = float(row.get("shap_value", 0.0))
            print(f"- {feat}: shap={sval:+.4f}, pct={pct:.2f}%")

    if args.json:
        # Full bundle can be large (input_df + raw features), so print last.
        print("\n=== FULL BUNDLE JSON ===")
        print(json.dumps(bundle, ensure_ascii=False, indent=2, default=str))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
