from __future__ import annotations

from typing import Any

import os
from pathlib import Path

import numpy as np
import pandas as pd


def _apply_pickle_compat_patches() -> None:
    """Best-effort compatibility patches for older sklearn/numpy pickles.

    This is intentionally local to inference code; it does not mutate project state.
    """

    # sklearn ColumnTransformer internal helper changed across versions
    try:
        import sklearn.compose._column_transformer

        try:
            from sklearn.compose._column_transformer import _RemainderColsList  # noqa: F401
        except Exception:
            class _RemainderColsList(list):
                pass

            sklearn.compose._column_transformer._RemainderColsList = _RemainderColsList  # type: ignore[attr-defined]
    except Exception:
        pass

    # sklearn SimpleImputer internals changed across versions
    try:
        from sklearn.impute import SimpleImputer

        if not hasattr(SimpleImputer, "_fill_dtype"):
            SimpleImputer._fill_dtype = property(lambda self: getattr(self, "_fit_dtype", None))  # type: ignore[attr-defined]
    except Exception:
        pass

    # numpy random bit_generator attribute in some older pickles
    try:
        if not hasattr(np.random, "bit_generator") and hasattr(np.random, "_bit_generator"):
            np.random.bit_generator = np.random._bit_generator  # type: ignore[attr-defined]
    except Exception:
        pass


def _sentiment_ohe(final_sentiment: Any) -> dict[str, float]:
    s = str(final_sentiment or "neutral").strip().lower()
    return {
        "sentiment_negative": 1.0 if s == "negative" else 0.0,
        "sentiment_neutral": 1.0 if s == "neutral" else 0.0,
        "sentiment_positive": 1.0 if s == "positive" else 0.0,
    }


def _default_value_for(col: str) -> object:
    c = str(col)
    if c == "final_sentiment":
        return "neutral"
    if c in {"lyric", "clean_lyric"}:
        return ""
    return 0.0


def _collect_required_columns(pipeline, *, df_module=pd) -> set[str]:
    cols: set[str] = set()
    try:
        pre = getattr(pipeline, "named_steps", {}).get("preprocessor")
        transformers = getattr(pre, "transformers_", None) if pre is not None else None
        if not transformers:
            return cols

        for _name, _tr, c in transformers:
            if isinstance(c, (list, tuple, np.ndarray, df_module.Index)):
                for x in list(c):
                    if isinstance(x, str):
                        cols.add(x)
    except Exception:
        return cols

    return cols


def _ensure_pipeline_input(df: pd.DataFrame, pipeline) -> pd.DataFrame:
    req = _collect_required_columns(pipeline)
    if not req:
        return df

    out = df.copy()
    for c in req:
        if c not in out.columns:
            out[c] = _default_value_for(c)
    return out


def _extract_multilabel_positive_proba(pipe, X: pd.DataFrame) -> np.ndarray | None:
    """Return positive-class probabilities per label for MultiOutput models.

    Many sklearn MultiOutput setups return list[n_labels] of (n_samples, 2) arrays.
    """

    if not hasattr(pipe, "predict_proba"):
        return None

    try:
        proba_list = pipe.predict_proba(X)
    except Exception:
        return None

    if not isinstance(proba_list, (list, tuple)):
        return None

    cols = []
    for p in proba_list:
        arr = np.asarray(p)
        if arr.ndim == 2 and arr.shape[1] >= 2:
            cols.append(arr[:, 1])
        elif arr.ndim == 1:
            cols.append(arr)
        else:
            return None

    try:
        return np.column_stack(cols)
    except Exception:
        return None


def _load_lyrics_text(*, lyric_text: str | None, lyric_path: str | None) -> str:
    if lyric_text is not None:
        text = str(lyric_text)
    elif lyric_path:
        text = Path(str(lyric_path)).read_text(encoding="utf-8", errors="ignore")
    else:
        raise ValueError("Missing lyrics: upload a .txt file (lyric_text/lyric_path required)")

    text = text.strip()
    if not text:
        raise ValueError("Lyrics are empty after reading; please upload a valid .txt")
    return text


def _completeness_from_required(raw_features: dict[str, Any], required: set[str]) -> dict[str, Any]:
    required = {c for c in required if isinstance(c, str) and c}
    present = {c for c in required if c in raw_features and raw_features.get(c) is not None}
    missing = sorted(required - present)
    pct = 0.0 if not required else round(100.0 * (len(present) / len(required)), 2)
    return {
        "required_count": int(len(required)),
        "present_count": int(len(present)),
        "missing_count": int(len(missing)),
        "percent": float(pct),
        "missing": missing,
    }


def run_analyze_ready(
    *,
    audio_path: str,
    lyric_text: str | None = None,
    lyric_path: str | None = None,
    supabase_client: Any | None = None,
    allow_download: bool = True,
    compute_shap: bool = True,
    shap_nsamples_kernel: int | None = None,
    export_features_path: str | None = None,
    force_storage: bool | None = None,
    skip_p1: bool = False,
) -> dict[str, Any]:
    """End-to-end analysis for a local audio file.

    Output bundle is compatible with `chatbot/app_chatbot.py` dashboard renderers:
    - keys: p0..p4, raw_features, input_df, shap_values
    """

    audio_path = str(audio_path or "").strip()
    if not audio_path or not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio not found: {audio_path}")

    # Best-effort: load .env (Streamlit already does this).
    try:
        from chatbot.env import load_env

        load_env()
    except Exception:
        pass

    # New requirement: do NOT use Speech-to-Text; lyrics must come from uploaded .txt.
    lyrics = _load_lyrics_text(lyric_text=lyric_text, lyric_path=lyric_path)

    _apply_pickle_compat_patches()

    import joblib

    # Ensure custom estimators for P4 can be unpickled (best-effort).
    try:
        import DA.tasks.Genres.analysis_Genre_Classification  # noqa: F401
    except Exception:
        pass

    from chatbot.model_store import resolve_model_paths
    from chatbot.shap_runtime import build_shap_payload_p0_only
    from chatbot.librosa import extract_audio_features_full
    from chatbot.nlp import analyze_lyrics
    from chatbot.topic import extract_topic_features

    # 1) Feature extraction (librosa + NLP + topic) best-effort
    # supabase_client is intentionally not used here.
    raw_features: dict[str, Any] = {}

    try:
        audio_features = extract_audio_features_full(audio_path)
    except Exception as ex:
        audio_features = {"audio_error": f"{type(ex).__name__}: {ex}"}
    raw_features.update(audio_features)

    try:
        nlp_features = analyze_lyrics(lyrics)
    except Exception as ex:
        nlp_features = {
            "final_sentiment": "neutral",
            "lexical_diversity": 0.0,
            "lyric_total_words": 0.0,
            "nlp_error": f"{type(ex).__name__}: {ex}",
        }
    raw_features.update(nlp_features)

    try:
        probs, main_label, meta = extract_topic_features(lyrics)
        topic_features: dict[str, Any] = {}
        if isinstance(probs, dict):
            topic_features.update(probs)
        topic_features["topic_main"] = str(main_label or "")
        topic_features["topic_status"] = str((meta or {}).get("status") or "ok")
    except Exception as ex:
        topic_features = {f"topic_prob_{i}": 0.0 for i in range(16)}
        topic_features.update({"topic_main": "", "topic_status": f"topic-error:{type(ex).__name__}"})
    raw_features.update(topic_features)

    # Backward-compatible field name used elsewhere in the app
    raw_features["main_topic"] = str(raw_features.get("main_topic") or raw_features.get("topic_main") or "").strip()
    if not raw_features["main_topic"]:
        raw_features["main_topic"] = "Chưa xác định"

    raw_features.update(_sentiment_ohe(raw_features.get("final_sentiment", "neutral")))

    # Some training pipelines expect genre one-hot columns.
    # At inference time we may not know the true genre -> default to 0.0 ("unknown").
    for _g in ["genre_Ballad", "genre_Indie", "genre_Pop", "genre_Rap/Hip-hop"]:
        raw_features.setdefault(_g, 0.0)

    # Derive lyrical_density (words per second) to match training features.
    try:
        dur = float(raw_features.get("duration_sec") or 0.0)
        words = float(raw_features.get("lyric_total_words") or 0.0)
        raw_features["lyrical_density"] = round(float(words / dur), 6) if dur > 0 else 0.0
    except Exception:
        raw_features["lyrical_density"] = 0.0

    # Optional: export feature table for verification
    features_table = [
        {"feature": str(k), "value": raw_features.get(k)}
        for k in sorted(raw_features.keys(), key=lambda x: str(x))
    ]
    if export_features_path:
        out_path = Path(str(export_features_path))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(features_table).to_csv(out_path, index=False, encoding="utf-8")

    input_df = pd.DataFrame([raw_features])

    # 2) Resolve model artifact paths (download from Supabase Storage if configured)
    model_sources: dict[str, str] = {}
    wanted_tasks = {"P0", "P2", "P3", "P4"} if bool(skip_p1) else {"P0", "P1", "P2", "P3", "P4"}
    model_paths = resolve_model_paths(
        allow_download=bool(allow_download),
        report=model_sources,
        force_storage=force_storage,
        tasks=sorted(wanted_tasks),
    )

    # 2b) Load/predict sequentially to reduce peak RAM (P1 can be large).
    import gc

    load_errors: list[str] = []
    required_by_task: dict[str, set[str]] = {}
    shap_payload: dict[str, Any] = {}

    def _safe_load_artifact(tid: str) -> dict | None:
        path = model_paths.get(str(tid).upper())
        if path is None:
            return None
        try:
            gc.collect()
            return joblib.load(str(path))
        except Exception as ex:
            load_errors.append(f"{tid}: {type(ex).__name__}: {ex}")
            return None

    def _capture_required(tid: str, art: dict) -> None:
        try:
            if tid in {"P0", "P1", "P3", "P4"}:
                pipe = art.get("pipeline")
                if pipe is not None:
                    required_by_task[tid] = _collect_required_columns(pipe)
            elif tid == "P2":
                cols = art.get("numeric_features")
                if isinstance(cols, list) and cols:
                    required_by_task["P2"] = {str(c) for c in cols if isinstance(c, str)}
        except Exception:
            return

    # 3) Predict P0..P4 (each task is fail-safe)
    p0_out: dict[str, Any]
    p0_art_for_shap: dict | None = None
    try:
        p0_art = _safe_load_artifact("P0") or {}
        _capture_required("P0", p0_art)
        p0_pipe = p0_art.get("pipeline")
        if p0_pipe is None:
            raise KeyError("Missing pipeline")
        df0 = _ensure_pipeline_input(input_df, p0_pipe)
        hit_p = float(np.asarray(p0_pipe.predict_proba(df0))[0, 1])
        thr = float(p0_art.get("optimal_threshold", 0.5))
        p0_out = {
            "hit_prob": float(hit_p * 100.0),
            "threshold": thr,
            "label": "Tiềm năng Hit cao" if hit_p >= thr else "Cần cải thiện thêm",
            "source": model_sources.get("P0", "model:p0"),
        }

        if compute_shap and not str(os.getenv("SHAP_DISABLE", "") or "").strip():
            p0_art_for_shap = p0_art
    except Exception as ex:
        p0_out = {
            "hit_prob": 0.0,
            "threshold": 0.5,
            "label": "N/A",
            "source": f"p0-error:{type(ex).__name__}",
        }

    p1_out: dict[str, Any]
    pop_score = 0.0
    if bool(skip_p1):
        p1_out = {
            "enabled": False,
            "popularity_score": None,
            "source": "disabled",
        }
    else:
        try:
            p1_art = _safe_load_artifact("P1") or {}
            _capture_required("P1", p1_art)
            p1_pipe = p1_art.get("pipeline")
            if p1_pipe is None:
                raise KeyError("Missing pipeline")
            df1 = _ensure_pipeline_input(input_df, p1_pipe)
            pop_score = float(np.asarray(p1_pipe.predict(df1)).reshape(-1)[0])
            if not np.isfinite(pop_score):
                pop_score = 0.0
            pop_score = float(np.clip(pop_score, 0.0, 100.0))
            p1_out = {
                "enabled": True,
                "popularity_score": pop_score,
                "source": model_sources.get("P1", "model:p1"),
            }
        except Exception as ex:
            p1_out = {
                "enabled": True,
                "popularity_score": 0.0,
                "source": f"p1-error:{type(ex).__name__}",
            }
        finally:
            try:
                del p1_art
            except Exception:
                pass
            gc.collect()

    # 4) SHAP only for P0 (cached background)
    if compute_shap and not str(os.getenv("SHAP_DISABLE", "") or "").strip():
        try:
            if p0_art_for_shap is not None:
                nsamples = int(shap_nsamples_kernel or int(os.getenv("SHAP_NSAMPLES_KERNEL", "80")))
                shap_input_df = input_df
                p0_pipe = p0_art_for_shap.get("pipeline")
                if p0_pipe is not None:
                    shap_input_df = _ensure_pipeline_input(shap_input_df, p0_pipe)
                shap_payload = build_shap_payload_p0_only(
                    p0_artifact=p0_art_for_shap,
                    input_raw_df=shap_input_df,
                    nsamples_kernel=nsamples,
                )
        except Exception as ex:
            shap_payload = {"tasks": {}, "error": f"{type(ex).__name__}: {ex}"}
    try:
        del p0_art_for_shap
    except Exception:
        pass
    gc.collect()

    # Feed predicted popularity for downstream tasks (or 0.0 if P1 disabled)
    try:
        input_df.loc[input_df.index[0], "spotify_popularity"] = float(pop_score)
        raw_features["spotify_popularity"] = float(pop_score)
    except Exception:
        pass

    p2_out: dict[str, Any]
    try:
        p2_art = _safe_load_artifact("P2") or {}
        _capture_required("P2", p2_art)
        cols = p2_art.get("numeric_features")
        if not isinstance(cols, list) or not cols:
            raise KeyError("Missing numeric_features")

        X = input_df.copy()
        for c in cols:
            if c not in X.columns:
                X[c] = 0.0
        X_num = X[cols].copy()

        imputer = p2_art.get("imputer")
        scaler = p2_art.get("scaler")
        pca = p2_art.get("pca")
        clusterer = p2_art.get("clusterer")
        if any(x is None for x in (imputer, scaler, pca, clusterer)):
            raise KeyError("Missing P2 components")

        Z = imputer.transform(X_num)
        Z = scaler.transform(Z)

        # Optional weights (if present in artifact)
        pre = p2_art.get("preprocess_artifacts")
        if isinstance(pre, dict) and pre.get("feature_weights") is not None:
            w = np.asarray(pre.get("feature_weights"), dtype=float).reshape(1, -1)
            if np.asarray(Z).shape[1] == w.shape[1]:
                Z = np.asarray(Z) * w

        Zp = pca.transform(Z)
        cid = int(np.asarray(clusterer.predict(Zp)).reshape(-1)[0])

        vibe_map = None
        meta = p2_art.get("meta")
        if isinstance(meta, dict) and isinstance(meta.get("vibe_map"), dict):
            vibe_map = meta.get("vibe_map")
        cluster_name = str(vibe_map.get(cid) if isinstance(vibe_map, dict) else "") or f"Cluster {cid}"

        p2_out = {
            "cluster_id": cid,
            "cluster_name": cluster_name,
            "source": model_sources.get("P2", "model:p2"),
        }
    except Exception as ex:
        p2_out = {
            "cluster_id": -1,
            "cluster_name": "",
            "source": f"p2-error:{type(ex).__name__}",
        }

    try:
        del p2_art
    except Exception:
        pass
    gc.collect()

    p3_out: dict[str, Any]
    try:
        p3_art = _safe_load_artifact("P3") or {}
        _capture_required("P3", p3_art)
        p3_pipe = p3_art.get("pipeline")
        if p3_pipe is None:
            raise KeyError("Missing pipeline")
        df3 = _ensure_pipeline_input(input_df, p3_pipe)

        if hasattr(p3_pipe, "predict_proba"):
            proba = np.asarray(p3_pipe.predict_proba(df3)).reshape(1, -1)
            cls = int(np.argmax(proba[0]))
            conf = float(np.max(proba[0]))
        else:
            cls = int(np.asarray(p3_pipe.predict(df3)).reshape(-1)[0])
            conf = 0.0

        class_names = ["Negative", "Neutral", "Positive"]
        label = class_names[cls] if 0 <= cls < len(class_names) else str(cls)

        p3_out = {
            "emotion_id": cls,
            "emotion_label": label,
            "confidence": conf,
            "source": model_sources.get("P3", "model:p3"),
        }
    except Exception as ex:
        p3_out = {
            "emotion_id": -1,
            "emotion_label": "",
            "confidence": 0.0,
            "source": f"p3-error:{type(ex).__name__}",
        }

    try:
        del p3_art
    except Exception:
        pass
    gc.collect()

    p4_out: dict[str, Any]
    try:
        p4_art = _safe_load_artifact("P4") or {}
        _capture_required("P4", p4_art)
        p4_pipe = p4_art.get("pipeline")
        if p4_pipe is None:
            raise KeyError("Missing pipeline")
        df4 = _ensure_pipeline_input(input_df, p4_pipe)

        label_names = p4_art.get("label_names")
        thresholds = p4_art.get("thresholds") if isinstance(p4_art.get("thresholds"), dict) else {}
        if not isinstance(label_names, list) or not label_names:
            raise KeyError("Missing label_names")

        proba_pos = _extract_multilabel_positive_proba(p4_pipe, df4)
        proba_by: dict[str, float] = {}
        if proba_pos is not None:
            row = np.asarray(proba_pos)[0]
            thr = np.asarray([float(thresholds.get(lbl, 0.5)) for lbl in label_names], dtype=float)
            chosen = [label_names[i] for i in range(len(label_names)) if bool(row[i] >= thr[i])]
            proba_by = {str(label_names[i]): float(row[i]) for i in range(len(label_names))}
        else:
            pred = p4_pipe.predict(df4)
            if hasattr(pred, "toarray"):
                pred = pred.toarray()
            pred_row = np.asarray(pred)[0]
            chosen = [label_names[i] for i, v in enumerate(pred_row.tolist()) if int(v) == 1 and i < len(label_names)]

        genres = [str(x).replace("genre_", "") for x in chosen if str(x).strip()]
        if not genres and proba_by:
            best = max(proba_by.items(), key=lambda kv: float(kv[1]))[0]
            genres = [str(best).replace("genre_", "")]
        if not genres:
            genres = ["V-Pop"]

        p4_out = {
            "genres": genres,
            "proba_by_genre": {k.replace("genre_", ""): float(v) for k, v in proba_by.items()},
            "source": model_sources.get("P4", "model:p4"),
        }
    except Exception as ex:
        p4_out = {
            "genres": [],
            "proba_by_genre": {},
            "source": f"p4-error:{type(ex).__name__}",
        }

    try:
        del p4_art
    except Exception:
        pass
    gc.collect()

    required_union: set[str] = set()
    for s in required_by_task.values():
        required_union |= set(s)

    # Feature completeness (% required columns truly available in the final raw_features)
    feature_completeness = {
        "union": _completeness_from_required(raw_features, required_union),
        "p0": _completeness_from_required(raw_features, required_by_task.get("P0", set())),
        "p1": _completeness_from_required(raw_features, required_by_task.get("P1", set())),
        "p2": _completeness_from_required(raw_features, required_by_task.get("P2", set())),
        "p3": _completeness_from_required(raw_features, required_by_task.get("P3", set())),
        "p4": _completeness_from_required(raw_features, required_by_task.get("P4", set())),
    }

    return {
        "p0": p0_out,
        "p1": p1_out,
        "p2": p2_out,
        "p3": p3_out,
        "p4": p4_out,
        "raw_features": raw_features,
        "features_table": features_table,
        "feature_completeness": feature_completeness,
        "input_df": input_df,
        "shap_values": shap_payload,
        "model_sources": model_sources,
        "load_errors": load_errors,
    }
