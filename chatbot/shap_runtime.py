from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


_FOCUS_FEATURES_DEFAULT: list[str] = [
    "tempo_bpm",
    "rms_energy",
    "beat_strength_mean",
    "lyric_total_words",
    "lexical_diversity",
]


_MAX_CONTRIB_FEATURES_DEFAULT = 200
_TOP_N_DEFAULT = 20


def _to_dense(x: Any) -> np.ndarray:
    if hasattr(x, "toarray"):
        return np.asarray(x.toarray())
    return np.asarray(x)


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _coerce_int(x: Any, default: int) -> int:
    try:
        v = int(x)
        return v
    except Exception:
        return int(default)


def _normalize_feature_names(names: list[str]) -> list[str]:
    # Strip common prefixes like "num__" or "cat__" and keep the business name.
    out: list[str] = []
    for n in names:
        s = str(n)
        # ColumnTransformer prefixes often look like "num__tempo_bpm"
        s = s.split("__")[-1]
        out.append(s)
    return out


def _aggregate_to_focus_features(
    shap_vec: np.ndarray,
    feature_names: list[str],
    focus_features: list[str],
    *,
    task_name: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    feature_names_l = [str(n).lower() for n in feature_names]

    for feat in focus_features:
        f_l = str(feat).lower()
        idxs = [i for i, nm in enumerate(feature_names_l) if f_l in nm]
        if not idxs:
            signed = 0.0
            abs_val = 0.0
        else:
            sub = np.asarray([shap_vec[i] for i in idxs], dtype=float)
            signed = float(np.sum(sub))
            abs_val = float(np.sum(np.abs(sub)))

        rows.append(
            {
                "feature": str(feat),
                "shap_value": float(signed),
                "abs_shap": float(abs_val),
                "task": str(task_name),
            }
        )

    abs_sum = float(sum(r["abs_shap"] for r in rows))
    denom = abs_sum if abs_sum > 1e-12 else 1.0
    for r in rows:
        r["contribution_percent"] = float(r["abs_shap"] / denom * 100.0)
        # drop internal key
        r.pop("abs_shap", None)

    rows.sort(key=lambda r: float(r.get("contribution_percent", 0.0)), reverse=True)
    return rows


def _build_per_feature_contributions(
    *,
    shap_vec: np.ndarray,
    feature_names: list[str],
    x_one: np.ndarray | None,
    task_name: str,
    max_features: int,
) -> tuple[list[dict[str, Any]], bool]:
    """Return (rows, truncated).

    Rows are sorted by |SHAP| desc and include a global % computed over all features.
    """

    vec = np.asarray(shap_vec, dtype=float).reshape(-1)
    n = int(vec.shape[0])
    names = list(feature_names)
    if len(names) != n:
        names = [f"f_{i}" for i in range(n)]

    x_vals: np.ndarray | None = None
    if x_one is not None:
        arr = np.asarray(x_one)
        if arr.ndim == 2 and arr.shape[0] >= 1:
            row0 = np.asarray(arr[0]).reshape(-1)
            if int(row0.shape[0]) == n:
                x_vals = row0

    abs_vals = np.abs(vec)
    abs_total = float(abs_vals.sum())
    denom = abs_total if abs_total > 1e-12 else 1.0

    order = np.argsort(abs_vals)[::-1]
    truncated = False
    if max_features and max_features > 0 and len(order) > int(max_features):
        order = order[: int(max_features)]
        truncated = True

    rows: list[dict[str, Any]] = []
    for i in order:
        i_int = int(i)
        shap_value = float(vec[i_int])
        abs_shap = float(abs_vals[i_int])
        row: dict[str, Any] = {
            "feature": str(names[i_int]),
            "shap_value": shap_value,
            "contribution_percent": float(abs_shap / denom * 100.0),
            "task": str(task_name),
        }
        if x_vals is not None:
            row["feature_value"] = float(x_vals[i_int])
        rows.append(row)

    return rows, truncated


def _kernel_shap_one_sample(
    *,
    predict_fn,
    background: np.ndarray,
    x_one: np.ndarray,
    nsamples: int,
) -> np.ndarray:
    import shap  # local import (heavy)

    # Reduce background with kmeans to keep KernelExplainer fast/stable.
    bg = np.asarray(background)
    if bg.ndim != 2:
        bg = bg.reshape((bg.shape[0], -1))

    k = min(25, len(bg))
    if k >= 5 and len(bg) > k:
        try:
            bg_small = shap.kmeans(bg, k)
        except Exception:
            bg_small = bg[:k]
    else:
        bg_small = bg

    explainer = shap.KernelExplainer(predict_fn, bg_small)
    vals = explainer.shap_values(x_one, nsamples=int(nsamples))

    arr = np.asarray(vals)
    # We expect predict_fn to be scalar -> arr shape (1, n_features)
    if arr.ndim == 2:
        return np.asarray(arr[0], dtype=float)

    # Some shap versions return list even for scalar; handle list/tuple.
    if isinstance(vals, (list, tuple)) and len(vals) >= 1:
        v0 = np.asarray(vals[0])
        if v0.ndim == 2:
            return np.asarray(v0[0], dtype=float)

    return np.asarray(arr).reshape(-1).astype(float)


def _is_tree_estimator(estimator: Any) -> bool:
    tree_tokens = (
        "xgb",
        "xgboost",
        "randomforest",
        "extratrees",
        "gradientboosting",
        "decisiontree",
        "histgradientboosting",
        "lightgbm",
        "lgbm",
        "catboost",
    )
    try:
        cls_name = estimator.__class__.__name__.lower()
        mod_name = estimator.__class__.__module__.lower()
    except Exception:
        return False
    return any(tok in cls_name or tok in mod_name for tok in tree_tokens)


def _as_1d_shap_vector(vals: Any, *, mode: str) -> np.ndarray:
    """Normalize SHAP outputs to 1D (n_features,) for a single sample.

    mode:
      - 'proba' (binary positive class)
      - 'regression'
    """

    # Many SHAP APIs return list[class] for classification.
    if isinstance(vals, (list, tuple)) and len(vals) >= 1:
        if mode == "proba" and len(vals) >= 2:
            arr = np.asarray(vals[1])
        else:
            arr = np.asarray(vals[0])
    else:
        arr = np.asarray(vals)

    # Common shapes:
    # - (1, n_features)
    # - (n_features,)
    # - (1, n_features, 2) for binary (take class-1)
    if arr.ndim == 3 and arr.shape[0] >= 1 and arr.shape[-1] >= 2:
        arr = arr[0, :, 1]
    elif arr.ndim == 2 and arr.shape[0] >= 1:
        arr = arr[0]
    return np.asarray(arr, dtype=float).reshape(-1)


def _compute_shap_one_sample(
    *,
    estimator: Any,
    mode: str,
    predict_fn,
    background: np.ndarray,
    x_one: np.ndarray,
    nsamples: int,
) -> np.ndarray:
    """Compute a SHAP vector for a single sample.

    Tries TreeExplainer for tree estimators first; falls back to KernelExplainer.
    """

    import shap  # local import (heavy)

    if _is_tree_estimator(estimator):
        try:
            explainer = shap.TreeExplainer(estimator, data=np.asarray(background))
            try:
                vals = explainer.shap_values(np.asarray(x_one), check_additivity=False)
            except TypeError:
                vals = explainer.shap_values(np.asarray(x_one))
            vec = _as_1d_shap_vector(vals, mode=mode)
            if vec.size:
                return vec
        except Exception:
            pass

    return _kernel_shap_one_sample(
        predict_fn=predict_fn,
        background=background,
        x_one=x_one,
        nsamples=int(nsamples),
    )


def build_shap_payload_p0_p1(
    *,
    p0_artifact: dict,
    p1_artifact: dict,
    input_raw_df: pd.DataFrame,
    focus_features: list[str] | None = None,
    nsamples_kernel: int = 80,
    top_n: int = _TOP_N_DEFAULT,
    max_contrib_features: int = _MAX_CONTRIB_FEATURES_DEFAULT,
) -> dict[str, Any]:
    """Compute a small SHAP payload for P0/P1 using artifact shap_cache.

    Output schema matches what `render_shap_payload_cached()` expects in `chatbot/app_chatbot.py`.
    """

    focus = list(focus_features) if focus_features is not None else list(_FOCUS_FEATURES_DEFAULT)

    out: dict[str, Any] = {
        "focus_features": list(focus),
        "tasks": {},
    }

    try:
        from DA.SHAP_explain.shap_artifact import transform_for_model, try_get_feature_names
    except Exception as ex:
        msg = str(ex) or ex.__class__.__name__
        out["tasks"]["p0"] = {
            "title": "P0 - Hit Probability",
            "status": "import-failed",
            "detail": msg,
            "is_real_shap": False,
            "contributions": [],
        }
        out["tasks"]["p1"] = {
            "title": "P1 - Popularity Score",
            "status": "import-failed",
            "detail": msg,
            "is_real_shap": False,
            "contributions": [],
        }
        return out

    def _task_block(*, key: str, title: str, artifact: dict, mode: str) -> None:
        # mode: 'proba' (binary positive) or 'regression'
        try:
            import shap  # noqa: F401
        except Exception as ex:
            out["tasks"][key] = {
                "title": title,
                "status": "shap-missing",
                "detail": f"{type(ex).__name__}: {ex}",
                "is_real_shap": False,
                "contributions": [],
            }
            return

        pipe = artifact.get("pipeline")
        cache = artifact.get("shap_cache")
        if pipe is None:
            out["tasks"][key] = {
                "title": title,
                "status": "model-missing",
                "detail": "Missing pipeline in artifact",
                "is_real_shap": False,
                "contributions": [],
            }
            return
        if not isinstance(cache, dict) or cache.get("X_background_raw") is None:
            out["tasks"][key] = {
                "title": title,
                "status": "cache-missing",
                "detail": "Missing shap_cache.X_background_raw in artifact",
                "is_real_shap": False,
                "contributions": [],
            }
            return

        X_bg_raw = cache.get("X_background_raw")
        if not isinstance(X_bg_raw, pd.DataFrame):
            out["tasks"][key] = {
                "title": title,
                "status": "cache-invalid",
                "detail": "shap_cache.X_background_raw is not a DataFrame",
                "is_real_shap": False,
                "contributions": [],
            }
            return

        # Transform raw -> estimator feature space
        X_bg = _to_dense(transform_for_model(pipe, X_bg_raw))
        X_one = _to_dense(transform_for_model(pipe, input_raw_df.iloc[[0]].copy()))

        # Feature names in estimator feature space
        names = try_get_feature_names(pipe)
        if names is None or len(names) != int(X_bg.shape[1]):
            feature_names = [f"f_{i}" for i in range(int(X_bg.shape[1]))]
        else:
            feature_names = _normalize_feature_names([str(x) for x in list(names)])

        estimator = getattr(pipe, "steps", [])[-1][1]

        if mode == "proba" and hasattr(estimator, "predict_proba"):
            def predict_fn(mat):
                p = estimator.predict_proba(mat)
                arr = np.asarray(p)
                if arr.ndim == 2 and arr.shape[1] >= 2:
                    return arr[:, 1]
                return arr.reshape(-1)

        else:
            def predict_fn(mat):
                p = estimator.predict(mat)
                arr = np.asarray(p)
                return arr.reshape(-1)

        try:
            shap_vec = _compute_shap_one_sample(
                estimator=estimator,
                mode=("proba" if mode == "proba" else "regression"),
                predict_fn=predict_fn,
                background=X_bg,
                x_one=X_one,
                nsamples=int(nsamples_kernel),
            )
            rows, truncated = _build_per_feature_contributions(
                shap_vec=shap_vec,
                feature_names=feature_names,
                x_one=X_one,
                task_name=title,
                max_features=int(max_contrib_features),
            )

            focus_rows = _aggregate_to_focus_features(
                shap_vec,
                feature_names,
                focus,
                task_name=title,
            )

            top_n_int = _coerce_int(top_n, _TOP_N_DEFAULT)
            if top_n_int <= 0:
                top_n_int = _TOP_N_DEFAULT
            top_rows = rows[: min(len(rows), top_n_int)]
            out["tasks"][key] = {
                "title": title,
                "status": "ok",
                "detail": "",
                "is_real_shap": True,
                "contributions": rows,
                "contributions_top": top_rows,
                "focus_contributions": focus_rows,
                "feature_count_total": int(len(feature_names)),
                "is_truncated": bool(truncated),
            }
        except Exception as ex:
            out["tasks"][key] = {
                "title": title,
                "status": "shap-failed",
                "detail": f"{type(ex).__name__}: {ex}",
                "is_real_shap": False,
                "contributions": [],
            }

    _task_block(key="p0", title="P0 - Hit Probability", artifact=p0_artifact, mode="proba")
    _task_block(key="p1", title="P1 - Popularity Score", artifact=p1_artifact, mode="regression")

    # Optional combined summary across P0/P1 for AI prompt.
    try:
        combined: dict[str, float] = {f: 0.0 for f in focus}
        counts: dict[str, int] = {f: 0 for f in focus}
        for k in ("p0", "p1"):
            task = out.get("tasks", {}).get(k) or {}
            contribs = task.get("contributions") or []
            if not isinstance(contribs, list):
                continue
            for r in contribs:
                feat = str(r.get("feature") or "")
                if feat in combined:
                    combined[feat] += _safe_float(r.get("contribution_percent"), 0.0)
                    counts[feat] += 1
        combined_rows = []
        for feat in focus:
            c = counts.get(feat, 0) or 0
            v = combined.get(feat, 0.0)
            avg = float(v / c) if c else 0.0
            combined_rows.append({"feature": feat, "contribution_percent": avg})
        combined_rows.sort(key=lambda r: float(r.get("contribution_percent", 0.0)), reverse=True)
        out["combined"] = {
            "title": "P0+P1 (combined)",
            "contributions": combined_rows,
        }
    except Exception:
        pass

    return out


def build_shap_payload_p0_only(
    *,
    p0_artifact: dict,
    input_raw_df: pd.DataFrame,
    focus_features: list[str] | None = None,
    nsamples_kernel: int = 80,
    top_n: int = _TOP_N_DEFAULT,
    max_contrib_features: int = _MAX_CONTRIB_FEATURES_DEFAULT,
) -> dict[str, Any]:
    """Compute a small SHAP payload for P0 only using artifact shap_cache.

    This avoids loading/running P1 when models are not compatible.
    Output schema matches what `render_shap_payload_cached()` expects in `chatbot/app_chatbot.py`.
    """

    focus = list(focus_features) if focus_features is not None else list(_FOCUS_FEATURES_DEFAULT)

    out: dict[str, Any] = {
        "focus_features": list(focus),
        "tasks": {},
    }

    try:
        from DA.SHAP_explain.shap_artifact import transform_for_model, try_get_feature_names
    except Exception as ex:
        msg = str(ex) or ex.__class__.__name__
        out["tasks"]["p0"] = {
            "title": "P0 - Hit Probability",
            "status": "import-failed",
            "detail": msg,
            "is_real_shap": False,
            "contributions": [],
        }
        return out

    try:
        import shap  # noqa: F401
    except Exception as ex:
        out["tasks"]["p0"] = {
            "title": "P0 - Hit Probability",
            "status": "shap-missing",
            "detail": f"{type(ex).__name__}: {ex}",
            "is_real_shap": False,
            "contributions": [],
        }
        return out

    pipe = p0_artifact.get("pipeline")
    cache = p0_artifact.get("shap_cache")
    if pipe is None:
        out["tasks"]["p0"] = {
            "title": "P0 - Hit Probability",
            "status": "model-missing",
            "detail": "Missing pipeline in artifact",
            "is_real_shap": False,
            "contributions": [],
        }
        return out
    if not isinstance(cache, dict) or cache.get("X_background_raw") is None:
        out["tasks"]["p0"] = {
            "title": "P0 - Hit Probability",
            "status": "cache-missing",
            "detail": "Missing shap_cache.X_background_raw in artifact",
            "is_real_shap": False,
            "contributions": [],
        }
        return out

    X_bg_raw = cache.get("X_background_raw")
    if not isinstance(X_bg_raw, pd.DataFrame):
        out["tasks"]["p0"] = {
            "title": "P0 - Hit Probability",
            "status": "cache-invalid",
            "detail": "shap_cache.X_background_raw is not a DataFrame",
            "is_real_shap": False,
            "contributions": [],
        }
        return out

    try:
        X_bg = _to_dense(transform_for_model(pipe, X_bg_raw))
        X_one = _to_dense(transform_for_model(pipe, input_raw_df.iloc[[0]].copy()))
    except Exception as ex:
        out["tasks"]["p0"] = {
            "title": "P0 - Hit Probability",
            "status": "transform-failed",
            "detail": f"{type(ex).__name__}: {ex}",
            "is_real_shap": False,
            "contributions": [],
        }
        return out

    names = try_get_feature_names(pipe)
    if names is None or len(names) != int(X_bg.shape[1]):
        feature_names = [f"f_{i}" for i in range(int(X_bg.shape[1]))]
    else:
        feature_names = _normalize_feature_names([str(x) for x in list(names)])

    estimator = getattr(pipe, "steps", [])[-1][1]

    if hasattr(estimator, "predict_proba"):
        def predict_fn(mat):
            p = estimator.predict_proba(mat)
            arr = np.asarray(p)
            if arr.ndim == 2 and arr.shape[1] >= 2:
                return arr[:, 1]
            return arr.reshape(-1)
    else:
        def predict_fn(mat):
            p = estimator.predict(mat)
            arr = np.asarray(p)
            return arr.reshape(-1)

    try:
        shap_vec = _compute_shap_one_sample(
            estimator=estimator,
            mode="proba",
            predict_fn=predict_fn,
            background=X_bg,
            x_one=X_one,
            nsamples=int(nsamples_kernel),
        )
        # Primary: real per-feature SHAP contributions (sorted by |SHAP|).
        rows, truncated = _build_per_feature_contributions(
            shap_vec=shap_vec,
            feature_names=feature_names,
            x_one=X_one,
            task_name="P0 - Hit Probability",
            max_features=int(max_contrib_features),
        )

        # Secondary: the previous "focus feature" aggregation (handy for business summary).
        focus_rows = _aggregate_to_focus_features(
            shap_vec,
            feature_names,
            focus,
            task_name="P0 - Hit Probability",
        )

        # For LLM prompts, expose a small top-N view to avoid huge payloads.
        top_n_int = _coerce_int(top_n, _TOP_N_DEFAULT)
        if top_n_int <= 0:
            top_n_int = _TOP_N_DEFAULT
        top_rows = rows[: min(len(rows), top_n_int)]
        out["tasks"]["p0"] = {
            "title": "P0 - Hit Probability",
            "status": "ok",
            "detail": "",
            "is_real_shap": True,
            "contributions": rows,
            "contributions_top": top_rows,
            "focus_contributions": focus_rows,
            "feature_count_total": int(len(feature_names)),
            "is_truncated": bool(truncated),
        }
    except Exception as ex:
        out["tasks"]["p0"] = {
            "title": "P0 - Hit Probability",
            "status": "shap-failed",
            "detail": f"{type(ex).__name__}: {ex}",
            "is_real_shap": False,
            "contributions": [],
        }

    return out
