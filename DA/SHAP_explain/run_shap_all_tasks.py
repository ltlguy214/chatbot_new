from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import transformers

_THIS_DIR = Path(__file__).resolve().parent

# Ensure repository root is on sys.path so `import DA...` works when run by file path.
_DA_DIR = _THIS_DIR.parent
_REPO_ROOT = _DA_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from DA.SHAP_explain.shap_artifact import transform_for_model, try_get_feature_names
from DA.SHAP_explain.shap_artifact import ShapCacheConfig, build_shap_cache
from DA.models.topic_mapping import rename_topics_in_feature_names
DPI = 300
TOP_N = 20
RANDOM_STATE = 42


def _force_utf8_stdio() -> None:
    """Avoid UnicodeEncodeError on some Windows consoles (cp1252/cp936...)."""
    for stream in (sys.stdout, sys.stderr):
        try:
            # Python 3.7+: TextIOWrapper supports reconfigure
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def _maybe_silence_warnings() -> None:
    """Hide noisy non-fatal warnings by default (can re-enable via env)."""
    if os.getenv("SHAP_SHOW_WARNINGS", "0") == "1":
        return

    warnings.filterwarnings(
        "ignore",
        message=r"X has feature names, but .* was fitted without feature names",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"The NumPy global RNG was seeded by calling `np\.random\.seed`",
        category=FutureWarning,
    )

def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _to_dense(matrix):
    if hasattr(matrix, "toarray"):
        return matrix.toarray()
    return matrix


def _infer_problem_type(pkl_obj: Dict[str, Any]) -> str:
    if "r2_score" in pkl_obj:
        return "regression"
    return "classification"


def _is_tree_model(estimator: Any) -> bool:
    name = estimator.__class__.__name__.lower()
    module = estimator.__class__.__module__.lower()

    if hasattr(estimator, "feature_importances_"):
        return True

    treeish = (
        "randomforest" in name
        or "extratrees" in name
        or "gradientboost" in name
        or "xgb" in name
        or "xgboost" in module
        or "lgbm" in name
        or "lightgbm" in module
        or "catboost" in module
    )
    return bool(treeish)


def _pick_feature_names(pipeline, n_features: int) -> list[str]:
    names = try_get_feature_names(pipeline)
    if names is None or len(names) != n_features:
        # PCA-heavy pipelines (e.g., P2) don't expose feature names; use PC labels.
        try:
            from sklearn.decomposition import PCA

            for _, step in getattr(pipeline, "steps", [])[:-1]:
                if isinstance(step, PCA):
                    return [f"PC{i+1}" for i in range(n_features)]
        except Exception:
            pass
        return [f"f_{i}" for i in range(n_features)]
    # Strip transformer prefixes if present
    base = [str(n).split("__")[-1] for n in names.tolist()]
    return rename_topics_in_feature_names(base)


def _compute_top_indices(shap_values_2d: np.ndarray, top_n: int) -> np.ndarray:
    mean_abs = np.abs(shap_values_2d).mean(axis=0).flatten()
    top_indices = np.argsort(mean_abs)[-top_n:][::-1]
    return top_indices


def _save_beeswarm(
    shap_values_2d: np.ndarray,
    X_explain_2d: np.ndarray,
    feature_names: list[str],
    title: str,
    out_path: Path,
) -> None:
    import shap

    _ensure_dir(out_path)
    plt.figure(figsize=(14, 10))
    shap.summary_plot(
        shap_values_2d,
        X_explain_2d,
        feature_names=feature_names,
        show=False,
        max_display=len(feature_names),
    )
    plt.title(title, fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close()


def _save_bar(
    shap_values_2d: np.ndarray,
    feature_names: list[str],
    title: str,
    out_path: Path,
) -> None:
    mean_abs = np.abs(shap_values_2d).mean(axis=0).flatten()
    _save_bar_from_importances(mean_abs, feature_names, title, out_path)


def _save_bar_from_importances(
    importances: np.ndarray,
    feature_names: list[str],
    title: str,
    out_path: Path,
) -> None:
    _ensure_dir(out_path)

    importances = np.asarray(importances).flatten()
    order = np.argsort(importances)  # ascending
    ordered_names = [feature_names[i] for i in order]
    ordered_vals = importances[order]

    fig, ax = plt.subplots(figsize=(12, 10))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(ordered_vals)))
    ax.barh(range(len(ordered_vals)), ordered_vals, color=colors, edgecolor="black", linewidth=0.8)
    ax.set_yticks(range(len(ordered_vals)))
    ax.set_yticklabels(ordered_names, fontsize=10)

    max_val = float(np.max(ordered_vals)) if len(ordered_vals) else 0.0
    for i, val in enumerate(ordered_vals):
        ax.text(val + max_val * 0.01, i, f"{float(val):.4f}", va="center", ha="left", fontsize=9)

    ax.set_xlim(0, max_val * 1.15 if max_val > 0 else 1.0)
    ax.set_xlabel("Mean |SHAP Value|", fontsize=12, fontweight="bold")
    ax.set_ylabel("Feature", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    ax.grid(axis="x", alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close()


def _save_drift_bar(
    *,
    drift: np.ndarray,
    feature_names: list[str],
    out_path: Path,
    title: str,
    top_n: int = 20,
) -> None:
    """Bar chart of top-N drift features (positive = increased importance)."""
    _ensure_dir(out_path)
    drift = np.asarray(drift).flatten()
    if len(drift) != len(feature_names):
        raise ValueError("Drift và feature_names không khớp kích thước")

    order = np.argsort(np.abs(drift))[-top_n:][::-1]
    vals = drift[order][::-1]  # plot bottom-to-top
    names = [feature_names[i] for i in order][::-1]

    colors = ["#1f77b4" if v >= 0 else "#d62728" for v in vals]
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.barh(range(len(vals)), vals, color=colors, edgecolor="black", linewidth=0.8)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(names, fontsize=10)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Drift = mean(|SHAP|)_TEST - mean(|SHAP|)_TRAIN", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=18)
    ax.grid(axis="x", alpha=0.25, linestyle="--")
    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close()


def _save_drift_scatter(
    *,
    train_importance: np.ndarray,
    test_importance: np.ndarray,
    out_path: Path,
    title: str,
) -> None:
    """Scatter plot comparing train vs test SHAP importances."""
    _ensure_dir(out_path)
    x = np.asarray(train_importance).flatten()
    y = np.asarray(test_importance).flatten()
    if len(x) != len(y):
        raise ValueError("train/test importance không khớp kích thước")

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.scatter(x, y, s=16, alpha=0.45)

    max_v = float(max(np.max(x), np.max(y))) if len(x) else 1.0
    ax.plot([0, max_v], [0, max_v], linestyle="--", color="black", linewidth=1, alpha=0.7)
    ax.set_xlim(0, max_v * 1.05)
    ax.set_ylim(0, max_v * 1.05)
    ax.set_xlabel("mean(|SHAP|) TRAIN", fontsize=11, fontweight="bold")
    ax.set_ylabel("mean(|SHAP|) TEST", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=18)
    ax.grid(alpha=0.25, linestyle="--")
    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close()


def _safe_filename(name: str) -> str:
    """Convert an arbitrary label to a filesystem-safe stem."""
    s = str(name).strip()
    # Replace path separators and other problematic chars
    for ch in ["/", "\\", ":", "*", "?", '"', "<", ">", "|", "\n", "\r", "\t"]:
        s = s.replace(ch, "-")
    s = "_".join(s.split())
    return s or "label"


def _extract_shap_cache(pkl_obj: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cache = pkl_obj.get("shap_cache")
    if not isinstance(cache, dict):
        raise KeyError("Missing 'shap_cache' in artifact. Re-run the training script once to regenerate the .pkl.")

    X_background_raw = cache.get("X_background_raw")
    X_explain_raw = cache.get("X_explain_raw")
    if not isinstance(X_background_raw, pd.DataFrame) or not isinstance(X_explain_raw, pd.DataFrame):
        raise TypeError("Invalid 'shap_cache' format; expected DataFrames.")

    return X_background_raw, X_explain_raw


def _ensure_sentiment_ohe_p2(df: pd.DataFrame) -> pd.DataFrame:
    """Recreate the fixed 3-dim sentiment one-hot used by the P2 preprocessing."""
    df = df.copy()
    if "final_sentiment" in df.columns:
        s = df["final_sentiment"].fillna("Neutral").astype(str)
        s = s.str.strip().str.lower().replace({"neg": "negative", "pos": "positive"})
        df["sentiment_negative"] = (s == "negative").astype(float)
        df["sentiment_neutral"] = (s == "neutral").astype(float)
        df["sentiment_positive"] = (s == "positive").astype(float)
    else:
        df["sentiment_negative"] = 0.0
        df["sentiment_neutral"] = 1.0
        df["sentiment_positive"] = 0.0
    return df


def _build_shap_cache_if_missing(
    *,
    pkl_obj: Dict[str, Any],
    root: Path,
    n_background: int,
    n_explain: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (X_bg_raw, X_ex_raw), building from data_source for P2 if needed."""

    try:
        return _extract_shap_cache(pkl_obj)
    except Exception:
        pass

    # P2 artifacts currently may not include shap_cache; rebuild from data_source.
    if str(pkl_obj.get("task_id", "")).upper() != "P2":
        raise KeyError("Missing 'shap_cache' in artifact.")

    data_source = str(pkl_obj.get("data_source", "")).strip()
    if not data_source:
        raise KeyError("Missing 'data_source' in P2 artifact; cannot rebuild shap_cache.")

    src_path = (root / data_source).resolve() if not Path(data_source).is_absolute() else Path(data_source)
    if not src_path.exists():
        raise FileNotFoundError(f"Missing data_source CSV: {src_path}")

    df = pd.read_csv(src_path)
    df = _ensure_sentiment_ohe_p2(df)

    cols = pkl_obj.get("numeric_features")
    if not isinstance(cols, list) or not cols:
        raise KeyError("Missing 'numeric_features' in P2 artifact; cannot rebuild shap_cache.")

    # Ensure all expected columns exist; fill missing with 0.
    for c in cols:
        if c not in df.columns:
            df[c] = 0.0

    X_raw = df[cols].copy()
    cache = build_shap_cache(
        X_raw,
        config=ShapCacheConfig(n_background=n_background, n_explain=n_explain, random_state=RANDOM_STATE),
    )
    return cache["X_background_raw"], cache["X_explain_raw"]


def _get_pipeline_from_artifact(pkl_obj: Dict[str, Any]):
    pipeline = pkl_obj.get("pipeline")
    if pipeline is not None:
        return pipeline

    # P2 clustering artifact: build a fitted sklearn Pipeline on-the-fly.
    if str(pkl_obj.get("task_id", "")).upper() != "P2":
        return None

    imputer = pkl_obj.get("imputer")
    scaler = pkl_obj.get("scaler")
    pca = pkl_obj.get("pca")
    clusterer = pkl_obj.get("clusterer")
    if any(x is None for x in (imputer, scaler, pca, clusterer)):
        return None

    weights = None
    pre = pkl_obj.get("preprocess_artifacts")
    if isinstance(pre, dict) and "feature_weights" in pre:
        weights = pre.get("feature_weights")

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer

    def _apply_weights(X):
        if weights is None:
            return X
        w = np.asarray(weights, dtype=float).reshape(1, -1)
        if X.shape[1] != w.shape[1]:
            return X
        return X * w

    return Pipeline(
        steps=[
            ("imputer", imputer),
            ("scaler", scaler),
            ("weight", FunctionTransformer(_apply_weights, validate=False)),
            ("pca", pca),
            ("clusterer", clusterer),
        ]
    )

def _run_shap_for_pipeline(
    *,
    pipeline,
    X_background_raw: pd.DataFrame,
    X_explain_raw: pd.DataFrame,
    task_label: str,
    out_beeswarm: Path,
    out_bar: Path,
    problem_type: str,
    class_mode: str = "binary_positive",
    nsamples_kernel: int = 300,
) -> None:
    import shap

    def _as_2d_shap(arr: Any) -> np.ndarray:
        """Normalize SHAP outputs to 2D (n_samples, n_features).

        Some explainers/versions return (n, f, c). For binary tasks we take class-1.
        """
        a = np.asarray(arr)
        if a.ndim == 3 and a.shape[-1] >= 2:
            return a[:, :, 1]
        return a

    X_bg = _to_dense(transform_for_model(pipeline, X_background_raw))
    X_ex = _to_dense(transform_for_model(pipeline, X_explain_raw))

    estimator = pipeline.steps[-1][1]
    feature_names_all = _pick_feature_names(pipeline, int(X_ex.shape[1]))

    if problem_type == "regression":
        predict_fn = estimator.predict

        shap_values = None
        if _is_tree_model(estimator):
            try:
                explainer = shap.TreeExplainer(estimator, data=X_bg)
                shap_values = explainer.shap_values(X_ex)
            except Exception:
                shap_values = None

        if shap_values is None:
            np.random.seed(RANDOM_STATE)
            background = shap.kmeans(X_bg, 25)
            explainer = shap.KernelExplainer(predict_fn, background)
            shap_values = explainer.shap_values(X_ex, nsamples=nsamples_kernel)

        shap_2d = _as_2d_shap(shap_values)
        mean_abs_base = np.abs(shap_2d).mean(axis=0).flatten()

    else:
        # classification
        if hasattr(estimator, "predict_proba"):
            predict_fn = estimator.predict_proba

            shap_values = None
            if _is_tree_model(estimator):
                try:
                    explainer = shap.TreeExplainer(estimator, data=X_bg)
                    shap_values = explainer.shap_values(X_ex)
                except Exception:
                    shap_values = None

            if shap_values is None:
                np.random.seed(RANDOM_STATE)
                background = shap.kmeans(X_bg, 25)
                explainer = shap.KernelExplainer(predict_fn, background)
                shap_values = explainer.shap_values(X_ex, nsamples=nsamples_kernel)

            shap_for_plot = None

            # Binary: list [class0, class1]
            # Multiclass: list per class
            if isinstance(shap_values, list) and len(shap_values) >= 2:
                if class_mode == "binary_positive":
                    shap_for_plot = shap_values[1]
                elif class_mode.startswith("class_"):
                    idx = int(class_mode.split("_")[-1])
                    shap_for_plot = shap_values[idx]
                else:
                    shap_for_plot = shap_values[0]

                # Multiclass ranking: average mean(|SHAP|) across classes
                if len(shap_values) > 2:
                    mean_abs_base = (
                        np.mean([np.abs(_as_2d_shap(sv)) for sv in shap_values], axis=0)
                        .mean(axis=0)
                        .flatten()
                    )
                    shap_2d = _as_2d_shap(shap_for_plot)
                else:
                    shap_2d = _as_2d_shap(shap_for_plot)
                    mean_abs_base = np.abs(shap_2d).mean(axis=0).flatten()
            else:
                shap_2d = _as_2d_shap(shap_values)
                mean_abs_base = np.abs(shap_2d).mean(axis=0).flatten()
        else:
            # Fallback: estimator has no predict_proba (e.g., clustering).
            # Explain predict() output with KernelExplainer (treat as regression-like signal).
            if hasattr(estimator, "transform"):
                # Prefer a continuous score: closeness to nearest centroid.
                def predict_fn(X):
                    d = estimator.transform(X)
                    return -np.min(d, axis=1)
            else:
                predict_fn = estimator.predict
            shap_values = None
            if _is_tree_model(estimator):
                try:
                    explainer = shap.TreeExplainer(estimator, data=X_bg)
                    shap_values = explainer.shap_values(X_ex)
                except Exception:
                    shap_values = None

            if shap_values is None:
                np.random.seed(RANDOM_STATE)
                background = shap.kmeans(X_bg, 25)
                explainer = shap.KernelExplainer(predict_fn, background)
                shap_values = explainer.shap_values(X_ex, nsamples=nsamples_kernel)

            shap_2d = _as_2d_shap(shap_values)
            mean_abs_base = np.abs(shap_2d).mean(axis=0).flatten()

    top_idx = np.argsort(mean_abs_base)[-TOP_N:][::-1]

    shap_top = shap_2d[:, top_idx]
    X_ex_top = np.asarray(X_ex)[:, top_idx]
    feat_top = [feature_names_all[int(i)] for i in top_idx]

    _save_beeswarm(
        shap_top,
        X_ex_top,
        feat_top,
        title=f"Top {len(feat_top)} Features - SHAP (Beeswarm)\n{task_label}",
        out_path=out_beeswarm,
    )

    _save_bar_from_importances(
        mean_abs_base[top_idx],
        feat_top,
        title=f"Top {len(feat_top)} Features - SHAP (Bar)\n{task_label}",
        out_path=out_bar,
    )


def _compute_mean_abs_shap_importance(
    *,
    pipeline,
    X_background_raw: pd.DataFrame,
    X_explain_raw: pd.DataFrame,
    problem_type: str,
    class_mode: str,
    nsamples_kernel: int,
) -> Tuple[np.ndarray, list[str]]:
    """Return (mean_abs_importance, feature_names_all)."""
    import shap

    def _as_2d(arr: Any) -> np.ndarray:
        a = np.asarray(arr)
        if a.ndim == 3 and a.shape[-1] >= 2:
            return a[:, :, 1]
        return a

    X_bg = _to_dense(transform_for_model(pipeline, X_background_raw))
    X_ex = _to_dense(transform_for_model(pipeline, X_explain_raw))

    estimator = pipeline.steps[-1][1]
    feature_names_all = _pick_feature_names(pipeline, int(np.asarray(X_ex).shape[1]))

    if problem_type == "regression":
        predict_fn = estimator.predict
        shap_values = None
        if _is_tree_model(estimator):
            try:
                explainer = shap.TreeExplainer(estimator, data=X_bg)
                shap_values = explainer.shap_values(X_ex)
            except Exception:
                shap_values = None

        if shap_values is None:
            np.random.seed(RANDOM_STATE)
            background = shap.kmeans(X_bg, min(25, len(np.asarray(X_bg))))
            explainer = shap.KernelExplainer(predict_fn, background)
            shap_values = explainer.shap_values(X_ex, nsamples=nsamples_kernel)
        shap_2d = _as_2d(shap_values)
        return np.abs(shap_2d).mean(axis=0).flatten(), feature_names_all

    # classification / clustering
    if hasattr(estimator, "predict_proba"):
        predict_fn = estimator.predict_proba
        shap_values = None
        if _is_tree_model(estimator):
            try:
                explainer = shap.TreeExplainer(estimator, data=X_bg)
                shap_values = explainer.shap_values(X_ex)
            except Exception:
                shap_values = None

        if shap_values is None:
            np.random.seed(RANDOM_STATE)
            background = shap.kmeans(X_bg, min(25, len(np.asarray(X_bg))))
            explainer = shap.KernelExplainer(predict_fn, background)
            shap_values = explainer.shap_values(X_ex, nsamples=nsamples_kernel)

        if isinstance(shap_values, list) and len(shap_values) >= 2:
            if class_mode == "binary_positive":
                shap_2d = _as_2d(shap_values[1])
            elif class_mode.startswith("class_"):
                idx = int(class_mode.split("_")[-1])
                shap_2d = _as_2d(shap_values[idx])
            else:
                shap_2d = _as_2d(shap_values[0])
        else:
            shap_2d = _as_2d(shap_values)

        return np.abs(shap_2d).mean(axis=0).flatten(), feature_names_all

    # clustering-like (no predict_proba)
    if hasattr(estimator, "transform"):
        def predict_fn(X):
            d = estimator.transform(X)
            return -np.min(d, axis=1)
    else:
        predict_fn = estimator.predict

    shap_values = None
    if _is_tree_model(estimator):
        try:
            explainer = shap.TreeExplainer(estimator, data=X_bg)
            shap_values = explainer.shap_values(X_ex)
        except Exception:
            shap_values = None

    if shap_values is None:
        np.random.seed(RANDOM_STATE)
        background = shap.kmeans(X_bg, min(25, len(np.asarray(X_bg))))
        explainer = shap.KernelExplainer(predict_fn, background)
        shap_values = explainer.shap_values(X_ex, nsamples=nsamples_kernel)

    shap_2d = _as_2d(shap_values)
    return np.abs(shap_2d).mean(axis=0).flatten(), feature_names_all


def _load_df_for_artifact(*, pkl_obj: Dict[str, Any], root: Path) -> pd.DataFrame:
    data_source = str(pkl_obj.get("data_source", "")).strip()
    if not data_source:
        raise KeyError("Thiếu 'data_source' trong artifact")
    src_path = (root / data_source).resolve() if not Path(data_source).is_absolute() else Path(data_source)
    if not src_path.exists():
        raise FileNotFoundError(f"Không tìm thấy data_source: {src_path}")
    df = pd.read_csv(src_path)
    if str(pkl_obj.get("task_id", "")).upper() == "P2":
        df = _ensure_sentiment_ohe_p2(df)
    return df


def _build_X_for_pipeline(*, pkl_obj: Dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
    """Build X_raw DataFrame to feed into the pipeline."""
    if str(pkl_obj.get("task_id", "")).upper() == "P2":
        cols = pkl_obj.get("numeric_features")
        if not isinstance(cols, list) or not cols:
            raise KeyError("P2: thiếu 'numeric_features'")
        for c in cols:
            if c not in df.columns:
                df[c] = 0.0
        return df[cols].copy()

    # For sklearn/imblearn pipelines with ColumnTransformer, passing the full df is ok;
    # the preprocessor selects needed columns.
    return df.copy()


def _split_train_test_by_date(
    *,
    df: pd.DataFrame,
    date_col: str,
    split_date: str,
    train_start: Optional[str] = None,
    test_end: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if date_col not in df.columns:
        raise KeyError(f"Không có cột ngày '{date_col}' trong dữ liệu")

    dt = pd.to_datetime(df[date_col], errors="coerce")
    split_dt = pd.to_datetime(split_date)

    mask_train = dt < split_dt
    if train_start:
        mask_train &= dt >= pd.to_datetime(train_start)

    mask_test = dt >= split_dt
    if test_end:
        mask_test &= dt <= pd.to_datetime(test_end)

    return df.loc[mask_train].copy(), df.loc[mask_test].copy()


def run_all() -> None:
    _force_utf8_stdio()
    _maybe_silence_warnings()

    # Keep plots consistent/HD across tasks
    matplotlib.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": DPI,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
        }
    )

    # Repo root (e.g., D:/Hit_songs_DA). NOTE: do NOT use parents[1] here,
    # because that points to the DA/ folder and would produce DA/DA/... paths.
    root = _REPO_ROOT

    # Optional: run a single artifact only.
    # Example (PowerShell):
    #   $env:SHAP_PKL='DA/models/best_model_p0.pkl'; $env:SHAP_N_EXPLAIN='200'; python DA/SHAP_explain/run_shap_all_tasks.py
    one_pkl = os.getenv("SHAP_PKL", "").strip()
    if one_pkl:
        pkl_path = (root / one_pkl).resolve() if not Path(one_pkl).is_absolute() else Path(one_pkl)
        if not pkl_path.exists():
            print(f"Không tìm thấy artifact: {pkl_path}")
            return

        stem = pkl_path.stem.lower()
        if "best_model_p0" in stem:
            task_label = "P0 - Hit Prediction"
            out_beeswarm = root / "DA" / "tasks" / "Hit" / "p0_shap_beeswarm.png"
            out_bar = root / "DA" / "tasks" / "Hit" / "p0_shap_bar.png"
            class_mode = "binary_positive"
        elif "best_model_p2" in stem:
            task_label = "P2 - Style"
            out_beeswarm = root / "DA" / "tasks" / "Style" / "p2_shap_beeswarm.png"
            out_bar = root / "DA" / "tasks" / "Style" / "p2_shap_bar.png"
            class_mode = "binary_positive"
        else:
            task_label = f"Single - {pkl_path.name}"
            out_dir = root / "DA" / "tasks" / "SHAP" / pkl_path.stem
            out_beeswarm = out_dir / "shap_beeswarm.png"
            out_bar = out_dir / "shap_bar.png"
            class_mode = "binary_positive"

        obj = joblib.load(pkl_path)
        pipeline = _get_pipeline_from_artifact(obj)
        if pipeline is None:
            print("  -> Không có pipeline (hoặc thiếu component P2) trong artifact; bỏ qua")
            return

        drift_mode = os.getenv("SHAP_DRIFT", "0") == "1"
        if drift_mode:
            date_col = os.getenv("SHAP_DATE_COL", "spotify_release_date")
            split_date = os.getenv("SHAP_SPLIT_DATE", "2025-02-25")
            train_start = os.getenv("SHAP_TRAIN_START", "") or None
            test_end = os.getenv("SHAP_TEST_END", "2026-03-18")

            n_bg = int(os.getenv("SHAP_N_BACKGROUND", "50"))
            n_ex = int(os.getenv("SHAP_N_EXPLAIN", "250"))
            nsamples_kernel = int(os.getenv("SHAP_NSAMPLES_KERNEL", "300"))

            df_all = _load_df_for_artifact(pkl_obj=obj, root=root)
            df_train, df_test = _split_train_test_by_date(
                df=df_all,
                date_col=date_col,
                split_date=split_date,
                train_start=train_start,
                test_end=test_end,
            )

            if len(df_train) == 0 or len(df_test) == 0:
                print(f"  -> Không đủ dữ liệu train/test sau khi split (train={len(df_train)}, test={len(df_test)})")
                return

            X_train = _build_X_for_pipeline(pkl_obj=obj, df=df_train)
            X_test = _build_X_for_pipeline(pkl_obj=obj, df=df_test)

            X_bg_raw = X_train.sample(n=min(n_bg, len(X_train)), random_state=RANDOM_STATE)
            X_ex_train = X_train.sample(n=min(n_ex, len(X_train)), random_state=RANDOM_STATE)
            X_ex_test = X_test.sample(n=min(n_ex, len(X_test)), random_state=RANDOM_STATE)

            problem_type = _infer_problem_type(obj)
            mean_abs_train, feature_names = _compute_mean_abs_shap_importance(
                pipeline=pipeline,
                X_background_raw=X_bg_raw,
                X_explain_raw=X_ex_train,
                problem_type=problem_type,
                class_mode=class_mode,
                nsamples_kernel=nsamples_kernel,
            )
            mean_abs_test, feature_names2 = _compute_mean_abs_shap_importance(
                pipeline=pipeline,
                X_background_raw=X_bg_raw,
                X_explain_raw=X_ex_test,
                problem_type=problem_type,
                class_mode=class_mode,
                nsamples_kernel=nsamples_kernel,
            )
            if feature_names2 != feature_names:
                feature_names = feature_names

            drift = mean_abs_test - mean_abs_train

            out_dir = out_beeswarm.parent
            out_bar_drift = out_dir / f"{pkl_path.stem}_shap_drift_bar.png"
            out_scatter = out_dir / f"{pkl_path.stem}_shap_drift_scatter.png"

            _save_drift_bar(
                drift=drift,
                feature_names=feature_names,
                out_path=out_bar_drift,
                title=f"Top 20 SHAP Drift (TEST - TRAIN)\n{task_label}",
                top_n=20,
            )
            _save_drift_scatter(
                train_importance=mean_abs_train,
                test_importance=mean_abs_test,
                out_path=out_scatter,
                title=f"So sánh SHAP importance TRAIN vs TEST\n{task_label}",
            )

            order_abs = np.argsort(np.abs(drift))[::-1][:10]
            print("\nTop 10 drift |lớn nhất| (theo |TEST-TRAIN|):")
            for i in order_abs:
                print(f"  - {feature_names[int(i)]}: {float(drift[int(i)]):+.6f}")

            order_inc = np.argsort(drift)[::-1][:10]
            order_dec = np.argsort(drift)[:10]
            print("\nTop 10 tăng importance (drift dương):")
            for i in order_inc:
                print(f"  - {feature_names[int(i)]}: {float(drift[int(i)]):+.6f}")
            print("\nTop 10 giảm importance (drift âm):")
            for i in order_dec:
                print(f"  - {feature_names[int(i)]}: {float(drift[int(i)]):+.6f}")

            print(f"\n✅ Đã lưu drift plots: {out_bar_drift} và {out_scatter}")
            return

        problem_type = _infer_problem_type(obj)

        # Allow overriding sample sizes at runtime.
        n_bg = int(os.getenv("SHAP_N_BACKGROUND", "200"))
        n_ex = int(os.getenv("SHAP_N_EXPLAIN", "200"))
        try:
            X_bg_raw, X_ex_raw = _build_shap_cache_if_missing(pkl_obj=obj, root=root, n_background=n_bg, n_explain=n_ex)
        except Exception as e:
            print(f"  -> {e}")
            return

        nsamples_kernel = int(os.getenv("SHAP_NSAMPLES_KERNEL", "300"))

        _run_shap_for_pipeline(
            pipeline=pipeline,
            X_background_raw=X_bg_raw,
            X_explain_raw=X_ex_raw,
            task_label=task_label,
            out_beeswarm=out_beeswarm,
            out_bar=out_bar,
            problem_type=problem_type,
            class_mode=class_mode,
            nsamples_kernel=nsamples_kernel,
        )
        print(f"✅ Đã lưu tại: {out_beeswarm} và {out_bar}")
        return

    tasks = [
        {
            "task_id": "P0",
            "label": "P0 - Hit Prediction",
            "pkl": root / "DA" / "models" / "best_model_p0.pkl",
            "out_beeswarm": root / "DA" / "tasks" / "Hit" / "p0_shap_beeswarm.png",
            "out_bar": root / "DA" / "tasks" / "Hit" / "p0_shap_bar.png",
            "class_mode": "binary_positive",
        },
        {
            "task_id": "P1",
            "label": "P1 - Popularity Prediction",
            "pkl": root / "DA" / "models" / "best_model_p1.pkl",
            "out_beeswarm": root / "DA" / "tasks" / "Popularity" / "p1_shap_beeswarm.png",
            "out_bar": root / "DA" / "tasks" / "Popularity" / "p1_shap_bar.png",
            "class_mode": "regression",
        },
        {
            "task_id": "P2",
            "label": "P2 - Style",
            "pkl": root / "DA" / "models" / "best_model_p2.pkl",
            "out_beeswarm": root / "DA" / "tasks" / "Style" / "p2_shap_beeswarm.png",
            "out_bar": root / "DA" / "tasks" / "Style" / "p2_shap_bar.png",
            "class_mode": "binary_positive",
        },
        {
            "task_id": "P3",
            "label": "P3 - Sentiment Classification",
            "pkl": root / "DA" / "models" / "best_model_p3.pkl",
            "out_beeswarm": root / "DA" / "tasks" / "Sentiment" / "p3_shap_beeswarm.png",
            "out_bar": root / "DA" / "tasks" / "Sentiment" / "p3_shap_bar.png",
            "class_mode": "class_1",  # match existing scripts: neutral class for beeswarm
        },
        {
            "task_id": "P4",
            "label": "P4 - Genre Multi-Label",
            "pkl": root / "DA" / "models" / "best_model_p4.pkl",
            "out_dir": root / "DA" / "tasks" / "Genres" / "04_explainable",
        },
    ]

    print("=" * 80)
    print("CHẠY SHAP (standalone) - tất cả tasks")
    print("=" * 80)

    # Optional task filter: SHAP_TASKS="P0,P2" to run only selected tasks.
    only = os.getenv("SHAP_TASKS", "").strip()
    only_set = {x.strip().upper() for x in only.split(",") if x.strip()} if only else None

    for t in tasks:
        if only_set is not None and t.get("task_id") not in only_set:
            continue
        if t.get("skip"):
            print(f"\n[{t['task_id']}] {t['label']} → skipped")
            continue

        pkl_path: Path = t["pkl"]
        if not pkl_path.exists():
            print(f"\n[{t['task_id']}] Không tìm thấy artifact: {pkl_path}")
            continue

        print(f"\n[{t['task_id']}] Đang load: {pkl_path}")
        obj = joblib.load(pkl_path)

        drift_mode = os.getenv("SHAP_DRIFT", "0") == "1"

        if t["task_id"] == "P4":
            if drift_mode:
                print("  -> SHAP_DRIFT hiện chưa hỗ trợ P4 (multi-label); bỏ qua drift")

            pipeline = obj.get("pipeline")
            if pipeline is None:
                print("  -> Không có pipeline trong artifact; bỏ qua")
                continue

            try:
                X_bg_raw, X_ex_raw = _extract_shap_cache(obj)
            except Exception as e:
                print(f"  -> {e}")
                continue

            # Multi-label: run each estimator separately
            labels = obj.get("labels") or obj.get("classes") or obj.get("label_names")
            if not labels:
                print("  -> Thiếu danh sách nhãn (labels) trong artifact; bỏ qua")
                continue

            out_dir: Path = t["out_dir"]
            out_dir.mkdir(parents=True, exist_ok=True)

            estimator_multi = pipeline.steps[-1][1]
            estimators = getattr(estimator_multi, "estimators_", None)
            if estimators is None:
                print("  -> Cần MultiOutputClassifier (có estimators_); bỏ qua")
                continue

            # Transform once
            X_bg = _to_dense(transform_for_model(pipeline, X_bg_raw))
            X_ex = _to_dense(transform_for_model(pipeline, X_ex_raw))
            feature_names_all = _pick_feature_names(pipeline, int(X_ex.shape[1]))

            import shap

            for idx, label_name in enumerate(labels):
                safe = _safe_filename(label_name)
                out_bee = out_dir / f"p4_shap_beeswarm_{safe}.png"
                out_bar = out_dir / f"p4_shap_bar_{safe}.png"

                clf = estimators[idx]
                if not hasattr(clf, "predict_proba"):
                    print(f"  -> [{label_name}] estimator không có predict_proba; bỏ qua")
                    continue

                try:
                    shap_values = None
                    if _is_tree_model(clf):
                        try:
                            explainer = shap.TreeExplainer(clf, data=X_bg)
                            # Avoid occasional numeric/additivity failures.
                            shap_values = explainer.shap_values(X_ex, check_additivity=False)
                        except Exception:
                            shap_values = None

                    if shap_values is None:
                        np.random.seed(RANDOM_STATE)
                        background = shap.kmeans(X_bg, 25)
                        explainer = shap.KernelExplainer(clf.predict_proba, background)
                        shap_values = explainer.shap_values(X_ex, nsamples=300)

                    if isinstance(shap_values, list) and len(shap_values) >= 2:
                        shap_2d = np.asarray(shap_values[1])
                    else:
                        shap_2d = np.asarray(shap_values)

                    if shap_2d.ndim == 3 and shap_2d.shape[-1] >= 2:
                        shap_2d = shap_2d[:, :, 1]

                    top_idx = _compute_top_indices(shap_2d, TOP_N)
                    shap_top = shap_2d[:, top_idx]
                    X_ex_top = np.asarray(X_ex)[:, top_idx]
                    feat_top = [feature_names_all[int(i)] for i in top_idx]

                    _save_beeswarm(
                        shap_top,
                        X_ex_top,
                        feat_top,
                        title=f"Top {len(feat_top)} Features - SHAP (Beeswarm)\nP4 - {label_name}",
                        out_path=out_bee,
                    )
                    _save_bar(
                        shap_top,
                        feat_top,
                        title=f"Top {len(feat_top)} Features - SHAP (Bar)\nP4 - {label_name}",
                        out_path=out_bar,
                    )

                    print(f"  -> [{label_name}] đã lưu: {out_bee.name}, {out_bar.name}")
                except Exception as e:
                    print(f"  -> [{label_name}] SHAP thất bại: {e}")

            continue

        # P0/P1/P2/P3
        pipeline = _get_pipeline_from_artifact(obj)
        if pipeline is None:
            print("  -> Không có pipeline trong artifact; bỏ qua")
            continue

        if drift_mode:
            date_col = os.getenv("SHAP_DATE_COL", "spotify_release_date")
            split_date = os.getenv("SHAP_SPLIT_DATE", "2025-02-25")
            train_start = os.getenv("SHAP_TRAIN_START", "") or None
            test_end = os.getenv("SHAP_TEST_END", "2026-03-18")

            n_bg = int(os.getenv("SHAP_N_BACKGROUND", "50"))
            n_ex = int(os.getenv("SHAP_N_EXPLAIN", "250"))
            nsamples_kernel = int(os.getenv("SHAP_NSAMPLES_KERNEL", "300"))

            try:
                df_all = _load_df_for_artifact(pkl_obj=obj, root=root)
                df_train, df_test = _split_train_test_by_date(
                    df=df_all,
                    date_col=date_col,
                    split_date=split_date,
                    train_start=train_start,
                    test_end=test_end,
                )
                if len(df_train) == 0 or len(df_test) == 0:
                    print(f"  -> Không đủ dữ liệu train/test sau khi split (train={len(df_train)}, test={len(df_test)})")
                    continue

                X_train = _build_X_for_pipeline(pkl_obj=obj, df=df_train)
                X_test = _build_X_for_pipeline(pkl_obj=obj, df=df_test)

                X_bg_raw = X_train.sample(n=min(n_bg, len(X_train)), random_state=RANDOM_STATE)
                X_ex_train = X_train.sample(n=min(n_ex, len(X_train)), random_state=RANDOM_STATE)
                X_ex_test = X_test.sample(n=min(n_ex, len(X_test)), random_state=RANDOM_STATE)

                problem_type = _infer_problem_type(obj)
                if problem_type == "regression":
                    class_mode = "regression"
                else:
                    class_mode = t.get("class_mode", "binary_positive")

                mean_abs_train, feature_names = _compute_mean_abs_shap_importance(
                    pipeline=pipeline,
                    X_background_raw=X_bg_raw,
                    X_explain_raw=X_ex_train,
                    problem_type=problem_type,
                    class_mode=class_mode,
                    nsamples_kernel=nsamples_kernel,
                )
                mean_abs_test, feature_names2 = _compute_mean_abs_shap_importance(
                    pipeline=pipeline,
                    X_background_raw=X_bg_raw,
                    X_explain_raw=X_ex_test,
                    problem_type=problem_type,
                    class_mode=class_mode,
                    nsamples_kernel=nsamples_kernel,
                )
                if feature_names2 != feature_names:
                    feature_names = feature_names

                drift = mean_abs_test - mean_abs_train

                out_dir = Path(t["out_beeswarm"]).parent
                out_bar_drift = out_dir / f"{t['task_id'].lower()}_shap_drift_bar.png"
                out_scatter = out_dir / f"{t['task_id'].lower()}_shap_drift_scatter.png"

                _save_drift_bar(
                    drift=drift,
                    feature_names=feature_names,
                    out_path=out_bar_drift,
                    title=f"Top 20 SHAP Drift (TEST - TRAIN)\n{t['label']}",
                    top_n=20,
                )
                _save_drift_scatter(
                    train_importance=mean_abs_train,
                    test_importance=mean_abs_test,
                    out_path=out_scatter,
                    title=f"So sánh SHAP importance TRAIN vs TEST\n{t['label']}",
                )

                order_abs = np.argsort(np.abs(drift))[::-1][:10]
                print("  Top 10 drift |lớn nhất| (|TEST-TRAIN|):")
                for i in order_abs:
                    print(f"    - {feature_names[int(i)]}: {float(drift[int(i)]):+.6f}")

                print(f"  ✅ Đã lưu drift plots: {out_bar_drift} và {out_scatter}")
            except Exception as e:
                print(f"  -> Drift thất bại: {e}")
            continue

        # Allow overriding sample sizes at runtime.
        n_bg = int(os.getenv("SHAP_N_BACKGROUND", "200"))
        n_ex = int(os.getenv("SHAP_N_EXPLAIN", "200"))
        try:
            X_bg_raw, X_ex_raw = _build_shap_cache_if_missing(pkl_obj=obj, root=root, n_background=n_bg, n_explain=n_ex)
        except Exception as e:
            print(f"  -> Lỗi: {e}")
            continue

        problem_type = _infer_problem_type(obj)
        if problem_type == "regression":
            class_mode = "regression"
        else:
            class_mode = t.get("class_mode", "binary_positive")

        try:
            nsamples_kernel = int(os.getenv("SHAP_NSAMPLES_KERNEL", "300"))
            _run_shap_for_pipeline(
                pipeline=pipeline,
                X_background_raw=X_bg_raw,
                X_explain_raw=X_ex_raw,
                task_label=t["label"],
                out_beeswarm=t["out_beeswarm"],
                out_bar=t["out_bar"],
                problem_type=problem_type,
                class_mode=class_mode,
                nsamples_kernel=nsamples_kernel,
            )
            print(f"  ✅ Đã lưu tại {t['out_beeswarm']} và {t['out_bar']}")
        except Exception as e:
            print(f"  -> SHAP thất bại: {e}")


if __name__ == "__main__":
    run_all()
