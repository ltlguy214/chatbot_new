from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ShapCacheConfig:
    n_background: int = 200
    n_explain: int = 200
    random_state: int = 42


def _sample_df(df: pd.DataFrame, n: int, random_state: int) -> pd.DataFrame:
    if n <= 0:
        return df.iloc[0:0].copy()
    if len(df) <= n:
        return df.copy()
    return df.sample(n=n, random_state=random_state)


def build_shap_cache(
    X_train: pd.DataFrame,
    X_explain_source: Optional[pd.DataFrame] = None,
    *,
    config: ShapCacheConfig = ShapCacheConfig(),
) -> Dict[str, Any]:
    """Build a tiny, SHAP-ready cache to embed into a .pkl artifact.

    Stores small raw samples only (DataFrame) so SHAP can be run later without retraining.
    """

    if not isinstance(X_train, pd.DataFrame):
        raise TypeError("build_shap_cache expects X_train as a pandas DataFrame")

    X_background_raw = _sample_df(X_train, config.n_background, config.random_state)

    if X_explain_source is None:
        X_explain_source = X_train
    if not isinstance(X_explain_source, pd.DataFrame):
        raise TypeError("build_shap_cache expects X_explain_source as a pandas DataFrame")

    X_explain_raw = _sample_df(X_explain_source, config.n_explain, config.random_state)

    return {
        "n_background": int(len(X_background_raw)),
        "n_explain": int(len(X_explain_raw)),
        "random_state": int(config.random_state),
        "X_background_raw": X_background_raw,
        "X_explain_raw": X_explain_raw,
    }


def try_get_feature_names(pipeline) -> Optional[np.ndarray]:
    """Best-effort extraction of feature names after all transforms (excluding estimator).

    Handles common imblearn pipelines that contain samplers (fit_resample) between
    transformer steps (e.g., SelectFromModel), where Pipeline.get_feature_names_out
    is not available.

    Returns None if names cannot be inferred.
    """

    steps = getattr(pipeline, "steps", None)
    if not steps:
        return None

    # 1) Seed names from a ColumnTransformer-like preprocessor.
    names: Optional[np.ndarray] = None
    named_steps = getattr(pipeline, "named_steps", None)
    pre = None
    if isinstance(named_steps, dict):
        pre = named_steps.get("preprocessor")
    if pre is None:
        pre = steps[0][1]

    getter = getattr(pre, "get_feature_names_out", None)
    if getter is not None:
        try:
            names = np.asarray(getter())
        except Exception:
            names = None

    if names is None:
        return None

    # 2) Walk subsequent transform steps and update names when possible.
    for _, step in steps[1:-1]:
        if step in (None, "passthrough"):
            continue

        # Skip samplers (imblearn) that alter rows, not columns.
        if hasattr(step, "fit_resample") and not hasattr(step, "transform"):
            continue

        # Preferred: sklearn transformers that support get_feature_names_out.
        step_getter = getattr(step, "get_feature_names_out", None)
        if step_getter is not None:
            try:
                names = np.asarray(step_getter(names))
                continue
            except Exception:
                pass

        # Common selectors (SelectFromModel, VarianceThreshold, SelectKBest, ...)
        # expose get_support(). Use it to subset names.
        support_getter = getattr(step, "get_support", None)
        if support_getter is not None:
            try:
                mask = np.asarray(support_getter(), dtype=bool)
                if mask.ndim == 1 and len(mask) == len(names):
                    names = names[mask]
            except Exception:
                pass

    return np.asarray(names)


def transform_for_model(pipeline, X_raw: pd.DataFrame):
    """Apply fitted pipeline transforms (excluding the final estimator) to X_raw.

    Notes:
    - imblearn Pipelines often contain samplers (fit_resample) that don't implement
      transform(). For SHAP we want the *feature space* of the estimator, so we
      skip samplers and keep applying transformer steps (e.g., SelectFromModel).
    """

    X = X_raw
    steps = getattr(pipeline, "steps", None)
    if not steps:
        # Fallback: behave like sklearn Pipeline
        return pipeline[:-1].transform(X_raw)

    for _, step in steps[:-1]:
        if step in (None, "passthrough"):
            continue

        # Skip imblearn samplers (row resampling) during SHAP transforms.
        if hasattr(step, "fit_resample") and not hasattr(step, "transform"):
            continue

        transformer = getattr(step, "transform", None)
        if transformer is None:
            # No-op for steps that don't transform features.
            continue

        X = transformer(X)

    return X
