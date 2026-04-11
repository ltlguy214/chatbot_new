from __future__ import annotations

from typing import Any


def sparse_to_dense(X: Any):
    """Convert sparse matrices to dense arrays for downstream estimators.

    Intended for use with scikit-learn's FunctionTransformer inside a Pipeline.
    Must live at module top-level so joblib/pickle can serialize the pipeline.

    - If X has .toarray() (scipy sparse), returns X.toarray().
    - Otherwise returns X unchanged.
    """

    try:
        toarray = getattr(X, "toarray", None)
        if callable(toarray):
            return toarray()
    except Exception:
        pass
    return X
