from __future__ import annotations

import json
from collections.abc import Callable
from functools import partial
from typing import Any

import numpy as np


def _ratio_sampling_strategy(y: Any, *, class_ids: list[int], ratios: list[float]) -> dict[int, int]:
    """Compute RandomUnderSampler sampling_strategy dict from desired class ratios.

    This is designed for *under-sampling* in multi-class single-label problems.

    Let base = min(counts) over classes present in the fold (non-zero counts).
    Target count per class k is:
        target_k = min(count_k, round(ratios[k] * base))

    Missing classes (count=0) are omitted.
    """

    y_arr = np.asarray(y)
    if y_arr.ndim != 1:
        y_arr = y_arr.reshape(-1)

    class_ids = [int(c) for c in class_ids]
    ratios = [float(r) for r in ratios]
    if len(ratios) != len(class_ids):
        raise ValueError(f"ratios length {len(ratios)} must equal class_ids length {len(class_ids)}")

    # Counts for classes present in this fold
    counts = {c: int(np.sum(y_arr == c)) for c in class_ids}
    present = [c for c, n in counts.items() if n > 0]
    if not present:
        return {}

    base = min(counts[c] for c in present)
    base = int(max(1, base))

    strategy: dict[int, int] = {}
    for i, c in enumerate(class_ids):
        n = counts.get(c, 0)
        if n <= 0:
            continue
        target = int(round(float(ratios[i]) * float(base)))
        target = max(1, min(int(n), target))
        strategy[int(c)] = int(target)

    return strategy


def make_ratio_sampling_strategy(*, class_ids: list[int], ratios: list[float]) -> Callable[[Any], dict[int, int]]:
    """Return a picklable callable for imblearn's sampling_strategy."""

    return partial(_ratio_sampling_strategy, class_ids=list(class_ids), ratios=list(ratios))


def parse_ratios_spec(
    spec: str,
    *,
    class_names: list[str] | None = None,
    class_ids: list[int] | None = None,
) -> list[float] | None:
    """Parse ratio spec into a list of floats.

    Supported formats:
    - "1.2,1.2,1" (positional)
    - "Negative=1.2,Neutral=1.2,Positive=1" (by name)
    - JSON dict: '{"0": 1.2, "1": 1.2, "2": 1}' (by id)

    Returns None if spec is empty.
    """

    if spec is None:
        return None
    raw = str(spec).strip()
    if not raw:
        return None

    raw2 = raw
    for pfx in ("ratio:", "ratios:"):
        if raw2.lower().startswith(pfx):
            raw2 = raw2[len(pfx) :].strip()
            break

    class_names = list(class_names) if class_names is not None else []
    class_ids = [int(x) for x in (class_ids or [])]

    # JSON dict
    if raw2.startswith("{") and raw2.endswith("}"):
        obj = json.loads(raw2)
        if not isinstance(obj, dict):
            raise ValueError("JSON ratios spec must be a dict")

        if class_ids:
            out = []
            for cid in class_ids:
                v = obj.get(str(cid), obj.get(cid))
                if v is None and class_names and 0 <= class_ids.index(cid) < len(class_names):
                    v = obj.get(class_names[class_ids.index(cid)])
                if v is None:
                    raise ValueError(f"Missing ratio for class id {cid}")
                out.append(float(v))
            return out

        # If ids not provided, just return values in sorted key order.
        keys = sorted(obj.keys(), key=lambda k: int(k) if str(k).lstrip("-").isdigit() else str(k))
        return [float(obj[k]) for k in keys]

    # Name=value pairs
    if "=" in raw2 and class_names:
        pairs = [p.strip() for p in raw2.split(",") if p.strip()]
        mapping: dict[str, float] = {}
        for p in pairs:
            if "=" not in p:
                continue
            k, v = p.split("=", 1)
            mapping[k.strip()] = float(v.strip())
        out = []
        for nm in class_names:
            if nm not in mapping:
                raise ValueError(f"Missing ratio for class name '{nm}'")
            out.append(float(mapping[nm]))
        return out

    # Positional list
    sep = ","
    if "," not in raw2 and ":" in raw2:
        # Allow '1.2:1.2:1.0' style
        sep = ":"
    parts = [p.strip() for p in raw2.split(sep) if p.strip()]
    vals = [float(p) for p in parts]
    return vals
