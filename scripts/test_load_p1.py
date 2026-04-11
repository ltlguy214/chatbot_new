from __future__ import annotations

import os
import sys

import joblib


_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def main() -> int:
    candidates = [
        os.path.join(_ROOT, '.model_cache', 'best_model_p1_compressed.pkl'),
        os.path.join(_ROOT, 'DA', 'models', 'best_model_p1_compressed.pkl'),
        os.path.join(_ROOT, 'DA', 'models', 'best_model_p1.pkl'),
    ]

    ok = False
    for p in candidates:
        print('PATH', p)
        if not os.path.exists(p):
            print('  MISSING')
            continue
        try:
            obj = joblib.load(p)
            if isinstance(obj, dict):
                print('  LOAD_OK', 'dict', 'keys=', list(obj.keys())[:8])
            else:
                print('  LOAD_OK', type(obj).__name__)
            ok = True
            break
        except Exception as ex:
            msg = str(ex)
            detail = f"{type(ex).__name__}: {msg}" if msg else type(ex).__name__
            print('  LOAD_FAIL', detail[:400])

    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
