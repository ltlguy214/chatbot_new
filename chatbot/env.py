from __future__ import annotations

import os
from pathlib import Path


def _find_dotenv_path(start: Path) -> Path | None:
    cur = start
    for _ in range(8):
        candidate = cur / '.env'
        if candidate.exists():
            return candidate
        if cur.parent == cur:
            break
        cur = cur.parent
    return None


class EnvLoader:
    def __init__(self, *, max_depth: int = 8) -> None:
        self.max_depth = int(max_depth)

    def _find_dotenv_path(self, start: Path) -> Path | None:
        cur = start
        for _ in range(max(1, self.max_depth)):
            candidate = cur / '.env'
            if candidate.exists():
                return candidate
            if cur.parent == cur:
                break
            cur = cur.parent
        return None

    def load(self) -> None:
        """Best-effort .env loader.

        Streamlit is often launched from different working directories; this tries:
        - current working directory and its parents
        - this file's directory and its parents

        If python-dotenv is missing, falls back to a minimal KEY=VALUE parser.
        """

        dotenv_path: Path | None = None
        try:
            cwd = Path.cwd()
            dotenv_path = self._find_dotenv_path(cwd)
            if dotenv_path is None:
                dotenv_path = self._find_dotenv_path(Path(__file__).resolve().parent)

            if dotenv_path is None:
                return

            # Expose which .env file was used for debugging (without printing secrets).
            try:
                os.environ.setdefault('DOTENV_PATH_USED', str(dotenv_path))
            except Exception:
                pass

            def _postprocess_multiline_values(path: Path) -> None:
                """Support non-standard multi-line values used in this repo's .env.

                Example pattern:
                  GEMINI_API_KEYS=
                  key1
                  key2

                `python-dotenv` ignores continuation lines; we reconstruct them here
                but only when the env var is currently missing/empty.
                """

                try:
                    if str(os.getenv('GEMINI_API_KEYS') or '').strip():
                        return

                    lines = path.read_text(encoding='utf-8').splitlines()
                    for idx, raw in enumerate(lines):
                        line = raw.strip()
                        if not line or line.startswith('#'):
                            continue
                        if not line.startswith('GEMINI_API_KEYS'):
                            continue
                        if '=' not in line:
                            continue

                        _, value = line.split('=', 1)
                        if str(value or '').strip():
                            return

                        keys: list[str] = []
                        for j in range(idx + 1, len(lines)):
                            nxt = lines[j].strip()
                            if not nxt or nxt.startswith('#') or '=' in nxt:
                                break
                            keys.append(nxt)

                        if keys:
                            os.environ['GEMINI_API_KEYS'] = "\n".join(keys)
                        return
                except Exception:
                    return

            try:
                from dotenv import load_dotenv  # type: ignore

                load_dotenv(dotenv_path=str(dotenv_path), override=False)
                _postprocess_multiline_values(dotenv_path)
                return
            except Exception:
                pass

            # Minimal parser fallback (KEY=VALUE per line).
            for line in dotenv_path.read_text(encoding='utf-8').splitlines():
                line = line.strip()
                if not line or line.startswith('#') or '=' not in line:
                    continue
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value

            _postprocess_multiline_values(dotenv_path)
        except Exception:
            return


def load_env() -> None:
    EnvLoader().load()
