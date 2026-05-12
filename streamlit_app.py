"""Streamlit Cloud entrypoint.

Why:
- Streamlit Cloud expects a single app entry script at repo root.
- This project reads configuration from environment variables / .env.
- On Streamlit Cloud, secrets are provided via `st.secrets`; we map them into
  `os.environ` (best-effort, without overwriting existing env vars).

Run:
  streamlit run streamlit_app.py
"""

from __future__ import annotations

import os
import runpy
from pathlib import Path

import streamlit as st


def _inject_streamlit_secrets_into_env() -> None:
    """Copy `st.secrets` values into `os.environ` if missing.

    Supports both flat secrets and one-level nested sections.
    """

    def _set_if_missing(key: str, value: object) -> None:
        key = str(key or "").strip()
        if not key:
            return
        if key in os.environ and str(os.environ.get(key) or "").strip() != "":
            return

        if isinstance(value, (str, int, float, bool)):
            os.environ[key] = str(value)
        else:
            # Ignore non-primitive types.
            return

    try:
        for key, value in st.secrets.items():
            if isinstance(value, dict):
                for sub_key, sub_val in value.items():
                    _set_if_missing(str(sub_key), sub_val)
            else:
                _set_if_missing(str(key), value)
    except Exception:
        return


_inject_streamlit_secrets_into_env()

# Execute the original Streamlit script with correct __file__/cwd semantics.
app_path = Path(__file__).resolve().parent / "chatbot" / "app_chatbot.py"
runpy.run_path(str(app_path), run_name="__main__")
