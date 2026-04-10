from __future__ import annotations


def format_int_vi(value: int | None) -> str:
    if value is None:
        return ''
    try:
        return f"{int(value):,}".replace(',', '.')
    except Exception:
        return str(value)


def format_duration_ms(duration_ms: int | None) -> str:
    if duration_ms is None:
        return ''
    try:
        total_seconds = int(duration_ms) // 1000
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        return f"{minutes:d}:{seconds:02d}"
    except Exception:
        return ''
