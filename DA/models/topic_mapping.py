"""Topic name mapping helpers for report/plots.

This project produces topic probability features like `topic_prob_0` from BERTopic + PhoBERT.
When plotting Feature Importance / SHAP, these raw names are hard to read.

Use:
- `rename_topics_for_report(df, column_name='Feature')`
- `rename_topics_in_feature_names(feature_names)`
"""

from __future__ import annotations

import re
from typing import Iterable, List

# Based on latest BERTopic + PhoBERT labeling provided by the user.
TOPIC_MAPPING: dict[str, str] = {
    "topic_prob_-1": "Topic -1: Nhiễu / Không phân loại",
    "topic_prob_0": "Topic 0: Ballad Thất tình / Chia tay",
    "topic_prob_1": "Topic 1: Tình yêu đôi lứa / Lãng mạn",
    "topic_prob_2": "Topic 2: Rap Hiphop Gai góc",
    "topic_prob_3": "Topic 3: Rap Đời sống / Hustle",
    "topic_prob_4": "Topic 4: Nhạc Trữ tình / Hoài cổ",
    "topic_prob_5": "Topic 5: Pop / R&B Âu Mỹ (English)",
    "topic_prob_6": "Topic 6: Tình cảm Gia đình / Cha mẹ",
    "topic_prob_7": "Topic 7: Nhạc Tết / Xuân",
    "topic_prob_8": "Topic 8: Hoài niệm / Kỷ niệm",
    "topic_prob_9": "Topic 9: Lòng yêu nước / Tự hào",
    "topic_prob_10": "Topic 10: Pop hiện đại / Thả thính",
    "topic_prob_11": "Topic 11: Cấu trúc Anh ngữ bổ trợ",
    "topic_prob_12": "Topic 12: Ad-libs / Biểu cảm",
    "topic_prob_13": "Topic 13: Tâm sự / Tỏ tình trực tiếp",
    "topic_prob_14": "Topic 14: Ngôn ngữ ngoại lai (Latin/Pháp)",
}

_TOPIC_PATTERN = re.compile(r"topic_prob_-?\d+")


def rename_topic_string(name: str) -> str:
    """Replace any `topic_prob_*` occurrence inside a string with a human label."""

    if not isinstance(name, str) or "topic_prob_" not in name:
        return name

    def _repl(match: re.Match[str]) -> str:
        key = match.group(0)
        mapped = TOPIC_MAPPING.get(key)
        if mapped is not None:
            return mapped

        # Fallback: keep a readable generic label if a new topic id appears.
        # Example: topic_prob_15 -> Topic 15
        try:
            idx = int(key.replace("topic_prob_", ""))
            return f"Topic {idx}"
        except Exception:
            return key

    return _TOPIC_PATTERN.sub(_repl, name)


def rename_topics_in_feature_names(feature_names: Iterable[str]) -> List[str]:
    """Map topic features in a list of feature names."""

    return [rename_topic_string(str(n)) for n in feature_names]


def rename_topics_for_report(df, column_name: str = "Feature"):
    """In-place rename topic feature labels in a DataFrame column.

    Works for exact matches (topic_prob_0) and for transformer-prefixed names
    (e.g., num__topic_prob_0).
    """

    if df is None or column_name not in df.columns:
        return df

    df[column_name] = df[column_name].astype(str).map(rename_topic_string)
    return df
