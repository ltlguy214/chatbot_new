from __future__ import annotations

from typing import Any

import os
import re

import numpy as np


# 1) Stopwords: copied from user's notebook (with small extension in-code)
_VI_STOPWORDS: list[str] = [
    "anh", "em", "là", "có", "không", "một", "cho", "đã", "đang", "sẽ", "cũng",
    "như", "những", "này", "với", "đến", "trong", "để", "của", "mà", "thì",
    "còn", "lại", "nào", "vậy", "đâu", "thế", "cứ", "vẫn", "người", "ta", "mình",
    "rồi", "lúc", "khi", "ngày", "đêm", "nơi", "này", "ấy", "ô", "oh", "yeah", "la",
    "verse", "chorus", "pre", "bridge", "intro", "outro",
]

_EXT_STOPWORDS = _VI_STOPWORDS + ["đi", "và", "nhau", "ai", "chỉ", "vì", "phải", "chẳng", "được", "làm"]

_RE_TAGS = re.compile(r"\[.*?\]")
_RE_KEEP_VI = re.compile(
    r"[^a-záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ\s]",
    flags=re.IGNORECASE,
)


def clean_vietnamese_text_for_topic(text: str) -> str:
    s = str(text or "").lower()
    s = _RE_TAGS.sub(" ", s)
    s = _RE_KEEP_VI.sub(" ", s)

    try:
        from pyvi import ViTokenizer

        s = ViTokenizer.tokenize(s)
    except Exception:
        pass

    return " ".join(s.split())


_TOPIC_MODEL = None


def _load_topic_model():
    global _TOPIC_MODEL
    if _TOPIC_MODEL is not None:
        return _TOPIC_MODEL

    model_path = str(os.getenv("BERTOPIC_MODEL_PATH") or "models/vpop_bertopic_model").strip()
    if not model_path:
        return None

    if not os.path.exists(model_path):
        return None

    try:
        from bertopic import BERTopic

        _TOPIC_MODEL = BERTopic.load(model_path)
        return _TOPIC_MODEL
    except Exception:
        return None


def extract_topic_features(lyric_text: str, *, n_topics: int = 16) -> tuple[dict[str, float], str, dict[str, Any]]:
    """Return (topic_prob_features, main_topic_label, meta).

    - Uses BERTopic.transform on a pre-trained model loaded from `BERTOPIC_MODEL_PATH`.
    - If the model is missing, we still attempt a lightweight keyword-based main topic label
      so the UI/web payload has a non-empty `main_topic`.
    """

    probs_out = {f"topic_prob_{i}": 0.0 for i in range(int(n_topics))}

    clean = clean_vietnamese_text_for_topic(lyric_text)
    if not clean:
        return probs_out, "", {"status": "empty-lyric"}

    model = _load_topic_model()
    if model is None:
        # Lightweight fallback: infer a coarse topic label from keywords.
        chosen_key = ""
        s = str(clean).lower()

        def _has(*terms: str) -> bool:
            return any(t and (t in s) for t in terms)

        # Heuristics roughly aligned with DA.models.topic_mapping.TOPIC_MAPPING
        if _has("tết", "tet", "mùa xuân", "mua_xuan", "xuân", "xuan"):
            chosen_key = "topic_prob_7"
        elif _has("mẹ", "me", "cha", "ba", "bố", "bo", "gia đình", "gia_dinh"):
            chosen_key = "topic_prob_6"
        elif _has(
            "chia tay",
            "chia_tay",
            "thất tình",
            "that_tinh",
            "rời xa",
            "roi_xa",
            "nước mắt",
            "nuoc_mat",
            "đau",
            "dau",
            "lụy",
            "luy",
            "cô đơn",
            "co_don",
        ):
            chosen_key = "topic_prob_0"
        elif _has("rap", "hiphop", "hip-hop", "trap", "flow", "hustle"):
            chosen_key = "topic_prob_2"
        elif _has("kỷ niệm", "ky_niem", "hoài niệm", "hoai_niem", "quá khứ", "qua_khu"):
            chosen_key = "topic_prob_8"
        elif _has("việt nam", "viet_nam", "tổ quốc", "to_quoc", "tự hào", "tu_hao"):
            chosen_key = "topic_prob_9"
        else:
            # If there's lots of English tokens, treat as English-leaning topic.
            try:
                ascii_letters = sum(1 for ch in s if "a" <= ch <= "z")
                total = max(1, len(s))
                if float(ascii_letters) / float(total) > 0.25:
                    chosen_key = "topic_prob_5"
            except Exception:
                chosen_key = ""

        main_topic_label = ""
        try:
            from DA.models.topic_mapping import TOPIC_MAPPING

            if chosen_key:
                main_topic_label = str(TOPIC_MAPPING.get(chosen_key, "") or "")
        except Exception:
            main_topic_label = ""

        if not main_topic_label:
            main_topic_label = "Chưa xác định"

        return (
            probs_out,
            main_topic_label,
            {
                "status": "fallback-rule:model-missing",
                "model_path": str(os.getenv("BERTOPIC_MODEL_PATH") or "models/vpop_bertopic_model"),
                "chosen_key": chosen_key,
            },
        )

    try:
        _topics, probs = model.transform([clean])
        if probs is None or len(probs) == 0:
            return probs_out, "", {"status": "no-probs"}

        arr = np.asarray(probs[0], dtype=float).reshape(-1)
        for i in range(min(int(n_topics), int(arr.shape[0]))):
            probs_out[f"topic_prob_{i}"] = round(float(arr[i]), 6)

        top_idx = int(np.argmax(arr[: int(min(int(n_topics), int(arr.shape[0])))])) if arr.size else -1
        main_topic_key = f"topic_prob_{top_idx}" if top_idx >= 0 else ""

        main_topic_label = ""
        try:
            from DA.models.topic_mapping import TOPIC_MAPPING

            main_topic_label = str(TOPIC_MAPPING.get(main_topic_key, "") or "")
        except Exception:
            main_topic_label = ""

        return probs_out, main_topic_label, {"status": "ok", "top_idx": top_idx, "top_key": main_topic_key}

    except Exception as ex:
        return probs_out, "", {"status": f"error:{type(ex).__name__}", "detail": str(ex)}
