from __future__ import annotations

from typing import Any

import os
import re

import numpy as np


_RE_KEEP_VI_NL = re.compile(
    r"[^a-záàảãạăắằẳẵặâấầẩẫậèéèẻẽẹêềếểễệóòỏõọôồốổỗộơờớởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ\n ]",
    flags=re.IGNORECASE,
)


def _extract_lyric_features(text: str) -> tuple[dict[str, Any], str]:
    """Mirror `extract_lyric_features` from nlp_colab.py.

    Returns: (features, lyric_clean_preserve_newlines)
    """

    raw = str(text or "").lower()
    text_clean = _RE_KEEP_VI_NL.sub(" ", raw)

    valid_lines = [l.strip() for l in text_clean.split("\n") if l.strip()]
    lyric_clean = "\n".join(valid_lines)
    text_for_nlp = lyric_clean.replace("\n", " ")

    try:
        from underthesea import word_tokenize, pos_tag

        tokens = word_tokenize(text_for_nlp)
        tags = pos_tag(text_for_nlp)
    except Exception:
        tokens = text_for_nlp.split()
        tags = []

    noun = sum(1 for _w, t in tags if t in ["N", "Np", "Nc", "Nu"])
    verb = sum(1 for _w, t in tags if t in ["V", "Vy"])
    adj = sum(1 for _w, t in tags if t in ["A", "Adj"])

    feats = {
        "lyric_total_words": int(len(tokens)),
        "lexical_diversity": round(float(len(set(tokens)) / len(tokens)), 4) if tokens else 0,
        "noun_count": int(noun),
        "verb_count": int(verb),
        "adj_count": int(adj),
    }
    return feats, lyric_clean


def _pos_counts(text_clean: str) -> tuple[int, float, int, int, int]:
    """Return (total_words, lexical_diversity, noun_count, verb_count, adj_count)."""

    if not text_clean:
        return 0, 0.0, 0, 0, 0

    try:
        import underthesea

        tokens = underthesea.word_tokenize(text_clean)
        tags = underthesea.pos_tag(text_clean)
    except Exception:
        tokens = text_clean.split()
        tags = []

    total = int(len(tokens))
    lex_div = float(len(set(tokens)) / total) if total else 0.0

    noun = sum(1 for _w, t in tags if t in ["N", "Np", "Nc", "Nu"])
    verb = sum(1 for _w, t in tags if t in ["V", "Vy"])
    adj = sum(1 for _w, t in tags if t in ["A", "Adj"])

    return total, round(lex_div, 4), int(noun), int(verb), int(adj)


def _analyze_lexicon(text_clean: str) -> tuple[str, float]:
    """Mirror `analyze_lexicon` from nlp_colab.py."""

    if not text_clean or len(str(text_clean).strip()) == 0:
        return "neutral", 0.0

    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        vader_analyzer = SentimentIntensityAnalyzer()
        vader_score = float(vader_analyzer.polarity_scores(text_clean).get("compound", 0.0))
    except Exception:
        vader_score = 0.0

    positive_words = [
        'vui_vẻ', 'hạnh_phúc', 'phấn_khởi',
        'yêu', 'thích', 'vui', 'đẹp', 'tuyệt', 'mơ', 'ước', 'cười',
        'rạng_rỡ', 'ngọt_ngào', 'nắng', 'xuân', 'hứng_khởi', 'rộn_ràng', 'tươi',
        'sáng', 'tình', 'hôn', 'ôm', 'bên_nhau', 'mãi_mãi', 'chung_đôi', 'thương',
        'chill', 'phiêu', 'cuốn', 'dính', 'mê', 'ngất_ngây', 'lâng_lâng', 'thả_thính',
        'bay', 'thăng_hoa', 'bùng_cháy', 'cháy', 'đỉnh', 'keo', 'lì', 'mlem',
        'bình_yên', 'an yên', 'ấm', 'ấm áp', 'dịu dàng', 'êm', 'nhẹ nhàng',
        'thảnh_thơi', 'tự_do', 'chữa_lành', 'an_ủi', 'vỗ_về', 'xinh', 'lung_linh',
        'lấp_lánh', 'rực_rỡ', 'tự_hào', 'kiêu_hãnh', 'thành_công', 'vinh_quang',
        'nồng_nàn', 'say_đắm', 'vẹn_tròn', 'đắm_say', 'ngây_ngất', 'nồng_cháy',
        'phi_thường'
    ]
    negative_words = [
        'buồn', 'đau', 'khóc', 'lỗi', 'sầu', 'thương_đau', 'đau_khổ', 'tê_tái',
        'nghẹn', 'thắt', 'nhói', 'buốt', 'xót', 'đắng', 'cay', 'tủi', 'hờn',
        'oán', 'trách', 'hận', 'tiếc', 'day_dứt', 'ám_ảnh', 'toang', 'xu',
        'suy', 'lụy', 'trầm_cảm', 'mệt', 'chán', 'nản', 'gãy', 'cút',
        'xa', 'mất', 'nhớ', 'quên', 'cô_đơn', 'lẻ_loi', 'vỡ', 'tan', 'chia',
        'lìa', 'giã_từ', 'biệt_ly', 'rời', 'bỏ', 'buông', 'lạc', 'trôi',
        'phôi_pha', 'nhạt', 'phai', 'tàn', 'úa', 'héo', 'biến_mất', 'cách_xa',
        'lệ', 'nước_mắt', 'ướt_mi', 'hoen_mi', 'đêm', 'mưa', 'bão', 'giông',
        'tối', 'đen', 'lạnh', 'giá', 'rét', 'vực', 'hố', 'mây_đen', 'bóng_tối',
        'chết', 'tử', 'gục', 'ngã', 'vỡ_nát', 'vô_vọng', 'tuyệt_vọng',
        'giá_như', 'sai', 'sai_lầm', 'hối_hận', 'muộn_màng', 'lầm_lỡ',
        'bên_lề', 'đứng_sau', 'lặng_lẽ', 'âm_thầm', 'vô_hình', 'xa_lạ',
        'ngừng_trôi', 'thước_phim', 'kỷ_niệm', 'quá_khứ'
    ]
    negation_words = [
        'không', 'không_cần', 'không_được', 'khỏi', 'khỏi_cần', 'khỏi_phải',
        'chẳng', 'chẳng_cần', 'chẳng_được', 'chẳng_phải', 'chẳng_thà', ' chẳng_có',
        'chả', 'chả_cần', 'chả_được', 'chả_phải',
        'đừng', 'chưa', 'thôi', 'hết', 'ngừng', 'dứt', 'đéo', 'đếch', 'chớ'
    ]

    ngram_sad_phrases = [
        'người đến sau', 'bên người mới', 'ai kia', 'người ta',
        'chúc em', 'chúc anh', 'giá như', 'bên lề', 'thước phim',
        'chẳng thể', 'không thể', 'xa lạ', 'quá khứ', 'làm bạn',
        'tình đơn phương', 'người cũ', 'từng là', 'lặng lẽ', 'rời xa'
    ]

    text_lower = str(text_clean).lower()
    try:
        from underthesea import word_tokenize

        original_tokens = word_tokenize(text_lower.replace("\n", " "))
    except Exception:
        original_tokens = text_lower.replace("\n", " ").split()

    total_original_words = int(len(original_tokens) if len(original_tokens) > 0 else 1)

    pos_count = 0.0
    neg_count = 0.0

    for phrase in ngram_sad_phrases:
        if phrase in text_lower:
            occurrences = int(text_lower.count(phrase))
            neg_count += float(occurrences) * 3.5
            text_lower = text_lower.replace(phrase, " ")

    text_for_tokens = text_lower.replace("\n", " ")
    try:
        from underthesea import word_tokenize

        tokens = word_tokenize(text_for_tokens)
    except Exception:
        tokens = text_for_tokens.split()

    neg_weight = 1.5
    for i, token in enumerate(tokens):
        context_before = tokens[max(0, i - 3): i]
        has_negation = any(neg in context_before for neg in negation_words)

        if token in positive_words:
            if has_negation:
                neg_count += 1 * neg_weight
            else:
                pos_count += 1
        elif token in negative_words:
            if has_negation:
                pos_count += 0.5
            else:
                neg_count += 1 * neg_weight

    vn_ratio = (pos_count - neg_count) / float(total_original_words)
    final_score = (vn_ratio * 0.7) + (vader_score * 0.05 * 0.3)

    if final_score > 0.01:
        sentiment = "positive"
    elif final_score < -0.01:
        sentiment = "negative"
    else:
        sentiment = "neutral"

    return sentiment, round(float(final_score), 4)


_PHOBERT_PIPE = None


def _get_phobert_pipe():
    global _PHOBERT_PIPE
    if _PHOBERT_PIPE is not None:
        return _PHOBERT_PIPE

    if str(os.getenv("NLP_DISABLE_PHOBERT", "") or "").strip().lower() in {"1", "true", "yes", "on"}:
        return None

    try:
        from transformers import pipeline

        _PHOBERT_PIPE = pipeline(
            "sentiment-analysis",
            model="wonrax/phobert-base-vietnamese-sentiment",
            tokenizer="wonrax/phobert-base-vietnamese-sentiment",
        )
        return _PHOBERT_PIPE
    except Exception:
        return None


def _analyze_phobert(text_clean: str) -> tuple[str, float]:
    """Mirror `analyze_phobert` from nlp_colab.py (chunking by lyric lines)."""

    if not text_clean:
        return "neutral", 0.0

    pipe = _get_phobert_pipe()
    if pipe is None:
        return "neutral", 0.0

    lines = str(text_clean).split("\n")

    max_chars = 400
    chunks: list[str] = []
    current_chunk = ""

    for line in lines:
        line = str(line).strip()
        if not line:
            continue
        if len(current_chunk) + len(line) + 1 > max_chars:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = line
        else:
            current_chunk = (current_chunk + " " + line) if current_chunk else line

    if current_chunk:
        chunks.append(current_chunk)

    score_map = {"POS": 1.0, "NEU": 0.0, "NEG": -1.0}
    try:
        chunk_results = [pipe(c)[0] for c in chunks]
        avg_val = float(np.mean([score_map.get(str(r.get("label")), 0.0) for r in chunk_results]))
        avg_conf = float(np.mean([float(r.get("score", 0.0)) for r in chunk_results]))
        sentiment = "positive" if avg_val > 0.2 else ("negative" if avg_val < -0.2 else "neutral")
        return sentiment, round(float(avg_conf), 4)
    except Exception:
        return "neutral", 0.0


def analyze_lyrics(text: str) -> dict[str, Any]:
    """NLP pipeline for ANALYZE_READY.

    Output keys align with the ML feature schema.
    """

    feats, lyr_clean = _extract_lyric_features(text)
    s_lex, score_lex = _analyze_lexicon(lyr_clean)
    s_pho, conf_pho = _analyze_phobert(lyr_clean)

    # Hybrid rule (same spirit as existing backend)
    if s_pho == s_lex:
        final_sent = s_pho
    elif s_pho == "positive" and score_lex < -0.02:
        final_sent = "negative"
    elif conf_pho > 0.95:
        final_sent = s_pho
    else:
        final_sent = s_lex

    out = {
        "sent_lexicon": s_lex,
        "score_lexicon": float(score_lex),
        "sent_phobert": s_pho,
        "conf_phobert": float(conf_pho),
        "final_sentiment": final_sent,
        **feats,
        # Match training export: `lyric` is the cleaned lyric (newline preserved)
        "lyric": lyr_clean,
        # Backward-compat: some pipelines look for this name.
        "clean_lyric": lyr_clean,
    }
    return out
