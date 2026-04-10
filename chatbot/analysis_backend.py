"""Backend phân tích Audio (Librosa) + Lyric (NLP).

Mục tiêu:
- Gom logic phân tích Librosa + NLP vào 1 class dùng lại ở mọi nơi.
- Dựa theo logic code bạn cung cấp (Hybrid Lexicon + PhoBERT, và bộ audio features).
- Có cơ chế lưu tạm (in-memory) và hook để lưu DB (Supabase) khi bạn tạo bảng.

Lưu ý:
- PhoBERT được lazy-load để tránh tải model nặng lúc import.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import os
import re


@dataclass
class SaveResult:
    ok: bool
    message: str


class VPopAnalysisBackend:
    """Một class chung cho phân tích NLP + Librosa."""

    # ===== Expected keys (để check "đủ thông số") =====
    NLP_KEYS: Tuple[str, ...] = (
        "sent_lexicon",
        "score_lexicon",
        "sent_phobert",
        "conf_phobert",
        "final_sentiment",
        "lyric_total_words",
        "lexical_diversity",
        "noun_count",
        "verb_count",
        "adj_count",
        "clean_lyric",
    )

    # Theo csv_header bạn gửi (không gồm file_name/spotify_track_id)
    AUDIO_KEYS: Tuple[str, ...] = (
        "duration_sec",
        "tempo_bpm",
        "musical_key",
        "rms_energy",
        "spectral_centroid_mean",
        "mfcc10_mean",
        "mfcc10_std",
        "mfcc11_mean",
        "mfcc11_std",
        "mfcc12_mean",
        "mfcc12_std",
        "mfcc13_mean",
        "mfcc13_std",
        "mfcc1_mean",
        "mfcc1_std",
        "mfcc2_mean",
        "mfcc2_std",
        "mfcc3_mean",
        "mfcc3_std",
        "mfcc4_mean",
        "mfcc4_std",
        "mfcc5_mean",
        "mfcc5_std",
        "mfcc6_mean",
        "mfcc6_std",
        "mfcc7_mean",
        "mfcc7_std",
        "mfcc8_mean",
        "mfcc8_std",
        "mfcc9_mean",
        "mfcc9_std",
        "zero_crossing_rate",
        "spectral_rolloff",
        "spectral_contrast_band1_mean",
        "spectral_contrast_band2_mean",
        "spectral_contrast_band3_mean",
        "spectral_contrast_band4_mean",
        "spectral_contrast_band5_mean",
        "spectral_contrast_band6_mean",
        "spectral_contrast_band7_mean",
        "spectral_flatness_mean",
        "beat_strength_mean",
        "onset_rate",
        "tempo_stability",
        "chroma1_mean",
        "chroma2_mean",
        "chroma3_mean",
        "chroma4_mean",
        "chroma5_mean",
        "chroma6_mean",
        "chroma7_mean",
        "chroma8_mean",
        "chroma9_mean",
        "chroma10_mean",
        "chroma11_mean",
        "chroma12_mean",
        "tonnetz1_mean",
        "tonnetz2_mean",
        "tonnetz3_mean",
        "tonnetz4_mean",
        "tonnetz5_mean",
        "tonnetz6_mean",
        "harmonic_percussive_ratio",
    )

    def __init__(
        self,
        *,
        enable_phobert: bool = True,
        phobert_model: str = "wonrax/phobert-base-vietnamese-sentiment",
        phobert_tokenizer: str = "wonrax/phobert-base-vietnamese-sentiment",
        supabase_client: Any = None,
        default_table: str = "",
        sample_rate: int = 22050,
        n_fft: int = 2048,
        hop_length: int = 512,
    ):
        self.enable_phobert = enable_phobert
        self.phobert_model = phobert_model
        self.phobert_tokenizer = phobert_tokenizer

        self.supabase = supabase_client
        self.default_table = default_table  # để trống theo yêu cầu (chưa upload bảng)

        self.sr = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length

        self._vader_analyzer = None
        self._phobert = None
        self.temp_table: List[Dict[str, Any]] = []  # "bảng tạm" in-memory

    # =====================
    # Lyrics IO helpers
    # =====================
    @staticmethod
    def get_lyrics_from_txt(file_name: str, lyric_dir: str) -> str:
        try:
            base_name = os.path.splitext(file_name)[0]
            txt_path = os.path.join(lyric_dir, base_name + ".txt")
            if os.path.exists(txt_path):
                with open(txt_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    return content if len(content) > 10 else ""
            return ""
        except Exception:
            return ""

    @staticmethod
    def get_lyrics_from_mp3(file_path: str) -> str:
        try:
            from mutagen.id3 import ID3

            tags = ID3(file_path)
            for key in tags.keys():
                if key.startswith("USLT"):
                    uslt = tags[key]
                    return uslt.text if hasattr(uslt, "text") else str(uslt)
            return ""
        except Exception:
            return ""

    # =====================
    # NLP analysis (dựa theo code bạn gửi)
    # =====================
    def _ensure_vader(self) -> None:
        if self._vader_analyzer is not None:
            return
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        self._vader_analyzer = SentimentIntensityAnalyzer()

    def _ensure_phobert(self) -> None:
        if not self.enable_phobert:
            return
        if self._phobert is not None:
            return
        from transformers import pipeline

        # Lazy-load để tránh nặng lúc import
        self._phobert = pipeline(
            "sentiment-analysis",
            model=self.phobert_model,
            tokenizer=self.phobert_tokenizer,
        )

    @staticmethod
    def extract_lyric_features(text: str) -> Tuple[Dict[str, Any], str]:
        """Bản rút gọn đúng theo đoạn code bạn dán (clean + POS + counts)."""
        if not text or not isinstance(text, str):
            return (
                {
                    "lyric_total_words": 0,
                    "lexical_diversity": 0.0,
                    "noun_count": 0,
                    "verb_count": 0,
                    "adj_count": 0,
                },
                "",
            )

        # Giữ chữ cái tiếng Việt + newline
        text_clean = re.sub(
            r"[^a-záàảãạăắằẳẵặâấầẩẫậèéèẻẽẹêềếểễệóòỏõọôồốổỗộơờớởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ\n ]",
            " ",
            text.lower(),
        )
        valid_lines = [l.strip() for l in text_clean.split("\n") if l.strip()]
        text_clean = "\n".join(valid_lines)
        text_for_nlp = text_clean.replace("\n", " ")

        from underthesea.pipeline.word_tokenize import word_tokenize
        from underthesea.pipeline.pos_tag import pos_tag

        tokens = word_tokenize(text_for_nlp)
        tags = pos_tag(text_for_nlp)
        noun = sum(1 for _, t in tags if t in ["N", "Np", "Nc", "Nu"])
        verb = sum(1 for _, t in tags if t in ["V", "Vy"])
        adj = sum(1 for _, t in tags if t in ["A", "Adj"])

        feats = {
            "lyric_total_words": len(tokens),
            "lexical_diversity": round(len(set(tokens)) / len(tokens), 4) if tokens else 0.0,
            "noun_count": noun,
            "verb_count": verb,
            "adj_count": adj,
        }
        return feats, text_clean

    def analyze_lexicon(self, lyric_text: str) -> Tuple[str, float]:
        """SENTIMENT ANALYSIS HYBRID V2: Lexicon Tiếng Việt + Slang + VADER."""
        if not lyric_text or len(lyric_text.strip()) == 0:
            return "neutral", 0.0

        self._ensure_vader()

        positive_words = [
            "vui_vẻ",
            "hạnh_phúc",
            "phấn_khởi",
            "yêu",
            "thích",
            "vui",
            "đẹp",
            "tuyệt",
            "mơ",
            "ước",
            "cười",
            "rạng_rỡ",
            "ngọt_ngào",
            "nắng",
            "xuân",
            "hứng_khởi",
            "rộn_ràng",
            "tươi",
            "sáng",
            "tình",
            "hôn",
            "ôm",
            "bên_nhau",
            "mãi_mãi",
            "chung_đôi",
            "thương",
            "chill",
            "phiêu",
            "cuốn",
            "dính",
            "mê",
            "ngất_ngây",
            "lâng_lâng",
            "thả_thính",
            "bay",
            "thăng_hoa",
            "bùng_cháy",
            "cháy",
            "đỉnh",
            "keo",
            "lì",
            "mlem",
            "bình_yên",
            "an yên",
            "ấm",
            "ấm áp",
            "dịu dàng",
            "êm",
            "nhẹ nhàng",
            "thảnh_thơi",
            "tự_do",
            "chữa_lành",
            "an_ủi",
            "vỗ_về",
            "xinh",
            "lung_linh",
            "lấp_lánh",
            "rực_rỡ",
            "tự_hào",
            "kiêu_hãnh",
            "thành_công",
            "vinh_quang",
            "nồng_nàn",
            "say_đắm",
            "vẹn_tròn",
            "đắm_say",
            "ngây_ngất",
            "nồng_cháy",
            "phi_thường",
        ]

        negative_words = [
            "buồn",
            "đau",
            "khóc",
            "lỗi",
            "sầu",
            "thương_đau",
            "đau_khổ",
            "tê_tái",
            "nghẹn",
            "thắt",
            "nhói",
            "buốt",
            "xót",
            "đắng",
            "cay",
            "tủi",
            "hờn",
            "oán",
            "trách",
            "hận",
            "tiếc",
            "day_dứt",
            "ám_ảnh",
            "toang",
            "xu",
            "suy",
            "lụy",
            "trầm_cảm",
            "mệt",
            "chán",
            "nản",
            "gãy",
            "cút",
            "xa",
            "mất",
            "nhớ",
            "quên",
            "cô_đơn",
            "lẻ_loi",
            "vỡ",
            "tan",
            "chia",
            "lìa",
            "giã_từ",
            "biệt_ly",
            "rời",
            "bỏ",
            "buông",
            "lạc",
            "trôi",
            "phôi_pha",
            "nhạt",
            "phai",
            "tàn",
            "úa",
            "héo",
            "biến_mất",
            "cách_xa",
            "lệ",
            "nước_mắt",
            "ướt_mi",
            "hoen_mi",
            "đêm",
            "mưa",
            "bão",
            "giông",
            "tối",
            "đen",
            "lạnh",
            "giá",
            "rét",
            "vực",
            "hố",
            "mây_đen",
            "bóng_tối",
            "chết",
            "tử",
            "gục",
            "ngã",
            "vỡ_nát",
            "vô_vọng",
            "tuyệt_vọng",
            "giá_như",
            "sai",
            "sai_lầm",
            "hối_hận",
            "muộn_màng",
            "lầm_lỡ",
            "bên_lề",
            "đứng_sau",
            "lặng_lẽ",
            "âm_thầm",
            "vô_hình",
            "xa_lạ",
            "ngừng_trôi",
            "thước_phim",
            "kỷ_niệm",
            "quá_khứ",
        ]

        negation_words = [
            "không",
            "không_cần",
            "không_được",
            "khỏi",
            "khỏi_cần",
            "khỏi_phải",
            "chẳng",
            "chẳng_cần",
            "chẳng_được",
            "chẳng_phải",
            "chẳng_thà",
            " chẳng_có",
            "chả",
            "chả_cần",
            "chả_được",
            "chả_phải",
            "đừng",
            "chưa",
            "thôi",
            "hết",
            "ngừng",
            "dứt",
            "đéo",
            "đếch",
            "chớ",
        ]

        ngram_sad_phrases = [
            "người đến sau",
            "bên người mới",
            "ai kia",
            "người ta",
            "chúc em",
            "chúc anh",
            "giá như",
            "bên lề",
            "thước phim",
            "chẳng thể",
            "không thể",
            "xa lạ",
            "quá khứ",
            "làm bạn",
            "tình đơn phương",
            "người cũ",
            "từng là",
            "lặng lẽ",
            "rời xa",
        ]

        vader_score = self._vader_analyzer.polarity_scores(lyric_text)["compound"]

        text_lower = lyric_text.lower()
        pos_count = 0.0
        neg_count = 0.0

        from underthesea.pipeline.word_tokenize import word_tokenize

        original_tokens = word_tokenize(text_lower.replace("\n", " "))
        total_original_words = len(original_tokens) if len(original_tokens) > 0 else 1

        # LỚP 1: n-grams
        for phrase in ngram_sad_phrases:
            if phrase in text_lower:
                occurrences = text_lower.count(phrase)
                neg_count += occurrences * 3.5
                text_lower = text_lower.replace(phrase, " ")

        # LỚP 2: token scan
        text_for_tokens = text_lower.replace("\n", " ")
        tokens = word_tokenize(text_for_tokens)
        neg_weight = 1.5

        for i, token in enumerate(tokens):
            context_before = tokens[max(0, i - 3) : i]
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

        vn_ratio = (pos_count - neg_count) / total_original_words
        final_score = (vn_ratio * 0.7) + (vader_score * 0.05 * 0.3)

        if final_score > 0.01:
            sentiment = "positive"
        elif final_score < -0.01:
            sentiment = "negative"
        else:
            sentiment = "neutral"

        return sentiment, round(final_score, 4)

    def analyze_phobert(self, text: str) -> Tuple[str, float]:
        """Chấm điểm bằng PhoBERT (chunking theo dòng \n)."""
        if not text:
            return "neutral", 0.0

        if not self.enable_phobert:
            return "neutral", 0.0

        try:
            self._ensure_phobert()
        except Exception:
            # Nếu model chưa tải được, không làm fail hệ thống
            return "neutral", 0.0

        lines = text.split("\n")
        max_chars = 400
        chunks: List[str] = []
        current_chunk = ""

        for line in lines:
            line = line.strip()
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

        score_map = {"POS": 1, "NEU": 0, "NEG": -1}

        try:
            chunk_results = [self._phobert(c)[0] for c in chunks]
            import numpy as np

            avg_val = np.mean([score_map.get(r.get("label", "NEU"), 0) for r in chunk_results])
            avg_conf = np.mean([r.get("score", 0.0) for r in chunk_results])
            sentiment = "positive" if avg_val > 0.2 else ("negative" if avg_val < -0.2 else "neutral")
            return sentiment, round(float(avg_conf), 4)
        except Exception:
            return "neutral", 0.0

    def analyze_lyrics(self, text: str) -> Dict[str, Any]:
        """Trả về NLP features phẳng (giống output CSV)."""
        feats, lyr_clean = self.extract_lyric_features(text)
        s_lex, score_lex = self.analyze_lexicon(lyr_clean)
        s_pho, conf_pho = self.analyze_phobert(lyr_clean)

        # Logic Hybrid V3 theo đoạn code bạn dán
        if s_pho == s_lex:
            final_sent = s_pho
        elif s_pho == "positive" and score_lex < -0.02:
            final_sent = "negative"
        elif conf_pho > 0.95:
            final_sent = s_pho
        else:
            final_sent = s_lex

        return {
            "sent_lexicon": s_lex,
            "score_lexicon": score_lex,
            "sent_phobert": s_pho,
            "conf_phobert": conf_pho,
            "final_sentiment": final_sent,
            "lyric_total_words": feats["lyric_total_words"],
            "lexical_diversity": feats["lexical_diversity"],
            "noun_count": feats["noun_count"],
            "verb_count": feats["verb_count"],
            "adj_count": feats["adj_count"],
            "clean_lyric": lyr_clean,
        }

    def extract_keywords(self, text: str, *, top_k: int = 8) -> List[str]:
        """Trích xuất keyword đơn giản (dùng cho truy vấn vector).

        - Ưu tiên underthesea tokenizer khi có.
        - Fallback về regex tokenizer khi thiếu dependency.
        """

        top_k = max(1, int(top_k))
        raw = str(text or "").strip()
        if not raw:
            return []

        # Danh sách stopwords rút gọn (tránh kéo thêm file).
        stop = {
            "và",
            "là",
            "của",
            "cho",
            "một",
            "những",
            "các",
            "tôi",
            "mình",
            "bạn",
            "anh",
            "em",
            "với",
            "ở",
            "đi",
            "nhé",
            "ạ",
            "ơi",
            "tìm",
            "gợi",
            "ý",
            "nhạc",
            "bài",
            "hát",
        }

        tokens: List[str] = []
        try:
            from underthesea.pipeline.word_tokenize import word_tokenize

            tokens = [t.strip().lower() for t in word_tokenize(raw) if isinstance(t, str)]
        except Exception:
            tokens = [t.lower() for t in re.findall(r"\w+", raw, flags=re.UNICODE)]

        cleaned: List[str] = []
        seen: set[str] = set()
        for tok in tokens:
            tok = tok.strip().strip("_-")
            if not tok or len(tok) < 3:
                continue
            if tok in stop:
                continue
            if tok in seen:
                continue
            cleaned.append(tok)
            seen.add(tok)
            if len(cleaned) >= top_k:
                break
        return cleaned

    # =====================
    # Audio analysis (Librosa)
    # =====================
    def extract_audio_features(self, y, sr: int) -> Dict[str, Any]:
        import librosa
        import numpy as np

        features: Dict[str, Any] = {}

        # 1) Temporal
        features["duration_sec"] = round(float(librosa.get_duration(y=y, sr=sr)), 2)
        features["rms_energy"] = round(
            float(np.mean(librosa.feature.rms(y=y, frame_length=self.n_fft, hop_length=self.hop_length))),
            6,
        )
        features["zero_crossing_rate"] = round(
            float(
                np.mean(
                    librosa.feature.zero_crossing_rate(
                        y, frame_length=self.n_fft, hop_length=self.hop_length
                    )
                )
            ),
            6,
        )

        # 2) Spectral
        cent = librosa.feature.spectral_centroid(
            y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length
        )[0]
        features["spectral_centroid_mean"] = round(float(np.mean(cent)), 2)

        rolloff = librosa.feature.spectral_rolloff(
            y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length
        )[0]
        features["spectral_rolloff"] = round(float(np.mean(rolloff)), 2)

        contrast = librosa.feature.spectral_contrast(
            y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length
        )
        for i in range(contrast.shape[0]):
            features[f"spectral_contrast_band{i + 1}_mean"] = round(float(np.mean(contrast[i])), 2)

        features["spectral_flatness_mean"] = round(
            float(np.mean(librosa.feature.spectral_flatness(y=y, n_fft=self.n_fft, hop_length=self.hop_length))),
            6,
        )

        # 3) Rhythm
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=self.hop_length)
        features["tempo_bpm"] = round(float(tempo), 2)

        onset_env = librosa.onset.onset_strength(
            y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length
        )
        features["beat_strength_mean"] = round(float(np.mean(onset_env)), 4)

        onset_frames = librosa.onset.onset_detect(
            onset_envelope=onset_env, sr=sr, hop_length=self.hop_length
        )
        features["onset_rate"] = round(
            float(len(onset_frames) / max(features["duration_sec"], 1.0)),
            2,
        )

        if len(beats) > 1:
            beat_times = librosa.frames_to_time(beats, sr=sr)
            beat_intervals = np.diff(beat_times)
            features["tempo_stability"] = (
                round(1 - (np.std(beat_intervals) / np.mean(beat_intervals)), 4)
                if np.mean(beat_intervals) > 0
                else 0.0
            )
        else:
            features["tempo_stability"] = 0.0

        # 4) Timbre
        mfccs = librosa.feature.mfcc(
            y=y, sr=sr, n_mfcc=13, n_fft=self.n_fft, hop_length=self.hop_length
        )
        for i in range(13):
            features[f"mfcc{i + 1}_mean"] = round(float(np.mean(mfccs[i])), 4)
            features[f"mfcc{i + 1}_std"] = round(float(np.std(mfccs[i])), 4)

        # 5) Harmony
        chroma = librosa.feature.chroma_stft(
            y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length
        )
        for i in range(12):
            features[f"chroma{i + 1}_mean"] = round(float(np.mean(chroma[i])), 4)

        y_harmonic = librosa.effects.harmonic(y)
        y_perc = librosa.effects.percussive(y)
        features["harmonic_percussive_ratio"] = round(
            float(np.mean(abs(y_harmonic)) / (np.mean(abs(y_perc)) + 1e-10)),
            4,
        )

        tonnetz = librosa.feature.tonnetz(y=y_harmonic, sr=sr)
        for i in range(6):
            features[f"tonnetz{i + 1}_mean"] = round(float(np.mean(tonnetz[i])), 4)

        # Musical key detection
        chroma_cens = librosa.feature.chroma_cens(y=y_harmonic, sr=sr)
        chroma_mean = np.mean(chroma_cens, axis=1)

        major_p = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        minor_p = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

        chroma_mean /= (np.linalg.norm(chroma_mean) + 1e-10)
        major_p /= np.linalg.norm(major_p)
        minor_p /= np.linalg.norm(minor_p)

        pitch_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        best_key, max_corr = "Unknown", -1.0

        for i in range(12):
            c_maj = float(np.dot(chroma_mean, np.roll(major_p, i)))
            c_min = float(np.dot(chroma_mean, np.roll(minor_p, i)))
            if c_maj > max_corr:
                max_corr, best_key = c_maj, f"{pitch_names[i]}:maj"
            if c_min > max_corr:
                max_corr, best_key = c_min, f"{pitch_names[i]}:min"

        features["musical_key"] = best_key

        return features

    def analyze_audio_file(self, file_path: str) -> Dict[str, Any]:
        import librosa

        y, sr = librosa.load(file_path, sr=self.sr, mono=True)
        return self.extract_audio_features(y, sr)


class AudioAnalyzer:
    """Compatibility wrapper used by the Streamlit app."""

    def __init__(self, backend: VPopAnalysisBackend | None = None) -> None:
        self.backend = backend or VPopAnalysisBackend()

    def process_audio_file(self, path: str) -> Dict[str, Any]:
        return self.backend.analyze_audio_file(path)


class NLPAnalyzer:
    """Compatibility wrapper used by the Streamlit app."""

    def __init__(self, backend: VPopAnalysisBackend | None = None) -> None:
        self.backend = backend or VPopAnalysisBackend()

    def analyze_full_lyrics(self, text: str) -> Dict[str, Any]:
        return self.backend.analyze_lyrics(text)

    def extract_keywords(self, text: str, *, top_k: int = 8) -> List[str]:
        return self.backend.extract_keywords(text, top_k=top_k)

    # =====================
    # Combined + persistence
    # =====================
    @staticmethod
    def check_missing_keys(features: Dict[str, Any], expected_keys: Tuple[str, ...]) -> List[str]:
        return [k for k in expected_keys if k not in features]

    def analyze_track(
        self,
        *,
        file_name: str = "",
        spotify_track_id: str = "",
        audio_path: Optional[str] = None,
        lyric_text: str = "",
        lyric_txt_dir: str = "",
        prefer_txt: bool = True,
    ) -> Dict[str, Any]:
        """Phân tích 1 track (audio + lyric nếu có)."""

        record: Dict[str, Any] = {
            "file_name": file_name,
            "spotify_track_id": spotify_track_id,
        }

        # Lyrics: ưu tiên txt -> mp3 metadata -> lyric_text
        final_lyric = lyric_text
        if prefer_txt and file_name and lyric_txt_dir:
            txt_lyr = self.get_lyrics_from_txt(file_name, lyric_txt_dir)
            if txt_lyr:
                final_lyric = txt_lyr

        if (not final_lyric) and audio_path:
            mp3_lyr = self.get_lyrics_from_mp3(audio_path)
            if mp3_lyr:
                final_lyric = mp3_lyr

        if final_lyric:
            record["lyric"] = final_lyric
            record.update(self.analyze_lyrics(final_lyric))
        else:
            record["lyric"] = ""
            record.update({k: ("neutral" if "sent" in k else 0) for k in self.NLP_KEYS if k not in ("clean_lyric",)})
            record["clean_lyric"] = ""
            record["final_sentiment"] = "neutral"

        if audio_path:
            record.update(self.analyze_audio_file(audio_path))

        return record

    def save_record(self, record: Dict[str, Any], table_name: str = "") -> SaveResult:
        """Lưu kết quả vào "bảng tạm" hoặc Supabase (nếu có bảng)."""
        # Luôn lưu vào buffer (tạm thời)
        self.temp_table.append(record)

        tn = table_name or self.default_table
        if not tn:
            return SaveResult(ok=True, message="Saved to temp_table (DB table not configured)")

        if self.supabase is None:
            return SaveResult(ok=False, message="No supabase_client provided")

        try:
            # insert trước; khi bạn muốn upsert thì đổi sang .upsert(...)
            self.supabase.table(tn).insert(record).execute()
            return SaveResult(ok=True, message=f"Saved to DB table '{tn}'")
        except Exception as e:
            return SaveResult(ok=False, message=f"DB save failed: {e}")
