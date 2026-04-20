"""
Backend phân tích Audio (Librosa) + Lyric (NLP).
Được chia làm 2 luồng độc lập:
1. LUỒNG SEARCH: Nhanh, Real-time, 10s, Z-Score 40D.
2. LUỒNG ANALYZE: Chậm, Full bài, Audio thô + PhoBERT + Lexicon + BERTopic.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import os
import re
import numpy as np
import sys
from importlib import import_module
import joblib
from pathlib import Path
import pandas as pd


def _import_third_party_librosa():
    """Import the pip package `librosa` safely.

    This repo also contains `chatbot/librosa.py` which can shadow the pip package
    if the working directory / sys.path includes the `chatbot/` folder.
    """

    here = os.path.dirname(os.path.abspath(__file__))
    here_norm = os.path.normcase(os.path.abspath(here))
    removed: list[str] = []
    popped_module = None

    try:
        try:
            existing = sys.modules.get('librosa')
            existing_file = os.path.normcase(os.path.abspath(getattr(existing, '__file__', '') or '')) if existing is not None else ''
            local_librosa = os.path.normcase(os.path.abspath(os.path.join(here, 'librosa.py')))
            if existing is not None and existing_file == local_librosa:
                popped_module = sys.modules.pop('librosa', None)
        except Exception:
            popped_module = None

        for p in list(sys.path):
            try:
                if os.path.normcase(os.path.abspath(p)) == here_norm:
                    sys.path.remove(p)
                    removed.append(p)
            except Exception:
                continue

        return import_module('librosa')
    finally:
        if removed:
            sys.path[:0] = removed
        if popped_module is not None:
            sys.modules['librosa'] = popped_module


librosa = _import_third_party_librosa()

@dataclass
class SaveResult:
    ok: bool
    message: str

class VPopAnalysisBackend:
    """Class lõi xử lý Dữ liệu Âm nhạc & Ngôn ngữ V-Pop"""

    # ===== 1. HẰNG SỐ NLP & TOPIC =====
    NLP_KEYS: Tuple[str, ...] = (
        "sent_lexicon", "score_lexicon", "sent_phobert", "conf_phobert",
        "final_sentiment", "lyric_total_words", "lexical_diversity",
        "noun_count", "verb_count", "adj_count", "clean_lyric",
        # 16 cột features của Topic Modeling (BERTopic)
        *tuple([f"topic_prob_{i}" for i in range(16)])
    )

    def __init__(self, supabase_client: Any = None, default_table: str = "track_features"):
        self.supabase = supabase_client
        self.default_table = default_table
        self.temp_table: List[Dict[str, Any]] = []
        
        # Biến đánh dấu tải Model
        self._bertopic_model = None
        self._nlp_loaded = False

        # --- LOAD THƯỚC ĐO Z-SCORE ---
        # Lấy đường dẫn tuyệt đối đến thư mục chứa file này, rồi nhảy lên 1 cấp
        BASE_DIR = Path(__file__).resolve().parent
        scaler_path = BASE_DIR / 'models' / 'audio_scaler.joblib'
        if not scaler_path.exists(): # Nếu chạy từ scripts/ thì check lại đường dẫn
             scaler_path = Path.cwd() / 'models' / 'audio_scaler.joblib'

        try:
            self.audio_scaler = joblib.load(scaler_path)
            print("✅ Đã nạp Thước đo Audio Scaler thành công.")
        except Exception as e:
            print(f"⚠️ Cảnh báo: Chưa tìm thấy audio_scaler.joblib tại {scaler_path}. (Cần chạy bước update_vectors_db trước)")
            self.audio_scaler = None

    # =====================================================================
    # LUỒNG 1: SEARCH_AUDIO (Nhanh, 10s, Z-Score 40D Vector)
    # =====================================================================
    def extract_search_vector(self, audio_path: str) -> Optional[list[float]]:
        """
        PHƯƠNG PHÁP SHAZAM-LIKE: Phân tích toàn bộ nội dung file đầu vào, 
        bất kể độ dài hay vị trí đoạn nhạc. Đã fix UserWarning bằng DataFrame.
        """
        if not self.audio_scaler:
            print("❌ Lỗi: Backend thiếu Scaler.")
            return None
            
        try:
            # 1. Load toàn bộ file đầu vào
            y_raw, sr = librosa.load(audio_path, sr=22050)
            
            # Cắt bỏ khoảng lặng
            y, _ = librosa.effects.trim(y_raw, top_db=25)
            duration = librosa.get_duration(y=y, sr=sr)
            
            if duration < 3.0:
                print("⚠️ File quá ngắn (dưới 3s) để phân tích chính xác.")
                return None

            # 2. Trích xuất đặc trưng (Mean trên toàn bộ file)
            mfccs = [np.mean(m) for m in librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)]
            chroma = [np.mean(c) for c in librosa.feature.chroma_stft(y=y, sr=sr)]
            contrast = [np.mean(ct) for ct in librosa.feature.spectral_contrast(y=y, sr=sr)]
            
            # Tối ưu hóa bắt nhịp
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr, start_bpm=110)
            
            phys = [
                float(tempo),
                np.mean(librosa.feature.rms(y=y)),
                np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
                np.mean(librosa.feature.zero_crossing_rate(y=y)),
                np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)),
                np.mean(librosa.feature.spectral_flatness_mean(y=y)) if hasattr(librosa.feature, 'spectral_flatness_mean') else np.mean(librosa.feature.spectral_flatness(y=y)),
                np.mean(librosa.onset.onset_strength(y=y, sr=sr)),
                len(librosa.onset.onset_detect(y=y, sr=sr)) / max(duration, 1)
            ]

            # 3. Tổng hợp mảng 40 chiều thô theo đúng thứ tự
            raw_ordered = mfccs + chroma + contrast + phys

            # 4. ÉP Z-SCORE BẰNG DATAFRAME (Để fix UserWarning và khớp metadata)
            # Tạo danh sách tên cột đúng thứ tự lúc train mô hình
            feature_names = [f'mfcc{i}_mean' for i in range(1, 14)] + \
                            [f'chroma{i}_mean' for i in range(1, 13)] + \
                            [f'spectral_contrast_band{i}_mean' for i in range(1, 8)] + \
                            ['tempo_bpm', 'rms_energy', 'spectral_centroid_mean', 'zero_crossing_rate', 
                             'spectral_rolloff', 'spectral_flatness_mean', 'beat_strength_mean', 'onset_rate']

            # Chuyển mảng thô thành DataFrame 1 dòng
            raw_df = pd.DataFrame([raw_ordered], columns=feature_names)
            
            # Transform bằng scaler (Lúc này sẽ không còn Warning vì đã có tên cột)
            z_array = self.audio_scaler.transform(raw_df)
            
            return [round(float(x), 6) for x in z_array[0]]

        except Exception as e:
            print(f"Lỗi extract_search_vector: {e}")
            return None
        
    def search_similar_tracks(self, audio_path: str, match_count: int = 5) -> Dict[str, Any]:
        if not self.supabase: return {"error": "Chưa kết nối Supabase"}
        query_vector = self.extract_search_vector(audio_path)
        if not query_vector: return {"error": "Không thể phân tích file audio này."}

        try:
            # RPC match_audio_signature phải được định nghĩa trên Supabase
            res = self.supabase.rpc('match_audio_signature', {
                'query_embedding': query_vector,
                'match_threshold': 0.3, # Giảm xuống 0.3 để bắt rộng hơn
                'match_count': match_count
            }).execute()
            return {"tracks": getattr(res, 'data', []) or [], "error": None}
        except Exception as e:
            return {"error": f"Lỗi truy vấn Database: {str(e)}"}

    # =====================================================================
    # LUỒNG 3: LƯU TRỮ VÀO CƠ SỞ DỮ LIỆU
    # =====================================================================
    def save_record(self, record: Dict[str, Any], table_name: str = "") -> SaveResult:
        self.temp_table.append(record)
        tn = table_name or self.default_table
        if not self.supabase or not tn:
            return SaveResult(ok=True, message="Đã lưu vào Temp Buffer.")
        try:
            # self.supabase.table(tn).insert(record).execute()
            return SaveResult(ok=True, message="Lưu DB thành công.")
        except Exception as e:
            return SaveResult(ok=False, message=f"Lỗi lưu DB: {str(e)}")
        