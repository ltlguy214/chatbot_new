import streamlit as st
import pandas as pd
import numpy as np
import librosa
import joblib
import re
import os
import warnings
from underthesea import word_tokenize, pos_tag
from underthesea import sentiment as uts_sentiment

# Tắt cảnh báo
warnings.filterwarnings('ignore')

# =============================================================================
# 1. CẤU HÌNH & TỪ ĐIỂN (Lấy từ code của bạn)
# =============================================================================
SR = 22050
N_FFT = 2048
HOP_LENGTH = 512

# Từ điển V-Pop (Lấy từ extract_lyrics_from_mp3.py)
POSITIVE_WORDS = [
    'yêu', 'thích', 'vui', 'hạnh phúc', 'đẹp', 'tuyệt', 'mơ', 'ước', 'cười',
    'rạng rỡ', 'ngọt ngào', 'nắng', 'xuân', 'hứng khởi', 'rộn ràng', 'tươi',
    'sáng', 'tình', 'hôn', 'ôm', 'bên nhau', 'mãi mãi', 'chung đôi', 'thương',
    'chill', 'phiêu', 'cuốn', 'dính', 'mê', 'ngất ngây', 'lâng lâng',
    'bay', 'thăng hoa', 'bùng cháy', 'cháy', 'đỉnh', 'keo', 'lì',
    'bình yên', 'an yên', 'ấm', 'ấm áp', 'dịu dàng', 'êm', 'nhẹ nhàng',
    'thảnh thơi', 'tự do', 'chữa lành', 'an ủi', 'vỗ về',
    'xinh', 'lung linh', 'lấp lánh', 'rực rỡ', 'tự hào', 'kiêu hãnh',
    'thành công', 'vinh quang', 'báu vật', 'thiên đường', 'ngọc ngà'
]
NEGATIVE_WORDS = [
    'buồn', 'đau', 'khóc', 'lỗi', 'sầu', 'thương đau', 'đau khổ', 'tê tái',
    'nghẹn', 'thắt', 'nhói', 'buốt', 'xót', 'đắng', 'cay', 'tủi', 'hờn',
    'oán', 'trách', 'hận', 'tiếc', 'day dứt', 'ám ảnh', 'day dứt',
    'suy', 'lụy', 'trầm cảm', 'mệt', 'chán', 'nản', 'toang', 'gãy', 'xu',
    'xa', 'mất', 'nhớ', 'quên', 'cô đơn', 'lẻ loi', 'vỡ', 'tan', 'chia',
    'lìa', 'giã từ', 'biệt ly', 'rời', 'bỏ', 'buông', 'lạc', 'trôi',
    'phôi pha', 'nhạt', 'phai', 'tàn', 'úa', 'héo', 'biến mất', 'cách xa',
    'lệ', 'nước mắt', 'ướt mi', 'hoen mi', 'đêm', 'mưa', 'bão', 'giông',
    'tối', 'đen', 'lạnh', 'giá', 'rét', 'vực', 'hố', 'mây đen', 'bóng tối',
    'chết', 'tử', 'gục', 'ngã', 'vỡ nát', 'vô vọng', 'tuyệt vọng',
    'trống rỗng', 'hư vô', 'cô độc', 'bơ vơ', 'lạc lõng', 'trái ngang'
]
NEGATION_WORDS = [
    'không', 'chẳng', 'chả', 'đừng', 'chưa', 'thôi',
    'hết', 'ngừng', 'dứt', 'bỏ', 'khỏi', 'cấm',
    'đéo', 'đếch', 'dek', 'nào', 'làm gì', 'đâu có', 'có đâu']

# =============================================================================
# 2. HÀM XỬ LÝ AUDIO (Lấy từ extract_feature.py)
# =============================================================================
def extract_audio_features_custom(audio_path):
    """Trích xuất Audio theo đúng chuẩn extract_feature.py"""
    try:
        y, sr = librosa.load(audio_path, sr=SR)
    except Exception as e:
        return None

    features = {}
    try:
        # 1. Temporal
        features['duration_sec'] = float(librosa.get_duration(y=y, sr=sr))
        features['rms_energy'] = float(np.mean(librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP_LENGTH)))
        features['zero_crossing_rate'] = float(np.mean(librosa.feature.zero_crossing_rate(y, frame_length=N_FFT, hop_length=HOP_LENGTH)))

        # 2. Spectral
        cent = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)[0]
        features['spectral_centroid_mean'] = float(np.mean(cent))
        
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)[0]
        features['spectral_rolloff'] = float(np.mean(rolloff))

        contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
        for i in range(contrast.shape[0]):
            features[f'spectral_contrast_band{i+1}_mean'] = float(np.mean(contrast[i]))

        features['spectral_flatness_mean'] = float(np.mean(librosa.feature.spectral_flatness(y=y, n_fft=N_FFT, hop_length=HOP_LENGTH)))

        # 3. Rhythm
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=HOP_LENGTH)
        features['tempo_bpm'] = float(tempo)

        onset_env = librosa.onset.onset_strength(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
        features['beat_strength_mean'] = float(np.mean(onset_env))
        
        onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=HOP_LENGTH)
        features['onset_rate'] = float(len(onset_frames) / features['duration_sec'])

        if len(beats) > 1:
            beat_times = librosa.frames_to_time(beats, sr=sr)
            beat_intervals = np.diff(beat_times)
            if np.mean(beat_intervals) > 0:
                features['tempo_stability'] = 1 - (np.std(beat_intervals) / np.mean(beat_intervals))
            else: features['tempo_stability'] = 0.0
        else: features['tempo_stability'] = 0.0

        # 4. Timbre & Harmony
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=N_FFT, hop_length=HOP_LENGTH)
        for i in range(13):
            features[f'mfcc{i+1}_mean'] = float(np.mean(mfccs[i]))
            features[f'mfcc{i+1}_std'] = float(np.std(mfccs[i]))

        chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
        for i in range(12):
            features[f'chroma{i+1}_mean'] = float(np.mean(chroma[i]))

        y_harmonic = librosa.effects.harmonic(y)
        tonnetz = librosa.feature.tonnetz(y=y_harmonic, sr=sr)
        for i in range(6):
            features[f'tonnetz{i+1}_mean'] = float(np.mean(tonnetz[i]))

        y_perc = librosa.effects.percussive(y)
        features['harmonic_percussive_ratio'] = float(np.mean(np.abs(y_harmonic)) / (np.mean(np.abs(y_perc)) + 1e-10))
        
    except Exception as e:
        st.error(f"Lỗi phân tích Audio: {e}")
        return None
    
    return features

# =============================================================================
# 3. HÀM XỬ LÝ LYRIC & SENTIMENT (Lấy từ extract_lyrics_from_mp3.py)
# =============================================================================
def analyze_lyrics_custom(lyric_text):
    """
    Kết hợp:
    1. Sentiment Advanced (từ extract_lyrics_from_mp3.py)
    2. Basic Linguistic (POS Tag) (từ extract_feature.py)
    """
    features = {}
    
    if not lyric_text or not isinstance(lyric_text, str):
        return features

    # --- A. SENTIMENT (Code chuẩn của bạn) ---
    features['sentiment'] = 'neutral'
    features['sentiment_score'] = 0.0
    features['sentiment_confidence'] = 0.0
    
    # 1. Underthesea Sentiment
    try:
        underthesea_result = uts_sentiment(lyric_text)
        features['sentiment'] = underthesea_result # Mặc định lấy cái này trước
    except:
        pass

    # 2. Dictionary-based Sentiment
    lyric_lower = lyric_text.lower()
    words = lyric_lower.split()
    pos_score = 0
    neg_score = 0
    
    for i, word in enumerate(words):
        context_before = words[max(0, i-3):i]
        has_negation = any(neg in context_before for neg in NEGATION_WORDS)
        
        if any(pos_word in word for pos_word in POSITIVE_WORDS):
            if has_negation: neg_score += 1
            else: pos_score += 1
        elif any(neg_word in word for neg_word in NEGATIVE_WORDS):
            if has_negation: pos_score += 0.5
            else: neg_score += 1
            
    total_score = pos_score + neg_score
    if total_score > 0:
        sentiment_ratio = (pos_score - neg_score) / total_score
        dict_score = round(sentiment_ratio, 4)
        dict_conf = round(total_score / (len(words) + 1), 4)
        
        # Logic cập nhật đè của bạn
        if features['sentiment'] == 'neutral':
            if sentiment_ratio > 0.2:
                features['sentiment'] = 'positive'
            elif sentiment_ratio < -0.2:
                features['sentiment'] = 'negative'
        
        features['sentiment_score'] = dict_score
        features['sentiment_confidence'] = dict_conf

    # --- B. LINGUISTIC STATS (Code từ extract_feature.py) ---
    text_clean = re.sub(r'[^\w\s]', '', lyric_text.lower())
    tokens = word_tokenize(text_clean)
    
    features['lyric_total_words'] = len(tokens)
    features['lexical_diversity'] = len(set(tokens)) / len(tokens) if len(tokens) > 0 else 0.0
    
    tags = pos_tag(text_clean)
    features['noun_count'] = sum(1 for word, tag in tags if tag in ['N', 'Np', 'Nc', 'Nu'])
    features['verb_count'] = sum(1 for word, tag in tags if tag in ['V', 'Vy'])
    features['adj_count'] = sum(1 for word, tag in tags if tag in ['A', 'Adj'])
    
    return features

def preprocess_text_for_model(text):
    """Hàm làm sạch để đưa vào TF-IDF (Phải khớp với file train)"""
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'[^a-záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# =============================================================================
# 4. GIAO DIỆN STREAMLIT
# =============================================================================

st.set_page_config(page_title="Hit Predictor Pro", page_icon="🎵")
st.title("🎵 AI Dự Đoán Hit (Full Algorithm)")
st.markdown("**Core:** Librosa (Audio) + Underthesea (Lyrics/Sentiment) + Random Forest")

# Load Model
@st.cache_resource
def load_model():
    try:
        data = joblib.load('DA\pkl_file\hit_song_model_full.pkl')
        return data['pipeline'], data['audio_features']
    except:
        return None, None

pipeline, model_audio_cols = load_model()

if pipeline is None:
    st.error("⚠️ Không tìm thấy file model `hit_song_model_full.pkl`.")
    st.stop()

# Input
col1, col2 = st.columns(2)
with col1:
    uploaded_audio = st.file_uploader("1. Upload nhạc (.mp3, .wav)", type=["mp3", "wav"])
with col2:
    lyrics_text = st.text_area("2. Nhập lời bài hát (Tiếng Việt)", height=150)

if st.button("🚀 PHÂN TÍCH & DỰ ĐOÁN", type="primary"):
    if not uploaded_audio or not lyrics_text.strip():
        st.warning("Vui lòng nhập đủ thông tin!")
    else:
        with st.status("🔍 Đang chạy thuật toán phân tích...", expanded=True) as status:
            try:
                # B1. Xử lý Audio (Dùng hàm của bạn)
                st.write("🎧 Trích xuất đặc trưng Audio (Librosa)...")
                with open("temp_app.mp3", "wb") as f:
                    f.write(uploaded_audio.getbuffer())
                
                audio_feats = extract_audio_features_custom("temp_app.mp3")
                if audio_feats is None:
                    st.stop()

                # B2. Xử lý Lyrics (Dùng hàm của bạn)
                st.write("📝 Phân tích ngữ nghĩa & cảm xúc (Underthesea)...")
                lyric_feats = analyze_lyrics_custom(lyrics_text)
                
                # B3. Tổng hợp dữ liệu
                # Gộp 2 dictionary lại
                full_feats = {**audio_feats, **lyric_feats}
                
                # Thêm cột clean_lyric cho TF-IDF
                full_feats['clean_lyric'] = preprocess_text_for_model(lyrics_text)
                
                # Tạo DataFrame
                df_input = pd.DataFrame([full_feats])
                
                # B4. Lọc cột để khớp với Model
                # (Model cần các cột MFCC, RMS... và clean_lyric)
                # Lưu ý: Các cột tính ra thừa (như sentiment_score) nếu model không dùng thì pipeline sẽ tự lờ đi
                # Nhưng các cột thiếu (nếu có) phải được điền 0
                for col in model_audio_cols:
                    if col not in df_input.columns:
                        df_input[col] = 0.0
                
                # B5. Dự đoán
                st.write("🔮 Đang chạy mô hình Random Forest...")
                pred = pipeline.predict(df_input)[0]
                proba = pipeline.predict_proba(df_input)[0]
                
                os.remove("temp_app.mp3")
                status.update(label="✅ Hoàn tất!", state="complete", expanded=False)
                
                # B6. Hiển thị
                st.divider()
                c1, c2 = st.columns([1, 1.5])
                with c1:
                    if pred == 1:
                        st.success(f"### 🌟 DỰ ĐOÁN: HIT\nĐộ tin cậy: {proba[1]*100:.1f}%")
                    else:
                        st.error(f"### 💤 DỰ ĐOÁN: NON-HIT\nĐộ tin cậy: {proba[0]*100:.1f}%")
                
                with c2:
                    st.caption("Thông số phân tích:")
                    st.write(f"- **Cảm xúc (Sentiment):** {lyric_feats['sentiment']} (Score: {lyric_feats['sentiment_score']})")
                    st.write(f"- **Từ vựng (Lexical Diversity):** {lyric_feats['lexical_diversity']:.2f}")
                    st.write(f"- **Năng lượng (RMS):** {audio_feats['rms_energy']:.4f}")
                    st.write(f"- **Tempo:** {audio_feats['tempo_bpm']:.1f} BPM")

            except Exception as e:
                st.error(f"Lỗi: {e}")
                print(e)