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
try:
    from mutagen.mp3 import MP3
    from mutagen.id3 import ID3, USLT
    MUTAGEN_AVAILABLE = True
except ImportError:
    MUTAGEN_AVAILABLE = False

warnings.filterwarnings('ignore')

# =============================================================================
# 1. CẤU HÌNH & TỪ ĐIỂN V-POP
# =============================================================================
SR = 22050
N_FFT = 2048
HOP_LENGTH = 512

# Từ điển V-Pop
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
    'đéo', 'đếch', 'dek', 'nào', 'làm gì', 'đâu có', 'có đâu'
]

# =============================================================================
# 2. HÀM XỬ LÝ AUDIO (FULL FEATURES)
# =============================================================================

def extract_audio_features_custom(audio_path):
    """Trích xuất Audio đầy đủ theo chuẩn extract_feature.py"""
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
            else: 
                features['tempo_stability'] = 0.0
        else: 
            features['tempo_stability'] = 0.0

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
# 3. HÀM XỬ LÝ LYRICS & SENTIMENT (ADVANCED)
# =============================================================================

def analyze_lyrics_custom(lyric_text):
    """Phân tích lyrics với Underthesea + Custom Dictionary + POS Tagging"""
    features = {}
    
    if not lyric_text or not isinstance(lyric_text, str):
        return features

    # A. SENTIMENT ANALYSIS
    features['sentiment'] = 'neutral'
    features['sentiment_score'] = 0.0
    features['sentiment_confidence'] = 0.0
    
    # 1. Underthesea Sentiment
    try:
        underthesea_result = uts_sentiment(lyric_text)
        features['sentiment'] = underthesea_result
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
            if has_negation: 
                neg_score += 1
            else: 
                pos_score += 1
        elif any(neg_word in word for neg_word in NEGATIVE_WORDS):
            if has_negation: 
                pos_score += 0.5
            else: 
                neg_score += 1
            
    total_score = pos_score + neg_score
    if total_score > 0:
        sentiment_ratio = (pos_score - neg_score) / total_score
        dict_score = round(sentiment_ratio, 4)
        dict_conf = round(total_score / (len(words) + 1), 4)
        
        if features['sentiment'] == 'neutral':
            if sentiment_ratio > 0.2:
                features['sentiment'] = 'positive'
            elif sentiment_ratio < -0.2:
                features['sentiment'] = 'negative'
        
        features['sentiment_score'] = dict_score
        features['sentiment_confidence'] = dict_conf

    # B. LINGUISTIC STATS (POS Tagging)
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
    """Làm sạch text cho TF-IDF"""
    if not isinstance(text, str): 
        return ""
    text = text.lower()
    text = re.sub(r'[^a-záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_lyrics_from_audio_metadata(audio_path):
    """Trích xuất lyrics từ metadata ID3 của file audio"""
    if not MUTAGEN_AVAILABLE:
        return None
    
    try:
        audio = MP3(audio_path, ID3=ID3)
        
        # Tìm USLT tag (Unsynchronized Lyrics)
        for tag in audio.tags.values():
            if isinstance(tag, USLT):
                lyrics = tag.text
                if lyrics and len(lyrics.strip()) > 10:
                    return lyrics
        
        # Thử tìm trong các tag khác
        if 'USLT::XXX' in audio.tags:
            return str(audio.tags['USLT::XXX'])
            
    except Exception as e:
        pass
    
    return None

# =============================================================================
# 2. GIAO DIỆN CHÍNH (FRONTEND)
# =============================================================================

st.set_page_config(page_title="V-Pop Hit Predictor AI", page_icon="🎧", layout="wide")

# Sidebar
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/19/Spotify_logo_without_text.svg/2048px-Spotify_logo_without_text.svg.png", width=50)
    st.title("Cấu Hình Dự Đoán")
    
    # Load Model
    try:
        data = joblib.load('DA\pkl_file\hit_song_model_full.pkl')
        pipeline = data['pipeline']
        model_audio_cols = data['audio_features']
        st.success("✅ Model Active: Random Forest")
    except:
        st.error("❌ Lỗi: Không tìm thấy file `hit_song_model_full.pkl`")
        st.stop()
    
    st.info("""
    **Hướng dẫn:**
    1. Upload file nhạc (.mp3)
    2. Nhập lời bài hát (Tiếng Việt)
    3. Nhấn 'Phân tích'
    
    **Công nghệ:**
    - Audio: Librosa (63 features)
    - Lyrics: Underthesea + V-Pop Dictionary
    - Model: Random Forest + TF-IDF
    """)

# Main Title
st.title("🎧 Hệ Thống Dự Đoán & Phân Tích Hit Nhạc Việt")
st.markdown("---")

# TẠO TABS
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔮 DỰ ĐOÁN HIT (Main)", 
    "📈 Dự đoán Doanh thu (P1)", 
    "🎨 Phân loại Phong cách (P2)", 
    "❤️ Phân tích Cảm xúc (P3)", 
    "guitar Genre Detection (P4)"
])

# --- TAB 1: CHỨC NĂNG CHÍNH ---
with tab1:
    col_in1, col_in2 = st.columns([1, 1])
    
    with col_in1:
        st.subheader("1. Audio Input")
        uploaded_file = st.file_uploader("Upload file nhạc (Bắt buộc)", type=['mp3', 'wav'])
        if uploaded_file:
            st.audio(uploaded_file, format='audio/mp3')

    with col_in2:
        st.subheader("2. Lyrics Input")
        lyrics_txt = st.text_area("Lời bài hát (Tùy chọn - Tự động extract từ file nếu có)", height=150, placeholder="Dán lời bài hát vào đây...\nHoặc để trống để app tự extract từ metadata file audio")
        
        if MUTAGEN_AVAILABLE:
            st.caption("✅ Hỗ trợ tự động đọc lyrics từ ID3 metadata")
        else:
            st.caption("⚠️ Cài `mutagen` để tự động đọc lyrics: pip install mutagen")

    btn_predict = st.button("🚀 Bắt đầu Phân tích", type="primary", use_container_width=True)

    if btn_predict:
        if uploaded_file is None and (not lyrics_txt or not lyrics_txt.strip()):
            st.error("⚠️ Vui lòng upload file nhạc HOẶC nhập lời bài hát (ít nhất 1 trong 2)!")
        else:
            with st.status("🔍 Đang chạy thuật toán phân tích...", expanded=True) as status:
                try:
                    audio_feats = None
                    lyric_feats = {}
                    final_lyrics = lyrics_txt
                    
                    # 1. XỬ LÝ AUDIO (nếu có)
                    if uploaded_file is not None:
                        st.write("🎧 Trích xuất đặc trưng Audio (Librosa)...")
                        with open("temp_app.mp3", "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        # Extract audio features
                        audio_feats = extract_audio_features_custom("temp_app.mp3")
                        
                        # Thử extract lyrics từ metadata nếu user chưa nhập
                        if not final_lyrics or not final_lyrics.strip():
                            st.write("📖 Đang thử đọc lyrics từ metadata file audio...")
                            extracted_lyrics = extract_lyrics_from_audio_metadata("temp_app.mp3")
                            if extracted_lyrics:
                                final_lyrics = extracted_lyrics
                                st.success("✅ Đã tìm thấy lyrics trong file audio!")
                            else:
                                st.info("ℹ️ Không tìm thấy lyrics trong metadata file")
                    else:
                        st.info("ℹ️ Không có file audio - Chỉ phân tích lyrics")
                    
                    # 2. XỬ LÝ LYRICS (nếu có)
                    if final_lyrics and final_lyrics.strip():
                        st.write("📝 Phân tích ngữ nghĩa & cảm xúc (Underthesea)...")
                        lyric_feats = analyze_lyrics_custom(final_lyrics)
                    else:
                        st.warning("⚠️ Không có lyrics - Chỉ dùng audio features")
                    
                    # 3. TỔNG HỢP DỮ LIỆU
                    full_feats = {}
                    
                    # Thêm audio features (hoặc điền 0 nếu không có)
                    if audio_feats:
                        full_feats.update(audio_feats)
                    else:
                        # Không có audio -> Điền 0 cho tất cả audio features
                        for col in model_audio_cols:
                            full_feats[col] = 0.0
                        st.warning("⚠️ Sử dụng giá trị mặc định cho audio features (accuracy sẽ thấp hơn)")
                    
                    # Thêm lyrics features
                    full_feats.update(lyric_feats)
                    
                    # Thêm clean_lyric cho TF-IDF
                    if final_lyrics and final_lyrics.strip():
                        full_feats['clean_lyric'] = preprocess_text_for_model(final_lyrics)
                    else:
                        full_feats['clean_lyric'] = ""  # Empty string nếu không có lyrics
                        st.warning("⚠️ Không có lyrics - TF-IDF sẽ không có dữ liệu (accuracy sẽ thấp hơn)")
                    
                    # 4. Tạo DataFrame
                    df_input = pd.DataFrame([full_feats])
                    
                    # 5. Đảm bảo đủ cột cho Model
                    for col in model_audio_cols:
                        if col not in df_input.columns:
                            df_input[col] = 0.0
                    
                    # 6. Dự đoán
                    st.write("🔮 Đang chạy mô hình Random Forest...")
                    pred = pipeline.predict(df_input)[0]
                    proba = pipeline.predict_proba(df_input)[0]
                    
                    # Xóa file tạm (nếu có)
                    if uploaded_file is not None and os.path.exists("temp_app.mp3"):
                        os.remove("temp_app.mp3")
                    
                    status.update(label="✅ Hoàn tất!", state="complete", expanded=False)

                    # === HIỂN THỊ KẾT QUẢ ===
                    st.divider()
                    c_res1, c_res2 = st.columns([2, 3])
                    
                    with c_res1:
                        st.subheader("KẾT QUẢ DỰ ĐOÁN")
                        if pred == 1:
                            st.success(f"# 🌟 HIT\nĐộ tin cậy: **{proba[1]*100:.1f}%**")
                        else:
                            st.error(f"# 💤 NON-HIT\nĐộ tin cậy: **{proba[0]*100:.1f}%**")
                        st.caption("Model: Random Forest + TF-IDF")

                    with c_res2:
                        st.subheader("THÔNG SỐ PHÂN TÍCH CHI TIẾT")
                        
                        col_a, col_b = st.columns(2)
                        with col_a:
                            if audio_feats:
                                st.metric("🎵 Năng lượng (RMS)", f"{audio_feats['rms_energy']:.4f}")
                                st.metric("🥁 Tempo", f"{audio_feats['tempo_bpm']:.1f} BPM")
                            else:
                                st.metric("🎵 Năng lượng (RMS)", "N/A")
                                st.metric("🥁 Tempo", "N/A")
                        with col_b:
                            st.metric("❤️ Cảm xúc", lyric_feats.get('sentiment', 'N/A'))
                            st.metric("📊 Lexical Diversity", f"{lyric_feats.get('lexical_diversity', 0):.2f}")
                        
                        st.write("---")
                        st.caption("**Phân tích Lời bài hát:**")
                        st.write(f"- Sentiment Score: {lyric_feats.get('sentiment_score', 0):.2f}")
                        st.write(f"- Confidence: {lyric_feats.get('sentiment_confidence', 0):.2f}")
                        st.write(f"- Tổng từ: {lyric_feats.get('lyric_total_words', 0)}")
                        st.write(f"- Danh từ: {lyric_feats.get('noun_count', 0)} | Động từ: {lyric_feats.get('verb_count', 0)} | Tính từ: {lyric_feats.get('adj_count', 0)}")

                except Exception as e:
                    st.error(f"Lỗi xử lý: {e}")
                    import traceback
                    st.code(traceback.format_exc())

# --- CÁC TAB TÍNH NĂNG MỞ RỘNG (PLACEHOLDERS) ---

with tab2:
    st.header("P1: Dự đoán Độ phổ biến (Spotify Popularity)")
    st.info("🚧 Tính năng đang phát triển")
    st.markdown("""
    **Mô tả:** Dự đoán con số cụ thể (Regression) về lượt stream hoặc điểm popularity (0-100).
    - **Input:** Audio, Lyric, Tên Ca sĩ, Thể loại.
    - **Model:** Gradient Boosting Regressor / Neural Network.
    """)
    st.text_input("Nhập tên ca sĩ (Ví dụ: Sơn Tùng M-TP)", disabled=True)
    st.button("Dự báo Doanh thu", disabled=True)

with tab3:
    st.header("P2: Phân cụm Phong cách (Style Clustering)")
    st.info("🚧 Tính năng đang phát triển")
    st.markdown("""
    **Mô tả:** Gom nhóm bài hát này vào các cụm phong cách tương đồng.
    - **Ví dụ:** Nhạc Buồn Lãng Mạn, Nhạc Quẩy Vinahouse, Nhạc Chill Lofi...
    - **Thuật toán:** K-Means Clustering hoặc DBSCAN trên vector đặc trưng.
    """)

with tab4:
    st.header("P3: Phân tích Cảm xúc Đa phương thức (Multimodal Sentiment)")
    st.info("🚧 Tính năng đang phát triển")
    st.markdown("""
    **Mô tả:** Kết hợp cảm xúc từ Giọng hát (Audio Sentiment) và Lời bài hát (Text Sentiment).
    - **Audio:** Giọng vui tươi hay u sầu?
    - **Text:** Lời tích cực hay tiêu cực?
    -> **Kết luận:** Bài hát Vui, Buồn, hay Mâu thuẫn (nhạc vui lời buồn).
    """)

with tab5:
    st.header("P4: Phân loại Thể loại Nhạc (Genre Classification)")
    st.info("🚧 Tính năng đang phát triển")
    st.markdown("""
    **Mô tả:** AI tự động nghe nhạc và gắn thẻ thể loại.
    - **Output:** Pop, Ballad, Rap, Indie, EDM...
    - **Model:** Convolutional Neural Network (CNN) trên Spectrogram hình ảnh.
    """)