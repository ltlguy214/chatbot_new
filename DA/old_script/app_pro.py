import streamlit as st
import pandas as pd
import numpy as np
import librosa
import joblib
import re
import os
import warnings
from underthesea import sentiment as uts_sentiment

warnings.filterwarnings('ignore')

# =============================================================================
# 1. HÀM XỬ LÝ (BACKEND)
# =============================================================================

def preprocess_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'[^a-záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_audio_features(audio_path, required_features):
    try:
        y, sr = librosa.load(audio_path, duration=60, offset=30)
    except:
        y, sr = librosa.load(audio_path) # Fallback nếu file ngắn

    features = {}
    # Cơ bản
    features['rms_energy'] = np.mean(librosa.feature.rms(y=y))
    features['zero_crossing_rate'] = np.mean(librosa.feature.zero_crossing_rate(y))
    features['duration_sec'] = librosa.get_duration(y=y, sr=sr)
    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        features['tempo_bpm'] = float(tempo)
    except: features['tempo_bpm'] = 120.0

    # Spectral
    features['spectral_centroid_mean'] = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    features['spectral_flatness_mean'] = np.mean(librosa.feature.spectral_flatness(y=y))
    features['spectral_rolloff'] = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    
    # MFCC (1-13)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    for i in range(1, 14):
        features[f'mfcc{i}_mean'] = np.mean(mfccs[i])
        features[f'mfcc{i}_std'] = np.std(mfccs[i])

    # Chroma & Contrast
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    for i in range(12): features[f'chroma{i+1}_mean'] = np.mean(chroma[i])
    
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    for i in range(7): features[f'spectral_contrast_band{i+1}_mean'] = np.mean(contrast[i])

    # Khớp features với model
    input_data = {}
    for col in required_features:
        input_data[col] = features.get(col, 0.0)
    
    return pd.DataFrame([input_data])

# =============================================================================
# 2. GIAO DIỆN CHÍNH (FRONTEND)
# =============================================================================

st.set_page_config(page_title="V-Pop Hit Predictor AI", page_icon="🎧", layout="wide")

# Sidebar
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/19/Spotify_logo_without_text.svg/2048px-Spotify_logo_without_text.svg.png", width=50)
    st.title("Cấu Hình Dự Đoán")
    
    # Load Model Dictionary
    try:
        data_pkg = joblib.load('DA\pkl_file\multi_model_hit_prediction.pkl')
        models_dict = data_pkg['models']
        audio_cols = data_pkg['audio_features']
        st.success("✅ Model Core: Active")
    except:
        st.error("❌ Lỗi: Chưa có file model `.pkl`")
        st.stop()

    # Chọn thuật toán
    selected_model_name = st.selectbox(
        "Chọn Thuật toán (Algorithm):",
        list(models_dict.keys()),
        index=0
    )
    
    st.info("""
    **Hướng dẫn:**
    1. Upload file nhạc (.mp3)
    2. Nhập lời bài hát (nếu có)
    3. Nhấn 'Phân tích'
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
        lyrics_txt = st.text_area("Lời bài hát (Tùy chọn)", height=150, placeholder="Dán lời bài hát vào đây...")

    btn_predict = st.button("🚀 Bắt đầu Phân tích", type="primary", use_container_width=True)

    if btn_predict:
        if uploaded_file is None:
            st.error("⚠️ Vui lòng upload file nhạc âm thanh!")
        else:
            with st.spinner(f"Đang chạy thuật toán {selected_model_name}..."):
                try:
                    # 1. Lưu file tạm
                    with open("temp.mp3", "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # 2. Trích xuất Audio Features
                    df_features = extract_audio_features("temp.mp3", audio_cols)
                    
                    # 3. Xử lý Lyric
                    clean_text = preprocess_text(lyrics_txt) if lyrics_txt else ""
                    df_features['clean_lyric'] = clean_text
                    
                    # 4. Dự đoán
                    pipeline = models_dict[selected_model_name]
                    # Pipeline tự động: Impute -> Scale -> TF-IDF -> Predict
                    pred = pipeline.predict(df_features)[0]
                    proba = pipeline.predict_proba(df_features)[0]
                    
                    # 5. Sentiment phụ trợ
                    try:
                        senti = uts_sentiment(lyrics_txt) if lyrics_txt else "N/A"
                    except: senti = "N/A"

                    # === HIỂN THỊ KẾT QUẢ ===
                    st.divider()
                    c_res1, c_res2 = st.columns([2, 3])
                    
                    with c_res1:
                        st.subheader("KẾT QUẢ DỰ ĐOÁN")
                        if pred == 1:
                            st.success(f"# 🌟 HIT\nĐộ tin cậy: **{proba[1]*100:.2f}%**")
                        else:
                            st.error(f"# 💤 NON-HIT\nĐộ tin cậy: **{proba[0]*100:.2f}%**")
                        st.caption(f"Model sử dụng: {selected_model_name}")

                    with c_res2:
                        st.subheader("THÔNG SỐ ĐÃ PHÂN TÍCH (Extracted Features)")
                        
                        # Hiển thị DataFrame rút gọn
                        display_cols = ['rms_energy', 'tempo_bpm', 'spectral_centroid_mean', 'mfcc1_mean']
                        st.dataframe(df_features[display_cols].style.format("{:.4f}"))
                        
                        st.write("---")
                        metric1, metric2, metric3 = st.columns(3)
                        metric1.metric("Năng lượng (RMS)", f"{df_features['rms_energy'].values[0]:.4f}")
                        metric2.metric("Tempo (BPM)", f"{df_features['tempo_bpm'].values[0]:.1f}")
                        metric3.metric("Cảm xúc Lời", senti)

                    # Xóa file tạm
                    os.remove("temp.mp3")

                except Exception as e:
                    st.error(f"Lỗi xử lý: {e}")

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