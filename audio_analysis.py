import os
import librosa
import numpy as np
import pandas as pd
import warnings
from pydub import AudioSegment
import tempfile
import soundfile as sf
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pyloudnorm as pyln

warnings.filterwarnings('ignore')

def extract_features(file_path):
    """Hàm hỗ trợ trích xuất đặc trưng âm thanh với xử lý lỗi tốt hơn."""
    
    # Lấy độ dài thực của file TRƯỚC KHI load
    try:
        info = sf.info(file_path)
        actual_duration_seconds = info.duration
        length_minutes = actual_duration_seconds / 60
    except:
        try:
            audio = AudioSegment.from_file(file_path)
            actual_duration_seconds = len(audio) / 1000.0
            length_minutes = actual_duration_seconds / 60
        except:
            length_minutes = None
    
    # Load audio - TỐI ƯU HÓA với sr=22050, mono=True, offset=0, duration=60
    try:
        # Tối ưu: sr=22050 (đủ cho analysis), mono=True (giảm dung lượng), duration=60 (chỉ lấy 60s)
        y, sr = librosa.load(file_path, sr=22050, mono=True, offset=0.0, duration=60)
        print(f"  ✓ Đọc thành công (librosa)")
        
    except Exception as e1:
        print(f"  ⚠ Librosa lỗi: {str(e1)[:50]}")
        
        try:
            print(f"  → Thử pydub...")
            audio = AudioSegment.from_file(file_path)
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                audio.export(tmp.name, format='wav')
                tmp_path = tmp.name
            
            y, sr = librosa.load(tmp_path, sr=22050, mono=True, offset=0.0, duration=60)
            os.unlink(tmp_path)
            print(f"  ✓ Đọc thành công (pydub)")
            
        except Exception as e2:
            print(f"  ✗ Pydub cũng lỗi: {str(e2)[:50]}")
            
            try:
                print(f"  → Thử load toàn bộ file...")
                y, sr = librosa.load(file_path, sr=22050, mono=True)
                if len(y) > sr * 60:
                    y = y[:sr * 60]
                print(f"  ✓ Đọc thành công (full load)")
                
            except Exception as e3:
                print(f"  ✗ THẤT BẠI HOÀN TOÀN")
                print(f"     Lý do: {str(e3)[:80]}")
                return [None] * 11  # 11 features (added LUFS)
    
    # Trích xuất đặc trưng
    try:
        # 1. Tempo (BPM)
        try:
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
            tempo_val = float(tempo[0] if isinstance(tempo, np.ndarray) else tempo)
        except:
            tempo_val = 120.0
        
        # 2. RMS - Cần cho cả LUFS và Energy
        rms = librosa.feature.rms(y=y)
        
        # 3. Loudness (LUFS) - CHUẨN ITU-R BS.1770-4 cho broadcast
        try:
            # LUFS chính xác hơn RMS/dB đơn giản
            meter = pyln.Meter(sr)  # Tạo loudness meter
            loudness_lufs = meter.integrated_loudness(y)  # Tính LUFS
            # Fallback nếu giá trị không hợp lệ
            if not np.isfinite(loudness_lufs):
                loudness_lufs = -23.0  # Standard target
        except:
            # Fallback về RMS nếu pyloudnorm lỗi
            loudness_avg_rms = float(np.mean(rms))
            loudness_lufs = 20 * np.log10(loudness_avg_rms + 1e-10)
        
        # 4. Energy (năng lượng) - Tính từ RMS
        try:
            # Sử dụng mean của RMS thay vì sum để có giá trị ổn định hơn
            rms_mean = float(np.mean(rms))
            # Normalize về scale 0-1 (RMS thường trong khoảng 0-0.3)
            energy = min(1.0, rms_mean / 0.3)
        except:
            energy = 0.5
        
        # 5. Danceability (khả năng nhảy) - Dựa vào tempo stability và beat strength
        try:
            # Tempogram - đo độ ổn định của nhịp
            tempogram = librosa.feature.tempogram(y=y, sr=sr)
            tempo_stability = 1 - np.std(tempogram) / (np.mean(tempogram) + 1e-8)
            
            # Beat strength
            beat_frames = librosa.beat.beat_track(y=y, sr=sr)[1]
            beat_strength = len(beat_frames) / (len(y) / sr) if len(beat_frames) > 0 else 0
            
            # Combine
            danceability = min(1.0, (tempo_stability * 0.6 + min(beat_strength / 3, 1) * 0.4))
        except:
            danceability = 0.5
        
        # 6. Valence (cảm xúc tích cực) - Dựa vào brightness và harmony
        try:
            # Spectral centroid (brightness) - cao hơn = vui hơn
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            brightness = np.mean(spectral_centroids) / (sr / 2)  # normalize
            
            # Chromagram - harmonic content
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            harmony = np.mean(chroma)
            
            valence = min(1.0, (brightness * 0.6 + harmony * 0.4))
        except:
            valence = 0.5
        
        # 7. Liveness - Dựa vào spectral variance và noise
        try:
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            sc_variance = np.var(spectral_centroids)
            
            zcr = librosa.feature.zero_crossing_rate(y)
            zcr_mean = np.mean(zcr)
            
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            rolloff_variance = np.var(spectral_rolloff)
            
            liveness = min(1.0, (sc_variance / 1e9 + zcr_mean * 10 + rolloff_variance / 1e11) / 3)
        except:
            liveness = 0.5
        
        # 8. Instrumentalness - Dựa vào harmonic content
        try:
            y_harmonic, y_percussive = librosa.effects.hpss(y)
            
            harmonic_energy = np.sum(y_harmonic ** 2)
            total_energy = np.sum(y ** 2)
            
            contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            contrast_mean = np.mean(contrast)
            
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_var = np.var(mfccs)
            
            harmonic_ratio = harmonic_energy / (total_energy + 1e-8)
            instrumentalness = min(1.0, (harmonic_ratio * 0.5 + contrast_mean / 50 + (1 - min(mfcc_var / 100, 1)) * 0.3))
        except:
            instrumentalness = 0.5
        
        # 9. Acousticness (tính acoustic) - Dựa vào harmonic vs percussive
        try:
            y_harmonic, y_percussive = librosa.effects.hpss(y)
            
            harmonic_energy = np.sum(y_harmonic ** 2)
            percussive_energy = np.sum(y_percussive ** 2)
            total = harmonic_energy + percussive_energy
            
            # Acoustic có nhiều harmonic hơn percussive
            acousticness = harmonic_energy / (total + 1e-8)
        except:
            acousticness = 0.5
        
        # 10. Speechiness (lượng lời hát) - Dựa vào MFCC patterns
        try:
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            
            # Speech có MFCC variance cao trong các band đầu (1-4)
            mfcc_var = np.var(mfccs, axis=1)
            # Normalize pattern score
            low_band_var = np.mean(mfcc_var[1:4])
            high_band_var = np.mean(mfcc_var[4:])
            mfcc_pattern = min(1.0, low_band_var / (high_band_var + 50))  # Normalize properly
            
            # Zero crossing rate cao = nhiều consonants (speech characteristic)
            zcr = librosa.feature.zero_crossing_rate(y)
            zcr_mean = np.mean(zcr)
            zcr_speech_indicator = min(1.0, zcr_mean * 10)  # Normalize to 0-1
            
            # Combine với weight hợp lý
            speechiness = min(1.0, mfcc_pattern * 0.5 + zcr_speech_indicator * 0.5)
        except:
            speechiness = 0.5
        
        # Nếu không lấy được length từ metadata
        if length_minutes is None:
            length_minutes = len(y) / sr / 60
        
        return (
            length_minutes, 
            tempo_val, 
            loudness_lufs,  # LUFS thay vì dB
            energy,
            danceability,
            valence,
            liveness, 
            instrumentalness,
            acousticness,
            speechiness
        )
        
    except Exception as e:
        print(f"  ✗ Lỗi khi trích xuất đặc trưng: {str(e)[:60]}")
        return [None] * 11  # 11 features

def visualize_features(df, output_dir):
    """Vẽ các biểu đồ visualization cho đặc trưng âm thanh."""
    
    print("\n" + "="*60)
    print("BẮT ĐẦU VẼ BIỂU ĐỒ VISUALIZATION")
    print("="*60)
    
    # Tạo thư mục output nếu chưa có
    viz_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    # Các cột đặc trưng (thang 0-100)
    feature_cols = ['Energy', 'Danceability', 'Happiness', 'Liveness', 
                    'Instrumentalness', 'Acousticness', 'Speechiness']
    
    # Lọc bỏ các dòng có giá trị null
    df_viz = df[feature_cols].dropna()
    
    if len(df_viz) == 0:
        print("⚠ Không có dữ liệu để vẽ!")
        return
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    # 1. RADAR CHART (Spider Chart) - Trung bình các đặc trưng
    print("\n1. Đang vẽ Radar Chart...")
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Tính trung bình
    values = df_viz[feature_cols].mean().values
    angles = np.linspace(0, 2 * np.pi, len(feature_cols), endpoint=False).tolist()
    values = np.concatenate((values, [values[0]]))  # Close the circle
    angles += angles[:1]
    
    ax.plot(angles, values, 'o-', linewidth=2, label='Trung bình', color='#FF6B6B')
    ax.fill(angles, values, alpha=0.25, color='#FF6B6B')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(feature_cols, size=10)
    ax.set_ylim(0, 100)
    ax.set_title('Radar Chart - Đặc trưng âm thanh trung bình (0-100)', size=14, pad=20, weight='bold')
    ax.grid(True)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, '01_radar_chart.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✓ Đã lưu: 01_radar_chart.png")
    
    # 2. DISTRIBUTION PLOTS - Phân phối từng đặc trưng
    print("\n2. Đang vẽ Distribution Plots...")
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    
    for idx, col in enumerate(feature_cols):
        sns.histplot(df_viz[col], kde=True, ax=axes[idx], color='skyblue', bins=30)
        axes[idx].set_title(f'Phân phối {col}', fontsize=12, weight='bold')
        axes[idx].set_xlabel('')
        axes[idx].axvline(df_viz[col].mean(), color='red', linestyle='--', label=f'Mean: {df_viz[col].mean():.2f}')
        axes[idx].legend()
    
    # Ẩn subplot thừa
    for idx in range(len(feature_cols), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Phân phối các đặc trưng âm thanh', fontsize=16, weight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, '02_distribution_plots.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✓ Đã lưu: 02_distribution_plots.png")
    
    # 3. CORRELATION HEATMAP
    print("\n3. Đang vẽ Correlation Heatmap...")
    fig, ax = plt.subplots(figsize=(12, 10))
    
    corr = df_viz[feature_cols].corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title('Ma trận tương quan giữa các đặc trưng', fontsize=14, weight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, '03_correlation_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✓ Đã lưu: 03_correlation_heatmap.png")
    
    # 4. BOX PLOTS - So sánh phân phối
    print("\n4. Đang vẽ Box Plots...")
    fig, ax = plt.subplots(figsize=(14, 8))
    
    df_viz_melted = df_viz[feature_cols].melt(var_name='Feature', value_name='Value')
    sns.boxplot(data=df_viz_melted, x='Feature', y='Value', ax=ax, palette='Set2')
    ax.set_title('Box Plot - So sánh phân phối các đặc trưng', fontsize=14, weight='bold')
    ax.set_xlabel('Đặc trưng', fontsize=12)
    ax.set_ylabel('Giá trị (0-1)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, '04_box_plots.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✓ Đã lưu: 04_box_plots.png")
    
    # 5. SCATTER MATRIX (Pairplot) - Mối quan hệ giữa các đặc trưng
    print("\n5. Đang vẽ Scatter Matrix (có thể mất vài giây)...")
    
    # Chọn một số đặc trưng quan trọng để tránh quá nhiều subplot
    important_features = ['Energy', 'Danceability', 'Happiness', 'Speechiness']
    pairplot = sns.pairplot(df_viz[important_features], diag_kind='kde', plot_kws={'alpha': 0.6})
    pairplot.fig.suptitle('Scatter Matrix - Mối quan hệ giữa các đặc trưng', 
                          fontsize=14, weight='bold', y=1.01)
    
    plt.savefig(os.path.join(viz_dir, '05_scatter_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✓ Đã lưu: 05_scatter_matrix.png")
    
    # 6. BAR CHART - Top songs theo từng đặc trưng
    print("\n6. Đang vẽ Top Songs Charts...")
    
    if 'Title' in df.columns:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        top_features = ['Energy', 'Danceability', 'Happiness', 'Speechiness']
        
        for idx, feature in enumerate(top_features):
            top_10 = df.nlargest(10, feature)[['Title', feature]].dropna()
            
            if len(top_10) > 0:
                axes[idx].barh(range(len(top_10)), top_10[feature].values, color='coral')
                axes[idx].set_yticks(range(len(top_10)))
                axes[idx].set_yticklabels([t[:30] + '...' if len(t) > 30 else t for t in top_10['Title'].values], 
                                         fontsize=9)
                axes[idx].set_xlabel('Giá trị', fontsize=10)
                axes[idx].set_title(f'Top 10 bài hát có {feature} cao nhất', fontsize=12, weight='bold')
                axes[idx].invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, '06_top_songs.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("   ✓ Đã lưu: 06_top_songs.png")
    
    print("\n" + "="*60)
    print(f"✓ HOÀN THÀNH! Đã tạo 6 biểu đồ visualization")
    print(f"📁 Thư mục: {viz_dir}")
    print("="*60 + "\n")

# ================================================================
# CẤU HÌNH
# ================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_FOLDER = r"H:\LTDH - UIT\HK3\BigData\hihi\Data"
MASTER_CSV = os.path.join(SCRIPT_DIR, "master_song_list.csv")
OUTPUT_CSV = os.path.join(SCRIPT_DIR, "master_song_list_1.csv")
LOG_FILE = os.path.join(SCRIPT_DIR, "processing_log.txt")
ERROR_LOG = os.path.join(SCRIPT_DIR, "error_details.txt")
# ================================================================

print(f"Thư mục làm việc: {SCRIPT_DIR}")
print(f"Đang đọc file {MASTER_CSV}...")

try:
    df_master = pd.read_csv(MASTER_CSV)
    print(f"Đã đọc {len(df_master)} bản ghi từ CSV gốc.")
except FileNotFoundError:
    print(f"LỖI: Không tìm thấy file {MASTER_CSV}")
    exit()

if not os.path.exists(AUDIO_FOLDER):
     print(f"LỖI: Thư mục âm thanh không tồn tại: {AUDIO_FOLDER}")
     exit()

audio_data = []
success_count = 0
fail_count = 0
log_lines = []
error_details = []

print(f"\nBắt đầu quét thư mục: {AUDIO_FOLDER}\n")

audio_files = [f for f in os.listdir(AUDIO_FOLDER) 
               if f.lower().endswith(('.mp3', '.wav', '.flac', '.m4a'))]
total_files = len(audio_files)

# Sử dụng tqdm cho progress bar
for filename in tqdm(audio_files, desc="Đang xử lý audio", unit="file"):
    file_path = os.path.join(AUDIO_FOLDER, filename)
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    
    tqdm.write(f"\n📁 {filename} ({file_size_mb:.2f} MB)")
    
    features = extract_features(file_path)
    length, tempo, loudness, energy, danceability, valence, liveness, instrumentalness, acousticness, speechiness = features
    
    if length is not None:
        name_only = os.path.splitext(filename)[0]
        audio_data.append({
            'Match_Key': name_only,
            'Length_minutes': round(length, 2),
            'Tempo': round(tempo, 1),
            'Loudness': round(loudness, 2),  # Loudness trong LUFS (chuẩn ITU-R BS.1770-4)
            'Energy': round(energy * 100, 1),
            'Danceability': round(danceability * 100, 1),
            'Happiness': round(valence * 100, 1),
            'Liveness': round(liveness * 100, 1),
            'Instrumentalness': round(instrumentalness * 100, 1),
            'Acousticness': round(acousticness * 100, 1),
            'Speechiness': round(speechiness * 100, 1)
        })
        success_count += 1
        log_lines.append(
            f"✓ {filename} | "
            f"Dur: {length:.2f}min | "
            f"LUFS: {loudness:.1f} | "
            f"Energy: {energy*100:.1f} | "
            f"Dance: {danceability*100:.1f} | "
            f"Happiness: {valence*100:.1f}"
        )
        tqdm.write(f"  ✓ Thành công")
    else:
        fail_count += 1
        log_lines.append(f"✗ {filename}")
        error_details.append(f"File: {filename} | Size: {file_size_mb:.2f}MB | Status: FAILED")
        tqdm.write(f"  ✗ Thất bại")

# Lưu log
print(f"\nĐang lưu log vào: {LOG_FILE}")
with open(LOG_FILE, 'w', encoding='utf-8') as f:
    f.write('\n'.join(log_lines))
print(f"✓ Đã lưu {LOG_FILE}")

print(f"\nĐang lưu log lỗi vào: {ERROR_LOG}")
with open(ERROR_LOG, 'w', encoding='utf-8') as f:
    f.write("DANH SÁCH FILE BỊ LỖI:\n")
    f.write("="*80 + "\n\n")
    if error_details:
        f.write('\n'.join(error_details))
    else:
        f.write("Không có file nào bị lỗi!")
print(f"✓ Đã lưu {ERROR_LOG}")

print(f"\n{'='*60}")
print(f"KẾT QUẢ:")
print(f"  ✓ Thành công: {success_count}/{total_files} ({success_count/total_files*100:.1f}%)")
print(f"  ✗ Thất bại: {fail_count}/{total_files} ({fail_count/total_files*100:.1f}%)")
print(f"  → Log tổng quát: {LOG_FILE}")
print(f"  → Log lỗi chi tiết: {ERROR_LOG}")
print(f"{'='*60}\n")

# Gộp dữ liệu
if audio_data:
    df_audio = pd.DataFrame(audio_data)
    print("Đang gộp dữ liệu...")
    
    # Tạo cột tạm để so khớp không phân biệt chữ hoa/thường
    df_master['title_lower'] = df_master['title'].str.lower()
    df_audio['Match_Key_lower'] = df_audio['Match_Key'].str.lower()
    
    df_combined = pd.merge(
        df_master, 
        df_audio, 
        left_on='title_lower',
        right_on='Match_Key_lower',
        how='left'
    )
    
    # Xóa các cột tạm
    df_combined.drop(columns=['title_lower', 'Match_Key_lower'], inplace=True, errors='ignore')
    
    if 'Match_Key' in df_combined.columns:
        df_combined.drop(columns=['Match_Key'], inplace=True)
    
    # ✅ XÓA CÁC CỘT TRỐNG
    unnamed_cols = [col for col in df_combined.columns if col.startswith('Unnamed')]
    if unnamed_cols:
        print(f"\nĐang xóa {len(unnamed_cols)} cột trống: {unnamed_cols}")
        df_combined.drop(columns=unnamed_cols, inplace=True)
    
    empty_cols = df_combined.columns[df_combined.isna().all()].tolist()
    if empty_cols:
        print(f"Đang xóa {len(empty_cols)} cột rỗng hoàn toàn: {empty_cols}")
        df_combined.drop(columns=empty_cols, inplace=True)
    
    # ✅ FILTER CHỈ GIỮ CÁC CỘT YÊU CẦU
    required_cols = ['title', 'artists', 'Length_minutes', 'Tempo', 'Popularity', 
                     'Energy', 'Danceability', 'Happiness', 'Acousticness', 
                     'Instrumentalness', 'Liveness', 'Speechiness', 'Loudness']
    
    # Chỉ giữ các cột có trong dataframe
    cols_to_keep = [col for col in required_cols if col in df_combined.columns]
    df_combined = df_combined[cols_to_keep]
    
    print(f"\n✓ Đã lọc và giữ lại {len(cols_to_keep)} cột: {cols_to_keep}")
    
    df_combined.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print(f"\n✓ HOÀN TẤT! File đã lưu: {OUTPUT_CSV}")
    print(f"Tổng số cột: {len(df_combined.columns)}")
    
    # Thống kê chi tiết
    feature_cols = ['Length_minutes', 'Tempo', 'Loudness', 'Energy', 
                    'Danceability', 'Happiness', 'Liveness', 'Instrumentalness', 
                    'Acousticness', 'Speechiness']
    
    null_count = df_combined[feature_cols].isnull().sum()
    print(f"\nThống kê dữ liệu thiếu:")
    print(null_count)
    
    print(f"\nThống kê giá trị:")
    print(df_combined[feature_cols].describe())
    
    # Liệt kê các bài hát không match được
    missing_data = df_combined[df_combined['Length_minutes'].isnull()]
    if len(missing_data) > 0:
        print(f"\n{len(missing_data)} bài hát không có dữ liệu âm thanh:")
        if 'title' in missing_data.columns:
            print(missing_data['title'].head(10).to_list())
    
    # ✅ VẼ VISUALIZATION
    # visualize_features(df_combined, SCRIPT_DIR)
    
else:
    print("Không có dữ liệu để gộp.")

print(f"\n📁 Kiểm tra các file log tại: {SCRIPT_DIR}")