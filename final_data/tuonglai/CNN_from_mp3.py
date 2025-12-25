import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os

def export_mel_for_cnn(file_path):
    if not os.path.exists(file_path):
        print(f"❌ Không tìm thấy file tại: {file_path}")
        return

    print(f"🎵 Đang phân tích: {file_path}...")
    
    # 1. Load âm thanh (30s)
    y, sr = librosa.load(file_path, duration=30)
    
    # 2. Tạo Mel-spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    
    # 3. Chuyển sang dB
    S_dB = librosa.power_to_db(S, ref=np.max)
    
    print(f"✅ Đã trích xuất ma trận đặc trưng: {S_dB.shape}")
    
    # 4. Vẽ và LƯU FILE ẢNH
    plt.figure(figsize=(12, 6))
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', cmap='magma')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Mel-spectrogram: {os.path.basename(file_path)}')
    plt.tight_layout()
    
    # Lưu kết quả ra file để bạn kiểm tra
    output_name = "mel_spectrogram_result.png"
    plt.savefig(output_name)
    print(f"📸 Đã lưu hình ảnh đặc trưng tại: {output_name}")
    plt.show()

# --- DÒNG QUAN TRỌNG NHẤT: BỎ DẤU # VÀ ĐIỀN ĐÚNG ĐƯỜNG DẪN ---
path = r'Audio_lyric\audio_final\Qua Khung Cửa Sổ.mp3'
export_mel_for_cnn(path)