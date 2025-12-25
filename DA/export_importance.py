import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# =============================================================================
# CẤU HÌNH
# =============================================================================
MODEL_PATH = 'DA/pkl_file/task_1_PB_RF.pkl'
CSV_OUTPUT = 'feature_importance_full.csv'   # Tên file Excel xuất ra
IMG_OUTPUT = 'feature_importance_chart.png'  # Tên ảnh biểu đồ xuất ra

def main():
    print("="*60)
    print("📊 CÔNG CỤ XUẤT BIỂU ĐỒ & SỐ LIỆU CHI TIẾT")
    print("="*60)

    # 1. LOAD MODEL
    if not os.path.exists(MODEL_PATH):
        print(f"❌ LỖI: Không tìm thấy file model tại '{MODEL_PATH}'")
        return

    print(f"⏳ Đang load model từ: {MODEL_PATH}...")
    try:
        model_data = joblib.load(MODEL_PATH)
        print("   ✓ Load thành công!")
    except Exception as e:
        print(f"❌ Lỗi đọc file model: {e}")
        return

    # 2. LẤY DỮ LIỆU
    pipeline = model_data.get('pipeline')
    audio_cols = model_data.get('audio_features', [])
    # Các cột ngôn ngữ (nếu model cũ không có key này thì dùng mặc định)
    linguistic_cols = model_data.get('linguistic_cols', ['sentiment_score', 'noun_ratio', 'verb_ratio', 'adj_ratio'])

    # 3. TẠO DANH SÁCH TÊN FEATURES ĐẦU VÀO
    # Thứ tự phải khớp lúc train: [Audio] + [Linguistic] + [PhoBERT]
    feature_names = []
    feature_names.extend(audio_cols)
    feature_names.extend(linguistic_cols)
    feature_names.extend([f'phobert_dim_{i+1}' for i in range(768)]) # 768 chiều PhoBERT
    
    all_features = np.array(feature_names)

    # 4. LẤY FEATURES ĐƯỢC GIỮ LẠI (SAU KHI LỌC)
    try:
        if 'feature_selection' in pipeline.named_steps:
            selector = pipeline.named_steps['feature_selection']
            selected_mask = selector.get_support()
            final_features = all_features[selected_mask]
        else:
            final_features = all_features
            
        # 5. LẤY ĐIỂM SỐ (IMPORTANCE)
        clf = pipeline.named_steps['classifier']
        importances = clf.feature_importances_
        
        # Tạo DataFrame
        df_imp = pd.DataFrame({
            'Feature': final_features,
            'Importance': importances
        })
        
        # Sắp xếp giảm dần
        df_imp = df_imp.sort_values(by='Importance', ascending=False)
        
        # ---------------------------------------------------------
        # A. XUẤT FILE CSV (TOÀN BỘ CÁC CỘT)
        # ---------------------------------------------------------
        df_imp.to_csv(CSV_OUTPUT, index=False)
        print(f"\n💾 Đã lưu danh sách chi tiết vào: {CSV_OUTPUT}")
        
        # ---------------------------------------------------------
        # B. VẼ BIỂU ĐỒ (TOP 20 QUAN TRỌNG NHẤT)
        # ---------------------------------------------------------
        print("\n🎨 Đang vẽ biểu đồ...")
        plt.figure(figsize=(12, 10))
        
        # Lấy Top 20 để vẽ cho đẹp
        top_20 = df_imp.head(20)
        
        sns.barplot(data=top_20, x='Importance', y='Feature', palette='viridis')
        plt.title('Top 20 Yếu Tố Quyết Định (Feature Importance)', fontsize=15)
        plt.xlabel('Mức độ ảnh hưởng', fontsize=12)
        plt.ylabel('Tên đặc trưng', fontsize=12)
        plt.tight_layout()
        
        # Lưu thành file ảnh (để chắc chắn xem được)
        plt.savefig(IMG_OUTPUT, dpi=300)
        print(f"🖼️  Đã lưu ảnh biểu đồ vào: {IMG_OUTPUT}")
        
        # Thử hiển thị cửa sổ (nếu môi trường hỗ trợ)
        try:
            plt.show()
        except:
            pass

        print("\n✅ HOÀN THÀNH! Hãy kiểm tra file ảnh và file csv trong thư mục.")

    except Exception as e:
        print(f"\n❌ Lỗi khi xử lý: {e}")

if __name__ == "__main__":
    main()