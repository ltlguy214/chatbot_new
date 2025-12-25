import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.neural_network import MLPClassifier

# Tắt cảnh báo cho gọn
warnings.filterwarnings('ignore')

# ==============================================================================
# 1. CẤU HÌNH ĐƯỜNG DẪN (THEO LOGIC CỦA BẠN)
# ==============================================================================
TRAIN_DATA = r'final_data\merged_balanced_500_500_scaled_ml.csv'   # File đã scale -> Dùng để Train
TEST_DATA = r'final_data\final_rejected_dataset.csv'                  # File rejected -> Dùng để Test

# Các cột cần loại bỏ (Update theo dữ liệu của bạn nếu cần)
IGNORE_COLS = ['filename', 'length', 'label', 'is_hit', 'Unnamed: 0']

# Biến toàn cục lưu model và danh sách cột feature
trained_models = {}

# ==============================================================================
# 2. HÀM TRAIN (DÙNG FILE ĐÃ SCALE SẴN)
# ==============================================================================
def train_all_models():
    global trained_models
    print(f"\n{'='*40}")
    print("🚀 BẮT ĐẦU TRAIN (Dữ liệu đã Scale sẵn)")
    print(f"{'='*40}")

    try:
        # Load file đã scale
        df = pd.read_csv(TRAIN_DATA)
        print(f"✅ Đã load file train: {df.shape}")

        # Xác định X và y
        # Lọc bỏ các cột không phải feature
        feature_cols = [c for c in df.columns if c not in IGNORE_COLS]
        
        X = df[feature_cols]
        y = df['is_hit']

        # Chia train/test (chỉ để validate nội bộ)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        # Định nghĩa các model
        models_def = {
            'Logistic Regression': LogisticRegression(),
            'Random Forest': RandomForestClassifier(),
            'Gradient Boosting': GradientBoostingClassifier(),
            'XGBoost': XGBClassifier(eval_metric='logloss', verbose=0),
            'LightGBM': LGBMClassifier(verbose=-1),
            'CatBoost': CatBoostClassifier(verbose=0),
            'MLP (sklearn)': MLPClassifier(max_iter=500)
        }

        # Loop train
        for name, model in models_def.items():
            print(f"⏳ Training {name}...")
            model.fit(X_train, y_train)
            trained_models[name] = {
                'model': model,
                'score': model.score(X_val, y_val)
            }
        
        # LƯU LẠI DANH SÁCH FEATURE ĐỂ DÙNG KHI TEST
        trained_models['_meta'] = {'feature_cols': feature_cols}
        
        print("\n✅ ĐÃ TRAIN XONG TẤT CẢ MODEL!")
    
    except Exception as e:
        print(f"❌ Lỗi nghiêm trọng khi train: {e}")

# ==============================================================================
# 3. HÀM TEST (LOGIC REVERSE-ENGINEERING CỦA BẠN)
# ==============================================================================
def test_random_songs():
    if not trained_models:
        print("❌ Chưa có model! Vui lòng chọn Option 1 để train trước.")
        return

    # Lấy danh sách cột feature mà model đã học
    feature_cols = trained_models['_meta']['feature_cols']

    print(f"\n{'='*80}")
    print("🔄 ĐANG CHUẨN BỊ DỮ LIỆU TEST (Fit Scaler gốc -> Transform Rejected)")
    
    scaler = StandardScaler()

    # --- BƯỚC A: FIT SCALER TRÊN FILE GỐC ---
    try:
        df_orig = pd.read_csv(TRAIN_DATA_ORIGINAL)
        # Chỉ lấy đúng các cột feature, fillna 0 để an toàn
        X_orig = df_orig[feature_cols].fillna(0)
        
        scaler.fit(X_orig) # <--- MẤU CHỐT: Học quy luật từ file gốc
        print(f"✅ Đã fit Scaler trên {len(df_orig)} dòng dữ liệu gốc.")
    except Exception as e:
        print(f"❌ Lỗi khi đọc file gốc ({TRAIN_DATA_ORIGINAL}): {e}")
        return

    # --- BƯỚC B: LOAD VÀ TRANSFORM FILE REJECTED ---
    try:
        df_test = pd.read_csv(TEST_DATA)
        X_test_raw = df_test[feature_cols].fillna(0)
        
        # Áp dụng Scaler vừa học được lên file rejected
        X_test_scaled_vals = scaler.transform(X_test_raw)
        
        # Đưa về DataFrame để dễ xử lý
        X_test_scaled = pd.DataFrame(X_test_scaled_vals, columns=feature_cols)
        print(f"✅ Đã transform {len(df_test)} bài hát rejected.")
    except Exception as e:
        print(f"❌ Lỗi khi đọc/xử lý file test ({TEST_DATA}): {e}")
        return

    # --- BƯỚC C: CHỌN NGẪU NHIÊN VÀ DỰ ĐOÁN ---
    while True:
        # Lấy random 5 bài
        random_indices = np.random.choice(df_test.index, size=min(5, len(df_test)), replace=False)
        random_songs = df_test.loc[random_indices].reset_index(drop=True)
        random_inputs = X_test_scaled.loc[random_indices].reset_index(drop=True)

        print(f"\n{'-'*80}")
        print("🎧 DANH SÁCH NGẪU NHIÊN TỪ REJECTED DATASET:")
        for i, row in random_songs.iterrows():
            print(f"[{i}] {str(row.get('title', 'No Title'))[:30]:<30} - {str(row.get('artists', 'Unknown'))[:20]}")
        
        choice = input("\n>>> Chọn số [0-4], 'r' để random lại, 'q' để thoát: ").strip().lower()
        
        if choice == 'q': break
        if choice == 'r': continue
        
        try:
            idx = int(choice)
            if 0 <= idx < len(random_songs):
                # Dữ liệu đầu vào cho model
                input_vector = random_inputs.iloc[idx].to_frame().T
                
                print(f"\n🎶 KẾT QUẢ DỰ ĐOÁN CHO: {random_songs.iloc[idx].get('title', 'Unknown')}")
                print("-" * 60)
                
                for name, info in trained_models.items():
                    if name == '_meta': continue # Bỏ qua biến meta
                    
                    model = info['model']
                    pred = model.predict(input_vector)[0]
                    proba = model.predict_proba(input_vector)[0].max() * 100 if hasattr(model, 'predict_proba') else 0
                    
                    res_str = "🔥 HIT" if pred == 1 else "🌑 NON-HIT"
                    print(f"{name:<25} | {res_str:<12} | Conf: {proba:.1f}%")
            else:
                print("❌ Số không hợp lệ.")
        except ValueError:
            print("❌ Vui lòng nhập số.")

# ==============================================================================
# 4. CHẠY CHƯƠNG TRÌNH
# ==============================================================================
if __name__ == "__main__":
    while True:
        print(f"\n{'='*30}")
        print("MENU CHÍNH")
        print("1. Train Models (Load file scaled)")
        print("2. Test Random Songs (Load file gốc fit scaler -> Test)")
        print("0. Thoát")
        
        opt = input(">>> Chọn: ")
        if opt == '1':
            train_all_models()
        elif opt == '2':
            test_random_songs()
        elif opt == '0':
            break