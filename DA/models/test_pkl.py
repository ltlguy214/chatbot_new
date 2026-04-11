import joblib
import pandas as pd

# Load thử model P2 (Hit Prediction)
model_path = 'DA/models/best_model_p2.pkl'
model_data = joblib.load(model_path)

print(f"--- KIỂM TRA MODEL: {model_data.get('model_name')} ---")

# Kiểm tra sự tồn tại của shap_cache
if 'shap_cache' in model_data and model_data['shap_cache'] is not None:
    cache = model_data['shap_cache']
    print("✅ SHAP Cache: Đã tìm thấy!")
    
    # Soi chi tiết dữ liệu nền (background)
    X_bg = cache.get('X_background')
    if X_bg is not None:
        print(f"   • Dữ liệu nền (Background): {X_bg.shape[0]} mẫu, {X_bg.shape[1]} tính năng.")
    
    # Soi chi tiết dữ liệu giải thích (explain)
    X_ex = cache.get('X_explain')
    if X_ex is not None:
        print(f"   • Dữ liệu giải thích (Explain): {X_ex.shape[0]} mẫu, {X_ex.shape[1]} tính năng.")
else:
    print("❌ SHAP Cache: Rỗng hoặc không tồn tại.")