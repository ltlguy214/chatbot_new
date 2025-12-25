'''
Docstring for DA.task_1_multi_model
Chưa phân tích Lyrics 
RFE ✅ 
Hybrid Models ✅ 
Permutation Importance ✅

rf_selector = RandomForestClassifier(n_estimators=100, random_state=42)

BẢNG XẾP HẠNG CUỐI CÙNG:
                    Model  Accuracy      AUC
1           Random Forest     0.610  0.59875
0     Logistic Regression     0.585  0.61810
4                CatBoost     0.580  0.58630
6     Voting (RF+XGB+Cat)     0.575  0.58770
7  Stacking (RF+LGBM+Cat)     0.575  0.58560
3                LightGBM     0.565  0.54460
5          Neural Network     0.565  0.59510
2                 XGBoost     0.555  0.57040

best_model: Random Forest   Accuracy: 0.6100 | AUC: 0.5988

'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn & Metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import RFE
from sklearn.inspection import permutation_importance
from sklearn.base import BaseEstimator, ClassifierMixin

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb

# CatBoost & LightGBM
try:
    import lightgbm as lgb
    from catboost import CatBoostClassifier
except ImportError:
    print("⚠️ Cần cài đặt: pip install lightgbm catboost")
    exit()

# --- CLASS WRAPPER ĐÃ FIX (ĐỔI THỨ TỰ KẾ THỪA) ---
class SklearnCatBoostWrapper(ClassifierMixin, BaseEstimator):
    """
    Wrapper để CatBoost tương thích với VotingClassifier/StackingClassifier.
    Fix lỗi: 'The estimator should be a classifier'.
    """
    _estimator_type = "classifier"  # Khẳng định chắc chắn đây là classifier

    def __init__(self, estimator=None):
        self.estimator = estimator
        
    def fit(self, X, y):
        # Fit model gốc
        self.estimator.fit(X, y)
        # Copy các thuộc tính quan trọng để Sklearn nhận diện đã train xong
        self.classes_ = self.estimator.classes_
        return self
    
    def predict(self, X):
        return self.estimator.predict(X)
    
    def predict_proba(self, X):
        return self.estimator.predict_proba(X)
    
    def __sklearn_tags__(self):
        # Hỗ trợ Sklearn 1.6+
        tags = super().__sklearn_tags__()
        tags.estimator_type = "classifier"
        return tags

# =============================================================================
# 1. LOAD DỮ LIỆU
# =============================================================================
try:
    # Dùng file merged_balanced_500_500.csv (Raw)
    df = pd.read_csv(r'final_data\merged_balanced_500_500.csv')
    print("✅ Đã load dữ liệu.")
except:
    print("❌ Lỗi file.")
    exit()

# Xử lý ngày tháng & Lọc cột số
if 'spotify_release_date' in df.columns:
    df['release_date_dt'] = pd.to_datetime(df['spotify_release_date'], errors='coerce')
    df['days_since_release'] = (pd.Timestamp.now() - df['release_date_dt']).dt.days
    df['release_year'] = df['release_date_dt'].dt.year

df_numeric = df.select_dtypes(include=[np.number])
target_col = 'is_hit'

# Loại bỏ Leakage & Time Bias
cols_to_drop = [target_col, 'spotify_popularity', 'total_plays', 'spotify_streams', 
                'release_year', 'days_since_release', 'Unnamed: 0']
X = df_numeric.drop(columns=[c for c in cols_to_drop if c in df_numeric.columns])
y = df_numeric[target_col]

# Chia tập
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Xử lý NaN & Scale
imputer = SimpleImputer(strategy='median')
X_train_imp = imputer.fit_transform(X_train)
X_test_imp = imputer.transform(X_test)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imp)
X_test_scaled = scaler.transform(X_test_imp)

# =============================================================================
# 2. LỌC BIẾN (RFE) & FIX FEATURE NAMES
# =============================================================================
print("\n⏳ Đang lọc 20 biến quan trọng (RFE)...")
rf_selector = RandomForestClassifier(n_estimators=100, random_state=42)
selector = RFE(estimator=rf_selector, n_features_to_select=20, step=1)
selector.fit(X_train_scaled, y_train)

feature_names = X.columns
selected_cols = feature_names[selector.support_]
print(f"✅ Biến được chọn: {list(selected_cols)}")

# Chuyển về DataFrame để LightGBM/XGBoost không báo lỗi thiếu tên cột
X_train_final = pd.DataFrame(selector.transform(X_train_scaled), columns=selected_cols)
X_test_final = pd.DataFrame(selector.transform(X_test_scaled), columns=selected_cols)

# =============================================================================
# 3. CẤU HÌNH MODELS
# =============================================================================
rf = RandomForestClassifier(n_estimators=200, random_state=42)
xgb_model = xgb.XGBClassifier(eval_metric='logloss', random_state=42)
lgbm_model = lgb.LGBMClassifier(random_state=42, verbose=-1, verbosity=-1)

# CatBoost: Tắt log file và log training
cat_base = CatBoostClassifier(verbose=0, allow_writing_files=False, random_state=42)
cat_wrapper = SklearnCatBoostWrapper(estimator=cat_base)

mlp = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=1000, random_state=42)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": rf,
    "XGBoost": xgb_model,
    "LightGBM": lgbm_model,
    "CatBoost": cat_wrapper,
    "Neural Network": mlp
}

# Hybrid Models
print("\n🔗 Khởi tạo Hybrid Models...")
voting_clf = VotingClassifier(
    estimators=[('rf', rf), ('xgb', xgb_model), ('cat', cat_wrapper)],
    voting='soft'
)
models["Voting (RF+XGB+Cat)"] = voting_clf

stacking_clf = StackingClassifier(
    estimators=[('rf', rf), ('lgbm', lgbm_model), ('cat', cat_wrapper)],
    final_estimator=LogisticRegression(),
    cv=5
)
models["Stacking (RF+LGBM+Cat)"] = stacking_clf

# =============================================================================
# 4. HUẤN LUYỆN
# =============================================================================
results = []
print("\n🚀 BẮT ĐẦU HUẤN LUYỆN...")

best_acc = 0
best_model = None
best_name = ""

for name, model in models.items():
    print(f"   Training {name}...")
    try:
        model.fit(X_train_final, y_train)
        y_pred = model.predict(X_test_final)
        acc = accuracy_score(y_test, y_pred)
        
        try:
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test_final)[:, 1]
            else:
                y_proba = model.decision_function(X_test_final)
            auc = roc_auc_score(y_test, y_proba)
        except:
            auc = 0.0
            
        print(f"     -> Accuracy: {acc:.4f} | AUC: {auc:.4f}")
        results.append({'Model': name, 'Accuracy': acc, 'AUC': auc})
        
        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_name = name
            
    except Exception as e:
        print(f"❌ Lỗi {name}: {e}")
        # In chi tiết lỗi để debug nếu cần
        import traceback
        traceback.print_exc()

# =============================================================================
# 5. KẾT QUẢ & PERMUTATION IMPORTANCE
# =============================================================================
if results:
    res_df = pd.DataFrame(results).sort_values(by='Accuracy', ascending=False)
    print("\n🏆 BẢNG XẾP HẠNG CUỐI CÙNG:")
    print(res_df)
    
    print(f"\n🔍 Phân tích tầm quan trọng biến (Permutation Importance) cho: {best_name}")
    try:
        # Tính toán importance trên tập test
        perm_importance = permutation_importance(best_model, X_test_final, y_test, n_repeats=10, random_state=42)
        sorted_idx = perm_importance.importances_mean.argsort()

        plt.figure(figsize=(10, 8))
        #  - Generated by matplotlib below
        plt.barh(X_test_final.columns[sorted_idx], perm_importance.importances_mean[sorted_idx])
        plt.xlabel("Mức độ giảm độ chính xác (Importance)")
        plt.title(f"Yếu tố quan trọng nhất ({best_name})")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"⚠️ Không vẽ được biểu đồ: {e}")