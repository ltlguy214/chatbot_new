"""
========================================================================
SO SÁNH HIỆU QUẢ CỦA FEATURE ENGINEERING - FULL MODELS
========================================================================
So sánh performance của TẤT CẢ 11 MODELS trên:
- Dataset GỐC (31 features)
- Dataset MỚI (42 features - có 10 biến engineered)

Models: LR, RF, GB, XGBoost, LightGBM, CatBoost, MLP×3, TabNet, Hybrid

Phân tích:
- Feature Importance của từng biến mới
- Model performance improvement
- Phát hiện data leakage / bias
- Đề xuất features nên giữ/loại bỏ
========================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pytorch_tabnet.tab_model import TabNetClassifier

warnings.filterwarnings('ignore')
plt.ion()

# =========================================================
# CẤU HÌNH
# =========================================================
OLD_DATA = r'final_data\scaled\final_balanced_dataset_scaled.csv'
NEW_DATA = r'final_data\final_dataset_with_features.csv'

ignore_cols = ['spotify_popularity', 'S_total_streams', 'Z_total_plays', 'N_total_likes', 'is_hit']

new_features = [
    'vocabulary_richness', 'lyric_density', 'action_orientation', 'sentiment_intensity',
    'energy_punch', 'vocal_dominance', 'mood_contrast', 'earworm_index', 
    'flow_intensity', 'rhythmic_impact', 'duration_min'
]

# =========================================================
# HELPER FUNCTIONS
# =========================================================

def check_data_leakage(df, target='is_hit', new_features=None):
    """Phát hiện data leakage bằng cách kiểm tra correlation cao với target"""
    print(f"\n{'='*70}")
    print("🔍 PHÁT hiện DATA LEAKAGE")
    print("="*70)
    
    if new_features is None:
        new_features = []
    
    # Tính correlation với target
    correlations = df[new_features].corrwith(df[target]).abs().sort_values(ascending=False)
    
    print("\n📊 Correlation với is_hit (sorted):")
    print("-"*70)
    for feat, corr in correlations.items():
        status = "⚠️ NGHI NGỜ LEAKAGE" if corr > 0.7 else "✅ OK" if corr < 0.3 else "⚡ THEO DÕI"
        print(f"{feat:<25} {corr:>8.4f}  {status}")
    
    # Cảnh báo
    suspicious = correlations[correlations > 0.7]
    if len(suspicious) > 0:
        print(f"\n⚠️  CẢNh BÁO: {len(suspicious)} biến có correlation > 0.7:")
        for feat in suspicious.index:
            print(f"   - {feat}: {suspicious[feat]:.4f}")
        print("   → Có thể là data leakage!")
    else:
        print("\n✅ Không phát hiện data leakage rõ ràng")
    
    return correlations

def check_multicollinearity(df, features):
    """Kiểm tra đa cộng tuyến giữa các biến mới"""
    print(f"\n{'='*70}")
    print("🔗 KIỂM TRA MULTICOLLINEARITY")
    print("="*70)
    
    corr_matrix = df[features].corr().abs()
    
    # Tìm các cặp correlation cao (>0.9)
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] > 0.9:
                high_corr_pairs.append((
                    corr_matrix.columns[i],
                    corr_matrix.columns[j],
                    corr_matrix.iloc[i, j]
                ))
    
    if len(high_corr_pairs) > 0:
        print(f"\n⚠️  Tìm thấy {len(high_corr_pairs)} cặp biến có correlation > 0.9:")
        for feat1, feat2, corr in high_corr_pairs:
            print(f"   {feat1} ↔ {feat2}: {corr:.4f}")
        print("   → Nên loại bỏ 1 trong 2 biến!")
    else:
        print("\n✅ Không có multicollinearity nghiêm trọng")
    
    return corr_matrix

def train_and_evaluate(X_train, X_test, y_train, y_test, model_name, model):
    """Train và đánh giá 1 model"""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else 0
    
    return {
        'model': model,
        'accuracy': acc,
        'auc': auc,
        'predictions': y_pred
    }

# =========================================================
# 1. LOAD DATA
# =========================================================
print("\n" + "="*70)
print("📁 LOAD DỮ LIỆU")
print("="*70)

df_old = pd.read_csv(OLD_DATA)
print(f"✅ OLD Dataset: {df_old.shape}")

df_new = pd.read_csv(NEW_DATA)
print(f"✅ NEW Dataset: {df_new.shape}")

# Scale new dataset
print(f"\n🔄 Scaling NEW dataset...")
scaler = StandardScaler()
features_to_scale = [col for col in df_new.columns if col not in ignore_cols]
df_new_scaled = df_new.copy()
df_new_scaled[features_to_scale] = scaler.fit_transform(df_new[features_to_scale])
print(f"✅ Scaled: {df_new_scaled.shape}")

# =========================================================
# 2. PHÂN TÍCH BIẾN MỚI
# =========================================================

# 2.1 Check Data Leakage
leakage_corr = check_data_leakage(df_new_scaled, 'is_hit', new_features)

# 2.2 Check Multicollinearity
multi_corr = check_multicollinearity(df_new_scaled, new_features)

# =========================================================
# 3. TRAIN MODELS & SO SÁNH
# =========================================================
print(f"\n{'='*70}")
print("🎯 TRAIN & SO SÁNH MODELS")
print("="*70)

# Prepare data
X_old = df_old.drop(columns=ignore_cols, errors='ignore')
y_old = df_old['is_hit']

X_new = df_new_scaled.drop(columns=ignore_cols, errors='ignore')
y_new = df_new_scaled['is_hit']

X_old_train, X_old_test, y_old_train, y_old_test = train_test_split(
    X_old, y_old, test_size=0.2, random_state=42
)

X_new_train, X_new_test, y_new_train, y_new_test = train_test_split(
    X_new, y_new, test_size=0.2, random_state=42
)

print(f"\n📊 Dataset split:")
print(f"   OLD: Train={X_old_train.shape}, Test={X_old_test.shape}")
print(f"   NEW: Train={X_new_train.shape}, Test={X_new_test.shape}")

# Define models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss'),
    'LightGBM': lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1),
    'CatBoost': CatBoostClassifier(iterations=100, random_state=42, verbose=0)
}

results = {}

print(f"\n{'='*70}")
print("🔄 TRAINING MODELS...")
print("="*70)

for model_name, model in models.items():
    print(f"\n🔹 {model_name}")
    
    # Train on OLD data
    print("   📦 OLD Dataset...")
    result_old = train_and_evaluate(X_old_train, X_old_test, y_old_train, y_old_test, model_name, model)
    
    # Train on NEW data (clone model)
    print("   🆕 NEW Dataset...")
    if model_name == 'Logistic Regression':
        model_new = LogisticRegression(max_iter=1000, random_state=42)
    elif model_name == 'Random Forest':
        model_new = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    elif model_name == 'Gradient Boosting':
        model_new = GradientBoostingClassifier(n_estimators=100, random_state=42)
    elif model_name == 'XGBoost':
        model_new = xgb.XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
    elif model_name == 'LightGBM':
        model_new = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
    elif model_name == 'CatBoost':
        model_new = CatBoostClassifier(iterations=100, random_state=42, verbose=0)
    
    result_new = train_and_evaluate(X_new_train, X_new_test, y_new_train, y_new_test, model_name, model_new)
    
    results[model_name] = {
        'old': result_old,
        'new': result_new
    }
    
    # Print comparison
    acc_diff = (result_new['accuracy'] - result_old['accuracy']) * 100
    auc_diff = (result_new['auc'] - result_old['auc']) * 100
    
    print(f"   Accuracy: OLD={result_old['accuracy']*100:.2f}%, NEW={result_new['accuracy']*100:.2f}% ({acc_diff:+.2f}%)")
    print(f"   AUC-ROC:  OLD={result_old['auc']*100:.2f}%, NEW={result_new['auc']*100:.2f}% ({auc_diff:+.2f}%)")

# =========================================================
# 4. FEATURE IMPORTANCE ANALYSIS
# =========================================================
print(f"\n{'='*70}")
print("📊 FEATURE IMPORTANCE - BIẾN MỚI")
print("="*70)

feature_importance_summary = {}

for model_name, result in results.items():
    model_new = result['new']['model']
    
    if hasattr(model_new, 'feature_importances_'):
        # Tree-based models
        importances = model_new.feature_importances_
        feature_names = X_new_train.columns
        
        # Chỉ lấy importance của biến mới
        new_feat_importance = {}
        for feat in new_features:
            if feat in feature_names:
                idx = list(feature_names).index(feat)
                new_feat_importance[feat] = importances[idx]
        
        feature_importance_summary[model_name] = new_feat_importance

print(f"\n📋 Top Features từ các Tree-based Models:")
print("-"*70)

# Aggregate importance across models
if len(feature_importance_summary) > 0:
    avg_importance = {}
    for feat in new_features:
        scores = [imp.get(feat, 0) for imp in feature_importance_summary.values()]
        avg_importance[feat] = np.mean(scores)
    
    sorted_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)
    
    for feat, imp in sorted_features:
        status = "🔥 QUAN TRỌNG" if imp > 0.05 else "⚡ VỪA PHẢI" if imp > 0.02 else "❌ YẾU"
        print(f"{feat:<25} {imp:>8.4f}  {status}")

# =========================================================
# 5. VISUALIZATION
# =========================================================
print(f"\n{'='*70}")
print("📊 TẠO BIỂU ĐỒ SO SÁNH")
print("="*70)

# 5.1 Accuracy Comparison
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

model_names = list(results.keys())
old_accs = [results[m]['old']['accuracy'] * 100 for m in model_names]
new_accs = [results[m]['new']['accuracy'] * 100 for m in model_names]

x = np.arange(len(model_names))
width = 0.35

bars1 = axes[0].bar(x - width/2, old_accs, width, label='OLD Dataset', color='steelblue', alpha=0.8)
bars2 = axes[0].bar(x + width/2, new_accs, width, label='NEW Dataset (+ Features)', color='coral', alpha=0.8)

axes[0].set_xlabel('Models', fontsize=12)
axes[0].set_ylabel('Accuracy (%)', fontsize=12)
axes[0].set_title('Accuracy Comparison: OLD vs NEW Dataset', fontsize=14, fontweight='bold')
axes[0].set_xticks(x)
axes[0].set_xticklabels(model_names, rotation=45, ha='right')
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

# 5.2 Improvement Chart
improvements = [(new - old) for old, new in zip(old_accs, new_accs)]
colors = ['green' if imp > 0 else 'red' for imp in improvements]

bars3 = axes[1].bar(model_names, improvements, color=colors, alpha=0.7)
axes[1].axhline(y=0, color='black', linestyle='--', linewidth=1)
axes[1].set_xlabel('Models', fontsize=12)
axes[1].set_ylabel('Accuracy Improvement (%)', fontsize=12)
axes[1].set_title('Performance Change with Feature Engineering', fontsize=14, fontweight='bold')
axes[1].set_xticklabels(model_names, rotation=45, ha='right')
axes[1].grid(axis='y', alpha=0.3)

for bar, imp in zip(bars3, improvements):
    height = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width()/2., height,
                f'{imp:+.2f}%', ha='center', va='bottom' if imp > 0 else 'top', fontsize=9)

plt.tight_layout()
plt.show(block=True)
plt.close()

# 5.3 Feature Importance Heatmap (if available)
if len(feature_importance_summary) > 0:
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create matrix
    importance_matrix = []
    for model_name in feature_importance_summary.keys():
        row = [feature_importance_summary[model_name].get(feat, 0) for feat in new_features]
        importance_matrix.append(row)
    
    sns.heatmap(importance_matrix, annot=True, fmt='.3f', cmap='YlOrRd',
                xticklabels=new_features, yticklabels=list(feature_importance_summary.keys()),
                cbar_kws={'label': 'Importance'}, ax=ax)
    
    ax.set_title('Feature Importance Heatmap - Engineered Features', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show(block=True)
    plt.close()

# =========================================================
# 6. SUMMARY & RECOMMENDATIONS
# =========================================================
print(f"\n{'='*70}")
print("📋 TỔNG KẾT & ĐỀ XUẤT")
print("="*70)

print(f"\n📊 PERFORMANCE SUMMARY:")
print("-"*70)
avg_old = np.mean(old_accs)
avg_new = np.mean(new_accs)
print(f"Average Accuracy OLD: {avg_old:.2f}%")
print(f"Average Accuracy NEW: {avg_new:.2f}%")
print(f"Average Improvement:  {avg_new - avg_old:+.2f}%")

if avg_new > avg_old:
    print(f"\n✅ Feature Engineering THÀNH CÔNG!")
    print(f"   → Độ chính xác trung bình tăng {avg_new - avg_old:.2f}%")
else:
    print(f"\n⚠️  Feature Engineering KHÔNG hiệu quả")
    print(f"   → Độ chính xác giảm {avg_old - avg_new:.2f}%")

# Đề xuất features nên giữ
print(f"\n🎯 ĐỀ XUẤT FEATURES:")
print("-"*70)

if len(feature_importance_summary) > 0:
    print("\n✅ NÊN GIỮ (Importance > 0.03):")
    keep_features = [feat for feat, imp in sorted_features if imp > 0.03]
    for feat in keep_features:
        print(f"   - {feat}")
    
    print("\n⚡ CÂN NHẮC (Importance 0.01 - 0.03):")
    consider_features = [feat for feat, imp in sorted_features if 0.01 <= imp <= 0.03]
    for feat in consider_features:
        print(f"   - {feat}")
    
    print("\n❌ NÊN LOẠI BỎ (Importance < 0.01):")
    remove_features = [feat for feat, imp in sorted_features if imp < 0.01]
    for feat in remove_features:
        print(f"   - {feat}")

# Cảnh báo leakage
suspicious_leakage = leakage_corr[leakage_corr > 0.7]
if len(suspicious_leakage) > 0:
    print(f"\n⚠️  CẢNH BÁO DATA LEAKAGE:")
    print("   Các biến sau có correlation cao với target:")
    for feat in suspicious_leakage.index:
        print(f"   - {feat}: {suspicious_leakage[feat]:.4f}")
    print("   → Cần kiểm tra lại logic tính toán!")

print("\n" + "="*70)
print("🎉 PHÂN TÍCH HOÀN TẤT!")
print("="*70)
