"""
Script train ALL MODELS with PhoBERT (Fix Error + Anti-Overfitting + Chart)
1. Bỏ CatBoost (đang lỗi version).
2. Tinh chỉnh tham số (Giảm max_depth, tăng min_samples) để chống học vẹt.
3. Lưu tất cả model.
4. Vẽ biểu đồ so sánh độ chính xác để báo cáo.
"""

import pandas as pd
import numpy as np
import joblib
import torch
from transformers import AutoModel, AutoTokenizer
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns

# Các thuật toán
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import lightgbm as lgb
# Tạm bỏ CatBoost do lỗi xung đột version sklearn 1.6+
# from catboost import CatBoostClassifier 

# Công cụ hỗ trợ
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm 
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# CẤU HÌNH & PHOBERT
# =============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  Hardware: {DEVICE}")

print("⏳ Đang tải PhoBERT...")
try:
    phobert_tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
    phobert_model = AutoModel.from_pretrained("vinai/phobert-base-v2").to(DEVICE)
except:
    phobert_tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    phobert_model = AutoModel.from_pretrained("vinai/phobert-base").to(DEVICE)

def preprocess_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'[^a-záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def get_phobert_embeddings(text_list, batch_size=32):
    print(f"🚀  [PhoBERT] Embedding {len(text_list)} lyrics...")
    phobert_model.eval()
    embeddings = []
    for i in tqdm(range(0, len(text_list), batch_size)):
        batch_texts = text_list[i : i + batch_size]
        inputs = phobert_tokenizer(batch_texts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = phobert_model(**inputs)
            embeddings.append(outputs.last_hidden_state[:, 0, :].cpu().numpy())
    return np.vstack(embeddings)

# =============================================================================
# MAIN PROGRAM
# =============================================================================
if __name__ == "__main__":
    print("="*80)
    print("TRAINING ALL MODELS (ANTI-OVERFITTING MODE + CHART)")
    print("="*80)
    
    # --- 1. Load Data ---
    print("\n[1/7] Load dữ liệu...")
    CSV_MAIN = r'final_data/mergerd_balanced_and_features.csv'
    CSV_LYRIC = r'Audio_lyric/nlp_analysis.csv'
    
    try:
        df = pd.read_csv(CSV_MAIN)
        if 'lyric' not in df.columns and 'clean_lyric' not in df.columns:
            if os.path.exists(CSV_LYRIC):
                print("   ⚠️ Lấy Lyric từ file nlp_analysis.csv...")
                df_lyric = pd.read_csv(CSV_LYRIC)
                df['file_name'] = df['file_name'].astype(str).str.strip().str.lower()
                df_lyric['file_name'] = df_lyric['file_name'].astype(str).str.strip().str.lower()
                df = pd.merge(df, df_lyric[['file_name', 'lyric']], on='file_name', how='left')
                df['clean_lyric'] = df['lyric']
            else:
                print("   ❌ LỖI: Thiếu file lyric!")
                exit()
        else:
             df['clean_lyric'] = df['lyric'] if 'lyric' in df.columns else df['clean_lyric']

        if 'hit' not in df.columns and 'is_hit' in df.columns: df['hit'] = df['is_hit']
        df = df.dropna(subset=['hit'])
        print(f"   ✓ Dữ liệu: {len(df)} dòng")
    except Exception as e:
        print(f"❌ Lỗi Data: {e}")
        exit()

    # --- 2. Features ---
    print("\n[2/7] Chuẩn bị Features...")
    exclude_cols = ['clean_lyric', 'lyrics', 'lyric', 'clean_title', 'title', 'file_name', 'hit', 'is_hit', 'song_name', 'artist', 'album', 'release_date', 'spotify_popularity', 'total_plays', 'spotify_streams', 'release_year', 'days_since_release', 'id', 'Unnamed: 0', 'sentiment', 'sentiment_score', 'sentiment_confidence', 'noun_count', 'verb_count', 'adj_count', 'lyric_total_words']
    audio_cols = [c for c in df.columns if c not in exclude_cols and df[c].dtype in ['float64', 'int64', 'int32', 'float32']]
    X_audio = df[audio_cols].values
    
    df['lyric_total_words'] = df['lyric_total_words'].replace(0, 1)
    noun_ratio = (df['noun_count'] / df['lyric_total_words']).values.reshape(-1, 1)
    verb_ratio = (df['verb_count'] / df['lyric_total_words']).values.reshape(-1, 1)
    adj_ratio  = (df['adj_count'] / df['lyric_total_words']).values.reshape(-1, 1)
    sent_score = df['sentiment_score'].values.reshape(-1, 1)
    X_linguistic = np.hstack((sent_score, noun_ratio, verb_ratio, adj_ratio))

    df['clean_lyric'] = df['clean_lyric'].fillna('').astype(str).apply(preprocess_text)
    X_phobert = get_phobert_embeddings(df['clean_lyric'].tolist())
    
    X = np.hstack((X_audio, X_linguistic, X_phobert))
    y = df['hit'].values

    # --- 3. Chia tập ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # --- 4. Định nghĩa Models (Đã tinh chỉnh tham số chống Overfitting) ---
    print("\n[4/7] Khởi tạo Models (Đã giảm độ sâu cây)...")
    
    # Selector: Lọc bớt rác để giảm nhiễu
    selector = SelectFromModel(RandomForestClassifier(n_estimators=50, max_depth=7, random_state=42, n_jobs=-1), threshold='median')

    models = {
        # RF: Giảm max_depth từ 15 -> 8, Tăng min_samples_leaf -> 4
        'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=4, class_weight='balanced', random_state=42, n_jobs=-1),
        
        # XGB: Giảm max_depth -> 6, Tăng reg_alpha (chống nhiễu)
        'XGBoost': xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05, reg_alpha=1, random_state=42, eval_metric='logloss'),
        
        # LGBM: Giảm num_leaves
        'LightGBM': lgb.LGBMClassifier(n_estimators=200, max_depth=6, num_leaves=30, random_state=42, verbosity=-1),
        
        # Các model khác
        'Logistic Regression': LogisticRegression(C=0.5, max_iter=1000, random_state=42), # C nhỏ hơn = Regularization mạnh hơn
        'SVM': SVC(C=1.0, kernel='rbf', probability=True, random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=6, min_samples_leaf=10, random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=9), # Tăng K để mượt hóa biên
        'MLP Neural Net': MLPClassifier(hidden_layer_sizes=(32,), alpha=0.01, max_iter=500, random_state=42),
    }
    
    # Voting (RF + XGB + LGBM)
    models['Voting'] = VotingClassifier(estimators=[
        ('rf', models['Random Forest']), 
        ('xgb', models['XGBoost']), 
        ('lgbm', models['LightGBM'])
    ], voting='soft')

    # --- 5. Train & So Sánh ---
    print("\n[5/7] Bắt đầu Train & Lưu trữ...")
    os.makedirs('DA/pkl_file/models', exist_ok=True)
    os.makedirs("DA/images", exist_ok=True)
    results = []
    
    print(f"{'Model Name':<20} | {'Train Acc':<10} | {'Test Acc':<10} | {'Gap (Overfit)':<12} | {'F1-Score':<10}")
    print("-" * 85)

    for name, clf in models.items():
        try:
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('feature_selection', selector), # Lọc tính năng quan trọng
                ('classifier', clf)
            ])
            
            pipeline.fit(X_train, y_train)
            
            y_train_pred = pipeline.predict(X_train)
            y_test_pred = pipeline.predict(X_test)
            
            train_acc = accuracy_score(y_train, y_train_pred)
            test_acc = accuracy_score(y_test, y_test_pred)
            gap = train_acc - test_acc
            f1 = f1_score(y_test, y_test_pred, zero_division=0)
            
            print(f"{name:<20} | {train_acc*100:.2f}%     | {test_acc*100:.2f}%     | {gap*100:.2f}%       | {f1*100:.2f}%")
            
            results.append({
                'Model': name, 'Train Accuracy': train_acc, 'Test Accuracy': test_acc,
                'Overfitting Gap': gap, 'F1-Score': f1,
                'Precision': precision_score(y_test, y_test_pred, zero_division=0),
                'Recall': recall_score(y_test, y_test_pred, zero_division=0)
            })

            # Lưu model
            safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("+", "")
            joblib.dump({
                'pipeline': pipeline,
                'audio_features': audio_cols,
                'linguistic_cols': ['sentiment_score', 'noun_ratio', 'verb_ratio', 'adj_ratio'],
                'model_type': 'hybrid_multi'
            }, f'DA/pkl_file/models/{safe_name}.pkl')
            
        except Exception as e:
            print(f"❌ Lỗi khi train {name}: {e}")

    # --- 6. Tổng kết và Vẽ biểu đồ ---
    print("-" * 85)
    print("\n[6/7] Hoàn tất và xuất CSV...")
    df_res = pd.DataFrame(results).sort_values(by='Test Accuracy', ascending=False)
    df_res.to_csv('DA/pkl_file/model_comparison_full.csv', index=False)
    print("   💾 Bảng kết quả: DA/pkl_file/model_comparison_full.csv")
    
    # --- 7. VẼ BIỂU ĐỒ SO SÁNH ---
    print("\n[7/7] Vẽ biểu đồ so sánh độ chính xác...")
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")
    
    # Vẽ biểu đồ cột
    chart = sns.barplot(data=df_res, x="Model", y="Test Accuracy", palette="viridis")
    
    plt.title("So sánh Độ chính xác (Accuracy) - PhoBERT Experiment", fontsize=14, fontweight='bold')
    plt.ylim(0, 1.1) # Trục Y từ 0 đến 1.1 để hiển thị số rõ ràng
    plt.xticks(rotation=45, ha='right')
    
    # Hiển thị số liệu trên đầu cột
    for i, v in enumerate(df_res['Test Accuracy']):
        plt.text(i, v + 0.01, f"{v*100:.1f}%", ha='center', fontweight='bold', fontsize=10)
        
    plt.tight_layout()
    save_img_path = 'DA/images/model_accuracy_comparison_phobert.png'
    plt.savefig(save_img_path)
    print(f"   📸 Đã lưu biểu đồ tại: {save_img_path}")
    
    # Lưu model tốt nhất
    if not df_res.empty:
        best_name = df_res.iloc[0]['Model']
        print(f"\n   🏆 MODEL VÔ ĐỊCH: {best_name}")
        best_safe_name = best_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("+", "")
        best_data = joblib.load(f'DA/pkl_file/models/{best_safe_name}.pkl')
        joblib.dump(best_data, 'DA/pkl_file/task_1_phoBert.pkl')
        print("   ✅ Đã cập nhật model mặc định: DA/pkl_file/task_1_phoBert.pkl")