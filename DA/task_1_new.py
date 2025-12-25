import pandas as pd
import numpy as np
import re
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

# Scikit-learn Imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectFromModel

# Classification Models (FULL LIST TỪ TASK 3)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, 
    ExtraTreesClassifier, VotingClassifier, StackingClassifier
)
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

# Metrics
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

warnings.filterwarnings('ignore')

# =============================================================================
# 1. CẤU HÌNH & LOAD DỮ LIỆU
# =============================================================================
FILE_AUDIO = 'mergerd_balanced_and_features.csv' 
FILE_NLP = 'nlp_analysis.csv'                   

def load_and_merge_data():
    print("⏳ Đang tải dữ liệu HIT PREDICTION...")
    try:
        # Xử lý đường dẫn linh hoạt
        path_audio = FILE_AUDIO if os.path.exists(FILE_AUDIO) else f'final_data/{FILE_AUDIO}'
        path_nlp = FILE_NLP if os.path.exists(FILE_NLP) else f'Audio_lyric/{FILE_NLP}'
        
        if not os.path.exists(path_audio) or not os.path.exists(path_nlp):
             print(f"❌ Không tìm thấy file dữ liệu.")
             return None

        df_audio = pd.read_csv(path_audio)
        df_nlp = pd.read_csv(path_nlp)
        
        # Chuẩn hóa tên file
        df_audio['file_name'] = df_audio['file_name'].astype(str).str.strip().str.lower()
        df_nlp['file_name'] = df_nlp['file_name'].astype(str).str.strip().str.lower()
        
        # Merge Audio và NLP (Lấy lyric)
        df = pd.merge(df_audio, df_nlp[['file_name', 'lyric']], on='file_name', how='left')
        print(f"✅ Đã merge thành công: {len(df)} bài hát.")
        return df
    except Exception as e:
        print(f"❌ Lỗi load data: {e}")
        return None

# =============================================================================
# 2. FEATURE ENGINEERING (TARGET: IS_HIT)
# =============================================================================
def preprocess_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'[^a-záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def prepare_data(df):
    # 1. TẠO TARGET: IS_HIT
    target_col = 'is_hit' if 'is_hit' in df.columns else 'hit'
    if target_col not in df.columns:
        raise ValueError("❌ Thiếu cột label 'is_hit'!")
        
    df['target'] = df[target_col].apply(lambda x: 1 if (x == 1 or str(x).lower() == 'true') else 0)
    print(f"📊 Phân phối nhãn Hit: \n{df['target'].value_counts()}")

    # 2. Xử lý text
    df['clean_lyric'] = df['lyric'].fillna('').apply(preprocess_text)
    
    # 3. Lọc Feature (BỎ DATA LEAKAGE)
    cols_ignore = [
        'file_name', 'title', 'artists', 'spotify_release_date', 'spotify_genres', 
        'is_hit', 'hit', 'target', 
        'total_plays', 'spotify_streams', 'spotify_popularity', # Leakage!
        'lyric', 'clean_lyric', 'Unnamed: 0', 'track_id', 
        'sentiment', 'sentiment_confidence',
        'cluster', 'musical_key'
    ]
    numeric_feats = [c for c in df.columns if c not in cols_ignore and pd.api.types.is_numeric_dtype(df[c])]
    
    return df, numeric_feats

# =============================================================================
# 3. QUY TRÌNH CHẠY FULL MODEL (GIỐNG TASK 3)
# =============================================================================
def run_full_analysis_task1():
    # 1. Load Data
    df = load_and_merge_data()
    if df is None: return

    df_clean, numeric_feats = prepare_data(df)

    # =========================================================================
    # [THÊM] KIỂM TRA SỐ LƯỢNG FEATURES (TASK 1)
    # =========================================================================
    print("\n" + "="*60)
    print("🔍 KIỂM TRA CHI TIẾT CÁC BIẾN ĐẦU VÀO (TASK 1 - HIT PREDICTION)")
    print("="*60)
    print(f"1️⃣ Số lượng biến số (Numeric Features): {len(numeric_feats)}")
    print(np.array(numeric_feats)) 
    print("-" * 60)
    print(f"2️⃣ Số lượng biến văn bản (TF-IDF): 100")
    total_feats = len(numeric_feats) + 100
    print("-" * 60)
    print(f"✅ TỔNG CỘNG SỐ BIẾN: {total_feats}")
    print("="*60 + "\n")
    
    X = df_clean[numeric_feats + ['clean_lyric']]
    y = df_clean['target']
    
    # Stratify theo target (Hit/Non-Hit)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 2. Pipeline
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')), 
        ('scaler', StandardScaler())
    ])
    
    # TF-IDF tối ưu cho Hit (100 features)
    text_transformer = TfidfVectorizer(max_features=100, ngram_range=(1, 2))

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_feats),
        ('text', text_transformer, 'clean_lyric')
    ])
    
    feature_selector = SelectFromModel(RandomForestClassifier(n_estimators=50, random_state=42), threshold="median")

    # 3. ĐỊNH NGHĨA TOÀN BỘ MODEL (CẤU HÌNH TUNED TỪ TASK 3)
    print("\n⚙️ Đang cấu hình toàn bộ models...")

    # --- Tree-based ---
    rf = RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=42)
    et = ExtraTreesClassifier(n_estimators=300, class_weight='balanced', random_state=42)
    gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, subsample=0.8, random_state=42)
    
    # AdaBoost (Rate 0.5 ổn định)
    ada = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=200, learning_rate=0.5, random_state=42
    )
    
    # --- Non-Tree ---
    svm = SVC(C=10, kernel='rbf', gamma='scale', probability=True, class_weight='balanced', random_state=42)
    mlp = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=1000, early_stopping=True, random_state=42)
    lr = LogisticRegression(random_state=42)

    # --- Ensembles ---
    voting_clf = VotingClassifier(
        estimators=[('et', et), ('rf', rf), ('ada', ada)], 
        voting='soft'
    )
    
    stacking_clf = StackingClassifier(
        estimators=[('et', et), ('svm', svm), ('rf', rf)],
        final_estimator=LogisticRegression(),
        cv=3
    )

    # DANH SÁCH MODEL ĐỂ SO SÁNH
    models_to_evaluate = {
        'Voting (ET+RF+ADA)': voting_clf, # Hy vọng là Champion
        'AdaBoost (Tuned)': ada,
        'Gradient Boosting': gb,
        'Random Forest': rf,
        'Extra Trees': et,
        'Logistic Regression': lr,
        'SVM (Geometric)': svm,
        'MLP (Neural Net)': mlp,
        'Stacking (ET+SVM+RF)': stacking_clf
    }

    # 4. CHẠY VÒNG LẶP HUẤN LUYỆN
    print("\n🚀 BẮT ĐẦU SO SÁNH TOÀN DIỆN CHO BÀI TOÁN HIT...")
    results_list = []
    fitted_pipelines = {}

    best_acc = 0
    best_model_name = "" # Thêm dòng này
    
    print(f"{'MODEL':<30} | {'ACC':<8} | {'F1':<8} | {'PREC':<8} | {'REC':<8}")
    print("-" * 75)

    for name, model in models_to_evaluate.items():
        try:
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('selector', feature_selector),
                ('clf', model)
            ])
            
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            rec = recall_score(y_test, y_pred, average='weighted')
            
            results_list.append({'Model': name, 'Accuracy': acc, 'F1-Score': f1})
            fitted_pipelines[name] = pipeline
            if acc > best_acc:
                best_acc = acc
                best_model_name = name

            print(f"{name:<30} | {acc:.4f}   | {f1:.4f}   | {prec:.4f}   | {rec:.4f}")
            
        except Exception as e:
            print(f"❌ Lỗi {name}: {e}")

    # 5. VẼ BIỂU ĐỒ SO SÁNH
    results_df = pd.DataFrame(results_list).sort_values(by='Accuracy', ascending=False)
    
    plt.figure(figsize=(14, 8))
    sns.set_theme(style="whitegrid")
    ax = sns.barplot(data=results_df, x='Accuracy', y='Model', palette='viridis')
    
    for i, v in enumerate(results_df['Accuracy']):
        ax.text(v + 0.002, i, f"{v:.4f}", color='black', va='center', fontweight='bold')
    
    min_acc = results_df['Accuracy'].min()
    max_acc = results_df['Accuracy'].max()
    plt.xlim(max(0, min_acc - 0.05), min(1.0, max_acc + 0.05))
    
    plt.title('So sánh Hiệu năng Model dự đoán HIT (Task 1)', fontsize=15, fontweight='bold')
    plt.xlabel('Độ chính xác (Accuracy)')
    plt.tight_layout()
    os.makedirs('DA/images', exist_ok=True)
    plt.savefig('DA/images/task1_full_comparison.png')
    print("\n✅ Đã lưu biểu đồ so sánh tại: DA/images/task1_full_comparison.png")
    print("🏆 Top 1 Model cho bài toán Hit:", best_model_name)
    print(results_df.head(1))

    save_dir = 'DA/pkl_file'
    os.makedirs(save_dir, exist_ok=True)
    
    # FILE 1: All models
    all_models_path = f'{save_dir}/all_models_p1_hit.pkl'
    joblib.dump({
        'models': fitted_pipelines,
        'numeric_feats': numeric_feats,
        'target_names': ['Non-Hit', 'Hit']
    }, all_models_path)
    print(f"💾 Đã lưu bộ toàn tập model tại: {all_models_path}")
    
    # FILE 2: Best model
    if best_model_name in fitted_pipelines:
        best_model_path = f'{save_dir}/best_model_p1_hit.pkl'
        joblib.dump({
            'pipeline': fitted_pipelines[best_model_name],
            'model_name': best_model_name,
            'accuracy': best_acc
        }, best_model_path)
        print(f"💾 Đã lưu model tốt nhất ({best_model_name}) tại: {best_model_path}")

if __name__ == "__main__":
    run_full_analysis_task1()