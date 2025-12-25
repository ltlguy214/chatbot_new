"""
Script Train Model (BM25 VERSION - FULL OPTION)
-----------------------------------------------
1. Thay thế TF-IDF bằng BM25 (Tối ưu hơn cho Hit Song).
2. Train & Lưu file .pkl cho TẤT CẢ model.
3. Xuất báo cáo CSV & Biểu đồ đầy đủ.
4. Tự động chọn Best Model.
"""

import pandas as pd
import numpy as np
import joblib
import os
import re
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

# ML Libraries
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier

# Processing
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_extraction.text import CountVectorizer # <--- Dùng cái này làm nền cho BM25
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score
)

warnings.filterwarnings('ignore')

# =============================================================================
# CÁC HÀM HỖ TRỢ & CLASS FIX LỖI
# =============================================================================
class FixedCatBoostClassifier(CatBoostClassifier, BaseEstimator, ClassifierMixin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

def preprocess_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'[^a-záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def to_dense_array(x):
    if hasattr(x, "toarray"):
        return x.toarray()
    return x

VN_STOPWORDS = [
    'anh', 'em', 'tôi', 'là', 'của', 'và', 'có', 'trong', 'đã', 'được', 
    'với', 'cho', 'từ', 'này', 'để', 'một', 'không', 'thì', 'những', 'trên',
    'sẽ', 'những', 'khi', 'người', 'các', 'về', 'ở', 'đến', 'ra', 'vào',
    'như', 'nếu', 'bởi', 'đang', 'mà', 'nó', 'hay', 'vì', 'theo', 'thế',
    'rằng', 'cũng', 'nhưng', 'bạn', 'họ', 'vẫn', 'chỉ', 'được', 'nào',
    'đều', 'rất', 'lại', 'thật', 'thêm', 'nữa', 'đây', 'đó', 'ấy', 'kia'
]

# =============================================================================
# CLASS BM25 VECTORIZER (THAY THẾ TF-IDF)
# =============================================================================
class BM25Vectorizer(BaseEstimator, TransformerMixin):
    """
    Triển khai thuật toán BM25 tương thích với Sklearn Pipeline.
    k1: Kiểm soát độ bão hòa tần suất từ (thường 1.2 - 2.0).
    b:  Kiểm soát độ phạt độ dài văn bản (0.75 là chuẩn).
    """
    def __init__(self, k1=1.5, b=0.75, max_features=500, ngram_range=(1, 2), stop_words=None):
        self.k1 = k1
        self.b = b
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.stop_words = stop_words
        # Dùng CountVectorizer để đếm từ trước
        self.vectorizer = CountVectorizer(
            max_features=max_features, 
            ngram_range=ngram_range, 
            stop_words=stop_words
        )

    def fit(self, X, y=None):
        self.vectorizer.fit(X)
        
        # Tính toán các chỉ số thống kê cho BM25
        X_counts = self.vectorizer.transform(X)
        self.doc_len_ = np.array(X_counts.sum(axis=1)).flatten()
        self.avgdl_ = self.doc_len_.mean()
        self.n_samples_ = X_counts.shape[0]
        
        # Tính IDF chuẩn BM25
        # df: số lượng văn bản chứa từ đó
        df = np.array((X_counts > 0).sum(axis=0)).flatten()
        self.idf_ = np.log((self.n_samples_ - df + 0.5) / (df + 0.5) + 1)
        return self

    def transform(self, X):
        X_counts = self.vectorizer.transform(X)
        X_dense = X_counts.toarray() # Chuyển sang dense để tính toán nhanh
        doc_len = np.array(X_counts.sum(axis=1)).flatten()
        
        # Công thức BM25:
        # score = IDF * (TF * (k1 + 1)) / (TF + k1 * (1 - b + b * (doc_len / avgdl)))
        
        numerator = X_dense * (self.k1 + 1)
        denominator = X_dense + self.k1 * (1 - self.b + self.b * (doc_len[:, None] / self.avgdl_))
        
        # Tránh chia cho 0
        with np.errstate(divide='ignore', invalid='ignore'):
             tf_part = np.where(denominator != 0, numerator / denominator, 0)
        
        return tf_part * self.idf_

    def get_feature_names_out(self, input_features=None):
        return self.vectorizer.get_feature_names_out()

# =============================================================================
# HÀM VẼ FEATURE IMPORTANCE
# =============================================================================
def visualize_importance(pipeline, numeric_cols, model_name):
    print(f"\n🎨 Đang vẽ Feature Importance cho NHÀ VÔ ĐỊCH: {model_name}...")
    try:
        model = pipeline.named_steps['clf']
        if not hasattr(model, 'feature_importances_'):
            print(f"   ⚠️ Best Model ({model_name}) không hỗ trợ vẽ Feature Importance.")
            return

        preprocessor = pipeline.named_steps['pre']
        # Lấy tên feature từ BM25 (bước 'text')
        bm25_transformer = preprocessor.named_transformers_['text']
        bm25_cols = list(bm25_transformer.get_feature_names_out())
        
        all_feature_names = numeric_cols + bm25_cols
        
        selector = pipeline.named_steps['sel']
        selected_mask = selector.get_support()
        final_features = np.array(all_feature_names)[selected_mask]
        
        importances = model.feature_importances_
        df_imp = pd.DataFrame({'Feature': final_features, 'Importance': importances})
        df_imp = df_imp.sort_values(by='Importance', ascending=False).head(20)
        
        plt.figure(figsize=(10, 8))
        sns.barplot(data=df_imp, x='Importance', y='Feature', palette='viridis') 
        plt.title(f'TOP 20 YẾU TỐ QUAN TRỌNG NHẤT ({model_name.upper()})', fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'DA/images/best_model_importance.png')
        print(f"   📸 Đã lưu biểu đồ tại: DA/images/best_model_importance.png")
    except Exception as e:
        print(f"   ❌ Lỗi vẽ biểu đồ: {e}")

# =============================================================================
# MAIN PROGRAM
# =============================================================================
if __name__ == "__main__":
    print("="*80)
    print("🚀 TRAINING ALL MODELS (BM25 VERSION)")
    print("="*80)

    # 1. Load Data
    print("\n[1/7] Load dữ liệu...")
    try:
        df_audio = pd.read_csv("final_data/mergerd_balanced_and_features.csv")
        df_nlp = pd.read_csv("Audio_lyric/nlp_analysis.csv")

        df_audio['file_name'] = df_audio['file_name'].astype(str).str.strip().str.lower()
        df_nlp['file_name'] = df_nlp['file_name'].astype(str).str.strip().str.lower()

        df = pd.merge(df_audio, df_nlp[['file_name', 'lyric']], on='file_name', how='left')
        df['clean_lyric'] = df['lyric'].fillna('').apply(preprocess_text)
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        exit()

    # 2. Features & Split
    print("[2/7] Chia tập dữ liệu...")
    if 'is_hit' not in df.columns:
        print("❌ Thiếu cột 'is_hit'")
        exit()

    excluded = ['file_name','title','artists','spotify_release_date','spotify_genres','is_hit','hit','lyric','clean_lyric','total_plays','spotify_streams','spotify_popularity','release_year','days_since_release']
    numeric_feats = [c for c in df.columns if c not in excluded and df[c].dtype in ['float64', 'int64']]
    
    X = df[numeric_feats + ['clean_lyric']]
    y = df['is_hit']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 3. Pipeline Setup (SỬ DỤNG BM25 THAY TF-IDF)
    print("[3/7] Cấu hình Pipeline (với BM25)...")
    
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric_feats),
        # --- Thay TfidfVectorizer bằng BM25Vectorizer ---
        ('text', BM25Vectorizer(max_features=500, ngram_range=(1, 2), stop_words=VN_STOPWORDS), 'clean_lyric')
    ])
    
    selector = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42), threshold="median")

    # 4. Models Definition
    rf_base = RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_leaf=2, random_state=42, n_jobs=-1, class_weight='balanced')
    xgb_base = xgb.XGBClassifier(n_estimators=300, max_depth=10, random_state=42, use_label_encoder=False, eval_metric='logloss')
    lgb_base = lgb.LGBMClassifier(n_estimators=300, max_depth=10, random_state=42, verbose=-1)
    cat_base = FixedCatBoostClassifier(verbose=0, random_state=42)

    models = {
        'Random Forest': rf_base,
        'XGBoost': xgb_base,
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'MLP': MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42),
        'Naive Bayes': GaussianNB(),
        'Voting (RF+XGB+LGBM)': VotingClassifier(estimators=[('rf', rf_base), ('xgb', xgb_base), ('lgb', lgb_base)], voting='soft'),
        'Stacking (RF+XGB+Cat)': StackingClassifier(estimators=[('rf', rf_base), ('xgb', xgb_base), ('cat', cat_base)], final_estimator=LogisticRegression(), cv=3)
    }

    # 5. Train Loop
    print("\n[4/7] Bắt đầu Training và Lưu từng Model...")
    os.makedirs("DA/pkl_file", exist_ok=True)
    os.makedirs("DA/images", exist_ok=True)
    
    results = []
    best_acc = 0
    best_model_name = ""
    best_pipeline = None

    for name, clf in models.items():
        steps = [('pre', preprocessor), ('sel', selector), ('clf', clf)]
        pipeline = Pipeline(steps)

        try:
            pipeline.fit(X_train, y_train)
        except TypeError: # Fix Naive Bayes
            steps.insert(2, ('dense', FunctionTransformer(to_dense_array, accept_sparse=True)))
            pipeline = Pipeline(steps)
            pipeline.fit(X_train, y_train)

        # Predict
        y_train_pred = pipeline.predict(X_train)
        y_test_pred = pipeline.predict(X_test)
        
        # Metrics
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        f1 = f1_score(y_test, y_test_pred, pos_label=1)
        prec = precision_score(y_test, y_test_pred, pos_label=1)
        rec = recall_score(y_test, y_test_pred, pos_label=1)

        print(f"\n📊 {name.upper()}")
        print(f"   ► Train Acc: {train_acc*100:.2f}% | Test Acc: {test_acc*100:.2f}%")
        print(f"   ► Precision: {prec:.4f} | Recall: {rec:.4f} | F1-Score: {f1:.4f}")
        
        print("   📉 Classification Report:")
        print(classification_report(y_test, y_test_pred, target_names=['Non-Hit', 'Hit']))
        
        tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
        print(f"   🧩 Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
        
        results.append({
            'Model': name, 
            'Accuracy': test_acc,
            'F1-Score': f1,
            'Precision': prec,
            'Recall': rec
        })
        
        clean_name = name.lower().replace(" ", "_").replace("+", "_").replace("(", "").replace(")", "")
        joblib.dump({
            'pipeline': pipeline, 
            'metrics': {'acc': test_acc, 'f1': f1}, 
            'features': numeric_feats,
            'model_name': name
        }, f"DA/pkl_file/task_1_{clean_name}.pkl")
        
        if test_acc > best_acc:
            best_acc = test_acc
            best_model_name = name
            best_pipeline = pipeline

    # 6. Xuất báo cáo tổng hợp (CSV)
    print("\n[5/7] Xuất file báo cáo tổng hợp (CSV)...")
    df_results = pd.DataFrame(results).sort_values(by='Accuracy', ascending=False)
    csv_path = "DA/pkl_file/all_model_metrics.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"   ✅ Đã lưu bảng so sánh tại: {csv_path}")

    # 7. Kết quả và Vẽ biểu đồ
    print("\n" + "="*80)
    print(f"🏆 NHÀ VÔ ĐỊCH: {best_model_name} (Accuracy: {best_acc*100:.2f}%)")
    print("="*80)

    # A. Vẽ biểu đồ so sánh
    print("[6/7] Vẽ biểu đồ so sánh...")
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_results, x="Model", y="Accuracy", palette="viridis")
    plt.title("So sánh Độ chính xác (Accuracy) khi dùng BM25", fontsize=14, fontweight='bold')
    plt.ylim(0, 1.1)
    for i, v in enumerate(df_results['Accuracy']):
        plt.text(i, v + 0.01, f"{v*100:.1f}%", ha='center', fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('DA/images/model_accuracy_comparison_BM25.png')
    
    # B. Vẽ Feature Importance
    if best_pipeline:
        visualize_importance(best_pipeline, numeric_feats, model_name=best_model_name)
    
    # C. Lưu Best Model
    print(f"\n[7/7] Đang lưu model tốt nhất ({best_model_name}) cho App...")
    best_model_path = 'DA/pkl_file/task_1_best_model_BM25.pkl'
    joblib.dump({
        'pipeline': best_pipeline,
        'metrics': {'acc': best_acc},
        'features': numeric_feats,
        'model_name': best_model_name
    }, best_model_path)
    print(f"   ✅ Đã lưu file chuẩn cho App: {best_model_path}")

    print("\n✅ HOÀN TẤT (PHIÊN BẢN BM25)!")