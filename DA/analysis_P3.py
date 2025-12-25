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

# Classification Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, 
    ExtraTreesClassifier, BaggingClassifier, VotingClassifier, StackingClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

# Metrics
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

warnings.filterwarnings('ignore')

# =============================================================================
# 1. CẤU HÌNH & LOAD DỮ LIỆU
# =============================================================================
FILE_AUDIO = 'mergerd_balanced_and_features.csv' 
FILE_NLP = 'nlp_analysis.csv'                   

# VN_STOPWORDS = [
#     'anh', 'em', 'tôi', 'là', 'của', 'và', 'có', 'trong', 'đã', 'được', 
#     'với', 'cho', 'từ', 'này', 'để', 'một', 'không', 'thì', 'những', 'trên',
#     'sẽ', 'những', 'khi', 'người', 'các', 'về', 'ở', 'đến', 'ra', 'vào',
#     'như', 'nếu', 'bởi', 'đang', 'mà', 'nó', 'hay', 'vì', 'theo', 'thế',
#     'rằng', 'cũng', 'nhưng', 'bạn', 'họ', 'vẫn', 'chỉ', 'được', 'nào',
#     'đều', 'rất', 'lại', 'thật', 'thêm', 'nữa', 'đây', 'đó', 'ấy', 'kia','ta'
# ]

def load_and_merge_data():
    print("⏳ Đang tải và xử lý dữ liệu...")
    try:
        # Xử lý đường dẫn linh hoạt (tìm trong folder con nếu không thấy)
        path_audio = FILE_AUDIO if os.path.exists(FILE_AUDIO) else f'final_data/{FILE_AUDIO}'
        path_nlp = FILE_NLP if os.path.exists(FILE_NLP) else f'Audio_lyric/{FILE_NLP}'
        
        if not os.path.exists(path_audio) or not os.path.exists(path_nlp):
             print(f"❌ Không tìm thấy file dữ liệu. Vui lòng kiểm tra lại!")
             return None

        df_audio = pd.read_csv(path_audio)
        df_nlp = pd.read_csv(path_nlp)
        
        # Chuẩn hóa tên file để merge
        df_audio['file_name'] = df_audio['file_name'].astype(str).str.strip().str.lower()
        df_nlp['file_name'] = df_nlp['file_name'].astype(str).str.strip().str.lower()
        
        # Xóa các cột NLP cũ trong file Audio (tránh trùng lặp)
        cols_to_remove = ['sentiment', 'lyric', 'clean_lyric', 'sentiment_score', 'sentiment_label']
        df_audio = df_audio.drop(columns=[c for c in cols_to_remove if c in df_audio.columns], errors='ignore')

        # Merge Audio và NLP (Chỉ lấy lyric và sentiment từ NLP)
        df = pd.merge(df_audio, df_nlp[['file_name', 'lyric', 'sentiment']], on='file_name', how='inner')
        print(f"✅ Đã merge thành công: {len(df)} bài hát.")
        return df
    
    except Exception as e:
        print(f"❌ Lỗi load data: {e}")
        return None

# =============================================================================
# 2. FEATURE ENGINEERING (BINARY TARGET)
# =============================================================================
def preprocess_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    # Giữ lại tiếng Việt, xóa ký tự đặc biệt
    text = re.sub(r'[^a-záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def prepare_data(df):
    # 1. Tạo biến mục tiêu Nhị phân (Binary Target)
    # Positive (Vui) = 1, Non-Positive (Buồn/Trung tính) = 0
    # Cần kiểm tra kỹ giá trị sentiment để map chính xác
    df['target'] = df['sentiment'].apply(lambda x: 1 if str(x).strip().lower() == 'positive' else 0)
    
    # 2. Xử lý text (Lyrics)
    df['clean_lyric'] = df['lyric'].fillna('').apply(preprocess_text)
    
    # 3. Lọc Feature số (Audio)
    # Loại bỏ các cột không dùng cho training
    cols_ignore = [
        'file_name', 'title', 'artists', 'spotify_release_date', 'spotify_genres', 
        'is_hit', 'total_plays', 'spotify_streams', 'spotify_popularity',
        'sentiment', 'sentiment_score', 'Unnamed: 0', 'lyric', 'clean_lyric', 
        'musical_key', 'target', 'sentiment_confidence', 'track_id'
    ]
    numeric_feats = [c for c in df.columns if c not in cols_ignore and pd.api.types.is_numeric_dtype(df[c])]
    
    return df, numeric_feats

# =============================================================================
# 3. QUY TRÌNH CHỌN LỌC & ĐÁNH GIÁ TOÀN DIỆN
# =============================================================================
def run_analysis_complete():
    # 1. Load & Prepare
    df = load_and_merge_data()
    if df is None: return

    df_clean, numeric_feats = prepare_data(df)

    print("\n" + "="*60)
    print("🔍 KIỂM TRA CHI TIẾT CÁC BIẾN ĐẦU VÀO (FEATURES - TASK 3)")
    print("="*60)
    print(f"1️⃣ Số lượng biến số (Numeric Features): {len(numeric_feats)}")
    # print(np.array(numeric_feats)) 
    print("-" * 60)
    print(f"2️⃣ Số lượng biến văn bản (TF-IDF): 100")
    total_feats = len(numeric_feats) + 100
    print("-" * 60)
    print(f"✅ TỔNG CỘNG SỐ BIẾN: {total_feats}")
    print("="*60 + "\n")
    print("📋 Danh sách chi tiết các biến số:")
    print(np.array(numeric_feats)) # In dạng array cho gọn
    
    X = df_clean[numeric_feats + ['clean_lyric']]
    y = df_clean['target']
    
    # Chia Train/Test (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 2. Pipeline Components
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')), 
        ('scaler', StandardScaler())
    ])
    # TF-IDF cho văn bản
    # text_transformer = TfidfVectorizer(max_features=100, ngram_range=(1, 2), stop_words=VN_STOPWORDS)
    text_transformer = TfidfVectorizer(max_features=100, ngram_range=(1, 2))
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_feats),
        ('text', text_transformer, 'clean_lyric')
    ])
    
    # Feature Selection tự động để loại bỏ nhiễu
    feature_selector = SelectFromModel(RandomForestClassifier(n_estimators=50, random_state=42), threshold="median")

    # 3. ĐỊNH NGHĨA MODEL
    print("\n⚙️ Đang cấu hình các mô hình...")

    # --- Nhóm Cây (Tree-based) ---
    
    # --- CẤU HÌNH TỐI ƯU HÓA (TUNED HYPERPARAMETERS) ---
    
    # 1. Random Forest: Tăng số cây lên 300 để kết quả ổn định hơn (ít biến động)
    rf = RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=42)
    
    # 2. Gradient Boosting: Dùng subsample=0.8 để tránh học vẹt (Stochastic Gradient Boosting)
    gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, subsample=0.8, random_state=42)
    
    # 3. Extra Trees: Tương tự RF, tăng số cây
    et = ExtraTreesClassifier(n_estimators=300, class_weight='balanced', random_state=42)
    
    # 4. AdaBoost (QUAN TRỌNG): 
    # - n_estimators=200: Tăng gấp đôi số cây để học kỹ hơn.
    # - learning_rate=0.5: Giảm tốc độ học xuống một nửa (so với mặc định 1.0) để tránh sai sót.
    ada = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=200, 
        learning_rate=0.5, 
        random_state=42
    )
    
    # 5. SVM: Tăng C lên 10 để mô hình "quyết liệt" hơn (giảm sai số trên tập train)
    svm = SVC(C=10, kernel='rbf', probability=True, class_weight='balanced', random_state=42)
    
    # 6. MLP: Tăng số vòng lặp (max_iter) để đảm bảo mạng nơ-ron hội tụ hoàn toàn
    mlp = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=1000, early_stopping=True, random_state=42)
    lr_meta = LogisticRegression(random_state=42) # Dùng cho meta-learner (Linear)

    # --- Ensemble (Tổ hợp) ---
    # Voting: Top 3 Trees tốt nhất (Dựa trên kết quả thực nghiệm: ET, Ada, GB)
    voting_clf = VotingClassifier(
        estimators=[('rf', rf), ('ada', ada), ('gb', gb)], # Thay 'et' bằng 'rf'
        voting='soft'
    )

    # Stacking: Kết hợp đa dạng (Diversity) = Cây (ET) + Hình học (SVM) + Nơ-ron (MLP)
    # Meta-learner: Logistic Regression
    stacking_clf = StackingClassifier(
        estimators=[('ada', ada), ('svm', svm), ('mlp', mlp)],
        final_estimator=lr_meta,
        cv=3
    )

    # 4. DANH SÁCH TOÀN BỘ MODEL CẦN CHẠY (TÁCH LẺ)
    # Bao gồm cả các thành phần bên trong Voting/Stacking để so sánh trực tiếp
    models_to_evaluate = {
        'AdaBoost (Tree)': ada,             # Đang Top 1
        'Random Forest (Tree)': rf,         # Đang Top 2
        'Gradient Boosting (Tree)': gb,     # Đang Top 3 (nhóm cây)
        'Logistic Regression (Linear)': lr_meta, # Đang Top 3 (tổng)
        'Extra Trees (Tree)': et,
        'SVM (Geometric)': svm,
        'MLP (Neural Net)': mlp,
        'Voting (RF+Ada+GB)': voting_clf,   # Đội hình mới
        'Stacking (Ada+SVM+MLP)': stacking_clf # Đổi cây chủ lực sang Ada
    }

    print("\n🚀 BẮT ĐẦU HUẤN LUYỆN & SO SÁNH (COMPONENTS vs ENSEMBLES)...")
    results_list = []

    fitted_pipelines = {}

    trained_pipelines = {} # Lưu lại để vẽ biểu đồ Feature Importance nếu cần
    
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
            
            # Metrics
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            rec = recall_score(y_test, y_pred, average='weighted')
            
            results_list.append({'Model': name, 'Accuracy': acc, 'F1-Score': f1, 'Precision': prec, 'Recall': rec})
            trained_pipelines[name] = pipeline
            
            print(f"{name:<30} | {acc:.4f}   | {f1:.4f}   | {prec:.4f}   | {rec:.4f}")
            
        except Exception as e:
            print(f"❌ Lỗi {name}: {e}")

    # 5. XUẤT KẾT QUẢ & BIỂU ĐỒ
    os.makedirs('DA/images', exist_ok=True)
    results_df = pd.DataFrame(results_list).sort_values(by='Accuracy', ascending=False)
    
    # Biểu đồ So sánh Hiệu năng
    plt.figure(figsize=(14, 8))
    sns.set_theme(style="whitegrid")
    # Vẽ biểu đồ barplot
    ax = sns.barplot(data=results_df, x='Accuracy', y='Model', palette='viridis')
    
    # Thêm nhãn giá trị lên cột
    for i, v in enumerate(results_df['Accuracy']):
        ax.text(v + 0.005, i, f"{v:.1%}", color='black', va='center', fontweight='bold')
    
    plt.title('So sánh Toàn diện: Thành phần lẻ vs Tổ hợp (Ensemble)', fontsize=15, fontweight='bold')
    plt.xlabel('Độ chính xác (Accuracy)')
    plt.xlim(0, 1.0) # Scale trục x từ 0 đến 100%
    plt.axvline(x=0.7, color='red', linestyle='--', label='Target 70%') # Đường tham chiếu
    plt.legend()
    plt.tight_layout()
    plt.savefig('DA/images/complete_model_comparison.png')
    
    print("\n✅ Đã lưu biểu đồ so sánh chi tiết tại: DA/images/complete_model_comparison.png")
    print("🏆 Top 3 Mô hình tốt nhất:")
    print(results_df[['Model', 'Accuracy', 'F1-Score']].head(3))
    
    try:
        # 1. Lấy tên model có độ chính xác cao nhất (Quán quân)
        best_model_name = results_df.iloc[0]['Model']
        print(f"\n📊 Đang kiểm tra vẽ Feature Importance cho model tốt nhất: {best_model_name}")
        
        target_name = None
        importances = None
        pipeline = None

        # 2. Kiểm tra xem quán quân có vẽ được không
        if best_model_name in trained_pipelines:
            temp_pipe = trained_pipelines[best_model_name]
            temp_clf = temp_pipe.named_steps['clf']
            
            if hasattr(temp_clf, 'feature_importances_'):
                target_name = best_model_name
                importances = temp_clf.feature_importances_
                pipeline = temp_pipe
            else:
                print(f"⚠️ Model quán quân ({best_model_name}) không hỗ trợ thuộc tính 'feature_importances_' (thường là Voting/Stacking/SVM/KNN).")
        
        save_dir = 'DA/pkl_file'
        os.makedirs(save_dir, exist_ok=True)
        
        # FILE 1: All models
        all_models_path = f'{save_dir}/all_models_p3_sentiment.pkl'
        joblib.dump({
            'models': fitted_pipelines,
            'numeric_feats': numeric_feats,
            'target_names': ['Non-Positive', 'Positive'] # Nhãn nhị phân
        }, all_models_path)
        print(f"💾 Đã lưu bộ toàn tập model tại: {all_models_path}")
        
        # FILE 2: Best model
        if best_model_name in fitted_pipelines:
            best_model_path = f'{save_dir}/best_model_p3_sentiment.pkl'
            joblib.dump({
                'pipeline': fitted_pipelines[best_model_name],
                'model_name': best_model_name,
                'accuracy': best_acc
            }, best_model_path)
            print(f"💾 Đã lưu model tốt nhất ({best_model_name}) tại: {best_model_path}")

        # 3. Nếu quán quân không vẽ được, tìm model CÂY/BOOSTING tốt nhất tiếp theo để thay thế
        if target_name is None:
            print("🔄 Đang tìm model Tree-based tốt nhất tiếp theo để minh họa Feature Importance...")
            for idx, row in results_df.iterrows():
                model_name = row['Model']
                # Bỏ qua model quán quân đã check rồi
                if model_name == best_model_name: continue
                
                # Chỉ xét các model có khả năng cao có feature importance (Tree, Forest, Boost)
                if any(k in model_name for k in ['Tree', 'Forest', 'Boost', 'Ada']):
                    if model_name in trained_pipelines:
                        temp_pipe = trained_pipelines[model_name]
                        temp_clf = temp_pipe.named_steps['clf']
                        if hasattr(temp_clf, 'feature_importances_'):
                            target_name = model_name
                            importances = temp_clf.feature_importances_
                            pipeline = temp_pipe
                            print(f"✅ Đã chọn model thay thế để vẽ: {target_name}")
                            break
        
        # 4. Tiến hành vẽ nếu tìm được model phù hợp
        if target_name and pipeline is not None:
            # Lấy tên features từ Preprocessor
            tf_idf_names = pipeline.named_steps['preprocessor'].named_transformers_['text'].get_feature_names_out()
            all_feat_names = np.array(numeric_feats + list(tf_idf_names))
            
            # Lọc theo Selector (nếu có trong pipeline)
            if 'selector' in pipeline.named_steps:
                mask = pipeline.named_steps['selector'].get_support()
                selected_feats = all_feat_names[mask]
            else:
                selected_feats = all_feat_names
            
            # Tạo DataFrame
            feat_imp_df = pd.DataFrame({'Feature': selected_feats, 'Importance': importances})
            feat_imp_df = feat_imp_df.sort_values(by='Importance', ascending=False).head(20)
            
            # Vẽ
            plt.figure(figsize=(10, 8))
            sns.barplot(data=feat_imp_df, x='Importance', y='Feature', palette='rocket')
            plt.title(f'Top 20 Features quan trọng nhất ({target_name})', fontsize=14)
            plt.tight_layout()
            plt.savefig('DA/images/feature_importance.png')
            print(f"✅ Đã lưu biểu đồ Feature Importance ({target_name}) tại: DA/images/feature_importance.png")
        else:
            print("❌ Không tìm thấy model nào hỗ trợ vẽ Feature Importance.")

    except Exception as e:
        print(f"⚠️ Lỗi khi vẽ Feature Importance: {e}")

if __name__ == "__main__":
    run_analysis_complete()