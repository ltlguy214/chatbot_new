import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 1. Load dữ liệu
df = pd.read_csv('DA\\final_data\\VPop_5_Vibes_Final.csv')

# 2. Chọn các đặc trưng số để chạy PCA (loại bỏ metadata)
features = df.select_dtypes(include=['float64', 'int64']).columns
# Loại bỏ các cột không phải đặc trưng âm thanh nếu có (ví dụ: cluster ID)
features = [f for f in features if f not in ['cluster', 'is_hit', 'duration_sec']]

x = df[features]
y = df['vibe'] # Tên cột vibe/cluster của bạn

# 3. Chuẩn hóa và giảm chiều về 2D
x_scaled = StandardScaler().fit_transform(x)
pca = PCA(n_components=2)
pca_res = pca.fit_transform(x_scaled)

df_pca = pd.DataFrame(data=pca_res, columns=['PC1', 'PC2'])
df_pca['Vibe'] = y

# 4. Vẽ biểu đồ
plt.figure(figsize=(10, 7))
sns.scatterplot(x='PC1', y='PC2', hue='Vibe', data=df_pca, palette='viridis', alpha=0.7)
plt.title('Trực quan hóa các cụm Vibe bằng PCA')

# lưu biểu đồ
plt.savefig('DA\\tasks\\Style\\pca_vibe_plot.png')
plt.close()