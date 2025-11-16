# CÁC BƯỚC CHẠY
- Bước 1: run scripts\run_all.py          (done)
- Bước 2: run scrapers\song_list_info.py  (done)
- Bước 3: run scripts\is_hit.py           (done)

# CHI TIẾT FILE is_hit.py

title,artists,featured_artists,release_date,genres,spotify_popularity,total_platforms,score_platform,A_total_appearances,Z_total_appearances,N_total_appearances,S_total_appearances,total_appearances,score_sustain,Z_best_rank,S_best_rank,N_best_rank,A_best_rank,score_rank,S_total_streams,Z_total_plays,score_historical,total_score,is-hit

total_score = [Rank(0-12)] + [Sustain(0-1)] + [Platform(0-1)] + [Historical(0-3)] = (0-17)

Threshold = 11

spotify_popularity = 50

[Rank(0-12)]
Bậc 1 (Top 1-10): 3 điểm
Bậc 2 (Top 11-20): 2 điểm
Bậc 3 (Top 21-40): 1 điểm
41 - 50: 0 điểm

[Sustain(0-1)]
Total_Appearances >= 30

[Platform(0-1)]
total_platforms >= 3

[Historical(0-3)]
Spotify Streams >= 10,000,000
Zing Plays >= 1,000,000
NCT Likes >= 30,000

FINAL_HIT_THRESHOLD = 9
POPULARITY_LIFELINE = 60 (total_score < 9)


## 🎯 MỤC TIÊU PHÂN TÍCH
## 📁 Cấu trúc

- `scrapers/` - Scripts thu thập dữ liệu
- `scripts/` - Scripts xử lý và tự động hóa
- `data/` - CSV files
- `analysis/` - Scripts phân tích
- `archive/` - Dữ liệu cũ

## 📖 Tài liệu

- [README.md](README.md) - Hướng dẫn chi tiết
- [SETUP_GUIDE.md](SETUP_GUIDE.md) - Hướng dẫn thiết lập

1. **Phân tích các yếu tố tạo hit** của bài hát Vpop
2. **Xu hướng thể loại nhạc hit** tại Việt Nam
3. **Dự đoán khả năng hit** của bài hát mới

---

## 📊 PHƯƠNG ÁN 1: MASTER DATASET (ĐÃ TRIỂN KHAI) ✅

### File: `data/vpop_master_dataset.csv`

**Cách tiếp cận:** Gộp tất cả nguồn → Tính Hit Score → Tạo metrics tổng hợp

### ✅ Ưu điểm:
- **Dữ liệu tập trung** - 1 file duy nhất
- **Metrics đầy đủ** - có tất cả thông số từ 5 nền tảng
- **Hit Score** - chỉ số định lượng độ hit
- **Cross-platform analysis** - phân tích xu hướng đa nền tảng
- **Dễ dùng** - import 1 file là có tất cả

### ⚠️ Hạn chế:
- **Mất thông tin timeline** - không theo dõi được biến động theo ngày
- **Không phân tích trend** - không biết bài hát tăng/giảm rank
- **Aggregate data** - chỉ có tổng hợp, không có chi tiết từng ngày

### 🎯 Use cases:
- ✅ **Xếp hạng hit nhất** (theo Total Hit Score)
- ✅ **Phân tích cross-platform** (xuất hiện trên bao nhiêu nền tảng)
- ✅ **Modeling cơ bản** (features tổng hợp)
- ❌ Không phù hợp cho **time series analysis**

---

## 📈 PHƯƠNG ÁN 2: TIME SERIES DATASET (ĐỀ XUẤT)

### File: `data/vpop_timeseries_dataset.csv`

**Cách tiếp cận:** Giữ nguyên timeline → Merge theo (Date, Title, Artist) → Pivot metrics theo platform

### Schema đề xuất:

| Column | Type | Mô tả | Ví dụ |
|--------|------|-------|-------|
| **Date** | date | Ngày thu thập | 2025-11-05 |
| **Title** | str | Tên bài hát | "Em" |
| **Artist** | str | Nghệ sĩ | "Binz" |
| **Duration** | str | Thời lượng | "03:45" |
| | | **SPOTIFY METRICS** | |
| **Spotify_Rank** | int | Vị trí Spotify | 2 |
| **Spotify_Peak_Position** | int | Vị trí peak | 1 |
| **Spotify_Days_On_Chart** | int | Số ngày trên BXH | 15 |
| **Spotify_Daily_Streams** | int | Lượt nghe hôm nay | 150000 |
| **Spotify_Total_Streams** | int | Tổng lượt nghe | 5000000 |
| **Spotify_Streams_Change** | int | Thay đổi so với hôm qua | +10000 |
| | | **APPLE MUSIC METRICS** | |
| **AppleMusic_Rank** | int | Vị trí Apple Music | 3 |
| | | **ZINGMP3 METRICS** | |
| **ZingMP3_Rank** | int | Vị trí ZingMP3 | 1 |
| **ZingMP3_Total_Plays** | int | Tổng lượt nghe | 3000000 |
| **ZingMP3_Total_Likes** | int | Tổng lượt thích | 50000 |
| | | **NCT METRICS** | |
| **NCT_Rank** | int | Vị trí NCT | 2 |
| **NCT_Total_Likes** | int | Tổng lượt thích | 40000 |
| | | **YOUTUBE METRICS** | |
| **YouTube_Rank** | int | Vị trí YouTube | 1 |
| **YouTube_Total_Views** | int | Tổng lượt xem | 2000000 |
| **YouTube_Total_Likes** | int | Tổng lượt thích | 30000 |
| | | **CALCULATED METRICS** | |
| **Daily_Hit_Score** | float | Điểm hit hôm nay | 385.5 |
| **Platform_Count** | int | Số nền tảng xuất hiện hôm nay | 4 |
| **Avg_Rank** | float | Vị trí trung bình | 2.0 |
| **Rank_Trend** | str | Xu hướng rank | "rising" / "stable" / "falling" |
| **Days_Trending** | int | Số ngày liên tục trending | 5 |

### ✅ Ưu điểm:
- **Phân tích xu hướng** - theo dõi biến động theo thời gian
- **Time series modeling** - ARIMA, LSTM, Prophet
- **Trend detection** - phát hiện bài hát đang lên/xuống
- **Seasonality analysis** - phân tích mùa vụ hit
- **Prediction** - dự đoán rank ngày tiếp theo

### ⚠️ Hạn chế:
- **File size lớn** - mỗi bài × mỗi ngày = nhiều rows
- **Sparse data** - nhiều null (không phải bài nào cũng có trên mọi platform mỗi ngày)
- **Phức tạp** - cần xử lý missing data, interpolation

### 🎯 Use cases:
- ✅ **Phân tích xu hướng** - bài hát tăng/giảm rank theo thời gian
- ✅ **Viral detection** - phát hiện bài hát đột ngột viral
- ✅ **Time series forecasting** - dự đoán rank tương lai
- ✅ **Seasonality** - phân tích mùa hit (Tết, lễ hội...)

---

## 🎨 PHƯƠNG ÁN 3: HYBRID DATASET (KHUYẾN NGHỊ) ⭐

### Kết hợp cả 2 phương án:

1. **Master Dataset** (`vpop_master_dataset.csv`)
   - Dùng cho: Overview, ranking, cross-platform analysis
   - Update: Hàng ngày (append mới + recalculate)

2. **Time Series Dataset** (`vpop_timeseries_dataset.csv`)
   - Dùng cho: Trend analysis, forecasting, viral detection
   - Update: Hàng ngày (append new rows)

3. **Features Dataset** (`vpop_features_dataset.csv`)
   - Spotify Audio Features: danceability, energy, tempo...
   - Artist metadata: followers, popularity, genres
   - Dùng cho: Machine learning modeling

### 📁 Final Structure:

```
data/
├── vpop_master_dataset.csv          # Tổng hợp tất cả (341 songs unique)
├── vpop_timeseries_dataset.csv      # Timeline đầy đủ (1399+ rows)
├── vpop_features_dataset.csv        # Audio features (Spotify API)
│
├── raw/                              # Raw data từng nguồn
│   ├── spotify_top100_kworb_vietnam.csv
│   ├── apple_music_top100_kworb_vietnam.csv
│   ├── zingmp3_top100.csv
│   ├── nct_top50.csv
│   └── youtube_trendz_top30.csv
│
└── backup/                           # Backup tự động
```

---

## 🔬 PHÂN TÍCH ĐỀ XUẤT

### 1️⃣ Exploratory Data Analysis (EDA)

**Dùng Master Dataset:**
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('data/vpop_master_dataset.csv')

# Top 20 hit nhất
top_20 = df.nlargest(20, 'Total_Hit_Score')

# Cross-platform distribution
plt.figure(figsize=(10, 6))
df['Platform_Count'].value_counts().plot(kind='bar')
plt.title('Phân bố số nền tảng')

# Correlation matrix
metrics = ['Total_Hit_Score', 'Platform_Count', 'Spotify_Total_Streams', 
           'ZingMP3_Total_Plays', 'YouTube_Total_Views']
sns.heatmap(df[metrics].corr(), annot=True)
```

### 2️⃣ Trend Analysis

**Dùng Time Series Dataset:**
```python
ts_df = pd.read_csv('data/vpop_timeseries_dataset.csv')
ts_df['Date'] = pd.to_datetime(ts_df['Date'])

# Theo dõi 1 bài hát theo thời gian
song = ts_df[ts_df['Title'] == 'Em'].sort_values('Date')

plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(song['Date'], song['Spotify_Rank'], label='Spotify')
plt.plot(song['Date'], song['ZingMP3_Rank'], label='ZingMP3')
plt.legend()
plt.title('Rank Trend - Em by Binz')

plt.subplot(2, 1, 2)
plt.plot(song['Date'], song['Spotify_Daily_Streams'])
plt.title('Daily Streams Trend')
```

### 3️⃣ Feature Engineering

**Tạo features cho ML:**
```python
# Từ Master Dataset
df['Avg_Rank_Per_Platform'] = df['Avg_Rank'] / df['Platform_Count']
df['Hit_Density'] = df['Total_Hit_Score'] / df['Date_Count']
df['Platform_Diversity'] = df['Platform_Count'] / 5  # Normalized

# Từ Time Series Dataset
ts_df['Rank_Velocity'] = ts_df.groupby('Title')['Avg_Rank'].diff()
ts_df['Streams_Growth_Rate'] = ts_df.groupby('Title')['Spotify_Daily_Streams'].pct_change()
ts_df['Is_Rising'] = ts_df['Rank_Velocity'] < 0  # Rank giảm = đang lên
```

### 4️⃣ Machine Learning Models

#### Model 1: Hit Classification
**Target:** Binary (hit / not hit)  
**Features:** Platform_Count, Total_Hit_Score, Audio Features  
**Models:** Random Forest, XGBoost, Neural Network

```python
from sklearn.ensemble import RandomForestClassifier

# Define hit: Total_Hit_Score > threshold
df['Is_Hit'] = df['Total_Hit_Score'] > df['Total_Hit_Score'].quantile(0.75)

features = ['Platform_Count', 'Avg_Rank', 'Spotify_Total_Streams', ...]
X = df[features]
y = df['Is_Hit']

model = RandomForestClassifier()
model.fit(X_train, y_train)
```

#### Model 2: Rank Prediction
**Target:** Next day rank  
**Features:** Historical ranks, streams, time features  
**Models:** LSTM, Prophet, ARIMA

```python
from prophet import Prophet

# Forecast Spotify rank
song_data = ts_df[ts_df['Title'] == 'Em'][['Date', 'Spotify_Rank']]
song_data.columns = ['ds', 'y']

model = Prophet()
model.fit(song_data)
future = model.make_future_dataframe(periods=7)
forecast = model.predict(future)
```

### 5️⃣ Genre & Artist Analysis

**Thêm metadata từ Spotify API:**
```python
# File: analysis/get_metadata.py
import spotipy

sp = spotipy.Spotify(auth_manager=...)

# Lấy genres
artist_info = sp.artist(artist_id)
genres = artist_info['genres']  # ['vpop', 'pop', 'r&b']

# Lấy audio features
features = sp.audio_features(track_id)
danceability = features[0]['danceability']
energy = features[0]['energy']
valence = features[0]['valence']  # Happiness
```

**Phân tích:**
- Genre distribution của hit songs
- Correlation giữa audio features và hit score
- Artist popularity vs song performance

---

## 📋 ACTION PLAN

### Phase 1: Data Collection (DONE ✅)
- [x] 5 scrapers hoạt động
- [x] Automated daily run
- [x] Master dataset created

### Phase 2: Time Series Dataset (NEXT)
```python
# Tạo script: scripts/create_timeseries_dataset.py
# Merge theo (Date, Title, Artist)
# Pivot metrics theo platform
# Add calculated fields
```

### Phase 3: Feature Dataset
```python
# Sử dụng analysis/get_metadata.py
# Thêm Spotify audio features
# Thêm artist metadata
# Merge vào master dataset
```

### Phase 4: Analysis & Modeling
- EDA notebooks
- Trend analysis
- ML models
- Dashboard (Streamlit/PowerBI)

---

## 🎯 KẾT LUẬN & KHUYẾN NGHỊ

### ✅ Nên làm ngay:

1. **Giữ Master Dataset hiện tại** - đã tốt cho overview
2. **Tạo Time Series Dataset** - cần thiết cho trend analysis
3. **Thu thập Spotify Features** - quan trọng cho modeling
4. **Setup monitoring** - theo dõi data quality

### 📊 Sử dụng dataset nào cho mục đích gì:

| Mục đích | Dataset | Script/Tool |
|----------|---------|-------------|
| Xếp hạng hit nhất | Master Dataset | Pandas groupby |
| Phân tích xu hướng | Time Series Dataset | Matplotlib, Seaborn |
| Dự đoán rank | Time Series Dataset | Prophet, LSTM |
| Phân loại hit/non-hit | Master + Features | Random Forest, XGBoost |
| Phân tích thể loại | Features Dataset | Spotify API |
| Dashboard | Master + Time Series | Streamlit, PowerBI |

### 🚀 Next Steps:

1. Tạo `create_timeseries_dataset.py`
2. Tạo `enrich_with_spotify_features.py`
3. Tạo Jupyter notebooks cho EDA
4. Build ML models
5. Create dashboard

---

**📅 Cập nhật:** 2025-11-06  
**📧 Contact:** [Your Email]
