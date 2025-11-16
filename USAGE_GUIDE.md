# VPOP HIT SONGS DATA ANALYSIS - HƯỚNG DẪN SỬ DỤNG

## 📋 TỔNG QUAN DỰ ÁN

Dự án thu thập và phân tích dữ liệu bảng xếp hạng âm nhạc Việt Nam từ nhiều nguồn để:
- **Phân tích các yếu tố tạo hit** của bài hát
- **Phát hiện xu hướng thể loại nhạc** hit tại Việt Nam
- **Dự đoán khả năng hit** của bài hát mới

### 🎯 Nguồn dữ liệu (5 nền tảng)

| Nền tảng | Số lượng | Metrics quan trọng | Tần suất cập nhật |
|----------|----------|-------------------|-------------------|
| **Spotify** | Top 100 | Streams, Peak Position, Days on Chart | Hàng ngày |
| **Apple Music** | Top 100 | Rank (Kworb không có streams) | Hàng ngày |
| **ZingMP3** | Top 100 | Total Plays, Total Likes | Hàng ngày |
| **NCT** | Top 50 | Total Likes, Duration | Hàng ngày |
| **YouTube Trendz** | Top 30 | Views, Likes | Hàng ngày |

---

## 📁 CẤU TRÚC DỰ ÁN

```
Hit_Songs_DA/
│
├── 📂 data/                          # DỮ LIỆU CHÍNH
│   ├── vpop_master_dataset.csv      # ★ DATASET TỔNG HỢP (341 bài hát duy nhất)
│   │
│   ├── spotify_top100_kworb.csv           # Spotify Top 100 (raw)
│   ├── spotify_top100_kworb_vietnam.csv   # Spotify Top 100 (chỉ nhạc VN)
│   │
│   ├── apple_music_top100_kworb.csv       # Apple Music Top 100 (raw)
│   ├── apple_music_top100_kworb_vietnam.csv # Apple Music (chỉ nhạc VN)
│   │
│   ├── zingmp3_top100.csv                 # ZingMP3 Top 100
│   ├── nct_top50.csv                      # NCT Top 50
│   ├── youtube_trendz_top30.csv           # YouTube Trendz Top 30
│   │
│   └── 📂 backup/                    # Backup tự động hàng ngày
│       └── backup_YYYYMMDD/
│
├── 📂 scrapers/                      # SCRIPTS THU THẬP DỮ LIỆU
│   ├── spotify_top100_kworb.py      # Scraper Spotify (Kworb)
│   ├── apple_music_top100_kworb.py  # Scraper Apple Music (Kworb)
│   ├── zingmp3_top100.py            # Scraper ZingMP3
│   ├── nct_top50.py                 # Scraper NCT
│   └── youtube_trendz_top30.py      # Scraper YouTube Trendz
│
├── 📂 scripts/                       # SCRIPTS XỬ LÝ DỮ LIỆU
│   ├── run_all_daily.ps1            # ★ CHẠY TẤT CẢ HÀNG NGÀY
│   ├── standardize_headers.py       # Chuẩn hóa headers
│   ├── merge_master_dataset.py      # Gộp thành master dataset
│   └── cleanup_files.py             # Dọn dẹp file
│
├── 📂 analysis/                      # SCRIPTS PHÂN TÍCH
│   ├── get_metadata.py              # Lấy metadata từ Spotify API
│   ├── process_hits_features.py     # Xử lý features cho modeling
│   └── vpop_hit_analysis.py         # Phân tích popularity score
│
├── 📂 archive/                       # DỮ LIỆU CŨ (tham khảo)
│
├── filter_vietnam_songs.py          # Lọc nhạc VN từ Spotify
└── filter_apple_music_vietnam.py    # Lọc nhạc VN từ Apple Music
```

---

## 🚀 CÁCH SỬ DỤNG

### ✅ Chạy thu thập dữ liệu hàng ngày (TỰ ĐỘNG)

```powershell
# Chạy tất cả scrapers + xử lý + tạo master dataset
.\scripts\run_all_daily.ps1
```

**Quy trình tự động:**
1. Backup dữ liệu hiện tại
2. Scrape NCT Top 50
3. Scrape ZingMP3 Top 100
4. Scrape Spotify Top 100
5. Scrape Apple Music Top 100
6. Scrape YouTube Trendz Top 30
7. Lọc nhạc Việt Nam (Spotify + Apple Music)
8. Chuẩn hóa headers
9. Gộp thành Master Dataset

### ✅ Chạy từng scraper riêng lẻ

```powershell
# Spotify
python scrapers\spotify_top100_kworb.py

# Apple Music
python scrapers\apple_music_top100_kworb.py

# ZingMP3
python scrapers\zingmp3_top100.py

# NCT
python scrapers\nct_top50.py

# YouTube Trendz
python scrapers\youtube_trendz_top30.py
```

### ✅ Lọc nhạc Việt Nam

```powershell
# Lọc Spotify
python filter_vietnam_songs.py

# Lọc Apple Music
python filter_apple_music_vietnam.py
```

### ✅ Tạo Master Dataset

```powershell
# Chuẩn hóa headers
python scripts\standardize_headers.py

# Merge thành master dataset
python scripts\merge_master_dataset.py
```

---

## 📊 MASTER DATASET - VPOP_MASTER_DATASET.CSV

### 🎯 Mục đích
File tổng hợp **TẤT CẢ DỮ LIỆU** từ 5 nền tảng, phân tích và tính toán metrics để:
- Xác định **bài hát hit nhất** (theo Hit Score)
- Phân tích **xu hướng cross-platform** (xuất hiện trên bao nhiêu nền tảng)
- So sánh **hiệu suất trên từng nền tảng**
- Cung cấp **metrics đầy đủ** cho machine learning

### 📋 Schema (27 columns)

| Column | Mô tả | Ví dụ |
|--------|-------|-------|
| **Title** | Tên bài hát | "chẳng phải tình đầu sao đau đến thế" |
| **Artist** | Nghệ sĩ | "MIN, Dangrangto & antransax" |
| **Appearance_Count** | Số lần xuất hiện (qua các ngày/nguồn) | 20 |
| **Platform_Count** | Số nền tảng xuất hiện | 4 |
| **Platforms** | Các nền tảng | "SPOTIFY, APPLE_MUSIC, ZINGMP3, NCT" |
| **Date_Count** | Số ngày xuất hiện | 5 |
| **First_Seen** | Ngày đầu tiên xuất hiện | "2025-10-31" |
| **Last_Seen** | Ngày cuối cùng xuất hiện | "2025-11-05" |
| **Best_Rank** | Vị trí cao nhất | 1 |
| **Avg_Rank** | Vị trí trung bình | 2.5 |
| **Best_Platform** | Nền tảng có rank tốt nhất | "SPOTIFY" |
| **Total_Hit_Score** | ★ Điểm hit tổng hợp | 1850.5 |
| **Avg_Hit_Score** | Điểm hit trung bình | 92.5 |
| **Spotify_Peak_Position** | Vị trí peak trên Spotify | 1 |
| **Spotify_Total_Streams** | Tổng lượt nghe Spotify | 5000000 |
| **Spotify_Days_On_Chart** | Số ngày trên BXH Spotify | 30 |
| **ZingMP3_Total_Plays** | Tổng lượt nghe ZingMP3 | 3000000 |
| **ZingMP3_Total_Likes** | Tổng lượt thích ZingMP3 | 50000 |
| **NCT_Total_Likes** | Tổng lượt thích NCT | 40000 |
| **YouTube_Total_Views** | Tổng lượt xem YouTube | 2000000 |
| **YouTube_Total_Likes** | Tổng lượt thích YouTube | 30000 |
| **Duration** | Thời lượng bài hát | "04:43" |
| **Spotify_URL** | Link Spotify | (chưa có) |
| **ZingMP3_URL** | Link ZingMP3 | "https://..." |
| **NCT_URL** | Link NCT | "https://..." |
| **YouTube_URL** | Link YouTube | "https://..." |

### 🏆 Hit Score Calculation

```
Hit Score = (101 - Rank) × Platform Weight
```

**Platform Weights:**
- Spotify: 1.0 (ưu tiên cao nhất)
- Apple Music: 0.95
- ZingMP3: 0.9
- NCT: 0.85
- YouTube: 0.8

**Total Hit Score** = Tổng điểm từ tất cả lần xuất hiện trên mọi nền tảng

---

## 🎯 PHÂN TÍCH DỮ LIỆU

### 1️⃣ Phân tích yếu tố tạo hit

```python
import pandas as pd

# Đọc master dataset
df = pd.read_csv('data/vpop_master_dataset.csv')

# Top 10 bài hit nhất
top_hits = df.nlargest(10, 'Total_Hit_Score')[
    ['Title', 'Artist', 'Total_Hit_Score', 'Platform_Count', 'Best_Rank']
]

# Phân tích correlation
# - Platform_Count vs Total_Hit_Score
# - Spotify_Total_Streams vs Total_Hit_Score
# - Duration vs Total_Hit_Score
```

### 2️⃣ Phân tích xu hướng cross-platform

```python
# Bài hát xuất hiện trên nhiều nền tảng nhất
cross_platform = df[df['Platform_Count'] >= 3].sort_values('Total_Hit_Score', ascending=False)

# Phân bố theo số nền tảng
platform_dist = df['Platform_Count'].value_counts()
```

### 3️⃣ Phân tích hiệu suất từng nền tảng

```python
# So sánh streams/plays/views
spotify_top = df.nlargest(10, 'Spotify_Total_Streams')
zingmp3_top = df.nlargest(10, 'ZingMP3_Total_Plays')
youtube_top = df.nlargest(10, 'YouTube_Total_Views')
```

### 4️⃣ Lấy Audio Features từ Spotify API

```python
# Sử dụng script có sẵn
python analysis/get_metadata.py
```

Metrics Spotify trả về:
- **danceability, energy, valence** (cảm xúc)
- **tempo, loudness, key** (âm nhạc)
- **acousticness, instrumentalness** (kiểu nhạc)
- **speechiness** (lời hát)

---

## 🔧 CẤU HÌNH

### Environment Variables (cho scrapers)

```powershell
# Spotify
$env:SPOTIFY_OUTPUT_FILE = "data\spotify_top100_kworb.csv"

# Apple Music
$env:APPLE_OUTPUT_FILE = "data\apple_music_top100_kworb.csv"

# ZingMP3
$env:ZING_OUTPUT_FILE = "data\zingmp3_top100.csv"

# NCT
$env:NCT_OUTPUT_FILE = "data\nct_top50.csv"

# YouTube
$env:YOUTUBE_OUTPUT_FILE = "data\youtube_trendz_top30.csv"
```

### Spotify API Credentials (cho analysis)

File: `analysis/vpop_hit_analysis.py`

```python
CLIENT_ID = "your_client_id"
CLIENT_SECRET = "your_client_secret"
```

---

## 📅 TỰ ĐỘNG HÓA

### Setup Windows Task Scheduler

```powershell
.\scripts\setup_daily_task.ps1
```

Lên lịch chạy:
- **Mỗi ngày lúc 18:00** (sau khi Kworb cập nhật lúc 16:44)
- Tự động backup
- Tự động merge master dataset

---

## 🎓 PHÂN TÍCH NÂNG CAO

### 1. Feature Engineering

- Tính **popularity trend** (tăng/giảm theo thời gian)
- Tính **viral coefficient** (tốc độ lan truyền)
- Tính **platform dominance** (nền tảng nào chiếm ưu thế)

### 2. Machine Learning Models

Dự đoán hit dựa trên:
- Audio features (từ Spotify API)
- Cross-platform metrics
- Artist popularity
- Release timing

### 3. Trend Analysis

- Thể loại nhạc hot (dựa trên genre từ Spotify)
- Nghệ sĩ đang lên
- Mùa vụ hit (theo tháng)

---

## 📞 HỖ TRỢ

### Các vấn đề thường gặp

**1. Scraper bị lỗi encoding Vietnamese:**
- Đảm bảo file CSV dùng `utf-8-sig`
- Check `encoding='utf-8-sig'` trong `pd.read_csv()`

**2. Duplicate data:**
- Chạy: `python scripts/standardize_headers.py` để deduplicate

**3. Missing data:**
- Check backup folder: `data/backup/`
- Restore từ backup gần nhất

**4. Spotify API 401 Unauthorized:**
- Kiểm tra CLIENT_ID và CLIENT_SECRET
- Tạo mới tại: https://developer.spotify.com/dashboard

---

## 📝 CHANGELOG

### v2.0 - 2025-11-06
- ✅ Thêm Apple Music scraper
- ✅ Chuẩn hóa headers tất cả file
- ✅ Tạo Master Dataset tổng hợp
- ✅ Thêm Hit Score calculation
- ✅ Cập nhật run_all_daily.ps1 (7 bước)
- ✅ Tự động backup hàng ngày

### v1.0 - 2025-10-31
- ✅ Spotify, ZingMP3, NCT, YouTube scrapers
- ✅ Vietnam filtering
- ✅ Basic data collection

---

## 🎯 ROADMAP

- [ ] Thêm genre classification
- [ ] Thêm artist metadata
- [ ] Dashboard visualization (Streamlit/PowerBI)
- [ ] Predictive modeling (hit/non-hit)
- [ ] Real-time alerts (bài hát mới trending)
- [ ] API endpoint (serve predictions)

---

**📧 Contact:** [Your Email]  
**📚 Documentation:** `README.md`, `QUICKSTART.md`  
**🔗 Repository:** [GitHub Link]
