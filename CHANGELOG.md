
**Thống kê:**
- 📊 **341 bài hát duy nhất** (từ 1399 records raw)
- 📅 Thời gian: 2025-10-31 → 2025-11-06
- 🏆 Top hit: "Anh Đã Không Biết Cách Yêu Em" - Quang Đăng Trần (1338.4 điểm)

**Metrics:**
- Platform distribution: 298 bài (1 platform), 32 bài (2 platforms), 8 bài (3 platforms), 3 bài (4 platforms)
- Cross-platform hits: 3 bài xuất hiện trên 4/5 nền tảng


### 6. TÀI LIỆU MỚI

**File mới tạo:**
1. ✅ `USAGE_GUIDE.md` - Hướng dẫn sử dụng chi tiết
2. ✅ `DATA_MERGE_PROPOSAL.md` - Đề xuất phương án gộp dữ liệu
3. ✅ `CHANGELOG.md` - File này

**Nội dung:**
- Hướng dẫn sử dụng từng script
- Schema của Master Dataset
- 3 phương án merge data (Master, Time Series, Hybrid)
- Đề xuất phân tích và modeling
- Roadmap phát triển

---

## 📁 CẤU TRÚC SAU KHI SẮP XẾP

```
Hit_Songs_DA/
│
├── 📂 data/                              # DỮ LIỆU CHÍNH
│   ├── ⭐ vpop_master_dataset.csv       # DATASET TỔNG HỢP (341 songs)
│   │
│   ├── spotify_top100_kworb.csv          (400 records)
│   ├── spotify_top100_kworb_vietnam.csv  (378 records)
│   ├── apple_music_top100_kworb.csv      (100 records)
│   ├── apple_music_top100_kworb_vietnam.csv (91 records)
│   ├── zingmp3_top100.csv                (700 records)
│   ├── nct_top50.csv                     (200 records)
│   ├── youtube_trendz_top30.csv          (30 records)
│   │
│   └── 📂 backup/
│       └── backup_20251106_172209/       (8 files)
│
├── 📂 scrapers/                          # THU THẬP DỮ LIỆU
│   ├── spotify_top100_kworb.py
│   ├── apple_music_top100_kworb.py       ← Đã sửa (ngày hôm qua)
│   ├── zingmp3_top100.py
│   ├── nct_top50.py
│   └── youtube_trendz_top30.py
│
├── 📂 scripts/                           # XỬ LÝ DỮ LIỆU
│   ├── ⭐ run_all_daily.ps1             # CHẠY TẤT CẢ (7 bước)
│   ├── standardize_headers.py            ← MỚI
│   ├── merge_master_dataset.py           ← MỚI
│   └── cleanup_files.py                  ← MỚI
│
├── 📂 analysis/                          # PHÂN TÍCH
│   ├── get_metadata.py                   (giữ lại)
│   ├── process_hits_features.py          (giữ lại)
│   └── vpop_hit_analysis.py              (giữ lại)
│
├── 📂 archive/                           # DỮ LIỆU CŨ
│
├── filter_vietnam_songs.py               (Spotify)
├── filter_apple_music_vietnam.py         (Apple Music)
│
└── 📝 TÀI LIỆU
    ├── USAGE_GUIDE.md                    ← MỚI
    ├── DATA_MERGE_PROPOSAL.md            ← MỚI
    ├── CHANGELOG.md                      ← MỚI
    ├── README.md
    ├── QUICKSTART.md
    └── SETUP_GUIDE.md
```

---

## 🎯 HEADERS CHUẨN (Sau khi standardize)

### Spotify
```
Date, Rank, Title, Artist, Featured_Artists, Days_On_Chart, 
Peak_Position, Peak_Count, Daily_Streams, Streams_Change, 
Seven_Day_Streams, Seven_Day_Change, Total_Streams, Source
```

### Apple Music
```
Date, Rank, Title, Artist, Source
```

### ZingMP3
```
Date, Rank, Title, Artists, Duration, URL, Total_Likes, Total_Plays, Source
```

### NCT
```
Date, Rank, Title, Artists, Duration, URL, Total_Likes, Source
```

### Master Dataset
```
Title, Artist, Appearance_Count, Platform_Count, Platforms, Date_Count,
First_Seen, Last_Seen, Best_Rank, Avg_Rank, Best_Platform,
Total_Hit_Score, Avg_Hit_Score,
Spotify_Peak_Position, Spotify_Total_Streams, Spotify_Days_On_Chart,
ZingMP3_Total_Plays, ZingMP3_Total_Likes,
NCT_Total_Likes,
YouTube_Total_Views, YouTube_Total_Likes,
Duration, Spotify_URL, ZingMP3_URL, NCT_URL, YouTube_URL
```

## 📊 ĐỀ XUẤT TIẾP THEO

### 1. TẠO TIME SERIES DATASET
**Mục đích:** Phân tích xu hướng, dự đoán rank

**Script cần tạo:** `scripts/create_timeseries_dataset.py`

**Output:** `data/vpop_timeseries_dataset.csv`

**Schema:**
```
Date, Title, Artist, Duration,
Spotify_Rank, Spotify_Daily_Streams, Spotify_Total_Streams,
AppleMusic_Rank,
ZingMP3_Rank, ZingMP3_Total_Plays, ZingMP3_Total_Likes,
NCT_Rank, NCT_Total_Likes,
YouTube_Rank, YouTube_Total_Views, YouTube_Total_Likes,
Daily_Hit_Score, Platform_Count, Avg_Rank, Rank_Trend
```

### 2. THU THẬP SPOTIFY AUDIO FEATURES
**Mục đích:** Feature engineering cho ML

**Script có sẵn:** `analysis/vpop_hit_analysis.py`

**Cần làm:**
- Lấy track IDs từ Spotify search
- Gọi API `audio_features`
- Merge vào master dataset

**Features:**
- danceability, energy, valence, tempo
- loudness, key, mode
- acousticness, instrumentalness, speechiness

### 3. PHÂN TÍCH & MODELING

**Notebooks cần tạo:**
1. `01_EDA.ipynb` - Exploratory Data Analysis
2. `02_Trend_Analysis.ipynb` - Time series analysis
3. `03_Feature_Engineering.ipynb` - Tạo features
4. `04_Hit_Classification.ipynb` - ML model (hit/non-hit)
5. `05_Rank_Prediction.ipynb` - LSTM/Prophet forecasting

### 4. DASHBOARD
**Tool:** Streamlit hoặc PowerBI

**Features:**
- Top 20 hit nhất (real-time)
- Trend charts (rank, streams theo thời gian)
- Platform comparison
- Artist performance
- Genre distribution

---

## ⚠️ LƯU Ý QUAN TRỌNG

### 1. Backup
- ✅ Tự động backup mỗi ngày khi chạy `run_all_daily.ps1`
- ✅ Lưu trong `data/backup/backup_YYYYMMDD/`
- ⚠️ Nên backup định kỳ ra ngoài project (Google Drive, OneDrive)

### 2. Data Quality
- ⚠️ Apple Music Kworb không có streams/plays (chỉ có rank)
- ⚠️ YouTube data có thể thiếu featured artists (parsing issue)
- ⚠️ Spotify featured_artists có khoảng trắng kép (cần clean)

### 3. Ghi đè dữ liệu
- ✅ Scrapers dùng **mode='a'** (append) - an toàn
- ✅ Có deduplication tự động
- ⚠️ Nếu chạy 2 lần trong ngày → duplicate → cần manual clean

### 4. Environment Variables
- ⚠️ Phải set đúng `$env:*_OUTPUT_FILE` trước khi chạy scraper
- ✅ Script `run_all_daily.ps1` đã set sẵn

---

## 📞 HỖ TRỢ

### Vấn đề thường gặp:

**1. "Không thu thập được dữ liệu!"**
- Check internet connection
- Check Kworb website có thay đổi HTML không
- Check encoding (phải là utf-8)

**2. Duplicate records**
- Chạy: `python scripts/standardize_headers.py` (có dedupe)
- Hoặc manual: `df.drop_duplicates(subset=['Date', 'Rank'])`

**3. Master dataset không cập nhật**
- Chạy lại: `python scripts/merge_master_dataset.py`
- Check file input có tồn tại không

**4. Spotify API 401**
- Kiểm tra CLIENT_ID, CLIENT_SECRET
- Tạo mới app tại: https://developer.spotify.com/dashboard

---

## 📅 LỊCH SỬ THAY ĐỔI CHI TIẾT

### 2025-11-06 - 17:22
1. ✅ Backup toàn bộ CSV (8 files)
2. ✅ Tạo `scripts/standardize_headers.py`
3. ✅ Chuẩn hóa headers tất cả files
4. ✅ Tạo `scripts/merge_master_dataset.py`
5. ✅ Generate Master Dataset (341 songs)
6. ✅ Cập nhật `run_all_daily.ps1` (7 bước)
7. ✅ Tạo `scripts/cleanup_files.py`
8. ✅ Dọn dẹp 4 files không cần thiết
9. ✅ Tạo `USAGE_GUIDE.md`
10. ✅ Tạo `DATA_MERGE_PROPOSAL.md`
11. ✅ Tạo `CHANGELOG.md`

### 2025-11-06 - Earlier
- ✅ Thêm Apple Music scraper
- ✅ Fix date logic (ngày hôm qua, không phải hôm nay)
- ✅ Tạo filter_apple_music_vietnam.py
- ✅ Lọc 91/100 bài nhạc VN

### 2025-11-05
- ✅ Fix Spotify peak_count = 0
- ✅ Fix NCT rank 30 missing
- ✅ Validate ZingMP3 (700 records)
- ✅ File organization (rename top200 → top100)

---

**📧 Liên hệ:** [Your Email]  
**📚 Docs:** USAGE_GUIDE.md, DATA_MERGE_PROPOSAL.md  
**🔗 Repo:** [GitHub URL]

---

✅ **ĐÃ HOÀN THÀNH TÁI TỔ CHỨC DỰ ÁN**
