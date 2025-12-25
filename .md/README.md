# Vietnamese Hit Songs Data Scraper

Bộ công cụ thu thập và phân tích dữ liệu bảng xếp hạng nhạc Việt từ nhiều nguồn.

## 📁 Cấu trúc thư mục

```
Hit_Songs_DA/
├── scrapers/           # Các script thu thập dữ liệu
│   ├── nct_top50.py           # NCT Top 50 chart scraper
│   ├── zingmp3_top100.py      # Zing MP3 Top 100 chart scraper
│   └── spotify_top200_kworb.py # Spotify Vietnam Top 200 (kworb.net)
│
├── scripts/            # Các script xử lý và tự động hóa
│   ├── run_all_daily.ps1              # Thu thập tự động 3 nguồn hàng ngày
│   ├── setup_daily_task.ps1           # Thiết lập Windows Task Scheduler
│   ├── compare_nct_charts.py          # So sánh thay đổi NCT BXH
│   ├── dedupe_nct_by_date_rank.py     # Loại bỏ trùng lặp NCT
│   ├── filter_spotify_kworb_date_rank.py # Loại bỏ trùng lặp Spotify
│   ├── standardize_schemas.py         # Chuẩn hóa schema CSV
│   └── backfill_nct_history.ps1      # Backfill dữ liệu NCT lịch sử
│
├── data/               # Dữ liệu đầu ra
│   ├── nct_top50.csv                 # NCT data (chuẩn hóa)
│   ├── zingmp3_top100.csv            # Zing MP3 data (chuẩn hóa)
│   ├── spotify_top200_kworb.csv      # Spotify raw data
│   └── spotify_top200_kworb_clean.csv # Spotify data (đã loại trùng)
│
├── analysis/           # Scripts phân tích dữ liệu
│   ├── get_metadata.py
│   ├── process_hits_features.py
│   └── vpop_hit_analysis.py
│
├── archive/            # Dữ liệu cũ/backup
├── .venv/              # Python virtual environment
├── README.md           # Hướng dẫn tổng quan
└── SETUP_GUIDE.md      # Hướng dẫn thiết lập chi tiết
```

## 🎯 Schema dữ liệu chuẩn

### NCT Top 50
```csv
Date,Rank,Title,Artists,Duration,URL,Total_Likes
```
- **Total_Likes**: Trích xuất chính xác từ JSON `__NUXT_DATA__`

### Zing MP3 Top 100 (Deep Scrape)
```csv
Date,Rank,Title,Artists,Duration,URL,Total_Likes,Total_Plays,Release_Date,Genre,Album
```
- **Total_Likes**: Lượt thích (210K → 210,000)
- **Total_Plays**: Tổng lượt nghe (9M → 9,000,000)  
- **Release_Date**: Ngày phát hành (YYYY-MM-DD)
- **Genre**: Thể loại nhạc
- **Album**: Tên album

**Note**: Scraper tự động nâng cấp schema cũ (7 cột) lên mới (11 cột) với backup.

### Spotify Vietnam
```csv
Date,rank,title,artist,featured_artists,days_on_chart,peak_position,
peak_count,daily_streams,streams_change,seven_day_streams,
seven_day_change,total_streams
```

## 🚀 Hướng dẫn sử dụng

### 1. Thu thập dữ liệu NCT Top 50

**Scrape ngày hiện tại:**
```powershell
python scrapers/nct_top50.py
```

**Scrape ngày cụ thể:**
```powershell
$env:NCT_DATE="2025-10-31"
python scrapers/nct_top50.py
```

**Single-file mode (append vào 1 file):**
```powershell
$env:NCT_OUTPUT_FILE="data/nct_top50.csv"
python scrapers/nct_top50.py
```

### 2. Thu thập dữ liệu Zing MP3 Top 100

**Deep Scrape với đầy đủ metadata:**
```powershell
python scrapers/zingmp3_top100.py
```

**Single-file append mode:**
```powershell
$env:ZING_OUTPUT_FILE="data/zingmp3_top100.csv"
python scrapers/zingmp3_top100.py
```

**Giới hạn số bài (test):**
```powershell
$env:ZING_MAX_ITEMS="10"
python scrapers/zingmp3_top100.py
```

**Deep scrape tự động:**
- Click vào nút "Khác" (3 chấm) cho mỗi bài
- Hover lên tiêu đề bài hát để hiện panel thông tin
- Trích xuất: Total_Likes, Total_Plays, Release_Date, Genre, Album
- Parse compact numbers: 210K → 210,000, 9M → 9,000,000

### 3. Thu thập dữ liệu Spotify Vietnam

**Scrape từ kworb.net:**
```powershell
$env:SPOTIFY_WITH_DATE="1"
$env:SPOTIFY_OUTPUT_FILE="data/spotify_top200_kworb.csv"
python scrapers/spotify_top200_kworb.py
```

### 4. Xử lý dữ liệu

**Loại bỏ trùng lặp NCT:**
```powershell
python scripts/dedupe_nct_by_date_rank.py
```

**Loại bỏ trùng lặp Spotify:**
```powershell
python scripts/filter_spotify_kworb_date_rank.py
```

**Chuẩn hóa schema:**
```powershell
python scripts/standardize_schemas.py
```

### 5. Backfill dữ liệu lịch sử

**NCT (khoảng ngày):**
```powershell
./scripts/backfill_nct_history.ps1 -StartDate "2025-10-01" -EndDate "2025-10-31"
```

### 6. Chạy tự động hàng ngày

**Chạy thủ công:**
```powershell
.\scripts\run_all_daily.ps1
```

**So sánh charts:**
```powershell
python scripts\compare_nct_charts.py
```

**Thiết lập Windows Task Scheduler (chạy lúc 6:00 AM):**
```powershell
# Chạy với quyền Administrator
.\scripts\setup_daily_task.ps1
```

Xem hướng dẫn chi tiết trong [SETUP_GUIDE.md](SETUP_GUIDE.md)

Script sẽ tự động:
- Thu thập NCT Top 50
- Thu thập Zing MP3 Top 100
- Thu thập Spotify Vietnam Top 200
- Loại bỏ trùng lặp và làm sạch dữ liệu

## 🔧 Biến môi trường

### NCT Top 50
- `NCT_DATE`: Ngày cần scrape (YYYY-MM-DD), mặc định = hôm nay
- `NCT_OUTPUT_FILE`: File đầu ra, mặc định = `nct_top50_{date}.csv`
- `NCT_MAX_ITEMS`: Giới hạn số bài hát (test), mặc định = 50

### Zing MP3
- `ZING_DATE`: Ngày cần scrape (YYYY-MM-DD), mặc định = hôm nay
- `ZING_OUTPUT_FILE`: File đầu ra, mặc định = `zingmp3_top100.csv`

### Spotify Vietnam
- `SPOTIFY_WITH_DATE`: Set = "1" để bật cột Date
- `SPOTIFY_OUTPUT_FILE`: File đầu ra, mặc định = `spotify_top200_kworb.csv`

## 📊 Tính năng chính

✅ **Thu thập dữ liệu tự động** từ 3 nguồn: NCT, Zing MP3, Spotify  
✅ **Schema chuẩn hóa** - Dễ dàng phân tích với pandas  
✅ **Loại bỏ trùng lặp** - Dữ liệu sạch theo (Date, Rank)  
✅ **Backfill lịch sử** - Thu thập dữ liệu quá khứ tự động  
✅ **Single-file mode** - Tất cả dữ liệu trong 1 file  
✅ **Stealth mode** - Bypass automation detection  

## 🛠 Yêu cầu hệ thống

- Python 3.8+
- Chrome browser
- PowerShell 5.1+

### Packages
```
selenium
webdriver-manager
beautifulsoup4
selenium-stealth
pandas
```

## 📝 Ghi chú

- NCT Top 50: Dữ liệu từ `nhaccuatui.com/chart`, bao gồm Total_Likes chính xác
- Zing MP3: Dữ liệu từ `zingmp3.vn/zing-chart`, Total_Likes để trống (không có API)
- Spotify: Dữ liệu từ `kworb.net/spotify/country/vn`, bao gồm streaming metrics

## 📈 Dữ liệu hiện có

- **NCT**: 200 bản ghi (31/10 - 05/11/2025)
- **Zing**: 100 bản ghi (05/11/2025)
- **Spotify**: 1000 bản ghi (đã loại trùng)

---

**Cập nhật:** 05/11/2025
