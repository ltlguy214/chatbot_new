import pandas as pd
import os
from datetime import datetime
from pathlib import Path

# Thử import pyuca
try:
    from pyuca import Collator
    pyuca_available = True
except ImportError:
    pyuca_available = False
    Collator = None # Đặt là None để kiểm tra sau

def format_number(num):
    """Format số với dấu phẩy ngăn cách hàng nghìn"""
    return f"{num:,}"

def get_file_size(filepath):
    """Lấy kích thước file theo KB"""
    try:
        size_bytes = os.path.getsize(filepath)
        return f"{size_bytes / 1024:.2f} KB"
    except Exception:
        return "Không rõ"

def analyze_csv_file(filepath, file_description, collator=None):
    """
    Phân tích một file CSV và trả về thống kê.
    'collator' là một đối tượng pyuca.Collator() hoặc None.
    """
    try:
        # Thử đọc với utf-8-sig trước, nếu lỗi thì thử utf-8
        try:
            df = pd.read_csv(filepath, encoding='utf-8-sig')
        except UnicodeDecodeError:
            df = pd.read_csv(filepath, encoding='utf-8')
        
        stats = {
            'file': os.path.basename(filepath),
            'description': file_description,
            'records': len(df),
            'columns': len(df.columns),
            'size': get_file_size(filepath),
            'column_names': list(df.columns)
        }
        
        # Thống kê theo ngày nếu có cột Date
        if 'Date' in df.columns:
            try:
                date_stats = df.groupby('Date').size().to_dict()
                stats['dates'] = sorted(date_stats.keys())
                stats['date_counts'] = date_stats
                stats['date_range'] = f"{min(stats['dates'])} → {max(stats['dates'])}"
            except Exception as e:
                stats['date_error'] = f"Lỗi xử lý cột Date: {e}"

        # Thống kê theo Source nếu có
        if 'Source' in df.columns:
            stats['sources'] = df['Source'].value_counts().to_dict()
        
        # Thống kê unique values cho các cột quan trọng
        if 'Title' in df.columns:
            stats['unique_titles'] = df['Title'].nunique()
        
        if 'Artist' in df.columns or 'Artists' in df.columns:
            artist_col = 'Artist' if 'Artist' in df.columns else 'Artists'
            stats['unique_artists'] = df[artist_col].nunique()
        
        # Top 5 bài hát nếu có cột Rank
        if 'Rank' in df.columns and 'Title' in df.columns:
            if 'Date' in df.columns and 'dates' in stats:
                # Lấy top 5 của ngày gần nhất
                latest_date = max(df['Date'].unique())
                df_copy = df[df['Date'] == latest_date].copy()
                
                # Convert Rank to numeric để xử lý lỗi
                df_copy['Rank'] = pd.to_numeric(df_copy['Rank'], errors='coerce')
                df_copy = df_copy.dropna(subset=['Rank'])
                
                if len(df_copy) > 0:
                    artist_col = 'Artist' if 'Artist' in df_copy.columns else 'Artists'
                    
                    # --- SỬA ĐỔI SẮP XẾP VỚI PYUCA ---
                    sort_columns = ['Rank']
                    ascending_order = [True]
                    
                    # Đảm bảo Title là string trước khi sắp xếp
                    df_copy['Title'] = df_copy['Title'].astype(str)

                    # Nếu có pyuca (collator != None), tạo sort key cho Title
                    if collator:
                        df_copy['title_sort_key'] = df_copy['Title'].apply(collator.sort_key)
                        sort_columns.append('title_sort_key')
                        ascending_order.append(True)
                    else:
                        # Nếu không có pyuca, sắp xếp theo Title mặc định
                        sort_columns.append('Title')
                        ascending_order.append(True)

                    df_sorted = df_copy.sort_values(by=sort_columns, ascending=ascending_order)
                    
                    # Đảm bảo cột artist tồn tại trong danh sách hiển thị
                    display_cols = ['Rank', 'Title']
                    if artist_col in df_copy.columns:
                        display_cols.append(artist_col)
                    
                    top_songs = df_sorted.head(5)[display_cols]
                    stats['latest_date'] = latest_date
                    stats['top_5'] = top_songs.to_dict('records')
                    # --- KẾT THÚC SỬA ĐỔI ---

        return stats
    
    except Exception as e:
        return {
            'file': os.path.basename(filepath),
            'description': file_description,
            'error': str(e)
        }

def main():
    current_dir = Path(__file__).parent  # Thư mục chứa file script (ví dụ: /project/scripts)
    root_dir = current_dir.parent      # Thư mục gốc của dự án (ví dụ: /project)
    
    # Sử dụng root_dir thống nhất cho cả data và output
    data_dir = root_dir / 'data'                   # Đường dẫn tới thư mục data (/project/data)
    output_file = root_dir / 'CSV_STATISTICS_REPORT.txt' # Lưu report ở thư mục gốc (/project/CSV_STATISTICS_REPORT.txt)
    
    # Khởi tạo Collator nếu thư viện có sẵn
    collator = Collator() if pyuca_available else None
    
    # Định nghĩa các file cần phân tích (dựa trên các file bạn đã tải lên)
    files_to_analyze = {
        'apple_music_top100_kworb_vn.csv': 'Apple Music Vietnam Top 100 (Đã lọc nghệ sĩ VN)',
        'spotify_top100_kworb_vn.csv': 'Spotify Vietnam Top 100 (Đã lọc nghệ sĩ VN)',
        'nct_top50.csv': 'NhacCuaTui Top 50',
        'zingmp3_top100.csv': 'Zing MP3 Top 100',
        'song_list_info.csv': 'Danh sách bài hát gốc & Data Spotify ',
        'new_releases_VPOP_only.csv': 'Các bài hát VPOP mới phát hành',
        'non_hits_bsides.csv': 'Các bài hát Non-hits & B-sides'
    }    
    # Mở file để ghi kết quả
    report_lines = []
    
    def print_and_save(text=""):
        """In ra console và lưu vào list"""
        print(text)
        report_lines.append(text)
    
    print_and_save("=" * 100)
    print_and_save("BÁO CÁO THỐNG KÊ TẤT CẢ FILE CSV")
    print_and_save("=" * 100)
    print_and_save(f"Thời gian tạo báo cáo: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print_and_save(f"Thư mục dữ liệu: {data_dir}")
    print_and_save("=" * 100)
    print_and_save()
    
    all_stats = []
    
    for filename, description in files_to_analyze.items():
        filepath = data_dir / filename
        
        if not filepath.exists():
            print_and_save(f"⚠️  FILE KHÔNG TỒN TẠI: {filename} (Bỏ qua)")
            print_and_save(f"   Đường dẫn kiểm tra: {filepath}")
            print_and_save()
            continue
        
        print_and_save(f"📊 {description}")
        print_and_save(f"   File: {filename}")
        print_and_save("-" * 100)
        
        # Truyền collator vào hàm phân tích
        stats = analyze_csv_file(filepath, description, collator)
        all_stats.append(stats)
        
        if 'error' in stats:
            print_and_save(f"   ❌ LỖI: {stats['error']}")
            print_and_save()
            continue
        
        # In thông tin cơ bản
        print_and_save(f"   📈 Tổng số bản ghi: {format_number(stats['records'])} records")
        print_and_save(f"   📋 Số cột: {stats['columns']} columns")
        print_and_save(f"   💾 Kích thước file: {stats['size']}")
        print_and_save(f"   📝 Các cột: {', '.join(stats['column_names'])}")
        
        # In thống kê theo ngày
        if 'dates' in stats:
            print_and_save(f"   📅 Khoảng thời gian: {stats['date_range']}")
            print_and_save(f"   📆 Số ngày: {len(stats['dates'])} ngày")
            print_and_save(f"   📊 Phân bổ theo ngày:")
            # Chỉ in 5 ngày đầu và 5 ngày cuối nếu có quá nhiều ngày
            if len(stats['dates']) > 10:
                for date in stats['dates'][:5]:
                    count = stats['date_counts'][date]
                    print_and_save(f"      • {date}: {format_number(count)} records")
                print_and_save(f"      ... (và {len(stats['dates']) - 10} ngày khác) ...")
                for date in stats['dates'][-5:]:
                    count = stats['date_counts'][date]
                    print_and_save(f"      • {date}: {format_number(count)} records")
            else:
                 for date in sorted(stats['dates']):
                    count = stats['date_counts'][date]
                    print_and_save(f"      • {date}: {format_number(count)} records")
        elif 'date_error' in stats:
            print_and_save(f"   📅 {stats['date_error']}")


        # In unique values
        if 'unique_titles' in stats:
            print_and_save(f"   🎵 Số bài hát unique: {format_number(stats['unique_titles'])}")
        
        if 'unique_artists' in stats:
            print_and_save(f"   🎤 Số nghệ sĩ unique: {format_number(stats['unique_artists'])}")
        
        # In thống kê theo source (cho master dataset)
        if 'sources' in stats:
            print_and_save(f"   🌐 Phân bổ theo nguồn:")
            for source, count in sorted(stats['sources'].items(), key=lambda x: x[1], reverse=True):
                print_and_save(f"      • {source}: {format_number(count)} records")
        
        # In top 5 bài hát
        if 'top_5' in stats:
            print_and_save(f"   🏆 Top 5 bài hát (ngày {stats['latest_date']}):")
            for song in stats['top_5']:
                artist_key = 'Artist' if 'Artist' in song else 'Artists'
                rank_str = f"{int(song['Rank']):2d}" if pd.notna(song['Rank']) else "??"
                title = str(song['Title'])[:50]
                # Kiểm tra xem artist_key có thực sự tồn tại trong dict 'song' không
                artist = "Unknown"
                if artist_key in song:
                    artist = str(song[artist_key])[:40] if pd.notna(song[artist_key]) else "Unknown"
                
                print_and_save(f"      {rank_str}. {title:<50} - {artist}")
        
        print_and_save()
    
    # Tổng kết
    print_and_save("=" * 30)
    print_and_save("TỔNG KẾT")
    print_and_save("=" * 30)
    
    # Thêm cảnh báo nếu chưa cài pyuca
    if not pyuca_available:
        print_and_save("⚠️  LƯU Ý: Thư viện 'pyuca' không được cài đặt.")
        print_and_save("   Thứ tự sắp xếp Top 5 theo Tiêu đề có thể không")
        print_and_save("   chính xác theo Tiếng Việt. Hãy chạy lệnh:")
        print_and_save("   pip install pyuca")
        print_and_save()

    total_records = sum(s['records'] for s in all_stats if 'records' in s)
    total_files = len([s for s in all_stats if 'records' in s])
    
    print_and_save(f"📁 Tổng số file đã phân tích: {total_files} files")
    print_and_save(f"📊 Tổng số bản ghi (tất cả file): {format_number(total_records)} records")
    
    # Thống kê chi tiết theo loại file
    print_and_save("📊 THỐNG KÊ CHI TIẾT:")

    # Raw chart data files
    chart_files = ['spotify_top100_kworb_vn.csv', 'apple_music_top100_kworb_vn.csv', 
                   'zingmp3_top100.csv', 'nct_top50.csv']
    chart_total = sum(s['records'] for s in all_stats if s['file'] in chart_files and 'records' in s)
    print_and_save(f"   🔴 Dữ liệu charts (4 nguồn): {format_number(chart_total)} records")

    # Master song lists
    master_list_files = ['master_song_list.csv', 'master_song_list_with_spotify_data.csv']
    master_list_total = sum(s['records'] for s in all_stats if s['file'] in master_list_files and 'records' in s)
    print_and_save(f"   🟡 Master Song Lists: {format_number(master_list_total)} records")

    # Master dataset (đã xử lý)
    master_ds = next((s for s in all_stats if s['file'] == 'master_dataset.csv'), None)
    if master_ds and 'records' in master_ds:
        print_and_save(f"   🔵 Master Dataset (Tổng hợp): {format_number(master_ds['records'])} records")
        if 'unique_titles' in master_ds:
             print_and_save(f"      • Trong đó: {format_number(master_ds['unique_titles'])} bài hát unique")
    
    # Other datasets
    new_releases_ds = next((s for s in all_stats if s['file'] == 'new_releases_VPOP_only.csv'), None)
    if new_releases_ds and 'records' in new_releases_ds:
        print_and_save(f"   🟢 New Releases: {format_number(new_releases_ds['records'])} records")

    non_hits_ds = next((s for s in all_stats if s['file'] == 'non_hits_bsides.csv'), None)
    if non_hits_ds and 'records' in non_hits_ds:
        print_and_save(f"   ⚪ Non-Hits / B-sides: {format_number(non_hits_ds['records'])} records")
    
    print_and_save()
    print_and_save("=" * 30)
    print_and_save("✅ BÁO CÁO HOÀN TẤT")
    print_and_save("=" * 30)
    
    # Lưu báo cáo ra file
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print()
        print(f"💾 Đã lưu báo cáo vào file: {output_file}")
        print(f"📄 Tổng số dòng: {len(report_lines)}")
    except Exception as e:
        print(f"❌ LỖI KHI LƯU FILE BÁO CÁO: {e}")

if __name__ == '__main__':
    main()