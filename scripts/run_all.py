import subprocess
import sys
import os


PYTHON_EXECUTABLE = sys.executable 
SCRIPTS_DIR = "scrapers"

SCRAPE_SCRIPTS = [
    "apple_music_top100_kworb.py",
    "spotify_top100_kworb.py",
    "nct_top50.py",
    "zingmp3_top100.py",
]

def run_script(script_name: str, use_scripts_dir: bool = True) -> bool:
    """
    Hàm helper để chạy một script Python và kiểm tra lỗi.
    """
    if use_scripts_dir:
        script_path = os.path.join(SCRIPTS_DIR, script_name)
    else:
        script_path = script_name # Sử dụng đường dẫn được cung cấp trực tiếp

    print("\n" + "="*50)
    print(f"🚀 Đang chạy: {script_path}")
    print("="*50)
    
    # Kiểm tra xem file có tồn tại không
    if not os.path.exists(script_path):
        print(f"❌ LỖI: Không tìm thấy file script '{script_path}'.")
        # Sửa lại thông báo lỗi cho chuẩn:
        print("Vui lòng kiểm tra lại đường dẫn và tên file.")
        return False # Báo thất bại

    # Xây dựng lệnh chạy
    # Thêm "-X", "utf8" để buộc Python chạy ở Chế độ UTF-8
    command = [PYTHON_EXECUTABLE, "-X", "utf8", script_path]
    
    # Chạy lệnh
    try:
        result = subprocess.run(
            command, 
            check=True,
            text=True, 
            encoding='utf-8',
            errors='replace', # Bỏ qua lỗi ký tự lạ
            capture_output=True
        )
        
        # In output (stdout) nếu chạy thành công
        print("--- Output ---")
        print(result.stdout)
        print("--------------")
        print(f"✅ Hoàn thành: {script_name}")
        return True # Báo thành công
        
    except subprocess.CalledProcessError as e:
        # ... (phần xử lý lỗi giữ nguyên) ...
        print(f"❌ LỖI NGHIÊM TRỌNG khi chạy '{script_name}':")
        print("--- Standard Output (stdout) ---")
        print(e.stdout)
        print("--- Standard Error (stderr) ---")
        print(e.stderr)
        print("-------------------------------")
        return False # Báo thất bại
    except Exception as e:
        # ... (phần xử lý lỗi giữ nguyên) ...
        print(f"❌ LỖI HỆ THỐNG khi cố chạy '{script_name}': {e}")
        return False # Báo thất bại

def main():
    print("Bắt đầu quy trình chạy tự động...")
    
    # 1. Chạy các file cào dữ liệu
    for script_file in SCRAPE_SCRIPTS:
        # Mặc định use_scripts_dir=True, sẽ tìm trong "scrapers"
        success = run_script(script_file) 
        
        if not success:
            print("\n" + "="*50)
            print(f"🛑 Dừng quy trình do có lỗi ở script: {script_file}")
            print("Vui lòng sửa lỗi và chạy lại 'run_all.py'.")
            print("="*50)
            return # Thoát hàm main

    print("\n" + "="*50)
    print("🎉 Tất cả script cào dữ liệu đã hoàn thành.")

if __name__ == "__main__":
    main()