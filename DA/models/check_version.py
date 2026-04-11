'''
    ✅ scikit-learn    == 1.8.0
   ✅ pandas          == 2.3.3
   ✅ numpy           == 2.3.5
   ✅ xgboost         == 3.1.2
   ✅ lightgbm        == 4.6.0
   ✅ optuna          == 4.6.0
   ✅ matplotlib      == 3.10.8
   ✅ seaborn         == 0.13.2
   ✅ joblib          == 1.5.2
'''
import sys
import platform

def check_environment():
    print("="*50)
    print("🚀 KIỂM TRA MÔI TRƯỜNG ĐỒ ÁN V-POP HIT PREDICTION")
    print("="*50)
    
    # 1. Thông tin hệ thống
    print("\n🖥️  THÔNG TIN HỆ THỐNG & PYTHON:")
    print(f"   • Hệ điều hành: {platform.system()} {platform.release()} ({platform.architecture()[0]})")
    print(f"   • Python version: {sys.version.split()[0]}")
    print("-" * 50)

    # 2. Danh sách các thư viện trích xuất từ file analysis_Hit_Prediction.py
    print("\n📦 CÁC THƯ VIỆN CỐT LÕI (REQUIREMENTS):")
    libraries = {
        "scikit-learn": "sklearn",
        "pandas": "pandas",
        "numpy": "numpy",
        "xgboost": "xgboost",
        "lightgbm": "lightgbm",
        "optuna": "optuna",
        "matplotlib": "matplotlib",
        "seaborn": "seaborn",
        "joblib": "joblib"
    }

    missing_libs = []

    for lib_name, module_name in libraries.items():
        try:
            module = __import__(module_name)
            # Một số thư viện lưu version ở thuộc tính khác, nhưng đa số dùng __version__
            version = getattr(module, '__version__', 'N/A')
            print(f"   ✅ {lib_name:<15} == {version}")
        except ImportError:
            print(f"   ❌ {lib_name:<15} == CHƯA CÀI ĐẶT (Missing)")
            missing_libs.append(lib_name)

    print("\n" + "="*50)
    if missing_libs:
        print(f"⚠️ CẢNH BÁO: Bạn cần cài đặt các thư viện bị thiếu bằng lệnh:")
        print(f"   pip install {' '.join(missing_libs)}")
    else:
        print("🎉 TUYỆT VỜI! Môi trường của bạn đã có đủ các thư viện cần thiết.")
        print("   Hãy copy các dòng có dấu ✅ phía trên để làm file requirements.txt nhé!")
    print("="*50)

if __name__ == "__main__":
    check_environment()