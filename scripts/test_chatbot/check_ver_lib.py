import importlib.metadata
import sys

# Danh sách các thư viện cốt lõi của dự án
libs_to_check = [
    "librosa", "pandas", "numpy", "tqdm", "mutagen", "underthesea", 
    "vaderSentiment", "transformers", "torch", "bertopic", 
    "pyvi", "sentence-transformers"
]

print(f"{'Thư viện':<25} | {'Phiên bản':<15} | {'Trạng thái'}")
print("-" * 55)

for lib in libs_to_check:
    try:
        version = importlib.metadata.version(lib)
        print(f"{lib:<25} | {version:<15} | OK")
    except importlib.metadata.PackageNotFoundError:
        print(f"{lib:<25} | {'Chưa cài đặt':<15} | ❌")

print("-" * 55)
print(f"Python version: {sys.version.split()[0]}")