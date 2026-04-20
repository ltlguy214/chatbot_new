import sys
import os
from pathlib import Path

# Fix đường dẫn cho module chatbot
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from chatbot.env import load_env
from chatbot.supabase import get_supabase_client
from analysis_backend import VPopAnalysisBackend

def run_test():
    load_env()
    supabase = get_supabase_client()
    backend = VPopAnalysisBackend(supabase_client=supabase)
    
    # File test (Đảm bảo file này tồn tại trong folder Test_app)
    audio_path = os.path.join(ROOT_DIR, "chatbot", "Test_app", "Em Có Còn Dùng Số Này Không.mp3")
    
    print(f"\n🔍 Đang tìm kiếm cho: {os.path.basename(audio_path)}")
    result = backend.search_similar_tracks(audio_path, match_count=5)
    
    if result.get("error"):
        print(f"❌ Lỗi: {result['error']}")
    else:
        tracks = result.get("tracks", [])
        print(f"🎉 Kết quả (Top {len(tracks)}):")
        for i, t in enumerate(tracks):
            print(f"#{i+1}: {t['title']} - {t['artists']} (Khớp: {t['similarity']:.4f})")

if __name__ == "__main__":
    run_test()