import os
import time
import warnings
from pathlib import Path

# Bịt miệng cảnh báo
warnings.filterwarnings("ignore")

import google.generativeai as genai
from supabase import create_client, Client
from dotenv import load_dotenv

# Load biến môi trường
current_dir = Path(__file__).parent
env_path = current_dir / ".env" if (current_dir / ".env").exists() else current_dir.parent / ".env"
load_dotenv(dotenv_path=env_path)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEYS", "").split(',')[0].strip()

if not SUPABASE_URL or not GEMINI_API_KEY:
    print("❌ LỖI: Thiếu URL Supabase hoặc API Key trong file .env")
    exit()

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
genai.configure(api_key=GEMINI_API_KEY)

# 1. TỰ ĐỘNG DÒ MODEL KHẢ DỤNG
AVAILABLE_EMBEDDING_MODEL = None
try:
    for m in genai.list_models():
        if 'embedContent' in m.supported_generation_methods:
            AVAILABLE_EMBEDDING_MODEL = m.name
            break
except Exception as e:
    print(f"❌ Lỗi dò model: {e}")
    exit()

if not AVAILABLE_EMBEDDING_MODEL:
    print("❌ LỖI: API Key không truy cập được model nhúng nào.")
    exit()

def generate_embedding(text, retries=3):
    """Lấy vector và đảm bảo nó CHUẨN 768 chiều"""
    for attempt in range(retries):
        try:
            # Tham số output_dimensionality đôi khi bị API phớt lờ đối với một số model cũ/mới
            result = genai.embed_content(
                model=AVAILABLE_EMBEDDING_MODEL,
                content=text,
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=768 
            )
            
            vector = result['embedding']
            
            # 2. CHỐNG LỖI CỨNG (FALLBACK): Cắt thủ công nếu API vẫn trả về 3072 chiều
            if len(vector) > 768:
                vector = vector[:768]
                
            return vector
        except Exception as e:
            print(f"⚠️ Lỗi Gemini (lần thử {attempt + 1}): {e}")
            time.sleep(2)
    return None

def main():
    print(f"🚀 Bắt đầu nhúng Vector...")
    print(f"✅ Đang sử dụng model: {AVAILABLE_EMBEDDING_MODEL} (Đã ép chuẩn 768 chiều)")

    check_resp = supabase.table("lyrics").select("spotify_track_id").is_("lyric_embedding_gemini", "null").execute()
    pending_ids = [item['spotify_track_id'] for item in check_resp.data]

    if not pending_ids:
        print("✅ Dữ liệu đã nhúng xong toàn bộ!")
        return

    print(f"🔎 Còn lại {len(pending_ids)} bài cần xử lý.")

    for index, track_id in enumerate(pending_ids):
        try:
            s_data = supabase.table("songs").select("title, artists, genres, vibe, main_topic, final_sentiment").eq("spotify_track_id", track_id).single().execute()
            l_data = supabase.table("lyrics").select("lyric").eq("spotify_track_id", track_id).single().execute()

            song = s_data.data
            lyric_row = l_data.data
            if not song or not lyric_row: continue

            enriched_text = (
                f"Tên bài: {song.get('title')} | "
                f"Ca sĩ: {song.get('artists')} | "
                f"Thể loại: {song.get('genres')} | "
                f"Cảm nhận: {song.get('vibe')} | "
                f"Cảm xúc: {song.get('final_sentiment')} | "
                f"Chủ đề: {song.get('main_topic')} | "
                f"Lời bài hát: {lyric_row.get('lyric')}" 
            )

            print(f"[{index+1}/{len(pending_ids)}] Đang nhúng: {song.get('title')}", end=" ")

            vector = generate_embedding(enriched_text)
            if vector:
                print(f"-> OK ({len(vector)} dims)")
                supabase.table("lyrics").update({"lyric_embedding_gemini": vector}).eq("spotify_track_id", track_id).execute()
            else:
                print("-> Thất bại")
            
            time.sleep(0.2)
        except Exception as ex:
            print(f"\n⚠️ Lỗi tại ID {track_id}: {ex}")
            continue

    print("🎉 HOÀN TẤT!")

if __name__ == "__main__":
    main()