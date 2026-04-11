# Báo cáo test local – VMusic AI Chatbot (Streamlit)

Ngày: 2026-04-11  
Repo: local workspace `Hit_songs_DA` (đẩy code sang `ltlguy214/chatbot_new.git` khi cần)

## 1) Tóm tắt nhanh
- Intent (Gemini) có thể bị **quota/429** ⇒ hệ thống đã có **fallback heuristic**.
- Đã fix để **tự xoay API key** khi quota (cooldown theo từng key, không khóa toàn cục).
- Đã fix intent cho câu kiểu: **“tìm cho tôi nhạc của Phùng Khánh Linh” → `RECOMMEND_ARTIST`** (kèm `params.artist`).
- UI: ẩn dòng debug “AI Action/Thought” mặc định + render markdown **giữ xuống dòng** trong lịch sử chat.

## 2) Môi trường chạy local
- Python: `.venv312` (Python 3.12)
- Dependency tối thiểu cho phần listener:
  - `sentence-transformers` (embedding)
  - `supabase` (DB + RPC)
  - `google-genai` (Gemini)
  - `rapidfuzz` (fuzzy match artist; có fallback `difflib` nếu thiếu)

Ghi chú: nếu thiếu `sentence-transformers` thì vector/RPC sẽ fail ngay ở bước embed.

## 3) Kiến trúc chức năng chính
- Intent routing: `parse_intent_llm()` trong chatbot/intent.py
  - Ưu tiên Gemini
  - Fallback `_heuristic_intent()` nếu thiếu key / quota / lỗi SDK
  - Post-process: tự fill thiếu `params.artist`, `params.lyric_snippet`, và giảm `CLARIFY` khi pattern rõ ràng
- Action execution: `handle_action()` trong chatbot/action_handler.py
  - Table: `songs`, `track_features`, `artists`
  - RPC: `match_vpop_tracks(query_embedding, match_threshold, match_count)`
- UI: `chatbot/app_chatbot.py`
  - Chat history lưu/đọc từ Supabase
  - Listener flow: gọi intent → handle_action → preview Spotify
  - Producer flow: `ANALYZE_READY` chạy pipeline P0–P4 khi có file

## 4) Kết quả test local (đã chạy)
### 4.1 Intent fallback – case trọng điểm
Đã xác nhận trong local rằng (khi Gemini fallback):
- “tìm cho tôi nhạc của phùng khánh linh” ⇒ `RECOMMEND_ARTIST`, `params.artist = phùng khánh linh`
- “tìm nhạc vui” ⇒ `RECOMMEND_MOOD`, `params.mood = vui`
- “tìm bài có lời "em có nghe"” ⇒ `SEARCH_LYRIC`, `params.lyric_snippet = em có nghe`

### 4.2 Smoke test end-to-end (Supabase + embedding + action handler)
Chạy: `python scripts/test_supabase_match_vpop_tracks.py`

Output thực tế (rút gọn):
- `EMBED_OK dim=384`
- `RPC_OK rows=5`
- `MOOD_OK source=live-supabase-table:track_features->songs tracks=5`
- `LYRIC_OK source=live-supabase-rpc:match_vpop_tracks tracks=5`
- `NAME_OK source=live-supabase-table:songs tracks=1`
- `GENRE_OK ... tracks=5`
- `ARTISTS_TABLE_OK ... count=1664`
- `ARTIST_OK ... tracks=5`
- `AUDIO_NOFILE_OK ... error=Bạn chưa tải file âm thanh lên!`
- `ANALYZE_NOFILE_OK ... error=Bạn chưa tải file âm thanh lên!`
- `CLARIFY_OK ...`
- `OOS_OK ...`

### 4.3 Sanity check webapp
Chạy headless: `python -m streamlit run chatbot/app_chatbot.py --server.headless true --server.port 8502`
- App boot OK, không crash ngay khi start.

## 5) Ma trận mức hoàn thiện ACTION
- `RECOMMEND_ARTIST`: OK (fuzzy artist + query `songs`; load artist list từ `artists` table; có fallback nếu thiếu rapidfuzz)
- `RECOMMEND_MOOD`: OK (ưu tiên `track_features->songs`, fallback vector RPC)
- `RECOMMEND_GENRE`: OK (ilike `songs.spotify_genres`; phụ thuộc dữ liệu genre trong DB)
- `SEARCH_LYRIC`: OK (vector RPC; phụ thuộc embedding)
- `SEARCH_NAME`: OK (table `songs`, ilike title/artist)
- `SEARCH_AUDIO`: Chưa implement pipeline (hiện trả missing-file hoặc TODO)
- `ANALYZE_READY`: OK trong UI (producer pipeline P0–P4 khi có file); trong `handle_action` vẫn là placeholder
- `MISSING_FILE`: OK
- `MUSIC_KNOWLEDGE`: Placeholder (chưa nối LLM/KB)
- `OUT_OF_SCOPE`: OK (fallback message)
- `CLARIFY`: OK (fallback message)

## 6) Vấn đề còn lại / rủi ro
- Gemini quota: không thể “fix quota” bằng code; chỉ có thể **xoay key** + fallback.
- Spotify preview: phụ thuộc token/credential; khi thiếu có thể chỉ hiện link search.
- `SEARCH_AUDIO` chưa có pipeline so sánh audio.
- Một số đoạn UI/markdown phụ thuộc theme Streamlit; đã cố định xuống dòng trong history.

## 7) Khuyến nghị next steps
- Nếu cần ổn định Gemini: thêm nhiều key vào `GEMINI_API_KEYS` để xoay vòng.
- Nếu cần hoàn thiện `SEARCH_AUDIO`: cần định nghĩa rõ embedding/audio-feature + Supabase schema (hoặc local index) rồi nối vào `handle_action`.
- Nếu muốn giảm “debug noise”: giữ `Debug: Hiện AI Action/Thought` mặc định OFF (đã làm).
