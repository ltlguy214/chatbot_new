import json
from intent import parse_intent_llm # Giả sử file gom chung của bạn tên là intent.py

# ==========================================
# BỘ DATA TEST (EASY & HARD CASES)
# ==========================================
TEST_CASES = [
    # 1. SEARCH_NAME
    {"q": "Tìm bài Nơi này có anh", "file": False, "expect": "SEARCH_NAME", "desc": "[Dễ] Tên bài rõ ràng"},
    {"q": "Tìm bài hát cắt đôi nỗi của Tăng Duy Tân", "file": False, "expect": "SEARCH_NAME", "desc": "[Trung bình] Tên bài và ca sĩ rõ ràng"},
    {"q": "Mở bài cái gì mà có anh á của sếp Tùng", "file": False, "expect": "SEARCH_NAME", "desc": "[Khó] Tên bài mập mờ, dùng biệt danh 'sếp Tùng'"},
    {"q": "tìm bài anh nhà ở đâu thế của đông nhi", "file": False, "expect": "SEARCH_NAME", "desc": "[Khó] Sai nghệ sĩ."},
    
    # 2. SEARCH_LYRIC
    {"q": "Tìm lời bài hát từng là của nhau", "file": False, "expect": "SEARCH_LYRIC", "desc": "[Dễ] Có từ khóa 'lời'"},
    {"q": "Bài gì có lời là 'đưa tay đây nào mãi bên nhau bạn nhé'", "file": False, "expect": "SEARCH_LYRIC", "desc": "[Dễ] Có từ khóa 'câu' / 'lời'"},
    {"q": "Tìm bài có câu nắng ấm xa dần", "file": False, "expect": "SEARCH_LYRIC", "desc": "[Dễ] Có từ khóa 'câu' / 'lời'"},
    {"q": "'mang tiền về cho mẹ đừng mang ưu phiền về cho mẹ' là bài gì?", "file": False, "expect": "SEARCH_LYRIC", "desc": "[Khó] Không có từ khóa báo hiệu, chỉ đưa thẳng câu quote"},

    # 3. SEARCH_AUDIO
    {"q": "Tìm bài giống file này", "file": True, "expect": "SEARCH_AUDIO", "desc": "[Dễ] Có từ 'giống' và có file"},
    {"q": "bài này là bài gì", "file": True, "expect": "SEARCH_AUDIO", "desc": "[Khó] Câu hỏi phổ thông khi đã có file nhưng không có từ 'giống'"},
    {"q": "Check xem cái beat mp3 mình vừa gửi là của bài nào", "file": True, "expect": "SEARCH_AUDIO", "desc": "[Khó] Dùng từ lóng 'beat', 'check xem'"},

    # 4. RECOMMEND_MOOD
    {"q": "Gợi ý nhạc buồn", "file": False, "expect": "RECOMMEND_MOOD", "desc": "[Dễ] Chỉ đích danh mood"},
    {"q": "tìm cho tôi bài nào sâu thúi ruột đi", "file": False, "expect": "RECOMMEND_MOOD", "desc": "[Khó] Mô tả cảm xúc qua từ khóa"},
    {"q": "Cho mình mấy bài nhạc nào chill chill nhẹ nhàng với", "file": False, "expect": "RECOMMEND_MOOD", "desc": "[Trung bình] Mô tả cảm xúc qua từ khóa"},
    {"q": "cho mình vài bài sâu lắng thấm thía", "file": False, "expect": "RECOMMEND_MOOD", "desc": "[Trung bình] Mô tả cảm xúc qua từ khóa"},
    {"q": "Mới chia tay bồ xong, chán quá có bài nào nghe cho khóc luôn không", "file": False, "expect": "RECOMMEND_MOOD", "desc": "[Khó] Ẩn ý tâm trạng lụy/suy, không có chữ 'buồn'"},
    {"q": "nhạc quằn quại", "file": False, "expect": "RECOMMEND_MOOD", "desc": "[Khó] Mô tả thể loại qua từ khóa"},

    # 5. RECOMMEND_ARTIST
    {"q": "gợi ý nhạc son tung", "file": False, "expect": "RECOMMEND_ARTIST", "desc": "[Dễ] Đúng tên nghệ sĩ"},
    {"q": "gợi ý nhạc của phanmạnhquỳnh", "file": False, "expect": "RECOMMEND_ARTIST", "desc": "[Dễ] Đúng tên nghệ sĩ"},
    {"q": "gợi ý nhạc của bíck phươn", "file": False, "expect": "RECOMMEND_ARTIST", "desc": "[Trung bình] Sai chính tả trong tên nghệ sĩ"},
    {"q": "Bật nhạc của Bích Phương", "file": False, "expect": "RECOMMEND_ARTIST", "desc": "[Dễ] Đúng cấu trúc"},
    {"q": "Dạo này Đen Vâu có ra bài gì mới không mở nghe coi", "file": False, "expect": "RECOMMEND_ARTIST", "desc": "[Khó] Trộn lẫn yêu cầu 'bài mới' và 'nghệ sĩ'"},

    # 6. RECOMMEND_GENRE
    {"q": "nhạc lofi", "file": False, "expect": "RECOMMEND_GENRE", "desc": "[Dễ] Đúng tên thể loại"},
    {"q": "nhạc lofi chill", "file": False, "expect": "ADVANCED_SEARCH", "desc": "[Trung bình] Mô tả thể loại qua từ khóa"},
    {"q": "Gợi ý nhạc trẻ", "file": False, "expect": "RECOMMEND_GENRE", "desc": "[Dễ] Đúng tên thể loại"},
    {"q": "Mở nhạc rap", "file": False, "expect": "RECOMMEND_GENRE", "desc": "[Dễ] Đúng tên thể loại"},
    {"q": "tìm nhạc chill indie", "file": False, "expect": "ADVANCED_SEARCH", "desc": "[Trung bình] Kết hợp từ khóa và thể loại"},
    {"q": "Kiếm mấy bài xập xình quẩy trong club hay mở á", "file": False, "expect": "RECOMMEND_GENRE", "desc": "[Khó] Không nói rõ EDM/Dance, chỉ mô tả ngữ cảnh"},

    # 7. ADVANCED_SEARCH
    {"q": "Nhạc rap buồn của Đen Vâu", "file": False, "expect": "ADVANCED_SEARCH", "desc": "[Dễ] Nêu rõ 3 yếu tố"},
    {"q": "Muốn nghe thể loại Indie mà giai điệu chill chill chữa lành của Chillies", "file": False, "expect": "ADVANCED_SEARCH", "desc": "[Khó] Trộn rất nhiều tính từ và thể loại"},

    # 8. RECOMMEND_SEED
    {"q": "Gợi ý bài giống '1000 Ánh Mắt'", "file": False, "expect": "RECOMMEND_SEED", "desc": "[Dễ] Có từ khóa rõ ràng"},
    {"q": "Có bài nào style tựa tựa như Waiting for you của Mono không", "file": False, "expect": "RECOMMEND_SEED", "desc": "[Khó] Kết hợp tên bài, tên ca sĩ và chữ 'style tựa tựa'"},
    {"q": "Tìm bài nào có vibe giống bài 'See Tình'", "file": False, "expect": "RECOMMEND_SEED", "desc": "[Trung bình] Dùng từ khóa 'giống vibe' để yêu cầu tìm theo DNA âm thanh"},
    {"q": "Có bài nào style tựa tựa như Waiting for you của Mono không", "file": False, "expect": "RECOMMEND_SEED", "desc": "[Khó] Kết hợp tên bài, tên ca sĩ và chữ 'style tựa tựa'"},

    # 9. RECOMMEND_ATTRIBUTES
    {"q": "Nhạc nhịp nhanh", "file": False, "expect": "RECOMMEND_ATTRIBUTES", "desc": "[Dễ] Tính chất rõ ràng"},
    {"q": "Tìm cho tôi bài nào nhịp nhanh cỡ 140 bpm", "file": False, "expect": "RECOMMEND_ATTRIBUTES", "desc": "[Trung bình] Kết hợp từ khóa và thông số kỹ thuật"},
    {"q": "Kiếm mấy bài bass căng, tempo dồn dập để tập gym", "file": False, "expect": "RECOMMEND_ATTRIBUTES", "desc": "[Khó] Dùng thuật ngữ chuyên môn (bass, tempo)"},

    # 10. RECOMMEND_POPULARITY
    {"q": "Dạo này có bài hit nào đang trending không? Cho xin top 5", "file": False, "expect": "RECOMMEND_POPULARITY", "desc": "[Dễ] Bắt ngay bằng Rule Engine"},
    {"q": "Top 10 bài hot nhất", "file": False, "expect": "RECOMMEND_POPULARITY", "desc": "[Dễ] Bắt ngay bằng Rule Engine"},
    {"q": "Dạo này giới trẻ trên tiktok đang rần rần bài nào nhất?", "file": False, "expect": "RECOMMEND_POPULARITY", "desc": "[Khó] Không dùng từ khóa 'top/hot/bxh', bắt buộc LLM phải suy luận"},

    # 11. ANALYZE_READY
    {"q": "Phân tích file này xem có hit không", "file": True, "expect": "ANALYZE_READY", "desc": "[Dễ] Bắt ngay bằng Rule Engine"},
    {"q": "Check cái demo này coi có viral tiktok được không", "file": True, "expect": "ANALYZE_READY", "desc": "[Khó] LLM phải hiểu 'viral' tương đương 'hit' và 'demo' là bài hát"},

    # 12. MISSING_FILE
    {"q": "Phân tích bài này xem", "file": False, "expect": "MISSING_FILE", "desc": "[Dễ] Thiếu file"},
    {"q": "Ủa cái giai điệu tèn ten ten ten này là bài gì vậy", "file": False, "expect": "MISSING_FILE", "desc": "[Khó] Tả bằng miệng nhưng bản chất là muốn search audio mà quên gửi file"},

    # 13. CLARIFY
    {"q": "Gợi ý nhạc đi", "file": False, "expect": "CLARIFY", "desc": "[Dễ] Quá chung chung"},
    {"q": "Ê bot", "file": False, "expect": "GREETING", "desc": "[Khó] Chỉ là câu chào"},

    # GREETING
    {"q": "Xin chào", "file": False, "expect": "GREETING", "desc": "[Dễ] Câu chào đơn giản"},
    {"q": "Hey bot", "file": False, "expect": "GREETING", "desc": "[Dễ] Câu chào thân mật"},
    {"q": "hello cưng ơi", "file": False, "expect": "GREETING", "desc": "[Khó] Câu chào lóng, không có từ khóa rõ ràng"},
    
    # 14. MUSIC_KNOWLEDGE
    {"q": "Hợp âm C thứ", "file": False, "expect": "MUSIC_KNOWLEDGE", "desc": "[Dễ] Bắt bằng Rule Engine"},
    {"q": "Hợp âm C7 gồm những nốt nào?", "file": False, "expect": "MUSIC_KNOWLEDGE", "desc": "[Trung bình] Câu hỏi về cấu trúc hợp âm"},
    {"q": "Vòng hòa âm của bài pop thường viết theo bậc mấy?", "file": False, "expect": "MUSIC_KNOWLEDGE", "desc": "[Khó] Câu hỏi nhạc lý chuyên sâu"},

    # 15. OUT_OF_SCOPE
    {"q": "Thời tiết hôm nay", "file": False, "expect": "OUT_OF_SCOPE", "desc": "[Dễ] Lạc đề rõ ràng"},
    {"q": "Chỉ mình cách xài Spotify premium miễn phí", "file": False, "expect": "OUT_OF_SCOPE", "desc": "[Khó] Có chữ Spotify (liên quan nhạc) nhưng là câu hỏi vi phạm/lạc đề"}
]

# ==========================================
# RUN TEST
# ==========================================
def run_tests():
    print("🚀 BẮT ĐẦU CHẠY BỘ TEST 15 ACTION (EASY & HARD)...\n")
    passed = 0
    
    for i, tc in enumerate(TEST_CASES, 1):
        print(f"[{i}/{len(TEST_CASES)}] {tc['desc']}")
        print(f"👤 User : '{tc['q']}' (File: {tc['file']})")
        
        # Gọi hàm parse_intent_llm từ hệ thống của bạn
        result = parse_intent_llm(tc['q'], has_file=tc['file'])
        
        actual_action = result['action']
        expected_action = tc['expect']
        
        if actual_action == expected_action:
            print(f"✅ PASS  : {actual_action}")
            passed += 1
        else:
            print(f"❌ FAIL  : Got '{actual_action}' | Expected '{expected_action}'")
            
        print(f"🧠 Kẻ xử lý: {result.get('thought', '')}")
        
        # In params ra xem LLM có bắt đúng Entity trong câu Khó không
        params = result.get('params', {})
        extracted = {k: v for k, v in params.items() if v}
        if extracted:
            print(f"📦 Params: {extracted}")
            
        print("-" * 60)

    print(f"🎉 TỔNG KẾT: PASS {passed}/{len(TEST_CASES)} CASES.")

if __name__ == "__main__":
    run_tests()