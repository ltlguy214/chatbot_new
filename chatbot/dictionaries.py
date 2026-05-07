# config/dictionaries.py

# Tổng hợp từ điển mapping Intent
MOOD_MAP = {
    "buồn": "buồn", "sad": "buồn", "suy": "buồn", "deep": "buồn", "lụy": "buồn", "luy": "buồn",
    "sầu": "buồn", "sau": "buồn", "đau": "buồn", "dau": "buồn",
    "đau lòng": "buồn", "dau long": "buồn", "thất tình": "buồn", "that tinh": "buồn", "cô đơn": "buồn",
    "co don": "buồn", "tâm trạng": "buồn", "tam trang": "buồn", "não nề": "buồn", "nao ne": "buồn",
    "dằn vặt": "buồn", "dan vat": "buồn", "khóc": "buồn", "khoc": "buồn", "thê lương": "buồn", "the luong": "buồn",
    "sầu thảm": "buồn", "nước mắt": "buồn", "tuyệt vọng": "buồn", "bi đát": "buồn",
    "vui": "vui", "happy": "vui", "yêu đời": "vui", "yeu doi": "vui", "tích cực": "vui", "tich cuc": "vui",
    "ngọt ngào": "vui", "ngot ngao": "vui", "lãng mạn": "vui", "lang man": "vui", "hạnh phúc": "vui",
    "hanh phuc": "vui", "động lực": "vui", "dong luc": "vui", "vui vẻ": "vui", "vui ve": "vui",
    "chill": "chill", "ngủ": "chill", "ru ngủ": "chill", "thư giãn": "chill", "thu gian": "chill", 
    "bình yên": "chill", "binh yen": "chill", "nhẹ nhàng": "chill", "nhe nhang": "chill", 
    "healing": "chill", "chữa lành": "chill", "chua lanh": "chill", "an ủi": "chill", "an ui": "chill", 
    "ru ngu": "chill", "thoải mái": "chill", "thoai mai": "chill",
    "quẩy": "quẩy", "quay": "quẩy", "dance": "quẩy", "sôi động": "quẩy", "soi dong": "quẩy",
    "bùng nổ": "quẩy", "bung no": "quẩy", "cháy": "quẩy", "chay": "quẩy", "xập xình": "quẩy", "xap xinh": "quẩy", 
    "sung": "quẩy", "tung nóc nhà": "quẩy", "tung nóc": "quẩy", "cực cháy": "quẩy", "cuc chay": "quẩy",
    "party": "quẩy", "club": "quẩy", "remix": "quẩy", "cháy máy": "quẩy",
    "kịch tính": "kich tinh", "tươi mới": "tuoi moi", "da diết": "da diết", "stress": "stress", "gym": "gym",
    "tình yêu": "tình yêu", "tinh yeu": "tình yêu", "chia tay": "chia tay",
    "tết": "tết", "tet": "tết", "xuân": "xuân", "xuan": "xuân", "năm mới": "xuân", "nam moi": "xuân",
    "gia đình": "gia đình", "gia dinh": "gia đình", "mẹ": "gia đình", "cha": "gia đình", 
    "quê hương": "gia đình", "que huong": "gia đình",
    "kỷ niệm": "kỷ niệm", "ky niem": "kỷ niệm", "hoài niệm": "kỷ niệm", "hoai niem": "kỷ niệm",
    "hoài cổ": "hoai co", "hoai co": "hoai co",
    "thả thính": "thả thính", "tha thinh": "thả thính", "crush": "thả thính",
    "tự hào": "tự hào", "tu hao": "tự hào", "yêu nước": "tự hào", "yeu nuoc": "tự hào"
}

GENRE_KEYWORDS = [
    'rap', 'hiphop', 'hip-hop', 'hip hop', 'underground', 'trap', 'ballad', 'r&b', 'indie', 
    'pop', 'vpop', 'nhạc trẻ', 'nhac tre', 'mainstream', 'hiện đại', 'hien dai', 
    'vinahouse', 'house', 'electronic', 'điện tử', 'dien tu'
]

POPULARITY_KEYWORDS = [
    'hot', 'top', 'bxh', 'viral', 'trending', 'pho bien', 'thinh hanh', 
    'hay nhat', 'nhieu view nhat', 'nghe gi nhieu nhat', 'lam mua lam gio', 
    'duoc yeu thich nhat', 'hit', 'nhieu nguoi nghe', 'quoc dan', 'leo chart',
    'noi tieng nhat', 'bang xep hang', 'dinh dam', 'sieu pham', 
    'thong tri', 'chay nhat', 'nghe nhieu nhat', 'chart'
]
ATTRIBUTE_KEYWORDS = [
    # tempo
    "nhanh", "cham", "fast", "slow",
    "nhip nhanh", "nhip cham", "don dap", "speed up",
    # energy
    "manh", "cang", "nhe",
    "nang luong", "uy luc", "cuc manh",
    # numeric
    "bpm", "tempo"
]
TEENCODE_MAP = {
    "rcm": "goi y",
    "recomend": "goi y",
    "recommend": "goi y",
    # =========================
    # KHÔNG / PHỦ ĐỊNH
    # =========================
    "ko": "khong",
    "k": "khong",
    "kh": "khong",
    "k0": "khong",
    "hok": "khong",
    "hem": "khong",
    "hong": "khong",
    "h0ng": "khong",
    "kg": "khong",
    "khum": "khong",
    "khongg": "khong",

    # =========================
    # ĐƯỢC
    # =========================
    "dc": "duoc",
    "đc": "duoc",
    "dk": "duoc",
    "dk": "duoc",
    "đk": "duoc",
    "duocj": "duoc",

    # =========================
    # MÌNH / TÔI
    # =========================
    "mik": "minh",
    "mjk": "minh",
    "mk": "minh",
    "tui": "toi",
    "toy": "toi",
    "t": "toi",

    # =========================
    # BẠN
    # =========================
    "bn": "ban",
    "b": "ban",
    "fen": "ban",
    "bro": "ban",
    "bruh": "ban",

    # =========================
    # GÌ
    # =========================
    "j": "gi",
    "gii": "gi",
    "gii?": "gi",
    "gii~": "gi",
    "qq": "gi",
    "clg": "cai gi",
    "cakgi": "cai gi",

    # =========================
    # VẬY
    # =========================
    "z": "vay",
    "zay": "vay",
    "v": "vay",
    "ậy": "vay",

    # =========================
    # BIẾT
    # =========================
    "bt": "biet",
    "biett": "biet",

    # =========================
    # RỒI
    # =========================
    "r": "roi",
    "ròi": "roi",
    "roii": "roi",

    # =========================
    # ĐI
    # =========================
    "dj": "di",
    "đii": "di",
    "dii": "di",

    # =========================
    # NHÉ / NHA
    # =========================
    "nhá": "nha",
    "nhee": "nhe",
    "nhaa": "nha",
    "nheee": "nhe",

    # =========================
    # THÌ
    # =========================
    "thoy": "thoi",
    "thui": "thoi",

    # =========================
    # YÊU / THÍCH
    # =========================
    "iu": "yeu",
    "thik": "thich",
    "thix": "thich",

    # =========================
    # NHẠC
    # =========================
    "nhaccc": "nhac",
    "nahc": "nhac",
    "nac": "nhac",
    "nhak": "nhac",
    "nhạcc": "nhac",

    # =========================
    # BÀI
    # =========================
    "baii": "bai",
    "bh": "bai hat",
    "baih": "bai",

    # =========================
    # NHANH / CHẬM
    # =========================
    "nhah": "nhanh",
    "nhan": "nhanh",
    "chm": "cham",

    # =========================
    # CĂNG / MẠNH
    # =========================
    "cag": "cang",
    "cănng": "cang",
    "manhg": "manh",

    # =========================
    # BUỒN / CHILL
    # =========================
    "bun": "buon",
    "bùn": "buon",
    "chil": "chill",
    "chilll": "chill",
    "relx": "relax",

    # =========================
    # SEARCH / PLAY
    # =========================
    "timm": "tim",
    "moe": "mo",
    "batt": "bat",
    "phatt": "phat",
    "nghee": "nghe",

    # =========================
    # ARTIST / SONG
    # =========================
    "cs": "ca si",
    "ns": "nhac si",

    # =========================
    # HOT / VIRAL
    # =========================
    "vr": "viral",
    "trl": "trending",
    "top top": "tiktok",


    # =========================
    # MUSIC DOMAIN
    # =========================
    "edmmm": "edm",
    "hiphop": "hip hop",
    "lofii": "lofi",
    "lofii": "lofi",
    "rappp": "rap",
    "ráp": "rap",
    # =========================
    # EMOTION
    # =========================
    "lụy": "sad",
    "sadg": "sad",
    "happyy": "happy",
    "heall": "healing",

    # =========================
    # PLAYLIST
    # =========================
    "pl": "playlist",
    "listt": "list", 'lít': 'list',

    # =========================
    # COMMON TYPOS
    # =========================
    "cx": "cũng", 'cũm': "cũng",
    "vs": "với", "vớii": "với",
    "ms": "mới", "mớii": "mới",
    "mn": "mọi người",
    "mọing": "mọi người",
    
}

KNOWLEDGE_KEYWORDS = [
        "là gì", "la gi", "thế nào", "the nao", "bao nhiêu", "bao nhieu",
        "tại sao", "tai sao", "ai là", "ai la", "hướng dẫn", "huong dan",
        "cách để", "cach de", "làm sao", "lam sao", "đặc điểm", "dac diem",
        "tiểu sử", "tieu su",
        "cách sử dụng", "cach su dung", "cách tạo", "cach tao", "cách viết", "cach viet",
        "cách cải thiện", "cach cai thien", "cách đăng ký", "cach dang ky", "cach dang ki",
        "khi nào", "khi nao", "từ khi nào", "tu khi nao", "bao giờ", "bao gio",
        "gồm", "gom", "loại nào", "loai nao", "nguồn gốc", "nguon goc",
        "sự nghiệp", "su nghiep", "phong cách", "phong cach", "tầm quan trọng", "tam quan trong",
        "nhánh nhỏ", "nhanh nho",
        "phát triển", "phat trien", "thế hệ", "the he", "hiện nay", "hien nay", 
        "viết", "viet",
    ]