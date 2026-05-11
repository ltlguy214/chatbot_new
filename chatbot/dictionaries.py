# config/dictionaries.py

# Tổng hợp từ điển mapping Intent
MOOD_MAP = {
    # MOOD - buồn
    "buồn": "buồn", "sad": "buồn", "suy": "buồn", "deep": "buồn", "lụy": "buồn", "luy": "buồn",
    "sầu": "buồn", "sau": "buồn", "đau": "buồn", "dau": "buồn",
    "đau lòng": "buồn", "dau long": "buồn", "thất tình": "buồn", "that tinh": "buồn",
    "tâm trạng": "buồn", "tam trang": "buồn", "não nề": "buồn", "nao ne": "buồn",
    "dằn vặt": "buồn", "dan vat": "buồn", "khóc": "buồn", "khoc": "buồn", "thê lương": "buồn", "the luong": "buồn",
    "sầu thảm": "buồn", "nước mắt": "buồn", "tuyệt vọng": "buồn", "bi đát": "buồn",
    
    # MOOD - vui
    "vui": "vui", "happy": "vui",  "tích cực": "vui", "tich cuc": "vui",
    "phấn khởi": "vui", "năng động": "vui", "vui vẻ": "vui", "vui ve": "vui",
    
    # VIBE - Lãng mạn/Tình yêu ra một mood riêng (ví dụ đặt là "lãng mạn" hoặc "tình yêu")
    "ngọt ngào": "tình yêu", "ngot ngao": "tình yêu", 
    "lãng mạn": "tình yêu", "lang man": "tình yêu", 
    "đáng yêu": "tình yêu", "dang yeu": "tình yêu",
    "tình yêu": "tình yêu", "tinh yeu": "tình yêu",
    
    # VIBE -Yêu đời / Tươi mới ra một mood riêng (ví dụ đặt là "tươi mới") để tránh nhầm lẫn với "vui"
    "yêu đời": "tươi mới", "yeu doi": "tươi mới",
    "tuoi moi": "tươi mới", "tươi mới": "tươi mới", "tươi moi": "tươi mới",
    "dong lực": "tươi mới", "dong luc": "tươi mới", "động lực": "tươi mới", "dong luc": "tươi mới",

    # VIBE - bùng nổ / Sôi động
    "gym" : "bùng nổ", "quay" : "bùng nổ", "quẩy" : "bùng nổ",
    "party": "bùng nổ", "sung" : "bùng nổ", "chay": "bùng nổ",
    "cháy": "bùng nổ", "soi dong": "bùng nổ", "sôi động": "bùng nổ",
    "soi động": "bùng nổ", "bung no": "bùng nổ",
    "xập xình": "bùng nổ", "xap xinh": "bùng nổ", 
    "tung nóc nhà": "bùng nổ", "tung nóc": "bùng nổ", "cực cháy": "bùng nổ", "cuc chay": "bùng nổ",
    "club": "bùng nổ", "cháy máy": "bùng nổ", "chay may": "bùng nổ",
    
    # VIBE - bình yên / Chữa lành
    "ru ngủ": "bình yên", "ru ngu": "bình yên", "ngủ":"bình yên", "an ui": "bình yên", "an ủi": "bình yên",
    "stress": "bình yên", "chill": "bình yên", "thu gian": "bình yên", "thư giãn": "bình yên",
    "binh yen": "bình yên", "bình yen": "bình yên", "nhe nhang": "bình yên", "nhẹ nhàng": "bình yên",
    "healing": "bình yên", "chua lanh": "bình yên", "chữa lành": "bình yên", "an ủi": "bình yên",
    "an ui": "bình yên", "thoải mái": "bình yên", "thoai mai": "bình yên",
    "bình yên": "bình yên",

    # VIBE - sâu lắng / Thấu cảm
    "sâu lắng": "sâu lắng",
    "sâu lắg": "sâu lắng", "sau lang": "sâu lắng",
    "thấu cảm": "sâu lắng",
    
    # VIBE - kịch tính / Da diết
    # nhóm da diet
    "da diết": "da diet", "dằn vặt": "da diet",
    "dan vat": "da diet", "não nề": "da diet",
    "nao ne": "da diet",
    
    # nhóm kịch tính
    "kịch tính": "kich tinh", "cao trao": "kich tinh",
    "cao trào": "kich tinh",

    # TOPIC
    "chia tay": "chia tay",
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
    'hot', 'top', 'bxh', 'viral', 'trending', 'pho bien', 'thinh hanh', 'thịnh hành', 
    'hay nhat', 'nhieu view nhat', 'nghe gi nhieu nhat', 'lam mua lam gio', 
    'duoc yeu thich nhat', 'hit', 'nhieu nguoi nghe', 'quoc dan', 'leo chart',
    'noi tieng nhat', 'bang xep hang', 'dinh dam', 'sieu pham', 
    'thong tri', 'chay nhat', 'nghe nhieu nhat', 'chart', 'nổi tiếng nhất', 
    'nổi tiếng', 'noi tieng', 'phổ biến', 'pho bien', 'thịnh hành', 'thinh hanh',
]
ATTRIBUTE_KEYWORDS = [
    # tempo
    "nhanh", "cham", "fast", "slow",
    "nhip nhanh", "nhip cham", "don dap", "speed up",
    # energy
    "manh", "cang", "nhe",
    "uy luc", "cuc manh",
    # numeric
    "bpm", "tempo"
]
TEENCODE_MAP = {
    "nhacquayparty": "nhạc quẩy party",
    "nhacquay": "nhạc quẩy",
    "nhacparty": "nhạc party",
    "quayparty": "quẩy party",
    "nhacbuon": "nhạc buồn",
    "nhacchill": "nhạc chill",
    "nhacvui": "nhạc vui",
    "nhachot": "nhạc hot",
    "bot": "bot", 
    "ad": "admin", 
    "admin": "admin",
    "mìn": "mình",
    "balat": "ballad",
    "balad": "ballad",
    "ballat": "ballad",
    "rcm bài": "gợi ý nhạc",
    "rcm": "gợi ý",
    "bùn": "buồn",
    "balad": "ballad",
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