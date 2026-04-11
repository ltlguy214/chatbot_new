import pandas as pd
import os
import re

# 1. CẤU HÌNH (Đọc thẳng vào file Final và Lưu đè)
INPUT_FILE = 'data/merged_inner_data_final.csv'       
ARTIST_DB_FILE = 'data/artists_vietnam.csv'  
OUTPUT_FILE = 'data/merged_inner_data_final.csv'       

def check_required_files():
    missing_files = []
    for file_path in [INPUT_FILE, ARTIST_DB_FILE]:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
            
    if missing_files:
        error_msg = "❌ LỖI NGHIÊM TRỌNG: Không tìm thấy các file sau:\n"
        error_msg += "\n".join([f"  - {f}" for f in missing_files])
        raise FileNotFoundError(error_msg)

def load_dynamic_artists():
    db = pd.read_csv(ARTIST_DB_FILE)
    db.columns = db.columns.str.strip().str.lower()
    genre_col = next((col for col in db.columns if 'genre' in col), 'genres')
    name_col = next((col for col in db.columns if 'name' in col or 'artist' in col), 'artist_name')
    db[genre_col] = db[genre_col].fillna('').str.lower()
    db[name_col] = db[name_col].fillna('').str.lower()
    
    is_rap = db[genre_col].str.contains('hip hop|rap|drill', na=False)
    is_ballad = db[genre_col].str.contains('bolero|ballad', na=False)
    is_indie = db[genre_col].str.contains('indie|lo-fi', na=False)
    is_pop = db[genre_col].str.contains('pop|vinahouse', na=False)

    rappers = db[is_rap][name_col].unique().tolist()
    ballads = db[is_ballad][name_col].unique().tolist()
    indies = db[is_indie][name_col].unique().tolist()
    pops = db[is_pop][name_col].unique().tolist()
    
    print(f"✅ Đã tải từ DB: {len(rappers)} Rappers, {len(pops)} Pop, {len(indies)} Indies, {len(ballads)} Ballads.")
    return rappers, pops, indies, ballads

def parse_display_artists(title_val, artists_val, featured_val):
    title_str = str(title_val) if pd.notna(title_val) else ""
    artists_str = str(artists_val) if pd.notna(artists_val) else ""
    feat_str = str(featured_val) if pd.notna(featured_val) else ""

    main_artist = ""
    feats = []

    if artists_str:
        parts = [p.strip() for p in artists_str.split(',')]
        main_artist = parts[0]
        if len(parts) > 1: feats.extend(parts[1:])

    feat_match = re.search(r'(?:\(|\[)?(?:feat|ft)\.?\s*([^)\]]+)(?:\)|\])?', title_str, flags=re.IGNORECASE)
    if feat_match:
        extracted_feats = re.split(r'[,&]', feat_match.group(1))
        feats.extend([f.strip() for f in extracted_feats])

    if feat_str:
        feats.extend([f.strip() for f in feat_str.split(',')])

    final_feats = []
    for f in feats:
        clean_f = f.replace(')', '').replace('(', '').strip()
        if clean_f and clean_f.lower() != main_artist.lower() and clean_f not in final_feats:
            final_feats.append(clean_f)

    return main_artist, ", ".join(final_feats)

def run_v61_final_engine():
    print("--- Đang khởi tạo V6.3 (Final Clean Edition) ---")
    check_required_files()
    
    df = pd.read_csv(INPUT_FILE, low_memory=False)
    db_rappers, db_pop, db_indie, db_ballad = load_dynamic_artists()

    print("⚡ Đang biên dịch bộ máy nhận diện nghệ sĩ (Pre-compiling Regex)...")
    
    def build_artist_pattern(artist_list):
        valid_artists = [re.escape(a) for a in artist_list if a]
        if not valid_artists: return re.compile(r'(?!)')
        valid_artists.sort(key=len, reverse=True)
        return re.compile(r'\b(?:' + '|'.join(valid_artists) + r')\b')

    rap_list = db_rappers + ['đen', 'b ray', 'mck', 'pjpo', 'lil wuyn', 'hustlang', 'seachains', '4god']
    pattern_rap = build_artist_pattern(rap_list)
    pattern_pop = build_artist_pattern(db_pop)
    pattern_ballad = build_artist_pattern(db_ballad)
    pattern_indie = build_artist_pattern(db_indie)

    def calculate_v61_scoring(row):
        s = {'Pop': 0, 'Ballad': 0, 'Indie': 0, 'Rap': 0}
        
        title = str(row.get('title', '')).lower()
        artist_info = f"{title} {str(row.get('artists', '')).lower()} {str(row.get('featured_artists', '')).lower()}"
        spotify_g = str(row.get('spotify_genres', '')).lower() 
        
        lyrical_density = 0
        if 'lyrical_density' in row and pd.notna(row['lyrical_density']):
            lyrical_density = row['lyrical_density']
        elif 'lyric_total_words' in row and 'duration_sec' in row:
            dur = row['duration_sec']
            if pd.notna(dur) and dur > 0 and pd.notna(row['lyric_total_words']):
                lyrical_density = row['lyric_total_words'] / dur

        energy = row.get('rms_energy', 0)
        beat = row.get('beat_strength_mean', 0)

        # 1. TẦNG NGHỆ SĨ (Bảo vệ thành quả lọc File DB của bạn)
        is_rap_track = bool(pattern_rap.search(artist_info))
        is_pop_track = bool(pattern_pop.search(artist_info))
        is_ballad_track = bool(pattern_ballad.search(artist_info))
        is_indie_track = bool(pattern_indie.search(artist_info))

        # NẾU nghệ sĩ không có trong file DB của bạn, LÚC ĐÓ MỚI dùng tag của Spotify để cứu cánh
        if not (is_rap_track or is_pop_track or is_ballad_track or is_indie_track):
            if 'hip hop' in spotify_g or 'rap' in spotify_g or 'drill' in spotify_g: is_rap_track = True
            if 'pop' in spotify_g or 'vinahouse' in spotify_g: is_pop_track = True
            if 'ballad' in spotify_g or 'bolero' in spotify_g: is_ballad_track = True
            if 'indie' in spotify_g or 'lo-fi' in spotify_g: is_indie_track = True

        if is_rap_track: s['Rap'] += 3.5 
        if is_pop_track: s['Pop'] += 3.0 
        if is_ballad_track: s['Ballad'] += 2.5
        if is_indie_track: s['Indie'] += 2.5

        # 2. TỐC ĐỘ HÁT (LYRICAL DENSITY) 
        if pd.notna(lyrical_density) and lyrical_density > 0:
            if 2.3 < lyrical_density < 5.0: 
                if beat > 1.6 and not is_rap_track:
                    s['Pop'] += 1.5 
                else:
                    s['Rap'] += 3.0
            elif 1.6 <= lyrical_density <= 2.3: 
                s['Rap'] += 1.5
            elif lyrical_density <= 1.2: 
                s['Rap'] -= 1.5    
                s['Ballad'] += 1.5 

        # 3. AUDIO FEATURES (Chia sẻ điểm công bằng Pop & Rap)
        if pd.notna(energy) and pd.notna(beat):
            if energy < 0.18 and beat < 1.25:
                s['Ballad'] += 2.0
            
            if energy > 0.20 and beat > 1.45:
                if is_rap_track: s['Rap'] += 1.5
                if is_pop_track or (not is_rap_track): s['Pop'] += 2.0 

        # 4. ANTI-CONFLICT LOGIC
        if pd.notna(energy) and pd.notna(beat):
            if energy > 0.22 and beat > 1.55: 
                s['Ballad'] -= 2.0 
                if is_rap_track: s['Rap'] += 1.5
                if is_pop_track or (not is_rap_track): s['Pop'] += 1.5

        if s['Rap'] > 3 and s['Ballad'] > 3:
            if pd.notna(lyrical_density) and lyrical_density > 2.2: 
                s['Ballad'] -= 2.0
            else: 
                s['Rap'] -= 2.0

        # TỔNG HỢP KẾT QUẢ
        mapping = {'Rap': 'Rap/Hip-hop', 'Pop': 'Pop', 'Indie': 'Indie', 'Ballad': 'Ballad'}
        max_score = max(s.values())
        if max_score <= 0: return "Pop"
        
        threshold = max(max_score * 0.60, 1.5)
        ordered_labels = [mapping[k] for k, v in sorted(s.items(), key=lambda x: x[1], reverse=True) if v >= threshold]
        
        return ", ".join(ordered_labels) if ordered_labels else "Pop"

    print("⏳ Đang quét Data và phân tích Genres (Vui lòng chờ)...")
    genres_series = df.apply(calculate_v61_scoring, axis=1)
    
    if 'genres' in df.columns:
        df.drop(columns=['genres'], inplace=True)
        
    if 'spotify_genres' in df.columns:
        insert_loc = df.columns.get_loc('spotify_genres') + 1
        df.insert(insert_loc, 'genres', genres_series)
    else:
        df['genres'] = genres_series 
        
    df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print(f"✅ ĐÃ CHÈN CỘT 'genres' VÀ LƯU ĐÈ THÀNH CÔNG VÀO: {OUTPUT_FILE}\n")
    
    # --- HIỂN THỊ BẢNG 20 BÀI HÁT NGẪU NHIÊN CHẾ ĐỘ CONSOLE ---
    print("20 BÀI HÁT NGẪU NHIÊN (REALITY CHECK)")
    print(f"{'TITLE':<35} | {'ARTISTS':<20} | {'FEAT':<25} | {'GENRES (Main -> Sub)':<22} | {'LD':<5} | {'ENG':<5} | {'BEAT':<5}")
    print("-" * 130)
    
    sample_size = min(20, len(df))
    sample_df = df.sample(n=sample_size) 
    
    for _, row in sample_df.iterrows():
        raw_title = row.get('title', 'Unknown')
        raw_artists = row.get('artists', 'Unknown')
        raw_feat = row.get('featured_artists', '')
        
        main_artist, feat_str = parse_display_artists(raw_title, raw_artists, raw_feat)
        title_print = str(raw_title)[:34]
        artist_print = main_artist[:19]
        feat_print = feat_str[:24] if feat_str else "-"
        genres_print = str(row.get('genres', ''))[:21]
        
        lyric_words = row.get('lyric_total_words', 0)
        dur = row.get('duration_sec', 1)
        ld_val = row.get('lyrical_density', lyric_words / dur if pd.notna(dur) and dur > 0 else 0)
        ld_str = f"{ld_val:.2f}" if pd.notna(ld_val) else "0.00"
        eng = f"{row.get('rms_energy', 0):.3f}" if pd.notna(row.get('rms_energy')) else "0.000"
        beat = f"{row.get('beat_strength_mean', 0):.2f}" if pd.notna(row.get('beat_strength_mean')) else "0.00"
        
        print(f"{title_print:<35} | {artist_print:<20} | {feat_print:<25} | {genres_print:<22} | {ld_str:<5} | {eng:<5} | {beat:<5}")
        
    print("-" * 130)

if __name__ == "__main__":
    run_v61_final_engine()