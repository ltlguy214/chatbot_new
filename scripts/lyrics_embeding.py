from sentence_transformers import SentenceTransformer
import pandas as pd

# 1. Load dữ liệu lyrics của bạn
df = pd.read_csv('lyrics_rows.csv')

# 2. Sử dụng model để tạo vector (384 chiều)
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# 3. Tạo embedding cho cột lyric
# Lưu ý: Nên làm sạch văn bản (loại bỏ \n) trước khi encode
df['lyric_embedding'] = df['lyric'].apply(lambda x: model.encode(str(x)).tolist())

# 4. Lưu lại để upload lên Supabase
df.to_csv('lyrics_ready_for_supabase.csv', index=False)