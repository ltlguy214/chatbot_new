import os

def split_file(file_path, chunk_size_mb=45):
    chunk_size = chunk_size_mb * 1024 * 1024 # 45MB
    file_name = os.path.basename(file_path)
    
    with open(file_path, 'rb') as f:
        chunk_num = 1
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            chunk_name = f"{file_name}.part{chunk_num}"
            with open(chunk_name, 'wb') as chunk_file:
                chunk_file.write(chunk)
            print(f"Đã tạo: {chunk_name}")
            chunk_num += 1

if __name__ == "__main__":
    # Chặt file P1 90MB ra
    split_file('DA/models/best_model_p1_compressed.pkl')