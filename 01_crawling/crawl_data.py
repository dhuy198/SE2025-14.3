
import os
import requests
import hashlib
import time
import random
from duckduckgo_search import DDGS
from PIL import Image
from io import BytesIO

# ================= CẤU HÌNH PRO =================
SAVE_FOLDER = "raw_data"
MIN_FILE_SIZE_KB = 40    # Bỏ qua ảnh dưới 40KB (ảnh icon/thumbnail)
MAX_IMAGES_PER_QUERY = 30 # Số ảnh tối đa cho 1 từ khóa ghép
TIMEOUT = 10             # Giây

# ================= MA TRẬN TỪ KHÓA (ĐỂ TĂNG ĐA DẠNG) =================
# Code sẽ tự ghép: [Subject] + [Actio_pron] + [Context]
SUBJECTS = ["Cảnh sát giao thông", "CSGT", "Chiến sĩ cảnh sát giao thông"]
ACTIONS = [
    "đang làm nhiệm vụ", "xử lý vi phạm", "điều tiết giao thông", 
    "thổi còi", "chào điều lệnh", "kiểm tra giấy tờ", "lái xe mô tô đặc chủng",
    "giúp đỡ người dân", "đội nắng", "dầm mưa"
]
CONTEXTS = ["đường phố", "ban đêm", "trời mưa", "nắng nóng", "cao tốc", "ngã tư"]

# ================= HÀM XỬ LÝ (HASHING & VALIDATION) =================

def get_image_hash(image_bytes):
    """Tạo mã MD5 từ nội dung ảnh để chống trùng lặp"""
    return hashlib.md5(image_bytes).hexdigest()

def is_valid_image(image_bytes):
    """Kiểm tra xem file tải về có phải ảnh lỗi không"""
    try:
        img = Image.open(BytesIO(image_bytes))
        img.verify() # Verify cấu trúc ảnh
        return True
    except:
        return False

def download_images():
    if not os.path.exists(SAVE_FOLDER): os.makedirs(SAVE_FOLDER)
    
    existing_hashes = set()
    # Load hash của các ảnh đã có trong folder (nếu chạy lại lần 2)
    print("-> Đang index dữ liệu cũ để tránh trùng lặp...")
    for f in os.listdir(SAVE_FOLDER):
        path = os.path.join(SAVE_FOLDER, f)
        if os.path.isfile(path):
            with open(path, "rb") as file:
                existing_hashes.add(get_image_hash(file.read()))
    
    print(f"-> Đã index {len(existing_hashes)} ảnh cũ.")

    # Tạo danh sách query từ ma trận
    queries = []
    for sub in SUBJECTS:
        for act in ACTIONS:
            queries.append(f"{sub} {act}") # Ghép cơ bản
            # Lấy ngẫu nhiên context để ghép thêm cho đa dạng
            if random.random() > 0.5:
                ctx = random.choice(CONTEXTS)
                queries.append(f"{sub} {act} {ctx}")

    print(f"=== TỔNG CỘNG {len(queries)} TRUY VẤN KHÁC NHAU ===")
    
    total_downloaded = 0
    ddgs = DDGS()

    for idx, query in enumerate(queries):
        print(f"\n[{idx+1}/{len(queries)}] Tìm: '{query}'")
        
        try:
            # DuckDuckGo Search
            results = ddgs.images(
                query, 
                region="vn-vi", # Ưu tiên kết quả Việt Nam
                safesearch="off", 
                max_results=MAX_IMAGES_PER_QUERY
            )
            
            for res in results:
                img_url = res.get("image")
                if not img_url: continue

                try:
                    # Giả lập Browser để không bị chặn tải
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                    }
                    response = requests.get(img_url, headers=headers, timeout=TIMEOUT)
                    
                    if response.status_code == 200:
                        img_data = response.content
                        
                        # 1. Lọc kích thước file (nhỏ quá là rác)
                        if len(img_data) < MIN_FILE_SIZE_KB * 1024:
                            continue

                        # 2. Check trùng lặp bằng Hash
                        img_hash = get_image_hash(img_data)
                        if img_hash in existing_hashes:
                            # print("   [SKIP] Ảnh trùng.")
                            continue
                        
                        # 3. Check ảnh lỗi
                        if not is_valid_image(img_data):
                            continue

                        # Lưu ảnh
                        filename = f"csgt_{total_downloaded}_{int(time.time())}.jpg"
                        with open(os.path.join(SAVE_FOLDER, filename), "wb") as f:
                            f.write(img_data)
                        
                        existing_hashes.add(img_hash)
                        total_downloaded += 1
                        print(f"   [OK] Đã tải: {filename} ({len(img_data)//1024} KB)")
                        
                except Exception as e:
                    # Lỗi mạng lẻ tẻ thì bỏ qua
                    continue
                
            # Nghỉ ngẫu nhiên 1 chút để server không chặn
            time.sleep(random.uniform(1, 2))

        except Exception as e:
            print(f"Lỗi query '{query}': {e}")
            time.sleep(5) # Nếu bị lỗi API thì nghỉ lâu hơn chút

    print(f"\n=== HOÀN TẤT. TỔNG CỘNG: {total_downloaded} ẢNH MỚI ===")

if __name__ == "__main__":
    download_images()
