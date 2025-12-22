import os
import time
import random
import hashlib
import requests
from duckduckgo_search import DDGS
from PIL import Image
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
import imagehash

# ================= CONFIG =================
SAVE_FOLDER = "raw_data"
MIN_FILE_SIZE_KB = 40
MIN_WIDTH = 400
MIN_HEIGHT = 400
MAX_IMAGES_PER_QUERY = 30
TIMEOUT = 10
MAX_WORKERS = 6
PHASH_THRESHOLD = 8   # càng nhỏ càng nghiêm

# ================= KEYWORDS =================
SUBJECTS = ["Cảnh sát giao thông", "CSGT", "Chiến sĩ cảnh sát giao thông"]
ACTIONS = [
    "đang làm nhiệm vụ", "xử lý vi phạm", "điều tiết giao thông",
    "thổi còi", "chào điều lệnh", "kiểm tra giấy tờ",
    "lái xe mô tô đặc chủng", "giúp đỡ người dân"
]
CONTEXTS = ["đường phố", "ban đêm", "trời mưa", "nắng nóng", "ngã tư"]

# ================= UTILS =================
def is_valid_image(img_bytes):
    try:
        img = Image.open(BytesIO(img_bytes))
        img.verify()
        return True
    except:
        return False

def preprocess_image(img_bytes):
    img = Image.open(BytesIO(img_bytes)).convert("RGB")
    if img.width < MIN_WIDTH or img.height < MIN_HEIGHT:
        return None, None
    phash = imagehash.phash(img)
    return img, phash

def is_duplicate(phash, existing_hashes):
    for h in existing_hashes:
        if abs(phash - h) <= PHASH_THRESHOLD:
            return True
    return False

# ================= DOWNLOAD =================
def download_one(session, url):
    try:
        r = session.get(url, timeout=TIMEOUT)
        if r.status_code != 200:
            return None
        if len(r.content) < MIN_FILE_SIZE_KB * 1024:
            return None
        if not is_valid_image(r.content):
            return None
        return r.content
    except:
        return None

# ================= MAIN =================
def crawl():
    os.makedirs(SAVE_FOLDER, exist_ok=True)

    existing_hashes = set()
    print("-> Load hash cũ...")
    for f in os.listdir(SAVE_FOLDER):
        try:
            img = Image.open(os.path.join(SAVE_FOLDER, f))
            existing_hashes.add(imagehash.phash(img))
        except:
            continue

    queries = set()
    for s in SUBJECTS:
        for a in ACTIONS:
            queries.add(f"{s} {a}")
            if random.random() > 0.5:
                queries.add(f"{s} {a} {random.choice(CONTEXTS)}")

    print(f"=== {len(queries)} QUERIES ===")

    total = 0
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 Chrome/120 Safari/537.36"
    })

    ddgs = DDGS()

    for qi, query in enumerate(queries):
        print(f"\n[{qi+1}/{len(queries)}] {query}")
        try:
            results = ddgs.images(
                query,
                region="vn-vi",
                safesearch="off",
                max_results=MAX_IMAGES_PER_QUERY
            )

            with ThreadPoolExecutor(MAX_WORKERS) as executor:
                futures = []
                for r in results:
                    if r.get("image"):
                        futures.append(
                            executor.submit(download_one, session, r["image"])
                        )

                for fut in as_completed(futures):
                    img_bytes = fut.result()
                    if not img_bytes:
                        continue

                    img, phash = preprocess_image(img_bytes)
                    if img is None:
                        continue

                    if is_duplicate(phash, existing_hashes):
                        continue

                    fname = f"csgt_{total}_{int(time.time())}.jpg"
                    img.save(os.path.join(SAVE_FOLDER, fname), "JPEG", quality=92)

                    existing_hashes.add(phash)
                    total += 1
                    print(f"   ✔ {fname}")

            time.sleep(random.uniform(1.5, 3))

        except Exception as e:
            print("Query lỗi:", e)
            time.sleep(5)

    print(f"\n=== DONE: {total} ẢNH ===")

if __name__ == "__main__":
    crawl()
