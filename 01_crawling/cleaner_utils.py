import os
import cv2
import imagehash
from PIL import Image
from tqdm import tqdm

class DataCleaner:
    def __init__(self, blur_threshold=100.0, hash_size=8):
        self.blur_threshold = blur_threshold
        self.hash_size = hash_size
        self.seen_hashes = set()

    def get_blur_score(self, image):
        """Tính toán độ sắc nét bằng phương pháp Laplacian."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def is_duplicate(self, image):
        """Dùng pHash để loại bỏ các ảnh tương đồng (quan trọng cho video)."""
        pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        h = str(imagehash.phash(pil_img, hash_size=self.hash_size))
        if h in self.seen_hashes:
            return True
        self.seen_hashes.add(h)
        return False

    def clean_directory(self, input_dir, output_dir):
        """Quét toàn bộ thư mục, lọc ảnh lỗi/nhòe/trùng."""
        os.makedirs(output_dir, exist_ok=True)
        files = []
        for root, _, filenames in os.walk(input_dir):
            for f in filenames:
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                    files.append(os.path.join(root, f))

        print(f"--- Đang lọc {len(files)} tệp tin ---")
        count = 0
        for path in tqdm(files):
            img = cv2.imread(path)
            if img is None: continue
            
            # 1. Kiểm tra độ nhòe
            if self.get_blur_score(img) < self.blur_threshold:
                continue
            
            # 2. Kiểm tra trùng lặp
            if self.is_duplicate(img):
                continue
                
            cv2.imwrite(os.path.join(output_dir, f"clean_{count:05d}.jpg"), img)
            count += 1
        return count