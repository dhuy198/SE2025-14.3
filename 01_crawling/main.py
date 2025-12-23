import os
import shutil
from crawler_utils import DataCollector
from cleaner_utils import DataCleaner
from processor_utils import FaceProcessor
from captioner_utils import VisionCaptioner

# ==================== CẤU HÌNH HỆ THỐNG ====================
BASE_DIR = "../02_dataset"
RAW_DIR = os.path.join(BASE_DIR, "1_raw")
CLEANED_DIR = os.path.join(BASE_DIR, "2_cleaned")
PROCESSED_DIR = os.path.join(BASE_DIR, "3_processed")
FINAL_DATASET = "vntrafficpolice_train_v1" # Folder cuối cùng để train

# Từ khóa thu thập
KEYWORDS_NAM = [
    "nam cảnh sát giao thông Việt Nam quân phục",
    "cảnh sát giao thông Việt Nam chân dung nam",
    "CSGT Việt Nam làm nhiệm vụ nam"
]

KEYWORDS_NU = [
    "nữ cảnh sát giao thông Việt Nam xinh đẹp",
    "nữ cảnh sát giao thông Việt Nam quân phục",
    "nữ CSGT Việt Nam điều phối giao thông"
]

VIDEO_PATH = "inputs/video_csgt.mp4" # Để None nếu không dùng video

# ==================== CHƯƠNG TRÌNH CHÍNH ====================

def run_pipeline():
    # 0. Khởi tạo các Module
    processor = FaceProcessor()
    gender_detector = GenderClassifier() # Khởi tạo bộ nhận diện giới tính
    collector = DataCollector(base_dir=RAW_DIR)
    cleaner = DataCleaner(blur_threshold=200.0) # Tesla P100 chạy rất nhanh nên lọc kỹ
    processor = FaceProcessor()
    
    # Khởi tạo Captioner (Tải model Qwen2-VL vào VRAM)
    print("\n[INFO] Đang tải Qwen2-VL vào Tesla P100...")
    captioner = VisionCaptioner()

    # --- BƯỚC 1: THU THẬP (CRAWL & VIDEO) ---
    print("\n>>> BƯỚC 1: Thu thập đa nguồn...")
    # Thu thập ảnh Nam
    collector.crawl_web(KEYWORDS_NAM, max_images=100)
    # Thu thập ảnh Nữ
    collector.crawl_web(KEYWORDS_NU, max_images=100)
    
    # Nếu có video, trích xuất (mặc định cho vào folder nam hoặc tạo folder riêng)
    if VIDEO_PATH and os.path.exists(VIDEO_PATH):
        collector.extract_video_frames(VIDEO_PATH, interval_sec=0.5)

    # --- BƯỚC 2: LỌC NHIỄU & TRÙNG LẶP ---
    print("\n>>> BƯỚC 2: Làm sạch dữ liệu (Blur & Deduplication)...")
    # Lặp qua các thư mục con trong RAW_DIR để lọc
    for sub in os.listdir(RAW_DIR):
        input_sub = os.path.join(RAW_DIR, sub)
        output_sub = os.path.join(CLEANED_DIR, sub)
        if os.path.isdir(input_sub):
            print(f"--- Đang dọn dẹp folder: {sub} ---")
            cleaner.clean_directory(input_sub, output_sub)

    # --- BƯỚC 3: RETINAFACE -> GENDER CHECK -> SORT ---
    print("\n>>> BƯỚC 3: Face Alignment & Gender Sorting...")
    
    # Tạo folder tạm để chứa ảnh đã crop nhưng chưa phân loại nếu cần
    # Hoặc xử lý trực tiếp từ CLEANED_DIR
    for sub in os.listdir(CLEANED_DIR):
        input_sub = os.path.join(CLEANED_DIR, sub)
        if not os.path.isdir(input_sub): continue
        
        print(f"--- Đang xử lý: {sub} ---")
        
        # 1. Đầu tiên, cho FaceProcessor crop và lưu vào một folder tạm
        temp_crop_dir = os.path.join(PROCESSED_DIR, "temp_crops")
        processor.process_all(input_sub, temp_crop_dir)
        
        # 2. Quét qua folder tạm để phân loại giới tính "thực tế"
        for img_name in os.listdir(temp_crop_dir):
            img_path = os.path.join(temp_crop_dir, img_name)
            
            # Dự đoán giới tính dựa trên ảnh mặt đã crop
            gender_label = gender_detector.predict(img_path)
            
            # Xác định folder đích: chan_dung_nam hoặc chan_dung_nu
            final_target = os.path.join(PROCESSED_DIR, f"chan_dung_{gender_label}")
            os.makedirs(final_target, exist_ok=True)
            
            # Di chuyển ảnh vào đúng vị trí
            shutil.move(img_path, os.path.join(final_target, img_name))
            
    # Xóa folder tạm sau khi xong
    if os.path.exists(temp_crop_dir):
        shutil.rmtree(temp_crop_dir)

    # --- BƯỚC 4: AI CAPTIONING (THEO CODE CŨ CỦA HIẾU) ---
    print("\n>>> BƯỚC 4: Qwen2-VL phân tích 5 đặc trưng & Ghép Caption...")
    # Captioner sẽ quét PROCESSED_DIR, tìm folder chan_dung_nam/nu
    # Sau đó resize 512x512 và lưu vào FINAL_DATASET
    captioner.process(PROCESSED_DIR, FINAL_DATASET)

    print(f"\n🚀 TẤT CẢ ĐÃ XONG!")
    print(f"Dữ liệu 'xịn xò' đã sẵn sàng tại: {FINAL_DATASET}")
    print(f"Tổng số ảnh đạt chuẩn: {len(os.listdir(FINAL_DATASET)) // 2}")

if __name__ == "__main__":
    run_pipeline()