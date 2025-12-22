"""
Usage:
1. Mở terminal và chuyển đến thư mục chứa file này:
   cd SE2025-14.3/02_dataset
2. Chạy script:
   python3 preprocess_images.py

Script sẽ tự động lấy dữ liệu từ folder 'raw_data' cùng cấp và lưu kết quả vào 'processed_data' 
với kích thước 512x512, giữ nguyên tỉ lệ ảnh (Letterboxing - thêm lề đen thay vì cắt hay kéo dãn).
"""

import os
from PIL import Image, ImageOps
from tqdm import tqdm

def letterbox(img, target_size=(512, 512), color=(0, 0, 0)):
    """
    Resizes image maintaining aspect ratio and pads with 'color' to reach 'target_size'.
    """
    iw, ih = img.size
    tw, th = target_size
    
    # Calculate scale and new dimensions
    scale = min(tw/iw, th/ih)
    nw = int(iw * scale)
    nh = int(ih * scale)
    
    # Resize image
    img = img.resize((nw, nh), Image.Resampling.LANCZOS)
    
    # Create new blank image and paste resized image onto center
    new_img = Image.new('RGB', target_size, color)
    new_img.paste(img, ((tw - nw) // 2, (th - nh) // 2))
    
    return new_img

def preprocess_images(input_dir, output_dir, target_size=(512, 512)):
    """
    Walks through input_dir, applies letterboxing, and saves as .png in output_dir.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    for root, dirs, files in os.walk(input_dir):
        # Create corresponding subdirectories in output_dir
        relative_path = os.path.relpath(root, input_dir)
        target_root = os.path.join(output_dir, relative_path)
        
        if not os.path.exists(target_root):
            os.makedirs(target_root, exist_ok=True)

        print(f"Processing folder: {relative_path if relative_path != '.' else 'root'}")
        
        for file in tqdm(files):
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff')):
                try:
                    input_path = os.path.join(root, file)
                    
                    # Open image
                    with Image.open(input_path) as img:
                        # Convert to RGB (to handle transparency/grayscale/P mode)
                        img = img.convert("RGB")
                        
                        # Apply Letterboxing
                        img = letterbox(img, target_size)
                        
                        # Define output filename (change extension to .png)
                        file_name_no_ext = os.path.splitext(file)[0]
                        output_filename = f"{file_name_no_ext}.png"
                        output_path = os.path.join(target_root, output_filename)
                        
                        # Save image
                        img.save(output_path, "PNG")
                except Exception as e:
                    print(f"Error processing {file}: {e}")

if __name__ == "__main__":
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define paths relative to the script directory
    RAW_DATA_DIR = os.path.join(script_dir, "raw_data")
    PROCESSED_DATA_DIR = os.path.join(script_dir, "processed_data")
    TARGET_SIZE = (512, 512)

    print(f"Input directory: {RAW_DATA_DIR}")
    print(f"Output directory: {PROCESSED_DATA_DIR}")

    if not os.path.exists(RAW_DATA_DIR):
        print(f"Error: {RAW_DATA_DIR} does not exist.")
    else:
        preprocess_images(RAW_DATA_DIR, PROCESSED_DATA_DIR, TARGET_SIZE)
        print("Preprocessing complete!")
