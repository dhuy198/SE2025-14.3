import os
import torch
from tqdm import tqdm
import re
from PIL import Image  # Thêm thư viện xử lý ảnh
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# ================= CẤU HÌNH =================
INPUT_FOLDER = "raw_data"       
OUTPUT_FOLDER = "train_new_v1"  
TRIGGER_WORD = "vntrafficpolice" 
TARGET_SIZE = (512, 512) # Kích thước mục tiêu

FOLDER_MAP = {
    # "bat_giu": "vntrafficpolicebatgiu, arresting, handcuffs, confrontation",
    "chan_dung_nam": "vntrafficpolicechandungnam",
    "chan_dung_nu": "vntrafficpolicechandungnu",
    # "dieu_phoi": "vntrafficpolicedieuphoi, directing traffic, street, standing, hand gesture",
    # "doan_xe": "vntrafficpolicedoanxe, motorcade, riding motorcycle, police bike, formation"
}

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ================= LOAD MODEL =================
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
)
processor = AutoProcessor.from_pretrained(MODEL_ID)

def get_qwen_tags(image_path):
    prompt = (
        "List 10-15 descriptive keywords for this image, separated by commas. "
        "Focus on: uniform details, accessories "
        "background, weather, and camera angle. No full sentences."
    )
    
    messages = [{"role": "user", "content": [
        {"type": "image", "image": f"file://{image_path}"},
        {"type": "text", "text": prompt}
    ]}]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt").to("cuda")

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=100)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        tags = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0].strip()
        tags = re.sub(r'^(Keywords|Tags|Mô tả):', '', tags, flags=re.IGNORECASE).strip()
        return tags

# ================= XỬ LÝ CHÍNH =================
def main():
    cnt = 0
    tasks = []
    for root, dirs, files in os.walk(INPUT_FOLDER):
        folder_name = os.path.basename(root)
        if folder_name in FOLDER_MAP:
            for file in files:
                if os.path.splitext(file)[1].lower() in IMAGE_EXTENSIONS:
                    tasks.append({
                        'full_path': os.path.join(root, file),
                        'folder_tags': FOLDER_MAP[folder_name],
                        'filename': file,
                        'folder': folder_name
                    })

    print(f"📦 Đang xử lý {len(tasks)} ảnh (Resize & Auto-caption)...")

    for task in tqdm(tasks):
        try:
            # 1. Lấy tag từ AI
            # ai_tags = get_qwen_tags(task['full_path'])
            final_caption = f"{task['folder_tags']}"
            final_caption = final_caption.replace(".", "").strip()
            
            # 2. Xử lý ảnh: Mở, Resize và Lưu
            new_base_name = f"{task['folder']}_{cnt}"
            # Lưu định dạng .jpg để đồng bộ và nhẹ (tùy chọn)
            output_img_path = os.path.join(OUTPUT_FOLDER, f"{new_base_name}.jpg")
            
            with Image.open(task['full_path']) as img:
                # Chuyển sang RGB (đề phòng ảnh PNG có kênh Alpha gây lỗi khi lưu JPG)
                img = img.convert("RGB")
                
                # Resize (Sử dụng Resampling.LANCZOS cho chất lượng tốt nhất)
                # Lưu ý: Code này sẽ nén ảnh về 512x512 (có thể gây méo nếu ảnh gốc không vuông)
                img_resized = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
                
                # Lưu ảnh
                img_resized.save(output_img_path, "JPEG", quality=95)

            # 3. Lưu file caption
            with open(os.path.join(OUTPUT_FOLDER, f"{new_base_name}.txt"), "w", encoding="utf-8") as f:
                f.write(final_caption)
            
            cnt += 1
                
        except Exception as e:
            print(f"❌ Lỗi: {task['filename']} - {e}")

    print(f"✅ Hoàn tất! Đã lưu {cnt} bộ dữ liệu vào '{OUTPUT_FOLDER}'")

if __name__ == "__main__":
    main()