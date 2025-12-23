import os
import torch
import re
from PIL import Image
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

class VisionCaptioner:
    def __init__(self, model_id="Qwen/Qwen2-VL-7B-Instruct"):
        self.target_size = (512, 512)
        # Load model tối ưu cho Tesla P100 (16GB)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
        )
        self.processor = AutoProcessor.from_pretrained(model_id)
        
        # Map tag cũ của Hiếu
        self.folder_map = {
            "chan_dung_nam": "a portrait photo of <police_token>, a male police officer",
            "chan_dung_nu": "a portrait photo of <police_token>, a female police officer",
        }

    def get_ai_properties(self, image_path):
        """Trích xuất 5 đặc tính từ AI để bổ trợ cho caption gốc."""
        prompt = (
            "Describe this image briefly with these 5 properties: "
            "uniform color, headwear detail, accessories (whistle/baton), background, and camera angle. "
            "Keywords only, separated by commas."
        )
        
        messages = [{"role": "user", "content": [
            {"type": "image", "image": f"file://{image_path}"},
            {"type": "text", "text": prompt}
        ]}]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        inputs = self.processor(text=[text], images=image_inputs, padding=True, return_tensors="pt").to("cuda")

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=80)
            generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
            tags = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0].strip()
            # Làm sạch chuỗi
            tags = re.sub(r'^(Keywords|Tags|Properties):', '', tags, flags=re.IGNORECASE).strip()
            return tags.lower()

    def process(self, input_folder, output_folder):
        os.makedirs(output_folder, exist_ok=True)
        tasks = []
        
        # Quét thư mục theo logic cũ của Hiếu
        for root, dirs, files in os.walk(input_folder):
            folder_name = os.path.basename(root)
            if folder_name in self.folder_map:
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                        tasks.append({
                            'full_path': os.path.join(root, file),
                            'folder_tags': self.folder_map[folder_name],
                            'filename': file,
                            'folder': folder_name
                        })

        print(f" Đang captioning cho {len(tasks)} ảnh...")
        cnt = 0
        for task in tqdm(tasks):
            try:
                # 1. Lấy tag AI (5 tính chất)
                ai_tags = self.get_ai_properties(task['full_path'])
                
                # 2. Hợp nhất: Tag cũ + AI tags
                final_caption = f"{task['folder_tags']}, {ai_tags}"
                final_caption = final_caption.replace(".", "").strip()
                
                # 3. Xử lý Ảnh theo chuẩn của Hiếu
                new_base_name = f"{task['folder']}_{cnt}"
                output_img_path = os.path.join(output_folder, f"{new_base_name}.jpg")
                
                with Image.open(task['full_path']) as img:
                    img = img.convert("RGB")
                    # Resize chuẩn 512x512 cho LoRA
                    img_resized = img.resize(self.target_size, Image.Resampling.LANCZOS)
                    img_resized.save(output_img_path, "JPEG", quality=95)

                # 4. Lưu file caption .txt
                with open(os.path.join(output_folder, f"{new_base_name}.txt"), "w", encoding="utf-8") as f:
                    f.write(final_caption)
                
                cnt += 1
            except Exception as e:
                print(f" Lỗi tại {task['filename']}: {e}")

        print(f"Xong! Đã lưu {cnt} bộ dataset vào {output_folder}")