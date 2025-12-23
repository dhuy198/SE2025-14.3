import os
import json
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm
from pathlib import Path

# Thư viện từ Hugging Face (vẫn cần dùng cho các thành phần của Stable Diffusion)
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel, StableDiffusionPipeline
from diffusers.optimization import get_scheduler
from peft import LoraConfig
from transformers import CLIPTextModel, CLIPTokenizer
from torchvision import transforms

class TrainingConfig:
    # Model SD v1.5 (Standard, nhẹ, ổn định)
    pretrained_model_name_or_path = "runwayml/stable-diffusion-v1-5"
    
    # Đường dẫn folder dữ liệu của bạn
    train_data_dir = "/datausers3/kttv/tien/ClassificationProjectHimawari/SE/SE2025-14.3/02_dataset/data_3"  
    output_dir = "lora-sd15-police"

    # Resolution: SD 1.5 hoạt động tốt nhất ở 512x512
    resolution = 1024
    train_batch_size = 1
    num_train_epochs = 100
    
    learning_rate = 5e-5
    gradient_accumulation_steps = 4 
    mixed_precision = "bfp16" 
    
    # LoRA params
    lora_rank = 64
    lora_alpha = 32
    seed = 42

config = TrainingConfig()

# ==============================================================================
# PHẦN 2: XỬ LÝ DỮ LIỆU (ĐƠN GIẢN HÓA CHO SD 1.5)
# ==============================================================================
class LocalImageDataset(Dataset):
    def __init__(self, root_dir, tokenizer, size=512):
        self.root_dir = root_dir
        self.tokenizer = tokenizer
        self.size = size
        self.image_paths = []
        self.prompts = []
        valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        print(f"-> Đang quét dữ liệu từ: {root_dir}")
        
        if not os.path.exists(root_dir):
             return

        # Duyệt folder con
        for folder_name in os.listdir(root_dir):
            
            if os.path.isdir(folder_path):
                # Duyệt từng file trong folder con
                for filename in os.listdir(folder_path):
                    file_ext = os.path.splitext(filename)[1].lower()
                    
                    # Nếu là file ảnh
                    if file_ext in valid_ext:
                        img_path = os.path.join(folder_path, filename)
                        
                        # Tạo đường dẫn file txt tương ứng
                        # Ví dụ: anh1.jpg -> anh1.txt
                        base_name = os.path.splitext(filename)[0]
                        txt_path = os.path.join(folder_path, base_name + ".txt")
                        
                        prompt = ""
                        # Kiểm tra file txt có tồn tại không
                        if os.path.exists(txt_path):
                            try:
                                with open(txt_path, 'r', encoding='utf-8') as f:
                                    prompt = f.read().strip()
                            except Exception as e:
                                continue
                        else:
                            # Nếu không có file txt, bỏ qua ảnh này (hoặc có thể dùng folder name làm fallback)
                            # print(f"Cảnh báo: Không tìm thấy file caption cho {filename}, bỏ qua.")
                            continue 
                        
                        # Chỉ thêm vào list nếu có prompt
                        if prompt:
                            self.image_paths.append(img_path)
                            self.prompts.append(prompt)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        prompt = self.prompts[idx]

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Lỗi ảnh {img_path}: {e}")
            # Nếu lỗi load ảnh, thử load ảnh kế tiếp
            return self.__getitem__((idx + 1) % len(self))

        pixel_values = self.transforms(image)

        # Tokenize
        text_inputs = self.tokenizer(
            prompt, 
            padding="max_length", 
            max_length=self.tokenizer.model_max_length, 
            truncation=True, 
            return_tensors="pt"
        )

        return {
            "pixel_values": pixel_values,
            "input_ids": text_inputs.input_ids[0],
        }