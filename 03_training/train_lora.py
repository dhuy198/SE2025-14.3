import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm.auto import tqdm
from accelerate import Accelerator
from diffusers import DDPMScheduler, UNet2DConditionModel, AutoencoderKL
from transformers import AutoTokenizer, CLIPTextModel # Chỉ dùng CLIPTextModel thường

# Import LoRA
from peft import LoraConfig, get_peft_model

# ==============================================================================
# PHẦN 1: CẤU HÌNH (CONFIG) - ĐÃ CHUYỂN SANG SD 1.5
# ==============================================================================
class TrainingConfig:
    # Model SD v1.5 (Standard, nhẹ, ổn định)
    pretrained_model_name_or_path = "runwayml/stable-diffusion-v1-5"
    
    # Đường dẫn folder dữ liệu của bạn
    train_data_dir = "../02_dataset/data"  
    output_dir = "lora-sd15-police"

    # Resolution: SD 1.5 hoạt động tốt nhất ở 512x512
    resolution = 512 
    train_batch_size = 1
    num_train_epochs = 60
    
    learning_rate = 1e-4
    gradient_accumulation_steps = 4 
    mixed_precision = "fp16" 
    
    # LoRA params
    lora_rank = 32
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
             print(f"LỖI: Không tìm thấy thư mục {root_dir}")
             return

        # Duyệt folder con
        for folder_name in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder_name)
            
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
                                print(f"Cảnh báo: Không đọc được file {txt_path}: {e}")
                                continue
                        else:
                            # Nếu không có file txt, bỏ qua ảnh này (hoặc có thể dùng folder name làm fallback)
                            # print(f"Cảnh báo: Không tìm thấy file caption cho {filename}, bỏ qua.")
                            continue 
                        
                        # Chỉ thêm vào list nếu có prompt
                        if prompt:
                            self.image_paths.append(img_path)
                            self.prompts.append(prompt)

        print(f"-> Tìm thấy {len(self.image_paths)} ảnh hợp lệ (có kèm file caption).")

        self.transforms = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

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

# ==============================================================================
# PHẦN 3: TRAINING LOOP (SD 1.5 LOGIC)
# ==============================================================================
def main():
    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=config.mixed_precision
    )
    
    print("-> Đang tải Model Stable Diffusion v1.5...")
    
    # 1. Load Tokenizers & Models
    tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(config.pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(config.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(config.pretrained_model_name_or_path, subfolder="unet")
    
    # Freeze base models
    vae.requires_grad_(True)
    text_encoder.requires_grad_(True)
    unet.requires_grad_(True)

    # 2. Add LoRA to UNet
    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"], 
        lora_dropout=0.0,
        bias="none"
    )
    unet = get_peft_model(unet, lora_config)
    
    # Xử lý Mixed Precision
    weight_dtype = torch.float32
    if config.mixed_precision == "fp16":
        weight_dtype = torch.float16
        
    unet.to(accelerator.device, dtype=weight_dtype) 
    vae.to(accelerator.device, dtype=torch.float32)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    for name, param in unet.named_parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.float32)
    
    unet.enable_gradient_checkpointing()

    # 3. Optimizer
    lora_layers = filter(lambda p: p.requires_grad, unet.parameters())
    optimizer = torch.optim.AdamW(lora_layers, lr=config.learning_rate)

    # 4. Data
    dataset = LocalImageDataset(config.train_data_dir, tokenizer, size=config.resolution)
    train_dataloader = DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=0)

    noise_scheduler = DDPMScheduler.from_pretrained(config.pretrained_model_name_or_path, subfolder="scheduler")

    # 5. Prepare Accelerator
    unet, optimizer, train_dataloader = accelerator.prepare(unet, optimizer, train_dataloader)
    
    # 6. Train Loop
    print(f"***** Bắt đầu huấn luyện SD v1.5 (Resolution: {config.resolution}) *****")
    
    for epoch in range(config.num_train_epochs):
        unet.train()
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{config.num_train_epochs}")
        
        for step, batch in enumerate(progress_bar):
            with accelerator.accumulate(unet):
                pixel_values = batch["pixel_values"].to(dtype=torch.float32)
                
                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor
                    latents = latents.to(dtype=weight_dtype)

                with torch.no_grad():
                    encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                model_pred = unet(
                    noisy_latents, 
                    timesteps, 
                    encoder_hidden_states=encoder_hidden_states
                ).sample

                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), 1.0)
                
                optimizer.step()
                optimizer.zero_grad()

            progress_bar.set_postfix(loss=loss.item())

        # ======================================================================
        # [MỚI] LƯU CHECKPOINT MỖI 10 EPOCH
        # ======================================================================
        if (epoch + 1) % 5 == 0:
            # Tạo đường dẫn ví dụ: lora-sd15-police/checkpoint-10
            checkpoint_dir = os.path.join(config.output_dir, f"checkpoint-{epoch+1}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            print(f"\n-> Đang lưu checkpoint tại Epoch {epoch+1} vào: {checkpoint_dir}")
            
            # Lưu model
            unet_to_save = accelerator.unwrap_model(unet)
            unet_to_save.save_pretrained(checkpoint_dir)
            
            # Quay lại chế độ train để tiếp tục epoch sau
            unet.train() 
        # ======================================================================

    # 7. Final Save
    print("-> Đang lưu Model cuối cùng...")
    os.makedirs(config.output_dir, exist_ok=True)
    
    unet_to_save = accelerator.unwrap_model(unet)
    unet_to_save.save_pretrained(config.output_dir)
    
    print(f"-> Xong! Saved final model to {config.output_dir}")

if __name__ == "__main__":
    main()