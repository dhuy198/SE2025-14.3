import os
import json
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm
from pathlib import Path

from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel, StableDiffusionXLPipeline
from diffusers.optimization import get_scheduler
from peft import LoraConfig
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

print("***** Bắt đầu huấn luyện SDXL LoRA *****")
class TrainingConfig:
    pretrained_model_name_or_path = "stabilityai/stable-diffusion-xl-base-1.0"
    train_data_dir = "../01_crawling/cong_an_dieu_huong" 
    output_dir = "sdxl-lora-cong-an"

    resolution = 512  # SDXL chuẩn là 1024
    train_batch_size = 64
    num_train_epochs = 100
    learning_rate = 1e-3
    lr_scheduler = "constant"
    lr_warmup_steps = 0
    
    lora_rank = 16
    lora_alpha = 16

config = TrainingConfig()

# ==============================================================================
# PHẦN 2: CHUẨN BỊ MODEL (SDXL có 2 Text Encoders)
# ==============================================================================
accelerator = Accelerator(
    gradient_accumulation_steps=4, # Tăng cái này nếu VRAM thấp (dưới 24GB)
    mixed_precision="fp16",
    project_config=ProjectConfiguration(project_dir=config.output_dir),
)

# Tải Tokenizers và Text Encoders (SDXL có 2 bộ)
tokenizer_one = CLIPTokenizer.from_pretrained(config.pretrained_model_name_or_path, subfolder="tokenizer")
tokenizer_two = CLIPTokenizer.from_pretrained(config.pretrained_model_name_or_path, subfolder="tokenizer_2")
text_encoder_one = CLIPTextModel.from_pretrained(config.pretrained_model_name_or_path, subfolder="text_encoder")
text_encoder_two = CLIPTextModelWithProjection.from_pretrained(config.pretrained_model_name_or_path, subfolder="text_encoder_2")

vae = AutoencoderKL.from_pretrained(config.pretrained_model_name_or_path, subfolder="vae")
unet = UNet2DConditionModel.from_pretrained(config.pretrained_model_name_or_path, subfolder="unet")
noise_scheduler = DDPMScheduler.from_pretrained(config.pretrained_model_name_or_path, subfolder="scheduler")

# Đóng băng các thành phần không train
vae.requires_grad_(False)
text_encoder_one.requires_grad_(False)
text_encoder_two.requires_grad_(False)
unet.requires_grad_(True)

# Thêm LoRA vào UNet (SDXL UNet lớn hơn nhiều)
lora_config = LoraConfig(
    r=config.lora_rank,
    lora_alpha=config.lora_alpha,
    init_lora_weights="gaussian",
    target_modules=["to_k", "to_q", "to_v", "to_out.0"], # Các lớp Attention
)
unet.add_adapter(lora_config)

# Optimizer
optimizer = torch.optim.AdamW(unet.parameters(), lr=config.learning_rate)

# ==============================================================================
# PHẦN 3: DATASET (Xử lý cho 2 Tokenizers)
# ==============================================================================
class SDXLDataset(Dataset):
    def __init__(self, data_dir, tokenizer_one, tokenizer_two, resolution):
        self.data_dir = data_dir
        self.tokenizer_one = tokenizer_one
        self.tokenizer_two = tokenizer_two
        self.resolution = resolution
        self.data = []
        
        metadata_path = os.path.join(data_dir, "metadata.jsonl")
        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                self.data.append(json.loads(line))

        self.transforms = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.data)

    def tokenize_prompt(self, tokenizer, prompt):
        return tokenizer(
            prompt, padding="max_length", max_length=tokenizer.model_max_length,
            truncation=True, return_tensors="pt"
        ).input_ids[0]

    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(os.path.join(self.data_dir, item["file_name"])).convert("RGB")
        
        return {
            "pixel_values": self.transforms(image),
            "input_ids_one": self.tokenize_prompt(self.tokenizer_one, item["text"]),
            "input_ids_two": self.tokenize_prompt(self.tokenizer_two, item["text"]),
        }

train_dataset = SDXLDataset(config.train_data_dir, tokenizer_one, tokenizer_two, config.resolution)
train_dataloader = DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True)

# Chuẩn bị với Accelerator
unet, optimizer, train_dataloader = accelerator.prepare(unet, optimizer, train_dataloader)
weight_dtype = torch.float16
vae.to(accelerator.device, dtype=torch.float32) # VAE SDXL nên để float32 để tránh lỗi hình đen
text_encoder_one.to(accelerator.device, dtype=weight_dtype)
text_encoder_two.to(accelerator.device, dtype=weight_dtype)

# ==============================================================================
# PHẦN 4: VÒNG LẶP HUẤN LUYỆN
# ==============================================================================
for epoch in range(config.num_train_epochs):
    unet.train()
    for batch in tqdm(train_dataloader, desc=f"Epoch {epoch}"):
        with accelerator.accumulate(unet):
            # 1. Encode ảnh sang Latents
            pixel_values = batch["pixel_values"].to(dtype=vae.dtype)
            latents = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor
            latents = latents.to(dtype=weight_dtype)

            # 2. Tạo Noise và Timesteps
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # 3. Get Text Embeddings (SDXL cần cả 2)
            prompt_embeds_list = []
            for encoder, input_ids in zip([text_encoder_one, text_encoder_two], [batch["input_ids_one"], batch["input_ids_two"]]):
                res = encoder(input_ids.to(accelerator.device), output_hidden_states=True)
                prompt_embeds_list.append(res.hidden_states[-2]) # SDXL dùng penultimate layer
            
            # SDXL đặc thù cần concat embeddings và add time IDs (đơn giản hóa ở đây)
            prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
            
            # Predict noise
            # Lưu ý: SDXL cần thêm 'added_cond_kwargs' cho đúng chuẩn, ở đây ta train LoRA cơ bản cho UNet
            noise_pred = unet(noisy_latents, timesteps, prompt_embeds).sample
            
            loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

# ==============================================================================
# PHẦN 5: LƯU VÀ TEST
# ==============================================================================
# Lưu trọng số LoRA
unet.save_attn_procs(config.output_dir)

# Chạy thử nghiệm
pipe = StableDiffusionXLPipeline.from_pretrained(config.pretrained_model_name_or_path, torch_dtype=torch.float16).to("cuda")
pipe.load_lora_weights(config.output_dir)

image = pipe("một chiến sĩ công an giao thông mặc quân phục, 4k", num_inference_steps=30).images[0]
image.save("sdxl_lora_result.png")