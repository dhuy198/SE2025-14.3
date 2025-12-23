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
# from datasets import load_dataset # <--- THAY ĐỔI: Không cần dùng nữa

# <--- THAY ĐỔI: Thêm thư viện Dataset của PyTorch
from torch.utils.data import Dataset, DataLoader

print("***** Bắt đầu huấn luyện *****")
# ==============================================================================
# PHẦN 1: CẤU HÌNH CÁC THAM SỐ HUẤN LUYỆN
# ==============================================================================
class TrainingConfig:
    # --- Model và đường dẫn ---
    pretrained_model_name_or_path = "runwayml/stable-diffusion-v1-5"
    train_data_dir = "../01_crawling/cong_an_dieu_huong" 
    output_dir = "lora-cong-an-giao-thong"

    # --- Tham số huấn luyện ---
    resolution = 512
    train_batch_size = 1
    num_train_epochs = 100
    learning_rate = 1e-4
    lr_scheduler = "constant"
    lr_warmup_steps = 0
    seed = 42

    # --- Tham số LoRA ---
    lora_rank = 8
    lora_alpha = 8

config = TrainingConfig()

# ==============================================================================
# PHẦN 0: TẠO DỮ LIỆU GIẢ LẬP
# ==============================================================================
def setup_dummy_data(data_dir):
    print("Tạo dữ liệu giả lập để kiểm tra...")
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    
    Image.new("RGB", (512, 512), "blue").save(data_path / "cong_an_1.jpg")
    Image.new("RGB", (512, 512), "red").save(data_path / "cong_an_2.jpg")

    metadata = [
        {"file_name": "cong_an_1.jpg", "text": "một chiến sĩ công an giao thông đang chỉ đường"},
        {"file_name": "cong_an_2.jpg", "text": "cận cảnh một công an giao thông mặc quân phục"},
    ]
    with open(data_path / "metadata.jsonl", "w", encoding="utf-8") as f:
        for item in metadata:
            f.write(json.dumps(item) + "\n")
    print("Đã tạo xong dữ liệu giả lập.")

if not os.path.exists(config.train_data_dir):
    setup_dummy_data(config.train_data_dir)

# ==============================================================================
# PHẦN 2: CHUẨN BỊ DỮ LIỆU, MODEL VÀ CÁC THÀNH PHẦN
# ==============================================================================

# Khởi tạo Accelerator
accelerator = Accelerator(
    gradient_accumulation_steps=1,
    mixed_precision="fp16",
    project_config=ProjectConfiguration(project_dir=config.output_dir),
)

# Tải các thành phần của Stable Diffusion
tokenizer = CLIPTokenizer.from_pretrained(config.pretrained_model_name_or_path, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(config.pretrained_model_name_or_path, subfolder="text_encoder")
vae = AutoencoderKL.from_pretrained(config.pretrained_model_name_or_path, subfolder="vae")
unet = UNet2DConditionModel.from_pretrained(config.pretrained_model_name_or_path, subfolder="unet")
noise_scheduler = DDPMScheduler.from_pretrained(config.pretrained_model_name_or_path, subfolder="scheduler")

# Đóng băng các tham số
vae.requires_grad_(False)
text_encoder.requires_grad_(False)
unet.requires_grad_(True) # Chỉ huấn luyện UNet

# Thêm LoRA adapter vào UNet
lora_config = LoraConfig(
    r=config.lora_rank,
    lora_alpha=config.lora_alpha,
    init_lora_weights="gaussian",
    target_modules=["to_k", "to_q", "to_v", "to_out.0"],
)
unet.add_adapter(lora_config)

# Optimizer
try:
    import bitsandbytes as bnb
    optimizer_cls = bnb.optim.AdamW8bit
    print("Sử dụng AdamW8bit optimizer.")
except ImportError:
    optimizer_cls = torch.optim.AdamW
    print("bitsandbytes không được cài đặt, sử dụng AdamW optimizer mặc định.")

optimizer = optimizer_cls(
    unet.parameters(),
    lr=config.learning_rate,
)

# Định nghĩa các bước xử lý ảnh
image_transforms = transforms.Compose(
    [
        transforms.Resize(config.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(config.resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)

# <--- THAY ĐỔI: Định nghĩa lớp Dataset tùy chỉnh
class CustomImageCaptionDataset(Dataset):
    def __init__(self, data_dir, tokenizer, transforms):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.transforms = transforms
        self.data = []
        
        # Đọc file metadata.jsonl
        metadata_path = os.path.join(data_dir, "metadata.jsonl")
        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Lấy đường dẫn ảnh và caption
        image_path = os.path.join(self.data_dir, item["file_name"])
        text = item["text"]
        
        # Mở và xử lý ảnh
        image = Image.open(image_path).convert("RGB")
        pixel_values = self.transforms(image)
        
        # Tokenize text
        input_ids = self.tokenizer(
            text,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids[0] # Lấy phần tử đầu tiên để loại bỏ batch dimension
        
        return {"pixel_values": pixel_values, "input_ids": input_ids}

# <--- THAY ĐỔI: Khởi tạo Dataset tùy chỉnh
train_dataset = CustomImageCaptionDataset(
    data_dir=config.train_data_dir,
    tokenizer=tokenizer,
    transforms=image_transforms
)

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    input_ids = torch.stack([example["input_ids"] for example in examples])
    return {"pixel_values": pixel_values, "input_ids": input_ids}

train_dataloader = DataLoader( # <--- THAY ĐỔI: Dùng DataLoader của PyTorch
    train_dataset, shuffle=True, collate_fn=collate_fn, batch_size=config.train_batch_size
)

# LR Scheduler
lr_scheduler = get_scheduler(
    config.lr_scheduler,
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=len(train_dataloader) * config.num_train_epochs,
)

# Chuẩn bị mọi thứ với Accelerator
unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
    unet, optimizer, train_dataloader, lr_scheduler
)

weight_dtype = torch.float16
text_encoder.to(accelerator.device, dtype=weight_dtype)
vae.to(accelerator.device, dtype=weight_dtype)

# ==============================================================================
# PHẦN 3: VÒNG LẶP HUẤN LUYỆN (KHÔNG THAY ĐỔI)
# ==============================================================================
# (Phần này giữ nguyên hoàn toàn)
print("***** Bắt đầu huấn luyện *****")
print(f"  Số mẫu = {len(train_dataset)}")
print(f"  Số epochs = {config.num_train_epochs}")
print(f"  Batch size = {config.train_batch_size}")
print(f"  Tổng số bước tối ưu hóa = {len(train_dataloader) * config.num_train_epochs}")

progress_bar = tqdm(range(len(train_dataloader) * config.num_train_epochs), disable=not accelerator.is_local_main_process)
progress_bar.set_description("Steps")

for epoch in range(config.num_train_epochs):
    unet.train()
    for step, batch in enumerate(train_dataloader):
        with accelerator.accumulate(unet):
            pixel_values = batch["pixel_values"].to(dtype=weight_dtype)
            latents = vae.encode(pixel_values).latent_dist.sample()
            latents = latents * vae.config.scaling_factor
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            encoder_hidden_states = text_encoder(batch["input_ids"])[0]
            noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
            loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(unet.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        if accelerator.sync_gradients:
            progress_bar.update(1)
        logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
        progress_bar.set_postfix(**logs)

    if accelerator.is_main_process:
        print(f"Epoch {epoch+1}/{config.num_train_epochs} - Loss: {loss.detach().item():.4f}")

# ==============================================================================
# PHẦN 4: LƯU LOA VÀ SỬ DỤNG (KHÔNG THAY ĐỔI)
# ==============================================================================
# (Phần này giữ nguyên hoàn toàn)
unet = accelerator.unwrap_model(unet)
# Tạo pipeline đầy đủ để lưu
pipeline = StableDiffusionPipeline.from_pretrained(
    config.pretrained_model_name_or_path,
    unet=unet,
    text_encoder=text_encoder,
    vae=vae,
    tokenizer=tokenizer,
    torch_dtype=weight_dtype, # Thêm dtype để nhất quán
)
# Lưu trọng số LoRA và toàn bộ pipeline
pipeline.save_pretrained(config.output_dir)
print(f"Đã lưu LoRA và pipeline vào thư mục: {config.output_dir}")

# Ví dụ sử dụng LoRA để tạo ảnh
print("\n***** Thử nghiệm tạo ảnh với LoRA *****")
pipe = StableDiffusionPipeline.from_pretrained(config.pretrained_model_name_or_path, torch_dtype=torch.float16).to("cuda")
pipe.load_lora_weights(config.output_dir)

prompt = "một chiến sĩ công an giao thông đang chỉ đường, ảnh chân thực, chất lượng cao"
image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
image.save("ket_qua_voi_lora.png")
print("Đã lưu ảnh kết quả vào file: ket_qua_voi_lora.png")