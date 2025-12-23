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
