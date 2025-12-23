import os
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from peft import PeftModel  # <--- BẮT BUỘC PHẢI CÓ DÒNG NÀY
from tqdm.auto import tqdm

# ==============================================================================
# CẤU HÌNH
# ==============================================================================
class InferenceConfig:
    # Model gốc SD 1.5
    base_model_path = "runwayml/stable-diffusion-v1-5"
    
    # Đường dẫn đến folder chứa file trong ảnh (adapter_model.safetensors)
    lora_folder_path = "../03_training/lora-sd15-police/checkpoint-15"  
    
    prompt_file = "test_prompts.txt"
    output_dir = "inference_results"
    
    # Tham số
    num_inference_steps = 40
    guidance_scale = 8.5
    seed = 42
    height = 512
    width = 512

cfg = InferenceConfig()

def main():
    # 1. Setup
    if not os.path.exists(cfg.lora_folder_path):
        print(f"❌ LỖI: Không tìm thấy folder {cfg.lora_folder_path}")
        return

    # Kiểm tra file safetensors (như trong ảnh của bạn)
    if not os.path.exists(os.path.join(cfg.lora_folder_path, "adapter_model.safetensors")):
        print("⚠️ CẢNH BÁO: Không thấy file 'adapter_model.safetensors'. Kiểm tra lại folder.")
    
    os.makedirs(cfg.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 2. Load Model Gốc (SD 1.5)
    print("-> 1. Đang tải Base Model...")
    pipe = StableDiffusionPipeline.from_pretrained(
        cfg.base_model_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)

    # 3. LOAD LORA (QUAN TRỌNG NHẤT)
    # Vì file của bạn là adapter_model.safetensors sinh ra bởi Peft
    # ta phải dùng PeftModel để load, tránh lỗi "No LoRA keys".
    print(f"-> 2. Đang nạp LoRA từ: {cfg.lora_folder_path}")
    
    try:
        # Load trọng số vào UNet
        pipe.unet = PeftModel.from_pretrained(pipe.unet, cfg.lora_folder_path)
        
        # Merge trọng số LoRA vào UNet gốc để chạy nhanh hơn
        pipe.unet = pipe.unet.merge_and_unload()
        
        print("✅ Load LoRA thành công (Đã merge vào UNet)!")
    except Exception as e:
        print(f"❌ LỖI Load LoRA: {e}")
        return

    # 4. Sinh ảnh
    if not os.path.exists(cfg.prompt_file):
        print(f"Tạo file mẫu {cfg.prompt_file}...")
        with open(cfg.prompt_file, "w") as f:
            f.write("traffic police control traffic, sunny day, 4k\n")
            f.write("checking alcohol level, close up, realistic\n")

    with open(cfg.prompt_file, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]

    print(f"-> 3. Bắt đầu sinh {len(prompts)} ảnh...")
    
    negative_prompt = "bad anatomy, low quality, blurred, text, watermark, cartoon, 3d, illustration"
    generator = torch.Generator(device=device).manual_seed(cfg.seed)

    for i, prompt in enumerate(tqdm(prompts)):
        image = pipe(
            prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=cfg.num_inference_steps,
            guidance_scale=cfg.guidance_scale,
            height=cfg.height,
            width=cfg.width,
            generator=generator
        ).images[0]

        safe_name = prompt.replace(" ", "_")[:40]
        image.save(os.path.join(cfg.output_dir, f"{i}.png"))

    print(f"\n✅ Xong! Ảnh lưu tại: {cfg.output_dir}")

if __name__ == "__main__":
    main()