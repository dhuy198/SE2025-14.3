import torch
import os
from diffusers import StableDiffusionPipeline
from tqdm import tqdm

# ==============================================================================
# 1. CẤU HÌNH (CONFIG)
# ==============================================================================
# Model SD 1.5 chuẩn (HuggingFace)
model_id = "runwayml/stable-diffusion-v1-5"

# Đường dẫn file chứa prompt
input_file = "/datausers3/kttv/tien/ClassificationProjectHimawari/SE/SE2025-14.3/04_inference/test_prompts.txt"
output_folder = "output_images_base_sd15"

# Cấu hình sinh ảnh
# Prompt phụ trợ để ảnh đẹp hơn (SD 1.5 gốc rất cần cái này)
style_suffix = ", highly detailed, 8k resolution, cinematic lighting, photorealistic" 
negative_prompt = "bad anatomy, blurry, low quality, watermark, text, signature, ugly, deformed, extra limbs"

device = "cuda"
batch_size = 1 # Xử lý từng ảnh một để tiết kiệm VRAM

# ==============================================================================
# 2. LOAD MODEL (BASE ONLY)
# ==============================================================================
print(f"-> Đang tải Base Model: {model_id}...")

pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16, # Dùng fp16 cho nhanh và nhẹ
    use_safetensors=True,
    safety_checker=None        # Tắt bộ lọc nội dung để tránh lỗi đen hình
)

# Tối ưu hóa bộ nhớ
pipe.to(device)
# pipe.enable_xformers_memory_efficient_attention() # Bật dòng này nếu cài xformers để render nhanh hơn

# ==============================================================================
# 3. SINH ẢNH
# ==============================================================================
if not os.path.exists(input_file):
    print(f"Lỗi: Không tìm thấy file {input_file}")
    # Nếu không có file, chạy thử 1 prompt mẫu
    lines = ["a futuristic city with flying cars"]
else:
    with open(input_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

os.makedirs(output_folder, exist_ok=True)
print(f"-> Tìm thấy {len(lines)} prompt. Bắt đầu sinh ảnh...")

for i, raw_prompt in enumerate(tqdm(lines)):
    # 1. Ghép prompt
    full_prompt = f"{raw_prompt}{style_suffix}"
    
    # 2. Sinh ảnh (SD 1.5 chuẩn 512x512)
    image = pipe(
        prompt=full_prompt,
        negative_prompt=negative_prompt,
        height=512,
        width=512,
        num_inference_steps=50, # SD 1.5 gốc nên để tầm 50 steps cho chi tiết tốt
        guidance_scale=7.5
    ).images[0]
    
    # 3. Lưu ảnh
    safe_name = raw_prompt.replace(" ", "_").replace("/", "-")[:50]
    filename = f"{output_folder}/{i+1:02d}_{safe_name}.png"
    image.save(filename)

print("\n" + "="*50)
print(f"-> HOÀN TẤT! Ảnh đã lưu tại: {output_folder}")