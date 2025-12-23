import torch
from diffusers import AutoPipelineForText2Image
import os
from tqdm import tqdm

# 1. Cấu hình
model_id = "stabilityai/sdxl-turbo" # Model thế hệ mới, cực nhanh
input_file = "test_prompts.txt"
output_folder = "output_sdxl_turbo"
os.makedirs(output_folder, exist_ok=True)

# 2. Tải Pipeline (Dùng SDXL để có chất lượng 1024x1024)
pipe = AutoPipelineForText2Image.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    variant="fp16"
)
pipe.to("cuda")

# Tối ưu thêm cho GPU yếu

# 3. Đọc dữ liệu
if os.path.exists(input_file):
    with open(input_file, "r") as f:
        prompts = [l.strip() for l in f.readlines() if l.strip()]
else:
    prompts = ["A cinematic shot of a futuristic robot in a jungle"]

# 4. Sinh ảnh
for i, prompt in enumerate(tqdm(prompts)):
    image = pipe(
        prompt=prompt,
        num_inference_steps=8, # Cực nhanh
        guidance_scale=0.5,
        width=512,
        height=512
    ).images[0]

    image.save(f"{output_folder}/img_{i}.png")

print("Hoàn thành!")
