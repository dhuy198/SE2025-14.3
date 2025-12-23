import os
import torch
import gradio as gr
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from peft import PeftModel
from PIL import Image

# ==============================================================================
# CẤU HÌNH & LOAD MODEL
# ==============================================================================
class GlobalState:
    pipe = None

cfg = {
    "base_model": "runwayml/stable-diffusion-v1-5",
    "lora_path": "../03_training/lora-sd15-police/checkpoint-15",
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

def load_model():
    print("-> Đang khởi tạo Model và LoRA...")
    pipe = StableDiffusionPipeline.from_pretrained(
        cfg["base_model"],
        torch_dtype=torch.float16 if cfg["device"] == "cuda" else torch.float32,
        safety_checker=None
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    
    # Nạp LoRA bằng Peft
    if os.path.exists(cfg["lora_path"]):
        pipe.unet = PeftModel.from_pretrained(pipe.unet, cfg["lora_path"])
        pipe.unet = pipe.unet.merge_and_unload()
        print("✅ Đã nạp và merge LoRA thành công!")
    else:
        print("⚠️ Cảnh báo: Không tìm thấy folder LoRA, chạy model gốc.")
        
    pipe.to(cfg["device"])
    GlobalState.pipe = pipe

# ==============================================================================
# HÀM XỬ LÝ GENERATE
# ==============================================================================
def generate_image(prompt, negative_prompt, steps, cfg_scale, seed):
    if GlobalState.pipe is None:
        return None
    
    generator = torch.Generator(device=cfg["device"]).manual_seed(int(seed))
    
    image = GlobalState.pipe(
        prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=int(steps),
        guidance_scale=cfg_scale,
        generator=generator
    ).images[0]
    
    return image

# ==============================================================================
# GIAO DIỆN GRADIO
# ==============================================================================
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 👮 Vietnamese Traffic Police Generator")
    gr.Markdown("Hệ thống tạo ảnh Cảnh sát Giao thông Việt Nam sử dụng SD 1.5 + LoRA.")

    with gr.Row():
        with gr.Column(scale=1):
            prompt = gr.Textbox(
                label="Prompt", 
                placeholder="Ví dụ: a vietnamese traffic police officer standing on the street, 4k, realistic...",
                lines=3
            )
            neg_prompt = gr.Textbox(
                label="Negative Prompt",
                value="bad anatomy, low quality, blurred, text, watermark, cartoon, 3d, illustration",
                lines=2
            )
            
            with gr.Accordion("Cấu hình nâng cao", open=False):
                steps = gr.Slider(minimum=20, maximum=100, value=40, step=1, label="Steps")
                cfg_scale = gr.Slider(minimum=1, maximum=20, value=8.5, step=0.5, label="Guidance Scale")
                seed = gr.Number(label="Seed", value=42, precision=0)
            
            generate_btn = gr.Button("Sinh ảnh 🚀", variant="primary")

        with gr.Column(scale=1):
            output_img = gr.Image(label="Kết quả", type="pil")

    # Xử lý khi click nút
    generate_btn.click(
        fn=generate_image,
        inputs=[prompt, neg_prompt, steps, cfg_scale, seed],
        outputs=[output_img]
    )

    # Thêm ví dụ mẫu
    gr.Examples(
        examples=[
            ["A traffic police officer control traffic in Hanoi, sunny day, high detail", "low quality", 40, 8.5, 42],
            ["Portrait of a female traffic police officer, smiling, realistic, 8k", "cartoon, blurry", 40, 8.0, 123]
        ],
        inputs=[prompt, neg_prompt, steps, cfg_scale, seed]
    )

if __name__ == "__main__":
    load_model()
    # Chạy trên local network (nếu dùng server thì set share=True để lấy link public)
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)