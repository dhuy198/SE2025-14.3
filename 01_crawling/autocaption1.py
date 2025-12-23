import os
from PIL import Image, ImageDraw, ImageFont

def add_caption_to_image(image_path, text, output_folder):
    try:
        img = Image.open(image_path).convert("RGBA")
        
        # Tạo lớp để vẽ
        txt_layer = Image.new("RGBA", img.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(txt_layer)
        
        # Tùy chỉnh Font (dùng font mặc định nếu không có font ttf)
        # Để đẹp hơn, bạn nên tải file font .ttf và dẫn link vào đây
        try:
            font = ImageFont.truetype("arial.ttf", 40) 
        except:
            font = ImageFont.load_default()

        # Tính toán vị trí vẽ (Góc dưới bên trái)
        # textbbox trả về (left, top, right, bottom)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        x = 20
        y = img.height - text_height - 20
        
        # Vẽ khung nền đen mờ cho chữ dễ đọc
        draw.rectangle((x-5, y-5, x + text_width + 5, y + text_height + 5), fill=(0, 0, 0, 128))
        
        # Vẽ chữ màu trắng
        draw.text((x, y), text, font=font, fill=(255, 255, 255, 255))
        
        # Gộp lớp chữ vào ảnh gốc
        out = Image.alpha_composite(img, txt_layer)
        
        # Lưu ảnh (convert lại RGB để lưu jpg)
        output_path = os.path.join(output_folder, os.path.basename(image_path))
        out.convert("RGB").save(output_path)
        print(f"✅ Đã xử lý: {os.path.basename(image_path)}")
        
    except Exception as e:
        print(f"❌ Lỗi: {e}")

# --- Cấu hình ---
input_folder = "dataset_csgt"   # Thư mục ảnh gốc
output_folder = "dataset_labeled" # Thư mục lưu ảnh đã có chữ
caption_text = "VI PHAM GIAO THONG" # Nội dung muốn ghi

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Chạy vòng lặp
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        add_caption_to_image(os.path.join(input_folder, filename), caption_text, output_folder)