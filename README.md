# 👮‍♂️ Vietnamese Traffic Police (CSGT) Portrait Generation

Dự án nghiên cứu fine-tune mô hình **Stable Diffusion** bằng kỹ thuật **LoRA** để tạo ra hình ảnh chân dung Cảnh sát Giao thông (CSGT) Việt Nam với độ chân thực cao, đúng quy chuẩn quân phục.
---

## 📂 Project Structure

.
├── 01_crawling/
│   ├── crawl_web.py          # Script crawl ảnh từ Google/Pinterest
│   ├── generate_synthetic.py  # Script dùng SDXL/Flux tạo ảnh mẫu
│   └── raw/                  # Thư mục chứa ảnh thô mới tải về
├── 02_dataset/
│   ├── quality_filter.py     # Lọc ảnh mờ, nhiễu, điểm thẩm mỹ thấp
│   ├── face_alignment.py     # Crop và căn chỉnh khuôn mặt (MediaPipe)
│   ├── captioning.py         # Gán nhãn tự động bằng Qwen2-VL
│   └── final_dataset/        # Dữ liệu sạch sẵn sàng để train
├── 03_training/
│   ├── train_lora.py         # Script huấn luyện chính
│   ├── config.yaml           # File cấu hình tham số (LR, Rank, Epoch)
│   └── checkpoints/          # Nơi lưu các file .safetensors
├── 04_inference/
│   ├── generate.py           # Script test model sau khi train
│   ├── prompt_library.md     # Bộ sưu tập các prompt hiệu quả
│   └── samples/              # Ảnh kết quả demo
└── README.md                 # Hướng dẫn sử dụng dự án

---

## 🚀 Pipeline Chi Tiết

### 1. Thu thập dữ liệu (01_crawling)
Kết hợp đa dạng nguồn dữ liệu để đảm bảo tính tổng quát:
- **Web Crawling:** Sử dụng Selenium/Playwright thu thập ảnh từ các trang báo, mạng xã hội.
- **Synthetic Data:** Lấy ảnh sinh ra từ các model AI khác (SDXL, Flux) để làm phong phú tư thế.
- **Manual Collection:** Tuyển chọn ảnh chất lượng cao để làm dữ liệu chuẩn (Anchor images).

### 2. Xây dựng Dataset & Tiền xử lý (02_dataset)
Đây là bước cốt lõi để đạt được độ chân thực:
- **Lọc trùng:** Loại bỏ ảnh tương đồng bằng thuật toán Perceptual Hash (pHash).
- **Lọc chất lượng (Quality Filter):**
    - Chấm điểm thẩm mỹ (Aesthetic Predictor) để giữ lại ảnh đẹp.
    - Dùng OpenCV lọc ảnh bị mờ, nhiễu.
- **Xử lý chân dung:** Sử dụng MediaPipe/RetinaFace để xác định vùng mặt và Crop về tỷ lệ 1:1.
- **Gán nhãn tự động (Auto-Captioning):** Sử dụng VLM (Qwen2-VL hoặc LLaVA) để mô tả chi tiết trang phục và bối cảnh.

### 3. Huấn luyện mô hình (03_training)
Cấu hình tối ưu cho GPU Tesla P100 16GB:
- **Kỹ thuật:** LoRA (Low-Rank Adaptation).
- **Base Model:** Stable Diffusion v1.5 / SDXL.
- **Thông số:** Rank 32, Alpha 32, Learning Rate 1e-4.

### 4. Kiểm thử & Suy luận (04_inference)
- Kiểm tra độ chính xác của các chi tiết: Sao trên mũ, màu áo vàng đặc trưng, phù hiệu CSGT.
- Hướng dẫn viết Prompt tối ưu để kích hoạt LoRA.

---

## 🛠 Hướng dẫn nhanh

1. Cài đặt thư viện:
   pip install -r requirements.txt

2. Chạy quy trình lọc ảnh:
   python 02_dataset/clean_data.py --input ./01_raw --output ./02_clean

3. Huấn luyện:
   accelerate launch 03_training/train_lora.py --config config.yaml

---

## ⚖️ Quy định sử dụng (Disclaimer)

Dự án này phục vụ mục đích nghiên cứu học thuật. Không sử dụng mô hình để tạo ra nội dung giả mạo, bôi nhọ hoặc vi phạm pháp luật. Người dùng tự chịu trách nhiệm về nội dung sinh ra.

