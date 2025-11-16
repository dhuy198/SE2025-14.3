import os
import json
from pathlib import Path

# ====================================================================================
# PHẦN CẤU HÌNH - BẠN CHỈ CẦN CHỈNH SỬA 2 DÒNG DƯỚI ĐÂY
# ====================================================================================

# 1. Đường dẫn đến folder gốc chứa các thư mục con (mỗi thư mục là một prompt)
#    Ví dụ: "./cong_an_dieu_huong"
INPUT_ROOT_FOLDER = "../01_crawling/cong_an_dieu_huong"

# 2. Đường dẫn đầy đủ đến file metadata sẽ được tạo ra
OUTPUT_FILE_PATH = "../02_dataset/train_data/metadata.jsonl"

# ====================================================================================
# PHẦN MÃ NGUỒN CHÍNH - BẠN KHÔNG CẦN CHỈNH SỬA PHẦN DƯỚI NÀY
# ====================================================================================

def create_metadata():
    """
    Quét qua thư mục đầu vào, lấy tên thư mục con làm prompt và tạo file metadata.jsonl.
    """
    # Sử dụng pathlib để xử lý đường dẫn một cách an toàn và đa nền tảng
    input_path = Path(INPUT_ROOT_FOLDER)
    output_path = Path(OUTPUT_FILE_PATH)

    # Kiểm tra xem thư mục đầu vào có tồn tại không
    if not input_path.is_dir():
        print(f"❌ Lỗi: Thư mục đầu vào '{input_path}' không tồn tại. Vui lòng kiểm tra lại.")
        return

    # Tự động tạo thư mục cha cho file output nếu nó chưa tồn tại
    # Ví dụ: sẽ tự tạo ra "../02_dataset/train_data/"
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"✅ Đã đảm bảo thư mục output tồn tại: '{output_path.parent}'")
    except Exception as e:
        print(f"❌ Lỗi: Không thể tạo thư mục output. Lỗi: {e}")
        return

    # Các định dạng file ảnh được chấp nhận
    image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    image_count = 0

    print(f"\n🚀 Bắt đầu quét thư mục: '{input_path}'...")

    # Mở file output để ghi
    with open(output_path, "w", encoding="utf-8") as f:
        # os.walk là công cụ tuyệt vời để duyệt qua cây thư mục
        # root: đường dẫn thư mục hiện tại (ví dụ: ./cong_an_dieu_huong/cảnh sát giao thông...)
        # dirs: danh sách các thư mục con bên trong 'root' (không dùng đến)
        # files: danh sách các file bên trong 'root'
        for root, _, files in os.walk(input_path):
            current_dir = Path(root)

            # Bỏ qua chính thư mục gốc ban đầu mà người dùng cung cấp
            if current_dir == input_path:
                continue

            # Tên của thư mục chứa ảnh chính là prompt của chúng ta
            # Ví dụ: "cảnh sát giao thông cầm gậy chỉ huy"
            prompt_text = current_dir.name.replace("_", " ")
            print(f"  📂 Đang xử lý prompt: '{prompt_text}'")

            for filename in files:
                file_path = current_dir / filename
                # Kiểm tra xem file có phải là ảnh không (dựa vào đuôi file)
                if file_path.suffix.lower() in image_extensions:
                    # Tạo đường dẫn tương đối của file ảnh so với thư mục gốc
                    # Đây là định dạng mà script training cần.
                    # Ví dụ: "cảnh sát giao thông cầm gậy chỉ huy/image01.jpg"
                    relative_path = file_path.relative_to(input_path).as_posix()

                    # Tạo một bản ghi (một dòng trong file jsonl)
                    record = {
                        "file_name": relative_path,
                        "text": prompt_text
                    }

                    # Ghi bản ghi dưới dạng một dòng JSON vào file
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    image_count += 1

    if image_count > 0:
        print(f"\n🎉 Hoàn thành! Đã xử lý tổng cộng {image_count} ảnh.")
        print(f"   File metadata đã được lưu tại: '{output_path}'")
    else:
        print("\n⚠️ Cảnh báo: Không tìm thấy ảnh nào để xử lý.")
        print("   Vui lòng kiểm tra lại đường dẫn đầu vào và đảm bảo các thư mục con có chứa file ảnh.")

# Dòng này để đảm bảo hàm create_metadata() chỉ chạy khi file này được thực thi trực tiếp
if __name__ == "__main__":
    create_metadata()