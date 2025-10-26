from simple_image_download import simple_image_download

# Khởi tạo trình tải ảnh
response = simple_image_download.simple_image_download()

# Danh sách chủ đề
topics = [
    "người đi bộ vượt đèn đỏ",
    "người đi bộ đi vào lòng đường",
    "leo qua dải phân cách",
    "tụ tập giữa đường",
    "mang vác vật cồng kềnh khi đi bộ",
    "đi bộ trên cao tốc",
    "người đi bộ nói chuyện điện thoại khi qua đường",
    "người đi bộ chạy băng qua đường",
    "người đi bộ băng qua ngã tư sai luật",
    "đi bộ dưới lòng đường vào ban đêm",
    "đi bộ trên cầu vượt xe máy",
    "đi bộ qua đường khi đèn đỏ"
]

# Số lượng ảnh mỗi chủ đề
num_images = 15

# Tải ảnh cho từng chủ đề
for topic in topics:
    print(f"🔽 Downloading {num_images} images for: {topic}")
    response.download(topic, num_images)
