from simple_image_download import simple_image_download

# Khởi tạo trình tải ảnh
response = simple_image_download.simple_image_download()

# Danh sách từ khóa chi tiết để có ảnh chính xác hơn
keywords = [
    "công an xử lý vi phạm giao thông",
    "cảnh sát giao thông lập biên bản",
    "CSGT dừng xe kiểm tra",
    "cảnh sát giao thông đo nồng độ cồn",
    "CSGT bắt lỗi xe máy",
    "cảnh sát giao thông xử phạt ô tô",
    "chốt cảnh sát giao thông",
    "CSGT kiểm tra giấy tờ xe",
    "công an bắt người không đội mũ bảo hiểm",
    "CSGT xử lý xe quá khổ quá tải"
]

# Số lượng ảnh mỗi từ khóa
num_images = 20

# Tải ảnh
for key in keywords:
    print(f"🔽 Đang tải {num_images} ảnh cho từ khóa: {key}")
    # extensions={'.jpg', '.png'} giúp lọc file rác nếu thư viện hỗ trợ bản mới
    response.download(key, num_images)

print("✅ Hoàn tất tải ảnh!")