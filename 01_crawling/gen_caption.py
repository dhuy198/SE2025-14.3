import os
from PIL import Image, ImageOps
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import traceback

# ================= CONFIG =================
INPUT_FOLDER = "raw_data"
OUTPUT_FOLDER = "train_new"
TRIGGER_WORD = "vntrafficpolice"
TARGET_SIZE = 512
JPEG_QUALITY = 95
MAX_WORKERS = 8

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}

FOLDER_MAP = {
    "bat_giu": "arresting, handcuffs, confrontation",
    "chan_dung_nam": "male officer, portrait, looking at camera",
    "chan_dung_nu": "female officer, portrait, looking at camera",
    "dieu_phoi": "directing traffic, street, hand gesture",
    "doan_xe": "motorcade, riding police motorcycle, formation"
}

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ================= IMAGE UTILS =================
def resize_with_padding(img, target=512):
    img = ImageOps.exif_transpose(img)
    img.thumbnail((target, target), Image.Resampling.LANCZOS)

    new_img = Image.new("RGB", (target, target), (127, 127, 127))
    offset = ((target - img.width) // 2, (target - img.height) // 2)
    new_img.paste(img, offset)
    return new_img

# ================= PROCESS ONE IMAGE =================
def process_task(task):
    try:
        base_name = task["base_name"]
        caption = task["caption"]
        src_path = task["path"]

        with Image.open(src_path) as img:
            img = img.convert("RGB")
            img = resize_with_padding(img, TARGET_SIZE)
            img.save(
                os.path.join(OUTPUT_FOLDER, f"{base_name}.jpg"),
                "JPEG",
                quality=JPEG_QUALITY,
                subsampling=0
            )

        with open(os.path.join(OUTPUT_FOLDER, f"{base_name}.txt"),
                  "w", encoding="utf-8") as f:
            f.write(caption)

        return True

    except Exception:
        traceback.print_exc()
        return False

# ================= MAIN =================
def main():
    tasks = []
    idx = 0

    for root, _, files in os.walk(INPUT_FOLDER):
        folder = os.path.basename(root)
        if folder not in FOLDER_MAP:
            continue

        for f in files:
            if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS:
                caption = f"{TRIGGER_WORD}, {FOLDER_MAP[folder]}"
                tasks.append({
                    "path": os.path.join(root, f),
                    "caption": caption,
                    "base_name": f"{folder}_{idx}"
                })
                idx += 1

    print(f"📦 Tổng ảnh: {len(tasks)}")

    success = 0
    with ThreadPoolExecutor(MAX_WORKERS) as executor:
        for ok in tqdm(executor.map(process_task, tasks), total=len(tasks)):
            success += int(ok)

    print(f"✅ Hoàn tất: {success}/{len(tasks)} ảnh")

if __name__ == "__main__":
    main()
