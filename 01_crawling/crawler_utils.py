import os
import cv2
from icrawler.builtin import BingImageCrawler
from tqdm import tqdm

class DataCollector:
    def __init__(self, base_dir="data"):
        self.base_dir = base_dir
        self.raw_web_dir = os.path.join(base_dir, "1_raw_web")
        self.raw_video_dir = os.path.join(base_dir, "1_raw_video")
        
        for d in [self.raw_web_dir, self.raw_video_dir]:
            os.makedirs(d, exist_ok=True)

    def crawl_web(self, keywords, max_images=100):
        """Crawl ảnh từ Bing cho các từ khóa CSGT Nam/Nữ."""
        for kw in keywords:
            save_path = os.path.join(self.raw_web_dir, kw.replace(" ", "_"))
            crawler = BingImageCrawler(storage={'root_dir': save_path})
            crawler.crawl(keyword=kw, max_num=max_images)

    def extract_video_frames(self, video_path, interval_sec=1.0):
        """Trích xuất frame từ video với định dạng hỗ trợ: .mp4, .mkv, .avi."""
        video_name = os.path.basename(video_path).split('.')[0]
        output_subfolder = os.path.join(self.raw_video_dir, video_name)
        os.makedirs(output_subfolder, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        interval_frames = int(fps * interval_sec)
        
        count = 0
        saved = 0
        pbar = tqdm(total=total_frames, desc=f"Processing {video_name}")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            if count % interval_frames == 0:
                frame_path = os.path.join(output_subfolder, f"frame_{count:06d}.jpg")
                cv2.imwrite(frame_path, frame)
                saved += 1
            
            count += 1
            pbar.update(1)
        
        cap.release()
        pbar.close()
        return saved