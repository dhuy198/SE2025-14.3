import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from tqdm import tqdm

class FaceProcessor:
    def __init__(self, det_size=(640, 640)):
        # Sử dụng GPU P100 của bạn qua CUDA
        self.app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=det_size)

    def align_and_crop(self, img, face, crop_size=512):
        """Xoay ảnh cho thẳng mắt và crop lấy cả mũ Kê-pi."""
        kps = face.kps
        left_eye, right_eye = kps[0], kps[1]
        
        # Tính toán góc xoay dựa trên độ dốc giữa 2 mắt
        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dy, dx))
        eye_center = tuple(np.mean(kps[:2], axis=0).astype(int))
        
        # Ma trận xoay
        M = cv2.getRotationMatrix2D(eye_center, angle, 1.0)
        rotated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_CUBIC)
        
        # Xác định vùng Crop (BBox) trên ảnh đã xoay
        bbox = face.bbox.astype(int)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        
        # "Smart Padding": Lấy cao lên 75% chiều cao mặt để không mất mũ Kê-pi
        y1 = max(0, bbox[1] - int(h * 0.75))
        y2 = min(rotated.shape[0], bbox[3] + int(h * 0.25))
        x1 = max(0, bbox[0] - int(w * 0.4))
        x2 = min(rotated.shape[1], bbox[2] + int(w * 0.4))
        
        cropped = rotated[y1:y2, x1:x2]
        if cropped.size == 0: return None
        return cv2.resize(cropped, (crop_size, crop_size))

    def process_all(self, input_dir, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        img_list = os.listdir(input_dir)
        
        for name in tqdm(img_list, desc="RetinaFace Processing"):
            img = cv2.imread(os.path.join(input_dir, name))
            if img is None: continue
            
            faces = self.app.get(img)
            for i, face in enumerate(faces):
                final_face = self.align_and_crop(img, face)
                if final_face is not None:
                    cv2.imwrite(os.path.join(output_dir, f"csgt_{i}_{name}"), final_face)