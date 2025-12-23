# File: gender_utils.py
from deepface import DeepFace
import cv2

class GenderClassifier:
    def __init__(self):
        # Model sẽ được tải ở lần chạy đầu tiên
        self.model_name = "gender" 

    def predict(self, img_path):
        try:
            # Phân tích giới tính
            results = DeepFace.analyze(img_path, actions=['gender'], enforce_detection=False)
            # Trả về 'Man' hoặc 'Woman'
            gender = results[0]['dominant_gender']
            return "nam" if gender == "Man" else "nu"
        except Exception as e:
            print(f"Lỗi khi nhận diện giới tính: {e}")
            return "unknown"