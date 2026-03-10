from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from transformers import pipeline
import cv2
import numpy as np

# Khởi tạo ứng dụng FastAPI
app = FastAPI(title="AI Utilities API", description="API tích hợp Cảm xúc và Nhận diện khuôn mặt (Bản tối ưu)")

# --- KHỞI TẠO CÁC MÔ HÌNH AI (Tối ưu cho Server Free 512MB) ---
print("Đang tải mô hình Phân tích cảm xúc (Bản nhẹ)...")
# Mô hình này tốn khoảng ~250MB RAM, vừa khít gói miễn phí
sentiment_model = pipeline("sentiment-analysis")

# Tải file Haar Cascade của OpenCV để nhận diện khuôn mặt (Rất nhẹ, chỉ vài MB)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# --- ĐỊNH NGHĨA DỮ LIỆU ĐẦU VÀO ---
class TextRequest(BaseModel):
    text: str

# ==========================================
# 1. API Phân tích cảm xúc (Sentiment Analysis)
# ==========================================
@app.post("/api/sentiment")
async def analyze_sentiment(req: TextRequest):
    # Model sẽ trả về dạng [{'label': 'POSITIVE', 'score': 0.99}]
    result = sentiment_model(req.text)[0]
    return {
        "text": req.text,
        "label": result['label'],
        "confidence_score": round(result['score'], 4)
    }

# ==========================================
# 2. API Nhận diện khuôn mặt (Face Detection)
# ==========================================
@app.post("/api/detect-faces")
async def detect_faces(file: UploadFile = File(...)):
    # Đọc dữ liệu file ảnh người dùng upload
    contents = await file.read()
    
    # Chuyển đổi dữ liệu ảnh sang định dạng OpenCV đọc được
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Chuyển ảnh sang thang độ xám (Gray) để nhận diện tốt hơn
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Nhận diện khuôn mặt
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Lấy tọa độ (x, y) và chiều rộng (w), chiều cao (h) của từng khuôn mặt
    face_coords = [{"x": int(x), "y": int(y), "width": int(w), "height": int(h)} for (x, y, w, h) in faces]
    
    return {
        "filename": file.filename,
        "faces_detected_count": len(face_coords),
        "coordinates": face_coords
    }