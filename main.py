from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from transformers import pipeline
import cv2
import numpy as np

# Khởi tạo ứng dụng FastAPI
app = FastAPI(title="AI Utilities API", description="API tích hợp NLP và Computer Vision")

# --- KHỞI TẠO CÁC MÔ HÌNH AI (Sẽ tốn chút thời gian tải model ở lần chạy đầu tiên) ---
print("Đang tải mô hình Phân tích cảm xúc...")
# sentiment_model = pipeline("sentiment-analysis")

print("Đang tải mô hình Tóm tắt văn bản...")
summarize_model = pipeline("summarization")

# Tải file Haar Cascade của OpenCV để nhận diện khuôn mặt
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
# 2. API Tóm tắt văn bản (Text Summarization)
# ==========================================
@app.post("/api/summarize")
async def summarize_text(req: TextRequest):
    # Giới hạn độ dài bản tóm tắt từ 10 đến 50 từ
    result = summarize_model(req.text, max_length=50, min_length=10, do_sample=False)
    return {
        "original_length": len(req.text),
        "summary": result[0]['summary_text']
    }

# ==========================================
# 3. API Nhận diện khuôn mặt (Face Detection)
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