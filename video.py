import sys
import numpy as np
import torch
import cv2
import os
from torchvision import transforms
from PIL import Image
import timm
import warnings


warnings.filterwarnings('ignore')
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def load_model(model_path):
    """Load the trained ViT model"""
    try:
        model = timm.create_model('vit_base_patch16_224', pretrained=False)
        model.head = torch.nn.Sequential(
            torch.nn.Dropout(0.3),
            torch.nn.Linear(in_features=768, out_features=2)
        )
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()
        return model
    except Exception as e:
        print(f"Error in load_model: {str(e)}")
        raise

def process_frame(frame):
    """Convert BGR to RGB and handle color space conversion safely"""
    try:
        if frame is None:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"Error in process_frame: {str(e)}")
        return None

def detect_deepfake(video_path, model, frame_skip=10):
    """Detect if a video is a deepfake"""
    try:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
            
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video file: {video_path}")
            
        frame_count = 0
        predictions = []
        
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if face_cascade.empty():
            raise ValueError("Failed to load face cascade classifier")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            if frame_count % frame_skip != 0:
                continue
                
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray_frame, 1.1, 4)
            
            for (x, y, w, h) in faces:
                face = frame[y:y+h, x:x+w]
                face_resized = cv2.resize(face, (224, 224))
                face_rgb = process_frame(face_resized)
                if face_rgb is None:
                    continue
                    
                face_pil = Image.fromarray(face_rgb)
                face_tensor = transform(face_pil).unsqueeze(0).to(device).float()
                
                with torch.no_grad():
                    outputs = model(face_tensor)
                    outputs = outputs.cpu()
                    probs = torch.softmax(outputs, dim=1)
                    pred = torch.argmax(probs, dim=1).item()
                    predictions.append(pred)
        
        cap.release()
        
        if not predictions:
            return "No faces detected", 0.0
            
        fake_count = predictions.count(1)
        total_predictions = len(predictions)
        confidence = fake_count / total_predictions
        
        if confidence > 0.4:
            return "FAKE", confidence
        else:
            return "REAL", 1 - confidence
            
    except Exception as e:
        print(f"Error in detect_deepfake: {str(e)}")
        raise

def main():
    model_path = 'vit_deepfake.pth'
    video_path = 'VIDEO-2025-01-24-02-15-50.mp4'  
    
    try:
        model = load_model(model_path)
        print("Model loaded successfully")
        
        result, confidence = detect_deepfake(video_path, model)
        print(f"Prediction: {result}")
        print(f"Confidence: {confidence:.2f}")
        
    except Exception as e:
        print(f"Error in main: {str(e)}")

if __name__ == "__main__":
    main()
