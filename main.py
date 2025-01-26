import streamlit as st
import os
import torch
import cv2
import librosa
import numpy as np
from PIL import Image
from torchvision import transforms, models
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor
import timm
import plotly.graph_objects as go
from datetime import datetime

# Check if MPS is available
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Metal) device")
else:
    device = torch.device("cpu")
    print("MPS device not found, using CPU")
def load_video_model(model_path):
    model = timm.create_model('vit_base_patch16_224', pretrained=False)
    model.head = torch.nn.Sequential(
        torch.nn.Dropout(0.3),
        torch.nn.Linear(in_features=768, out_features=2)
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()
    return model

def load_image_model(model_path):
    class ResNet50DeepfakeModel(torch.nn.Module):
        def __init__(self, num_classes=2):
            super(ResNet50DeepfakeModel, self).__init__()
            self.resnet50 = models.resnet50(pretrained=False)
            self.resnet50.fc = torch.nn.Linear(self.resnet50.fc.in_features, num_classes)

        def forward(self, x):
            return self.resnet50(x)

    model = ResNet50DeepfakeModel(num_classes=2)
    checkpoint = torch.load(model_path, map_location=device)
    if any(key.startswith("module.") for key in checkpoint.keys()):
        checkpoint = {key.replace("module.", ""): value for key, value in checkpoint.items()}
    model.load_state_dict(checkpoint, strict=False)
    model = model.to(device)
    model.eval()
    return model

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    if image.mode != 'RGB':
        image = image.convert('RGB')

    image_tensor = transform(image).unsqueeze(0)
    return image_tensor

def detect_deepfake_video(video_path, model):
    try:
        cap = cv2.VideoCapture(video_path)
        frame_count, predictions, confidences = 0, [], []
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            if frame_count % 10 != 0:
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray_frame, 1.1, 4)

            frame_predictions, frame_confidences = [], []

            for (x, y, w, h) in faces:
                face = frame_rgb[y:y + h, x:x + w]
                face_pil = Image.fromarray(face)
                face_tensor = preprocess_image(face_pil).to(device)

                with torch.no_grad():
                    outputs = model(face_tensor)
                    probs = torch.softmax(outputs, dim=1)
                    pred = torch.argmax(probs, dim=1).item()
                    conf = probs[0][pred].item()

                    frame_predictions.append(pred)
                    frame_confidences.append(conf)

            if frame_predictions:
                predictions.extend(frame_predictions)
                confidences.extend(frame_confidences)

        cap.release()

        if not predictions:
            return "No faces detected", 0.0

        fake_count = predictions.count(1)
        total_predictions = len(predictions)
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        if total_predictions > 0:
            fake_ratio = fake_count / total_predictions
            result = "FAKE" if fake_ratio > 0.5 else "REAL"
            return result, avg_confidence
        else:
            return "Unable to process video", 0.0

    except Exception as e:
        st.error(f"Error processing video: {str(e)}")
        return "Error", 0.0

def detect_deepfake_image(image, model):
    try:
        image_tensor = preprocess_image(image).to(device)
        with torch.no_grad():
            outputs = model(image_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            prediction = torch.argmax(probs, dim=1).item()
            confidence = probs[0][prediction].item()
        return ("FAKE" if prediction == 1 else "REAL"), confidence
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return "Error", 0.0

def detect_deepfake_audio(audio_path):
    filename = os.path.basename(audio_path)
    model_name = "facebook/wav2vec2-base-960h"
    model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name).to(device)
    processor = Wav2Vec2Processor.from_pretrained(model_name)

    audio_input, sr = librosa.load(audio_path, sr=16000)
    input_values = processor(audio_input, return_tensors="pt", sampling_rate=sr).input_values.to(device)

    with torch.no_grad():
        logits = model(input_values).logits
        prediction = torch.argmax(logits, dim=1).item()
        confidence = torch.softmax(logits, dim=1).max().item()

    return "REAL" if prediction == 1 else "FAKE", confidence


def main():
    st.set_page_config(page_title="Deepfake Detection System", page_icon="🕵️", layout="wide")

    st.markdown(
        """
        <style>
        .stApp { background-color: #000000; color: #FFFFFF; }
        .css-1d391kg { background-color: #1E1E1E; }
        .stButton>button { 
            background-color: #FFFFFF; 
            color: #000000; 
            border: 2px solid #FFFFFF;
            width: 100%;
            margin: 0 auto;
            display: block;
        }
        .stButton>button:hover { 
            background-color: #000000; 
            color: #FFFFFF; 
            border: 2px solid #FFFFFF; 
        }
        .stColumns {
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 1rem;
            padding: 1rem;
        }
        </style>
        """, unsafe_allow_html=True
    )

    st.markdown("<h1 style='text-align: center; color: white; text-shadow: 2px 2px 4px #000000;'>🔍 Deepfake Detection System</h1>", unsafe_allow_html=True)

    # Initialize session state
    if "detection_type" not in st.session_state:
        st.session_state.detection_type = None
    if "file_uploaded" not in st.session_state:
        st.session_state.file_uploaded = None
    if "result" not in st.session_state:
        st.session_state.result = None
    if "confidence" not in st.session_state:
        st.session_state.confidence = None

    
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
        if st.button("📷 Image Detection"):
            st.session_state.detection_type = "image"
            st.session_state.file_uploaded = None
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
        if st.button("🎥 Video Detection"):
            st.session_state.detection_type = "video"
            st.session_state.file_uploaded = None
        st.markdown("</div>", unsafe_allow_html=True)

    with col3:
        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
        if st.button("🎵 Audio Detection"):
            st.session_state.detection_type = "audio"
            st.session_state.file_uploaded = None
        st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.detection_type == "image":
        st.markdown("### 📷 Image Deepfake Detection")
        uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"], key="image_uploader")
        if uploaded_file is not None:
            st.session_state.file_uploaded = uploaded_file
            with st.spinner('Processing image... Please wait'):
                model = load_image_model("Resnet50_Final.pth")
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", width=400)
                result, confidence = detect_deepfake_image(image, model)
                st.session_state.result = result
                st.session_state.confidence = confidence

        if st.session_state.result is not None:
            st.markdown("---")
            st.subheader("Detection Results")
            if st.session_state.result == "FAKE":
                st.error(f"🚨 This image appears to be **FAKE**!")
            else:
                st.success(f"✅ This image appears to be **REAL**!")
            st.markdown(f"**Confidence Score:** {st.session_state.confidence:.2%}")

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=st.session_state.confidence * 100,
                title={'text': "Confidence Score"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "white"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 100], 'color': "gray"}
                    ],
                }
            ))
            fig.update_layout(
                paper_bgcolor="black", 
                font={'color': "white", 'family': "Arial"},
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)

    elif st.session_state.detection_type == "video":
        st.markdown("### 🎥 Video Deepfake Detection")
        uploaded_file = st.file_uploader("Upload a video...", type=["mp4", "avi", "mov"], key="video_uploader")
        if uploaded_file is not None:
            st.session_state.file_uploaded = uploaded_file
            video_path = f"temp_video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{uploaded_file.name.split('.')[-1]}"
            with open(video_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            with st.container():
                col1, col2, col3 = st.columns([1,2,1])
                with col2:
                    st.video(video_path)

            with st.spinner('Processing video... Please wait'):
                model = load_video_model("vit_deepfake.pth")
                result, confidence = detect_deepfake_video(video_path, model)
                st.session_state.result = result
                st.session_state.confidence = confidence

            if os.path.exists(video_path):
                os.remove(video_path)

        if st.session_state.result is not None:
            st.markdown("---")
            st.subheader("Detection Results")
            if st.session_state.result == "FAKE":
                st.error(f"🚨 This video appears to be **FAKE**!")
            else:
                st.success(f"✅ This video appears to be **REAL**!")
            st.markdown(f"**Confidence Score:** {st.session_state.confidence:.2%}")

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=st.session_state.confidence * 100,
                title={'text': "Confidence Score"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "white"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 100], 'color': "gray"}
                    ],
                }
            ))
            fig.update_layout(
                paper_bgcolor="black", 
                font={'color': "white", 'family': "Arial"},
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)

    elif st.session_state.detection_type == "audio":
        st.markdown("### 🎵 Audio Deepfake Detection")
        uploaded_file = st.file_uploader("Upload an audio file...", type=["wav", "mp3"], key="audio_uploader")
        if uploaded_file is not None:
            st.session_state.file_uploaded = uploaded_file
            audio_path = f"temp_audio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{uploaded_file.name.split('.')[-1]}"
            with open(audio_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            st.audio(audio_path)
            
            with st.spinner('Processing audio... Please wait'):
                result, confidence = detect_deepfake_audio(audio_path)
                st.session_state.result = result
                st.session_state.confidence = confidence

            if os.path.exists(audio_path):
                os.remove(audio_path)

        if st.session_state.result is not None:
            st.markdown("---")
            st.subheader("Detection Results")
            if st.session_state.result == "FAKE":
                st.error(f"🚨 This audio appears to be **FAKE**!")
            else:
                st.success(f"✅ This audio appears to be **REAL**!")
            st.markdown(f"**Confidence Score:** {st.session_state.confidence:.2%}")

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=st.session_state.confidence * 100,
                title={'text': "Confidence Score"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "white"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 100], 'color': "gray"}
                    ],
                }
            ))
            fig.update_layout(
                paper_bgcolor="black", 
                font={'color': "white", 'family': "Arial"},
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
