import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import numpy as np


class ResNet50DeepfakeModel(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNet50DeepfakeModel, self).__init__()
        self.resnet50 = models.resnet50(pretrained=False)
        self.resnet50.fc = nn.Linear(self.resnet50.fc.in_features, num_classes)
    def forward(self, x):
        return self.resnet50(x)

def load_resnet50_model(model_path, device):
    try:
        model = ResNet50DeepfakeModel(num_classes=2)
        checkpoint = torch.load(model_path, map_location=device)
        if all(k.startswith('module.') for k in checkpoint.keys()):
            checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}
        model.load_state_dict(checkpoint, strict=False)
        model = model.to(device)
        model.eval()  
        print("ResNet50 model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading ResNet50 model: {e}")
        return None

def preprocess_image(image_path, input_size=(224, 224)):
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Unable to load image from {image_path}.")
            return None
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ResNet50 normalization
        ])
        image = preprocess(image)
        image = image.unsqueeze(0)
        return image
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def classify_image(image_path, model, device):
    image = preprocess_image(image_path)
    if image is None:
        return None
    image = image.to(device)

    with torch.no_grad():
        output = model(image)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, prediction = torch.max(probabilities, 1)

    result = "Fake" if prediction.item() == 1 else "Real"
    confidence_percent = confidence.item() * 100

    return result, confidence_percent

if __name__ == "__main__":
    device = torch.device("cpu")
    print(f"Using device: {device}")

    model_path = "Resnet50_Final.pth" 
    model = load_resnet50_model(model_path, device)
    if model is None:
        exit()

    image_path = "WhatsApp Image 2025-01-25 at 16.25.10.jpeg"  

    result, confidence = classify_image(image_path, model, device)
    if result is not None:
        print(f"Classification Result: {result} with confidence {confidence:.2f}%")