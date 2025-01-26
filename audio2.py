import torch
import librosa
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor

def detect_audio_spoof(audio_path):
    try:
       
        model_name = "facebook/wav2vec2-base-960h"
        model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)
        processor = Wav2Vec2Processor.from_pretrained(model_name)

        
        audio_input, sr = librosa.load(audio_path, sr=16000)
        
    
        input_values = processor(audio_input, return_tensors="pt", sampling_rate=sr).input_values

        
        with torch.no_grad():
            logits = model(input_values).logits
            prediction = torch.argmax(logits, dim=1).item()
            confidence = torch.softmax(logits, dim=1).max().item()

        
        result = {
            "is_bonafide": prediction == 1,
            "confidence": confidence
        }
        return result

    except Exception as e:
        print(f"Error processing audio: {e}")
        return None


audio_path = "ai_voice.wav"
result = detect_audio_spoof(audio_path)

if result:
    status = "✅ Bonafide" if result['is_bonafide'] else "❌ Spoofed"
    print(f"{status} Audio (Confidence: {result['confidence']:.2f})")