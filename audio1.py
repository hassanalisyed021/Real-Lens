import torch
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForSequenceClassification.from_pretrained(
    "facebook/wav2vec2-base-960h", num_labels=2  # 2 labels: spoof (0) or bonafide (1)
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)