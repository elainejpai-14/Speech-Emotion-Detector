import os
import torch
import torchaudio
import torchaudio.transforms as T
import torch.nn as nn
from torch.cuda.amp import autocast
import gradio as gr

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Emotion mapping
emotion_dict = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
}
label_map = list(emotion_dict.values())

# Model definition (must match your training)
class EmotionCNN(nn.Module):
    def __init__(self, num_classes):
        super(EmotionCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.3),
            nn.Flatten(),
            nn.Linear(32 * 32 * 100, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# Load model
model = EmotionCNN(num_classes=8).to(device)
model.load_state_dict(torch.load("best_model.pt", map_location=device))
model.eval()

# Prediction function
def predict_emotion(audio):
    waveform, sr = torchaudio.load(audio)
    waveform = waveform.mean(dim=0).unsqueeze(0)

    mel_transform = T.MelSpectrogram(
        sample_rate=22050, n_fft=1024, hop_length=512, n_mels=128
    )
    db_transform = T.AmplitudeToDB()
    mel_spec = mel_transform(waveform)
    mel_spec_db = db_transform(mel_spec)

    # Pad or trim
    if mel_spec_db.size(2) < 400:
        mel_spec_db = nn.functional.pad(mel_spec_db, (0, 400 - mel_spec_db.size(2)))
    else:
        mel_spec_db = mel_spec_db[:, :, :400]

    mel_spec_db = mel_spec_db.unsqueeze(0).to(device)

    with torch.no_grad():
        with autocast():
            output = model(mel_spec_db)
            pred = output.argmax(1).item()
    
    return label_map[pred]

# Gradio Interface
interface = gr.Interface(
    fn=predict_emotion,
    inputs=gr.Audio(type="filepath", label="🎙️ Record or Upload Your Voice"),
    outputs=gr.Text(label="Detected Emotion"),
    title="🎙️ Speech Emotion Recognition",
    description="This app uses a CNN trained on RAVDESS to classify emotions from your voice."
)

interface.launch()
