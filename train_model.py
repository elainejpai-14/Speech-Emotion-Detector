import os
import torch
import torchaudio
import torchaudio.transforms as T
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# Emotion mapping
emotion_dict = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
}

# Dataset
class RAVDESSDataset(Dataset):
    def __init__(self, file_list, labels, max_len=400, train=False):
        self.file_list = file_list
        self.labels = labels
        self.max_len = max_len
        self.train = train

        self.mel_transform = T.MelSpectrogram(
            sample_rate=22050, n_fft=1024, hop_length=512, n_mels=128
        )
        self.db_transform = T.AmplitudeToDB()

        # Optional augmentations
        self.freq_mask = T.FrequencyMasking(freq_mask_param=15)
        self.time_mask = T.TimeMasking(time_mask_param=35)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        waveform, sr = torchaudio.load(file_path)
        waveform = waveform.mean(dim=0).unsqueeze(0)

        mel_spec = self.mel_transform(waveform)
        if self.train:
            mel_spec = self.freq_mask(mel_spec)
            mel_spec = self.time_mask(mel_spec)

        mel_spec_db = self.db_transform(mel_spec)

        if mel_spec_db.size(2) < self.max_len:
            mel_spec_db = nn.functional.pad(mel_spec_db, (0, self.max_len - mel_spec_db.size(2)))
        else:
            mel_spec_db = mel_spec_db[:, :, :self.max_len]

        label = torch.tensor(self.labels[idx])
        return mel_spec_db, label

# CNN Model with Dropout
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

# Load dataset
def load_ravdess_dataset(root_dir):
    file_paths, labels = [], []
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".wav"):
                emotion = int(file.split("-")[2]) - 1
                file_paths.append(os.path.join(subdir, file))
                labels.append(emotion)
    return file_paths, labels

# Training function
def train(model, loader, criterion, optimizer, device, scaler):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        with autocast():
            out = model(x)
            loss = criterion(out, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
    return total_loss / len(loader), correct / total

# Evaluation function
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            with autocast():
                out = model(x)
                loss = criterion(out, y)
            total_loss += loss.item()
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
    return total_loss / len(loader), correct / total

# Main
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root_dir = "/kaggle/input/ravdess-emotional-speech-audio"
    file_paths, labels = load_ravdess_dataset(root_dir)

    # First split: 70% train, 30% temp
    train_files, temp_files, train_labels, temp_labels = train_test_split(
        file_paths, labels, test_size=0.3, random_state=42, stratify=labels
    )

    # Second split: 15% val, 15% test (from temp)
    val_files, test_files, val_labels, test_labels = train_test_split(
        temp_files, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
    )

    train_dataset = RAVDESSDataset(train_files, train_labels, train=True)
    val_dataset   = RAVDESSDataset(val_files, val_labels)
    test_dataset  = RAVDESSDataset(test_files, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=16)
    test_loader  = DataLoader(test_dataset, batch_size=16)

    model = EmotionCNN(num_classes=8).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)
    scaler = GradScaler()

    best_val_loss = float('inf')
    patience, wait = 3, 0

    for epoch in range(30):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Acc={train_acc:.4f} | "
              f"Val Loss={val_loss:.4f}, Acc={val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            wait = 0
            torch.save(model.state_dict(), "best_model.pt")
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping triggered.")
                break

    # Final test evaluation (after early stopping)
    model.load_state_dict(torch.load("best_model.pt"))
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\nFinal Test Accuracy: {test_acc:.4f}, Loss: {test_loss:.4f}")
