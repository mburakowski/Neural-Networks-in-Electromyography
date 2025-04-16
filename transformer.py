import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from copy import deepcopy
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from scipy.io import loadmat
from scipy.signal import stft
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

def load_data(folder_path):
    X, y = [], []
    classes = sorted([d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))])
    for class_name in classes:
        class_folder = os.path.join(folder_path, class_name)
        for file in os.listdir(class_folder):
            if file.endswith(".mat"):
                data = loadmat(os.path.join(class_folder, file))
                if 'emg' not in data:
                    print(f"Brak klucza 'emg' w {file}")
                    continue
                X.append(data['emg'])
                y.append(class_name)
    return np.array(X), np.array(y)

def signal_to_spectrogram(emg_signal, fs=200, nperseg=64, noverlap=32):
    channels = emg_signal.shape[1]
    spectrograms = []
    for ch in range(channels):
        _, _, Zxx = stft(emg_signal[:, ch], fs=fs, window='hamming', nperseg=nperseg, noverlap=noverlap)
        S = np.log1p(np.abs(Zxx)) 
        spectrograms.append(S)
    return np.stack(spectrograms, axis=0)  

def augment_emg(emg_signal, noise_std=0.01, scale_range=(0.8, 1.2), time_shift_max=20):
    noise = np.random.normal(0, noise_std, emg_signal.shape)
    emg_aug = emg_signal + noise

    scale_factor = np.random.uniform(*scale_range)
    emg_aug *= scale_factor

    shift = np.random.randint(-time_shift_max, time_shift_max)
    if shift > 0:
        emg_aug = np.pad(emg_aug, ((shift, 0), (0, 0)), mode='constant')[:-shift]
    elif shift < 0:
        emg_aug = np.pad(emg_aug, ((0, -shift), (0, 0)), mode='constant')[-shift:]
    
    return emg_aug

class EMGDataset(Dataset):
    def __init__(self, X_raw, y_raw, label_encoder, max_val=None, augment=False):
        self.augment = augment
        self.X_raw = X_raw
        self.y = y_raw
        self.label_encoder = label_encoder
        self.max_val = max_val or np.max([signal_to_spectrogram(x).max() for x in X_raw])

    def __len__(self):
        return len(self.X_raw)

    def __getitem__(self, idx):
        x = self.X_raw[idx]
        if self.augment:
            x = augment_emg(x)
        spec = signal_to_spectrogram(x)
        spec = spec / self.max_val

        c, f, t = spec.shape
        spec = spec.transpose(2, 0, 1).reshape(t, c * f)
        return torch.tensor(spec, dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.long)

class EMGTransformer(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.embedding = nn.Linear(input_dim, 64)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=2,
            dim_feedforward=64,
            dropout=0.5,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.classifier(x)

def train_model(model, train_loader, val_loader, epochs=30, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        val_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
                preds = outputs.argmax(dim=1)
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)

        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct / total
        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, Val Acc={val_acc:.2f}")
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)

    return history

def evaluate_model(model, loader, label_encoder=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            preds = model(X_batch).argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())

    acc = np.mean(np.array(all_preds) == np.array(all_targets))
    print(f"Dokładność na zbiorze testowym: {acc * 100:.2f}%")

    if label_encoder:
        cm = confusion_matrix(all_targets, all_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
        disp.plot(xticks_rotation='vertical', cmap='Blues')
        plt.title("Macierz pomyłek")
        plt.grid(False)
        plt.show()

    return all_preds, all_targets

def train_one_fold(model, train_loader, val_loader, device, epochs=50, patience=5, lr=1e-3):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_model = deepcopy(model.state_dict())
    best_val_loss = float('inf')
    wait = 0
    history = []

    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            out = model(X_batch)
            loss = criterion(out, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Walidacja
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                out = model(X_batch)
                val_loss += criterion(out, y_batch).item()
                preds = out.argmax(1)
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)

        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct / total
        print(f"Epoka {epoch}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, Val Acc={val_acc:.2f}")

        history.append((avg_train_loss, avg_val_loss, val_acc))

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model = deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_model)
    return model, history


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Używane urządzenie:", device)

    train_path = 'pomiary_chwyty/trening'
    test_path = 'pomiary_chwyty/test'

    X_train_raw, y_train_raw = load_data(train_path)
    X_test_raw, y_test_raw = load_data(test_path)

    le = LabelEncoder()
    le.fit(np.concatenate((y_train_raw, y_test_raw)))
    num_classes = len(le.classes_)

    X = np.array(X_train_raw)
    y = np.array(le.transform(y_train_raw))

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_models = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n===== Fold {fold} =====")

        train_set = EMGDataset(X[train_idx], y[train_idx], le, augment=True)
        val_set = EMGDataset(X[val_idx], y[val_idx], le, max_val=train_set.max_val)

        train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=16)

        input_dim = train_set[0][0].shape[1]
        model = EMGTransformer(input_dim=input_dim, num_classes=num_classes)

        model, _ = train_one_fold(model, train_loader, val_loader, device)
        fold_models.append(deepcopy(model))

    print("\n=== Ewaluacja na zbiorze testowym ===")
    y_test_encoded = le.transform(y_test_raw)
    test_set = EMGDataset(X_test_raw, y_test_encoded, le, max_val=train_set.max_val)
    test_loader = DataLoader(test_set, batch_size=16)
    model = fold_models[-1]
    preds, targets = evaluate_model(model, test_loader, label_encoder=le)

    print("\n=== Classification Report ===")
    print(classification_report(targets, preds, target_names=le.classes_))


if __name__ == "__main__":
    main()
