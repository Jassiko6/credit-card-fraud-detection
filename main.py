import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, precision_recall_curve,
    auc, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.decomposition import PCA
import kagglehub


path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")

df = pd.read_csv(f"{path}/creditcard.csv")

normal_transaction_data = df[df['Class'] == 0].drop('Class', axis=1)
fraud_transaction_data = df[df['Class'] == 1].drop('Class', axis=1)

X_train_norm, X_temp_norm = train_test_split(normal_transaction_data, test_size=0.2)
X_val_norm, X_test_norm = train_test_split(X_temp_norm, test_size=0.5)

scaler = StandardScaler()

scaler.fit(X_train_norm)

X_train_norm_scaled = scaler.transform(X_train_norm)
X_val_norm_scaled = scaler.transform(X_val_norm)
X_test_norm_scaled = scaler.transform(X_test_norm)
X_fraud_scaled = scaler.transform(fraud_transaction_data)


X_test_scaled = np.vstack([X_test_norm_scaled, X_fraud_scaled])
y_test = np.array([0] * len(X_test_norm_scaled) + [1] * len(X_fraud_scaled))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Używane urządzenie: {device}")

X_val_tensor = torch.FloatTensor(X_val_norm_scaled).to(device)
X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train_norm_scaled).to(device)), batch_size=1024, shuffle=True)

class SimpleNeuralNet(nn.Module):
  def __init__(self):
       super().__init__()
       self.encoder = nn.Sequential(
           nn.Linear(30, 15),
           nn.ReLU(),
           nn.Linear(15, 7),
           nn.ReLU(),
       )
       self.decoder = nn.Sequential(
           nn.Linear(7, 15),
           nn.ReLU(),
           nn.Linear(15, 30),
       )
  def forward(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded


class DeepNeuralNet(nn.Module):
  def __init__(self):
       super().__init__()
       self.encoder = nn.Sequential(
           nn.Linear(30, 20),
           nn.ReLU(),
           nn.Dropout(0.2),
           nn.Linear(20, 10),
           nn.ReLU(),
           nn.Linear(10, 5),
           nn.ReLU()
       )
       self.decoder = nn.Sequential(
           nn.Linear(5, 10),
           nn.ReLU(),
           nn.Linear(10, 20),
           nn.ReLU(),
           nn.Linear(20, 30),
       )
  def forward(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

class SparseNeuralNet(nn.Module):
  def __init__(self):
       super().__init__()
       self.encoder = nn.Sequential(
           nn.Linear(30, 15),
           nn.ReLU(),
           nn.Linear(15, 7),
           nn.ReLU()
       )
       self.decoder = nn.Sequential(
           nn.Linear(7, 15),
           nn.ReLU(),
           nn.Linear(15, 30),
       )
  def forward(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return encoded, decoded

def train_model(model, train_loader, val_data, epochs=50, is_sparse=False):
    print(f"\nRozpoczynam trening: {model.__class__.__name__}...")
    criterion = nn.MSELoss()
    criterion_none = nn.MSELoss(reduction='none')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        correct_train = 0
        total_train = 0

        # --- Pętla treningowa ---
        for batch in train_loader:
            x = batch[0]
            optimizer.zero_grad()

            if is_sparse:
                encoded, decoded = model(x)
                loss = criterion(decoded, x)
                l1_reg = 1e-4 * torch.norm(encoded, 1)
                loss += l1_reg
                output = decoded
            else:
                decoded = model(x)
                loss = criterion(decoded, x)
                output = decoded

            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

            # --- Accuracy ---
            with torch.no_grad():
                batch_mse = torch.mean((x - output) ** 2, dim=1)
                limit = batch_mse.mean() + 2 * batch_mse.std()
                correct_train += (batch_mse < limit).sum().item()
                total_train += x.size(0)

        # --- Walidacja ---
        model.eval()
        with torch.no_grad():
            # Forward pass
            val_out = model(val_data)
            val_decoded = val_out[1] if is_sparse else val_out

            # Obliczenie Loss
            val_loss = criterion(val_decoded, val_data).item()

            # Obliczenie Accuracy (na podstawie dynamicznego progu, tak jak w treningu)
            val_mse = torch.mean((val_data - val_decoded) ** 2, dim=1)
            val_limit = val_mse.mean() + 2 * val_mse.std()

            val_acc = (val_mse < val_limit).float().mean().item()

        # --- Zapisywanie wyników ---
        avg_train_loss = sum(epoch_losses) / len(epoch_losses)
        train_acc = correct_train / total_train

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_train_loss:.5f} | Val Loss: {val_loss:.5f} | Train Acc: {train_acc:.3f} | Val Acc: {val_acc:.3f}")

    return history

# Obliczanie MSE dla architektur
def get_mse(model, data_tensor, is_sparse=False):
    model.eval()
    with torch.no_grad():
        output = model(data_tensor)
        decoded = output[1] if is_sparse else output
        return torch.mean((data_tensor - decoded)**2, dim=1).cpu().numpy()

# Architektury i placeholder na wyniki
models = {
    "Basic": (SimpleNeuralNet(), False),
    "Deep": (DeepNeuralNet(), False),
    "Sparse": (SparseNeuralNet(), True)
}

results = {}

# Trenowanie naszych architektur
for name, (architecture, sparse) in models.items():
    print(f"\nTrenowanie {name}...")
    hist = train_model(architecture.to(device), train_loader, X_val_tensor, is_sparse=sparse)
    mse = get_mse(architecture, X_test_tensor, is_sparse=sparse)
    results[name] = {"mse": mse, "history": hist}

# PCA - to jest upraszczacz, który uczy się jak zapamiętywać, używając 10 bardzo ważnych informacji.
# Jeżeli nowe dane są podobne do początkowych, MSE jest niskie.
pca = PCA(n_components=10).fit(X_train_norm_scaled)
pca_rec = pca.inverse_transform(pca.transform(X_test_scaled)) # Opisujemy za pomocą 10 ważnych informacji a potem próbujemy odbudować dane z pamięci.
results["PCA"] = {"mse": np.mean((X_test_scaled - pca_rec)**2, axis=1), "history": None}

fig, axes = plt.subplots(len(results), 3, figsize=(18, 5 * len(results)))

for i, (name, result) in enumerate(results.items()):

    # A. Wykres Loss
    ax_loss = axes[i, 0]
    if result['history']:
        ax_loss.plot(result['history']['train_loss'], label='Train Loss')
        ax_loss.plot(result['history']['val_loss'], label='Val Loss')
        ax_loss.set_title(f"{name}: Loss (MSE)")
        ax_loss.legend()
    else:
        ax_loss.text(0.5, 0.5, "PCA - brak Loss w czasie", ha='center')

    # B. Wykres Accuracy
    ax_acc = axes[i, 1]
    if result['history']:
        ax_acc.plot(result['history']['train_acc'], label='Train Acc')
        ax_acc.plot(result['history']['val_acc'], label='Val Acc')
        ax_acc.set_title(f"{name}: Accuracy (Reconstruction)")
        ax_acc.set_ylim(0.5, 1.05)
        ax_acc.legend()
    else:
        ax_acc.text(0.5, 0.5, "PCA - brak Acc w czasie", ha='center')

    # C. Confusion Matrix
    # Wyznaczanie progu: 95 percentyl błędów na próbkach normalnych
    mse_normal_only = result['mse'][y_test == 0]
    threshold = np.percentile(mse_normal_only, 90)

    y_pred = (result['mse'] > threshold).astype(int)

    ax_cm = axes[i, 2]
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=["Normal", "Fraud"]).plot(ax=ax_cm, cmap='Blues', colorbar=False)
    ax_cm.set_title(f"{name}: Confusion Matrix")

    # Wypisanie metryk w konsoli
    print(f"\n{'='*20}")
    print(f"RAPORT DLA MODELU: {name}")
    print(f"Ustalony próg błędu (Threshold): {threshold:.4f}")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Fraud"]))

plt.tight_layout()
plt.show()

# Histogram błędów dla najlepszego modelu (np. Deep)
best_mse = results['Deep']['mse']
plt.figure(figsize=(10, 6))
plt.hist(best_mse[y_test == 0], bins=100, alpha=0.5, label='Normalne', density=True, range=(0, 5))
plt.hist(best_mse[y_test == 1], bins=100, alpha=0.5, label='Oszustwa', density=True, range=(0, 5))
plt.title("Rozkład błędów rekonstrukcji (Deep Neural Net)")
plt.xlabel("Błąd MSE")
plt.legend()
plt.show()

# Oblicz średni błąd dla obu grup osobno (dla architektury Deep)
mse_normal = results['Deep']['mse'][y_test == 0]
mse_fraud = results['Deep']['mse'][y_test == 1]

print(f"Średni błąd dla normalnych: {mse_normal.mean():.6f}")
print(f"Średni błąd dla oszustw: {mse_fraud.mean():.6f}")
print(f"Różnica (Ratio): {mse_fraud.mean() / mse_normal.mean():.2f}x")
# Jeżeli Różnica jest 40-50x większa, to dobrze odróżnia anomalie