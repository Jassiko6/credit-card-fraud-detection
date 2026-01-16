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


# Download latest version
path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")

df = pd.read_csv(f"{path}/creditcard.csv")

normal_transaction_data = df[df['Class'] == 0].drop('Class', axis=1)
fraud_transaction_data = df[df['Class'] == 1].drop('Class', axis=1)

scaler = StandardScaler()


scaler.fit(df.drop('Class', axis=1))


X_normal_scaled = scaler.transform(normal_transaction_data)
X_fraud_scaled = scaler.transform(fraud_transaction_data)


X_train_norm, X_temp_norm = train_test_split(X_normal_scaled, test_size=0.2)
X_val_norm, X_test_norm = train_test_split(X_temp_norm, test_size=0.5)

X_test = np.vstack([X_test_norm, X_fraud_scaled])
y_test = np.array([0] * len(X_test_norm) + [1] * len(X_fraud_scaled))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Używane urządzenie: {device}")

X_val_tensor = torch.FloatTensor(X_val_norm).to(device)
X_test_tensor = torch.FloatTensor(X_test).to(device)
train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train_norm).to(device)), batch_size=1024, shuffle=True)
