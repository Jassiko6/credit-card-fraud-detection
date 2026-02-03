import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import kagglehub
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from tensorflow.keras import layers, losses, Model
from sklearn.decomposition import PCA
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)
from sklearn.metrics import roc_curve, roc_auc_score

path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")

df = pd.read_csv(f"{path}/creditcard.csv")

X = df.drop(columns=["Class"])
y = df["Class"].values


X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=0
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.2, stratify=y_temp, random_state=0
)

scaler = StandardScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
n_neighbors = 1
knn = KNeighborsClassifier(n_neighbors=n_neighbors)
knn.fit(X_train_scaled, y_train)
y_proba = knn.predict_proba(X_test_scaled)[:, 1]
fpr_knn, tpr_knn, thresholds_knn = roc_curve(y_test, y_proba)
auc_knn = roc_auc_score(y_test, y_proba)

print(auc_knn)

fig, ax = plt.subplots()
ax.plot(fpr_knn, tpr_knn)
ax.text(
    0.8, 0.1,
    f"AUC = {auc_knn:.3f}",
    size='small'
)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve for K-NN Classifier")
plt.savefig(f"roc_curve_knn_{n_neighbors}_neighbors_auc_{round(auc_knn, 5)}.png")
plt.show()


# "nf" means "not fraudulent"
X_train_nf = X_train[y_train == 0]
X_val_nf = X_val[y_val == 0]

scaler = StandardScaler()
X_train_nf_scaled = scaler.fit_transform(X_train_nf)
X_val_nf_scaled = scaler.transform(X_val_nf)
X_test_scaled = scaler.transform(X_test)


X_train_nf_tensor = tf.convert_to_tensor(X_train_nf_scaled, dtype=tf.float32)
X_val_nf_tensor = tf.convert_to_tensor(X_val_nf_scaled, dtype=tf.float32)
X_test_tensor = tf.convert_to_tensor(X_test_scaled, dtype=tf.float32)

class SimpleAutoencoder(Model):
    def __init__(self):
        super().__init__()
        self.encoder = tf.keras.Sequential([
            layers.Dense(30, activation="relu"),
            layers.Dense(15, activation="relu"),
            layers.Dense(7, activation="relu")
        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(7, activation="relu"),
            layers.Dense(15, activation="relu"),
            layers.Dense(30)
        ])

    def call(self, x):
        encoded = self.encoder(x)
        return self.decoder(encoded)


class DeepAutoencoder(Model):
    def __init__(self):
        super().__init__()
        self.encoder = tf.keras.Sequential([
            layers.Dense(30, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(20, activation="relu"),
            layers.Dense(10, activation="relu"),
            layers.Dense(5, activation="relu")
        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(5, activation="relu"),
            layers.Dense(10, activation="relu"),
            layers.Dense(20, activation="relu"),
            layers.Dense(30)
        ])

    def call(self, x):
        encoded = self.encoder(x)
        return self.decoder(encoded)


class SparseAutoencoder(Model):
    def __init__(self, l1_lambda=1e-4):
        super().__init__()
        self.l1_lambda = l1_lambda

        self.encoder = tf.keras.Sequential([
            layers.Dense(30, activation="relu"),
            layers.Dense(15, activation="relu"),
            layers.Dense(7, activation="relu")
        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(7, activation="relu"),
            layers.Dense(15, activation="relu"),
            layers.Dense(30)
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        self.add_loss(self.l1_lambda * tf.reduce_sum(tf.abs(encoded)))
        return decoded



def train_model(model, X_train, X_val, epochs=50):
    model.compile(
        optimizer="adam",
        loss=losses.MeanSquaredError()
    )

    history = model.fit(
        X_train,
        X_train,
        validation_data=(X_val, X_val),
        epochs=epochs,
        batch_size=256,
        shuffle=True,
        verbose=1
    )

    return history




def get_mse(model, X):
    reconstructions = model(X, training=False)
    mse = tf.reduce_mean(tf.square(X - reconstructions), axis=1)
    return mse.numpy()



models = {
    "Basic": SimpleAutoencoder(),
    "Deep": DeepAutoencoder(),
    "Sparse": SparseAutoencoder()
}

results = {}

for name, model in models.items():
    print(f"\nTraining {name} autoencoder...")
    history = train_model(model, X_train_nf_tensor, X_val_nf_tensor)

    mse = get_mse(model, X_test_tensor)

    results[name] = {
        "history": history.history,
        "mse": mse
    }



pca = PCA(n_components=10).fit(X_train_nf_scaled)
pca_rec = pca.inverse_transform(pca.transform(X_test_scaled))

results["PCA"] = {
    "mse": np.mean((X_test_scaled - pca_rec) ** 2, axis=1),
    "history": None
}

fig, axes = plt.subplots(len(results), 3, figsize=(18, 5 * len(results)))

for i, (name, result) in enumerate(results.items()):

    ax_loss = axes[i, 0]
    if result["history"]:
        ax_loss.plot(result["history"]["loss"], label="Train")
        ax_loss.plot(result["history"]["val_loss"], label="Val")
        ax_loss.set_title(f"{name}: Loss")
        ax_loss.legend()
    else:
        ax_loss.text(0.5, 0.5, "PCA (no training loss)", ha="center")

    ax_acc = axes[i, 1]


    mse_normal = result["mse"][y_test == 0]
    threshold = np.percentile(mse_normal, 90)

    y_pred = (result["mse"] > threshold).astype(int)

    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(
        cm,
        display_labels=["Normal", "Fraud"]
    ).plot(ax=axes[i, 1], colorbar=False)

    axes[i, 1].set_title(f"{name}: Confusion Matrix")

    ax_roc = axes[i, 2]
    fpr, tpr, thresholds = roc_curve(y_test, result['mse'])
    roc_auc = roc_auc_score(y_test, result['mse'])
    ax_roc.plot(fpr, tpr)
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title(f"ROC Curve for {name}")
    ax_roc.text(
        0.8, 0.1,
        f"AUC = {roc_auc:.3f}",
        size='small'
    )


    print(f"\n{'='*30}")
    print(f"MODEL: {name}")
    print(f"Threshold: {threshold:.6f}")
    print(classification_report(y_test, y_pred))


plt.tight_layout()
plt.show()
plt.savefig("autoencoders_train_val_losses_roc_curves_and_confusion_matrices.png")


best_mse = results["Deep"]["mse"]

plt.figure(figsize=(10, 6))
plt.hist(best_mse[y_test == 0], bins=100, alpha=0.5, label="Normal", density=True)
plt.hist(best_mse[y_test == 1], bins=100, alpha=0.5, label="Fraud", density=True)
plt.title("Reconstruction Error Distribution (Deep AE)")
plt.xlabel("MSE")
plt.legend()
plt.show()

print(f"Normal mean MSE: {best_mse[y_test == 0].mean():.6f}")
print(f"Fraud mean MSE: {best_mse[y_test == 1].mean():.6f}")
print(f"Ratio: {(best_mse[y_test == 1].mean() / best_mse[y_test == 0].mean()):.2f}x")
