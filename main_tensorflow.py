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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.metrics import roc_curve, roc_auc_score

import joblib
from model_definitions import SimpleAutoencoder, DeepAutoencoder, SparseAutoencoder

np.random.seed(0)
tf.random.set_seed(0)

models = {}

path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")

df = pd.read_csv(f"{path}/creditcard.csv")

X = df.drop(columns=["Class"])
y = df["Class"].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, stratify=y_test, random_state=0)

scalers = {}

scaler = StandardScaler()
scaler.fit(X_train)

scalers["k-NN"] = scaler

joblib.dump(scaler, "serialized_objects/scalers/scaler_knn.joblib")

X_train_scaled = scaler.transform(X_train)
X_test_scaled_knn = scaler.transform(X_test)

n_neighbors = 5
knn = KNeighborsClassifier(n_neighbors=n_neighbors)
knn.fit(X_train_scaled, y_train)

models["k-NN"] = knn

joblib.dump(knn, "serialized_objects/models/knn_model.joblib")

y_proba = knn.predict_proba(X_test_scaled_knn)[:, 1]
fpr_knn, tpr_knn, thresholds_knn = roc_curve(y_test, y_proba)
auc_knn = roc_auc_score(y_test, y_proba)


# "nf" means "not fraudulent"
X_train_nf = X_train[y_train == 0]
X_val_nf = X_val[y_val == 0]

scaler = StandardScaler()
X_train_nf_scaled = scaler.fit_transform(X_train_nf)
X_val_nf_scaled = scaler.transform(X_val_nf)
X_test_scaled_ae = scaler.transform(X_test)

scalers["ae"] = scaler

joblib.dump(scaler, "serialized_objects/scalers/scaler_ae.joblib")

X_train_nf_tensor = tf.convert_to_tensor(X_train_nf_scaled, dtype=tf.float32)
X_val_nf_tensor = tf.convert_to_tensor(X_val_nf_scaled, dtype=tf.float32)
X_test_tensor = tf.convert_to_tensor(X_test_scaled_ae, dtype=tf.float32)

def train_model(model, X_train, X_val, epochs=50):
    model.compile(optimizer="adam", loss=losses.MeanSquaredError())

    history = model.fit(
        X_train,
        X_train,
        validation_data=(X_val, X_val),
        epochs=epochs,
        batch_size=256,
        shuffle=True,
        verbose=1,
    )

    return history


def get_mse(model, X):
    reconstructions = model(X, training=False)
    mse = tf.reduce_mean(tf.square(X - reconstructions), axis=1)
    return mse.numpy()


autoencoders = {
    "Basic": SimpleAutoencoder(),
    "Deep": DeepAutoencoder(),
    "Sparse": SparseAutoencoder(),
}

results = {}
thresholds = {}

for name, model in autoencoders.items():
    print(f"\nTraining {name} autoencoder...")
    history = train_model(model, X_train_nf_tensor, X_val_nf_tensor)

    models[name] = model

    mse = get_mse(model, X_test_tensor)

    results[name] = {"history": history.history, "mse": mse}

    mse_normal = mse[y_test == 0]
    threshold = np.percentile(mse_normal, 90)
    thresholds[name] = threshold


pca = PCA(n_components=10, random_state=0).fit(X_train_nf_scaled)
pca_rec = pca.inverse_transform(pca.transform(X_test_scaled_ae))
models["PCA"] = pca

print("Saving models...")
for name, model in models.items():
     if isinstance(model, tf.keras.Model):
         model.save(f"{name}_model.keras")
     else:
         joblib.dump(model, f"serialized_objects/models/{name}_model.joblib")

mse_pca = np.mean((X_test_scaled_ae - pca_rec) ** 2, axis=1)
mse_pca_normal = mse_pca[y_test == 0]
threshold_pca = np.percentile(mse_pca_normal, 90)
thresholds["PCA"] = threshold_pca

results["PCA"] = {
    "mse": mse_pca,
    "history": None,
}

joblib.dump(thresholds, "serialized_objects/thresholds/thresholds.joblib")


plt.figure(figsize=(10, 6))
for name, result in results.items():
    if result["history"]:
        plt.plot(result["history"]["loss"], label=f"{name} Train")
        plt.plot(result["history"]["val_loss"], linestyle="--", label=f"{name} Val")
plt.title("Model Losses")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig("results/metrics_imgs/model_losses.png")
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(fpr_knn, tpr_knn, label=f"k-NN (AUC = {auc_knn:.3f})")

for name, result in results.items():
    fpr, tpr, thresholds = roc_curve(y_test, result["mse"])
    roc_auc = roc_auc_score(y_test, result["mse"])
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")

plt.plot([0, 1], [0, 1], "k--", label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves")
plt.legend()
plt.tight_layout()
plt.savefig("results/metrics_imgs/roc_curves.png")
plt.show()

model_predictions = {}

for name, result in results.items():
    mse_normal = result["mse"][y_test == 0]
    threshold = np.percentile(mse_normal, 90)
    y_pred = (result["mse"] > threshold).astype(int)
    model_predictions[name] = y_pred

model_predictions["k-NN"] = knn.predict(X_test_scaled_knn)

num_models = len(model_predictions)
cols = 2
rows = (num_models + 1) // 2

fig, axes = plt.subplots(rows, cols, figsize=(12, 6 * rows))
axes = axes.flatten()

for i, (name, y_pred) in enumerate(model_predictions.items()):
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=["Normal", "Fraud"]).plot(
        ax=axes[i], colorbar=False
    )

    axes[i].set_title(f"{name}: Confusion Matrix")

    print(f"\n{'=' * 30}")
    print(f"MODEL: {name}")

    if name in results:
        mse_normal = results[name]["mse"][y_test == 0]
        threshold = np.percentile(mse_normal, 90)
        print(f"Threshold: {threshold:.6f}")

    print(classification_report(y_test, y_pred))

for j in range(num_models, len(axes)):
    axes[j].axis("off")

plt.tight_layout()
plt.savefig("results/metrics_imgs/confusion_matrices.png")
plt.show()


best_mse = results["Basic"]["mse"]

plt.figure(figsize=(10, 6))
plt.hist(best_mse[y_test == 0], bins=100, alpha=0.5, label="Normal", density=True)
plt.hist(best_mse[y_test == 1], bins=100, alpha=0.5, label="Fraud", density=True)
plt.title("Reconstruction Error Distribution (Simple AE)")
plt.xlabel("MSE")
plt.legend()
plt.tight_layout()
plt.savefig("results/metrics_imgs/reconstruction_error_distribution.png")
plt.show()

print(f"Normal mean MSE: {best_mse[y_test == 0].mean():.6f}")
print(f"Fraud mean MSE: {best_mse[y_test == 1].mean():.6f}")
print(f"Ratio: {(best_mse[y_test == 1].mean() / best_mse[y_test == 0].mean()):.2f}x")
