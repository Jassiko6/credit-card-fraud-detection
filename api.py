from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import HTMLResponse
import pandas as pd
import joblib
import tensorflow as tf
import numpy as np
from io import StringIO
from model_definitions import SimpleAutoencoder, DeepAutoencoder, SparseAutoencoder

app = FastAPI()

try:
    scaler_knn = joblib.load("serialized_objects/scalers/scaler_knn.joblib")
    scaler_ae = joblib.load("serialized_objects/scalers/scaler_ae.joblib")
    thresholds = joblib.load("serialized_objects/thresholds/thresholds.joblib")
    knn_model = joblib.load("serialized_objects/models/knn_model.joblib")

    models = {}
    tf.keras.utils.get_custom_objects().update({
        "SimpleAutoencoder": SimpleAutoencoder,
        "DeepAutoencoder": DeepAutoencoder,
        "SparseAutoencoder": SparseAutoencoder
    })

    for name in ["Basic", "Deep", "Sparse"]:
        models[name] = tf.keras.models.load_model(f"serialized_objects/models/{name}_model.keras")
    models["PCA"] = joblib.load("serialized_objects/models/PCA_model.joblib")
except Exception as e:
    print(f"Error loading models or scalers: {e}")
    models = {}
    knn_model = None
    scaler_knn = None
    scaler_ae = None
    thresholds = {}

@app.get("/", response_class=HTMLResponse)
async def main():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Fraud Detection API</title>
        <style>
            body { font-family: sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            .container { border: 1px solid #ccc; padding: 20px; border-radius: 5px; }
            select, input[type="file"], button { margin: 10px 0; display: block; }
            #result { margin-top: 20px; font-weight: bold; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Credit Card Fraud Detection</h1>
            <p>Upload a CSV file containing a single transaction to check if it's fraudulent.</p>
            
            <label for="model-select">Choose a model:</label>
            <select id="model-select">
                <option value="k-NN">k-NN</option>
                <option value="Basic">Basic Autoencoder</option>
                <option value="Deep">Deep Autoencoder</option>
                <option value="Sparse">Sparse Autoencoder</option>
                <option value="PCA">PCA</option>
            </select>
            
            <label for="csv-file">Upload CSV:</label>
            <input type="file" id="csv-file" accept=".csv">
            
            <button onclick="checkFraud()">Check Transaction</button>
            
            <div id="result"></div>
        </div>

        <script>
            async function checkFraud() {
                const fileInput = document.getElementById('csv-file');
                const modelSelect = document.getElementById('model-select');
                const resultDiv = document.getElementById('result');
                
                if (fileInput.files.length === 0) {
                    resultDiv.innerText = "Please select a CSV file.";
                    resultDiv.style.color = "red";
                    return;
                }

                const formData = new FormData();
                formData.append('file', fileInput.files[0]);
                formData.append('model_name', modelSelect.value);

                try {
                    resultDiv.innerText = "Processing...";
                    resultDiv.style.color = "black";
                    
                    const response = await fetch('/predict', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const data = await response.json();
                    
                    if (response.ok) {
                        const isFraud = data.is_fraud;
                        resultDiv.innerText = isFraud ? "Fraud detected!" : "Transaction is Normal.";
                        resultDiv.style.color = isFraud ? "red" : "green";
                        if (data.details) {
                             resultDiv.innerText += ` (Details: ${data.details})`;
                        }
                    } else {
                        resultDiv.innerText = `Error: ${data.detail || 'Unknown error'}`;
                        resultDiv.style.color = "red";
                    }
                } catch (error) {
                    resultDiv.innerText = `Error: ${error.message}`;
                    resultDiv.style.color = "red";
                }
            }
        </script>
    </body>
    </html>
    """

@app.post("/predict")
async def predict(model_name: str = Form("k-NN"), file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV.")

    try:
        content = await file.read()
        df = pd.read_csv(StringIO(content.decode('utf-8')))

        if "Class" in df.columns:
            df = df.drop(columns=["Class"])

        if df.shape[1] != 30:
             raise HTTPException(status_code=400, detail=f"Expected 30 feature columns, got {df.shape[1]}.")

        is_fraud = False
        details = ""

        if model_name == "k-NN":
            if knn_model is None or scaler_knn is None:
                raise HTTPException(status_code=500, detail="k-NN model or scaler not loaded.")
            
            X_scaled = scaler_knn.transform(df)
            prediction = knn_model.predict(X_scaled)
            is_fraud = bool(prediction[0] == 1)
            
        elif model_name in ["Basic", "Deep", "Sparse"]:
            if model_name not in models or scaler_ae is None:
                 raise HTTPException(status_code=500, detail=f"{model_name} model or scaler not loaded.")
            
            model = models[model_name]
            X_scaled = scaler_ae.transform(df)
            X_tensor = tf.convert_to_tensor(X_scaled, dtype=tf.float32)
            
            reconstructions = model(X_tensor, training=False)
            mse = tf.reduce_mean(tf.square(X_tensor - reconstructions), axis=1).numpy()[0]
            
            threshold = thresholds.get(model_name)
            if threshold is None:
                 raise HTTPException(status_code=500, detail=f"Threshold for {model_name} not found.")

            is_fraud = bool(mse > threshold)
            details = f"MSE: {mse:.6f}, Threshold: {threshold:.6f}"

        elif model_name == "PCA":
            if "PCA" not in models or scaler_ae is None:
                 raise HTTPException(status_code=500, detail="PCA model or scaler not loaded.")
            
            pca = models["PCA"]
            X_scaled = scaler_ae.transform(df)
            
            pca_rec = pca.inverse_transform(pca.transform(X_scaled))
            mse = np.mean((X_scaled - pca_rec) ** 2, axis=1)[0]
            
            threshold = thresholds.get("PCA")
            if threshold is None:
                 raise HTTPException(status_code=500, detail="Threshold for PCA not found.")
                 
            is_fraud = bool(mse > threshold)
            details = f"MSE: {mse:.6f}, Threshold: {threshold:.6f}"

        else:
            raise HTTPException(status_code=400, detail=f"Unknown model: {model_name}")

        return {"is_fraud": is_fraud, "model": model_name, "details": details}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
