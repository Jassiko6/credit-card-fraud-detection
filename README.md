
# Credit Card Fraud Detection - Autoencoder

A comparative study of unsupervised anomaly detection techniques using PyTorch. This project evaluates different Autoencoder architectures and PCA to identify fraudulent transactions by measuring reconstruction error.


## Authors

- [@Wiktor Wilk](https://github.com/wiktorw95)
- [@Mikołaj Jassowicz](https://github.com/Jassiko6)


## Project Overview

In the financial industry, fraudulent transactions are rare events (anomalies). This dataset contains transactions made by European cardholders, where frauds make up a small percentage of all transactions.

* An Autoencoder is trained to compress and then reconstruct **normal** transactions with high precision.
* When the model encounters a **fraudulent** transaction (which it hasn't seen before), it fails to reconstruct it accurately.
* A high **Mean Squared Error (MSE)** between the input and the output indicates a high probability of fraud.
## Model Architectures

| Model | Architecture | Key Characteristic |
| :--- | :--- | :--- |
| **Basic AE** | 30 → 15 → 7 | Standard bottleneck design to learn essential features. |
| **Deep AE** | 30 → 20 → 10 → 5 | Deeper hierarchy with **Dropout (0.2)** to prevent overfitting. |
| **Sparse AE** | 30 → 15 → 7 | Uses **L1 Regularization** to force the model to use only the most important neurons. |
| **PCA** | Linear Baseline | Principal Component Analysis with 10 components for statistical comparison. |

---
## Tech Stack

* **Deep Learning:** PyTorch
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn (StandardScaler, PCA, Train-Test Split)
* **Visualization:** Matplotlib
* **Dataset Handling:** KaggleHub
## Installation & Usage

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/Jassiko6/credit-card-fraud-detection
    cd credit-card-fraud-detection
    ```

2.  **Install dependencies**:
    ```bash
    pip install torch pandas numpy matplotlib scikit-learn kagglehub
    ```

3.  **Run the script**:
    ```python
    python main.py
    ```

    The model with deep architecture on hugging face: https://huggingface.co/Jassiko6/credit-card-fraud-detection
