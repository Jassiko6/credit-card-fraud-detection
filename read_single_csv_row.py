import pandas as pd
import kagglehub
import numpy as np
import os


def create_sample_transactions(output_dir="sample_transactions"):
    """
    Reads the credit card fraud dataset, selects one random normal transaction
    and one random fraudulent transaction, and saves them to separate CSV files.
    """
    print("Downloading dataset from KaggleHub...")
    path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
    df = pd.read_csv(f"{path}/creditcard.csv")
    print("Dataset loaded successfully.")

    # Separate normal and fraudulent transactions
    normal_transactions = df[df['Class'] == 0]
    fraudulent_transactions = df[df['Class'] == 1]

    # Check if there are enough transactions of each type
    if normal_transactions.empty:
        print("No normal transactions found in the dataset.")
        return
    if fraudulent_transactions.empty:
        print("No fraudulent transactions found in the dataset.")
        return

    # Select one random normal transaction
    # Using .sample(1) to get a DataFrame with one row
    random_normal_transaction = normal_transactions.sample(1, random_state=42)

    # Select one random fraudulent transaction
    random_fraudulent_transaction = fraudulent_transactions.sample(1, random_state=42)

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Define output file paths
    normal_output_path = os.path.join(output_dir, "sample_normal_transaction.csv")
    fraud_output_path = os.path.join(output_dir, "sample_fraudulent_transaction.csv")

    # Save to CSV files
    # index=False prevents pandas from writing the DataFrame index as a column
    random_normal_transaction.to_csv(normal_output_path, index=False)
    random_fraudulent_transaction.to_csv(fraud_output_path, index=False)

    print(f"Sample normal transaction saved to: {normal_output_path}")
    print(f"Sample fraudulent transaction saved to: {fraud_output_path}")


if __name__ == "__main__":
    # Set a random seed for reproducibility of the sample selection
    np.random.seed(42)
    create_sample_transactions()