# Naive approach using a foundation BERT model without fine-tuning

import pandas as pd
import torch
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from transformers import DistilBertTokenizerFast, DistilBertModel

# ---------- Load and Prepare Data ----------
def load_data(file_path):
    df = pd.read_csv(file_path)
    df = df[["message", "gender", "Conv Length"]].dropna()
    return df

def encode_gender(df):
    """
    Encode the gender column in the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing the raw data.

    Returns:
        pd.DataFrame: The DataFrame with an additional column for encoded gender.
    """
    label_encoder = LabelEncoder()
    df["gender_encoded"] = label_encoder.fit_transform(df["gender"])
    return df

# ---------- Load Pre-trained DistilBERT ----------
def initialize_model():
    """
    Load the pre-trained DistilBERT model and tokenizer.

    Returns:
        tuple: A tuple containing:
            - tokenizer (DistilBertTokenizerFast): The tokenizer for DistilBERT.
            - model (DistilBertModel): The pre-trained DistilBERT model.
    """
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    model = DistilBertModel.from_pretrained("distilbert-base-uncased")
    model.eval()  # Set to inference mode
    return tokenizer, model

# ---------- Generate Embeddings ----------
def get_cls_embedding(text, tokenizer, model, device):
    """
    Generate the CLS token embedding for a given text using DistilBERT.

    Args:
        text (str): The input text to be tokenized and embedded.
        tokenizer (DistilBertTokenizerFast): The tokenizer for DistilBERT.
        model (DistilBertModel): The pre-trained DistilBERT model.
        device (torch.device): The device to run the model on (CPU or GPU).

    Returns:
        np.ndarray: The CLS token embedding as a NumPy array.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()  # CLS token

def generate_embeddings(df, tokenizer, model, device):
    """
    Generate embeddings for all messages in the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing the messages.
        tokenizer (DistilBertTokenizerFast): The tokenizer for DistilBERT.
        model (DistilBertModel): The pre-trained DistilBERT model.
        device (torch.device): The device to run the model on (CPU or GPU).

    Returns:
        np.ndarray: A 2D array of embeddings for each message.
    """
    embeddings = np.vstack([get_cls_embedding(text, tokenizer, model, device) for text in df["message"]])
    return embeddings

# ---------- Train and Evaluate Model ----------
def train_model(X, y):
    """
    Train a linear regression model on the provided features and target.

    Args:
        X (np.ndarray): The feature matrix.
        y (np.ndarray): The target variable.

    Returns:
        LinearRegression: The trained linear regression model.
    """
    regressor = LinearRegression()
    regressor.fit(X, y)
    return regressor

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model using RMSE.

    Args:
        model (LinearRegression): The trained linear regression model.
        X_test (np.ndarray): The test feature matrix.
        y_test (np.ndarray): The true target values for the test set.

    Returns:
        float: The RMSE of the model's predictions.
    """
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return rmse

# ---------- Main Function ----------
def main():
    data_file_path = "../../data/Tinder_Conv_Data_v4.csv"
    
    # Load and prepare data
    df = load_data(data_file_path)
    df = encode_gender(df)

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize tokenizer and model
    tokenizer, bert_model = initialize_model()
    bert_model.to(device)

    # Generate embeddings
    embeddings = generate_embeddings(df, tokenizer, bert_model, device)
    gender_column = df["gender_encoded"].values.reshape(-1, 1)
    X = np.hstack([embeddings, gender_column])
    y = df["Conv Length"].values

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = train_model(X_train, y_train)

    # Evaluate the model
    rmse = evaluate_model(model, X_test, y_test)
    print(f"Benchmark DistilBERT + Linear Regression RMSE: {rmse:.2f}")

# ---------- Entry Point ----------
if __name__ == "__main__":
    main()