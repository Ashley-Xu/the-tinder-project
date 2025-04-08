
# Deep learning approach using a foundation model (DistilledBERT) fine-tuned with the Tinder Data_v4

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import numpy as np

# ---------- Load and prepare data ----------
def load_data(file_path):
    df = pd.read_csv(file_path)
    df = df[["message", "gender", "Conv Length"]].dropna()
    df["input_text"] = df.apply(lambda row: f"{row['message']} Gender: {row['gender']}", axis=1)
    return df

# ---------- Tokenization ----------
def tokenize_data(texts, tokenizer):
    """
    Tokenize the input texts using the provided tokenizer.

    Args:
        texts (pd.Series): The input texts to be tokenized.
        tokenizer (DistilBertTokenizerFast): The tokenizer for DistilBERT.

    Returns:
        dict: A dictionary containing tokenized inputs.
    """
    return tokenizer(texts.tolist(), truncation=True, padding=True, max_length=128)

class TinderDataset(Dataset):
    """
    Custom Dataset class for loading and processing the Tinder data.

    Args:
        texts (pd.Series): The input texts.
        targets (pd.Series): The target values.
        tokenizer (DistilBertTokenizerFast): The tokenizer for DistilBERT.
    """
    def __init__(self, texts, targets, tokenizer):
        self.encodings = tokenize_data(texts, tokenizer)
        self.labels = torch.tensor(targets.values, dtype=torch.float)

    def __getitem__(self, idx):
        """
        Get a single item from the dataset.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            dict: A dictionary containing the input encodings and the label.
        """
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.labels)

# ---------- Load Pretrained DistilBERT for Regression ----------
def load_model():
    """
    Load the pre-trained DistilBERT model for sequence classification.

    Returns:
        DistilBertForSequenceClassification: The pre-trained DistilBERT model.
    """
    return DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=1,
        problem_type="regression"
    )

# ---------- Training Setup ----------
def setup_training_args():
    return TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_dir="./logs",
        metric_for_best_model="eval_loss"
    )

# ---------- Evaluation Metrics ----------
def compute_metrics(eval_pred):
    """
    Compute evaluation metrics for the model.

    Args:
        eval_pred (tuple): A tuple containing predictions and true labels.

    Returns:
        dict: A dictionary containing the RMSE of the predictions.
    """
    predictions, labels = eval_pred
    predictions = predictions.flatten()
    mse = ((predictions - labels) ** 2).mean()
    rmse = np.sqrt(mse)
    return {"rmse": rmse}

# ---------- Main Training Function ----------
def train_model(df):
    """
    Train the model using the provided DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing the input texts and target values.
    """
    # Train/Validation Split
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df["input_text"], df["Conv Length"], test_size=0.2, random_state=42
    )

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    train_dataset = TinderDataset(train_texts, train_labels, tokenizer)
    val_dataset = TinderDataset(val_texts, val_labels, tokenizer)

    model = load_model()
    training_args = setup_training_args()

    # ---------- Trainer ----------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    # ---------- Train the Model ----------
    trainer.train()
    trainer.evaluate()

    return model, tokenizer

def save_model(model, tokenizer):
    model.save_pretrained("final")
    tokenizer.save_pretrained("final")

# ---------- Entry Point ----------
if __name__ == "__main__":
    data_file_path = "../../data/Tinder_Conv_Data_v4.csv"
    df = load_data(data_file_path)
    model, tokenizer = train_model(df)
    save_model(model, tokenizer)