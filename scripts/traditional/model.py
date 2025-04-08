

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error

def load_and_preprocess_data(filepath):
    """
    Load and preprocess the dataset.
    Returns features (X) and target (y).
    """
    df = pd.read_csv(filepath)
    df = df[[
        "message", "gender", "Opener Length", "Basic Opener",
        "Gif Opener", "question", "Pickup Line", "Conv Length"
    ]].dropna()

    # Encode gender
    label_encoder = LabelEncoder()
    df["gender_encoded"] = label_encoder.fit_transform(df["gender"])

    X = df[[
        "message", "gender_encoded", "Opener Length", "Basic Opener",
        "Gif Opener", "question", "Pickup Line"
    ]]
    y = df["Conv Length"]
    return X, y

def build_pipeline():
    """
    Build a scikit-learn pipeline with TF-IDF and Random Forest.
    """
    tfidf = TfidfVectorizer(max_features=100)

    preprocessor = ColumnTransformer(transformers=[
        ("tfidf", tfidf, "message"),
        ("pass", "passthrough", [
            "gender_encoded", "Opener Length", "Basic Opener",
            "Gif Opener", "question", "Pickup Line"
        ])
    ])

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    return pipeline

def train_and_evaluate_model(X, y):
    """
    Train the model and evaluate RMSE on test set.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Random Forest RMSE on test set: {rmse:.2f}")

def main():
    """
    Main function to run the full pipeline.
    """
    filepath = "../../data/Tinder_Conv_Data_v4.csv"
    X, y = load_and_preprocess_data(filepath)
    train_and_evaluate_model(X, y)

if __name__ == "__main__":
    main()
