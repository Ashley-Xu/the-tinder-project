import streamlit as st
import pandas as pd
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

# Load the trained model and tokenizer
@st.cache_resource
def load_model():
    model = DistilBertForSequenceClassification.from_pretrained("Ashley-Xu/opening-line-strength-predictor")
    tokenizer = DistilBertTokenizerFast.from_pretrained("Ashley-Xu/opening-line-strength-predictor")
    model.eval()  # Set to evaluation mode
    return model, tokenizer

# Function to predict conversational length
def predict_conversational_length(model, tokenizer, message):
    # Prepare input text
    input_text = f"{message}"
    
    # Tokenize input
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    
    # Move inputs to the appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = outputs.logits.flatten().cpu().numpy()
    
    return prediction[0]  # Return the predicted conversational length

# Streamlit app layout
def main():
    st.title("Tinder Conversational Length Predictor")
    st.write("Enter your opening line and gender to predict the length of the conversation.")

    # User input
    message = st.text_area("Opening Line", "This is a cool place to try out snowboarding tricks.")
    #gender = st.selectbox("Gender", ["Male", "Female", "Other"])

    if st.button("Predict"):
        model, tokenizer = load_model()
        predicted_length = predict_conversational_length(model, tokenizer, message)
        st.success(f"Predicted Conversational Length: {predicted_length:.2f} rounds")

if __name__ == "__main__":
    main()