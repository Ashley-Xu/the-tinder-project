import streamlit as st
import pandas as pd
import torch
from transformers import AutoTokenizer, DistilBertForSequenceClassification
import base64
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Tinder Conversational Length Predictor",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load custom CSS
def load_css():
    css_file = Path(__file__).parent / "style.css"
    if css_file.exists():
        with open(css_file) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load SVG file
def load_svg(svg_file):
    with open(svg_file, "r") as f:
        return f.read()

# Load the trained model and tokenizer
@st.cache_resource
def load_model():
    model = DistilBertForSequenceClassification.from_pretrained("Ashley-Xu/opening-line-strength-predictor")
    tokenizer = AutoTokenizer.from_pretrained("Ashley-Xu/opening-line-strength-predictor")
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
    # Load custom CSS
    load_css()

    # Header with hearts decoration
    col1, col2, col3 = st.columns([1, 3, 1])
    with col1:
        st.markdown('<p class="heart">💜</p>', unsafe_allow_html=True)
    with col2:
        st.markdown('<h1>Project Tinder</h1>', unsafe_allow_html=True)
        st.markdown('<p class="subtitle">Predict conversation length based on your opening line</p>', unsafe_allow_html=True)
    with col3:
        st.markdown('<p class="heart">💗</p>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Main content area
    col_left, col_center, col_right = st.columns([1, 2, 1])

    with col_center:

        # Info box
        st.markdown('''
        <div class="info-box">
            <p style="margin: 0; text-align: center;">
                💬 This AI-powered tool uses a fine-tuned DistilBERT model to predict how long your
                Tinder conversation might last based on your opening line. Get instant feedback to
                optimize your messaging strategy!
            </p>
        </div>
        ''', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # User input
        message = st.text_area(
            "✨ Your Opening Line",
            placeholder="Try: 'This is a cool place to try out snowboarding tricks.' or create your own!",
            height=120,
            help="Enter the message you'd like to send as your first message on Tinder"
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # Center the button
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
        with col_btn2:
            predict_button = st.button("🔮 Predict Length", use_container_width=True)

        # Prediction
        if predict_button:
            if message.strip():
                with st.spinner("🤔 Analyzing your opening line..."):
                    model, tokenizer = load_model()
                    predicted_length = predict_conversational_length(model, tokenizer, message)

                # Display result with custom styling
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(f'''
                <div class="success-box">
                    <h2>🎉 Predicted Conversation Length</h2>
                    <h1 style="color: #9333EA; font-size: 3.5rem; margin: 1rem 0;">{predicted_length:.1f}</h1>
                    <p style="font-size: 1.3rem; color: #6B7280; margin: 0;">message rounds</p>
                </div>
                ''', unsafe_allow_html=True)

                # Interpretation guide
                st.markdown("<br>", unsafe_allow_html=True)
                if predicted_length < 5:
                    interpretation = "💭 Short conversation - Consider adding more personality or a question!"
                    color = "#EF4444"
                elif predicted_length < 10:
                    interpretation = "👍 Moderate conversation - You're on the right track!"
                    color = "#F59E0B"
                else:
                    interpretation = "🌟 Great conversation starter! This could lead to a meaningful connection!"
                    color = "#10B981"

                st.markdown(f'''
                <div class="info-box" style="border-color: {color}; background: linear-gradient(135deg, white 0%, {color}15 100%);">
                    <p style="margin: 0; text-align: center; font-size: 1.1rem; color: {color}; font-weight: 600;">
                        {interpretation}
                    </p>
                </div>
                ''', unsafe_allow_html=True)
            else:
                st.warning("⚠️ Please enter an opening line to get a prediction!")

        # Model info
        st.markdown("<br><br>", unsafe_allow_html=True)
        with st.expander("ℹ️ About the Model"):
            st.markdown("""
            ### Model Details
            - **Architecture**: Fine-tuned DistilBERT for regression
            - **Performance**: 10.38 RMSE on test set
            - **Training Data**: Real Tinder conversation data (anonymized)
            - **Metric**: Predicts number of message exchanges (rounds)

            ### How it works
            The model analyzes your opening line's:
            - Language patterns and sentiment
            - Question-asking behavior
            - Creativity and engagement level
            - Overall message quality
            """)

    # Additional illustration before footer
    couple_image_path = Path(__file__).parent / "assets" / "person_phone.png"
    if couple_image_path.exists():
        col_img1, col_img2, col_img3 = st.columns([1, 2, 1])
        with col_img2:
            st.image(str(couple_image_path), use_container_width=True)

    # Footer
    st.markdown('''
    <div class="footer">
        <p>💜 Built with Streamlit and Transformers | Model: https://huggingface.co/Ashley-Xu/opening-line-strength-predictor 💗</p>
        <p style="font-size: 0.9rem; margin-top: 0.5rem;">
            This tool is for educational purposes. All training data was anonymized and ethically collected.
        </p>
    </div>
    ''', unsafe_allow_html=True)

if __name__ == "__main__":
    main()