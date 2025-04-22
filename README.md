# Tinder Conversational Length Predictor

This project aims to predict the length of conversations based on users' opening lines and gender using various machine learning approaches. The project includes a Streamlit web application for user interaction and three different modeling approaches: naive, traditional, and deep learning.

A user-friendly web interface for the recommender system is available at: https://ashley-xu-the-tinder-project-app-dev-raysbj.streamlit.app/

## Installation

To set up the project, follow these steps:

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Create a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
   ```

3. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

To run the Streamlit application locally, execute the following command:
    ```bash
    streamlit run app.py
    ```

This will start the web application, allowing users to input their opening lines and gender to predict the conversational length.


## Data Collection and Handling

This project analyzes Tinder conversation data with strict adherence to privacy and ethical guidelines:

- All personal identifiers (names, contact information, locations) have been removed from messages
- No raw conversation data is published online
- Aggregated insights and models are presented without connection to individual users
- Data is stored securely with access limited to authorized researchers only (Dr. Bent via Google Drive)


## File Descriptions

### app.py

This file contains the Streamlit web application that serves as the user interface for the conversational length predictor. Users can input their opening line and select their gender to receive a prediction of the conversation length based on the trained models.

### naive/model.py

This file implements a naive approach using a pre-trained DistilBERT model without fine-tuning. It loads the dataset, generates embeddings for messages using DistilBERT, and trains a linear regression model to predict the conversational length. The model is evaluated using RMSE (Root Mean Squared Error).

### traditional/model.py

This file implements a traditional machine learning approach using a Random Forest classifier. It loads the dataset, preprocesses the data, and trains the model using various features such as "message", "gender_encoded", "Opener Length", "Basic Opener","Gif Opener", "question", "Pickup Line". 

### deep/model.py

This file implements a deep learning approach using a fine-tuned DistilBERT model for sequence classification. It loads the dataset, tokenizes the input, and trains the model using the Hugging Face Transformers library. The model is evaluated using RMSE, and the training process is managed with the Trainer API.


## Evaluation

The system is evaluated using Root Mean Square Error (RMSE) as the primary metric. RMSE is chosen because:
- It penalizes larger errors more heavily than smaller ones
- It's in the same units as the output target (conversation length)
- It's widely used in regression tasks
- It provides an intuitive measure of prediction accuracy

### Evaluation Results
**Naive Approach:** 11.05 RMSE

**Deep Learning Approach:** 10.38 RMSE

**Traditional Approach:** 10.41 RMSE

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
