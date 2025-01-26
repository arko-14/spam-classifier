import streamlit as st
import os
from tensorflow.keras.models import load_model
import joblib

# Define relative paths for model files
MODEL_PATH = os.path.join("spam_classifier", "spam_classifier_ann.h5")
VECTORIZER_PATH = os.path.join("spam_classifier", "vectorizer(4).pkl")

# Custom CSS for a better frontend experience
st.markdown("""
    <style>
    .title {
        text-align: center;
        color: #1E90FF;
        font-size: 36px;
        font-weight: bold;
    }
    .header {
        font-size: 24px;
        font-weight: bold;
        color: #333;
    }
    .info-text {
        font-size: 18px;
        color: #555;
    }
    .input-container {
        margin-top: 30px;
    }
    .input-field {
        padding: 10px;
        font-size: 18px;
        width: 100%;
        border: 1px solid #ddd;
        border-radius: 5px;
    }
    .classify-btn {
        background-color: #1E90FF;
        color: white;
        font-size: 18px;
        border-radius: 5px;
        padding: 10px;
        width: 100%;
    }
    .result {
        font-size: 20px;
        font-weight: bold;
        text-align: center;
        padding: 20px;
        border-radius: 10px;
    }
    .spam {
        background-color: #ff6347;
        color: white;
    }
    .ham {
        background-color: #32cd32;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Title and Description
st.markdown('<p class="title">Spam Classifier</p>', unsafe_allow_html=True)
st.write("### Enter a message to check if it's Spam or Ham!")

# Sidebar for instructions
st.sidebar.title("Instructions")
st.sidebar.write("1. Type or paste a message into the input box below.")
st.sidebar.write("2. Press the 'Classify' button to determine if it's Spam or Ham.")
st.sidebar.write("3. The result will be displayed below after classification.")

# Load the model and vectorizer
try:
    # Add a loading indicator for model loading
    with st.spinner('Loading the model and vectorizer...'):
        if not os.path.exists(MODEL_PATH):
            st.error(f"Model file not found at: {MODEL_PATH}")
            st.info("Please ensure you have trained the model and saved it to the correct location.")
            st.stop()
        
        if not os.path.exists(VECTORIZER_PATH):
            st.error(f"Vectorizer file not found at: {VECTORIZER_PATH}")
            st.info("Please ensure you have saved the vectorizer to the correct location.")
            st.stop()
        
        model = load_model(MODEL_PATH)
        with open(VECTORIZER_PATH, 'rb') as f:
            vectorizer = joblib.load(f)

except Exception as e:
    st.error(f"Error loading model or vectorizer: {str(e)}")
    st.stop()

# Input from the user
user_input = st.text_input("Enter your message:", "Type your message here", key="message", label_visibility="collapsed", placeholder="Type a message...")

# Input Container
with st.container():
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    if user_input.strip():
        # Classify the message
        if st.button("Classify", key="classify-btn", help="Classify the message"):
            try:
                # Preprocess the input using the vectorizer
                user_input_vectorized = vectorizer.transform([user_input])

                # Predict using the model
                prediction = model.predict(user_input_vectorized)
                result = "Spam" if prediction[0] > 0.5 else "Ham"  # Use a threshold for ANN output

                # Display the result
                result_class = "spam" if result == "Spam" else "ham"
                st.markdown(f'<div class="result {result_class}">The message is classified as: {result}</div>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"An error occurred during classification: {str(e)}")
    else:
        st.warning("Please enter a message to classify.")

st.markdown('<p class="info-text">Thank you for using this website.</p>', unsafe_allow_html=True)

