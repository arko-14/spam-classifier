import streamlit as st
import os
from tensorflow.keras.models import load_model
#from tensorflow.keras.models import load_model
import joblib

# Define relative paths for model files
MODEL_PATH = os.path.join("spam_classifier", "spam_classifier_ann.h5")
VECTORIZER_PATH = os.path.join("spam_classifier", "vectorizer(4).pkl")

st.title("Spam Classifier")
st.write("Enter a message to check if it's Spam or Ham!")

# Load the model and vectorizer
try:
    # Load the TensorFlow ANN model
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
user_input = st.text_input("Message", "Type your message here")

# Classify the message
if st.button("Classify"):
    if user_input.strip():  # Ensure input is not empty
        try:
            # Preprocess the input using the vectorizer
            user_input_vectorized = vectorizer.transform([user_input])

            # Predict using the model
            prediction = model.predict(user_input_vectorized)
            result = "Spam" if prediction[0] > 0.5 else "Ham"  # Use a threshold for ANN output
            st.write(f"The message is classified as: **{result}**")
        except Exception as e:
            st.error(f"An error occurred during classification: {str(e)}")
    else:
        st.write("Please enter a valid message!")
