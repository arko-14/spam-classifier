# Spam Classifier using ANN

This project is a high-performance spam classifier built using an Artificial Neural Network (ANN) with TensorFlow and Keras. The classifier distinguishes between spam and ham (non-spam) messages and achieved an accuracy of **98.19%** on the test set. The application is deployed using Streamlit, offering an interactive web interface for users.

## Overview

Spam messages are a pervasive problem in digital communications. This project aims to automatically classify incoming messages as "Spam" or "Ham" by leveraging an ANN model trained on preprocessed text data. The overall pipeline includes:
- **Text Preprocessing**: Cleaning and preparing raw messages.
- **Feature Extraction**: Converting text into numerical features using TF-IDF vectorization.
- **Model Training**: Building and training an ANN for binary classification.
- **Deployment**: Creating an interactive web application with Streamlit.

## Technologies Used

- **Python 3.8+** (tested with Python 3.12.8 locally)
- **TensorFlow & Keras** for building and training the ANN
- **Scikit-learn** for TF-IDF vectorization and additional utilities
- **Joblib** for saving and loading the vectorizer
- **Streamlit** for deploying the interactive web application
- **Pandas** and **NumPy** for data manipulation

## Model Details

- **Architecture**:
  - Input layer: Size matches the number of TF-IDF features.
  - Two hidden Dense layers with ReLU activation and Dropout for regularization.
  - Output layer: A single neuron with Sigmoid activation for binary classification.
- **Performance**: The model achieved an accuracy of **98.19%** on the test dataset.



## Usage

- **Input**: Open the app in your browser, type or paste a message in the input field.
- **Classification**: Click the "Classify" button to determine if the message is **Spam** or **Ham**.
- **Result**: The classification result is displayed prominently on the screen.

## Deployment

This project is deployedon Streamlit(frontend) and deployed on Render

## License

This project is licensed under the MIT License.

## Acknowledgements

- Inspired by real-world spam filtering challenges.
- Built as part of my continuous effort to improve machine learning and deployment skills.
