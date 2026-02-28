# 🐦 Twitter Sentiment Analysis AI

An end-to-end Machine Learning project that uses a **Deep Learning (LSTM)** model to analyze the sentiment of tweets and a Streamlit web interface for real-time predictions.

## 🚀 Features
* Neural Network: Built using TensorFlow/Keras with an LSTM (Long Short-Term Memory) architecture.
* Real-time Prediction: Enter any text in the web app and get instant sentiment classification.
* Data Processing: Handles text cleaning, tokenization, and padding for natural language processing (NLP).
* Categories: Classifies tweets into **Positive**, **Neutral**, or **Negative**.

## 🛠️ Tech Stack
* Language: Python 3.12
* Libraries: TensorFlow, Pandas, NumPy, Scikit-Learn, Streamlit
* Deployment: Streamlit Community Cloud

## 📂 Project Structure
* `twitter_sentiment_main.py`: The training script that builds the model.
* `twitter_sentiment_app.py`: The Streamlit web application.
* `sentiment_model.h5`: The saved trained neural network.
* `tokenizer.joblib`: The saved text-to-numeric dictionary.
* `requirements.txt`: List of necessary Python packages.
