import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from joblib import load  # Import load to bring in your saved tokenizer

# 1. Load the saved model
@st.cache_resource  # This makes the app faster by only loading the model once
def load_my_model():
    return tf.keras.models.load_model('sentiment_model.h5')

model = load_my_model()

# 2. FIX: Load the EXACT tokenizer used during training
try:
    tokenizer = load('tokenizer.joblib')
except FileNotFoundError:
    st.error("Error: 'tokenizer.joblib' not found. Please run your training script first!")
    st.stop()

max_length = 100 

def predict_sentiment(text):
    # Prepare the text
    sequences = tokenizer.texts_to_sequences([text])
    # Use the same padding style used in training ('post')
    padded = pad_sequences(sequences, maxlen=max_length, padding='post')
    
    # Predict
    prediction = model.predict(padded, verbose=0)
    return prediction

# --- Streamlit UI ---
st.set_page_config(page_title="Twitter Sentiment AI", page_icon="🐦")
st.title("🐦 Twitter Sentiment Analysis App")
st.markdown("Enter a tweet below to see if the AI thinks it's **Positive**, **Neutral**, or **Negative**.")

user_input = st.text_area("What's on your mind?", placeholder="Type your tweet here...")

if st.button("Analyze Sentiment"):
    if user_input.strip():
        with st.spinner('AI is thinking...'):
            prediction = predict_sentiment(user_input)
            
            # Map result to label
            # Based on your previous run: 0=Neg, 1=Neu, 2=Pos
            labels = ["Negative", "Neutral", "Positive"]
            result = labels[np.argmax(prediction)]
            confidence = np.max(prediction) * 100

            # Display result with some styling
            if result == "Positive":
                st.success(f"Result: {result} ({confidence:.1f}% confidence)")
            elif result == "Negative":
                st.error(f"Result: {result} ({confidence:.1f}% confidence)")
            else:
                st.info(f"Result: {result} ({confidence:.1f}% confidence)")
    else:
        st.warning("Please enter some text first!")