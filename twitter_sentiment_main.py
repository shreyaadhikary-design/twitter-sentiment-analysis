import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from joblib import dump

# --- FIX 1: Load the dataset safely ---
try:
    data = pd.read_csv('Twitter_Data.csv')
except FileNotFoundError:
    print("Error: Twitter_Data.csv not found in this folder!")
    exit()

# --- FIX 2 (Line 21): Added 'r' and 'regex=True' to stop the \s warning ---
data['clean_text'] = data['selected_text'].astype(str)
data['clean_text'] = data['clean_text'].str.replace(r'[^a-zA-Z\s]', '', regex=True).str.lower()

# Check unique values
unique_sentiments = data['sentiment'].unique()
print("Unique Sentiments found:", unique_sentiments)

# 1. Drop any rows that have empty text or empty sentiment labels
data = data.dropna(subset=['clean_text', 'sentiment'])

# 2. Ensure everything in clean_text is definitely a string
X = data['clean_text'].astype(str)

# 3. CONVERT WORDS TO NUMBERS (Fixes the ValueError: 'positive')
# This turns 'negative', 'neutral', 'positive' into 0, 1, 2
le = LabelEncoder()
y = le.fit_transform(data['sentiment'].astype(str)) 

# Print a check to see which number belongs to which word
print("Label Mapping:", dict(zip(le.classes_, le.transform(le.classes_))))

# 4. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Tokenize and pad
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)
X_train_pad = pad_sequences(X_train_seq, maxlen=100, padding='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=100, padding='post')

# 6. One-hot encode labels (Now y is numeric, so this will work!)
num_classes = 3 
y_train_onehot = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
y_test_onehot = tf.keras.utils.to_categorical(y_test, num_classes=num_classes)

# Build LSTM model
model = tf.keras.Sequential([
    Embedding(input_dim=5000, output_dim=100),
    LSTM(128),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
print("Starting training now...")
model.fit(X_train_pad, y_train_onehot, epochs=5, batch_size=32, validation_data=(X_test_pad, y_test_onehot))

# Save the results
model.save('sentiment_model.h5')
dump(tokenizer, 'tokenizer.joblib')
print("Finished! Model saved as sentiment_model.h5")