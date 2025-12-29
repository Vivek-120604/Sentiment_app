import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence
import pickle

# Configuration
MAX_LEN = 500
MODEL_PATH = 'simple_rnn_imdb.keras'
WORD_INDEX_PATH = 'word_to_index.pkl'

st.title('IMDB Movie Review Sentiment Analysis')

@st.cache_resource
def load_resources():
    try:
        model = load_model(MODEL_PATH)
    except Exception as e:
        model = None
    try:
        with open(WORD_INDEX_PATH, 'rb') as f:
            word_to_index = pickle.load(f)
    except Exception as e:
        word_to_index = None
    return model, word_to_index

model, word_to_index = load_resources()

if model is None or word_to_index is None:
    st.error('Model or word index not found. Ensure `simple_rnn_imdb.keras` and `word_to_index.pkl` exist in the app directory.')
else:
    st.write('Enter a movie review to predict its sentiment (Positive or Negative).')
    user_input = st.text_area('Enter your review here:')
    if st.button('Predict'):
        if user_input.strip():
            words = user_input.lower().split()
            indexed = [word_to_index.get(w, 3) for w in words]
            padded = sequence.pad_sequences([indexed], maxlen=MAX_LEN)
            pred = model.predict(padded, verbose=0)
            score = float(pred[0][0])
            sentiment = 'Positive' if score > 0.5 else 'Negative'
            st.write(f'Prediction score: {score:.4f}')
            st.write(f'Predicted sentiment: {sentiment}')
        else:
            st.write('Please enter a review to predict.')
