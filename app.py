
import os
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence
import pickle
import requests

# Configuration
MAX_LEN = 500
MODEL_PATH = 'simple_rnn_imdb.keras'
WORD_INDEX_PATH = 'word_to_index.pkl'

st.title('IMDB Movie Review Sentiment Analysis')

def download_file(url, dest_path):
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    with open(dest_path, 'wb') as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

@st.cache_resource
def load_resources():
    # Allow model and word index to be provided via Streamlit secrets or environment variables
    model_url = st.secrets.get('MODEL_URL') if st.secrets and 'MODEL_URL' in st.secrets else os.environ.get('MODEL_URL')
    word_index_url = st.secrets.get('WORD_INDEX_URL') if st.secrets and 'WORD_INDEX_URL' in st.secrets else os.environ.get('WORD_INDEX_URL')

    # Download model if missing and URL provided
    if not os.path.exists(MODEL_PATH) and model_url:
        try:
            download_file(model_url, MODEL_PATH)
        except Exception:
            pass

    # Download word index if missing and URL provided
    if not os.path.exists(WORD_INDEX_PATH) and word_index_url:
        try:
            download_file(word_index_url, WORD_INDEX_PATH)
        except Exception:
            pass

    model = None
    word_to_index = None
    try:
        if os.path.exists(MODEL_PATH):
            model = load_model(MODEL_PATH)
    except Exception:
        model = None
    try:
        if os.path.exists(WORD_INDEX_PATH):
            with open(WORD_INDEX_PATH, 'rb') as f:
                word_to_index = pickle.load(f)
    except Exception:
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
