
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

# We'll prefer using the Hugging Face Inference API on Streamlit Cloud
# (so we don't need to install heavy `transformers`/`torch` there).
HF_API_TOKEN = None
if st.secrets and 'HF_API_TOKEN' in st.secrets:
    HF_API_TOKEN = st.secrets['HF_API_TOKEN']
else:
    HF_API_TOKEN = os.environ.get('HF_API_TOKEN')

def hf_inference_api_call(text, token):
    """Call Hugging Face Inference API for sentiment. Returns (label, score) or raises."""
    headers = {"Authorization": f"Bearer {token}"}
    api_url = 'https://api-inference.huggingface.co/models/distilbert-base-uncased-finetuned-sst-2-english'
    payload = {"inputs": text}
    r = requests.post(api_url, headers=headers, json=payload, timeout=30)
    r.raise_for_status()
    data = r.json()
    # Example response: [{'label':'POSITIVE','score':0.9998}]
    if isinstance(data, dict) and data.get('error'):
        raise RuntimeError(data.get('error'))
    if not isinstance(data, list) or len(data) == 0:
        raise RuntimeError('Invalid HF response')
    label = data[0].get('label')
    score = float(data[0].get('score', 0.0))
    return label, score

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
                raw_index = pickle.load(f)
            # Keras' imdb sequences reserve indices 0-3 for special tokens.
            # The mapping returned by `imdb.get_word_index()` maps words -> integer starting at 1.
            # To match sequences used during training we must shift these by +3.
            word_to_index = {w: (i + 3) for w, i in raw_index.items()}
            word_to_index["<PAD>"] = 0
            word_to_index["<START>"] = 1
            word_to_index["<UNK>"] = 2
    except Exception:
        word_to_index = None
    return model, word_to_index

model, word_to_index = load_resources()

hf_sentiment = None

if model is None or word_to_index is None:
    st.error('Model or word index not found. Ensure `simple_rnn_imdb.keras` and `word_to_index.pkl` exist in the app directory.')
else:
    st.write('Enter a movie review to predict its sentiment (Positive or Negative).')
    user_input = st.text_area('Enter your review here:')
    if st.button('Predict'):
        if not user_input.strip():
            st.write('Please enter a review to predict.')
        else:
            # If HF API token is provided in secrets/env, use the Inference API (no heavy deps)
            if HF_API_TOKEN:
                try:
                    label, score = hf_inference_api_call(user_input[:1000], HF_API_TOKEN)
                    sentiment = 'Positive' if label.upper().startswith('POS') else 'Negative'
                    st.write(f'Prediction score: {score:.4f} (model: DistilBERT via HF Inference API)')
                    st.write(f'Predicted sentiment: {sentiment}')
                    hf_fallback = False
                except Exception as e:
                    st.write('HF Inference API failed — falling back to local Keras model.')
                    hf_fallback = True
            else:
                hf_fallback = True

            if hf_fallback:
                # simple tokenization: keep alphanumerics and apostrophes
                import re
                tokens = re.findall(r"\w+'?\w+|\w+", user_input.lower())
                # map tokens to shifted indices; unknown -> 2 (`<UNK>`)
                indexed = [word_to_index.get(w, 2) for w in tokens]
                padded = sequence.pad_sequences([indexed], maxlen=MAX_LEN)
                pred = model.predict(padded, verbose=0)
                score = float(pred[0][0])
                sentiment = 'Positive' if score > 0.5 else 'Negative'
                st.write(f'Prediction score: {score:.4f} (model: Keras SimpleRNN)')
                st.write(f'Predicted sentiment: {sentiment}')
