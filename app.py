import gradio as gr
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence
import pickle

# Define max_len
max_len = 500

# Load the saved model
model = load_model('simple_rnn_imdb_lstm_tanh.keras')

# Load the word index
with open('indexed_word_dictionary.pkl', 'rb') as f:
    word_to_index = pickle.load(f)

def preprocess_text(text, word_to_index, max_len):
   words = text.lower().split()
   encoded_review = [word_to_index.get(word, 3) for word in words]
   padded_review = sequence.pad_sequences([encoded_review], maxlen=max_len)
   return padded_review

def predict_sentiment(text_input):
    preprocessed_text = preprocess_text(text_input, word_to_index, max_len)
    prediction = model.predict(preprocessed_text, verbose=0) # Added verbose=0 to suppress output
    sentiment = 'Positive' if prediction[0] >= 0.5 else 'Negative'
    return sentiment

# Create the Gradio interface
iface = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=5, placeholder='Enter your movie review here...'),
    outputs='text',
    title='IMDB Movie Review Sentiment Analysis with LSTM (Tanh)',
    description='Enter a movie review to classify its sentiment as Positive or Negative using an LSTM model with Tanh activation.'
)

# Launch the Gradio app
iface.launch(share=True)
