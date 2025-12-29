import pickle
import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.callbacks import EarlyStopping

# Config
MAX_FEATURES = 10000
MAX_LEN = 500
EMBED_DIM = 128
RNN_UNITS = 128
EPOCHS = 3
BATCH_SIZE = 64
MODEL_PATH = 'simple_rnn_imdb.keras'
WORD_INDEX_PATH = 'word_to_index.pkl'

print('Loading IMDB dataset...')
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=MAX_FEATURES)
print('Padding sequences...')
X_train = sequence.pad_sequences(X_train, maxlen=MAX_LEN)
X_test = sequence.pad_sequences(X_test, maxlen=MAX_LEN)

print('Building model...')
model = Sequential()
model.add(Embedding(MAX_FEATURES, EMBED_DIM, input_length=MAX_LEN))
model.add(SimpleRNN(RNN_UNITS, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.build((None, MAX_LEN))
model.summary()

print('Preparing callbacks...')
earlystopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

print('Starting training...')
model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2, callbacks=[earlystopping])

print(f'Saving model to {MODEL_PATH}...')
model.save(MODEL_PATH)

print(f'Getting word index and saving to {WORD_INDEX_PATH}...')
words_index = imdb.get_word_index()
# Convert to a mapping consistent with how imdb sequences are encoded in Keras (shifted by 3 in decoding)
with open(WORD_INDEX_PATH, 'wb') as f:
    pickle.dump(words_index, f)

print('Done. Model and word index saved.')
