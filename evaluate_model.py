import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from sklearn.metrics import confusion_matrix, classification_report

MODEL_PATH = 'simple_rnn_imdb.keras'
MAX_FEATURES = 10000
MAX_LEN = 500

print('Loading test data...')
(_, _), (X_test, y_test) = imdb.load_data(num_words=MAX_FEATURES)
X_test = sequence.pad_sequences(X_test, maxlen=MAX_LEN)

print('Loading model...')
model = load_model(MODEL_PATH)

print('Predicting...')
probs = model.predict(X_test, verbose=0)
preds = (probs.flatten() > 0.5).astype(int)

print('Confusion matrix:')
cm = confusion_matrix(y_test, preds)
print(cm)

print('\nClassification report:')
print(classification_report(y_test, preds, digits=4))

# Show some misclassified examples (indices)
mis_idx = np.where(preds != y_test)[0]
print(f'Number of misclassified: {len(mis_idx)} / {len(y_test)}')
print('Sample misclassified indices:', mis_idx[:20])
