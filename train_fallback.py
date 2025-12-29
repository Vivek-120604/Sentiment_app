import pickle
from tensorflow.keras.datasets import imdb
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

MAX_FEATURES = 10000

def decode_sequences(sequences, word_index):
    reverse_index = { (i+3): w for w,i in word_index.items() }
    reverse_index[0] = '<PAD>'
    reverse_index[1] = '<START>'
    reverse_index[2] = '<UNK>'
    texts = []
    for seq in sequences:
        words = [reverse_index.get(i, '?') for i in seq if i != 0]
        texts.append(' '.join(words))
    return texts

def main():
    print('Loading IMDB dataset...')
    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=MAX_FEATURES)
    with open('word_to_index.pkl','rb') as f:
        raw = pickle.load(f)
    print('Decoding sequences to raw text (may be lossy)...')
    train_texts = decode_sequences(X_train, raw)
    test_texts = decode_sequences(X_test, raw)

    print('Vectorizing with TF-IDF...')
    vect = TfidfVectorizer(max_features=20000, ngram_range=(1,2))
    Xv = vect.fit_transform(train_texts)

    print('Training LogisticRegression fallback...')
    clf = LogisticRegression(max_iter=1000)
    clf.fit(Xv, y_train)

    print('Saving fallback artifacts...')
    with open('fallback_vect.pkl','wb') as f:
        pickle.dump(vect, f)
    with open('fallback_clf.pkl','wb') as f:
        pickle.dump(clf, f)

    print('Done. Fallback trained.')

if __name__ == '__main__':
    main()
