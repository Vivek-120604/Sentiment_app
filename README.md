
   
# Sentiment_app — Technical Overview

This repository demonstrates practical approaches to text sentiment classification using lightweight and experimental models. It contains a saved Keras-based RNN model for offline inference, scripts for training/evaluation, and small experiments with transformer-based inference.

This README focuses on the algorithms, preprocessing, training techniques, inference modes, and how the included scripts use them.

## Algorithms & Techniques used

- Embedding layers: maps integer token ids to dense vectors (learned embeddings) to represent words in a low-dimensional continuous space.
- Recurrent neural networks (RNNs): a compact sequence model (SimpleRNN / GRU / LSTM family) used in `simple_rnn_imdb.keras` to model order and temporal dependencies in text sequences.
- Sequence preprocessing: tokenization, integer encoding (word→index), sequence truncation/padding to fixed length, and optional out-of-vocabulary handling.
- Word-index mapping: a `word_to_index` style lookup is used to encode raw text into model-ready sequences.
- Training methods: cross-entropy loss for classification, optimizer (e.g., Adam), early stopping and model checkpointing to avoid overfitting, and class-balanced sampling for skewed datasets if required.
- Evaluation metrics: accuracy, precision, recall, F1-score, and confusion matrices computed by `evaluate_model.py` on held-out test data.
- Model persistence: Keras `.keras` model saving and loading for reproducible inference and deployment.
- Transformer experiments: lightweight calls and test harnesses in `test_transformer.py` demonstrating how to swap in transformer encoders (from `transformers`) for higher accuracy when resources permit.
- External inference option: integration with the Hugging Face Inference API for using large transformer models without local `transformers`/`torch` installs (controlled via `HF_API_TOKEN`).

## Files and their roles

- `app.py` — main web UI / demo entry. May be implemented with Streamlit or Flask to let users enter text and view predicted sentiment.
- `debug_inference.py` — small script to run quick local inferences against the included model; useful for sanity checks and examples.
- `evaluate_model.py` — runs evaluation on sample/test datasets; reports accuracy and other metrics and can save confusion matrices.
- `train_and_save.py` — training script that builds the model (embedding + RNN layers), trains on a dataset (e.g., IMDB), and saves the trained model to `simple_rnn_imdb.keras`.
- `train_fallback.py` — alternate training/fallback configuration (smaller model or alternate hyperparameters) to provide a lightweight option for constrained environments.
- `test_transformer.py` — experimental transformer-based example demonstrating how to switch from an RNN to a transformer encoder for inference/training.
- `simple_rnn_imdb.keras` — included trained Keras model for offline inference.
- `requirements.txt` — Python dependencies used by the project.
- `sample_data/` — small CSVs and datasets for quick experiments and evaluation.

## Typical data preprocessing pipeline

1. Normalize text (lowercasing, optional punctuation removal).
2. Tokenize text into words or subwords.
3. Map tokens to integers via a `word_index` dictionary (unknown tokens map to an OOV id).
4. Pad or truncate sequences to a fixed `max_len` (e.g. 200 tokens).
5. Batch and feed into model; embedding layer performs lookup to dense vectors and the RNN consumes the sequence.

Code sketch (preprocessing):

```python
# tokenize + map to ints
sequence = [word_index.get(w, oov_id) for w in tokenizer(text)]
sequence = pad_sequences([sequence], maxlen=MAX_LEN)
pred = model.predict(sequence)
```

## How to run (commands)

Install dependencies and create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Run the app locally (Streamlit or Flask depending on `app.py` implementation):

```bash
python app.py
# or if app uses Streamlit
streamlit run app.py
```

Quick inference example:

```bash
python debug_inference.py --text "I loved the movie, it was fantastic"
```

Evaluate the model on sample data:

```bash
python evaluate_model.py --data sample_data/mnist_test.csv
```

Train a new model (will create a new saved model file):

```bash
python train_and_save.py --epochs 10 --batch-size 64
```

## Inference modes (local vs remote)

- Local Keras model: the default path if `simple_rnn_imdb.keras` is present — fast and offline but limited by model capacity.
- Transformer / HF API: for higher accuracy, use `transformers` locally (requires `torch`), or call Hugging Face Inference API by setting `HF_API_TOKEN`.

To use Hugging Face Inference API (no heavy local dependencies):

```bash
export HF_API_TOKEN="hf_..."
```

Then the app will route requests to the HF inference endpoint when the token is present.

## Training notes & best practices

- Use a validation split and early stopping to prevent overfitting.
- Save the `word_index`/tokenizer together with the model so inference preserves the same encoding.
- Track experiments (hyperparameters, random seed) for reproducibility.
- For small datasets, prefer lower-capacity models (small RNN or lightweight transformer) to avoid overfitting.

## Metrics and evaluation

- Primary metrics: accuracy, precision, recall, F1.
- Use confusion matrices to inspect per-class errors and tune class weights if necessary.

## Extending the project

- Swap the embedding + RNN stack for a transformer encoder in `test_transformer.py` to compare performance.
- Add a `Dockerfile` for reproducible deployments.
- Add a unit test suite and CI pipeline to run `evaluate_model.py` and `debug_inference.py` in CI.

## Security & secrets

- Never commit `HF_API_TOKEN` to the repo. Use Streamlit secrets or environment variables for deployments.

## Where to look next

- Try `python debug_inference.py` to confirm local inference works.
- Run `python evaluate_model.py` to see metrics on the sample data.
- Inspect `train_and_save.py` to see the exact model architecture and hyperparameters used to create `simple_rnn_imdb.keras`.

---

_This README replaces the previous high-level summary with a concise technical reference describing algorithms and usage._