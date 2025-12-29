# Sentiment_app

Sentiment_app is a web application designed to analyze and classify the sentiment of user-entered text as positive, negative, or neutral. It uses natural language processing (NLP) and machine learning techniques to provide insights on textual data.

## Features

- **Sentiment Analysis:** Classifies input text into positive, negative, or neutral sentiment.
- **User-friendly Interface:** Simple and intuitive UI for entering text and viewing results.
- **Real-time Feedback:** Get instant sentiment results as you type or submit text.
- **Visualization:** Graphical representation of sentiment trends (if implemented).

## Getting Started

### Prerequisites

- Python 3.x
- Required Python packages (see [requirements.txt](requirements.txt) if available)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Vivek-120604/Sentiment_app.git
   cd Sentiment_app
   ```
2. (Optional) Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

You can run the application locally:

```bash
python app.py
```

Then open your browser and navigate to `http://127.0.0.1:5000/` (or the address shown in the terminal).

## Deploy to Streamlit Cloud

1. Push this repository to GitHub (already on `main`).
2. Go to https://streamlit.io/cloud and sign in with your GitHub account.
3. Create a new app and select the `Vivek-120604/Sentiment_app` repository and the `main` branch.
4. For the **main file**, set `app.py`.
5. Leave build command empty; Streamlit Cloud will install dependencies from `requirements.txt`.

Notes & recommendations:
- This repo includes a Hugging Face `transformers` + `torch` dependency which can make the build large and longer on Streamlit Cloud. If you encounter build or memory issues, either:
- The app prefers a high-accuracy model. To avoid installing `transformers`/`torch` on Streamlit Cloud (large build), the app can call the Hugging Face Inference API. Recommended setup:
   - Add `HF_API_TOKEN` as a secret in Streamlit (Settings → Secrets) with a Hugging Face API token.
   - The app will use the HF Inference API when `HF_API_TOKEN` is present, avoiding heavy dependencies and enabling immediate click-to-use deployment.
   - If you don't provide `HF_API_TOKEN`, the app falls back to the included `simple_rnn_imdb.keras` model.

If you do want to install `transformers`/`torch` locally (not recommended for Streamlit Cloud), add them to `requirements.txt`.
- Model artifacts `simple_rnn_imdb.keras` and `word_to_index.pkl` are included in the repo so the app can run without external downloads. If you prefer hosting the model elsewhere, set `MODEL_URL` and `WORD_INDEX_URL` as Streamlit secrets or environment variables.

After creating the app, open the app URL Streamlit provides. If the build fails due to `torch` size, try the first recommendation above.

### Setting `HF_API_TOKEN` (exact steps)

1. Go to your Streamlit Cloud dashboard and open the app you created for this repo.
2. Click **Settings** (gear icon) → **Secrets** (or "Settings & secrets").
3. Add a new secret with key `HF_API_TOKEN` and value set to your Hugging Face API token (starts with `hf_...`).
4. Save the secret and click **Deploy** (or re-run) to pick up the secret in the running app.

Note: Do NOT commit your real `HF_API_TOKEN` to the repository. Use the `secrets` UI. A template is provided at `.streamlit/secrets_template.toml` for local reference.

## Project Structure

- `app.py` — Main application file.
- `templates/` — HTML templates for the web interface.
- `static/` — CSS, JS, and image files.
- `model/` — Trained sentiment analysis models (if any).
- `requirements.txt` — List of required Python packages.

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a new branch (`git checkout -b feature-name`)
3. Commit your changes
4. Push to the branch (`git push origin feature-name`)
5. Open a pull request

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

For questions, feedback, or collaboration, please contact [Vivek-120604](https://github.com/Vivek-120604).

---

*Happy Sentiment Analyzing!*