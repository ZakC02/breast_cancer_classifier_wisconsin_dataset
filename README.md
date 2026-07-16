# Breast Cancer Wisconsin Dashboard

A lightweight Streamlit web app for exploring the Breast Cancer Wisconsin dataset, training a machine learning model, and reviewing evaluation metrics.

## Features
- Explore the dataset through interactive charts and tables
- Train either a Logistic Regression or XGBoost model
- View model accuracy and a confusion matrix

## Run locally
1. Open a terminal in the project root.
2. Install the Python dependencies:

```bash
pip install -r utils/requirements.txt

```

3. Start the app:

```bash
cd webapp
streamlit run Home.py
```

## Project structure
- `Home.py` — landing page for the dashboard
- `pages/` — EDA, training, and results pages
- `utils/` — data loading, preprocessing, training, and prediction helpers
- `models/` — saved trained model files

## Notes
- The app loads the dataset from the Hugging Face dataset hub at runtime.
- Model files are written to the local `models` directory after training.