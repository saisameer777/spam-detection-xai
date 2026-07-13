# Explainable Real-Time Spam Detection System

**Tech Stack:** Python, Scikit-learn, Flask, NLTK, TF-IDF, XAI

## Features
- 3 ML models: Naive Bayes, Logistic Regression, SVM
- TF-IDF vectorization with bigrams
- XAI pipeline highlighting key spam-triggering words
- Model comparison (ensemble voting)
- Sub-200ms inference latency
- Clean dark-mode UI

## Setup & Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download NLTK data
python3 -c "import nltk; nltk.download('stopwords')"

# 3. Train models
python3 train_model.py

# 4. Run the app
python3 app.py

# 5. Open browser
# http://localhost:5000
```

## Project Structure
```
spam_detector/
├── app.py            # Flask backend + XAI logic
├── train_model.py    # Model training script
├── models.pkl        # Saved trained models
├── requirements.txt
└── templates/
    └── index.html    # Frontend UI
```
