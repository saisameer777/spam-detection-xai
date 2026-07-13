"""
Explainable Real-Time Spam Detection System
Flask backend with XAI word highlighting
"""
import pickle
import re
import json
import numpy as np
from flask import Flask, request, jsonify, render_template
import nltk
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

app = Flask(__name__)

# ── Define preprocess locally (not loaded from pickle) ──
stemmer    = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text   = text.lower()
    text   = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()
    tokens = [stemmer.stem(w) for w in tokens if w not in stop_words]
    return ' '.join(tokens)

# ── Load only vectorizer and models from pickle ──
with open('models.pkl', 'rb') as f:
    data = pickle.load(f)

vectorizer = data['vectorizer']
models     = data['models']

# ── XAI: get top words contributing to prediction ──
def get_xai_words(text, model_name, label):
    model         = models[model_name]
    processed     = preprocess(text)
    vec           = vectorizer.transform([processed])
    feature_names = vectorizer.get_feature_names_out()

    if model_name == 'Naive Bayes':
        class_idx = list(model.classes_).index(label)
        log_probs  = model.feature_log_prob_[class_idx]
        scores     = (vec.toarray()[0] * log_probs)
    elif model_name == 'Logistic Regression':
        class_idx = list(model.classes_).index(label)
        coefs     = model.coef_[class_idx]
        scores    = (vec.toarray()[0] * coefs)
    else:  # SVM
        coefs  = model.coef_[0] if label == 'spam' else -model.coef_[0]
        scores = (vec.toarray()[0] * coefs)

    top_indices = np.argsort(scores)[-8:][::-1]
    top_words   = [(feature_names[i], float(scores[i]))
                   for i in top_indices if scores[i] > 0]
    return top_words[:6]

def highlight_text(text, xai_words):
    words      = text.split()
    xai_set    = {w[0].lower() for w in xai_words}
    highlighted = []
    for word in words:
        clean = re.sub(r'[^a-zA-Z]', '', word).lower()
        stem  = stemmer.stem(clean)
        if stem in xai_set or clean in xai_set:
            highlighted.append({'word': word, 'highlight': True})
        else:
            highlighted.append({'word': word, 'highlight': False})
    return highlighted

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    body       = request.get_json()
    text       = body.get('text', '').strip()
    model_name = body.get('model', 'Naive Bayes')

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    processed  = preprocess(text)
    vec        = vectorizer.transform([processed])
    model      = models[model_name]
    prediction = model.predict(vec)[0]

    if hasattr(model, 'predict_proba'):
        proba      = model.predict_proba(vec)[0]
        class_idx  = list(model.classes_).index(prediction)
        confidence = float(proba[class_idx]) * 100
    else:
        confidence = 85.0

    xai_words   = get_xai_words(text, model_name, prediction)
    highlighted = highlight_text(text, xai_words)

    return jsonify({
        'prediction':  prediction,
        'confidence':  round(confidence, 1),
        'xai_words':   xai_words,
        'highlighted': highlighted,
        'model_used':  model_name,
    })

@app.route('/compare', methods=['POST'])
def compare():
    body = request.get_json()
    text = body.get('text', '').strip()
    if not text:
        return jsonify({'error': 'No text provided'}), 400

    processed = preprocess(text)
    vec       = vectorizer.transform([processed])
    results   = {}

    for name, model in models.items():
        pred = model.predict(vec)[0]
        if hasattr(model, 'predict_proba'):
            proba     = model.predict_proba(vec)[0]
            class_idx = list(model.classes_).index(pred)
            conf      = round(float(proba[class_idx]) * 100, 1)
        else:
            conf = 85.0
        results[name] = {'prediction': pred, 'confidence': conf}

    votes   = [r['prediction'] for r in results.values()]
    final   = 'spam' if votes.count('spam') >= 2 else 'ham'
    results['Ensemble (Vote)'] = {'prediction': final, 'confidence': None}

    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
