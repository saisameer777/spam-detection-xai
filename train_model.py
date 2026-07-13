"""
Train spam detection models and save them.
Uses SMS Spam Collection dataset format.
"""
import pickle
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# ── Sample training data (spam + ham) ──
# In production, replace with full SMS Spam Collection dataset
data = [
    # SPAM
    ("Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005", "spam"),
    ("WINNER!! You have been selected to receive a $1000 Walmart Gift Card. Click here now!", "spam"),
    ("Congratulations! You've won a free iPhone. Claim your prize now at freeprize.com", "spam"),
    ("URGENT: Your account has been suspended. Verify immediately or lose access forever.", "spam"),
    ("You have won 1 million dollars. Send your bank details to claim.", "spam"),
    ("FREE ringtones! Text YES to 12345 to get free ringtones every week!", "spam"),
    ("Claim your free vacation package now. Limited time offer expires tonight!", "spam"),
    ("ALERT: Suspicious activity detected. Click here to secure your account immediately.", "spam"),
    ("You are selected for a cash prize of $5000. Reply NOW to claim!", "spam"),
    ("Get rich quick! Earn $500 per day working from home. No experience needed!", "spam"),
    ("FINAL NOTICE: Your loan application approved. Get $10000 instantly. Call now!", "spam"),
    ("Win a brand new BMW! Enter our free competition. Text CAR to 80888.", "spam"),
    ("Your mobile number has been awarded a prize. Call 09061743386 to claim.", "spam"),
    ("FREE MESSAGE: Congratulations, you have won a 2 week holiday to Benidorm!", "spam"),
    ("REMINDER: You have a PENDING transaction. Click the link to verify your details.", "spam"),
    ("Hot singles in your area are waiting to meet you! Join now for FREE!", "spam"),
    ("Buy cheap meds online! No prescription needed. Lowest prices guaranteed!", "spam"),
    ("Your account will be closed unless you verify your information. Act now!", "spam"),
    ("Limited time: 90% off all items. Shop now before stocks run out!", "spam"),
    ("You've been chosen for our exclusive rewards program. Claim $500 today!", "spam"),
    ("Double your income in 30 days with our proven investment strategy!", "spam"),
    ("FREE credit check! See your score instantly. No obligation required.", "spam"),
    ("You owe $0 in taxes! Get your refund now by clicking this secure link.", "spam"),
    ("Congratulations dear customer, you have won the weekly prize draw!", "spam"),
    ("Send this to 10 friends and receive $100 in your PayPal account instantly!", "spam"),
    # HAM
    ("Hey, are you coming to the party tonight?", "ham"),
    ("Can you please send me the report by end of day?", "ham"),
    ("I will be late for dinner tonight, please start without me.", "ham"),
    ("The meeting has been rescheduled to 3pm on Thursday.", "ham"),
    ("Happy birthday! Hope you have a wonderful day.", "ham"),
    ("Can we reschedule our call to tomorrow morning?", "ham"),
    ("I'm on my way, be there in 10 minutes.", "ham"),
    ("Did you finish the assignment? Let me know if you need help.", "ham"),
    ("Thanks for the update. I'll review and get back to you.", "ham"),
    ("Let's grab lunch tomorrow, I know a great new place.", "ham"),
    ("Please remember to submit your timesheet by Friday.", "ham"),
    ("I loved the movie last night! We should watch the sequel together.", "ham"),
    ("Can you pick up some groceries on your way home?", "ham"),
    ("The project deadline has been extended to next week.", "ham"),
    ("Good morning! How are you doing today?", "ham"),
    ("Don't forget we have a dentist appointment at 2pm.", "ham"),
    ("I sent you an email with the details. Please check when you get a chance.", "ham"),
    ("The kids are asking when you'll be home. They miss you!", "ham"),
    ("Thank you for your help yesterday, I really appreciate it.", "ham"),
    ("Can you review my code and let me know your thoughts?", "ham"),
    ("Looking forward to seeing you at the conference next week.", "ham"),
    ("I've attached the invoice to this email. Please process at your earliest.", "ham"),
    ("Do you want to study together at the library this evening?", "ham"),
    ("Just checking in to see how the project is progressing.", "ham"),
    ("Your package has been delivered and left at the front door.", "ham"),
    ("The flight is confirmed for Saturday. See you at the airport!", "ham"),
    ("I'll be working from home tomorrow if you need to reach me.", "ham"),
    ("Reminder: Team standup at 9am tomorrow. Please be on time.", "ham"),
    ("Great job on the presentation today! The client was very impressed.", "ham"),
    ("Can you send me the login credentials for the new system?", "ham"),
]

texts = [d[0] for d in data]
labels = [d[1] for d in data]

# ── Preprocessing ──
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()
    tokens = [stemmer.stem(w) for w in tokens if w not in stop_words]
    return ' '.join(tokens)

texts_clean = [preprocess(t) for t in texts]

X_train, X_test, y_train, y_test = train_test_split(
    texts_clean, labels, test_size=0.2, random_state=42, stratify=labels
)

# ── Train models ──
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)

models = {
    'Naive Bayes':        MultinomialNB(alpha=0.1),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'SVM':                LinearSVC(max_iter=1000, random_state=42),
}

trained = {}
print("\n── Model Performance ──")
for name, model in models.items():
    model.fit(X_train_vec, y_train)
    preds = model.predict(X_test_vec)
    acc = accuracy_score(y_test, preds)
    trained[name] = model
    print(f"{name}: {acc*100:.1f}% accuracy")

# ── Save ──
with open('models.pkl', 'wb') as f:
    pickle.dump({'vectorizer': vectorizer, 'models': trained, 'preprocess': preprocess}, f)

print("\nModels saved to models.pkl")
