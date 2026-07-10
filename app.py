from flask import Flask, render_template, request
import joblib
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os

# -----------------------------
# Initialize Flask App
# -----------------------------
app = Flask(__name__)

# -----------------------------
# Download NLTK resources
# -----------------------------
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet")

try:
    nltk.data.find("corpora/omw-1.4")
except LookupError:
    nltk.download("omw-1.4")

# -----------------------------
# Load Models
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

category_model = joblib.load(
    os.path.join(BASE_DIR, "models", "category_model.pkl")
)

priority_model = joblib.load(
    os.path.join(BASE_DIR, "models", "priority_model.pkl")
)

tfidf = joblib.load(
    os.path.join(BASE_DIR, "models", "tfidf_vectorizer.pkl")
)

# -----------------------------
# NLP Tools
# -----------------------------
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


# -----------------------------
# Text Cleaning Function
# -----------------------------
def clean_text(text):

    text = text.lower()

    text = re.sub(r"http\S+|www\S+", "", text)

    text = re.sub(r"\d+", "", text)

    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    text = re.sub(r"\s+", " ", text).strip()

    words = text.split()

    words = [
        word
        for word in words
        if word not in stop_words
    ]

    words = [
        lemmatizer.lemmatize(word)
        for word in words
    ]

    return " ".join(words)


# -----------------------------
# Home Page
# -----------------------------
@app.route("/")
def home():

    return render_template("index.html")


# -----------------------------
# Prediction Route
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():

    subject = request.form.get("subject", "")

    description = request.form.get("description", "")

    text = subject + " " + description

    cleaned = clean_text(text)

    vector = tfidf.transform([cleaned])

    category = category_model.predict(vector)[0]

    priority = priority_model.predict(vector)[0]

    recommendation = ""

    if priority.lower() == "high":
        recommendation = "Immediate attention required."

    elif priority.lower() == "medium":
        recommendation = "Assign to the respective support team."

    else:
        recommendation = "Can be handled in the normal support queue."

    return render_template(
        "index.html",
        subject=subject,
        description=description,
        category=category,
        priority=priority,
        recommendation=recommendation
    )


# -----------------------------
# Run Application
# -----------------------------
if __name__ == "__main__":
    app.run(
        debug=False,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 7860))
    )
