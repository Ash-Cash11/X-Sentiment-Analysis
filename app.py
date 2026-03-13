from flask import Flask, render_template, request, redirect, url_for, session, flash
import pandas as pd
import os
import re
import nltk
import matplotlib.pyplot as plt
from fpdf import FPDF

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ----------------------------------------
# APP CONFIG
# ----------------------------------------

app = Flask(__name__)
app.secret_key = "super_secret_key"

nltk.download("stopwords")
nltk.download("wordnet")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# Demo User
USER = {
    "username": "admin",
    "password": "admin123"
}

# ----------------------------------------
# TEXT CLEANING
# ----------------------------------------

def clean_tweet(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)

    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]

    return " ".join(words)

# ----------------------------------------
# TRAIN MODEL ON STARTUP
# ----------------------------------------

def train_model():
    print("Training model...")

    df = pd.read_csv(
        "data/sentiment140.csv",
        encoding="latin-1",
        header=None,
        sep=",",
        usecols=[0, 5],
        names=["sentiment", "tweet"]
    )

    df["sentiment"] = df["sentiment"].map({0: 0, 4: 1})

    df = pd.concat([
        df[df["sentiment"] == 0].sample(20000, random_state=42),
        df[df["sentiment"] == 1].sample(20000, random_state=42)
    ])

    df["cleaned"] = df["tweet"].apply(clean_tweet)

    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df["cleaned"])
    y = df["sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    print("Model Accuracy:", accuracy_score(y_test, model.predict(X_test)))
    print("Model ready.\n")

    return model, vectorizer

model, vectorizer = train_model()

# ----------------------------------------
# LOGIN
# ----------------------------------------

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        if username == USER["username"] and password == USER["password"]:
            session["user"] = username
            return redirect(url_for("index"))
        else:
            flash("Invalid Credentials! Use admin / admin123")

    return render_template("login.html")

@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))

# ----------------------------------------
# MAIN DASHBOARD
# ----------------------------------------

@app.route("/", methods=["GET", "POST"])
def index():

    if "user" not in session:
        return redirect(url_for("login"))

    if request.method == "POST":

        file = request.files.get("file")

        if not file or file.filename == "":
            flash("Please upload a CSV file.")
            return redirect(url_for("index"))

        df = pd.read_csv(file)

        if "tweet" not in df.columns:
            flash("CSV must contain a column named 'tweet'")
            return redirect(url_for("index"))

        df["cleaned"] = df["tweet"].apply(clean_tweet)
        X = vectorizer.transform(df["cleaned"])
        probs = model.predict_proba(X)

        sentiments = []
        confidences = []

        for prob in probs:
            neg, pos = prob
            confidence = max(neg, pos)

            if abs(neg - pos) < 0.15:
                sentiments.append("Neutral")
            elif neg > pos:
                sentiments.append("Negative")
            else:
                sentiments.append("Positive")

            confidences.append(round(confidence * 100, 2))

        df["sentiment"] = sentiments
        df["confidence"] = confidences

        os.makedirs("static/results", exist_ok=True)
        csv_path = "static/results/analyzed_output.csv"
        df.to_csv(csv_path, index=False)

        counts = df["sentiment"].value_counts()
        negative = counts.get("Negative", 0)
        neutral = counts.get("Neutral", 0)
        positive = counts.get("Positive", 0)
        total = negative + neutral + positive

        os.makedirs("static/charts", exist_ok=True)

        plt.figure()
        plt.bar(["Negative", "Neutral", "Positive"],
                [negative, neutral, positive])
        bar_chart = "static/charts/bar.png"
        plt.savefig(bar_chart)
        plt.close()

        plt.figure()
        plt.pie([negative, neutral, positive],
                labels=["Negative", "Neutral", "Positive"],
                autopct="%1.1f%%")
        pie_chart = "static/charts/pie.png"
        plt.savefig(pie_chart)
        plt.close()

        # PDF Export
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=14)
        pdf.cell(200, 10, txt="Sentiment Analysis Report", ln=True, align="C")
        pdf.cell(200, 10, txt=f"Total: {total}", ln=True)
        pdf.cell(200, 10, txt=f"Negative: {negative}", ln=True)
        pdf.cell(200, 10, txt=f"Neutral: {neutral}", ln=True)
        pdf.cell(200, 10, txt=f"Positive: {positive}", ln=True)

        pdf_path = "static/results/report.pdf"
        pdf.output(pdf_path)

        preview = df.head(5).to_dict(orient="records")

        return render_template("index.html",
                               total=total,
                               negative=negative,
                               neutral=neutral,
                               positive=positive,
                               bar_chart=bar_chart,
                               pie_chart=pie_chart,
                               preview=preview,
                               download_path=csv_path,
                               pdf_path=pdf_path)

    return render_template("index.html")

# ----------------------------------------

if __name__ == "__main__":
    app.run(debug=True)