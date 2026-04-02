import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Sample data
data = {
    "text": ["Breaking news: market crashes", "Celebrity gossip goes viral"],
    "label": [0, 1]  # 0 = Real, 1 = Fake
}

df = pd.DataFrame(data)

# Features
X = df["text"]
y = df["label"]

# Convert text to numbers
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

# Train model
model = LogisticRegression()
model.fit(X, y)

print("Model trained successfully!")

# 🔥 Take user input
news = input("Enter a news headline: ")

# Transform input
news_vector = vectorizer.transform([news])

# Predict
prediction = model.predict(news_vector)

if prediction[0] == 0:
    print("This news is REAL ✅")
else:
    print("This news is FAKE ❌")