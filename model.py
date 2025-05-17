from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# Tiny dataset
texts = [
    "Love it, so happy!",
    "Really good experience",
    "This is awful",
    "Very disappointing"
]
labels = [1, 1, 0, 0]  # 1 = Positive, 0 = Negative

# Vectorize and train (fix: use fit_transform instead of separate transform)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)  # Fit and transform in one step
model = LogisticRegression()
model.fit(X, labels)

# Save model and vectorizer
joblib.dump(model, 'sentiment_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')