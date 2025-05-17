from flask import Flask, request, render_template
import joblib
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)

# Load pre-trained model and vectorizer
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    text = ''
    if request.method == 'POST':
        text = request.form['text'].strip()
        if text:
            text_vector = vectorizer.transform([text])
            prediction = 'Positive' if model.predict(text_vector)[0] == 1 else 'Negative'
    return render_template('index.html', prediction=prediction, text=text)

if __name__ == '__main__':
    app.run(debug=True)
        