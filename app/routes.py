from flask import Blueprint, render_template, request
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import load

routes = Blueprint('routes', __name__)

# Load the model and vocabulary
with open('models/kbest_feature.pickle', 'rb') as f:
    kbest_feature = pickle.load(f)

model = load('models/sentiment_model.joblib')

@routes.route('/', methods=['GET', 'POST'])
def index():
    sentiment = None
    if request.method == 'POST':
        input_text = request.form['input_text']
        # Preprocess the input text
        tf_idf_vec = TfidfVectorizer(vocabulary=set(kbest_feature))
        text_vector = tf_idf_vec.fit_transform([input_text])
        # Predict sentiment
        result = model.predict(text_vector)
        sentiment = 'Sentimen Positif' if result[0] == 'positive' else 'Sentimen Negatif'
    return render_template('index.html', sentiment=sentiment)