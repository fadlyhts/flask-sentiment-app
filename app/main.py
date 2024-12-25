from flask import Flask, render_template, request
import pickle
import re
import os
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from joblib import load

# Download NLTK data
nltk.download('stopwords')

app = Flask(__name__)
app.config['SECRET_KEY'] = 'dev'

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KBEST_PATH = os.path.join(BASE_DIR, 'models', 'kbest_feature.pickle')
RF_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'random_forest_model.joblib')
KNN_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'knn_model.joblib')
NB_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'naive_bayes_model.joblib')

# Load pre-trained models
try:
    with open(KBEST_PATH, 'rb') as f:
        kbest_feature = pickle.load(f)
    models = {
        'rf': {'name': 'Random Forest', 'model': load(RF_MODEL_PATH)},
        'knn': {'name': 'KNN', 'model': load(KNN_MODEL_PATH)},
        'nb': {'name': 'Naive Bayes', 'model': load(NB_MODEL_PATH)}
    }
except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Please make sure model files exist in the models directory.")
    raise

# Initialize Sastrawi stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Load stopwords
stopwords_ind = stopwords.words('indonesian')

def casefolding(text):
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[-+]?[0-9]+', '', text)
    text = re.sub(r'[^\w\s]','', text)
    return text.strip()

def remove_stop_words(text):
    return " ".join([word for word in text.split() if word not in stopwords_ind])

def text_preprocessing(text):
    text = casefolding(text)
    text = remove_stop_words(text)
    text = stemmer.stem(text)
    return text

def predict_sentiment(input_text, model_choice='rf'):
    """Fungsi untuk memprediksi sentimen dari input teks"""
    print("\nProses Preprocessing Input Text:")
    print("1. Teks Original:", input_text)
    
    # Case folding
    case_folded = casefolding(input_text)
    print("2. Setelah Case Folding:", case_folded)
    
    # Stopword removal
    stopped = remove_stop_words(case_folded)
    print("3. Setelah Stopword Removal:", stopped)
    
    # Stemming
    stemmed = stemmer.stem(stopped)
    print("4. Setelah Stemming:", stemmed)
    
    # Get selected model
    selected_model = models[model_choice]
    
    # Transform menggunakan TF-IDF
    tf_idf = TfidfVectorizer(vocabulary=set(kbest_feature))
    text_vector = tf_idf.fit_transform([stemmed])
    
    # Predict using selected model
    prediction = selected_model['model'].predict(text_vector)[0]
    sentiment = 'Sentimen Positif' if prediction == 'Positive' else 'Sentimen Negatif'
    
    return {
        'steps': {
            'original': input_text,
            'case_folded': case_folded,
            'stopped': stopped,
            'stemmed': stemmed
        },
        'final_text': stemmed,
        'sentiment': sentiment,
        'model_name': selected_model['name']
    }

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        text = request.form['text']
        model_choice = request.form.get('model', 'rf')
        
        try:
            # Get prediction and preprocessing steps
            prediction_result = predict_sentiment(text, model_choice)
            
            # Format the result
            result = f"""
            <h3>Proses Preprocessing Input Text:</h3>
            <p>1. Teks Original: {prediction_result['steps']['original']}</p>
            <p>2. Setelah Case Folding: {prediction_result['steps']['case_folded']}</p>
            <p>3. Setelah Stopword Removal: {prediction_result['steps']['stopped']}</p>
            <p>4. Setelah Stemming: {prediction_result['steps']['stemmed']}</p>
            
            <h3>Hasil Akhir:</h3>
            <p>Model yang digunakan: {prediction_result['model_name']}</p>
            <p>Text Preprocessing: {prediction_result['final_text']}</p>
            <p>Hasil prediksi untuk "{text}" adalah {prediction_result['sentiment']}</p>
            """
            
        except Exception as e:
            result = f"Error in processing: {str(e)}"
    
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
