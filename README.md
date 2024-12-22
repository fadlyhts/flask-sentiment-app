# Flask Sentiment Analysis Web Application

This project is a web application built using Flask that classifies input text into positive or negative sentiment. It utilizes a pre-trained sentiment analysis model and a vocabulary file for feature extraction.

## Project Structure

```
flask-sentiment-app
├── app
│   ├── __init__.py
│   ├── routes.py
│   ├── static
│   │   └── style.css
│   └── templates
│       ├── base.html
│       └── index.html
├── models
│   ├── kbest_feature.pickle
│   ├── random_forest_model.joblib
│   ├── knn_model.joblib
│   └── naive_bayes_model.joblib
├── app.py
├── requirements.txt
└── README.md
```

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd flask-sentiment-app
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Running the Application

1. Start the Flask application:
   ```
   python app.py
   ```

2. Open your web browser and go to `http://127.0.0.1:5000`.

## Usage

- Enter the text you want to analyze in the input field.
- Click the "Analyze" button to see the sentiment classification (Positive or Negative).

## Dependencies

- Flask
- scikit-learn
- pandas
- numpy
- Sastrawi
- nltk

## License

This project is licensed under the MIT License.
