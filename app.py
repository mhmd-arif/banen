from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os
from math import radians, sin, cos, sqrt, atan2

# Ensure NLTK knows where to download data
nltk.data.path.append("/usr/local/nltk_data")

# Download necessary NLTK resources
nltk.download('punkt', download_dir='/usr/local/nltk_data')
nltk.download('stopwords', download_dir='/usr/local/nltk_data')
nltk.download('punkt_tab')  # Explicit download

app = Flask(__name__)
CORS(app)

# Preprocessing text function
def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens]
    tokens = [word for word in tokens if word.isalnum()]
    tokens = [word for word in tokens if word not in stopwords.words('indonesian')]
    return ' '.join(tokens)

# Haversine function to calculate distances between coordinates
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Radius of Earth in km
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

# Load dataset and preprocess it
file_path = 'datafix2.csv'
if os.path.exists(file_path):
    df = pd.read_csv(file_path, encoding='ISO-8859-1')
    df['processed_description'] = df['Description'].apply(preprocess_text)
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df['processed_description'])
else:
    df = pd.DataFrame()
    vectorizer = None

# Route to recommend places based on description
@app.route('/recommend', methods=['POST'])
def recommend_museum():
    if not vectorizer or df.empty:
        return jsonify({'error': 'Data not found or model not loaded'}), 500

    input_desc = request.json['description']
    input_desc_processed = preprocess_text(input_desc)
    input_vector = vectorizer.transform([input_desc_processed])
    similarities = cosine_similarity(input_vector, tfidf_matrix).flatten()
    sorted_indices = similarities.argsort()[::-1]

    results = []
    for i in sorted_indices:
        results.append({
            'Place_Name': df.iloc[i]['Place_Name'],
            'Description': df.iloc[i]['Description'],
            'Similarity_Score': similarities[i]
        })

    return jsonify(results)

# Route to fetch all places in the dataset
@app.route('/places', methods=['GET'])
def get_all_places():
    if df.empty:
        return jsonify({'error': 'Data not found'}), 500

    places = []
    for _, row in df.iterrows():
        places.append({
            'Place_Name': row['Place_Name'],
            'Description': row['Description'],
            'Lat': row['Lat'],
            'Long': row['Long'],
            'Image': row['Image']
        })
    return jsonify(places)

# Route to calculate distances using Haversine formula
@app.route('/distance', methods=['POST'])
def calculate_distance():
    if df.empty:
        return jsonify({'error': 'Data not found'}), 500

    start_point = request.json
    lat1 = start_point['lat']
    lon1 = start_point['lon']

    df['Distance'] = df.apply(lambda row: haversine(lat1, lon1, row['Lat'], row['Long']), axis=1)
    sorted_places = df.sort_values('Distance')

    results = []
    for _, row in sorted_places.iterrows():
        results.append({
            'Place_Name': row['Place_Name'],
            'Distance': row['Distance'],
            'Lat': row['Lat'],
            'Long': row['Long']
        })

    return jsonify(results)

# Run the app with Gunicorn or locally for testing
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

def handler(event, context):
    return app(event, context)
