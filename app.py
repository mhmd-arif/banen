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

app = Flask(__name__)
CORS(app)

nltk.download('stopwords')
nltk.download('punkt')

# Fungsi preprocessing text
def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens]
    tokens = [word for word in tokens if word.isalnum()]
    tokens = [word for word in tokens if word not in stopwords.words('indonesian')]
    return ' '.join(tokens)

# Fungsi Haversine untuk menghitung jarak antara dua koordinat
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Radius of the earth in km
    dlat = radians(lat2 - lat1)
    dlon = radians(lon1 - lon2)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

# Load dataset dan preprocess
file_path = 'datafix2.csv'
if os.path.exists(file_path):
    df = pd.read_csv(file_path, encoding='ISO-8859-1')
    df['processed_description'] = df['Description'].apply(preprocess_text)
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df['processed_description'])
else:
    df = pd.DataFrame()
    vectorizer = None

# Route untuk rekomendasi museum
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

# Route untuk mengambil semua tempat wisata
@app.route('/places', methods=['GET'])
def get_all_places():
    if df.empty:
        return jsonify({'error': 'Data not found'}), 500

    places = []
    for _, row in df.iterrows():
        places.append({
            'Place_Name': row['Place_Name'],
            'Description': row['Description'],
            'Lat': row['Lat'],     # Tambahkan koordinat Lat
            'Long': row['Long'],    # Tambahkan koordinat Long
            'Image': row['Image']  # Tambahkan ImageURL
        })
    return jsonify(places)

# Endpoint baru untuk menghitung jarak menggunakan Haversine
@app.route('/distance', methods=['POST'])
def calculate_distance():
    if df.empty:
        return jsonify({'error': 'Data not found'}), 500

    # Ambil titik awal dari request
    start_point = request.json
    lat1 = start_point['lat']
    lon1 = start_point['lon']

    # Hitung jarak ke semua tempat di dataset
    df['Distance'] = df.apply(lambda row: haversine(lat1, lon1, row['Lat'], row['Long']), axis=1)

    # Sortir berdasarkan jarak terdekat
    sorted_places = df.sort_values('Distance')

    # Buat list hasil dengan jarak
    results = []
    for _, row in sorted_places.iterrows():
        results.append({
            'Place_Name': row['Place_Name'],
            'Distance': row['Distance'],
            'Lat': row['Lat'],
            'Long': row['Long']
        })

    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)

def handler(event, context):
    return app(event, context)
