from flask import Flask, request, jsonify
from flask_cors import CORS  # Tambahkan Flask-CORS untuk menangani CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os

app = Flask(__name__)
CORS(app)  # Enable CORS agar request dari React bisa diterima

# Pastikan NLTK resources sudah di-download
nltk.download('stopwords')
nltk.download('punkt')

# Fungsi preprocessing text
def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens]
    tokens = [word for word in tokens if word.isalnum()]
    tokens = [word for word in tokens if word not in stopwords.words('indonesian')]
    return ' '.join(tokens)

# Load dataset dan preprocess
file_path = 'yogyakarta_budaya.csv'
if os.path.exists(file_path):
    df = pd.read_csv(file_path)
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
    top_indices = similarities.argsort()[-3:][::-1]
    
    results = []
    for i in top_indices:
        results.append({
            'Place_Name': df.iloc[i]['Place_Name'],
            'Similarity_Score': similarities[i]
        })
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
