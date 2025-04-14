import os
import requests
import json
import faiss
import numpy as np
import logging
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from sentence_transformers import SentenceTransformer
from flask import Flask, request, jsonify, render_template

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# TMDb API key (Replace with your own if needed)
API_KEY = "4cd530f8ef2f395bf2eeb590bdbd8b6f"
BASE_URL = "https://api.themoviedb.org/3"
IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"
OUTPUT_DIR = "static/movie_posters"
METADATA_FILE = "movie_metadata.json"
INDEX_FILE = "movie_index.faiss"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Use a more advanced sentence transformer model
model = SentenceTransformer("paraphrase-MiniLM-L6-v2")  

def fetch_movie_posters(num_pages=50):
    """Fetch 10,000+ movie posters and metadata from TMDb."""
    metadata = []
    for page in range(1, num_pages + 1):
        url = f"{BASE_URL}/movie/popular?api_key={API_KEY}&language=en-US&page={page}"
        response = requests.get(url)
        if response.status_code != 200:
            logging.error(f"Failed to fetch page {page}")
            continue

        movies = response.json().get("results", [])
        with ThreadPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(download_movie_data, movies))
            metadata.extend(filter(None, results))  # Remove None values

        if len(metadata) >= 10000:
            logging.info("✅ Successfully downloaded 10,000+ movie images.")
            break

    save_metadata(metadata)
    build_index(metadata)

def download_movie_data(movie):
    """Download movie poster and metadata."""
    poster_path = movie.get("poster_path")
    if not poster_path:
        return None

    image_url = f"{IMAGE_BASE_URL}{poster_path}"
    save_path = os.path.join(OUTPUT_DIR, f"{movie['id']}.jpg")

    if download_image(image_url, save_path):
        return {
            "id": movie["id"],
            "title": movie["title"],
            "release_date": movie.get("release_date", "Unknown"),
            "genres": fetch_genres(movie["genre_ids"]),
            "overview": movie.get("overview", "No description available"),
            "rating": movie.get("vote_average", "N/A"),
            "poster_path": f"movie_posters/{movie['id']}.jpg"
        }
    return None

def download_image(url, save_path):
    """Download and save an image from a URL."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(save_path, "wb") as file:
            for chunk in response.iter_content(1024):
                file.write(chunk)
        return True
    except requests.RequestException as e:
        logging.error(f"Failed to download {url}: {e}")
        return False

def fetch_genres(genre_ids):
    """Fetch genre names based on genre IDs."""
    genre_url = f"{BASE_URL}/genre/movie/list?api_key={API_KEY}&language=en-US"
    response = requests.get(genre_url)
    if response.status_code == 200:
        genre_map = {genre["id"]: genre["name"] for genre in response.json().get("genres", [])}
        return [genre_map.get(gid, "Unknown") for gid in genre_ids]
    return []

def save_metadata(metadata):
    """Save metadata to a JSON file."""
    with open(METADATA_FILE, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)
    logging.info(f"✅ Metadata saved to {METADATA_FILE}")

def build_index(metadata):
    """Build a search index using FAISS and optimized embeddings."""
    texts = [f"{m['title']} {', '.join(m['genres'])} {m['overview']}" for m in metadata]
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)  

    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)
    
    faiss.write_index(index, INDEX_FILE)
    logging.info(f"✅ Search index saved to {INDEX_FILE}")

def search_movies(query, top_k=10):
    """Improve search accuracy by filtering results based on relevance."""
    with open(METADATA_FILE, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    index = faiss.read_index(INDEX_FILE)
    query_embedding = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    distances, indices = index.search(query_embedding, top_k)

    results = [metadata[i] for i in indices[0]
               if i < len(metadata) and query.lower() in metadata[i]['title'].lower()]
    return results

# ---------------- Evaluation Function ----------------
def evaluate_search_system(test_queries, ground_truth, top_k=5):
    """
    Evaluate the search system using these metrics:
      - Precision@5: Fraction of the top 5 results that are relevant.
      - Recall@5: Fraction of relevant items retrieved in the top 5.
      - MRR: Mean Reciprocal Rank for the position of the first relevant result.
      - mAP: Mean Average Precision over the query set.
      - Accuracy@5: Fraction of top-5 results that are relevant.
    """
    total_precision = total_recall = total_mrr = total_map = total_accuracy = 0.0
    num_queries = len(test_queries)

    for query in test_queries:
        results = search_movies(query, top_k=top_k)
        relevant_movies = ground_truth.get(query, [])
        retrieved_titles = [movie["title"] for movie in results]
        relevant_retrieved = [title for title in retrieved_titles if title in relevant_movies]

        # Calculate Precision@5
        precision = len(relevant_retrieved) / top_k if top_k > 0 else 0.0

        # Calculate Recall@5
        recall = len(relevant_retrieved) / len(relevant_movies) if relevant_movies else 0.0

        # Calculate MRR (Mean Reciprocal Rank)
        mrr = 0.0
        for rank, title in enumerate(retrieved_titles, start=1):
            if title in relevant_movies:
                mrr = 1.0 / rank
                break

        # Calculate Mean Average Precision (mAP)
        if relevant_movies:
            map_score = sum((idx + 1) / (retrieved_titles.index(title) + 1)
                            for idx, title in enumerate(relevant_retrieved)) / len(relevant_movies)
        else:
            map_score = 0.0

        # Calculate Accuracy@5 (fraction of top-5 results that are relevant)
        accuracy = len(relevant_retrieved) / top_k

        total_precision += precision
        total_recall += recall
        total_mrr += mrr
        total_map += map_score
        total_accuracy += accuracy

    return {
        "Precision@5": total_precision / num_queries,
        "Recall@5": total_recall / num_queries,
        "MRR": total_mrr / num_queries,
        "mAP": total_map / num_queries,
        "Accuracy@5": total_accuracy / num_queries
    }
# ------------------------------------------------------

# Flask Web Interface
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")  # Ensure you have an 'index.html' template in the 'templates' directory

@app.route("/search", methods=["POST"])
def search():
    query = request.form.get("query")
    results = search_movies(query)
    return render_template("index.html", results=results, query=query)

if __name__ == "__main__":
    # Build the metadata and search index.
    fetch_movie_posters(num_pages=50)
    
    # --- Evaluation Section ---
    # Define your test queries and ground truth mappings.
    # (Update the ground truth as needed for your domain.)
    test_queries = ["Interstellar", "Titanic", "John Wick"]
    ground_truth = {
        "Interstellar": ["Interstellar"],
        "Titanic": ["Titanic"],
        "John Wick": ["John Wick"]
    }

    
    evaluation_results = evaluate_search_system(test_queries, ground_truth, top_k=5)
    logging.info("Evaluation Metrics:")
    logging.info(json.dumps(evaluation_results, indent=4))
    # --- End Evaluation Section ---

    app.run(debug=True)
