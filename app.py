# app.py — FINAL FULL WORKING CODE (POSTERS WILL LOAD)

import streamlit as st
import pickle
import pandas as pd
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import os

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(page_title="Movie Recommender System", layout="wide")

# --------------------------------------------------
# TMDB API KEY (LOCAL + CLOUD, NO CRASH)
# --------------------------------------------------
def get_api_key():
    # Streamlit Cloud
    try:
        return st.secrets["TMDB_API_KEY"]
    except Exception:
        pass

    # Environment variable (optional)
    if os.getenv("TMDB_API_KEY"):
        return os.getenv("TMDB_API_KEY")

    # LOCAL FALLBACK — REPLACE WITH YOUR REAL KEY
    return "7a749a98b6a42a373b6f8d826113d28f"

TMDB_API_KEY = get_api_key()

# --------------------------------------------------
# REQUEST SESSION (RETRY + SSL SAFE)
# --------------------------------------------------
session = requests.Session()
retry = Retry(
    total=3,
    backoff_factor=0.5,
    status_forcelist=[500, 502, 503, 504],
)
adapter = HTTPAdapter(max_retries=retry)
session.mount("https://", adapter)

# --------------------------------------------------
# FETCH POSTER
# --------------------------------------------------
def fetch_poster(movie_id):
    if TMDB_API_KEY == "YOUR_REAL_TMDB_API_KEY_HERE":
        return "https://upload.wikimedia.org/wikipedia/commons/6/65/No-Image-Placeholder.svg"

    url = (
        f"https://api.themoviedb.org/3/movie/{movie_id}"
        f"?api_key={TMDB_API_KEY}&language=en-US"
    )

    try:
        response = session.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException:
        return "https://upload.wikimedia.org/wikipedia/commons/6/65/No-Image-Placeholder.svg"

    poster_path = data.get("poster_path")
    if not poster_path:
        return "https://upload.wikimedia.org/wikipedia/commons/6/65/No-Image-Placeholder.svg"

    return "https://image.tmdb.org/t/p/w500" + poster_path

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
movie_dict = pickle.load(open("movie_dict.pkl", "rb"))
movie = pd.DataFrame(movie_dict)

# ensure tag is text
movie["tag"] = movie["tag"].apply(
    lambda x: " ".join(x) if isinstance(x, list) else str(x)
)

# --------------------------------------------------
# BUILD SIMILARITY AT RUNTIME
# --------------------------------------------------
cv = CountVectorizer(max_features=5000, stop_words="english")
vectors = cv.fit_transform(movie["tag"]).toarray()
similarity = cosine_similarity(vectors)

# --------------------------------------------------
# RECOMMENDER
# --------------------------------------------------
def recommend(selected_movie):
    movie_index = movie[movie["title"] == selected_movie].index[0]
    distances = similarity[movie_index]

    movie_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    names = []
    posters = []

    for i in movie_list:
        movie_id = movie.iloc[i[0]].movie_id
        names.append(movie.iloc[i[0]].title)
        posters.append(fetch_poster(movie_id))

    return names, posters

# --------------------------------------------------
# UI
# --------------------------------------------------
st.title("Movie Recommender System")

selected_movie = st.selectbox(
    "Which movie",
    movie["title"].values
)

if st.button("Recommend"):
    names, posters = recommend(selected_movie)
    cols = st.columns(5)

    for i in range(5):
        with cols[i]:
            st.text(names[i])
            st.image(posters[i])
