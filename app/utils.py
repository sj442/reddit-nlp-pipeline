import logging

def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    return logging.getLogger(__name__)



from sentence_transformers import SentenceTransformer

def load_model(model_name="all-MiniLM-L6-v2"):
    return SentenceTransformer(model_name)

def get_embeddings(model, texts):
    return model.encode(texts, show_progress_bar=True)



import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

STOPWORDS = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    if not text:
        return ""
    text = re.sub(r"http\S+", "", text)         # remove URLs
    text = re.sub(r"[^a-zA-Z\s]", "", text)     # remove punctuation/numbers
    text = text.lower()
    words = [lemmatizer.lemmatize(w) for w in text.split() if w not in STOPWORDS]
    return " ".join(words)



from sqlalchemy import text

def fetch_unprocessed_posts(engine, limit=500):
    query = text("SELECT id, title, selftext FROM reddit_posts WHERE processed = false LIMIT :limit;")
    with engine.connect() as conn:
        return conn.execute(query, {"limit": limit}).fetchall()

def mark_as_processed(engine, post_ids):
    query = text("UPDATE reddit_posts SET processed = true WHERE id = ANY(:ids);")
    with engine.begin() as conn:
        conn.execute(query, {"ids": post_ids})
