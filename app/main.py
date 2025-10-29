import os
import pandas as pd
from sqlalchemy import create_engine
from sentence_transformers import SentenceTransformer
from pgvector.sqlalchemy import Vector

import os
import logging

def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    return logging.getLogger(__name__)

# Sentiment Analysis
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')

sent_analyzer = SentimentIntensityAnalyzer()

def get_sentiment(text):
    score = sent_analyzer.polarity_scores(text)['compound']
    return score  # ranges from -1 (neg) to +1 (pos)

# Generate Embeddings
from sentence_transformers import SentenceTransformer

def load_model(model_name="all-MiniLM-L6-v2"):
    return SentenceTransformer(model_name)

def get_embeddings(model, texts):
    return model.encode(texts, show_progress_bar=True)

# Text preprocessing
import re
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

def update_clean_text_db(DB_CONN, post):
    clean_title = clean_text(post[1])
    clean_selftext = clean_text(post[2])
    post_id = post[0]
    with DB_CONN.cursor() as cur:
        cur.execute("""
            UPDATE reddit
            SET title = %s, selftext = %s
            WHERE id = %s
        """, (clean_title, clean_selftext, post_id))
    DB_CONN.commit()

# Keyword Extraction
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer

def extract_keywords(corpus, top_n=5):
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(corpus)
    feature_array = vectorizer.get_feature_names_out()
    keywords_per_doc = []
    for row in X:
        tfidf_scores = zip(feature_array, row.toarray()[0])
        top_terms = sorted(tfidf_scores, key=lambda x: x[1], reverse=True)[:top_n]
        keywords_per_doc.append([term for term, score in top_terms])
    return keywords_per_doc

# Database connection
import psycopg2

def connect_to_db():
    DB_CONN = psycopg2.connect(
    host=os.environ['DB_HOST'],
    dbname=os.environ['DB_NAME'],
    user=os.environ['DB_USER'],
    password=os.environ['DB_PASSWORD'])
    return DB_CONN

def fetch_unprocessed(DB_CONN):
    with DB_CONN.cursor() as cur:
        cur.execute("SELECT id, title, selftext FROM reddit WHERE processed = FALSE LIMIT 500")
        return cur.fetchall()

def process_post(model, post):
    text = (post[1] or '') + ' ' + (post[2] or '')
    sentiment = sent_analyzer.polarity_scores(text)['compound']
    embedding = get_embeddings(model, text).tolist()
    return sentiment, embedding

def update_post(DB_CONN, post_id, sentiment, embedding):
    with DB_CONN.cursor() as cur:
        cur.execute("""
            UPDATE reddit
            SET sentiment_score = %s, embedding = %s::vector, processed = TRUE
            WHERE id = %s
        """, (sentiment, embedding, post_id))
    DB_CONN.commit()


def main():

    print("Loading sentence transformers model..")
    model = load_model()

    print("Connecting to DB...")
    DB_CONN = connect_to_db()

    print("Fetching new posts...")
    posts = fetch_unprocessed(DB_CONN)
    for post in posts:

        update_clean_text_db(DB_CONN, post)

        sentiment_score, embedding = process_post(model, post)
        update_post(DB_CONN, post[0], sentiment_score, embedding)

    print(f"Processed {len(posts)} posts")

    print("Done!")

if __name__ == "__main__":
    main()
