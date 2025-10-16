import os
import pandas as pd
from sqlalchemy import create_engine
from sentence_transformers import SentenceTransformer
from pgvector.sqlalchemy import Vector

DB_URL = os.getenv("DB_URL")  # set as env variable in ECS

def fetch_unprocessed_posts(engine):
    query = "SELECT id, title, selftext FROM reddit_posts WHERE processed = false;"
    return pd.read_sql(query, engine)

def store_embeddings(engine, df, embeddings):
    df["embedding"] = embeddings.tolist()
    for i, row in df.iterrows():
        engine.execute(
            "UPDATE reddit_posts SET embedding = %s, processed = true WHERE id = %s",
            (row.embedding, row.id)
        )

def main():
    print("Connecting to DB...")
    engine = create_engine(DB_URL)

    print("Loading model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Fetching new posts...")
    df = fetch_unprocessed_posts(engine)

    print("Generating embeddings...")
    embeddings = model.encode(df["title"] + " " + df["selftext"], show_progress_bar=True)

    print("Storing embeddings...")
    store_embeddings(engine, df, embeddings)
    print("Done!")

if __name__ == "__main__":
    main()
