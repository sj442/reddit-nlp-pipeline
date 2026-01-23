import streamlit as st
import pandas as pd
import psycopg2
from sqlalchemy import create_engine
import plotly.express as px
from sklearn.decomposition import PCA
import numpy as np
import os
import nltk
from dotenv import load_dotenv


# Load .env variables
load_dotenv()

# ---------------------------
# DATABASE CONNECTION
# ---------------------------
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

@st.cache_data(ttl=600)
def load_data():
    DB_CONN = psycopg2.connect(
    host=DB_HOST,
    dbname=DB_NAME,
    user=DB_USER,
    password=DB_PASSWORD
)

# Fetch posts
    with DB_CONN.cursor() as cur:
        cur.execute("""
            SELECT id, title, sentiment_score, created_dt, subreddit, score, embedding
            FROM reddit 
            WHERE processed = TRUE 
            AND sentiment_score IS NOT NULL
        """)
        rows = cur.fetchall()
        columns = [desc[0] for desc in cur.description]

    # Convert to DataFrame
    df = pd.DataFrame(rows, columns=columns)

    # Close connection
    DB_CONN.close()
    return df

# ---------------------------
# STREAMLIT APP
# ---------------------------
st.set_page_config(page_title="Reddit NLP Dashboard", layout="wide")
st.title("🧠 Reddit NLP Sentiment Dashboard")

with st.spinner("Loading data from RDS..."):
    df = load_data()

st.sidebar.header("Filters")
subreddits = st.sidebar.multiselect(
    "Select Subreddit(s):", options=df["subreddit"].unique(), default=df["subreddit"].unique()
)
# topics = st.sidebar.multiselect(
#     "Select Topic(s):", options=df["topic"].dropna().unique(), default=df["topic"].dropna().unique()
# )

filtered = df[df["subreddit"].isin(subreddits)]

st.markdown(f"**Total posts:** {len(filtered):,}")

# ---------------------------
# SENTIMENT DISTRIBUTION
# ---------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Sentiment Score Distribution")
    fig = px.histogram(filtered, x="sentiment_score", nbins=30, color="subreddit",
                       title="Sentiment Distribution by Subreddit", barmode="overlay")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Average Sentiment Over Time")
    filtered["created_dt"] = pd.to_datetime(filtered["created_dt"])
    sentiment_trend = (
        filtered.groupby(filtered["created_dt"].dt.date)["sentiment_score"].mean().reset_index()
    )
    fig = px.line(sentiment_trend, x="created_dt", y="sentiment_score", title="Average Sentiment Over Time")
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# TOP SUBREDDITS
# ---------------------------
st.subheader("Average Sentiment by Subreddit")
avg_sent = filtered.groupby("subreddit")["sentiment_score"].mean().sort_values(ascending=False).reset_index()
fig = px.bar(avg_sent, x="subreddit", y="sentiment_score", title="Average Sentiment by Subreddit")
st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# TOPICS
# ---------------------------
# st.subheader("Topic-wise Sentiment")
# topic_sent = filtered.groupby("topic")["sentiment_score"].mean().sort_values(ascending=False).reset_index()
# fig = px.bar(topic_sent, x="topic", y="sentiment_score", title="Average Sentiment by Topic")
# st.plotly_chart(fig, use_container_width=True)


# ---------------------------
# WORD CLOUDS
# ---------------------------
from wordcloud import WordCloud
import matplotlib.pyplot as plt

def generate_wordcloud(text, color):
    return WordCloud(
        width=800,
        height=400,
        background_color='white',
        colormap=color,  # uses matplotlib color maps
        max_words=100
    ).generate(text)

def plot_wordcloud(wc, title):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis("off")
    ax.set_title(title, fontsize=18, weight='bold')
    st.pyplot(fig)

# Example usage:
positive_text = " ".join(df[df['sentiment_score'] > 0.2]['selftext'].dropna())
negative_text = " ".join(df[df['sentiment_score'] < -0.2]['selftext'].dropna())

positive_wc = generate_wordcloud(positive_text, color='Greens')
negative_wc = generate_wordcloud(negative_text, color='Reds')

st.subheader("Positive Word Cloud")
plot_wordcloud(positive_wc, "Positive Posts")

st.subheader("Negative Word Cloud")
plot_wordcloud(negative_wc, "Negative Posts")

# ---------------------------
# OPTIONAL: EMBEDDING VISUALIZATION
# ---------------------------
if st.checkbox("Show Embedding Visualization (PCA 2D)"):
    st.info("Reducing 768D embeddings to 2D via PCA (this may take a few seconds)...")
    embeddings = np.vstack(filtered["embedding"].values)
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(embeddings)
    reduced_df = pd.DataFrame(reduced, columns=["x", "y"])
    reduced_df["sentiment_score"] = filtered["sentiment_score"].values
    reduced_df["subreddit"] = filtered["subreddit"].values

    fig = px.scatter(
        reduced_df,
        x="x", y="y",
        color="sentiment_score",
        hover_data=["subreddit"],
        title="Embedding Clusters (colored by sentiment)"
    )
    st.plotly_chart(fig, use_container_width=True)
