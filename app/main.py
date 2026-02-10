import os
import json
import uuid
import logging
import boto3
import pandas as pd
import csv
from io import StringIO
from datetime import datetime, timezone
import re

import pyarrow as pa
import pyarrow.parquet as pq

# =========================
# Logging
# =========================
def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    return logging.getLogger(__name__)

logger = setup_logger()

# =========================
# S3 config (env vars)
# =========================
SOURCE_BUCKET = os.environ["SOURCE_BUCKET"]
SOURCE_PREFIX = os.environ.get("SOURCE_PREFIX", "reddit_csv/")

CURATED_BUCKET = os.environ.get("CURATED_BUCKET", SOURCE_BUCKET)
CURATED_PREFIX = os.environ.get("CURATED_PREFIX", "curated/reddit_posts/")

s3 = boto3.client("s3")

# =========================
# NLP: Sentiment
# =========================
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download("vader_lexicon", quiet=True)
sent_analyzer = SentimentIntensityAnalyzer()

def get_sentiment(text):
    return sent_analyzer.polarity_scores(text)["compound"]

# =========================
# NLP: Text preprocessing
# =========================
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

ENGLISH_STOPWORDS = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

CUSTOM_STOPWORDS = {
    "other","like","using","use","look","need","help","best",
    "year","day","post","reddit","people","thing","things",
    "way","make","know","want","good","bad","new","old",
    "really","also","one","two","first","second"
}

GENAI_FILLER = {
    "ai","model","models","llm","llms","gpt","chatgpt",
    "openai","anthropic","claude"
}

STOPWORDS = ENGLISH_STOPWORDS.union(CUSTOM_STOPWORDS)
STOPWORDS = STOPWORDS.union(GENAI_FILLER)


def clean_text(text):
    if not text:
        return ""
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.lower()
    words = [lemmatizer.lemmatize(w) for w in text.split() if w not in STOPWORDS]
    return " ".join(words)

# =========================
# NLP: Keywords
# =========================
from sklearn.feature_extraction.text import TfidfVectorizer

def extract_keywords(corpus, top_n=5):
    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(corpus)
    feature_array = vectorizer.get_feature_names_out()

    keywords_per_doc = []
    for row in X:
        scores = zip(feature_array, row.toarray()[0])
        top_terms = sorted(scores, key=lambda x: x[1], reverse=True)[:top_n]
        keywords_per_doc.append([term for term, _ in top_terms])

    return keywords_per_doc

# =========================
# S3 helpers
# =========================
DATE_RE = re.compile(r"^ingest_dt=(\d{4}-\d{2}-\d{2})/$")

def _ensure_trailing_slash(p: str) -> str:
    return p if p.endswith("/") else p + "/"

def list_ingest_dt_partitions(bucket: str, base_prefix: str) -> list[str]:
    prefix = _ensure_trailing_slash(base_prefix)
    token = None
    dates = set()

    while True:
        kwargs = {"Bucket": bucket, "Prefix": prefix, "Delimiter": "/"}
        if token:
            kwargs["ContinuationToken"] = token

        resp = s3.list_objects_v2(**kwargs)
        for cp in resp.get("CommonPrefixes", []):
            folder = cp["Prefix"][len(prefix):]  # e.g. "ingest_dt=2026-01-18/"
            m = DATE_RE.match(folder)
            if m:
                dates.add(m.group(1))

        if resp.get("IsTruncated"):
            token = resp.get("NextContinuationToken")
        else:
            break

    return sorted(dates)

def ingest_partition_exists(bucket: str, base_prefix: str, ingest_dt: str) -> bool:
    prefix = f"{_ensure_trailing_slash(base_prefix)}ingest_dt={ingest_dt}/"
    resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1)
    return "Contents" in resp and len(resp["Contents"]) > 0

def list_csvs_for_ingest_dt(bucket: str, base_prefix: str, ingest_dt: str) -> list[str]:
    prefix = f"{_ensure_trailing_slash(base_prefix)}ingest_dt={ingest_dt}/"
    keys = []
    token = None

    while True:
        kwargs = {"Bucket": bucket, "Prefix": prefix}
        if token:
            kwargs["ContinuationToken"] = token

        resp = s3.list_objects_v2(**kwargs)
        for obj in resp.get("Contents", []):
            k = obj["Key"]
            if k.endswith(".csv"):
                keys.append(k)

        if resp.get("IsTruncated"):
            token = resp.get("NextContinuationToken")
        else:
            break

    return keys

def load_csv_from_s3(bucket, key):
    obj = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_csv(obj["Body"])


def write_parquet_partitioned_to_s3(df: pd.DataFrame, bucket: str, base_prefix: str):
    """
    Writes Parquet files partitioned by created_year/created_month
    to: s3://bucket/base_prefix/created_year=YYYY/created_month=MM/part-....parquet
    """
    base_prefix = _ensure_trailing_slash(base_prefix)

    # Ensure created_dt exists and is datetime
    if "created_dt" not in df.columns:
        raise ValueError("created_dt column is required for parquet partitioning")

    df["created_dt"] = pd.to_datetime(df["created_dt"], errors="coerce", utc=True)
    df = df[df["created_dt"].notna()].copy()

    df["created_year"] = df["created_dt"].dt.year.astype(int)
    df["created_month"] = df["created_dt"].dt.month.apply(lambda m: f"{m:02d}")

    # Write each partition group as its own parquet file
    for (yr, mo), part_df in df.groupby(["created_year", "created_month"], dropna=False):
        local_path = f"/tmp/part-{uuid.uuid4().hex}.parquet"
        table = pa.Table.from_pandas(part_df, preserve_index=False)
        pq.write_table(table, local_path, compression="snappy")

        out_key = (
            f"{base_prefix}"
            f"created_year={int(yr)}/created_month={mo}/"
            f"part-{uuid.uuid4().hex}.parquet"
        )

        with open(local_path, "rb") as f:
            s3.put_object(Bucket=bucket, Key=out_key, Body=f)

        os.remove(local_path)

    return {"bucket": bucket, "prefix": base_prefix}


def write_csv_to_s3(df, bucket, base_prefix, ingest_dt):
    out_key = f"{_ensure_trailing_slash(base_prefix)}ingest_dt={ingest_dt}/part-{uuid.uuid4().hex}.csv"

    buf = StringIO()
    df.to_csv(
        buf,
        index=False,
        quoting=csv.QUOTE_ALL,
        lineterminator="\n",
    )

    s3.put_object(
        Bucket=bucket,
        Key=out_key,
        Body=buf.getvalue().encode("utf-8"),
        ContentType="text/csv",
    )
    return out_key

# =========================
# Pipeline
# =========================
def run_for_ingest_dt(ingest_dt: str):
    logger.info(f"Running NLP for ingest_dt={ingest_dt}")

    csv_keys = list_csvs_for_ingest_dt(SOURCE_BUCKET, SOURCE_PREFIX, ingest_dt)
    logger.info(f"Found {len(csv_keys)} input CSV(s)")

    if not csv_keys:
        logger.warning("No CSV files found for this ingest_dt. Exiting.")
        return {"rows_in": 0, "rows_out": 0, "ingest_dt": ingest_dt}

    df = pd.concat([load_csv_from_s3(SOURCE_BUCKET, k) for k in csv_keys], ignore_index=True)
    logger.info(f"Loaded {len(df)} rows")

    for col in ["id", "title", "selftext"]:
        if col not in df.columns:
            raise ValueError(f"Missing expected column '{col}' in input CSV schema")

    # Ensure created_dt exists (your weekly CSVs should already contain it)
    if "created_dt" not in df.columns:
        raise ValueError("Missing created_dt in weekly CSV. Required for curated parquet output.")


    BASE_COLUMNS = ["id", "title", "selftext", "created_dt", "score", "subreddit"]
    for c in BASE_COLUMNS:
        if c not in df.columns:
            raise ValueError(f"Missing expected column '{c}' in input CSV schema")

    df = df[BASE_COLUMNS].copy()

    df["clean_title"] = df["title"].fillna("").apply(clean_text)
    df["clean_selftext"] = df["selftext"].fillna("").apply(clean_text)
    combined = (df["clean_title"] + " " + df["clean_selftext"]).fillna("")

    logger.info("Running sentiment analysis")
    df["sentiment_score"] = combined.apply(get_sentiment)

    logger.info("Extracting keywords")
    df["keywords"] = extract_keywords(combined.tolist(), top_n=5)
    df["keywords"] = df["keywords"].apply(json.dumps)


    df["processed_dt"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    write_parquet_partitioned_to_s3(df, CURATED_BUCKET, CURATED_PREFIX)
    logger.info(f"Wrote curated parquet to s3://{CURATED_BUCKET}/{CURATED_PREFIX}")
    return {"rows_in": int(len(df)), "rows_out": int(len(df)), "ingest_dt": ingest_dt,
            "output": f"s3://{CURATED_BUCKET}/{CURATED_PREFIX}"}


def pick_ingest_dt(event: dict) -> str:
    """
    Priority:
      1) If event explicitly provides ingest_dt, use it.
      2) Prefer today's UTC ingest_dt if that partition exists.
      3) Otherwise fall back to latest available partition.
    """
    if isinstance(event, dict) and event.get("ingest_dt"):
        ingest_dt = event["ingest_dt"]
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", ingest_dt):
            raise ValueError(f"Invalid ingest_dt format: {ingest_dt}")
        return ingest_dt

    today_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if ingest_partition_exists(SOURCE_BUCKET, SOURCE_PREFIX, today_utc):
        return today_utc

    dates = list_ingest_dt_partitions(SOURCE_BUCKET, SOURCE_PREFIX)
    if not dates:
        raise ValueError(f"No ingest_dt partitions found under s3://{SOURCE_BUCKET}/{SOURCE_PREFIX}")
    return dates[-1]

# Lambda entrypoint for EventBridge schedule
def lambda_handler(event, context):
    ingest_dt = pick_ingest_dt(event or {})
    result = run_for_ingest_dt(ingest_dt)
    return {"statusCode": 200, "body": json.dumps(result)}

# Local/ECS entrypoint (optional)
def main():
    """
    Local testing options:
      - Set TEST_INGEST_DT=YYYY-MM-DD
      - OR set TEST_EVENT_JSON='{"ingest_dt":"YYYY-MM-DD"}'
      - Otherwise it will try today's UTC, else latest partition.
    """
    test_event_json = os.environ.get("TEST_EVENT_JSON")
    if test_event_json:
        event = json.loads(test_event_json)
    else:
        test_ingest_dt = os.environ.get("TEST_INGEST_DT")
        event = {"ingest_dt": test_ingest_dt} if test_ingest_dt else {}

    ingest_dt = pick_ingest_dt(event)
    print(run_for_ingest_dt(ingest_dt))

if __name__ == "__main__":
    mode = os.environ.get("MODE", "weekly").lower()

    if mode == "backfill":
        # runs the one-off historical/legacy reprocess job
        from backfill_historical import backfill
        print(backfill())
    else:
        # runs the normal weekly pipeline
        main()