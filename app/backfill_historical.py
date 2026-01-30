import os, json, logging, uuid
import boto3
import pandas as pd
from datetime import datetime, timezone
import pyarrow as pa
import pyarrow.parquet as pq

# import NLP functions from main.py
from main import clean_text, get_sentiment, extract_keywords

logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3 = boto3.client("s3")

INPUT_BUCKET = os.environ["INPUT_BUCKET"]
CURATED_BUCKET = os.environ.get("CURATED_BUCKET", INPUT_BUCKET)
CURATED_PREFIX = os.environ.get("CURATED_PREFIX", "curated/reddit_posts/")

BASE_COLUMNS = ["id", "title", "selftext", "created_dt", "score", "subreddit"]

def load_csv_from_s3(bucket, key):
    obj = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_csv(obj["Body"])

def list_csv_keys_under_prefix(bucket: str, prefix: str) -> list[str]:
    prefix = prefix if prefix.endswith("/") else prefix + "/"
    token = None
    keys = []
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
            token = resp["NextContinuationToken"]
        else:
            break
    return keys

def get_input_csv_keys() -> list[str]:
    prefixes_json = os.environ.get("INPUT_PREFIXES_JSON")
    if prefixes_json:
        prefixes = json.loads(prefixes_json)
        keys = []
        for p in prefixes:
            keys.extend(list_csv_keys_under_prefix(INPUT_BUCKET, p))
        keys = sorted(set(keys))
        if not keys:
            raise ValueError(f"No CSV files found under prefixes: {prefixes}")
        return keys

    input_key = os.environ.get("INPUT_KEY")
    if input_key:
        return [input_key]

    raise ValueError("Provide INPUT_PREFIXES_JSON or INPUT_KEY")


def write_parquet_partitioned(df: pd.DataFrame):
    groups = df.groupby(["created_year", "created_month"], dropna=False)

    for (yr, mo), part_df in groups:
        local_path = f"/tmp/part-{uuid.uuid4().hex}.parquet"
        table = pa.Table.from_pandas(part_df, preserve_index=False)
        pq.write_table(table, local_path, compression="snappy")

        s3_key = (
            f"{CURATED_PREFIX.rstrip('/')}/"
            f"created_year={int(yr)}/created_month={mo}/"
            f"part-{uuid.uuid4().hex}.parquet"
        )

        with open(local_path, "rb") as f:
            s3.put_object(Bucket=CURATED_BUCKET, Key=s3_key, Body=f)

        os.remove(local_path)

        logger.info(f"Wrote {len(part_df)} rows to s3://{CURATED_BUCKET}/{s3_key}")


def backfill():
    csv_keys = get_input_csv_keys()

    logger.info(f"Backfill reading {len(csv_keys)} CSV(s)")
    df = pd.concat([load_csv_from_s3(INPUT_BUCKET, k) for k in csv_keys], ignore_index=True)

    # keep only base columns (drops old processed cols)
    missing = [c for c in BASE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required base columns: {missing}")
    df = df[BASE_COLUMNS].copy()

    # normalize
    df["id"] = df["id"].astype(str)
    df["title"] = df["title"].fillna("")
    df["selftext"] = df["selftext"].fillna("")
    df["subreddit"] = df["subreddit"].fillna("")
    df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0).astype(int)
    df["created_dt"] = pd.to_datetime(df["created_dt"], errors="coerce", utc=True)
    df = df[df["created_dt"].notna()].copy()

    # process
    df["clean_title"] = df["title"].apply(clean_text)
    df["clean_selftext"] = df["selftext"].apply(clean_text)
    combined = (df["clean_title"] + " " + df["clean_selftext"]).fillna("")

    df["sentiment_score"] = combined.apply(get_sentiment)
    df["keywords"] = [json.dumps(x) for x in extract_keywords(combined.tolist(), top_n=5)]
    df["processed_dt"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    # partition columns from created_dt
    df["created_year"] = df["created_dt"].dt.year.astype(int)
    df["created_month"] = df["created_dt"].dt.month.apply(lambda m: f"{m:02d}")

    # dedupe across historical + weekly overlap
    df = df.sort_values("created_dt").drop_duplicates(subset=["id"], keep="last")

    write_parquet_partitioned(df)
    
    return {"rows": int(len(df)), "curated": f"s3://{CURATED_BUCKET}/{CURATED_PREFIX}"}


