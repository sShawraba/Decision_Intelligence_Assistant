import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from app.services.chroma_service import chroma_service

DATA_PATH = "/home/soup/decision-intelligence-assistant/data/cleaned_tweets_sample.csv"  # update this

def run_ingestion():
    print("Loading dataset...")

    df = pd.read_csv(DATA_PATH)

    # optional but recommended (speed + quality)
    df = df[df["clean_text"].notna()]
    df = df.sample(2000, random_state=42)

    texts = df["rag_text"].tolist()
    ids = df["tweet_id"].astype(str).tolist()

    print(f"Ingesting {len(texts)} documents...")
    chroma_service.clear()

    batch_size = 1000

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_ids = ids[i:i+batch_size]

        chroma_service.add_documents(batch_texts, batch_ids)

        print(f"Ingested {i + len(batch_texts)}/{len(texts)}")

    print("Done ✔ RAG database built")

if __name__ == "__main__":
    run_ingestion()
