"""Stage 2 (run once): embed all impressions into a persistent Chroma DB.
Usage:  python -m src.build_db
Re-running is safe: it skips ids already stored, so a crashed run resumes.
"""
import torch
import pandas as pd
import chromadb
from tqdm import tqdm

from encoders import load_model, text_vector

CONFIG_PATH = 'configs/Retrieval_flickr.yaml'
CHECKPOINT  = 'chkpts/checkpoint_59.pth'
CSV_PATH    = 'data/mimic_train_impressions.csv'   # df with an 'impression' column
DB_PATH     = './cxr_db'
BATCH       = 64


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device, "will be used for encoding.")

    model, tokenizer, _ = load_model(CONFIG_PATH, CHECKPOINT, device)

    df = pd.read_csv(CSV_PATH)
    impressions_df = df["report"].drop_duplicates().dropna().reset_index(drop = True)
    impressions = impressions_df.tolist()
    ids = [str(i) for i in impressions_df.index]

    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_or_create_collection(
        name="impressions",
        metadata={"hnsw:space": "cosine"},
    )

    # figure out which ids are already stored
    existing = set(collection.get(include=[])['ids']) if collection.count() else set()

    for s in tqdm(range(0, len(impressions), BATCH), desc="embedding"):
        batch_ids = ids[s:s + BATCH]
        # keep only rows not already in the DB
        keep = [(i, t) for i, t in zip(batch_ids, impressions[s:s + BATCH])
                if i not in existing]
        if not keep:
            continue
        b_ids, b_texts = zip(*keep)

        vecs = text_vector(model, tokenizer, list(b_texts), device).tolist()
        collection.add(embeddings=vecs, documents=list(b_texts), ids=list(b_ids))

    print(f"Done. Collection holds {collection.count()} impressions.")
    sanity_check(collection, model, tokenizer, impressions[0], device)


def sanity_check(collection, model, tokenizer, sample_text, device):
    """Embed an impression already in the DB and query with it.
    It should retrieve itself as the #1 hit — proves encoding + storage line up."""
    vec = text_vector(model, tokenizer, sample_text, device)[0].tolist()
    res = collection.query(query_embeddings=[vec], n_results=1)
    top = res['documents'][0][0]
    print("Sanity check:", "PASS" if top == sample_text else "CHECK — top hit differs")


if __name__ == "__main__":
    main()