"""retrieve the top-k most similar impressions for a query chest X-ray.
"""
import torch
import chromadb
from encoders import load_model, image_vector
from encoders import build_image_transform   # if it's defined in encoders.py

CONFIG_PATH = 'configs/Retrieval_flickr.yaml'
CHECKPOINT  = 'chkpts/checkpoint_59.pth'
DB_PATH     = './cxr_db'


def get_retriever():
    """Load model + DB once, return a function you can call many times."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tokenizer, config = load_model(CONFIG_PATH, CHECKPOINT, device)
    img_tf = build_image_transform(config['image_res'])

    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_collection(name="impressions")   # must already exist

    @torch.no_grad()
    def retrieve(image_path, k=5):
        vec = image_vector(model, img_tf, image_path, device)[0].tolist()  # [256]
        res = collection.query(
            query_embeddings=[vec],
            n_results=k,
            include=['documents', 'distances'],
        )
        reports   = res['documents'][0]    # the k impression texts
        distances = res['distances'][0]    # cosine distance (lower = more similar)
        return list(zip(reports, distances))

    return retrieve


if __name__ == "__main__":
    retrieve = get_retriever()
    results = retrieve('data/sample_cxr.jpg', k=5)
    for i, (report, dist) in enumerate(results, 1):
        print(f"\n--- rank {i}  (distance {dist:.3f}) ---\n{report}")