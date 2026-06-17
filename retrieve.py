"""retrieve top-k impressions for query images.
"""
import torch
import torch.nn.functional as F
import chromadb
from torch.utils.data import DataLoader

from encoders import load_model, build_image_transform
from image_dataset import CXRDataset

CONFIG_PATH = 'configs/Retrieval_flickr.yaml'
CHECKPOINT  = 'chkpts/checkpoint_59.pth'
DB_PATH     = './cxr_db'
H5_PATH     = 'data/cxr.h5'


class Retriever():
    """Encode all given image keys once, then return a fast lookup-based retriever."""

    def __init__(self, batch_size=64, num_workers=4):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model, self.tokenizer, config = load_model(CONFIG_PATH, CHECKPOINT, self.device)
        img_tf = build_image_transform(config['image_res'])

        # --- precompute every image embedding in batches ---
        self.ds = CXRDataset(H5_PATH, img_tf)
        self.n = len(self.ds)
        loader = DataLoader(self.ds, batch_size=batch_size, num_workers=num_workers,
                            pin_memory=True, shuffle=False)

        self.feats_all = []
        self.model.eval()
        with torch.no_grad():
            for imgs in loader:
                imgs = imgs.to(self.device, non_blocking=True)
                embeds = self.model.visual_encoder(imgs)
                feats = F.normalize(self.model.vision_proj(embeds[:, 0, :]), dim=-1)  # [B,256]
                self.feats_all.append(feats.cpu())

            self.feats_all = torch.cat(self.feats_all, dim=0)          # [N, 256]

        client = chromadb.PersistentClient(path=DB_PATH)
        self.collection = client.get_collection(name="impressions")

    def retrieve(self, i, k=5):
        res = self.collection.query(
            query_embeddings=[self.feats_all[i].tolist()],     # lookup, already encoded
            n_results=k,
            include=['documents', 'distances'],
        )
        return list(zip(res['documents'][0], res['distances'][0]))
    
    def retrieve_rerank(self, i, k=5, n_candidates=20):
        # Stage 1: ITC — fast, get a wide candidate net
        res = self.collection.query(
            query_embeddings=[self.feats_all[i].tolist()],
            n_results=n_candidates,
            include=['documents', 'distances'],
        )
        candidates = res['documents'][0]                       # n_candidates reports

        # Stage 2: ITM — rescore those candidates with the fusion network
        img = self.ds[i].unsqueeze(0).to(self.device)          # raw image -> [1,3,H,W]
        image_embeds = self.model.visual_encoder(img)          # [1, patches+1, 768]
        text = self.tokenizer(candidates, padding='max_length', truncation=True,
                              max_length=40, return_tensors='pt').to(self.device)
        scores = self.itm_score(image_embeds, text)            # [n_candidates]

        # re-sort candidates by ITM score, take top-k
        order = scores.argsort(descending=True)[:k]
        return [(candidates[j], scores[j].item()) for j in order]
    
    @torch.no_grad()
    def itm_score(self, image_embeds, text):
        """Fusion score for one image against a batch of candidate reports.
        image_embeds: [1, patches+1, 768] from visual_encoder (the FULL sequence, not the vector)
        text: tokenized candidate reports, batch of M
        returns: [M] match probabilities
        """
        M = text.input_ids.size(0)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(self.device)

        output = self.model.text_encoder(
            text.input_ids,
            attention_mask=text.attention_mask,
            encoder_hidden_states=image_embeds.repeat(M, 1, 1),   # same image vs each report
            encoder_attention_mask=image_atts.repeat(M, 1),
            return_dict=True,
            mode='fusion',                                        # the cross-attention path
        )
        vl = output.last_hidden_state[:, 0, :]                    # [M, 768] fused CLS
        logits = self.model.itm_head(vl)                          # [M, 2]
        return logits.softmax(dim=1)[:, 1]                        # P(match) per report


if __name__ == "__main__":
    # quick test on a few keys
    retriever = Retriever()
    for idx in range(10):
        print(f"\n=== {idx} ===")
        for rank, (rep, dist) in enumerate(retriever.retrieve(idx, k=5), 1):
            print(f"[{rank}] ({dist:.3f}) {rep}...")