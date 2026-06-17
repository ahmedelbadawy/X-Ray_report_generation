"""generate a report per test image from re-ranked retrieved impressions.
"""
import os
import pandas as pd
from tqdm import tqdm
import ollama

from retrieve import Retriever

MODEL    = 'qwen2.5:32b'
# TEST_CSV = 'data/test_set.csv'      # ground-truth reports, row order matches the H5 array
OUT_CSV  = 'results/generations_itm.csv'
K        = 3


def build_prompt(retrieved):
    evidence = "\n".join(f"- (confidence {s:.3f}) {r}" for r, s in retrieved)
    return f"""You are a radiologist writing the Impression section of a chest X-ray report.

Below are impressions from prior chest X-rays whose images closely match the current case. Each includes a confidence score (higher = better match). These are your ONLY source of information.

{evidence}

Write a concise, clinically realistic Impression for the current study.

Rules:
- Use ONLY findings explicitly supported by the evidence. Do NOT infer, add, or assume any detail (laterality, location, severity, or measurements) not clearly stated.
- Combine overlapping or similar findings into one coherent statement; avoid repetition.
- Prioritize findings that recur across items or have higher confidence scores.
- If findings conflict, follow the higher-confidence evidence.
- Preserve uncertainty from the evidence (e.g. "possible", "may represent") rather than overstating; do not add certainty the evidence does not have.
- If the evidence indicates no acute abnormality, state that clearly and concisely.
- Describe only the current study. Do NOT refer to prior studies or comparisons (e.g. "unchanged", "stable", "redemonstrated", "previously seen").
- Output only the Impression text — no preamble, headers, quotes, confidence scores, or mention of the evidence.

Impression:"""

def generate_one(retrieved):
    resp = ollama.chat(
        model=MODEL,
        messages=[{'role': 'user', 'content': build_prompt(retrieved)}],
        options={'temperature': 0.2},          # low temp = faithful, not creative
    )
    return resp['message']['content'].strip()


def main():
    os.makedirs('results', exist_ok=True)
    # test = pd.read_csv(TEST_CSV)

    retriever = Retriever()                     # encodes all test images once
    # assert len(test) == retriever.n, \
    #     f"CSV rows ({len(test)}) != images ({retriever.n}) — ordering must match"

    done = set()
    if os.path.exists(OUT_CSV):
        done = set(pd.read_csv(OUT_CSV)['index'])

    rows = []
    for i in tqdm(range(retriever.n), desc="generating"):
        if i in done:
            continue
        retrieved = retriever.retrieve_rerank(i, k=K)    
        rows.append({
            'index':          i,
            # 'true_report':    test.iloc[i]['report'],      # ground truth by row position
            'retrieved_top1': f"{retrieved[0][0]}## {retrieved[1][0]} ## {retrieved[2][0]}",             # retrieval-only baseline
            'generated':      generate_one(retrieved),     # RAG output
        })
        if len(rows) % 25 == 0:
            _append(rows); rows = []
    if rows:
        _append(rows)
    print(f"Done. Saved to {OUT_CSV}")


def _append(rows):
    pd.DataFrame(rows).to_csv(OUT_CSV, mode='a',
                              header=not os.path.exists(OUT_CSV), index=False)


if __name__ == "__main__":
    main()