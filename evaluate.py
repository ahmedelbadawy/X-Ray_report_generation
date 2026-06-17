"""evaluate generated reports vs retrieval-only baseline, against ground truth.
"""
import pandas as pd
from f1chexbert import F1CheXbert
from bert_score import score as bertscore
import inspect
GEN_CSV = 'results/generations.csv' 
TRUE_CSV = "data/mimic_test_impressions.csv"

def bertscore_f1(preds, refs):
    P, R, F1 = bertscore(preds, refs, lang='en', rescale_with_baseline=True)
    return F1.mean().item()

def main():
    df = pd.read_csv(TRUE_CSV).dropna(subset=['report'])
    truth = df['report'].astype(str).tolist()
    df_gen = pd.read_csv(GEN_CSV)
    print(inspect.signature(F1CheXbert.__init__))
    chexbert = F1CheXbert(device='cuda')   

    print(f"Evaluating {len(df)}, {len(df_gen)} reports on CheXbert F1\n")


    for name, col in [('Retrieval-only (top-1)', 'retrieved_top1'),
                      ('RAG (generated)',        'generated')]:
        preds = df_gen[col].astype(str).tolist()
        # f1chexbert returns (accuracy, per-class F1, macro/micro reports)
        accuracy, acc_per_class, class_report, class_report_5 = chexbert(
            hyps=preds, refs=truth
        )
        micro_f1 = class_report['micro avg']['f1-score']
        macro_f1 = class_report['macro avg']['f1-score']
        print(f"{name}:")
        print(f"  micro-F1: {micro_f1:.3f}   macro-F1: {macro_f1:.3f}")
        print(f"  (5-finding micro-F1: {class_report_5['micro avg']['f1-score']:.3f})\n")

        print(f"{name} BERTScore F1: {bertscore_f1(preds, truth):.3f}\n")

if __name__ == "__main__":
    main()


# Evaluating 3678, 3678 reports on CheXbert F1

# Retrieval-only (top-1):
#   micro-F1: 0.375   macro-F1: 0.266
#   (5-finding micro-F1: 0.451)

# RAG (generated):
#   micro-F1: 0.363   macro-F1: 0.249
#   (5-finding micro-F1: 0.438)
