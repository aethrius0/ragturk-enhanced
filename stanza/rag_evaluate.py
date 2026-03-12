"""
RAGTurk - RAG Evaluation (Stanza)
==================================
rag_generate.py'nin ürettiği LLM cevaplarını gold answer ile
ROUGE-1, ROUGE-L ve BERTScore kullanarak değerlendirir.
Sonuçları index tipi ve classifier grubuna göre raporlar.
"""
import os, json
import numpy as np

from rouge_score import rouge_scorer
from bert_score import score as bert_score_fn

# ── Paths ──────────────────────────────────────────────
ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "stanza", "results")

INDEX_MODES = ["raw", "lemma", "hybrid"]
CLF_LABELS  = ["entity-heavy", "long-morph", "cultural", "other"]


def load_generation(mode: str) -> list[dict]:
    path = os.path.join(RESULTS, f"rag_generation_{mode}.json")
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_rouge(predictions: list[str], references: list[str]) -> dict:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=False)
    r1_scores, rl_scores = [], []
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        r1_scores.append(scores["rouge1"].fmeasure)
        rl_scores.append(scores["rougeL"].fmeasure)
    return {
        "ROUGE-1": float(np.mean(r1_scores)) if r1_scores else 0,
        "ROUGE-L": float(np.mean(rl_scores)) if rl_scores else 0,
        "num_queries": len(predictions),
    }


def compute_bertscore(predictions: list[str], references: list[str]) -> float:
    if not predictions:
        return 0.0
    P, R, F1 = bert_score_fn(
        predictions, references, lang="tr", verbose=True, batch_size=64
    )
    return float(F1.mean())


def evaluate_group(data: list[dict]) -> dict:
    """Bir grup icin ROUGE + BERTScore hesapla."""
    preds = [d["llm_answer"] for d in data]
    refs  = [d["gold_answer"] for d in data]
    rouge = compute_rouge(preds, refs)
    bs = compute_bertscore(preds, refs)
    rouge["BERTScore-F1"] = bs
    return rouge


def main():
    print("=" * 60)
    print("  RAGTurk - RAG Evaluation (Stanza)")
    print("=" * 60)

    all_results = {}

    for mode in INDEX_MODES:
        data = load_generation(mode)
        if not data:
            print(f"\n  {mode.upper()}: generation dosyasi bulunamadi, atlaniyor.")
            continue

        print(f"\n{'='*60}")
        print(f"  INDEX: {mode.upper()} — Evaluation ({len(data)} sorgu)")
        print("=" * 60)

        mode_results = {}

        # ── Genel ─────────────────────────────────────
        print("\n  [TUM SORGULAR]", flush=True)
        overall = evaluate_group(data)
        mode_results["overall"] = overall
        print(f"    ROUGE-1={overall['ROUGE-1']:.4f}  "
              f"ROUGE-L={overall['ROUGE-L']:.4f}  "
              f"BERTScore-F1={overall['BERTScore-F1']:.4f}  "
              f"(n={overall['num_queries']})")

        # ── Kategori (FACTUAL / INTERPRETATION) ───────
        categories = set(d["category"] for d in data if d.get("category"))
        cat_results = {}
        for cat in sorted(categories):
            subset = [d for d in data if d["category"] == cat]
            print(f"\n  [{cat}] ({len(subset)} sorgu)", flush=True)
            cat_eval = evaluate_group(subset)
            cat_results[cat] = cat_eval
            print(f"    ROUGE-1={cat_eval['ROUGE-1']:.4f}  "
                  f"ROUGE-L={cat_eval['ROUGE-L']:.4f}  "
                  f"BERTScore-F1={cat_eval['BERTScore-F1']:.4f}")
        mode_results["by_category"] = cat_results

        # ── Classifier grubu ──────────────────────────
        clf_results = {}
        for label in CLF_LABELS:
            subset = [d for d in data if label in d.get("clf_labels", [])]
            if not subset:
                continue
            print(f"\n  [CLF:{label}] ({len(subset)} sorgu)", flush=True)
            clf_eval = evaluate_group(subset)
            clf_results[label] = clf_eval
            print(f"    ROUGE-1={clf_eval['ROUGE-1']:.4f}  "
                  f"ROUGE-L={clf_eval['ROUGE-L']:.4f}  "
                  f"BERTScore-F1={clf_eval['BERTScore-F1']:.4f}")
        mode_results["by_clf_label"] = clf_results

        all_results[mode] = mode_results

    # ── Kaydet ─────────────────────────────────────────
    out_path = os.path.join(RESULTS, "rag_evaluation_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n  Sonuclar kaydedildi: {out_path}")

    # ── Ozet tablo ─────────────────────────────────────
    print("\n" + "=" * 70)
    print("  OZET — RAG Generation Kalitesi")
    print("=" * 70)
    print(f"  {'Index':<10} {'ROUGE-1':>10} {'ROUGE-L':>10} {'BERTScore':>10}")
    print("  " + "-" * 42)
    for mode in INDEX_MODES:
        if mode in all_results:
            o = all_results[mode]["overall"]
            print(f"  {mode.upper():<10} {o['ROUGE-1']:>10.4f} {o['ROUGE-L']:>10.4f} {o['BERTScore-F1']:>10.4f}")
    print()


if __name__ == "__main__":
    main()
