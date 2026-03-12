"""
RAGTurk – Semantik Arama
=========================
FAISS index üzerinde Türkçe kültürel arama yapar.
"""
import os
os.environ["HF_HUB_OFFLINE"] = "1"

import pickle, tempfile, shutil
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VDB  = os.path.join(ROOT, "vector_db")
MODEL = "intfloat/multilingual-e5-base"


def load():
    # FAISS — temp'e kopyala oku (OneDrive Türkçe path fix)
    src = os.path.join(VDB, "cultural_faiss.index")
    tmp = os.path.join(tempfile.gettempdir(), "ragturk_read.index")
    shutil.copy(src, tmp)
    idx = faiss.read_index(tmp)
    os.remove(tmp)

    meta = pickle.load(open(os.path.join(VDB, "metadata.pkl"), "rb"))
    model = SentenceTransformer(MODEL, local_files_only=True)
    return idx, meta, model


def search(idx, meta, model, query, k=5):
    qe = model.encode([f"query: {query}"], normalize_embeddings=True).astype("float32")
    scores, ids = idx.search(qe, k)
    results = []
    for ii, sc in zip(ids[0], scores[0]):
        m = meta[ii]
        results.append({**m, "score": float(sc)})
    return results


def main():
    print("Model ve index yükleniyor...", flush=True)
    idx, meta, model = load()
    print(f"Hazır – {idx.ntotal} chunk, {len(set(m['file'] for m in meta))} dosya\n")

    while True:
        q = input("Sorgu (çıkış: q): ").strip()
        if q.lower() in ("q", "quit", "exit", ""):
            break
        res = search(idx, meta, model, q)
        print()
        for i, r in enumerate(res, 1):
            print(f"  {i}. [{r['score']:.3f}] {r['title']}")
            print(f"     {r['content'][:120]}...")
        print()


if __name__ == "__main__":
    main()
