"""
RAGTurk – 3-Index Embedding + FAISS (raw / lemma / hybrid) - STANZA VERSION
============================================================================
cultural_selected/ içindeki belgelerin chunk'larını 3 farklı metin
temsiliyle encode edip stanza/vector_db/ altına 3 ayrı FAISS index kaydeder:

  1. raw_faiss.index        → Orijinal metin
  2. lemma_faiss.index      → Stanza ile lemmatize edilmiş metin
  3. hybrid_faiss.index     → raw + "\\n" + lemma

Tüm index'ler için ortak metadata.pkl ve stats.json yazılır.
"""
import sys, os, re, json, time, pickle, tempfile, shutil
os.environ["HF_HUB_OFFLINE"] = "1"

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ── Paths ──────────────────────────────────────────────
ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CULTURAL = os.path.join(ROOT, "cultural_selected")
OUT      = os.path.join(ROOT, "stanza", "vector_db")
os.makedirs(OUT, exist_ok=True)

MODEL = "intfloat/multilingual-e5-base"   # 768-d, Türkçe destekli
BATCH = 32

# ── Stanza ──────────────────────────────────────────────
import stanza

print("Stanza yukleniyor...", flush=True)
nlp = stanza.Pipeline('tr', processors='tokenize,lemma', verbose=False, use_gpu=False)
print("Stanza hazir.", flush=True)


# ═══════════════════════════════════════════════════════
#  Lemmatization
# ═══════════════════════════════════════════════════════
def lemmatize(text: str) -> str:
    """Stanza ile metni lemmatize et. Her kelimeyi kök formuna çevir."""
    try:
        doc = nlp(text)
        lemmas = []
        for sent in doc.sentences:
            for word in sent.words:
                lemmas.append(word.lemma.lower() if word.lemma else word.text.lower())
        return " ".join(lemmas)
    except Exception:
        return text.lower()


# ═══════════════════════════════════════════════════════
#  Chunk yükleme
# ═══════════════════════════════════════════════════════
def load_chunks():
    """Tüm dokümanlardan chunk'ları yükle ve 3 metin versiyonunu üret."""
    raw_texts = []
    lemma_texts = []
    hybrid_texts = []
    meta = []

    files = sorted(f for f in os.listdir(CULTURAL) if f.endswith(".json"))
    total_files = len(files)
    print(f"  {total_files} dosya okunuyor...", flush=True)
    t0 = time.time()

    for fi, fn in enumerate(files):
        try:
            data = json.load(open(os.path.join(CULTURAL, fn), "r", encoding="utf-8"))
        except Exception:
            continue

        article_id = data.get("article", {}).get("id", fn.replace(".json", ""))
        title = data.get("article", data).get("title", "")
        chunks = data.get("chunks", [])

        pieces = []
        if chunks:
            for ch in chunks:
                c = ch.get("content", "")
                if len(c) > 50:
                    pieces.append((c, ch.get("id", ""), ch.get("section", "")))
        else:
            c = data.get("article", data).get("content", "")
            if len(c) > 50:
                pieces.append((c, "full", "article"))

        for content, chunk_id, section in pieces:
            raw = content[:500]
            lem = lemmatize(content[:500])
            hyb = raw + "\n" + lem

            raw_texts.append(f"passage: {raw}")
            lemma_texts.append(f"passage: {lem}")
            hybrid_texts.append(f"passage: {hyb}")
            meta.append({
                "file": fn,
                "title": title,
                "article_id": article_id,
                "chunk_id": chunk_id,
                "global_id": f"{article_id}_{chunk_id}",
                "section": section,
                "content": content
            })

        if (fi + 1) % 10 == 0 or (fi + 1) == total_files:
            elapsed = time.time() - t0
            eta = (total_files - fi - 1) * elapsed / (fi + 1)
            print(f"    [{fi+1:4d}/{total_files}] {len(meta):5d} chunk | {elapsed:.0f}s gecti | ~{eta/60:.1f}dk kaldi", flush=True)

    return raw_texts, lemma_texts, hybrid_texts, meta


# ═══════════════════════════════════════════════════════
#  Encoding
# ═══════════════════════════════════════════════════════
def encode_texts(model, texts, label):
    """Metin listesini batch halinde encode et."""
    n = len(texts)
    print(f"\n  [{label}] {n} chunk encode ediliyor...", flush=True)
    t0 = time.time()
    parts = []
    for i in range(0, n, BATCH):
        batch = texts[i:i + BATCH]
        emb = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        parts.append(emb)
        done = min(i + BATCH, n)
        el = time.time() - t0
        eta = (n - done) * el / max(done, 1) / 60
        if (i // BATCH) % 10 == 0 or done == n:
            print(f"    [{done:>5}/{n}]  {el:.0f}s  ~{eta:.1f}dk kaldı", flush=True)

    embeddings = np.vstack(parts).astype("float32")
    dur = time.time() - t0
    print(f"  [{label}] Bitti: {dur:.0f}s  shape={embeddings.shape}", flush=True)
    return embeddings


# ═══════════════════════════════════════════════════════
#  FAISS kaydet
# ═══════════════════════════════════════════════════════
def save_faiss(embeddings, name):
    """FAISS index oluştur ve kaydet (OneDrive Türkçe path fix)."""
    idx = faiss.IndexFlatIP(embeddings.shape[1])
    idx.add(embeddings)
    tmp = os.path.join(tempfile.gettempdir(), f"ragturk_{name}.index")
    faiss.write_index(idx, tmp)
    dest = os.path.join(OUT, f"{name}.index")
    shutil.move(tmp, dest)
    print(f"  -> {dest}", flush=True)
    return idx


# ═══════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  RAGTurk – 3-Index Embedding + FAISS")
    print("  (raw / lemma / hybrid)")
    print("=" * 60, flush=True)

    # Model yükle
    print(f"\n  Model: {MODEL}", flush=True)
    model = SentenceTransformer(MODEL, local_files_only=True)
    print("  Model yüklendi.", flush=True)

    # Chunk'ları yükle + lemmatize
    print("\n  Chunk'lar yükleniyor ve lemmatize ediliyor...", flush=True)
    t_load = time.time()
    raw_texts, lemma_texts, hybrid_texts, meta = load_chunks()
    n = len(meta)
    dur_load = time.time() - t_load
    print(f"\n  {n} chunk bulundu ({dur_load:.0f}s).", flush=True)

    if n == 0:
        print("  HATA: Chunk bulunamadı!")
        return

    # 3 farklı encoding
    emb_raw    = encode_texts(model, raw_texts,    "RAW")
    emb_lemma  = encode_texts(model, lemma_texts,  "LEMMA")
    emb_hybrid = encode_texts(model, hybrid_texts, "HYBRID")

    # FAISS index'leri kaydet
    print("\n  FAISS index'leri kaydediliyor...", flush=True)
    idx_raw    = save_faiss(emb_raw,    "raw_faiss")
    idx_lemma  = save_faiss(emb_lemma,  "lemma_faiss")
    idx_hybrid = save_faiss(emb_hybrid, "hybrid_faiss")

    # Metadata ve embedding'leri kaydet
    pickle.dump(meta, open(os.path.join(OUT, "metadata.pkl"), "wb"))
    np.save(os.path.join(OUT, "embeddings_raw.npy"),    emb_raw)
    np.save(os.path.join(OUT, "embeddings_lemma.npy"),  emb_lemma)
    np.save(os.path.join(OUT, "embeddings_hybrid.npy"), emb_hybrid)

    json.dump({
        "chunks": n,
        "dim": int(emb_raw.shape[1]),
        "model": MODEL,
        "files": len(set(m["file"] for m in meta)),
        "indices": ["raw_faiss", "lemma_faiss", "hybrid_faiss"],
        "date": time.strftime("%Y-%m-%d %H:%M"),
    }, open(os.path.join(OUT, "stats.json"), "w", encoding="utf-8"),
       ensure_ascii=False, indent=2)

    n_files = len(set(m["file"] for m in meta))
    print(f"\n{'='*60}")
    print(f"  TAMAMLANDI – {n} chunk, {n_files} dosya, 3 index")
    print(f"  Çıktı: {OUT}")
    print(f"{'='*60}", flush=True)

    # Hızlı test (hybrid index üzerinden)
    print("\n  Hızlı arama testi (hybrid index):", flush=True)
    for q in ["Osmanlı İmparatorluğu tarihi",
              "Türk müziği ve geleneksel enstrümanlar",
              "Anadolu uygarlıkları ve arkeolojik kazılar"]:
        qe = model.encode([f"query: {q}"], normalize_embeddings=True).astype("float32")
        sc, ix = idx_hybrid.search(qe, 3)
        print(f"\n  Sorgu: {q}")
        for j, (ii, s) in enumerate(zip(ix[0], sc[0]), 1):
            print(f"    {j}. [{s:.3f}] {meta[ii]['title'][:50]} – {meta[ii]['content'][:80]}...")


if __name__ == "__main__":
    main()
