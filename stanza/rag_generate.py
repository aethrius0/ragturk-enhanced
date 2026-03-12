"""
RAGTurk - RAG Generation (Stanza)
==================================
Top-3 chunk'ı GPT-4o mini'ye vererek cevap üretir.
Her index tipi (raw, lemma, hybrid) için ayrı ayrı çalışır.
"""
import sys, os, json, pickle, time, tempfile, shutil
os.environ["HF_HUB_OFFLINE"] = "1"

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv

# ── Paths ──────────────────────────────────────────────
ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CULTURAL = os.path.join(ROOT, "cultural_selected")
VECTOR   = os.path.join(ROOT, "stanza", "vector_db")
RESULTS  = os.path.join(ROOT, "stanza", "results")
STANZA_DIR = os.path.join(ROOT, "stanza")
os.makedirs(RESULTS, exist_ok=True)

MODEL = "intfloat/multilingual-e5-base"
TOP_K = 3
LLM_MODEL = "gpt-4o-mini"

# ── .env & OpenAI ──────────────────────────────────────
load_dotenv(os.path.join(ROOT, ".env"))
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# ── Stanza ─────────────────────────────────────────────
import stanza

print("Stanza yukleniyor...", flush=True)
nlp = stanza.Pipeline('tr', processors='tokenize,lemma', verbose=False, use_gpu=False)
print("Stanza hazir.", flush=True)

# ── Classifier ────────────────────────────────────────
sys.path.insert(0, STANZA_DIR)
from turkish_query_classifier_stanza import TurkishQueryClassifierStanza

classifier = TurkishQueryClassifierStanza()


def lemmatize(text: str) -> str:
    try:
        doc = nlp(text)
        lemmas = []
        for sent in doc.sentences:
            for word in sent.words:
                lemmas.append(word.lemma.lower() if word.lemma else word.text.lower())
        return " ".join(lemmas)
    except Exception:
        return text.lower()


def load_questions():
    """Tum sorgulari yukle ve classifier ile etiketle."""
    questions = []
    files = sorted(f for f in os.listdir(CULTURAL) if f.endswith(".json"))
    total_files = len(files)

    for fi, fname in enumerate(files, 1):
        path = os.path.join(CULTURAL, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                doc = json.load(f)
        except Exception:
            continue

        article_id = doc.get("article", {}).get("id", fname.replace(".json", ""))
        q_data = doc.get("questions", {})
        items = q_data.get("items", [])

        for item in items:
            question = item.get("question", "")
            answer = item.get("answer", "")
            related_chunks = item.get("related_chunk_ids", [])
            category = item.get("category", "")

            global_chunk_ids = [f"{article_id}_{cid}" for cid in related_chunks]

            clf_result = classifier.classify(question)
            clf_labels = clf_result.labels

            questions.append({
                "question": question,
                "answer": answer,
                "ground_truth": global_chunk_ids,
                "category": category,
                "clf_labels": clf_labels,
                "article_id": article_id
            })

        if fi % 50 == 0 or fi == total_files:
            print(f"  {fi}/{total_files} dosya, {len(questions)} sorgu islendi...", flush=True)

    return questions


def build_prompt(question: str, chunks: list[dict]) -> str:
    """Soru + top-3 chunk'tan LLM promptu olustur."""
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(f"[Kaynak {i}]\n{chunk['content']}")

    context = "\n\n".join(context_parts)

    return (
        "Aşağıda bir soruya cevap vermek için kullanılabilecek kaynak metinler verilmiştir.\n"
        "Bu kaynaklara dayanarak soruyu Türkçe olarak cevaplayın.\n"
        "Cevabınız kısa ve öz olsun.\n\n"
        f"### Kaynaklar\n{context}\n\n"
        f"### Soru\n{question}\n\n"
        "### Cevap"
    )


def call_llm(prompt: str, max_retries: int = 20) -> str:
    """GPT-4o mini'yi cagir (retry destekli)."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=512,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            wait = min(2 ** attempt, 120)
            print(f"    API hatasi (deneme {attempt+1}/{max_retries}): {e}", flush=True)
            print(f"    {wait}s bekleniyor...", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"{max_retries} denemede de basarisiz oldu.")


def main():
    print("=" * 60)
    print("  RAGTurk - RAG Generation (Stanza)")
    print("=" * 60)

    # Embedding model
    print(f"\n  Embedding model: {MODEL}", flush=True)
    emb_model = SentenceTransformer(MODEL)
    print("  Model yuklendi.", flush=True)

    # Metadata
    meta_path = os.path.join(VECTOR, "metadata.pkl")
    with open(meta_path, "rb") as f:
        metadata = pickle.load(f)
    print(f"  {len(metadata)} chunk metadata yuklendi.", flush=True)

    # Sorgular
    print("\n  Sorgular yukleniyor...", flush=True)
    questions = load_questions()
    print(f"  {len(questions)} sorgu yuklendi.", flush=True)

    # Index'ler
    index_configs = [
        ("raw_faiss.index", "raw"),
        ("lemma_faiss.index", "lemma"),
        ("hybrid_faiss.index", "hybrid"),
    ]

    for index_file, mode in index_configs:
        index_path = os.path.join(VECTOR, index_file)
        if not os.path.exists(index_path):
            print(f"\n  UYARI: {index_file} bulunamadi, atlaniyor.")
            continue

        out_file = os.path.join(RESULTS, f"rag_generation_{mode}.json")
        if os.path.exists(out_file):
            with open(out_file, "r", encoding="utf-8") as f:
                existing = json.load(f)
            if len(existing) >= 2558:
                print(f"\n  {mode.upper()} zaten tamamlanmis ({len(existing)} sonuc), atlaniyor.")
                continue

        print(f"\n{'='*60}")
        print(f"  INDEX: {mode.upper()} — Generation")
        print("=" * 60)

        # Index yukle
        temp_index = os.path.join(tempfile.gettempdir(), f"ragturk_{index_file}")
        shutil.copy(index_path, temp_index)
        index = faiss.read_index(temp_index)
        os.remove(temp_index)
        print(f"  Index yuklendi: {index.ntotal} vektor", flush=True)

        # Sorgulari encode et
        texts = []
        for q in questions:
            text = q["question"]
            if mode == "lemma":
                text = lemmatize(text)
            elif mode == "hybrid":
                text = text + "\n" + lemmatize(text)
            texts.append("query: " + text)

        print(f"  {len(texts)} sorgu encode ediliyor ({mode})...", flush=True)
        query_embeddings = emb_model.encode(
            texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True
        ).astype(np.float32)

        # Top-3 arama
        print(f"  Top-{TOP_K} arama yapiliyor...", flush=True)
        distances, indices = index.search(query_embeddings, TOP_K)

        # Generation
        generated = []
        out_file = os.path.join(RESULTS, f"rag_generation_{mode}.json")
        print(f"  LLM generation basliyor...", flush=True)
        t0 = time.time()

        for qi in range(len(questions)):
            q = questions[qi]
            top_indices = indices[qi]

            # Top-3 chunk bilgilerini al
            top_chunks = []
            for idx in top_indices:
                if 0 <= idx < len(metadata):
                    top_chunks.append(metadata[idx])

            prompt = build_prompt(q["question"], top_chunks)
            llm_answer = call_llm(prompt)

            generated.append({
                "question": q["question"],
                "gold_answer": q["answer"],
                "llm_answer": llm_answer,
                "ground_truth": q["ground_truth"],
                "category": q["category"],
                "clf_labels": q["clf_labels"],
                "article_id": q["article_id"],
                "retrieved_chunks": [
                    {"global_id": m.get("global_id", ""), "content": m.get("content", "")}
                    for m in top_chunks
                ],
            })

            if (qi + 1) % 50 == 0 or (qi + 1) == len(questions):
                with open(out_file, "w", encoding="utf-8") as f:
                    json.dump(generated, f, ensure_ascii=False, indent=2)
                elapsed = time.time() - t0
                qps = (qi + 1) / elapsed if elapsed > 0 else 0
                print(
                    f"    [{mode.upper()}] {qi+1}/{len(questions)}  "
                    f"({elapsed:.0f}s, {qps:.1f} q/s)",
                    flush=True,
                )

        print(f"  {mode.upper()} tamamlandi: {len(generated)} sonuc.", flush=True)

    print("\n  Generation bitti!")


if __name__ == "__main__":
    main()
