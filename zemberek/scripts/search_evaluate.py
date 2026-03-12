"""
RAGTurk - Search & Evaluation - ZEMBEREK VERSION
=================================================
FAISS index'lerinde sorgu arar, top-k chunk bulur ve 
ground truth ile karşılaştırıp metrikler hesaplar.

Metrikler:
  - Recall@k: İlgili chunk'ların kaçta kaçı top-k'da bulundu
  - MRR (Mean Reciprocal Rank): İlk doğru sonucun sırası
  - Precision@k: Top-k sonuçların kaçta kaçı doğru
  - Hit@k: En az 1 doğru sonuç bulundu mu

Kategoriler:
  - FACTUAL / INTERPRETATION (veri kaynaklı)
  - entity-heavy / long-morph / cultural / other (classifier kaynaklı)
"""
import sys, os, json, pickle, time, tempfile, shutil
os.environ["HF_HUB_OFFLINE"] = "1"

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ── Paths ──────────────────────────────────────────────
ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CULTURAL = os.path.join(ROOT, "cultural_selected")
VECTOR   = os.path.join(ROOT, "vector_db")
RESULTS  = os.path.join(ROOT, "results")
os.makedirs(RESULTS, exist_ok=True)

MODEL = "intfloat/multilingual-e5-base"
TOP_K_VALUES = [1, 3, 5, 10, 20]

# ── JPype + Zemberek ──────────────────────────────────
import re
import jpype, jpype.imports

JAR     = os.path.join(ROOT, "lib", "zemberek-full.jar")
JVM_DLL = r"C:\Program Files\Java\jdk-17\bin\server\jvm.dll"
_WORD   = re.compile(r'[\wçğıöşüÇĞİÖŞÜ]+')

if not jpype.isJVMStarted():
    jpype.startJVM(JVM_DLL, classpath=[JAR])

from zemberek.morphology import TurkishMorphology
morphology = TurkishMorphology.createWithDefaults()
print("Zemberek hazir.", flush=True)

# ── Classifier ────────────────────────────────────────
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS_DIR)
from turkish_query_classifier_option_b import TurkishQueryClassifierOptionB

print("Query Classifier yukleniyor...", flush=True)
classifier = TurkishQueryClassifierOptionB()
print("Classifier hazir.", flush=True)


def lemmatize(text: str) -> str:
    """Zemberek ile metni lemmatize et."""
    words = _WORD.findall(text)
    out = []
    for w in words:
        if len(w) < 2:
            out.append(w)
            continue
        try:
            results = list(morphology.analyze(w))
            if not results:
                out.append(w.lower())
                continue
            best = results[0]
            lems = list(best.getLemmas())
            lemma = str(lems[0]).lower() if lems else w.lower()
            out.append(lemma)
        except Exception:
            out.append(w.lower())
    return " ".join(out)


def load_questions():
    """Tum sorgulari yukle ve classifier ile etiketle."""
    questions = []
    files = sorted(f for f in os.listdir(CULTURAL) if f.endswith(".json"))
    total_q = 0
    
    for fi, fname in enumerate(files):
        path = os.path.join(CULTURAL, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                doc = json.load(f)
        except:
            continue
        
        article_id = doc.get("article", {}).get("id", fname.replace(".json", ""))
        q_data = doc.get("questions", {})
        items = q_data.get("items", [])
        
        for item in items:
            question = item.get("question", "")
            answer = item.get("answer", "")
            related_chunks = item.get("related_chunk_ids", [])
            category = item.get("category", "")
            
            # Global chunk ID olustur: article_id + chunk_id
            global_chunk_ids = [f"{article_id}_{cid}" for cid in related_chunks]
            
            # Classifier ile etiketle
            clf_result = classifier.classify(question)
            clf_labels = clf_result.labels  # entity-heavy, long-morph, cultural, other
            
            questions.append({
                "question": question,
                "answer": answer,
                "ground_truth": global_chunk_ids,
                "category": category,  # FACTUAL / INTERPRETATION
                "clf_labels": clf_labels,  # entity-heavy, long-morph, cultural, other
                "article_id": article_id
            })
            total_q += 1
        
        if (fi + 1) % 50 == 0:
            print(f"    {fi+1}/{len(files)} dosya, {total_q} sorgu islendi...", flush=True)
    
    return questions


def encode_queries(model, questions, mode="raw"):
    """Sorgulari encode et. mode: raw, lemma, hybrid"""
    texts = []
    for q in questions:
        text = q["question"]
        if mode == "lemma":
            text = lemmatize(text)
        elif mode == "hybrid":
            text = text + "\n" + lemmatize(text)
        texts.append("query: " + text)  # E5 modeli icin prefix
    
    print(f"  {len(texts)} sorgu encode ediliyor ({mode})...", flush=True)
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
    return embeddings


def search_faiss(index, query_embeddings, k):
    """FAISS'te arama yap, top-k sonuc don."""
    distances, indices = index.search(query_embeddings, k)
    return distances, indices


def calculate_metrics(retrieved_ids_list, ground_truth_list, k):
    """
    Metrikleri hesapla:
    - Recall@k
    - Precision@k  
    - MRR
    - Hit@k
    """
    recalls = []
    precisions = []
    reciprocal_ranks = []
    hits = []
    
    for retrieved, gt in zip(retrieved_ids_list, ground_truth_list):
        retrieved_set = set(retrieved[:k])
        gt_set = set(gt)
        
        if len(gt_set) == 0:
            continue
        
        # Recall: kac tane GT chunk bulundu / toplam GT
        found = len(retrieved_set & gt_set)
        recall = found / len(gt_set)
        recalls.append(recall)
        
        # Precision: kac tane dogru / k
        precision = found / k
        precisions.append(precision)
        
        # MRR: ilk dogru sonucun sirasi
        rr = 0
        for i, rid in enumerate(retrieved[:k]):
            if rid in gt_set:
                rr = 1 / (i + 1)
                break
        reciprocal_ranks.append(rr)
        
        # Hit: en az 1 dogru var mi
        hit = 1 if found > 0 else 0
        hits.append(hit)
    
    return {
        f"Recall@{k}": np.mean(recalls) if recalls else 0,
        f"Precision@{k}": np.mean(precisions) if precisions else 0,
        f"MRR@{k}": np.mean(reciprocal_ranks) if reciprocal_ranks else 0,
        f"Hit@{k}": np.mean(hits) if hits else 0,
        "num_queries": len(recalls)
    }


def main():
    print("=" * 60)
    print("  RAGTurk - Search & Evaluation")
    print("=" * 60)
    
    # Model yukle
    print(f"\n  Model: {MODEL}", flush=True)
    model = SentenceTransformer(MODEL)
    print("  Model yuklendi.", flush=True)
    
    # Metadata yukle
    meta_path = os.path.join(VECTOR, "metadata.pkl")
    if not os.path.exists(meta_path):
        print("HATA: metadata.pkl bulunamadi! Once create_embeddings.py calistir.")
        return
    
    with open(meta_path, "rb") as f:
        metadata = pickle.load(f)
    print(f"  {len(metadata)} chunk metadata yuklendi.", flush=True)
    
    # Chunk ID listesi olustur - dosyadaki article_id'yi kullan
    print("  Chunk ID'leri olusturuluyor...", flush=True)
    chunk_ids = []
    article_id_cache = {}  # dosya adi -> article_id
    
    for m in metadata:
        fname = m.get("file", "")
        chunk_id = m.get("chunk_id", "")
        
        # Cache'te yoksa dosyadan oku
        if fname not in article_id_cache:
            fpath = os.path.join(CULTURAL, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    doc = json.load(f)
                article_id_cache[fname] = doc.get("article", {}).get("id", fname.replace(".json", ""))
            except:
                article_id_cache[fname] = fname.replace(".json", "")
        
        article_id = article_id_cache[fname]
        global_id = f"{article_id}_{chunk_id}"
        chunk_ids.append(global_id)
    
    # Sorgulari yukle
    print("\n  Sorgular yukleniyor...", flush=True)
    questions = load_questions()
    print(f"  {len(questions)} sorgu yuklendi.", flush=True)
    
    # Index'leri yukle ve evaluate et
    index_configs = [
        ("raw_faiss.index", "raw"),
        ("lemma_faiss.index", "lemma"),
        ("hybrid_faiss.index", "hybrid")
    ]
    
    all_results = {}
    
    for index_file, mode in index_configs:
        index_path = os.path.join(VECTOR, index_file)
        if not os.path.exists(index_path):
            print(f"\n  UYARI: {index_file} bulunamadi, atlaniyor.")
            continue
        
        print(f"\n{'='*60}")
        print(f"  INDEX: {mode.upper()}")
        print("=" * 60)
        
        # Index yukle (Turkce karakter path sorunu icin temp'e kopyala)
        temp_index = os.path.join(tempfile.gettempdir(), f"ragturk_{index_file}")
        shutil.copy(index_path, temp_index)
        index = faiss.read_index(temp_index)
        os.remove(temp_index)
        print(f"  Index yuklendi: {index.ntotal} vektor", flush=True)
        
        # Sorgulari encode et
        query_embeddings = encode_queries(model, questions, mode=mode)
        query_embeddings = query_embeddings.astype(np.float32)
        
        # Arama yap
        max_k = max(TOP_K_VALUES)
        print(f"  Top-{max_k} arama yapiliyor...", flush=True)
        distances, indices = search_faiss(index, query_embeddings, max_k)
        
        # Bulunan chunk ID'leri
        retrieved_ids_list = []
        for query_indices in indices:
            retrieved = [chunk_ids[i] if i < len(chunk_ids) else "" for i in query_indices]
            retrieved_ids_list.append(retrieved)
        
        # Ground truth
        ground_truth_list = [q["ground_truth"] for q in questions]
        categories = [q["category"] for q in questions]
        clf_labels_list = [q["clf_labels"] for q in questions]
        
        # Her k icin metrik hesapla
        mode_results = {}
        print(f"\n  Metrikler (Tum Sorgular):")
        for k in TOP_K_VALUES:
            metrics = calculate_metrics(retrieved_ids_list, ground_truth_list, k)
            mode_results[f"k={k}"] = metrics
            print(f"    k={k:2d}  Recall={metrics[f'Recall@{k}']:.4f}  "
                  f"MRR={metrics[f'MRR@{k}']:.4f}  "
                  f"Hit={metrics[f'Hit@{k}']:.4f}")
        
        # Kategori bazli metrikler (FACTUAL / INTERPRETATION)
        unique_cats = list(set(categories))
        category_results = {}
        
        for cat in unique_cats:
            if not cat:
                continue
            # Bu kategorideki sorgulari filtrele
            cat_indices = [i for i, c in enumerate(categories) if c == cat]
            cat_retrieved = [retrieved_ids_list[i] for i in cat_indices]
            cat_gt = [ground_truth_list[i] for i in cat_indices]
            
            print(f"\n  Metrikler ({cat} - {len(cat_indices)} sorgu):")
            cat_metrics = {}
            for k in TOP_K_VALUES:
                metrics = calculate_metrics(cat_retrieved, cat_gt, k)
                cat_metrics[f"k={k}"] = metrics
                print(f"    k={k:2d}  Recall={metrics[f'Recall@{k}']:.4f}  "
                      f"MRR={metrics[f'MRR@{k}']:.4f}  "
                      f"Hit={metrics[f'Hit@{k}']:.4f}")
            category_results[cat] = cat_metrics
        
        mode_results["by_category"] = category_results
        
        # Classifier label bazli metrikler (entity-heavy, long-morph, cultural, other)
        clf_label_types = ["entity-heavy", "long-morph", "cultural", "other"]
        clf_label_results = {}
        
        for label in clf_label_types:
            # Bu labele sahip sorgulari filtrele
            label_indices = [i for i, labels in enumerate(clf_labels_list) if label in labels]
            if not label_indices:
                continue
            label_retrieved = [retrieved_ids_list[i] for i in label_indices]
            label_gt = [ground_truth_list[i] for i in label_indices]
            
            print(f"\n  Metrikler ({label} - {len(label_indices)} sorgu):")
            label_metrics = {}
            for k in TOP_K_VALUES:
                metrics = calculate_metrics(label_retrieved, label_gt, k)
                label_metrics[f"k={k}"] = metrics
                print(f"    k={k:2d}  Recall={metrics[f'Recall@{k}']:.4f}  "
                      f"MRR={metrics[f'MRR@{k}']:.4f}  "
                      f"Hit={metrics[f'Hit@{k}']:.4f}")
            clf_label_results[label] = label_metrics
        
        mode_results["by_clf_label"] = clf_label_results
        all_results[mode] = mode_results
    
    # Sonuclari kaydet
    results_file = os.path.join(RESULTS, "evaluation_results.json")
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n  Sonuclar kaydedildi: {results_file}")
    
    # Ozet tablo
    print("\n" + "=" * 70)
    print("  OZET (Recall@5, MRR@5)")
    print("=" * 70)
    print("\n  [TUM SORGULAR]")
    for mode in ["raw", "lemma", "hybrid"]:
        if mode in all_results:
            r5 = all_results[mode].get("k=5", {}).get("Recall@5", 0)
            mrr5 = all_results[mode].get("k=5", {}).get("MRR@5", 0)
            print(f"    {mode.upper():8s}  Recall@5={r5:.4f}  MRR@5={mrr5:.4f}")
    
    # Kategori bazli ozet
    for cat in ["FACTUAL", "INTERPRETATION"]:
        print(f"\n  [{cat}]")
        for mode in ["raw", "lemma", "hybrid"]:
            if mode in all_results:
                cat_data = all_results[mode].get("by_category", {}).get(cat, {}).get("k=5", {})
                r5 = cat_data.get("Recall@5", 0)
                mrr5 = cat_data.get("MRR@5", 0)
                print(f"    {mode.upper():8s}  Recall@5={r5:.4f}  MRR@5={mrr5:.4f}")
    
    # Classifier label bazli ozet
    for label in ["entity-heavy", "long-morph", "cultural", "other"]:
        print(f"\n  [{label}]")
        for mode in ["raw", "lemma", "hybrid"]:
            if mode in all_results:
                label_data = all_results[mode].get("by_clf_label", {}).get(label, {}).get("k=5", {})
                r5 = label_data.get("Recall@5", 0)
                mrr5 = label_data.get("MRR@5", 0)
                n = label_data.get("num_queries", 0)
                print(f"    {mode.upper():8s}  Recall@5={r5:.4f}  MRR@5={mrr5:.4f}  ({n} sorgu)")
    
    print("\n  Tamamlandi!")


if __name__ == "__main__":
    main()
