"""
RAGTurk - Semantic Search
=========================
FAISS index kullanarak kültürel belgelerde arama yapar.
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"  # Offline mod - cache'den yükle

import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VECTOR_DB = os.path.join(BASE_DIR, "vector_db")

# Model
MODEL_NAME = "intfloat/multilingual-e5-base"

class CulturalSearch:
    def __init__(self):
        import tempfile
        import shutil
        
        print("Model yükleniyor (cache)...")
        self.model = SentenceTransformer(MODEL_NAME, local_files_only=True)
        
        print("FAISS index yükleniyor...")
        index_path = os.path.join(VECTOR_DB, "cultural_faiss.index")
        
        # Türkçe path sorunu için temp'e kopyala
        temp_dir = tempfile.gettempdir()
        temp_index = os.path.join(temp_dir, "cultural_faiss_read.index")
        shutil.copy(index_path, temp_index)
        self.index = faiss.read_index(temp_index)
        
        print("Metadata yükleniyor...")
        metadata_path = os.path.join(VECTOR_DB, "metadata.pkl")
        with open(metadata_path, "rb") as f:
            self.metadata = pickle.load(f)
        
        print(f"Hazır! {self.index.ntotal} chunk aranabilir.\n")
    
    def search(self, query, top_k=5):
        """Sorgu ile en yakın belgeleri bul"""
        # E5 modeli için query prefix
        query_text = f"query: {query}"
        query_embedding = self.model.encode([query_text], normalize_embeddings=True).astype('float32')
        
        # FAISS arama
        scores, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for idx, score in zip(indices[0], scores[0]):
            meta = self.metadata[idx]
            results.append({
                "score": float(score),
                "title": meta["title"],
                "section": meta["section"],
                "content": meta["content"],
                "file": meta["file"]
            })
        
        return results
    
    def print_results(self, results):
        """Sonuçları güzel formatta yazdır"""
        print("-" * 60)
        for i, r in enumerate(results, 1):
            print(f"\n{i}. [{r['score']:.4f}] {r['title']}")
            print(f"   Bölüm: {r['section']}")
            print(f"   {r['content'][:200]}...")
        print("\n" + "-" * 60)


def main():
    searcher = CulturalSearch()
    
    print("=" * 60)
    print("RAGTurk Semantic Search")
    print("Çıkmak için 'q' yazın")
    print("=" * 60)
    
    while True:
        print()
        query = input("Sorgu: ").strip()
        
        if query.lower() in ['q', 'quit', 'exit', 'çık']:
            print("Görüşürüz!")
            break
        
        if not query:
            continue
        
        results = searcher.search(query, top_k=5)
        searcher.print_results(results)


if __name__ == "__main__":
    main()
