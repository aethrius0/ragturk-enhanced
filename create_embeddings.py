"""
RAGTurk - Multilingual E5 Embedding + FAISS Index
==================================================
Seçilen kültürel belgeler için embedding oluşturur ve FAISS index'e kaydeder.
"""
import sys, os
os.environ["HF_HUB_OFFLINE"] = "1"  # Offline mod

import json, time, pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
print("Modüller yüklendi.", flush=True)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CULTURAL_PATH = os.path.join(BASE_DIR, "cultural_selected")
OUTPUT_DIR = os.path.join(BASE_DIR, "vector_db")

# Klasörü oluştur
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"Klasör oluşturuldu: {OUTPUT_DIR}")

# Model - Multilingual E5
MODEL_NAME = "intfloat/multilingual-e5-base"  # 768 dim, Türkçe desteği

def load_documents():
    """Kültürel belgelerden chunk'ları yükle"""
    documents = []
    metadata = []
    
    files = [f for f in os.listdir(CULTURAL_PATH) if f.endswith(".json") and not f.startswith("_")]
    print(f"Toplam {len(files)} dosya yükleniyor...")
    
    for file in files:
        file_path = os.path.join(CULTURAL_PATH, file)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            title = ""
            if "article" in data:
                title = data["article"].get("title", "")
            
            # Chunk'ları al
            if "chunks" in data:
                for chunk in data["chunks"]:
                    content = chunk.get("content", "")
                    if content and len(content) > 50:  # Çok kısa chunk'ları atla
                        # E5 modeli için prefix ekle
                        text = f"passage: {content}"
                        documents.append(text)
                        metadata.append({
                            "file": file,
                            "title": title,
                            "chunk_id": chunk.get("id", ""),
                            "section": chunk.get("section", ""),
                            "content": content  # Orijinal içerik
                        })
            
            # Eğer chunk yoksa article content kullan
            elif "article" in data and "content" in data["article"]:
                content = data["article"]["content"]
                if content and len(content) > 50:
                    text = f"passage: {content}"
                    documents.append(text)
                    metadata.append({
                        "file": file,
                        "title": title,
                        "chunk_id": "full",
                        "section": "article",
                        "content": content
                    })
                    
        except Exception as e:
            print(f"Hata {file}: {e}")
            continue
    
    return documents, metadata

def create_embeddings(documents, model):
    """Embedding'leri oluştur"""
    print(f"\n{len(documents)} chunk için embedding oluşturuluyor...")
    print(f"Model: {MODEL_NAME}")
    
    start_time = time.time()
    
    # Batch halinde embedding oluştur
    batch_size = 32
    all_embeddings = []
    
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        embeddings = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        all_embeddings.append(embeddings)
        
        if (i + batch_size) % 200 == 0 or i + batch_size >= len(documents):
            elapsed = time.time() - start_time
            progress = min(i + batch_size, len(documents))
            print(f"  İşlenen: {progress}/{len(documents)} | Geçen süre: {elapsed:.1f}s")
    
    embeddings = np.vstack(all_embeddings).astype('float32')
    
    total_time = time.time() - start_time
    print(f"\nEmbedding tamamlandı: {total_time:.1f} saniye")
    print(f"Embedding boyutu: {embeddings.shape}")
    
    return embeddings

def create_faiss_index(embeddings):
    """FAISS index oluştur"""
    print("\nFAISS index oluşturuluyor...")
    
    dimension = embeddings.shape[1]
    
    # IndexFlatIP - Inner Product (cosine similarity için normalize edilmiş vektörler)
    index = faiss.IndexFlatIP(dimension)
    
    # Vektörleri ekle
    index.add(embeddings)
    
    print(f"Index oluşturuldu: {index.ntotal} vektör, {dimension} boyut")
    
    return index

def save_index(index, metadata, embeddings):
    """Index ve metadata'yı kaydet"""
    import tempfile
    import shutil
    
    # Klasörün var olduğundan emin ol
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Kayıt klasörü: {OUTPUT_DIR}")
    
    # FAISS - temp dizine kaydet sonra taşı (OneDrive/Türkçe path sorunu için)
    temp_dir = tempfile.gettempdir()
    temp_index = os.path.join(temp_dir, "cultural_faiss.index")
    final_index = os.path.join(OUTPUT_DIR, "cultural_faiss.index")
    
    print(f"FAISS index kaydediliyor (temp): {temp_index}")
    faiss.write_index(index, temp_index)
    shutil.move(temp_index, final_index)
    print(f"FAISS index taşındı: {final_index}")
    
    # Metadata kaydet
    metadata_path = os.path.join(OUTPUT_DIR, "metadata.pkl")
    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)
    print(f"Metadata kaydedildi: {metadata_path}")
    
    # Embeddings kaydet (opsiyonel - debug için)
    embeddings_path = os.path.join(OUTPUT_DIR, "embeddings.npy")
    np.save(embeddings_path, embeddings)
    print(f"Embeddings kaydedildi: {embeddings_path}")
    
    # İstatistikler
    stats = {
        "total_chunks": len(metadata),
        "embedding_dim": embeddings.shape[1],
        "model": MODEL_NAME,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "unique_files": len(set(m["file"] for m in metadata))
    }
    
    stats_path = os.path.join(OUTPUT_DIR, "index_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"İstatistikler kaydedildi: {stats_path}")
    
    return stats

def test_search(index, metadata, model, query="Osmanlı İmparatorluğu tarihi"):
    """Örnek sorgu testi"""
    print(f"\n{'='*60}")
    print("TEST SORGUSU")
    print(f"{'='*60}")
    print(f"Sorgu: {query}\n")
    
    # E5 için query prefix
    query_text = f"query: {query}"
    query_embedding = model.encode([query_text], normalize_embeddings=True).astype('float32')
    
    # Top 5 sonuç
    k = 5
    scores, indices = index.search(query_embedding, k)
    
    print("En yakın 5 sonuç:")
    print("-" * 40)
    
    for i, (idx, score) in enumerate(zip(indices[0], scores[0])):
        meta = metadata[idx]
        print(f"\n{i+1}. Skor: {score:.4f}")
        print(f"   Başlık: {meta['title']}")
        print(f"   Bölüm: {meta['section']}")
        print(f"   İçerik: {meta['content'][:150]}...")

def main():
    print("=" * 60)
    print("RAGTurk - Multilingual E5 + FAISS")
    print("=" * 60)
    
    # 1. Model yükle (cache'den, offline)
    print(f"\nModel yükleniyor: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME, local_files_only=True)
    print("Model yüklendi!")
    
    # 2. Belgeleri yükle  
    documents, metadata = load_documents()
    print(f"Yüklenen chunk sayısı: {len(documents)}")
    
    if len(documents) == 0:
        print("HATA: Hiç belge bulunamadı!")
        return
    
    # 3. Embedding oluştur
    embeddings = create_embeddings(documents, model)
    
    # 4. FAISS index oluştur
    index = create_faiss_index(embeddings)
    
    # 5. Kaydet
    stats = save_index(index, metadata, embeddings)
    
    print(f"\n{'='*60}")
    print("TAMAMLANDI!")
    print(f"{'='*60}")
    print(f"Toplam chunk: {stats['total_chunks']}")
    print(f"Benzersiz dosya: {stats['unique_files']}")
    print(f"Embedding boyutu: {stats['embedding_dim']}")
    print(f"Çıktı klasörü: {OUTPUT_DIR}")
    
    # 6. Test
    test_search(index, metadata, model)
    
    # İkinci test
    test_search(index, metadata, model, query="Türk müziği ve geleneksel enstrümanlar") 

if __name__ == "__main__":
    main()
