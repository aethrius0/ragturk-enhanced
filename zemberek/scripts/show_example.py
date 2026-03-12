import json

with open("cultural_selected/1048_Feodosia.json", "r", encoding="utf-8") as f:
    doc = json.load(f)

print("=== MAKALE ===")
print("Baslik:", doc["article"]["title"])
print()

print("=== CHUNKS (dokuman parcalari) ===")
for ch in doc["chunks"][:4]:
    cid = ch["chunk_id"]
    content = ch["content"][:200]
    print(f"  chunk_id: {cid}")
    print(f"  icerik: {content}...")
    print()

print("=== SORULAR ve GOLD ANSWER ===")
for i, item in enumerate(doc["questions"]["items"]):
    q = item["question"]
    a = item["answer"][:200]
    rc = item["related_chunk_ids"]
    cat = item["category"]
    print(f"Soru {i+1}: {q}")
    print(f"Cevap: {a}")
    print(f"Kategori: {cat}")
    print(f"related_chunk_ids (GOLD ANSWER): {rc}")
    print(f"  -> Bu sorunun cevabi {len(rc)} chunk icinde bulunuyor")
    print()
