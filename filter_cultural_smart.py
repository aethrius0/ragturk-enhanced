"""
RAGTurk Kültürel Belge Filtreleme - Akıllı NER Versiyonu
========================================================
NLTK NER kullanarak SADECE kültürel belgeleri ayıklar.
Bilimsel, teknik, spor vb. konuları eler.
"""

import os
import json
import shutil
import nltk
import time
import re
from nltk import word_tokenize, pos_tag, ne_chunk
from collections import defaultdict

# NLTK indirme
def setup_nltk():
    packages = [
        "punkt", "punkt_tab", 
        "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng",
        "maxent_ne_chunker", "maxent_ne_chunker_tab", "words"
    ]
    for pkg in packages:
        try:
            nltk.download(pkg, quiet=True)
        except:
            pass

setup_nltk()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "ragturk", "formal_5k", "dataset", "json")
OUTPUT_PATH = os.path.join(BASE_DIR, "cultural_selected")

os.makedirs(OUTPUT_PATH, exist_ok=True)

# =====================================================
# KÜLTÜREL ANAHTAR KELİMELER (yüksek öncelik)
# =====================================================
STRONG_CULTURAL_KEYWORDS = [
    # Tarih ve Medeniyet
    "osmanlı", "selçuklu", "bizans", "hitit", "frigya", "lidya", "urartu",
    "imparatorluk", "sultanat", "hanedan", "padişah", "sultan", "vezir",
    "sadrazam", "paşa", "beylik", "hanedanlık", "hilafet", "saltanat",
    
    # Müze ve Miras
    "müze", "müzesi", "anıt", "anıtkabir", "miras", "unesco", "koruma",
    "restorasyon", "arkeoloji", "arkeolojik", "kazı", "höyük", "ören",
    
    # Mimari
    "cami", "camii", "mescit", "medrese", "kervansaray", "han", "hamam",
    "kale", "kalesi", "saray", "köşk", "türbe", "külliye", "minare",
    "kubbe", "kemer", "çeşme", "köprü", "sur", "kule",
    
    # Sanat
    "minyatür", "hat", "hattat", "tezhip", "ebru", "çini", "seramik",
    "heykel", "resim", "ressam", "tablo", "fresk", "mozaik",
    
    # Edebiyat
    "şair", "şiir", "divan", "kaside", "gazel", "masal", "destan",
    "efsane", "halk edebiyatı", "tekke edebiyatı", "yazar", "roman",
    
    # Müzik
    "türkü", "şarkı", "beste", "besteci", "makam", "usul", "saz",
    "bağlama", "ud", "ney", "kanun", "kemençe", "klasik müzik",
    "halk müziği", "sanat müziği", "fasıl",
    
    # Din ve Tasavvuf
    "tasavvuf", "sufi", "mevlevi", "mevlana", "bektaşi", "tekke",
    "dergah", "tarikat", "şeyh", "derviş", "sema", "zikir",
    
    # Gelenek ve Folklor
    "folklor", "gelenek", "görenek", "adet", "örf", "halk oyunu",
    "kına", "düğün", "bayram", "kutlama", "festival", "şenlik",
    "el sanatları", "kilim", "halı", "nakış", "oya",
    
    # Mutfak
    "mutfak", "yemek kültürü", "geleneksel", "osmanlı mutfağı",
    
    # Coğrafi Kültürel Yerler
    "kapadokya", "efes", "truva", "bergama", "afrodisias",
    "göbeklitepe", "çatalhöyük", "hattuşa", "nemrut", "zeugma"
]

# Orta seviye kültürel kelimeler
MEDIUM_CULTURAL_KEYWORDS = [
    "tarih", "tarihi", "kültür", "kültürel", "sanat", "edebiyat",
    "mimari", "mimar", "antik", "medeniyet", "tören", "ritüel",
    "giysi", "kostüm", "ziyafet", "şölen", "opera", "tiyatro", "bale"
]

# =====================================================
# ELENECEK KONULAR (kara liste)
# =====================================================
BLACKLIST_KEYWORDS = [
    # Bilim ve Matematik
    "problem", "denklem", "formül", "algoritma", "teorem", "ispat",
    "hesaplama", "fonksiyon", "integral", "türev", "matris", "vektör",
    "polinom", "geometri", "cebir", "olasılık", "istatistik",
    
    # Fizik ve Kimya
    "atom", "molekül", "elektron", "proton", "nötron", "kuantum",
    "termodinamik", "elektromanyetik", "radyasyon", "izotop",
    "reaksiyon", "bileşik", "element", "periyodik",
    
    # Teknoloji ve Yazılım
    "yazılım", "donanım", "programlama", "kod", "veritabanı",
    "sunucu", "ağ", "protokol", "şifreleme", "api", "framework",
    "android", "ios", "windows", "linux", "python", "java",
    
    # Spor (genel spor haberleri)
    "maç", "lig", "sezon", "playoff", "şampiyonluk", "turnuva",
    "futbol", "basketbol", "voleybol", "tenis", "formula",
    "olimpiyat", "fifa", "uefa", "nba", "transfer",
    
    # Video Oyunları
    "video oyun", "playstation", "xbox", "nintendo", "steam",
    "rpg", "fps", "mmorpg", "esport",
    
    # Güncel Politika (kültürel olmayan)
    "seçim", "oy", "referandum", "parti", "koalisyon",
    "cumhurbaşkanlığı seçimi", "genel seçim", "yerel seçim",
    
    # Ekonomi ve Finans
    "borsa", "hisse", "döviz", "enflasyon", "faiz", "kredi",
    "banka", "yatırım", "piyasa",
    
    # Tıp (genel)
    "hastalık", "sendrom", "tedavi", "ilaç", "ameliyat",
    "enfeksiyon", "virüs", "bakteri", "tanı", "semptom"
]

# NER için maksimum karakter
MAX_CHARS_FOR_NER = 4000

def analyze_text_with_ner(text):
    """NLTK NER ile metin analizi"""
    try:
        full_tokens = text.split()
        full_token_count = len(full_tokens)
        
        if full_token_count <= 2:
            return None
        
        text_for_ner = text[:MAX_CHARS_FOR_NER]
        tokens = word_tokenize(text_for_ner)
        tagged = pos_tag(tokens)
        entities = ne_chunk(tagged)
        
        named_entities = []
        multi_token_entities = []
        
        for subtree in entities:
            if hasattr(subtree, 'label'):
                entity_tokens = [token for token, pos in subtree]
                entity = " ".join(entity_tokens)
                entity_type = subtree.label()
                
                named_entities.append((entity, entity_type))
                
                if len(entity_tokens) > 2:
                    multi_token_entities.append((entity, entity_type))
        
        return {
            "token_count": full_token_count,
            "entities": named_entities,
            "multi_token_entities": multi_token_entities,
            "entity_count": len(named_entities),
            "multi_token_count": len(multi_token_entities)
        }
    except:
        return None

def is_blacklisted(text, title):
    """Kara listedeki konuları kontrol et"""
    text_lower = text.lower()
    title_lower = title.lower()
    
    blacklist_count = 0
    for kw in BLACKLIST_KEYWORDS:
        if kw in title_lower:
            blacklist_count += 3  # Başlıkta varsa ağır ceza
        elif kw in text_lower:
            blacklist_count += 1
    
    return blacklist_count >= 5  # 5 veya daha fazla kara liste eşleşmesi

def calculate_cultural_score(text, analysis, title=""):
    """Kültürel skor hesapla - NER ve keyword bazlı"""
    text_lower = text.lower()
    title_lower = title.lower()
    
    score = 0
    strong_matches = []
    medium_matches = []
    
    # 1. Güçlü kültürel anahtar kelimeler (yüksek puan)
    for kw in STRONG_CULTURAL_KEYWORDS:
        if kw in title_lower:
            score += 10  # Başlıkta varsa çok yüksek puan
            strong_matches.append(kw)
        elif kw in text_lower:
            count = min(text_lower.count(kw), 3)
            score += count * 3
            if kw not in strong_matches:
                strong_matches.append(kw)
    
    # 2. Orta seviye kültürel kelimeler
    for kw in MEDIUM_CULTURAL_KEYWORDS:
        if kw in title_lower:
            score += 5
            medium_matches.append(kw)
        elif kw in text_lower:
            score += 1
            if kw not in medium_matches:
                medium_matches.append(kw)
    
    # 3. NER varlık skoru
    person_count = 0
    gpe_count = 0
    org_count = 0
    
    for entity, label in analysis["entities"]:
        if label == "PERSON":
            person_count += 1
            score += 2
        elif label == "GPE":
            gpe_count += 1
            score += 2
        elif label == "ORGANIZATION":
            org_count += 1
            score += 1
        elif label == "LOCATION":
            score += 2
    
    # 4. Token > 2 NER varlıkları için bonus
    score += analysis["multi_token_count"] * 4
    
    # 5. Uzun belge bonusu
    if analysis["token_count"] > 500:
        score += 3
    if analysis["token_count"] > 1000:
        score += 2
    
    # 6. Kara liste cezası
    blacklist_penalty = 0
    for kw in BLACKLIST_KEYWORDS:
        if kw in title_lower:
            blacklist_penalty += 15
        elif kw in text_lower:
            blacklist_penalty += 2
    
    score = max(0, score - blacklist_penalty)
    
    return {
        "score": score,
        "strong_matches": strong_matches,
        "medium_matches": medium_matches,
        "person_count": person_count,
        "gpe_count": gpe_count,
        "org_count": org_count,
        "entity_count": analysis["entity_count"],
        "multi_token_entities": analysis["multi_token_entities"],
        "token_count": analysis["token_count"],
        "blacklist_penalty": blacklist_penalty
    }

def main():
    print("=" * 60)
    print("RAGTurk Kültürel Belge Filtreleme (Akıllı NER)")
    print("=" * 60)
    
    scored_documents = []
    total_files = 0
    skipped_files = 0
    blacklisted = 0
    ner_failed = 0
    
    files = [f for f in os.listdir(DATASET_PATH) if f.endswith(".json")]
    total_files_count = len(files)
    
    print(f"\nToplam {total_files_count} JSON dosyası taranıyor...")
    print("NLTK NER analizi yapılıyor...\n")
    
    start_time = time.time()
    
    for idx, file in enumerate(files):
        current_idx = idx + 1
        
        if current_idx % 50 == 0 or current_idx == 10:
            elapsed = time.time() - start_time
            avg_per_file = elapsed / current_idx
            remaining = total_files_count - current_idx
            eta_min = (remaining * avg_per_file) / 60
            
            print(f"İşlenen: {current_idx}/{total_files_count} | "
                  f"Seçilen: {len(scored_documents)} | "
                  f"Elenen: {blacklisted} | "
                  f"Kalan: ~{eta_min:.1f} dk")
        
        file_path = os.path.join(DATASET_PATH, file)
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            if "article" in data and "content" in data["article"]:
                text = data["article"]["content"]
                title = data["article"].get("title", "")
            elif "content" in data:
                text = data["content"]
                title = data.get("title", "")
            else:
                skipped_files += 1
                continue
            
            total_files += 1
            
            # Kara liste kontrolü
            if is_blacklisted(text, title):
                blacklisted += 1
                continue
            
            # NER analizi
            analysis = analyze_text_with_ner(text)
            if not analysis:
                ner_failed += 1
                continue
            
            # Kültürel skor hesapla
            result = calculate_cultural_score(text, analysis, title)
            
            # Seçim kriterleri:
            # 1. En az 1 güçlü kültürel keyword VE skor >= 15
            # 2. VEYA en az 3 orta seviye keyword VE NER varlığı VE skor >= 10
            # 3. VEYA token > 2 NER varlığı VE skor >= 12
            
            has_strong = len(result["strong_matches"]) >= 1
            has_medium = len(result["medium_matches"]) >= 3
            has_multi_ner = analysis["multi_token_count"] > 0
            has_ner = analysis["entity_count"] > 0
            score = result["score"]
            
            selected = False
            reason = ""
            
            if has_strong and score >= 15:
                selected = True
                reason = "strong_cultural_keyword"
            elif has_medium and has_ner and score >= 10:
                selected = True
                reason = "medium_keywords_with_ner"
            elif has_multi_ner and score >= 12:
                selected = True
                reason = "multi_token_ner"
            
            if selected:
                scored_documents.append({
                    "file": file,
                    "file_path": file_path,
                    "title": title,
                    "score": score,
                    "strong_matches": result["strong_matches"],
                    "medium_matches": result["medium_matches"],
                    "multi_token_entities": analysis["multi_token_entities"],
                    "token_count": analysis["token_count"],
                    "entity_count": analysis["entity_count"],
                    "multi_token_count": analysis["multi_token_count"],
                    "selection_reason": reason
                })
                
        except Exception as e:
            skipped_files += 1
            continue
    
    total_time = time.time() - start_time
    
    print(f"\n{'=' * 60}")
    print("Tarama tamamlandı!")
    print(f"Toplam süre: {total_time/60:.1f} dakika")
    print(f"Toplam dosya: {total_files_count}")
    print(f"İşlenen: {total_files}")
    print(f"Kara listeye takılan: {blacklisted}")
    print(f"NER başarısız: {ner_failed}")
    print(f"Atlanan: {skipped_files}")
    print(f"Kriterleri karşılayan: {len(scored_documents)}")
    print(f"{'=' * 60}\n")
    
    # Skora göre sırala
    scored_documents.sort(key=lambda x: x["score"], reverse=True)
    
    # 300-500 arası belge seç
    MIN_DOCS = 300
    MAX_DOCS = 500
    
    selected_count = min(len(scored_documents), MAX_DOCS)
    if selected_count < MIN_DOCS:
        print(f"UYARI: Sadece {selected_count} belge kriterleri karşılıyor")
    
    selected_documents = scored_documents[:selected_count]
    
    # Mevcut dosyaları temizle
    print(f"{selected_count} kültürel belge kopyalanıyor...\n")
    
    for existing in os.listdir(OUTPUT_PATH):
        existing_path = os.path.join(OUTPUT_PATH, existing)
        if os.path.isfile(existing_path):
            os.remove(existing_path)
    
    # İstatistikler
    reason_counts = defaultdict(int)
    
    for idx, doc in enumerate(selected_documents):
        shutil.copy(doc["file_path"], OUTPUT_PATH)
        reason_counts[doc["selection_reason"]] += 1
        
        if idx < 20:
            print(f"{idx + 1}. {doc['title'][:55]}...")
            print(f"   Skor: {doc['score']:.0f} | NER: {doc['entity_count']} | "
                  f"Multi-NER: {doc['multi_token_count']}")
            if doc["strong_matches"][:3]:
                print(f"   Güçlü: {', '.join(doc['strong_matches'][:3])}")
            print()
    
    avg_score = sum(d["score"] for d in selected_documents) / selected_count if selected_count > 0 else 0
    
    print("=" * 60)
    print("SONUÇ RAPORU")
    print("=" * 60)
    print(f"Seçilen toplam belge: {selected_count}")
    print(f"Güçlü kültürel keyword: {reason_counts['strong_cultural_keyword']}")
    print(f"Orta keyword + NER: {reason_counts['medium_keywords_with_ner']}")
    print(f"Multi-token NER: {reason_counts['multi_token_ner']}")
    print(f"Ortalama skor: {avg_score:.2f}")
    print(f"Çıktı: {OUTPUT_PATH}")
    print("=" * 60)
    
    # Rapor dosyası
    report_path = os.path.join(OUTPUT_PATH, "_rapor.json")
    report = {
        "toplam_secilen": selected_count,
        "guclu_kulturel": reason_counts["strong_cultural_keyword"],
        "orta_keyword_ner": reason_counts["medium_keywords_with_ner"],
        "multi_token_ner": reason_counts["multi_token_ner"],
        "ortalama_skor": avg_score,
        "belgeler": [
            {
                "dosya": doc["file"],
                "baslik": doc["title"],
                "skor": doc["score"],
                "token_sayisi": doc["token_count"],
                "ner_sayisi": doc["entity_count"],
                "multi_token_ner": doc["multi_token_count"],
                "guclu_kelimeler": doc["strong_matches"][:5],
                "secim_nedeni": doc["selection_reason"]
            }
            for doc in selected_documents
        ]
    }
    
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\nDetaylı rapor: {report_path}")
    return selected_count

if __name__ == "__main__":
    main()
