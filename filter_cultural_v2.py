"""
RAGTurk Kültürel Belge Filtreleme v2
=====================================
NLTK NER + keyword analizi ile kültürel belgeleri seçer.
Tek thread, sade ve hızlı.

Çıktılar:
  cultural_selected/            -> Seçilen JSON belgeler
  cultural_selected/_rapor.json -> Detaylı rapor
  logs/morphological_queries.json -> Uzun morfolojik sorgu logu
"""

import os, json, shutil, time, re
import nltk
from nltk import word_tokenize, pos_tag, ne_chunk
from collections import defaultdict

# ── NLTK Setup ──────────────────────────────────────────
for pkg in ["punkt", "punkt_tab", "averaged_perceptron_tagger",
            "averaged_perceptron_tagger_eng", "maxent_ne_chunker",
            "maxent_ne_chunker_tab", "words"]:
    try:
        nltk.download(pkg, quiet=True)
    except:
        pass

# ── Paths ───────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATASET   = os.path.join(BASE_DIR, "ragturk", "formal_5k", "dataset", "json")
OUTPUT    = os.path.join(BASE_DIR, "cultural_selected")
LOG_DIR   = os.path.join(BASE_DIR, "logs")
os.makedirs(OUTPUT, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# NER için max karakter (hız için)
MAX_NER_CHARS = 5000

# ═════════════════════════════════════════════════════════
#  BAŞLIK KARA LİSTESİ  – bunlar varsa doğrudan ele
# ═════════════════════════════════════════════════════════
TITLE_BLACKLIST = [
    # Spor
    "futbol", "basketbol", "voleybol", "tenis", "rugby", "formula",
    "olimpiyat", "fifa", "uefa", "nba", "lig", "maç",
    "turnuva", "playoff", "transfer", "kadrosu", "sezonu",
    "stadyum", "derbi", "atletizm", "yüzme", "boks", "güreş", "halter",
    "gol", "penaltı", "hakem", "forvet", "defans", "kaleci",
    "kupası", "kupa", "yarışması", "şampiyonası", "şampiyonluk",
    "takımı", "takımının", "yarış", "spor kulübü",
    "galatasaray", "fenerbahçe", "beşiktaş", "trabzonspor",
    "futbol takımı", "futbol kulübü", "futbol ligi", "milli futbol",
    "basketbol takımı", "voleybol takımı", "milli takım",
    "süper lig", "premier league", "la liga", "bundesliga", "serie a",
    "türkiye kupası", "dünya kupası", "avrupa şampiyonası",
    "hat trick", "hat-trick", "gol krallığı", "grand prix",
    "puan durumu", "fikstür", "deplasman",

    # Bilim / Matematik / Fizik
    "problem", "problemi", "denklem", "formül", "teorem", "algoritma",
    "matris", "vektör", "polinom", "integral", "türev", "fonksiyon",
    "kuantum", "termodinamik", "elektromanyetik", "atom", "molekül",
    "izotop", "reaksiyon", "cisim",
    "cisim problemi", "diferansiyel denklem", "lineer cebir",
    "kuantum mekaniği", "görelilik teorisi", "karanlık madde",
    "parçacık fiziği",

    # Teknoloji / Yazılım
    "yazılım", "programlama", "framework", "api", "veritabanı",
    "android", "ios", "windows", "linux", "python", "java", "c++",
    "chrome", "firefox", "browser",
    "yazılım geliştirme", "programlama dili", "işletim sistemi",
    "yapay zeka", "makine öğrenmesi", "derin öğrenme",

    # Video Oyunları
    "playstation", "xbox", "nintendo", "steam",
    "rpg", "fps", "mmorpg", "esport", "chapter",
    "video oyun", "bilgisayar oyunu",

    # Tıp
    "sendrom", "hastalık", "enfeksiyon", "virüs", "bakteri",
    "tedavi", "ameliyat", "tanı", "semptom",
    "hastalığı", "sendromu", "tedavi yöntemi", "cerrahi müdahale",
    "bulaşıcı hastalık", "genetik hastalık",

    # Ekonomi / Finans
    "borsa", "hisse", "döviz", "enflasyon", "kripto", "bitcoin",
    "ekonomik mucizesi", "ekonomik krizi", "borsa endeksi",
    "kripto para", "merkez bankası",
    "ekonomik", "mucizesi", "ekonomisi",

    # Askeri / Savaş
    "muharebesi", "savaşı", "cephesi", "operasyonu",
    "muharebeleri", "çarpışması", "kuşatması", "baskını",
    "savaş", "harekatı", "antlaşması",
    "askeri operasyon", "silahlı kuvvetleri",
    "barış antlaşması", "ateşkes anlaşması",

    # Politik
    "hükümeti", "kabinesi", "paktı",
    "genel seçimleri", "yerel seçimleri", "referandumu",

    # Liste / İstatistik
    "listesi", "listeleri", "kronoloji", "istatistik",

    # Biyoloji
    "flora", "fauna", "kuşları", "balıkları",
    "böcekleri", "sürüngenleri",

    # Coğrafi / İdari
    "nüfusu", "seçimi", "seçimleri", "referandum",
    "ilçesi", "mahallesi", "köyü", "bucağı",
    "ip adresi", "protokol",

    # Film/Dizi bölümleri
    "bölümü", "sezon", "episode",

    # Kaza / Felaket
    "kazası", "faciası", "uçak kazası", "tren kazası",
]

# ═════════════════════════════════════════════════════════
#  BAŞLIK BEYAZ LİSTESİ  – bunlar varsa kültürel say
# ═════════════════════════════════════════════════════════
TITLE_WHITELIST = [
    "müze", "müzesi", "anıt", "anıtkabir", "camii", "cami",
    "türbe", "külliye", "medrese", "kervansaray", "hamam", "höyük",
    "unesco", "arkeoloji", "opera", "bale", "folklor",
    "osmanlı", "selçuklu", "bizans", "hitit",
    "minyatür", "tezhip", "ebru", "çini",
    "mimari", "mevlana", "göbeklitepe", "kapadokya", "efes",
    "sanat müzesi", "arkeoloji müzesi", "etnografya müzesi",
    "anıt mezar", "ulu cami", "selimiye camii", "sultanahmet",
    "ören yeri", "dünya mirası", "kültür mirası",
    "divan edebiyatı", "halk edebiyatı", "türk edebiyatı",
    "devlet operası", "devlet tiyatrosu",
    "halk oyunları", "türk mutfağı", "türk hamamı",
    "karagöz", "sema töreni", "mevlevi",
    "osmanlı devleti", "osmanlı imparatorluğu", "osmanlı mimarisi",
    "selçuklu devleti", "anadolu selçuklu",
    "bizans imparatorluğu",
    "hat sanatı", "tezhip sanatı", "ebru sanatı", "çini sanatı",
    "türk müziği", "halk müziği", "sanat müziği",
    "mimar sinan", "antik kent", "antik şehir",
]

# ═════════════════════════════════════════════════════════
#  İÇERİK – GÜÇLÜ KÜLTÜREL KELİMELER
# ═════════════════════════════════════════════════════════
STRONG_CULTURAL = [
    # Tarih & Medeniyet
    "osmanlı", "selçuklu", "bizans", "hitit", "frigya", "lidya", "urartu",
    "imparatorluk", "sultanat", "hanedan", "padişah",
    "sadrazam", "beylik", "hilafet", "saltanat",
    # Müze & Miras
    "müze", "müzesi", "anıt", "anıtkabir", "miras", "unesco",
    "restorasyon", "arkeoloji", "arkeolojik", "höyük",
    # Mimari
    "camii", "mescit", "medrese", "kervansaray", "hamam",
    "kalesi", "köşk", "türbe", "külliye", "minare", "kubbe",
    # Sanat
    "minyatür", "hattat", "tezhip", "ebru", "çini", "seramik",
    "heykel", "ressam", "tablo", "fresk", "mozaik",
    # Edebiyat
    "şair", "şiir", "divan", "kaside", "gazel", "masal", "destan",
    "efsane", "roman", "hikâye",
    # Müzik
    "türkü", "beste", "besteci", "makam", "saz",
    "bağlama", "ney", "kanun", "kemençe",
    # Din & Tasavvuf
    "tasavvuf", "sufi", "mevlevi", "mevlana", "bektaşi", "tekke",
    "dergah", "tarikat", "derviş", "sema",
    # Gelenek & Folklor
    "folklor", "gelenek", "görenek",
    "düğün", "bayram", "festival", "şenlik",
    "kilim", "halı", "nakış", "oya",
    # Antik yerler
    "kapadokya", "efes", "truva", "bergama", "afrodisias",
    "göbeklitepe", "çatalhöyük", "hattuşa", "nemrut", "zeugma",
    # Çoklu kalıplar
    "osmanlı devleti", "osmanlı imparatorluğu",
    "selçuklu devleti", "anadolu selçuklu",
    "bizans imparatorluğu", "hitit uygarlığı",
    "hat sanatı", "tezhip sanatı", "ebru sanatı", "çini sanatı",
    "divan edebiyatı", "halk edebiyatı",
    "türk müziği", "halk müziği", "sanat müziği",
    "halk oyunları", "türk folkloru", "halk kültürü",
    "antik kent", "ören yeri", "kültürel miras",
    "topkapı sarayı", "mimar sinan",
    "arkeolojik kazı", "dünya mirası",
]

MEDIUM_CULTURAL = [
    "tarih", "tarihi", "kültür", "kültürel", "sanat", "edebiyat",
    "mimari", "mimar", "antik", "medeniyet", "tören", "ritüel",
    "opera", "tiyatro", "bale", "sinema",
    "tarihi eser", "tarihi yapı", "sanat eseri",
    "antik şehir", "dini tören", "tiyatro oyunu",
]

# İçerik kara listesi (skor düşürücü)
CONTENT_BLACKLIST = [
    "futbol", "basketbol", "voleybol", "tenis", "maç", "lig",
    "gol", "penaltı", "hakem", "stadyum", "playoff", "sezon",
    "transfer", "şampiyonluk", "turnuva",
    "problem", "denklem", "algoritma", "teorem", "formül",
    "matris", "vektör", "integral", "türev", "fonksiyon",
    "kuantum", "elektron", "proton",
    "yazılım", "programlama", "veritabanı",
    "hastalık", "sendrom", "tedavi", "enfeksiyon", "virüs",
    "borsa", "hisse", "enflasyon", "faiz",
    "ekonomi", "ekonomik", "nüfus",
    "savaş", "muharebe", "cephe", "operasyon", "kuşatma",
    "darbe", "isyan", "deprem", "sel", "yangın",
    "şirket", "firma", "holding",
    "milletvekili", "başbakan", "cumhurbaşkanı",
    "mahkeme", "dava", "yargıtay",
]


# ═════════════════════════════════════════════════════════
#  NER ANALİZİ
# ═════════════════════════════════════════════════════════
def run_ner(text):
    """NLTK NER analizi - entity'leri döndür."""
    try:
        words = text.split()
        if len(words) < 3:
            return None

        snippet = text[:MAX_NER_CHARS]
        tokens  = word_tokenize(snippet)
        tagged  = pos_tag(tokens)
        tree    = ne_chunk(tagged)

        entities    = []
        multi_token = []

        for subtree in tree:
            if hasattr(subtree, 'label'):
                ent_words = [w for w, _ in subtree]
                name  = " ".join(ent_words)
                label = subtree.label()
                entities.append((name, label))
                if len(ent_words) > 2:
                    multi_token.append((name, label))

        return {
            "entities":    entities,
            "multi_token": multi_token,
            "count":       len(entities),
            "multi_count": len(multi_token),
        }
    except Exception:
        return None


# ═════════════════════════════════════════════════════════
#  BAŞLIK KONTROLÜ
# ═════════════════════════════════════════════════════════
def check_title(title):
    """'reject' / 'accept' / 'neutral'"""
    t = title.lower().strip()

    # Yıl / yıl aralığı
    if re.match(r'^\d{4}$', t) or re.match(r'^\d{4}[-–]\d{2,4}', t):
        return "reject"

    # Whitelist önce
    for kw in TITLE_WHITELIST:
        if kw in t:
            return "accept"

    # Blacklist
    for kw in TITLE_BLACKLIST:
        if kw in t:
            return "reject"

    return "neutral"


# ═════════════════════════════════════════════════════════
#  KÜLTÜREL SKOR
# ═════════════════════════════════════════════════════════
def cultural_score(text, title, ner):
    t_lower  = text.lower()
    tl_lower = title.lower()

    score       = 0
    strong_hits = []
    medium_hits = []

    # Güçlü kelimeler
    for kw in STRONG_CULTURAL:
        if kw in tl_lower:
            score += 12
            strong_hits.append(kw)
        elif kw in t_lower:
            cnt = min(t_lower.count(kw), 3)
            score += cnt * 3
            if kw not in strong_hits:
                strong_hits.append(kw)

    # Orta kelimeler
    for kw in MEDIUM_CULTURAL:
        if kw in tl_lower:
            score += 5
            medium_hits.append(kw)
        elif kw in t_lower:
            score += 1
            if kw not in medium_hits:
                medium_hits.append(kw)

    # NER skoru
    person = gpe = org = 0
    for _, label in ner["entities"]:
        if   label == "PERSON":       person += 1; score += 2
        elif label == "GPE":          gpe    += 1; score += 2
        elif label == "ORGANIZATION": org    += 1; score += 1
        elif label == "LOCATION":     score  += 2

    # Multi-token NER bonus
    score += ner["multi_count"] * 4

    # Blacklist cezası
    penalty = 0
    for kw in CONTENT_BLACKLIST:
        if kw in tl_lower:
            penalty += 20
        elif kw in t_lower:
            penalty += 3

    score = max(0, score - penalty)

    return {
        "score":   score,
        "strong":  strong_hits,
        "medium":  medium_hits,
        "person":  person,
        "gpe":     gpe,
        "org":     org,
        "penalty": penalty,
    }


# ═════════════════════════════════════════════════════════
#  MORFOLOJİK SORGU ANALİZİ
# ═════════════════════════════════════════════════════════
def analyze_queries(data, file_name):
    logs = []
    questions = data.get("questions", {})
    items = questions.get("items", []) if isinstance(questions, dict) else (questions if isinstance(questions, list) else [])

    for idx, item in enumerate(items):
        q = item.get("question", "")
        wc = len(q.split())
        if wc < 15:
            continue

        key_terms = []
        try:
            tokens = word_tokenize(q)
            tagged = pos_tag(tokens)
            tree   = ne_chunk(tagged)
            for sub in tree:
                if hasattr(sub, 'label'):
                    key_terms.append(" ".join(w for w, _ in sub))
            for tok, p in tagged:
                if p in ("NNP", "NNPS") and tok not in " ".join(key_terms):
                    key_terms.append(tok)
        except Exception:
            pass

        logs.append({
            "file": file_name, "query_id": f"q{idx}",
            "query": q, "word_count": wc,
            "key_terms": key_terms[:10],
            "category": item.get("category", ""),
            "is_long_morphological": wc > 25,
        })
    return logs


# ═════════════════════════════════════════════════════════
#  TEK DOSYA İŞLE
# ═════════════════════════════════════════════════════════
def process_file(file_name, file_path):
    """Dosyayı oku, analiz et, karar ver. dict döndür."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {"status": "skip"}

    # İçerik ve başlık
    if "article" in data and "content" in data["article"]:
        text  = data["article"]["content"]
        title = data["article"].get("title", "")
    elif "content" in data:
        text  = data["content"]
        title = data.get("title", "")
    else:
        return {"status": "skip"}

    qlogs = analyze_queries(data, file_name)

    # 1) Başlık kontrolü
    verdict = check_title(title)
    if verdict == "reject":
        return {"status": "title_reject", "qlogs": qlogs}

    # 2) NER
    ner = run_ner(text)
    if ner is None:
        return {"status": "ner_fail", "qlogs": qlogs}

    # 3) Skor
    res = cultural_score(text, title, ner)
    score = res["score"]

    # 4) Seçim kararı
    selected = False
    reason   = ""

    if verdict == "accept" and score >= 15:
        selected = True
        reason   = "title_whitelist"

    if not selected:
        has_strong      = len(res["strong"]) >= 1
        has_multi_strong= len(res["strong"]) >= 2
        has_medium      = len(res["medium"]) >= 2
        has_ner         = ner["count"] > 0
        has_multi_ner   = ner["multi_count"] > 0
        has_person      = res["person"] > 0
        has_gpe         = res["gpe"] > 0

        # NER ZORUNLU
        if not has_ner:
            pass
        elif has_multi_strong and score >= 20:
            selected = True; reason = "multi_strong_ner"
        elif has_strong and has_multi_ner and score >= 15:
            selected = True; reason = "strong_multi_ner"
        elif has_strong and (has_person or has_gpe) and score >= 15:
            selected = True; reason = "strong_person_gpe"
        elif has_medium and has_ner and score >= 15:
            selected = True; reason = "medium_keyword_ner"

    if not selected:
        return {"status": "content_reject", "qlogs": qlogs}

    return {
        "status": "selected",
        "file":   file_name,
        "path":   file_path,
        "title":  title,
        "score":  score,
        "reason": reason,
        "strong": res["strong"],
        "medium": res["medium"],
        "person": res["person"],
        "gpe":    res["gpe"],
        "org":    res["org"],
        "penalty":res["penalty"],
        "ner_count":    ner["count"],
        "multi_ner":    ner["multi_count"],
        "qlogs": qlogs,
    }


# ═════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("RAGTurk Kültürel Belge Filtreleme v2")
    print("=" * 60, flush=True)

    files = sorted(f for f in os.listdir(DATASET) if f.endswith(".json"))
    total = len(files)
    print(f"Toplam {total} JSON dosyası taranacak.\n", flush=True)

    selected_docs  = []
    all_qlogs      = []
    stats = defaultdict(int)

    t0 = time.time()

    for i, fname in enumerate(files, 1):
        fpath  = os.path.join(DATASET, fname)
        result = process_file(fname, fpath)

        status = result["status"]
        stats[status] += 1

        if "qlogs" in result:
            all_qlogs.extend(result["qlogs"])

        if status == "selected":
            selected_docs.append(result)

        # İlerleme
        if i % 100 == 0 or i == total:
            elapsed = time.time() - t0
            eta     = (total - i) * elapsed / i / 60
            print(f"[{i}/{total}]  seçilen={len(selected_docs)}  "
                  f"elenen={stats['title_reject']+stats['content_reject']}  "
                  f"~{eta:.1f} dk kaldı", flush=True)

    elapsed_total = time.time() - t0

    # Skora göre sırala
    selected_docs.sort(key=lambda d: d["score"], reverse=True)
    sel_count = len(selected_docs)

    # ── Eski dosyaları temizle, yenileri kopyala ────────
    for f in os.listdir(OUTPUT):
        fp = os.path.join(OUTPUT, f)
        if os.path.isfile(fp):
            os.remove(fp)

    reason_counts = defaultdict(int)
    for doc in selected_docs:
        shutil.copy(doc["path"], OUTPUT)
        reason_counts[doc["reason"]] += 1

    # ── Rapor yazdır ────────────────────────────────────
    print(f"\n{'='*60}")
    print("SONUÇ RAPORU")
    print(f"{'='*60}")
    print(f"Süre           : {elapsed_total/60:.1f} dakika")
    print(f"Toplam dosya   : {total}")
    print(f"Başlıkla elenen: {stats['title_reject']}")
    print(f"İçerikle elenen: {stats['content_reject']}")
    print(f"NER başarısız  : {stats['ner_fail']}")
    print(f"Atlanan        : {stats['skip']}")
    print(f"SEÇİLEN        : {sel_count}")
    print()

    print("Seçim nedeni dağılımı:")
    for r, c in sorted(reason_counts.items(), key=lambda x: -x[1]):
        print(f"  {r}: {c}")

    print(f"\nEn yüksek skorlu 15 belge:")
    for i, d in enumerate(selected_docs[:15], 1):
        print(f"  {i}. [{d['score']}] {d['title'][:60]}")
        print(f"     {d['reason']} | NER:{d['ner_count']} | {', '.join(d['strong'][:3])}")

    # ── Rapor JSON ──────────────────────────────────────
    report = {
        "secilen": sel_count,
        "taranan": total,
        "baslik_elenen":  stats["title_reject"],
        "icerik_elenen":  stats["content_reject"],
        "sure_dk": round(elapsed_total / 60, 1),
        "neden_dagilimi": dict(reason_counts),
        "ort_skor": round(sum(d["score"] for d in selected_docs) / sel_count, 1) if sel_count else 0,
        "belgeler": [
            {"dosya": d["file"], "baslik": d["title"], "skor": d["score"],
             "neden": d["reason"], "ner": d["ner_count"], "guclu": d["strong"][:5],
             "ceza": d["penalty"]}
            for d in selected_docs
        ],
    }
    rpath = os.path.join(OUTPUT, "_rapor.json")
    with open(rpath, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nRapor: {rpath}")

    # ── Morfolojik sorgu logu ───────────────────────────
    long_qs = [q for q in all_qlogs if q["is_long_morphological"]]
    morph = {
        "toplam_analiz": len(all_qlogs),
        "uzun_morfolojik": len(long_qs),
        "ort_kelime": round(sum(q["word_count"] for q in long_qs) / len(long_qs), 1) if long_qs else 0,
        "sorgular": sorted(long_qs, key=lambda x: -x["word_count"]),
    }
    mpath = os.path.join(LOG_DIR, "morphological_queries.json")
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(morph, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print("MORFOLOJİK SORGU LOGU")
    print(f"{'='*60}")
    print(f"Analiz edilen : {len(all_qlogs)}")
    print(f"Uzun (>25 kel): {len(long_qs)}")
    if long_qs:
        print(f"\nEn uzun 5 sorgu:")
        for q in long_qs[:5]:
            print(f"  [{q['word_count']} kel] {q['file']}")
            print(f"    {q['query'][:100]}...")
    print(f"Log: {mpath}")

    print(f"\n{'='*60}")
    print(f"TAMAMLANDI  –  {sel_count} kültürel belge seçildi")
    print(f"{'='*60}")
    return sel_count


if __name__ == "__main__":
    main()
