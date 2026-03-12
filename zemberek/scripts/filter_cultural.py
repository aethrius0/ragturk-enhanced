"""
RAGTurk – Kültürel Belge Filtreleme (Zemberek + NLTK NER)
==========================================================
2790 JSON belgesinden kültürel olanları seçip cultural_selected/ klasörüne koyar.

Mantık:
  1. Başlık blacklist/whitelist kontrolü (erken eleme)
  2. Keyword ön-elek (hiç keyword bulamazsa Zemberek/NER'i çağırmaz → hız)
  3. Zemberek morfolojik analiz (Prop tespiti + lemma)
  4. NLTK NER (PERSON / GPE / ORGANIZATION)
  5. Skor hesaplama + seçim kararı
  6. Sorgu analizi (kısa: token>2, uzun morfolojik: token>25)

Çıktılar:
  cultural_selected/              → Seçilen JSON belgeler
  cultural_selected/_rapor.json   → Detaylı rapor
  logs/morphological_queries.json → Uzun morfolojik sorgu logu
"""

import os, sys, json, shutil, time, re
from collections import defaultdict

# ── NLTK ────────────────────────────────────────────────
import nltk
for _r in ["punkt", "punkt_tab", "averaged_perceptron_tagger",
           "averaged_perceptron_tagger_eng", "maxent_ne_chunker",
           "maxent_ne_chunker_tab", "words"]:
    try:
        nltk.data.find(
            f"tokenizers/{_r}" if "punkt" in _r else
            f"taggers/{_r}"    if "tagger" in _r else
            f"chunkers/{_r}"   if "chunker" in _r else
            f"corpora/{_r}")
    except LookupError:
        nltk.download(_r, quiet=True)
print("NLTK hazır.", flush=True)

# ── JPype + Zemberek ────────────────────────────────────
import jpype, jpype.imports

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JAR      = os.path.join(ROOT, "lib", "zemberek-full.jar")
JVM_DLL  = r"C:\Program Files\Java\jdk-17\bin\server\jvm.dll"

if not jpype.isJVMStarted():
    jpype.startJVM(JVM_DLL, classpath=[JAR])

from zemberek.morphology import TurkishMorphology
morphology = TurkishMorphology.createWithDefaults()
print("Zemberek hazır.", flush=True)

# ── Paths ───────────────────────────────────────────────
DATASET = os.path.join(ROOT, "ragturk", "formal_5k", "dataset", "json")
OUTPUT  = os.path.join(ROOT, "cultural_selected")
LOG_DIR = os.path.join(ROOT, "logs")
os.makedirs(OUTPUT, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

SNIPPET = 800
_WORD   = re.compile(r'[\wçğıöşüÇĞİÖŞÜ]+')

# ═════════════════════════════════════════════════════════
#  KEYWORD LİSTELERİ
# ═════════════════════════════════════════════════════════
TITLE_BLACKLIST = {
    # Spor
    "futbol","basketbol","voleybol","tenis","rugby","formula",
    "olimpiyat","fifa","uefa","nba","lig","maç",
    "turnuva","playoff","transfer","kadrosu","sezonu",
    "stadyum","derbi","atletizm","yüzme","boks","halter",
    "gol","penaltı","hakem","forvet","defans","kaleci",
    "kupası","kupa","yarışması","şampiyonası","şampiyonluk",
    "takımı","takımının","yarış","spor kulübü","spor",
    "galatasaray","fenerbahçe","beşiktaş","trabzonspor",
    "süper lig","premier league","la liga","bundesliga","serie a",
    "grand prix","puan durumu","fikstür","deplasman",
    # Matematik / Fen
    "problem","problemi","denklem","formül","teorem","algoritma",
    "matris","vektör","polinom","integral","türev","fonksiyon",
    "kuantum","termodinamik","elektromanyetik","molekül","atom",
    "izotop","reaksiyon","cisim","fizik","kimya",
    # Teknoloji / Bilişim
    "yazılım","programlama","framework","api","veritabanı",
    "android","ios","windows","linux","python","java","c++",
    "chrome","firefox","browser","sunucu","ağ protokolü",
    # Oyun / Medya
    "playstation","xbox","nintendo","steam","rpg","fps","mmorpg","esport","chapter",
    # Tıp
    "sendrom","hastalık","enfeksiyon","virüs","bakteri",
    "tedavi","ameliyat","tanı","semptom","hastalığı","sendromu",
    # Ekonomi / Finans
    "borsa","hisse","döviz","enflasyon","kripto","bitcoin",
    "ekonomisi","ekonomik","gsyih","bütçe",
    # Doğa / Biyoloji
    "flora","fauna","kuşları","balıkları","böcekleri","sürüngenleri",
    "memelileri","bitkileri","hayvanları","türleri",
    # Jenerik listeler / istatistik
    "listesi","listeleri","kronoloji","kronolojisi","istatistik",
    "sıralaması","tablosu","dizini","indeksi","kadrosu",
    # Coğrafi / İdari (jenerik)
    "nüfusu","ilçesi","mahallesi","köyü","bucağı","mezrası",
    "havaalanı","havalimanı","istasyonu","garı",
    # Seçim / Politik (jenerik)
    "seçimi","seçimleri","referandum","kabinesi","hükümeti",
    # Diğer
    "bölümü","sezon","episode",
    "ip adresi","protokol",
    # Savaş / Askeri (jenerik)
    "muharebesi","savaşı","cephesi","operasyonu","harekatı",
    "muharebeleri","çarpışması","kuşatması","baskını",
    "kazası","faciası",
}

TITLE_WHITELIST = [
    # Yapılar / Mimari
    "müze","müzesi","anıt","anıtkabir","camii","cami",
    "türbe","külliye","medrese","kervansaray","hamam","höyük",
    "sarayı","köprüsü","kalesi","antik kent","ören yeri",
    # Tarihsel kavramlar
    "osmanlı","selçuklu","bizans","hitit",
    "unesco","dünya mirası","kültür mirası",
    # Sanat / Edebiyat
    "minyatür","tezhip","ebru","çini",
    "divan edebiyatı","halk edebiyatı","türk edebiyatı",
    "hat sanatı","mimar sinan","karagöz",
    # Kültür / Gelenek
    "halk oyunları","türk mutfağı","türk hamamı",
    "sema töreni","mevlevi","mevlana",
    "folklor","opera","bale",
    # Coğrafi / Tarihi yerler
    "göbeklitepe","kapadokya","efes","topkapı",
    "ayasofya","sultanahmet","dolmabahçe",
    # Kişiler (kültürel bağlam)
    "atatürk",
]

STRONG_KEYWORDS = [
    # Devletler / İmparatorluklar
    "osmanlı","selçuklu","bizans","hitit","frigya","lidya","urartu",
    "sasani","emevi","abbasi","moğol",
    "karahanlı","gazneli","memlük","akkoyunlu","karakoyunlu",
    "imparatorluk","sultanat","hanedan","padişah",
    "sadrazam","beylik","hilafet","saltanat","vezir",
    "şehzade","valide sultan","harem","ferman","berat",
    # Mimari / Yapılar
    "müze","müzesi","anıt","anıtkabir","unesco",
    "restorasyon","arkeoloji","arkeolojik","höyük",
    "camii","mescit","medrese","kervansaray","hamam",
    "kalesi","köşk","türbe","külliye","minare","kubbe",
    "sarayı","köprüsü","çeşme","şadırvan",
    "bedesten","kümbet","ribat","zaviye",
    "antik kent","ören yeri",
    # Görsel Sanatlar (spesifik)
    "minyatür","hattat","tezhip","ebru","çini",
    "fresk","mozaik","kaligrafi","hat sanatı",
    # Edebiyat (spesifik)
    "kaside","gazel","destan",
    "mesnevi","rubai","koşma",
    "halk edebiyatı","divan edebiyatı","tanzimat",
    # Müzik (spesifik)
    "türkü","besteci","makam",
    "bağlama","ney","kemençe","tanbur","zurna",
    "mehter","fasıl",
    # Din / Tasavvuf
    "tasavvuf","sufi","mevlevi","mevlana","bektaşi","tekke",
    "dergah","tarikat","derviş","sema",
    "kuran","hadis","tefsir",
    "manastır","patrik",
    # Gelenek / Halk Kültürü (spesifik)
    "folklor","görenek",
    "kına gecesi","mevlit","aşık","ozan","halk oyunları",
    "zeybek","horon","halay","semah","karagöz","hacivat",
    "ortaoyunu","meddah",
    # El Sanatları (spesifik)
    "kilim","nakış","oya","dokuma",
    "bakırcılık","kuyumculuk","oymacılık","telkari",
    # Mutfak (spesifik)
    "kebap","baklava","lokum","şerbet",
    "lahmacun","künefe",
    # Giyim (spesifik)
    "kaftan","fes","sarık","bindallı","şalvar",
    # Coğrafi / Tarihi Yerler
    "kapadokya","efes","truva","bergama","afrodisias",
    "göbeklitepe","çatalhöyük","hattuşa","nemrut","zeugma",
    "topkapı","dolmabahçe","ayasofya","sultanahmet","galata",
    "anıtkabir","gelibolu",
    # Önemli Kişiler
    "atatürk","fatih","kanuni","yavuz","abdülhamid",
    "mimar sinan","kurtuluş savaşı",
    "istiklal","cumhuriyet","inkılap",
    # Spor kültürü (geleneksel)
    "kırkpınar","cirit","okçuluk","ata sporları",
]

MEDIUM_KEYWORDS = [
    "tarih","tarihi","kültür","kültürel","sanat","edebiyat",
    "mimari","mimar","antik","medeniyet","tören","ritüel",
    "opera","tiyatro","bale","sinema","sahne",
    "müzik","enstrüman","çalgı","nota",
    "din","ibadet","inanç","mezhep","cemaat",
    "halk","köylü","yerel","yöresel","bölgesel",
    "giysi","kostüm","kıyafet","süsleme","bezeme",
    "şehir","kasaba","yerleşim","kent","mahalle",
    "lezzet","tat","aşçı","pişirme","gastronomi",
    "eser","yapıt","koleksiyon","sergi",
    "millet","ulus","toplum","halk bilimi",
    "lehçe","şive","ağız","dil","alfabe",
    "mitoloji","efsane","söylence","inanış",
]

PENALTY_KEYWORDS = {
    "futbol","basketbol","voleybol","tenis","maç","lig",
    "gol","penaltı","hakem","stadyum","playoff","sezon",
    "transfer","şampiyonluk","turnuva",
    "problem","denklem","algoritma","teorem","formül",
    "matris","vektör","integral","türev","fonksiyon",
    "kuantum","elektron","proton",
    "yazılım","programlama","veritabanı",
    "hastalık","sendrom","tedavi","enfeksiyon","virüs",
    "borsa","hisse","enflasyon","faiz",
    "şirket","firma","holding",
}


# ═════════════════════════════════════════════════════════
#  KEYWORD ÖN-ELEK  (Pure Python, anında)
# ═════════════════════════════════════════════════════════
def keyword_precheck(text, title):
    """
    Hızlı keyword taraması. Hiç kültürel keyword bulamazsa
    False döner → Zemberek/NER çağrısına gerek kalmaz.
    """
    tl = title.lower()
    tx = text[:2000].lower()
    for kw in STRONG_KEYWORDS:
        if kw in tl or kw in tx:
            return True
    # Whitelist başlık da geçer
    for kw in TITLE_WHITELIST:
        if kw in tl:
            return True
    return False


# ═════════════════════════════════════════════════════════
#  ZEMBEREK – kelime bazlı
# ═════════════════════════════════════════════════════════
def zemberek_analyze(text):
    """Kelime bazlı Zemberek → (prop_set, lemma_list)"""
    words = _WORD.findall(text[:SNIPPET])
    props = set()
    lemmas = []
    for w in words:
        if len(w) < 2:
            continue
        try:
            results = list(morphology.analyze(w))
            if not results:
                continue
            best = results[0]
            fmt = str(best.formatLong())
            lems = list(best.getLemmas())
            lemma = str(lems[0]).lower() if lems else w.lower()
            lemmas.append(lemma)
            if "Prop" in fmt:
                props.add(w)
        except Exception:
            continue
    return props, lemmas


# ═════════════════════════════════════════════════════════
#  NLTK NER
# ═════════════════════════════════════════════════════════
def nltk_ner(text):
    """NLTK NER → {PERSON: set, GPE: set, ORGANIZATION: set}"""
    ents = {"PERSON": set(), "GPE": set(), "ORGANIZATION": set()}
    try:
        sents = nltk.sent_tokenize(text[:SNIPPET])
        for s in sents[:5]:
            tree = nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(s)))
            for sub in tree:
                if hasattr(sub, "label") and sub.label() in ents:
                    name = " ".join(w for w, _ in sub.leaves())
                    ents[sub.label()].add(name)
    except Exception:
        pass
    return ents


# ═════════════════════════════════════════════════════════
#  BAŞLIK KONTROLÜ
# ═════════════════════════════════════════════════════════
def check_title(title):
    t = title.lower().strip()
    if re.match(r"^\d{4}$", t) or re.match(r"^\d{4}[-–]\d{2,4}", t):
        return "reject"
    for kw in TITLE_WHITELIST:
        if kw in t:
            return "accept"
    for kw in TITLE_BLACKLIST:
        if kw in t:
            return "reject"
    return "neutral"


# ═════════════════════════════════════════════════════════
#  SKOR HESAPLA
# ═════════════════════════════════════════════════════════
def score_document(text, title, props, lemmas, ner_ents):
    tl = title.lower()
    tx = text.lower()
    lemma_str = " ".join(lemmas)

    score = 0
    strong_hits, medium_hits = [], []

    for kw in STRONG_KEYWORDS:
        if kw in tl:
            score += 12; strong_hits.append(kw)
        elif kw in tx or kw in lemma_str:
            score += min(tx.count(kw), 3) * 3
            strong_hits.append(kw)

    for kw in MEDIUM_KEYWORDS:
        if kw in tl:
            score += 5; medium_hits.append(kw)
        elif kw in tx or kw in lemma_str:
            score += 1; medium_hits.append(kw)

    # Zemberek Prop
    pc = len(props)
    score += min(pc, 10) * 2
    if pc >= 5:
        score += 5

    # NER bonus
    gpe_c  = len(ner_ents["GPE"])
    per_c  = len(ner_ents["PERSON"])
    org_c  = len(ner_ents["ORGANIZATION"])
    if gpe_c >= 2: score += 4
    elif gpe_c >= 1: score += 2
    if per_c >= 2: score += 3
    elif per_c >= 1: score += 1
    if org_c >= 1: score += 2

    # Ceza (hafif)
    penalty = 0
    for kw in PENALTY_KEYWORDS:
        if kw in tl: penalty += 10
        elif kw in tx: penalty += 1
    score = max(0, score - penalty)

    return score, strong_hits, medium_hits, pc, gpe_c, per_c, org_c, penalty


# ═════════════════════════════════════════════════════════
#  SORGU ANALİZİ
# ═════════════════════════════════════════════════════════
def analyze_queries(data, file_name):
    """
    JSON'daki soruları analiz et.
    - token > 2 → kısa sorgu olarak kaydet
    - token > 25 → uzun morfolojik sorgu olarak işaretle
    """
    questions = data.get("questions", {})
    items = questions.get("items", []) if isinstance(questions, dict) else (
        questions if isinstance(questions, list) else []
    )

    short_queries = []
    long_morph_queries = []

    for idx, item in enumerate(items):
        q = item.get("question", "")
        tokens = q.split()
        tc = len(tokens)

        if tc > 25:
            # Uzun morfolojik sorgu
            long_morph_queries.append({
                "file": file_name,
                "query_id": f"q{idx}",
                "query": q,
                "token_count": tc,
                "category": item.get("category", ""),
            })
        elif tc > 2:
            # Kısa sorgu
            short_queries.append({
                "file": file_name,
                "query_id": f"q{idx}",
                "query": q,
                "token_count": tc,
                "category": item.get("category", ""),
            })

    return short_queries, long_morph_queries


# ═════════════════════════════════════════════════════════
#  TEK DOSYA İŞLE
# ═════════════════════════════════════════════════════════
def process(path, file_name):
    try:
        data = json.load(open(path, "r", encoding="utf-8"))
    except Exception:
        return None, [], []

    art = data.get("article", data)
    text  = art.get("content", "")
    title = art.get("title", "")
    if not text:
        return None, [], []

    # Sorgu analizi (hafif, her dosyada yapılır)
    short_q, long_q = analyze_queries(data, file_name)

    # 1) Başlık kontrolü
    tv = check_title(title)
    if tv == "reject":
        return {"ok": False, "why": "title"}, short_q, long_q

    # 2) Keyword ön-elek — hiç keyword yoksa Zemberek/NER çağırmadan ele
    if tv != "accept" and not keyword_precheck(text, title):
        return {"ok": False, "why": "no_keyword"}, short_q, long_q

    # 3) Zemberek morfolojik analiz
    props, lemmas = zemberek_analyze(text)

    # 4) NLTK NER
    ner = nltk_ner(text)

    # 5) Skor
    score, strong, medium, pc, gpe, per, org, pen = \
        score_document(text, title, props, lemmas, ner)

    # 6) Karar
    ns = len(strong)
    nm = len(medium)
    selected = False
    reason = ""

    if tv == "accept" and score >= 12:
        selected, reason = True, "title_whitelist"
    elif ns >= 3 and score >= 25:
        selected, reason = True, "multi_strong"
    elif ns >= 2 and score >= 20:
        selected, reason = True, "strong_double"
    elif ns >= 2 and pc >= 3 and score >= 18:
        selected, reason = True, "strong_prop"
    elif ns >= 2 and (gpe + per) >= 2 and score >= 15:
        selected, reason = True, "strong_ner"

    if not selected:
        return {"ok": False, "why": "score"}, short_q, long_q

    return {
        "ok": True, "title": title, "score": score,
        "reason": reason, "props": len(props),
        "prop_ex": list(props)[:5],
        "strong": strong[:5], "medium": medium[:3],
        "ner_p": per, "ner_g": gpe, "ner_o": org,
        "ner_ex": {
            "PERSON": list(ner["PERSON"])[:3],
            "GPE": list(ner["GPE"])[:3],
            "ORG": list(ner["ORGANIZATION"])[:3],
        },
        "penalty": pen,
    }, short_q, long_q


# ═════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════
def main():
    print("=" * 55)
    print("  RAGTurk Kültürel Filtre – Zemberek + NLTK NER")
    print("=" * 55, flush=True)

    files = sorted(f for f in os.listdir(DATASET) if f.endswith(".json"))
    total = len(files)
    print(f"  {total} belge taranacak.\n", flush=True)

    selected = []
    all_short_q = []
    all_long_q  = []
    counts = defaultdict(int)
    t0 = time.time()

    for i, fn in enumerate(files, 1):
        fp = os.path.join(DATASET, fn)
        r, sq, lq = process(fp, fn)

        all_short_q.extend(sq)
        all_long_q.extend(lq)

        if r is None:
            counts["skip"] += 1
        elif not r["ok"]:
            counts[r["why"]] += 1
        else:
            r["file"] = fn
            selected.append(r)

        if i == 1 or i % 50 == 0 or i == total:
            el = time.time() - t0
            eta = (total - i) * el / max(i, 1) / 60
            print(f"  [{i:>4}/{total}]  seçilen={len(selected):>4}  "
                  f"elenen={counts.get('title',0)+counts.get('score',0)+counts.get('no_keyword',0):>4}  "
                  f"{el:.0f}s geçti  ~{eta:.1f}dk kaldı", flush=True)

    dur = time.time() - t0
    selected.sort(key=lambda d: d["score"], reverse=True)
    sel_count = len(selected)

    # ── Eski dosyaları temizle, yenileri kopyala ────────
    for f in os.listdir(OUTPUT):
        p = os.path.join(OUTPUT, f)
        if os.path.isfile(p): os.remove(p)

    rc = defaultdict(int)
    for d in selected:
        shutil.copy(os.path.join(DATASET, d["file"]), OUTPUT)
        rc[d["reason"]] += 1

    # ── Uzun morfolojik sorgu istatistikleri ─────────────
    all_long_q.sort(key=lambda x: -x["token_count"])
    longest = all_long_q[0] if all_long_q else None

    # ── EKRANA RAPOR ────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  SONUÇ: {sel_count} kültürel belge seçildi  ({dur:.0f}s)")
    print(f"{'='*55}")
    print(f"  Toplam taranan    : {total}")
    print(f"  Başlıkla elenen   : {counts.get('title',0)}")
    print(f"  Keyword'süz elenen: {counts.get('no_keyword',0)}")
    print(f"  Skorla elenen     : {counts.get('score',0)}")
    print(f"  Atlanan           : {counts.get('skip',0)}")
    print(f"  SEÇİLEN           : {sel_count}")
    print()

    print("  Seçim nedeni dağılımı:")
    for r, c in sorted(rc.items(), key=lambda x: -x[1]):
        print(f"    {r}: {c}")
    print()

    print(f"  SORGU ANALİZİ:")
    print(f"    Kısa sorgular (token>2)       : {len(all_short_q)}")
    print(f"    Uzun morfolojik (token>25)     : {len(all_long_q)}")
    if longest:
        print(f"    En uzun sorgu                  : {longest['token_count']} token")
        print(f"    En uzun sorgu dosyası          : {longest['file']}")
        print(f"    En uzun sorgu metni            : {longest['query'][:120]}...")
    print()

    print(f"  En yüksek skorlu 10 belge:")
    for i, d in enumerate(selected[:10], 1):
        print(f"    {i:>2}. [{d['score']:>3}] {d['title'][:50]}")
        print(f"        {d['reason']} | Prop:{d['props']} "
              f"NER(P:{d['ner_p']},G:{d['ner_g']},O:{d['ner_o']})")

    # ── JSON RAPOR ──────────────────────────────────────
    report = {
        "motor": "Zemberek + NLTK NER",
        "secilen": sel_count,
        "taranan": total,
        "sure_saniye": round(dur, 1),
        "baslik_elenen": counts.get("title", 0),
        "keywordsuz_elenen": counts.get("no_keyword", 0),
        "skorla_elenen": counts.get("score", 0),
        "neden_dagilimi": dict(rc),
        "ort_skor": round(sum(d["score"] for d in selected) / sel_count, 1) if sel_count else 0,
        "sorgu_analizi": {
            "kisa_sorgu_sayisi": len(all_short_q),
            "uzun_morfolojik_sayisi": len(all_long_q),
            "en_uzun_token": longest["token_count"] if longest else 0,
            "en_uzun_dosya": longest["file"] if longest else "",
            "en_uzun_sorgu": longest["query"] if longest else "",
        },
        "belgeler": [
            {"dosya": d["file"], "baslik": d["title"], "skor": d["score"],
             "neden": d["reason"], "prop": d["props"],
             "prop_ornekler": d["prop_ex"],
             "ner": d["ner_ex"], "guclu": d["strong"], "ceza": d["penalty"]}
            for d in selected
        ],
    }
    rp = os.path.join(OUTPUT, "_rapor.json")
    json.dump(report, open(rp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n  Rapor: {rp}")

    # ── MORFOLOJİK SORGU LOGU ──────────────────────────
    morph_log = {
        "motor": "Zemberek + NLTK NER",
        "kisa_sorgu_sayisi": len(all_short_q),
        "uzun_morfolojik_sayisi": len(all_long_q),
        "en_uzun_token": longest["token_count"] if longest else 0,
        "en_uzun_dosya": longest["file"] if longest else "",
        "en_uzun_sorgu": longest["query"] if longest else "",
        "ort_token_uzun": round(
            sum(q["token_count"] for q in all_long_q) / len(all_long_q), 1
        ) if all_long_q else 0,
        "uzun_sorgular": all_long_q[:50],  # En uzun 50 tanesi
    }
    mp = os.path.join(LOG_DIR, "morphological_queries.json")
    json.dump(morph_log, open(mp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"  Morfolojik sorgu logu: {mp}")

    print(f"\n{'='*55}")
    print(f"  TAMAMLANDI – {sel_count} belge (Zemberek + NLTK NER)")
    print(f"{'='*55}")
    return sel_count


if __name__ == "__main__":
    main()
