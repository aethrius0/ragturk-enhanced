"""
Turkish Query Classifier - STANZA ONLY VERSION
===============================================
Sorguları analiz eder ve sınıflandırır.
Zemberek yerine sadece Stanza kullanır.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Tuple, Dict, Set, Optional

import stanza


# =========================
# Cultural lexicon
# =========================
TURKEY_PROVINCES: Set[str] = {
    "adana", "adiyaman", "afyonkarahisar", "ağrı", "aksaray", "amasya", "ankara",
    "antalya", "ardahan", "artvin", "aydın", "balıkesir", "bartın", "batman",
    "bayburt", "bilecik", "bingöl", "bitlis", "bolu", "burdur", "bursa", "çanakkale",
    "çankırı", "çorum", "denizli", "diyarbakır", "düzce", "edirne", "elazığ",
    "erzincan", "erzurum", "eskişehir", "gaziantep", "giresun", "gümüşhane",
    "hakkari", "hatay", "ığdır", "isparta", "istanbul", "izmir", "kahramanmaraş",
    "karabük", "karaman", "kars", "kastamonu", "kayseri", "kilis", "kırıkkale",
    "kırklareli", "kırşehir", "kocaeli", "konya", "kütahya", "malatya", "manisa",
    "mardin", "mersin", "muğla", "muş", "nevşehir", "niğde", "ordu", "osmaniye",
    "rize", "sakarya", "samsun", "siirt", "sinop", "sivas", "şanlıurfa", "şırnak",
    "tekirdağ", "tokat", "trabzon", "tunceli", "uşak", "van", "yalova", "yozgat",
    "zonguldak"
}

TURKISH_INSTITUTIONS: Set[str] = {
    "tbmm", "türk dil kurumu", "tdk", "yök", "tübitak", "diyanet", "adalet bakanlığı",
    "milli eğitim bakanlığı", "sağlık bakanlığı", "içişleri bakanlığı",
    "cumhurbaşkanlığı", "anayasa mahkemesi", "danıştay", "yargıtay",
    "rtük", "sgk", "türkiye cumhuriyet merkez bankası", "merkez bankası"
}

TURKISH_FOODS: Set[str] = {
    "mantı", "iskender", "cağ kebabı", "adana kebap", "urfa kebap", "lahmacun",
    "çiğ köfte", "baklava", "kadayıf", "künefe", "keşkek", "tarhana", "menemen",
    "imam bayıldı", "hünkar beğendi", "etli ekmek", "pişmaniye", "mesir macunu",
    "aşure", "ezogelin", "çiğ börek", "boyoz", "kokoreç", "simit"
}

TURKISH_CULTURE_TERMS: Set[str] = {
    "atatürk", "cumhuriyet", "osmanlı", "selçuklu", "türkiye", "anadolu",
    "karadeniz", "ege", "iç anadolu", "doğu anadolu", "güneydoğu anadolu",
    "marmara", "akdeniz", "kurtuluş savaşı", "istiklal marşı", "mevlana",
    "yunus emre", "nasreddin hoca", "hacı bektaş veli", "kapadokya", "ayasofya",
    "topkapı sarayı", "anıtkabir", "boğaziçi", "galata", "çanakkale", "nemrut"
}

CULTURAL_LEXICON: Set[str] = (
    TURKEY_PROVINCES
    | TURKISH_INSTITUTIONS
    | TURKISH_FOODS
    | TURKISH_CULTURE_TERMS
)


# =========================
# Data structure
# =========================
@dataclass
class QueryClassification:
    question: str
    tokens: List[str]
    token_count: int
    entities: List[Tuple[str, str]]
    entity_count: int
    morph_density: float
    cultural_matches: List[str]
    labels: List[str]
    lemmas: List[str]
    debug: Dict


# =========================
# Classifier (Stanza Only)
# =========================
class TurkishQueryClassifierStanza:
    def __init__(
        self,
        entity_threshold: int = 2,
        long_token_threshold: int = 18,
        morph_density_threshold: float = 1.35,
        cultural_match_threshold: int = 1,
    ) -> None:
        self.entity_threshold = entity_threshold
        self.long_token_threshold = long_token_threshold
        self.morph_density_threshold = morph_density_threshold
        self.cultural_match_threshold = cultural_match_threshold

        self._load_stanza()

    # -------------------------
    # Initialization
    # -------------------------
    def _load_stanza(self) -> None:
        # NER + Lemma pipeline
        self.nlp = stanza.Pipeline(
            lang="tr",
            processors="tokenize,lemma,ner",
            use_gpu=False,
            verbose=False,
        )
        print("Stanza hazir (NER + Lemma).", flush=True)

    # -------------------------
    # Normalization / tokenization
    # -------------------------
    @staticmethod
    def normalize_text(text: str) -> str:
        text = text.strip()
        text = text.replace("'", "'").replace("`", "'")
        text = re.sub(r"\s+", " ", text)
        return text

    @staticmethod
    def normalize_for_match(text: str) -> str:
        text = TurkishQueryClassifierStanza.normalize_text(text).lower()
        text = re.sub(r"[^\w\sçğıöşü]", " ", text, flags=re.UNICODE)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def simple_tokenize(text: str) -> List[str]:
        text = TurkishQueryClassifierStanza.normalize_text(text)
        return re.findall(r"\b[\wçğıöşüÇĞİÖŞÜ']+\b", text, flags=re.UNICODE)

    # -------------------------
    # NER with Stanza
    # -------------------------
    def extract_entities(self, doc) -> List[Tuple[str, str]]:
        entities: List[Tuple[str, str]] = []
        for ent in doc.ents:
            entities.append((ent.text, ent.type))
        return entities

    # -------------------------
    # Stanza morphology (lemma + complexity estimate)
    # -------------------------
    def analyze_with_stanza(self, doc) -> Tuple[float, List[Dict], List[str]]:
        """
        Stanza ile morfolojik analiz yap.
        Complexity: token uzunluğu / lemma uzunluğu oranına dayalı tahmin.
        """
        details: List[Dict] = []
        lemmas: List[str] = []
        complexities: List[float] = []

        for sent in doc.sentences:
            for word in sent.words:
                token = word.text
                lemma = word.lemma.lower() if word.lemma else token.lower()
                
                # Complexity estimate: token/lemma length ratio
                if len(lemma) > 0:
                    complexity = 1.0 + 0.15 * (len(token) - len(lemma))
                    complexity = max(1.0, complexity)  # minimum 1.0
                else:
                    complexity = 1.0
                
                details.append({
                    "token": token,
                    "lemma": lemma,
                    "upos": word.upos,
                    "feats": word.feats,
                    "complexity": complexity,
                })
                lemmas.append(lemma)
                complexities.append(complexity)

        if not complexities:
            return 0.0, [], []

        density = sum(complexities) / len(complexities)
        return density, details, lemmas

    # -------------------------
    # Cultural matching
    # -------------------------
    def find_cultural_matches(
        self,
        question: str,
        entities: List[Tuple[str, str]],
    ) -> List[str]:
        text_norm = self.normalize_for_match(question)
        matches: Set[str] = set()

        for phrase in CULTURAL_LEXICON:
            if phrase in text_norm:
                matches.add(phrase)

        for ent_text, _ in entities:
            ent_norm = self.normalize_for_match(ent_text)
            if ent_norm in CULTURAL_LEXICON:
                matches.add(ent_norm)

        return sorted(matches)

    # -------------------------
    # Main classification
    # -------------------------
    def classify(self, question: str) -> QueryClassification:
        question = self.normalize_text(question)
        tokens = self.simple_tokenize(question)
        
        # Single Stanza call for all analysis
        doc = self.nlp(question)
        
        entities = self.extract_entities(doc)
        morph_density, morph_details, lemmas = self.analyze_with_stanza(doc)
        cultural_matches = self.find_cultural_matches(question, entities)

        labels: List[str] = []

        # 1) entity-heavy
        if len(entities) >= self.entity_threshold:
            labels.append("entity-heavy")

        # 2) long-morph
        if len(tokens) >= self.long_token_threshold or morph_density >= self.morph_density_threshold:
            labels.append("long-morph")

        # 3) cultural
        if len(cultural_matches) >= self.cultural_match_threshold:
            labels.append("cultural")

        if not labels:
            labels.append("other")

        return QueryClassification(
            question=question,
            tokens=tokens,
            token_count=len(tokens),
            entities=entities,
            entity_count=len(entities),
            morph_density=round(morph_density, 4),
            cultural_matches=cultural_matches,
            labels=labels,
            lemmas=lemmas,
            debug={
                "morph_details": morph_details,
                "entity_threshold": self.entity_threshold,
                "long_token_threshold": self.long_token_threshold,
                "morph_density_threshold": self.morph_density_threshold,
                "cultural_match_threshold": self.cultural_match_threshold,
            },
        )

    def classify_many(self, questions: List[str]) -> List[QueryClassification]:
        return [self.classify(q) for q in questions]


# =========================
# Demo
# =========================
if __name__ == "__main__":
    clf = TurkishQueryClassifierStanza(
        entity_threshold=2,
        long_token_threshold=18,
        morph_density_threshold=1.35,
        cultural_match_threshold=1,
    )

    examples = [
        "Ankara Üniversitesi hangi yılda kurulmuştur?",
        "İstanbul ve Ankara üniversiteleri karşılaştırıldığında hangisi daha eskidir?",
        "Trabzon'a özgü yemeklerden biri hangisidir?",
        "Türkiye Cumhuriyet Merkez Bankası'nın görevleri nelerdir?",
        "Ankara'da bulunmakta olan üniversitelerin kuruluş tarihlerine göre sıralanması istendiğinde en eski üniversite hangisidir?",
        "Üç öğrencinin ikişer kalemi varsa toplam kaç kalemleri vardır?"
    ]

    for q in examples:
        result = clf.classify(q)
        print("=" * 90)
        print("Soru:", result.question)
        print("Etiketler:", result.labels)
        print("Token sayısı:", result.token_count)
        print("Entity sayısı:", result.entity_count)
        print("Entity'ler:", result.entities)
        print("Morph density:", result.morph_density)
        print("Lemmas:", result.lemmas)
        print("Cultural matches:", result.cultural_matches)
