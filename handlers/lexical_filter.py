#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LexiForge 12-Language Lexical Quality Filter & Normalizer
=========================================================
Shared by clean.py, clean_meanings.py, extractor.py, and nim_sql_sync.py.

Covers all 12 Glossa languages (DE, TR, EN, FR, ES, IT, PT, RO, LA, RU, SR, MK)
and all 66 Glossa Prisma Domain enums.
"""

from __future__ import annotations

import re
from typing import Any, Tuple, List

SUPPORTED_LANGS = ("DE", "TR", "EN", "FR", "ES", "IT", "PT", "RO", "LA", "RU", "SR", "MK")
VALID_LANGS = set(SUPPORTED_LANGS)

VALID_POS = {
    "NOUN", "VERB", "ADJECTIVE", "ADVERB", "PRONOUN", "PREPOSITION",
    "CONJUNCTION", "INTERJECTION", "ARTICLE", "NUMERAL", "PARTICLE",
    "PHRASE", "IDIOM", "PREFIX", "SUFFIX",
}

VALID_GENDER = {"MASCULINE", "FEMININE", "NEUTER", "COMMON", "AMBIGENERIC"}
VALID_ANIMACY = {"ANIMATE", "INANIMATE", "NOT_APPLICABLE"}
VALID_ASPECT = {"IMPERFECTIVE", "PERFECTIVE", "BIASPECTUAL"}
VALID_REGISTER = {
    "NEUTRAL", "FORMAL", "INFORMAL", "COLLOQUIAL", "SLANG",
    "ARCHAIC", "LITERARY", "TECHNICAL", "VULGAR",
}
VALID_SCRIPT = {"LATIN", "CYRILLIC", "LATIN_CYRILLIC"}
VALID_LEVEL = {"A1", "A2", "B1", "B2", "C1", "C2", "UNKNOWN"}

LANG_DISPLAY_TO_CODE = {
    "almanca": "DE", "german": "DE", "deutsch": "DE", "de": "DE",
    "türkçe": "TR", "turkce": "TR", "turkish": "TR", "tr": "TR",
    "ingilizce": "EN", "english": "EN", "en": "EN",
    "fransızca": "FR", "fransizca": "FR", "french": "FR", "français": "FR", "fr": "FR",
    "ispanyolca": "ES", "spanish": "ES", "español": "ES", "es": "ES",
    "italyanca": "IT", "italian": "IT", "italiano": "IT", "it": "IT",
    "portekizce": "PT", "portuguese": "PT", "português": "PT", "pt": "PT",
    "romence": "RO", "rumence": "RO", "romanian": "RO", "română": "RO", "ro": "RO",
    "latince": "LA", "latin": "LA", "la": "LA",
    "rusça": "RU", "rusca": "RU", "russian": "RU", "русский": "RU", "ru": "RU",
    "sırpça": "SR", "sirpca": "SR", "serbian": "SR", "српски": "SR", "sr": "SR",
    "makedonca": "MK", "macedonian": "MK", "македонски": "MK", "mk": "MK",
}

POS_LABEL_TO_ENUM = {
    "n.": "NOUN", "n": "NOUN", "noun": "NOUN", "isim": "NOUN", "substantiv": "NOUN",
    "v.": "VERB", "v": "VERB", "verb": "VERB", "fiil": "VERB",
    "adj.": "ADJECTIVE", "adj": "ADJECTIVE", "adjective": "ADJECTIVE", "sıfat": "ADJECTIVE",
    "adv.": "ADVERB", "adv": "ADVERB", "adverb": "ADVERB", "zarf": "ADVERB",
    "pron.": "PRONOUN", "pron": "PRONOUN", "pronoun": "PRONOUN", "zamir": "PRONOUN",
    "prep.": "PREPOSITION", "prep": "PREPOSITION", "preposition": "PREPOSITION", "edat": "PREPOSITION",
    "conj.": "CONJUNCTION", "conj": "CONJUNCTION", "conjunction": "CONJUNCTION", "bağlaç": "CONJUNCTION",
    "int.": "INTERJECTION", "interj.": "INTERJECTION", "interjection": "INTERJECTION", "ünlem": "INTERJECTION",
    "art.": "ARTICLE", "article": "ARTICLE", "artikel": "ARTICLE",
    "num.": "NUMERAL", "num": "NUMERAL", "numeral": "NUMERAL", "sayı": "NUMERAL",
    "part.": "PARTICLE", "part": "PARTICLE", "particle": "PARTICLE",
    "phr.": "PHRASE", "phr": "PHRASE", "phrase": "PHRASE", "ifade": "PHRASE",
    "idiom": "IDIOM", "deyim": "IDIOM",
    "prefix": "PREFIX", "pref.": "PREFIX", "öneki": "PREFIX",
    "suffix": "SUFFIX", "suf.": "SUFFIX", "soneki": "SUFFIX",
}

DOMAIN_TR_TO_ENUM = {
    "genel": "GENERAL", "tıp": "MEDICAL", "hukuk": "LEGAL", "teknik": "TECHNICAL",
    "akademik": "ACADEMIC", "felsefe": "PHILOSOPHY", "ticaret": "BUSINESS",
    "edebiyat": "LITERATURE", "bilim": "SCIENCE", "politika": "POLITICS",
    "din": "RELIGION", "askeri": "MILITARY", "spor": "SPORTS", "müzik": "MUSIC",
    "sanat": "ART", "mimarlık": "ARCHITECTURE", "gastronomi": "GASTRONOMY",
    "tarım": "AGRICULTURE", "matematik": "MATHEMATICS", "kimya": "CHEMISTRY",
    "fizik": "PHYSICS", "biyoloji": "BIOLOGY", "coğrafya": "GEOGRAPHY",
    "tarih": "HISTORY", "psikoloji": "PSYCHOLOGY", "sosyoloji": "SOCIOLOGY",
    "eğitim": "EDUCATION", "dilbilim": "LINGUISTICS", "bilişim": "COMPUTER_SCIENCE",
    "ekonomi": "ECONOMICS", "mühendislik": "ENGINEERING", "çevre": "ENVIRONMENTAL",
    "medya": "MEDIA", "moda": "FASHION", "ulaşım": "TRANSPORT",
    "gündelik": "COLLOQUIAL", "argo": "SLANG", "resmi": "FORMAL",
    "arkaik": "ARCHAIC", "astronomi": "ASTRONOMY", "eczacılık": "PHARMACY",
    "teoloji": "THEOLOGY", "anatomi": "ANATOMY", "zooloji": "ZOOLOGY",
    "botanik": "BOTANY", "mineraloji": "MINERALOGY", "meteoroloji": "METEOROLOGY",
    "elektronik": "ELECTRONICS", "telekom": "TELECOMMUNICATIONS",
    "havacılık": "AVIATION", "denizcilik": "MARITIME", "otomotiv": "AUTOMOTIVE",
    "tekstil": "TEXTILE", "matbaacılık": "PRINTING", "fotoğrafçılık": "PHOTOGRAPHY",
    "sinema": "CINEMA", "tiyatro": "THEATER", "dans": "DANCE",
    "mutfak": "CUISINE", "turizm": "TOURISM", "diplomasi": "DIPLOMACY",
    "pedagoji": "PEDAGOGY", "retorik": "RHETORIC", "mitoloji": "MYTHOLOGY",
}

VALID_DOMAIN = {
    "GENERAL", "MEDICAL", "LEGAL", "TECHNICAL", "ACADEMIC", "PHILOSOPHY",
    "BUSINESS", "LITERARY", "LITERATURE", "SCIENCE", "POLITICS", "RELIGION",
    "MILITARY", "SPORTS", "MUSIC", "ART", "ARCHITECTURE", "GASTRONOMY",
    "AGRICULTURE", "MATHEMATICS", "CHEMISTRY", "PHYSICS", "BIOLOGY",
    "GEOGRAPHY", "HISTORY", "PSYCHOLOGY", "SOCIOLOGY", "EDUCATION",
    "LINGUISTICS", "COMPUTER_SCIENCE", "ECONOMICS", "ENGINEERING",
    "ENVIRONMENTAL", "MEDIA", "FASHION", "TRANSPORT", "COLLOQUIAL",
    "SLANG", "FORMAL", "ARCHAIC", "ASTRONOMY", "MEDICINE", "PHARMACY",
    "THEOLOGY", "ANATOMY", "ZOOLOGY", "BOTANY", "MINERALOGY",
    "METEOROLOGY", "ELECTRONICS", "TELECOMMUNICATIONS", "AVIATION",
    "MARITIME", "AUTOMOTIVE", "TEXTILE", "PRINTING", "PHOTOGRAPHY",
    "CINEMA", "THEATER", "DANCE", "CUISINE", "TOURISM", "DIPLOMACY",
    "PEDAGOGY", "RHETORIC", "MYTHOLOGY",
}

DOMAIN_SYNONYMS = {
    "SPACE": "ASTRONOMY", "ASTROPHYSICS": "ASTRONOMY", "COSMOS": "ASTRONOMY",
    "ARMY": "MILITARY", "WARFARE": "MILITARY", "DEFENSE": "MILITARY", "NAVY": "MILITARY",
    "LAW": "LEGAL", "JURISPRUDENCE": "LEGAL", "CRIMINAL": "LEGAL",
    "TECH": "TECHNICAL", "TECHNOLOGY": "TECHNICAL",
    "IT": "COMPUTER_SCIENCE", "SOFTWARE": "COMPUTER_SCIENCE", "INFORMATICS": "COMPUTER_SCIENCE",
    "HEALTH": "MEDICAL", "CLINICAL": "MEDICINE",
    "COMMERCE": "BUSINESS", "FINANCE": "ECONOMICS",
    "ECOLOGY": "ENVIRONMENTAL", "NATURE": "BIOLOGY", "GRAMMAR": "LINGUISTICS",
    "COOKING": "CUISINE", "FOOD": "GASTRONOMY",
}

LANGUAGE_ARTICLE_RULES = {
    "DE": {
        "articles": {"der", "die", "das", "der/die", "der/das", "die/das", "der/die/das"},
        "genders": {"MASCULINE", "FEMININE", "NEUTER"},
        "a2g": {"der": "MASCULINE", "die": "FEMININE", "das": "NEUTER"},
    },
    "FR": {"articles": {"le", "la", "l'", "les", "le/la"}, "genders": {"MASCULINE", "FEMININE", "COMMON"}, "a2g": {"le": "MASCULINE", "la": "FEMININE"}},
    "ES": {"articles": {"el", "la", "los", "las", "el/la"}, "genders": {"MASCULINE", "FEMININE", "COMMON"}, "a2g": {"el": "MASCULINE", "la": "FEMININE"}},
    "IT": {"articles": {"il", "lo", "la", "l'", "i", "gli", "le", "il/la"}, "genders": {"MASCULINE", "FEMININE", "COMMON"}, "a2g": {"il": "MASCULINE", "lo": "MASCULINE", "la": "FEMININE"}},
    "PT": {"articles": {"o", "a", "os", "as", "o/a"}, "genders": {"MASCULINE", "FEMININE", "COMMON"}, "a2g": {"o": "MASCULINE", "a": "FEMININE"}},
    "RO": {"articles": {"un", "o"}, "genders": {"MASCULINE", "FEMININE", "NEUTER", "AMBIGENERIC"}, "a2g": {}},
    "LA": {"articles": set(), "genders": {"MASCULINE", "FEMININE", "NEUTER", "COMMON"}, "a2g": {}},
    "RU": {"articles": set(), "genders": {"MASCULINE", "FEMININE", "NEUTER", "COMMON"}, "a2g": {}},
    "SR": {"articles": set(), "genders": {"MASCULINE", "FEMININE", "NEUTER"}, "a2g": {}},
    "MK": {"articles": set(), "genders": {"MASCULINE", "FEMININE", "NEUTER"}, "a2g": {}},
    "TR": {"articles": set(), "genders": set(), "a2g": {}},
    "EN": {"articles": set(), "genders": set(), "a2g": {}},
}

POS_GARBAGE_ARTICLES = {"f", "m", "n", "s", "adj", "adv", "v", "is", "pl", "pron", "prep", "gsz"}

# ─── Headword Regexes ─────────────────────────────────────────────────────────
RE_ORDINAL_OR_BARE_NUM = re.compile(
    r"^\s*\d+(?:[.,]\d+)*\s*(?:st|nd|rd|th|['’]?(?:inci|ıncı|uncu|üncü|nci|ncı|ncu|ncü)|\.|°|%)?\s*$",
    re.IGNORECASE,
)
RE_PURE_NUM_PUNCT = re.compile(r"^[0-9\W_]+$")
# Running PDF dictionary headers like "136brodeln –B ruc h" or "152dampfen –d ann"
RE_PDF_RUNNING_HEADER = re.compile(
    r"^\s*\d{1,4}\s*[A-Za-zÄÖÜäöüßÇĞİÖŞÜçğıöşü]{2,}.*[–—]\s*[A-Za-zÄÖÜäöüßÇĞİÖŞÜçğıöşü\s]{2,}\s*$"
)
# Glued leading PDF page numbers like "107Cam elması" or "150daher" (NOT "10-Cent-Münze" or "1,2-Butadien")
RE_GLUED_PAGE_NUM = re.compile(
    r"^(?P<num>\d{2,4})(?=[A-ZÇĞİÖŞÜa-zçğıöşüÄÖÜäöüß\u0400-\u04FF]{3,})(?!\s*[-,])"
)
# Glued OCR notes like "(0. PI.) Ehrenhaftigkeit" or "(o. Pl.) ..."
RE_LEADING_OCR_NOTE = re.compile(
    r"^\(\s*(?:0|o)\.\s*P[lI]\.?\s*\)\s*", re.IGNORECASE
)
RE_LEADING_ZERO_FRAGMENT = re.compile(r"^0(?:\s+[^\d]|$)|^0\d\s+olan$", re.IGNORECASE)
RE_METADATA_PAGE = re.compile(
    r"^(?:seite|page|sayfa)\s*\d+\b|^(?:inhaltsverzeichnis|projektleitung|alle rechte vorbehalten|lektorat)$|^©\s*\d{4}",
    re.IGNORECASE,
)

# Exceptions that are genuine words/compounds even though they start with digits
VALID_DIGIT_COMPOUND_RE = re.compile(
    r"^(?:\d+(?:,\d+)*-[A-Za-zÄÖÜäöüß][-‑A-Za-zÄÖÜäöüß]*|\d+\s*-\s*[A-Za-zÄÖÜäöüß][-‑A-Za-zÄÖÜäöüß]*|\d+\.\s*(?:Januar|Februar|März|April|Mai|Juni|Juli|August|September|Oktober|November|Dezember|January|February|March|April|May|June|July|August|September|October|November|December|Ocak|Şubat|Mart|Nisan|Mayıs|Haziran|Temmuz|Ağustos|Eylül|Ekim|Kasım|Aralık)|\d+‑[A-Za-zÄÖÜäöüß][-‑A-Za-zÄÖÜäöüß]*|\d{1,2}\s+Uhr(?:\s+[^\d].*)?|\d+\s+minutes\s+of\s+fame|\d{1,2}K|\d{1,2}mo|24/7|9/11)$",
    re.IGNORECASE,
)


def is_protected_headword(text: str) -> bool:
    """Return True if headword matches a valid digit-containing chemical/date/time/compound pattern."""
    raw = re.sub(r"\s+", " ", (text or "")).strip()
    return bool(raw and VALID_DIGIT_COMPOUND_RE.match(raw))


def normalize_domain(raw_domain: Any, fallback: str = "GENERAL") -> str:
    if not raw_domain:
        return fallback
    low = str(raw_domain).strip().lower()
    if low in DOMAIN_TR_TO_ENUM:
        return DOMAIN_TR_TO_ENUM[low]
    key = re.sub(r"[^A-Z0-9]+", "_", str(raw_domain).upper().strip()).strip("_")
    key = DOMAIN_SYNONYMS.get(key, key)
    return key if key in VALID_DOMAIN else fallback


def is_junk_meaning(meaning: str, term_text: str = "") -> bool:
    m = (meaning or "").strip()
    if not m or m.lower() in {"null", "none", "n/a", "-", "—", "undefined"}:
        return True
    if RE_ORDINAL_OR_BARE_NUM.match(m) or RE_PURE_NUM_PUNCT.match(m):
        return True
    if term_text and m.lower() == term_text.strip().lower():
        return True
    return False


def classify_headword(
    text: str,
    lang: str = "DE",
    pos: str | None = None,
    definitions: List[dict] | None = None,
) -> Tuple[str, str, List[str]]:
    """
    Classify a dictionary headword as ("keep" | "clean" | "delete", cleaned_text, reasons).
    Catches PDF/OCR garbage like "11th", "11.", "1337", "136brodeln –B ruc h",
    glued page numbers like "107Cam elması", and preserves valid chemical/date/compound terms.
    """
    raw = re.sub(r"\s+", " ", (text or "")).strip()
    if not raw:
        return "delete", "", ["empty"]

    if VALID_DIGIT_COMPOUND_RE.match(raw):
        return "keep", raw, []

    reasons: List[str] = []
    if RE_ORDINAL_OR_BARE_NUM.match(raw):
        reasons.append("ordinal_or_bare_number")
    elif RE_PURE_NUM_PUNCT.match(raw):
        reasons.append("pure_symbol_or_numeric")
    elif RE_PDF_RUNNING_HEADER.match(raw):
        reasons.append("pdf_running_header")
    elif RE_METADATA_PAGE.match(raw):
        reasons.append("metadata_or_page")
    elif RE_LEADING_ZERO_FRAGMENT.match(raw):
        reasons.append("ocr_zero_fragment")

    if reasons:
        return "delete", raw, reasons

    # Single-letter alphabet section headers ("A".."Z", "u") unless genuine 1-letter valid word with real defs
    if len(raw) == 1:
        def_meanings = [(d.get("meaning") or "").strip().lower() for d in (definitions or []) if isinstance(d, dict)]
        non_echo_meanings = [
            m for m in def_meanings
            if m and m not in {raw.lower(), "ß", "ẞ", f"{raw.lower()} harfi"} and not RE_PURE_NUM_PUNCT.match(m)
        ]
        # Pure empty / self-echo single character -> section header or OCR artifact
        if not non_echo_meanings:
            return "delete", raw, ["single_char_header"]
        # Function words / affixes / lowercase articles/pronouns with real non-echo meanings are valid
        if pos in {"ARTICLE", "PRONOUN", "PREPOSITION", "CONJUNCTION", "INTERJECTION", "PARTICLE", "SUFFIX", "PREFIX"}:
            return "keep", raw, []
        if (lang == "EN" and raw in {"a", "I"}) or (lang == "TR" and raw in {"o"}) or (
            lang in {"FR", "ES", "IT", "PT", "RO", "LA", "RU", "SR", "MK"} and raw.islower()
        ):
            return "keep", raw, []
        # Single uppercase/isolated ASCII letter with only trivial <=2 char glosses (e.g. 'A' -> ['I','M','a','O'])
        if all(len(m) <= 2 for m in non_echo_meanings):
            return "delete", raw, ["single_char_header"]
        # Bare isolated letter like 'e', 'I' (DE: ['I', 'eek']) without real multi-char dictionary gloss (>= 5 chars)
        if lang in {"DE", "EN"} and pos in {None, "NOUN", "ADJECTIVE"} and not any(len(m) >= 5 for m in non_echo_meanings):
            return "delete", raw, ["single_char_nonword"]

    cleaned = raw
    clean_reasons: List[str] = []

    if RE_LEADING_OCR_NOTE.match(cleaned):
        cleaned = RE_LEADING_OCR_NOTE.sub("", cleaned).strip()
        clean_reasons.append("stripped_leading_ocr_note")

    if RE_GLUED_PAGE_NUM.match(cleaned):
        # Do not touch chemical prefixes or 14K / 16mo already caught by VALID_DIGIT_COMPOUND_RE
        candidate = RE_GLUED_PAGE_NUM.sub("", cleaned).strip()
        if len(candidate) >= 3:
            cleaned = candidate
            clean_reasons.append("stripped_glued_page_number")

    # Strip stray outer brackets/bullets
    bracket_clean = re.sub(r"^\s*[\[\]{}|<>•*]+|[\[\]{}|<>•*]+\s*$", "", cleaned).strip()
    if bracket_clean and bracket_clean != cleaned:
        cleaned = bracket_clean
        clean_reasons.append("stripped_stray_brackets")

    if not cleaned or RE_ORDINAL_OR_BARE_NUM.match(cleaned) or RE_PURE_NUM_PUNCT.match(cleaned):
        return "delete", raw, clean_reasons + ["became_empty_or_numeric"]

    if cleaned != raw:
        return "clean", cleaned, clean_reasons
    return "keep", raw, []


def clean_definitions(term_text: str, definitions: List[dict]) -> Tuple[List[dict], dict]:
    """Filter non-lexical / ordinal / self-echo definitions and normalize domains."""
    kept: List[dict] = []
    removed = 0
    seen = set()

    for d in definitions or []:
        if not isinstance(d, dict):
            continue
        meaning = re.sub(r"\s+", " ", (d.get("meaning") or "")).strip()
        if is_junk_meaning(meaning, term_text):
            removed += 1
            continue
        dom = normalize_domain(d.get("domain"), "GENERAL")
        key = (meaning.lower(), dom)
        if key in seen:
            removed += 1
            continue
        seen.add(key)
        copy_d = dict(d)
        copy_d["meaning"] = meaning
        copy_d["domain"] = dom
        kept.append(copy_d)

    return kept, {"removed": removed}


def sanitize_article_and_gender(term_obj: dict, src_lang: str = "DE") -> dict:
    """Apply 12-language grammatical consistency rules to article/gender/partOfSpeech."""
    lang = (term_obj.get("lang") or src_lang or "DE").strip().upper()
    if lang not in VALID_LANGS:
        lang = "DE"
    rules = LANGUAGE_ARTICLE_RULES.get(lang, LANGUAGE_ARTICLE_RULES["DE"])

    pos = (term_obj.get("partOfSpeech") or "").strip().upper() or None
    if pos not in VALID_POS:
        pos = None

    art = (term_obj.get("article") or term_obj.pop("artikel", "") or "").strip().lower()
    if art in {"null", "none", "der/die/das/null", "n/a", "no article", "undefined", ""} or art in POS_GARBAGE_ARTICLES:
        art = ""

    gender = (term_obj.get("gender") or "").strip().upper()
    if gender not in VALID_GENDER:
        gender = ""

    if pos is not None and pos != "NOUN":
        art = ""
        gender = ""
    elif not rules["genders"]:
        art = ""
        gender = ""
    else:
        if gender and gender not in rules["genders"]:
            gender = ""
        if art and (not rules["articles"] or art not in rules["articles"]):
            art = ""
        if art and art in rules["a2g"]:
            gender = rules["a2g"][art]

    if "article" in term_obj or art:
        term_obj["article"] = art or None
    if "gender" in term_obj or gender:
        term_obj["gender"] = gender or None
    if pos:
        term_obj["partOfSpeech"] = pos
    return term_obj


_RE_CYRILLIC = re.compile(r"[\u0400-\u04FF]")
_RE_LATIN_LETTER = re.compile(r"[A-Za-zÄÖÜäöüßÇĞİÖŞÜçğıöşüÀ-ÿ]")
_RE_TURKISH_SPEC = re.compile(r"[ğĞşŞıİ]")
_RE_DE_ESZETT = re.compile(r"[ßẞ]")


def detect_language_anomaly(
    text: str,
    lang: str,
    article: str | None = None,
    definitions: List[dict] | None = None,
) -> List[str]:
    """Detect script/orthography/article/translation language mismatches for any of 12 Glossa languages."""
    issues: List[str] = []
    t = (text or "").strip()
    src = (lang or "").strip().upper()
    art = (article or "").strip().lower()

    has_cyr = bool(_RE_CYRILLIC.search(t))
    has_lat = bool(_RE_LATIN_LETTER.search(t))

    if has_cyr and src not in {"RU", "SR", "MK"}:
        issues.append("cyrillic_in_non_slavic")
    if src in {"RU", "MK"} and has_lat and not has_cyr:
        issues.append("missing_cyrillic_script")
    if _RE_TURKISH_SPEC.search(t) and src in {"DE", "EN", "FR", "ES", "IT", "PT", "LA"}:
        issues.append("turkish_chars_in_foreign_term")
    if _RE_DE_ESZETT.search(t) and src in {"TR", "EN", "FR", "ES", "IT", "PT", "RO", "LA"}:
        issues.append("german_eszett_in_non_de")

    if art:
        if src in {"TR", "EN"} and art not in {"null", "none"}:
            issues.append("article_on_genderless_lang")
        elif art in POS_GARBAGE_ARTICLES:
            issues.append("pos_garbage_in_article")
        elif src in LANGUAGE_ARTICLE_RULES:
            allowed_arts = LANGUAGE_ARTICLE_RULES[src]["articles"]
            if allowed_arts and art not in allowed_arts and art not in {"null", "none", ""}:
                issues.append("foreign_article_for_lang")

    for d in definitions or []:
        if not isinstance(d, dict):
            continue
        dlang = (d.get("language") or "").strip().upper()
        meaning = (d.get("meaning") or "").strip()
        if dlang and dlang == src and "same_src_tgt_lang" not in issues:
            issues.append("same_src_tgt_lang")
        if t and meaning and meaning.lower() == t.lower() and "self_echo_meaning" not in issues:
            issues.append("self_echo_meaning")

    return issues


def _clean_opt_str(val: Any, max_len: int = 120) -> str | None:
    if val is None:
        return None
    s = re.sub(r"\s+", " ", str(val)).strip()
    if not s or s.lower() in {"null", "none", "n/a", "undefined", "-"}:
        return None
    return s[:max_len]


def sanitize_grammar_fields(term_obj: dict, src_lang: str = "DE") -> dict:
    """Validate and normalize all 12-language Prisma Term & Definition grammar fields."""
    sanitize_article_and_gender(term_obj, src_lang=src_lang)
    lang = (term_obj.get("lang") or src_lang or "DE").strip().upper()
    if lang not in VALID_LANGS:
        lang = "DE"
    pos = term_obj.get("partOfSpeech")

    # Script & transliteration
    raw_script = str(term_obj.get("script") or "").strip().upper()
    if raw_script not in VALID_SCRIPT:
        if lang in {"RU", "MK"}:
            raw_script = "CYRILLIC"
        elif lang == "SR":
            raw_script = "LATIN_CYRILLIC"
        else:
            raw_script = "LATIN"
    term_obj["script"] = raw_script

    translit = _clean_opt_str(term_obj.get("transliteration"), 180)
    term_obj["transliteration"] = translit if lang in {"RU", "SR", "MK"} else None

    # Canonical lemma
    head_text = _clean_opt_str(term_obj.get("text") or term_obj.get("term"), 180)
    term_obj["lemma"] = _clean_opt_str(term_obj.get("lemma"), 160) or head_text

    # Slavic animacy (RU, SR nouns) and verb aspect (RU, SR, MK verbs)
    anim = str(term_obj.get("animacy") or "").strip().upper()
    if lang in {"RU", "SR"} and (pos in {None, "NOUN"}) and anim in VALID_ANIMACY:
        term_obj["animacy"] = anim
    else:
        term_obj["animacy"] = None

    aspect = str(term_obj.get("aspect") or "").strip().upper()
    if lang in {"RU", "SR", "MK"} and (pos in {None, "VERB"}) and aspect in VALID_ASPECT:
        term_obj["aspect"] = aspect
        term_obj["aspectPair"] = _clean_opt_str(term_obj.get("aspectPair"), 140)
    else:
        term_obj["aspect"] = None
        term_obj["aspectPair"] = None

    # Declension/conjugation/definiteForm/vowelHarmony
    decl_cls = _clean_opt_str(term_obj.get("declensionClass"), 64)
    term_obj["declensionClass"] = decl_cls if lang in {"LA", "RU", "RO", "SR", "DE"} else None

    conj_cls = _clean_opt_str(term_obj.get("conjugationClass"), 64)
    term_obj["conjugationClass"] = conj_cls if (pos in {None, "VERB"}) else None

    def_form = _clean_opt_str(term_obj.get("definiteForm"), 120)
    term_obj["definiteForm"] = def_form if lang in {"RO", "MK"} else None

    vh = _clean_opt_str(term_obj.get("vowelHarmony"), 160)
    term_obj["vowelHarmony"] = vh if lang == "TR" else None

    # Auxiliary sanitization for DE / FR / IT verbs
    aux = _clean_opt_str(term_obj.get("auxiliary"), 32)
    if aux:
        aux_low = aux.lower()
        allowed_aux = {
            "DE": {"haben", "sein", "haben/sein"},
            "FR": {"avoir", "être", "avoir/être"},
            "IT": {"avere", "essere", "avere/essere"},
            "ES": {"haber"},
            "PT": {"ter", "haver"},
            "RO": {"avea", "fi"},
        }.get(lang, set())
        term_obj["auxiliary"] = aux_low if (pos in {None, "VERB"} and aux_low in allowed_aux) else None
    else:
        term_obj["auxiliary"] = None

    # Textual inflection/phonetics fields
    for fname, mlen in (
        ("plural", 120), ("genitive", 120), ("comparative", 120), ("superlative", 120),
        ("pastTense", 120), ("pastParticiple", 120), ("presentParticiple", 120),
        ("verbPrefix", 40), ("pronunciation", 160), ("syllables", 160),
    ):
        term_obj[fname] = _clean_opt_str(term_obj.get(fname), mlen)

    for bname in ("isRegular", "isSeparable", "isReflexive"):
        bval = term_obj.get(bname)
        term_obj[bname] = bval if isinstance(bval, bool) else None

    # Structured JSON fields
    for jname in ("grammarProfile", "caseMatrix", "declension", "conjugation", "semanticRelations"):
        jval = term_obj.get(jname)
        term_obj[jname] = jval if isinstance(jval, (dict, list)) and jval else None

    # Level, domain, secondaryDomains, contextTags
    lvl = str(term_obj.get("level") or "UNKNOWN").strip().upper()
    term_obj["level"] = lvl if lvl in VALID_LEVEL else "UNKNOWN"

    pdom = normalize_domain(term_obj.get("primaryDomain"), "GENERAL")
    term_obj["primaryDomain"] = pdom

    sec_out: List[str] = []
    for sd in term_obj.get("secondaryDomains") or []:
        nd = normalize_domain(sd, "")
        if nd and nd != pdom and nd not in sec_out:
            sec_out.append(nd)
        if len(sec_out) >= 4:
            break
    term_obj["secondaryDomains"] = sec_out

    tags_out: List[str] = []
    for tag in term_obj.get("contextTags") or []:
        ts = re.sub(r"\s+", " ", str(tag)).strip().lower()[:48]
        if ts and ts not in tags_out:
            tags_out.append(ts)
        if len(tags_out) >= 6:
            break
    term_obj["contextTags"] = tags_out

    # Normalize definition registers if definitions list present
    if isinstance(term_obj.get("definitions"), list):
        for d in term_obj["definitions"]:
            if isinstance(d, dict):
                reg = str(d.get("register") or "NEUTRAL").strip().upper()
                d["register"] = reg if reg in VALID_REGISTER else "NEUTRAL"
                d["domain"] = normalize_domain(d.get("domain"), pdom)

    return term_obj
