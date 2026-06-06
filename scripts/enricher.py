#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Glossa Master Enricher / Glossa Master Zenginleştirici
======================================================
Adds grammatical and semantic enrichment to vocabulary entries in JSON dictionary files
(partOfSpeech, gender, etymology, synonyms, etc.) using fallback chain.

Mevcut JSON sözlük dosyalarındaki terimlere dilbilgisel ve anlamsal zenginleştirme ekler
(partOfSpeech, gender, etymology, synonyms vb.) ve yedek zinciri (fallback chain) kullanır.
"""

import os
import sys
import orjson
import argparse
import time
import re
import hashlib
import asyncio
from tqdm.asyncio import tqdm as tqdm_async

# Prepend parent directory to sys.path to resolve imports from handlers/
# Üst dizini sys.path'e ekleyerek handlers/ modüllerini yüklemeyi sağlar
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Windows console UTF-8 support / Windows konsol UTF-8 desteği
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from handlers.fallback_chain import FallbackChain, parse_json_response, load_config


# ─── Redis Helpers / Redis Yardımcıları ────────────────────────────────────────

async def get_redis_client(config):
    try:
        import redis.asyncio as aioredis
        rcfg = config.get("redis", {})
        host = rcfg.get("host", "127.0.0.1")
        port = rcfg.get("port", 6379)
        db = rcfg.get("db", 0)
        password = rcfg.get("password", None)
        
        r = aioredis.Redis(host=host, port=port, db=db, password=password, protocol=2, socket_timeout=2.0)
        await r.ping()
        return r
    except Exception as e:
        print(f"\033[1;31m[Redis] ERROR: Redis connection failed / HATA: Redis bağlantısı başarısız oldu: {e}\033[0m")
        return None


def get_enrich_cache_key_sha256(src_lang, tgt_lang, term, definitions):
    def_meanings = []
    for d in definitions:
        if isinstance(d, dict):
            def_meanings.append(f"{d.get('meaning', '')}:{d.get('domain', '')}")
        else:
            def_meanings.append(str(d))
    meaning_str = "||".join(sorted(def_meanings))
    input_str = f"{(term or '').strip().lower()}||{meaning_str}"
    sha = hashlib.sha256(input_str.encode('utf-8')).hexdigest()
    return f"sözlük:{src_lang.upper()}:{tgt_lang.upper()}:{sha}"


# ─── Helpers / Yardımcılar ─────────────────────────────────────────────────────

STOPWORDS = {
    "DE": {"sich", "ein", "eine", "einen", "einem", "einer", "der", "die", "das",
           "den", "dem", "des", "zu", "auf", "an", "in", "mit", "von", "für",
           "über", "unter", "vor", "nach", "bei", "bis", "um", "aus", "durch",
           "gegen", "ohne", "es", "er", "sie", "wir", "ihr", "man", "jdn", "jdm",
           "etw", "jd", "nicht", "kein", "keine", "keinen", "keinem"},
    "TR": {"bir", "bu", "şu", "o", "ve", "ile", "için", "de", "da", "den",
           "dan", "ye", "ya", "ki", "mi", "mı", "mu", "mü", "ne", "her"},
    "EN": {"a", "an", "the", "to", "of", "in", "on", "at", "for", "with",
           "by", "from", "up", "about", "into", "through", "it", "is", "be"},
}

GERMAN_VERB_PREFIXES = [
    "ab", "an", "auf", "aus", "bei", "ein", "mit", "nach", "vor", "zu",
    "durch", "hinter", "über", "unter", "um", "wider",
    "zer", "be", "emp", "ent", "er", "ge", "ver"
]

GERMAN_ADJ_PREFIXES = [
    "un", "ur", "hyper", "super", "pseudo", "inter", "trans", "sub"
]


def find_parent_terms(term_obj, all_terms_by_text, src_lang):
    text = (term_obj.get("term") or "").strip()
    pos = (term_obj.get("partOfSpeech") or "").strip().upper()
    if not text:
        return None

    parents = []
    text_lower = text.lower()

    if " " in text:
        stops = STOPWORDS.get(src_lang, set())
        clean_text = re.sub(r'[^\w\s]', ' ', text_lower)
        tokens = clean_text.split()
        meaningful = [t for t in tokens if t not in stops and len(t) > 1]
        
        for token in meaningful:
            if token in all_terms_by_text:
                parents.append(all_terms_by_text[token]["term"])
                
    elif src_lang == "DE":
        if pos == "VERB":
            for prefix in GERMAN_VERB_PREFIXES:
                if text_lower.startswith(prefix) and len(text_lower) > len(prefix) + 2:
                    base_verb = text_lower[len(prefix):]
                    if base_verb in all_terms_by_text:
                        parent_obj = all_terms_by_text[base_verb]
                        if (parent_obj.get("partOfSpeech") or "").strip().upper() == "VERB":
                            parents.append(parent_obj["term"])
                            break

        elif pos == "NOUN":
            for other_text, other_obj in all_terms_by_text.items():
                if (other_obj.get("partOfSpeech") or "").strip().upper() == "NOUN" and len(text_lower) > len(other_text) + 2:
                    if text_lower.endswith(other_text):
                        parents.append(other_obj["term"])
                        break

        elif pos in ("ADJECTIVE", "ADVERB"):
            for prefix in GERMAN_ADJ_PREFIXES:
                if text_lower.startswith(prefix) and len(text_lower) > len(prefix) + 2:
                    base_word = text_lower[len(prefix):]
                    if base_word in all_terms_by_text:
                        parent_obj = all_terms_by_text[base_word]
                        if (parent_obj.get("partOfSpeech") or "").strip().upper() in ("ADJECTIVE", "ADVERB"):
                            parents.append(parent_obj["term"])
                            break

    parents = list(set([p for p in parents if p.lower() != text_lower]))
    return parents if parents else None


def run_parent_terms_prepass(terms, src_lang):
    all_terms_by_text = {}
    for t in terms:
        term_text = (t.get("term") or "").strip()
        if term_text:
            all_terms_by_text[term_text.lower()] = t

    changed = False
    for t in terms:
        parents = find_parent_terms(t, all_terms_by_text, src_lang)
        existing = t.get("parentTerms")
        if parents != existing:
            if parents:
                t["parentTerms"] = parents
            elif "parentTerms" in t:
                del t["parentTerms"]
            changed = True
            
    return changed


def save_dict(file_path, terms):
    temp_file = file_path + ".tmp"
    with open(temp_file, "wb") as f:
        f.write(orjson.dumps(terms, option=orjson.OPT_INDENT_2))
    os.replace(temp_file, file_path)


def split_coordinate_terms(terms):
    NEW_FIELDS = [
        "partOfSpeech", "gender", "article", "plural", "genitive", "comparative", "superlative",
        "pastTense", "pastParticiple", "presentParticiple", "auxiliary", "conjugation",
        "isRegular", "isSeparable", "isReflexive", "verbPrefix", "pronunciation", "syllables",
        "level", "etymology", "declension", "caseMatrix", "semanticRelations"
    ]
    processed_terms = []
    for item in terms:
        term_text = item.get("term")
        if not term_text:
            processed_terms.append(item)
            continue
        
        pos = (item.get("partOfSpeech") or "").strip().upper()
        if pos in ("IDIOM", "PHRASE"):
            processed_terms.append(item)
            continue
            
        raw_parts = re.split(r'[,;/]|\s+(?:und|&|and|ve)\s+', term_text, flags=re.IGNORECASE)
        parts = [p.strip() for p in raw_parts if p.strip()]
        
        should_split = len(parts) > 1 and all(len(p.split()) <= 2 for p in parts)
        
        if should_split:
            print(f"\033[1;35m  [Split] Splitting / Ayrıştırılıyor: '{term_text}' -> {parts}\033[0m")
            for part in parts:
                new_item = item.copy()
                new_item["term"] = part
                for f in NEW_FIELDS:
                    if f in new_item:
                        del new_item[f]
                if "synonyms" in new_item:
                    del new_item["synonyms"]
                if "antonyms" in new_item:
                    del new_item["antonyms"]
                processed_terms.append(new_item)
        else:
            processed_terms.append(item)
    return processed_terms


def get_batch_enrich_prompt(batch_items, src_lang, tgt_lang, src_name, tgt_name):
    cleaned_items = []
    for item in batch_items:
        cleaned_items.append({
            "term": item.get("term"),
            "definitions": item.get("definitions", [])
        })
    batch_json = orjson.dumps(cleaned_items, option=orjson.OPT_INDENT_2).decode('utf-8')
    
    return f"""You are an expert lexicographer and linguist.
You are performing a MAINTENANCE, AUDIT, and ENRICHMENT task for a batch of dictionary entries.
Analyze the following JSON array of dictionary entries:
{batch_json}

Source Language: {src_name} ({src_lang})
Target/Translation Language: {tgt_name} ({tgt_lang})

TASK INSTRUCTIONS:
For each entry in the input array, perform the following steps:

1. VALIDITY AUDIT (isValid):
   Evaluate if the 'term' is a valid dictionary entry. Mark it as `"isValid": false` if it is:
   - OCR noise (e.g., random characters, page footer fragments).
   - A book metadata/imprint entry leaked from the book structure.
   - A page number or stray numeric reference.
   - A standalone single character or punctuation.
   - Contact details (phone numbers, emails, websites, address blocks, social handles).
   - Publisher info or publishing/printing metadata (e.g., "x yayınları", "yayıncılık", "basımevi", copyright warnings, ISBN codes, etc.).
   If `"isValid": false`, provide a clear explanation in Turkish in the `"reason"` field.

2. CORRECT EXISTING FIELDS:
   Audit and correct the following fields if they are present and incorrect, or fill them if missing:
   - `term`: Fix any spelling mistakes, typos, casing, or possessive forms.
   - `article`: German nouns must have 'der', 'die', or 'das' corresponding to their gender. Correct if wrong.
   - `definitions[].meaning`: Ensure accurate dictionary translation. If the input `meaning` contains multiple comma-separated translations representing distinct nuances or concepts (e.g., "zihin, ruh" for "Geist"), split them into separate definition objects in the `"definitions"` array, and generate a distinct example sentence (example_source, example_target), domain, and usage note for each.
   - `definitions[].domain`: Ensure appropriate domains from the database list: GENERAL, MEDICAL, LEGAL, TECHNICAL, ACADEMIC, PHILOSOPHY, BUSINESS, LITERARY, LITERATURE, SCIENCE, POLITICS, RELIGION, MILITARY, SPORTS, MUSIC, ART, ARCHITECTURE, GASTRONOMY, AGRICULTURE, MATHEMATICS, CHEMISTRY, PHYSICS, BIOLOGY, GEOGRAPHY, HISTORY, PSYCHOLOGY, SOCIOLOGY, EDUCATION, LINGUISTICS, COMPUTER_SCIENCE, ECONOMICS, ENGINEERING, ENVIRONMENTAL, MEDIA, FASHION, TRANSPORT, COLLOQUIAL, SLANG, FORMAL, ARCHAIC, ASTRONOMY, MEDICINE, PHARMACY, THEOLOGY, ANATOMY, ZOOLOGY, BOTANY, MINERALOGY, METEOROLOGY, ELECTRONICS, TELECOMMUNICATIONS, AVIATION, MARITIME, AUTOMOTIVE, TEXTILE, PRINTING, PHOTOGRAPHY, CINEMA, THEATER, DANCE, CUISINE, TOURISM, DIPLOMACY, PEDAGOGY, RHETORIC, MYTHOLOGY.

3. COORDINATE SPLITTING:
   - If the `term` contains coordinate terms, split them into separate term objects in the `"terms"` array.
   - If no split is needed, the `"terms"` array should contain exactly one object.
   - If `"isValid": false`, the `"terms"` array should be empty `[]`.

4. GRAMMATICAL & SEMANTIC ENRICHMENT:
   Determine or correct all linguistic attributes for the term object:
   - `partOfSpeech`: Must be one of: NOUN, VERB, ADJECTIVE, ADVERB, PRONOUN, PREPOSITION, CONJUNCTION, INTERJECTION, ARTICLE, NUMERAL, PARTICLE, PHRASE, IDIOM, PREFIX, SUFFIX.
   - `gender`: For nouns, one of: MASCULINE, FEMININE, NEUTER, COMMON, or null.
   - `article`: Appropriate article (e.g. "der", "die", "das" for German) or null.
   - `plural`: Plural form of the noun or null.
   - `genitive`: Genitive form of the noun (e.g. "des Hauses") or null.
   - `comparative`: Comparative form of the adjective/adverb (e.g. "schöner") or null.
   - `superlative`: Superlative form of the adjective/adverb (e.g. "am schönsten") or null.
   - `pastTense`: Präteritum (past tense) form of the verb (e.g. "ging") or null.
   - `pastParticiple`: Partizip II (past participle) form of the verb (e.g. "gegangen") or null.
   - `presentParticiple`: Partizip I (present participle) form of the verb (e.g. "gehend") or null.
   - `auxiliary`: Auxiliary verb used with the verb ("haben" or "sein") or null.
   - `conjugation`: A JSON object detailing the verb conjugation table or null if not a verb.
   - `isRegular`: boolean (true/false) indicating if verb is regular, or null.
   - `isSeparable`: boolean (true/false) indicating if verb is separable, or null.
   - `isReflexive`: boolean (true/false) indicating if verb is reflexive, or null. Note: Check both the `term` and the provided definitions. If a definition includes a reflexive pronoun (like 'sich' in German, or similar in other languages) or indicates a reflexive meaning/usage, capture it here.
   - `verbPrefix`: separable prefix of the verb (e.g. "ab", "auf") or null.
   - `pronunciation`: IPA pronunciation string or null.
   - `syllables`: Hyphenated syllables (e.g. "A·bend·es·sen") or null.
   - `level`: Language level (A1, A2, B1, B2, C1, C2, UNKNOWN).
   - `etymology`: Short text explaining the origin of the word (in Turkish) or null.
   - `declension`: A JSON object of the noun or adjective declension table (cases: Nominative, Accusative, Dative, Genitive for Singular and Plural) or null.
   - `caseMatrix`: A JSON array of case mappings or null. If the verb is reflexive, governs a preposition, or has specific case requirements, add them here. Each object must have:
     - `reflexive`: The reflexive pronoun (e.g., "sich", "myself", or null).
     - `reflexiveCase`: The case of the reflexive pronoun (e.g., "Akk", "Dat", or null).
     - `prep`: Preposition governed by the verb (e.g., "an", "auf", "mit", or null).
     - `case`: Case governed by the preposition/verb (e.g., "Akk", "Dat", or null).
     - `meaning`: Turkish translation/meaning of this usage.
   - `semanticRelations`: A JSON object representing multilingual semantic relations or null.
   - `synonyms`: An array of synonyms or null.
   - `antonyms`: An array of antonyms or null.

OUTPUT REQUIREMENTS:
- Return ONLY a raw JSON array of objects. Do NOT wrap in ```json or any markdown blocks.
- The length of the output JSON array must be EXACTLY {len(batch_items)}, matching the input batch in the exact same order.
- Explanations, reasons, etymology, prepositions, usageNotes/usageNote must be in {tgt_lang} (Turkish).

JSON Schema for the output array:
[
  {{
    "original_term": "input term",
    "isValid": true | false,
    "reason": "Türkçe gerekçe veya null",
    "terms": [
      {{
        "term": "word",
        "partOfSpeech": "NOUN | VERB | ADJECTIVE | etc.",
        "gender": "MASCULINE | FEMININE | NEUTER | null",
        "article": "appropriate article or null",
        "plural": "plural form or null",
        "genitive": "genitive form or null",
        "comparative": "comparative form or null",
        "superlative": "superlative form or null",
        "pastTense": "past tense form or null",
        "pastParticiple": "past participle form or null",
        "presentParticiple": "present participle form or null",
        "auxiliary": "haben | sein | null",
        "conjugation": {{ ... }} or null,
        "isRegular": true | false | null,
        "isSeparable": true | false | null,
        "isReflexive": true | false | null,
        "verbPrefix": "separable prefix or null",
        "pronunciation": "IPA or null",
        "syllables": "A·bend·es·sen or null",
        "level": "A1 | A2 | B1 | B2 | C1 | C2 | UNKNOWN",
        "etymology": "Türkçe köken açıklaması veya null",
        "declension": {{ ... }} or null,
        "caseMatrix": [ {{ "reflexive": "sich | myself | null", "reflexiveCase": "Akk | Dat | null", "prep": "prep or null", "case": "case of prep or null", "meaning": "Turkish meaning of this usage" }} ] or null,
        "semanticRelations": {{ ... }} or null,
        "synonyms": ["syn1", "syn2"] or null,
        "antonyms": ["ant1", "ant2"] or null,
        "definitions": [
          {{
            "meaning": "translation meaning",
            "domain": "GENERAL | MEDICAL | LEGAL | etc.",
            "usageNote": "appropriate usage note in Turkish or null",
            "example_source": "example sentence in source language",
            "example_target": "translation of example sentence"
          }}
        ]
      }}
    ]
  }}
]
"""


def extract_pair_from_filename(file_path):
    base_name = os.path.basename(file_path).replace(".json", "")
    parts = base_name.split("_")
    for index, part in enumerate(parts):
        if part.lower() == "sozluk" and index + 2 < len(parts):
            src = parts[index + 1].upper()
            return src, parts[index + 2].upper()
    return "DE", "TR"


def apply_enrichment_result(orig_item, response_item, NEW_FIELDS):
    is_valid = response_item.get("isValid", True)
    if not is_valid:
        reason = response_item.get("reason") or "Belirtilmemiş"
        term = orig_item.get("term", "")
        print(f"\033[1;31m  [DELETE] '{term}' is invalid. Reason / Gerekçe: {reason}\033[0m")
        return []

    response_terms = response_item.get("terms")
    if not isinstance(response_terms, list) or len(response_terms) == 0:
        return [orig_item]

    updated_items = []
    for u_idx, new_data in enumerate(response_terms):
        item_copy = orig_item.copy() if u_idx == 0 else {}
        if u_idx > 0:
            for k in ["source", "createdAt", "editedById", "lang"]:
                if k in orig_item:
                    item_copy[k] = orig_item[k]

        for field in NEW_FIELDS:
            if field in new_data:
                item_copy[field] = new_data[field]
            else:
                item_copy.setdefault(field, None)

        if "term" in new_data:
            item_copy["term"] = new_data["term"]
        if "article" in new_data:
            item_copy["article"] = new_data["article"]
        if "definitions" in new_data:
            item_copy["definitions"] = new_data["definitions"]

        for field in ["synonyms", "antonyms"]:
            vals = new_data.get(field)
            if isinstance(vals, list):
                item_copy[field] = ", ".join(vals)
            elif isinstance(vals, str):
                item_copy[field] = vals
            else:
                item_copy[field] = new_data.get(field)

        item_copy.pop("isValid", None)
        item_copy.pop("reason", None)
        updated_items.append(item_copy)
        
    return updated_items


def check_enrich_file_status(file_path):
    try:
        with open(file_path, "rb") as f:
            terms = orjson.loads(f.read())
        if not isinstance(terms, list):
            return None
        
        needs_processing = False
        for term_obj in terms:
            term = (term_obj.get("term") or "").strip()
            if term and not term_obj.get("partOfSpeech"):
                needs_processing = True
                break
        
        if not needs_processing:
            return "COMPLETE"
    except Exception:
        pass
    return None


# ─── Async Main Loop / Asenkron Ana Döngü ──────────────────────────────────────

async def main_async():
    parser = argparse.ArgumentParser(description="LexiForge Master Enricher — Fallback Chain")
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--file", help="JSON dictionary file name or path")
    group.add_argument("--all", action="store_true", help="Process all JSON files")
    parser.add_argument("--limit", type=int, default=0, help="Max terms to enrich")
    parser.add_argument("--batch-size", type=int, help="Batch size for single query")
    parser.add_argument("--provider", help="Filter by a specific provider (openrouter, cerebras, groq, google, github, sambanova, mistral)")
    parser.add_argument("--model", help="Filter by a specific model")
    parser.add_argument("--skip-existing", action="store_true", help="Skip already enriched terms (default: False)")
    parser.add_argument("--semaphore", type=int, default=8, help="Max concurrent requests (Semaphore)")
    args = parser.parse_args()

    from handlers.selector import select_item_interactive, select_boolean_interactive, select_file_interactive

    config = load_config()
    json_dir = os.path.join("data", "final")
    json_files = []

    # 1. JSON Selection
    if args.file:
        if os.path.exists(args.file):
            json_files.append(args.file)
        else:
            full_path = os.path.join(json_dir, args.file)
            if os.path.exists(full_path):
                json_files.append(full_path)
            else:
                print(f"Error: JSON file '{args.file}' not found. / Hata: '{args.file}' JSON dosyası bulunamadı.")
                sys.exit(1)
    elif args.all:
        if os.path.exists(json_dir):
            for f in os.listdir(json_dir):
                if f.endswith(".json") and not f.endswith(".tmp") and "sozluk_" in f and "_progress" not in f:
                    json_files.append(os.path.join(json_dir, f))
    else:
        action_options = [
            "Enrich a single JSON file / Tek bir JSON dosyasını zenginleştir", 
            "Enrich all JSON files / Tüm JSON dosyalarını zenginleştir"
        ]
        selected_action = select_item_interactive(action_options, "Please select an action / Lütfen yapmak istediğiniz işlemi seçin:")
        if not selected_action:
            print("Selection cancelled. / Seçim iptal edildi.")
            sys.exit(0)

        if "Enrich all" in selected_action or "Tüm JSON" in selected_action:
            args.all = True
            if os.path.exists(json_dir):
                for f in os.listdir(json_dir):
                    if f.endswith(".json") and not f.endswith(".tmp") and "sozluk_" in f and "_progress" not in f:
                        json_files.append(os.path.join(json_dir, f))
        else:
            all_jsons = []
            if os.path.exists(json_dir):
                for f in os.listdir(json_dir):
                    if f.endswith(".json") and not f.endswith(".tmp") and "sozluk_" in f and "_progress" not in f:
                        all_jsons.append(os.path.join(json_dir, f))
            if not all_jsons:
                print("No JSON files to enrich found. / Zenginleştirilecek JSON dosyası bulunamadı.")
                sys.exit(1)

            status_dict = { f: check_enrich_file_status(f) for f in all_jsons }
            selected = select_file_interactive(all_jsons, "Select the JSON dictionary file to enrich / Zenginleştirilecek JSON sözlük dosyasını seçin", status_dict=status_dict)
            if not selected:
                print("Selection cancelled. / Seçim iptal edildi.")
                sys.exit(0)
            json_files.append(selected)
            args.file = selected

    # 2. Skip Existing
    if "--skip-existing" not in sys.argv:
        args.skip_existing = select_boolean_interactive(
            "Should already enriched terms be skipped? / Daha önce zenginleştirilmiş terimler atlansın mı (Skip already enriched)?",
            default=True
        )
    args.maintenance = not args.skip_existing

    # 3. Limit
    if "--limit" not in sys.argv:
        limit_confirm = select_boolean_interactive("Do you want to set a maximum limit? / Zenginleştirme işlemi için maksimum sınır (Limit) belirlemek ister misiniz?", default=False)
        if limit_confirm:
            try:
                val = input("Enter max term limit (e.g. 50) / Maksimum terim sınırı girin (örn. 50): ").strip()
                args.limit = int(val) if val else 0
            except ValueError:
                args.limit = 0
                print("Invalid value. Limit removed. / Geçersiz değer. Sınır kaldırıldı.")

    # 4. Batch Size
    if "--batch-size" not in sys.argv:
        batch_options = [
            "30 (Recommended) / 30 (Önerilen)", 
            "10", 
            "50", 
            "Enter custom value... / Özel Değer Girin..."
        ]
        selected_batch = select_item_interactive(batch_options, "What should be the batch size? / Tek seferde gönderilecek paket boyutu (Batch Size) ne olsun?")
        if not selected_batch or "30" in selected_batch:
            args.batch_size = 30
        elif selected_batch == "10":
            args.batch_size = 10
        elif selected_batch == "50":
            args.batch_size = 50
        else:
            try:
                val = input("Enter custom batch size / Özel paket boyutu girin: ").strip()
                args.batch_size = int(val) if val else 30
            except ValueError:
                args.batch_size = 30
                print("Invalid value. Defaulting to 30. / Geçersiz değer. Paket boyutu 30 olarak belirlendi.")

    if "--semaphore" not in sys.argv:
        try:
            sem_input = input("Enter max concurrent requests (Semaphore) (Default 8) / Maksimum eş zamanlı istek sayısını (Semaphore) girin (Varsayılan 8): ").strip()
            args.semaphore = int(sem_input) if sem_input else 8
        except ValueError:
            args.semaphore = 8

    redis_enabled = False
    redis_client = None
    try:
        import redis
        has_redis_lib = True
    except ImportError:
        has_redis_lib = False
        print("\033[1;33m[Warning] Python 'redis' package is not installed. Running file-based mode. / [Warning] Python 'redis' paketi yüklü değil. Dosya tabanlı çalışılacak.\033[0m")

    if has_redis_lib:
        if len(sys.argv) > 1:
            redis_enabled = True
        else:
            try:
                redis_enabled = select_boolean_interactive("Should Redis caching be used? / Redis önbellekleme kullanılsın mı?", default=True)
            except Exception:
                redis_enabled = True

        if redis_enabled:
            redis_client = await get_redis_client(config)
            if redis_client:
                print("\033[1;32m[Redis] Active. Cache will be used. / [Redis] Aktif. Önbellek kullanılacak.\033[0m")
            else:
                print("\033[1;31m[Redis] Connection failed. Continuing file-based... / [Redis] Bağlantı kurulamadı. Dosya tabanlı devam ediliyor...\033[0m")
                redis_enabled = False

    provider_filter = args.provider
    if not provider_filter:
        api_providers = config.get("api_providers", {})
        enabled_providers = [k for k, v in api_providers.items() if v.get("enabled", True)]
        if enabled_providers:
            options = ["All Providers (Full Fallback Chain) / Tüm Sağlayıcılar (Tam Yedek Zinciri)"] + enabled_providers
            try:
                selected_opt = select_item_interactive(options, "Select a Provider to enrich terms with / Terimlerin zenginleştirileceği Sağlayıcıyı seçin")
                if not selected_opt:
                    print("Selection cancelled. / Seçim iptal edildi.")
                    sys.exit(0)
                if not selected_opt.startswith("All Providers"):
                    provider_filter = selected_opt
            except ImportError:
                pass

    model_filter = args.model
    if provider_filter and not model_filter:
        models = config.get("api_providers", {}).get(provider_filter, {}).get("models", [])
        if models:
            options = ["All Models (Fallback Chain) / Tüm Modeller (Yedek Zinciri)"] + models
            try:
                selected_opt = select_item_interactive(options, f"Select a Model from {provider_filter} / {provider_filter} için bir Model seçin")
                if not selected_opt:
                    print("Selection cancelled. / Seçim iptal edildi.")
                    sys.exit(0)
                if not selected_opt.startswith("All Models"):
                    model_filter = selected_opt
            except ImportError:
                pass

    chain = FallbackChain(config, provider_filter=provider_filter, model_filter=model_filter)

    if chain.get_total_model_count() == 0:
        print("\033[1;31m[Enricher] ERROR: No available provider/model found! / HATA: Hiç kullanılabilir provider/model bulunamadı!\033[0m")
        sys.exit(1)

    print(f"\033[1;36m[Enricher] Fallback chain ready: {chain.get_total_model_count()} models ({chain.get_available_count()} active) / "
          f"Fallback zinciri hazır: {chain.get_total_model_count()} model ({chain.get_available_count()} aktif)\033[0m")
    print(f"\033[36m[Enricher] Provider order: {', '.join(dict.fromkeys(t.provider_key for t in chain.trackers))} / "
          f"Sağlayıcı sırası: {', '.join(dict.fromkeys(t.provider_key for t in chain.trackers))}\033[0m")

    if not json_files:
        print("No JSON files found to enrich. / Zenginleştirilecek JSON dosyası bulunamadı.")
        return

    NEW_FIELDS = [
        "partOfSpeech", "gender", "article", "plural", "genitive", "comparative", "superlative",
        "pastTense", "pastParticiple", "presentParticiple", "auxiliary", "conjugation",
        "isRegular", "isSeparable", "isReflexive", "verbPrefix", "pronunciation", "syllables",
        "level", "etymology", "declension", "caseMatrix", "semanticRelations"
    ]

    for file_path in json_files:
        file_name = os.path.basename(file_path)
        print(f"\n\033[1;33m{'='*60}\033[0m")
        print(f"\033[1;33m  Processing / İşleniyor: {file_name}\033[0m")
        print(f"\033[1;33m{'='*60}\033[0m")

        try:
            with open(file_path, "rb") as f:
                terms = orjson.loads(f.read())
            if not isinstance(terms, list):
                print(f"  Skipping {file_name} (not a list) / Atlanıyor (liste değil)")
                continue
            
            terms_before_len = len(terms)
            terms = split_coordinate_terms(terms)
            if len(terms) != terms_before_len:
                print(f"  Split coordinate terms: {terms_before_len} -> {len(terms)} terms. / Koordinat terimler ayrıldı: {terms_before_len} -> {len(terms)} terim.")
                save_dict(file_path, terms)
        except Exception as e:
            print(f"  Error reading {file_name}: {e}")
            continue

        source_lang, target_lang = extract_pair_from_filename(file_path)
        languages_cfg = config.get("languages", {})
        src_name = languages_cfg.get(source_lang, source_lang)
        tgt_name = languages_cfg.get(target_lang, target_lang)

        print(f"  [Python] Parent terms pre-pass starting... / Parent terms ön geçişi (pre-pass) başlatılıyor Metodlar...")
        changed_parents = run_parent_terms_prepass(terms, source_lang)
        if changed_parents:
            save_dict(file_path, terms)
            print(f"  [Python] Parent terms pre-pass completed and saved. / Parent terms ön geçişi tamamlandı ve kaydedildi.")
        else:
            print(f"  [Python] No changes in parent-child relations. / Parent-child ilişkilerinde değişiklik olmadı.")

        cooldown = chain.get_cooldown()
        batch_size = args.batch_size

        term_infos = []
        for idx, term_obj in enumerate(terms):
            term = (term_obj.get("term") or "").strip()
            needs_processing = True
            if not term:
                needs_processing = False
            elif not args.maintenance and term_obj.get("partOfSpeech"):
                needs_processing = False
            
            if needs_processing:
                term_infos.append((idx, term, term_obj))

        cached_responses = {}
        if redis_client and term_infos:
            cache_keys = []
            key_to_info = {}
            for idx, term, term_obj in term_infos:
                key = get_enrich_cache_key_sha256(source_lang, target_lang, term, term_obj.get("definitions", []))
                cache_keys.append(key)
                key_to_info[key] = (idx, term, term_obj)
            
            try:
                print(f"  [Redis] Querying cache (MGET) for {len(cache_keys)} terms... / {len(cache_keys)} terim için toplu zenginleştirme cache (MGET) sorgulanıyor...")
                redis_results = await redis_client.mget(cache_keys)
                for key, val in zip(cache_keys, redis_results):
                    if val is not None:
                        cached_responses[key] = orjson.loads(val)
            except Exception as ce:
                print(f"  [Redis] MGET cache check error: {ce}")

        cache_hits = {}
        pending_api_items = []
        pending_api_idx = []
        
        for idx, term, term_obj in term_infos:
            key = get_enrich_cache_key_sha256(source_lang, target_lang, term, term_obj.get("definitions", []))
            if key in cached_responses:
                cache_hits[idx] = cached_responses[key]
            else:
                if args.limit > 0 and len(pending_api_items) >= args.limit:
                    continue
                pending_api_items.append(term_obj)
                pending_api_idx.append(idx)

        batches = [pending_api_items[i:i + batch_size] for i in range(0, len(pending_api_items), batch_size)]
        sem = asyncio.Semaphore(args.semaphore)

        async def enrich_batch_with_split(batch_items, depth=0):
            if not batch_items:
                return []
                
            prompt = get_batch_enrich_prompt(batch_items, source_lang, target_lang, src_name, tgt_name)
            
            async with sem:
                resp_text = await asyncio.to_thread(
                    chain.call_with_fallback,
                    prompt,
                    context={"phase": "enrich", "target_lang": target_lang, "depth": depth, "batch_size": len(batch_items)}
                )

            is_valid = False
            results = []
            if resp_text is not None:
                try:
                    results = parse_json_response(resp_text)
                    if isinstance(results, list) and len(results) == len(batch_items):
                        is_valid = True
                except Exception:
                    pass

            if is_valid:
                if redis_client:
                    try:
                        pipe = redis_client.pipeline(transaction=False)
                        for idx, item in enumerate(batch_items):
                            res = results[idx]
                            cache_key = get_enrich_cache_key_sha256(source_lang, target_lang, item["term"], item.get("definitions", []))
                            pipe.set(cache_key, orjson.dumps(res))
                        await pipe.execute()
                    except Exception as ce:
                        print(f"Redis pipeline write error: {ce}")
                return results
            else:
                if len(batch_items) > 1:
                    mid = len(batch_items) // 2
                    left_half = batch_items[:mid]
                    right_half = batch_items[mid:]
                    print(f"\033[1;33m  [Split] Enrichment validation failed. Splitting batch (Size: {len(batch_items)}) in two... Depth / Derinlik: {depth+1}\033[0m")
                    
                    left_res = await enrich_batch_with_split(left_half, depth + 1)
                    right_res = await enrich_batch_with_split(right_half, depth + 1)
                    return left_res + right_res
                else:
                    print(f"\033[1;31m  [Split-FAIL] Term '{batch_items[0]['term']}' could not be enriched. / Terim '{batch_items[0]['term']}' zenginleştirilemedi.\033[0m")
                    return [{"isValid": True, "terms": [batch_items[0]]}]

        async def run_batch(b_idx, batch_items):
            res = await enrich_batch_with_split(batch_items)
            if cooldown > 0:
                await asyncio.sleep(cooldown)
            return b_idx, res

        terms_map = [item for item in terms]
        
        for idx, response_item in cache_hits.items():
            orig_item = terms_map[idx]
            updated = apply_enrichment_result(orig_item, response_item, NEW_FIELDS)
            terms_map[idx] = updated

        def flatten_terms(terms_list):
            flat = []
            for item in terms_list:
                if isinstance(item, list):
                    flat.extend(item)
                else:
                    flat.append(item)
            return flat

        if batches:
            save_lock = asyncio.Lock()
            last_save_time = time.time()
            save_interval = 15.0
            completed_batches = 0
            total_batches = len(batches)

            tasks = [run_batch(b_idx, b) for b_idx, b in enumerate(batches)]
            
            for future in asyncio.as_completed(tasks):
                b_idx, b_res = await future
                batch_indices = pending_api_idx[b_idx * batch_size : (b_idx + 1) * batch_size]
                
                async with save_lock:
                    for idx, response_item in zip(batch_indices, b_res):
                        orig_item = terms_map[idx]
                        updated = apply_enrichment_result(orig_item, response_item, NEW_FIELDS)
                        terms_map[idx] = updated
                        
                    completed_batches += 1
                    current_time = time.time()
                    is_last = (completed_batches >= total_batches)
                    
                    if is_last or (current_time - last_save_time >= save_interval):
                        flat_terms = flatten_terms(terms_map)
                        run_parent_terms_prepass(flat_terms, source_lang)
                        save_dict(file_path, flat_terms)
                        last_save_time = current_time
        else:
            if cache_hits:
                flat_terms = flatten_terms(terms_map)
                run_parent_terms_prepass(flat_terms, source_lang)
                save_dict(file_path, flat_terms)

        terms = flatten_terms(terms_map)
        print(f"\n  Finished enriching / Zenginleştirme tamamlandı: {file_name}. {len(terms)} terms in dictionary.")

    if redis_client:
        await redis_client.close()

    chain.print_stats()
    print("\n\033[1;32mEnrichment finished / Zenginleştirme tamamlandı.\033[0m")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
