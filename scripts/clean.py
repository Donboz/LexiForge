#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Glossa Advanced Cleaner & Consolidator / Gelişmiş Temizlik ve Birleştirici
========================================================================
1. Cleans terms in JSON dictionary files (article/artikel mapping, domain normalization).
2. Performs coordinate splitting on non-idiomatic terms (comma, semicolon, slash, etc.).
3. Performs LLM-based language detection and rebucketing if requested.
4. Consolidates and deduplicates all files for each language pair into 'data/merged/sozluk_[SRC]_[TGT].json'.

1. JSON sözlük dosyalarındaki terimleri temizler (artikel -> article, root domain -> definitions[i].domain).
2. POS tag IDIOM veya PHRASE olmayan terimlerde koordinat ayırma (virgül, noktalı virgül, eğik çizgi vb.) yapar.
3. Dil tespiti ve doğru dil grubuna taşıma (rebucketing) yapar.
4. Aynı dil çiftine ait tüm dosyaları birleştirir ve tekilleştirir.
"""

import os
import sys
import orjson
import argparse
import time
import re
import shutil
from collections import defaultdict

# Prepend parent directory to sys.path to resolve imports from handlers/
# Üst dizini sys.path'e ekleyerek handlers/ modüllerini yüklemeyi sağlar
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Reconfigure stdout/stderr to use UTF-8 / UTF-8 rekonfigürasyonu
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from handlers.run_logger import RunLogger, now_iso, save_json_atomic
from handlers.fallback_chain import FallbackChain, parse_json_response, load_config


# ─── Helpers / Yardımcılar ─────────────────────────────────────────────────────

def is_consolidated_filename(filename):
    name = os.path.splitext(os.path.basename(filename))[0]
    parts = name.split("_")
    return (
        len(parts) == 3
        and parts[0].lower() == "sozluk"
        and len(parts[1]) == 2
        and len(parts[2]) == 2
    )


def extract_pair_from_filename(filename):
    base_name = os.path.splitext(os.path.basename(filename))[0]
    parts = base_name.split("_")
    for index, part in enumerate(parts):
        if part.lower() == "sozluk" and index + 2 < len(parts):
            return parts[index + 1].upper(), parts[index + 2].upper()
    return "DE", "TR"


def make_record_key(term_obj):
    term = (term_obj.get("term") or "").strip().lower()
    article = (term_obj.get("article") or "").strip().lower()
    if article in ("null", "none", "der/die/das/null", "n/a", "no article", "undefined"):
        article = ""
    return term, article


def definition_key(def_obj):
    meaning = (def_obj.get("meaning") or "").strip().lower()
    domain = (def_obj.get("domain") or "GENERAL").strip().upper()
    ex_src = (def_obj.get("example_source") or "").strip().lower()
    ex_tgt = (def_obj.get("example_target") or "").strip().lower()
    return meaning, domain, ex_src, ex_tgt


def normalize_term(term_obj, src_lang="DE", default_domain="GENERAL"):
    if not isinstance(term_obj, dict):
        return term_obj

    if "artikel" in term_obj:
        term_obj["article"] = term_obj.pop("artikel")

    for k, v in list(term_obj.items()):
        if isinstance(v, str):
            term_obj[k] = v.strip()

    if "partOfSpeech" in term_obj and term_obj["partOfSpeech"]:
        term_obj["partOfSpeech"] = str(term_obj["partOfSpeech"]).strip().upper()

    term_val = (term_obj.get("term") or "").strip()
    existing_art = (term_obj.get("article") or "").strip().lower()
    if existing_art in ("null", "none", "der/die/das/null", "n/a", "no article", "undefined"):
        existing_art = ""

    if term_val:
        extracted_art = None
        rem_term = term_val
        if src_lang == "DE":
            m_lead = re.match(r'^(der|die|das)\s+(.+)$', term_val, re.IGNORECASE)
            m_trail = re.match(r'^(.+),\s+(der|die|das)$', term_val, re.IGNORECASE)
            if m_lead:
                rem_term, extracted_art = m_lead.group(2).strip(), m_lead.group(1).lower()
            elif m_trail:
                rem_term, extracted_art = m_trail.group(1).strip(), m_trail.group(2).lower()
        elif src_lang == "FR":
            m_lead = re.match(r'^(le|la|les|l\')\s+(.+)$', term_val, re.IGNORECASE)
            m_trail = re.match(r'^(.+),\s+(le|la|les)$', term_val, re.IGNORECASE)
            if m_lead:
                rem_term, extracted_art = m_lead.group(2).strip(), m_lead.group(1).lower()
            elif m_trail:
                rem_term, extracted_art = m_trail.group(1).strip(), m_trail.group(2).lower()
        elif src_lang == "ES":
            m_lead = re.match(r'^(el|la|los|las)\s+(.+)$', term_val, re.IGNORECASE)
            m_trail = re.match(r'^(.+),\s+(el|la|los|las)$', term_val, re.IGNORECASE)
            if m_lead:
                rem_term, extracted_art = m_lead.group(2).strip(), m_lead.group(1).lower()
            elif m_trail:
                rem_term, extracted_art = m_trail.group(1).strip(), m_trail.group(2).lower()
        elif src_lang == "IT":
            m_lead = re.match(r'^(il|lo|la|i|gli|le|l\')\s+(.+)$', term_val, re.IGNORECASE)
            m_trail = re.match(r'^(.+),\s+(il|lo|la|i|gli|le)$', term_val, re.IGNORECASE)
            if m_lead:
                rem_term, extracted_art = m_lead.group(2).strip(), m_lead.group(1).lower()
            elif m_trail:
                rem_term, extracted_art = m_trail.group(1).strip(), m_trail.group(2).lower()

        if extracted_art:
            term_obj["term"] = rem_term
            term_obj["article"] = extracted_art
        else:
            term_obj["article"] = existing_art
    else:
        term_obj["article"] = existing_art

    root_domain = term_obj.pop("domain", None) or default_domain
    if isinstance(root_domain, str):
        root_domain = root_domain.strip().upper()
    else:
        root_domain = default_domain

    definitions = term_obj.setdefault("definitions", [])
    if not isinstance(definitions, list):
        definitions = []
        term_obj["definitions"] = definitions

    for d in definitions:
        if isinstance(d, dict):
            if "meaning" in d and isinstance(d["meaning"], str):
                d["meaning"] = d["meaning"].strip()
            if "domain" in d and d["domain"]:
                d["domain"] = str(d["domain"]).strip().upper()
            else:
                d["domain"] = root_domain
            if "example_source" in d and isinstance(d["example_source"], str):
                d["example_source"] = d["example_source"].strip()
            if "example_target" in d and isinstance(d["example_target"], str):
                d["example_target"] = d["example_target"].strip()

    return term_obj


def split_coordinate_terms(terms):
    NEW_FIELDS = [
        "partOfSpeech", "gender", "article", "plural", "genitive", "comparative", "superlative",
        "pastTense", "pastParticiple", "presentParticiple", "auxiliary", "conjugation",
        "isRegular", "isSeparable", "isReflexive", "verbPrefix", "pronunciation", "syllables",
        "level", "etymology", "declension", "caseMatrix", "semanticRelations"
    ]
    ARTICLES = {"der", "die", "das", "ein", "eine", "den", "dem", "des", "einem", "einer", "eines"}
    processed_terms = []
    
    for item in terms:
        if not isinstance(item, dict):
            processed_terms.append(item)
            continue
            
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
        
        should_split = len(parts) > 1 and all(len(p.split()) <= 2 for p in parts) and not any(p.lower() in ARTICLES for p in parts)
        
        if should_split:
            print(f"  [Split] Splitting / Ayrıştırılıyor: '{term_text}' -> {parts}")
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


def merge_term_objects(existing, incoming):
    existing_defs = existing.setdefault("definitions", [])
    if not isinstance(existing_defs, list):
        existing_defs = []
        existing["definitions"] = existing_defs

    existing_def_keys = {definition_key(d) for d in existing_defs if isinstance(d, dict)}

    for new_def in incoming.get("definitions", []):
        if not isinstance(new_def, dict):
            continue
        d_key = definition_key(new_def)
        if d_key not in existing_def_keys:
            existing_defs.append(new_def)
            existing_def_keys.add(d_key)

    existing_pos = (existing.get("partOfSpeech") or "").strip().upper()
    incoming_pos = (incoming.get("partOfSpeech") or "").strip().upper()
    if not existing_pos and incoming_pos:
        existing["partOfSpeech"] = incoming_pos
    elif existing_pos and not incoming_pos:
        existing["partOfSpeech"] = existing_pos

    for field, val in incoming.items():
        if field in ("term", "partOfSpeech", "article", "definitions"):
            continue
        if val is not None and val != "":
            if existing.get(field) in (None, ""):
                existing[field] = val
            elif isinstance(existing.get(field), str) and isinstance(val, str) and len(val) > len(existing.get(field)):
                existing[field] = val
            elif isinstance(existing.get(field), dict) and isinstance(val, dict):
                existing_dict = existing.setdefault(field, {})
                for k, v in val.items():
                    if v not in (None, "") and existing_dict.get(k) in (None, ""):
                        existing_dict[k] = v


def get_lang_detect_prompt(source_lang, target_lang, allowed_langs, batch):
    lines = []
    for row in batch:
        idx = row["index"]
        term = row["term"]
        meaning = row.get("meaning", "")
        lines.append(f"{idx}. term=\"{term}\" | meaning_hint=\"{meaning}\"")

    return f"""You are a strict lexical language detector.
Source dictionary file is nominally {source_lang}->{target_lang}.
Task: detect ONLY the language of each term headword (not the meaning language).

Allowed language codes: {', '.join(allowed_langs)}
If uncertain, return UNKNOWN.

Items:
{chr(10).join(lines)}

Return ONLY JSON array in same order:
[
  {{"index": 1, "lang": "DE", "confidence": 0.98}},
  {{"index": 2, "lang": "TR", "confidence": 0.90}}
]

Rules:
- lang must be one of allowed codes or UNKNOWN.
- Do not include markdown.
"""


def detect_term_languages(candidates, allowed_langs, chain, batch_size=20):
    detected_langs = {}
    batch_rows = []
    for idx, cand in enumerate(candidates):
        term_text = cand["term"]
        meaning_hint = cand["meaning_hint"]
        batch_rows.append({"index": idx, "term": term_text, "meaning": meaning_hint})
        
    print(f"\n[Language Detector] Running detection on {len(batch_rows)} candidates... / "
          f"Dil Tespiti: {len(batch_rows)} aday üzerinde çalıştırılıyor...")
    
    for start in range(0, len(batch_rows), batch_size):
        batch = batch_rows[start:start + batch_size]
        if not batch:
            continue
            
        first_cand = candidates[batch[0]["index"]]
        src_lang = first_cand["src_lang"]
        tgt_lang = first_cand["tgt_lang"]
        
        prompt = get_lang_detect_prompt(src_lang, tgt_lang, allowed_langs, batch)
        print(f"  Batch {start // batch_size + 1}/{(len(batch_rows) + batch_size - 1) // batch_size} ({len(batch)} terms/terim)...")
        
        resp = chain.call_with_fallback(
            prompt,
            context={"phase": "lang_detect", "batch_start": start},
        )
        
        if resp is None:
            print("    Warning: Language detection batch failed. / Uyarı: Dil tespiti paketi başarısız oldu.")
            continue
            
        try:
            results = parse_json_response(resp)
        except Exception as e:
            print(f"    Error parsing LLM response: {e} / Hata: LLM yanıtı ayrıştırılamadı: {e}")
            continue
            
        if not isinstance(results, list):
            print("    Error: LLM response is not a list. / Hata: LLM yanıtı liste formatında değil.")
            continue
            
        for res in results:
            if not isinstance(res, dict):
                continue
            res_idx = res.get("index")
            lang = (res.get("lang") or "UNKNOWN").strip().upper()
            if isinstance(res_idx, int) and 0 <= res_idx < len(candidates):
                if lang not in allowed_langs:
                    lang = "UNKNOWN"
                detected_langs[res_idx] = lang
                
        time.sleep(chain.get_cooldown())
        
    return detected_langs


def main():
    parser = argparse.ArgumentParser(description="Glossa Advanced Cleaner & Consolidator")
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--file", help="Specific JSON dictionary file name or path")
    group.add_argument("--all", action="store_true", help="Clean and merge all JSON files")
    parser.add_argument("--detect-languages", action="store_true", help="Enable LLM-based language rebucketing")
    parser.add_argument("--batch-size", type=int, default=20, help="LLM language detection batch size")
    parser.add_argument("--limit", type=int, default=0, help="Max term rows to evaluate for language detection")
    parser.add_argument("--dry-run", action="store_true", help="Only report changes, do not write files")
    parser.add_argument("--provider", help="Provider filter for language detection")
    parser.add_argument("--model", help="Model filter for language detection")
    args = parser.parse_args()

    config = load_config()
    json_dir = os.path.join("data", "json")
    merged_dir = os.path.join("data", "merged")
    os.makedirs(merged_dir, exist_ok=True)

    # 1. Identify input files / Girdi dosyalarını belirle
    json_files = []
    if args.file:
        if os.path.exists(args.file):
            json_files.append(args.file)
        else:
            full_path = os.path.join(json_dir, args.file)
            if os.path.exists(full_path):
                json_files.append(full_path)
            else:
                print(f"Error: File '{args.file}' not found. / Hata: '{args.file}' dosyası bulunamadı.")
                sys.exit(1)
    elif args.all:
        if os.path.exists(json_dir):
            for f in os.listdir(json_dir):
                if f.endswith(".json") and not f.endswith(".tmp") and "_sozluk_" in f:
                    if any(x in f for x in ("_progress", "_translate_progress", "_clean_progress", "_clean_report")):
                        continue
                    json_files.append(os.path.join(json_dir, f))
    else:
        # Interactive selector / İnteraktif seçici
        all_jsons = []
        if os.path.exists(json_dir):
            for f in os.listdir(json_dir):
                if f.endswith(".json") and not f.endswith(".tmp") and "_sozluk_" in f:
                    if any(x in f for x in ("_progress", "_translate_progress", "_clean_progress", "_clean_report")):
                        continue
                    all_jsons.append(os.path.join(json_dir, f))
        if not all_jsons:
            print("No JSON files found to clean. / Temizlenecek JSON dosyası bulunamadı.")
            sys.exit(1)
        
        try:
            from handlers.selector import select_file_interactive
            selected = select_file_interactive(all_jsons, "Select a JSON dictionary to clean / Temizlenecek JSON sözlüğü seçin")
            if not selected:
                print("Selection cancelled. / İptal edildi.")
                sys.exit(0)
            json_files.append(selected)
        except ImportError:
            print("Error: selector handler not found. Please provide --file or --all. / Hata: selector modülü bulunamadı. Lütfen --file veya --all parametresi kullanın.")
            sys.exit(1)

    if not json_files:
        print("No JSON files found to process. / İşlenecek JSON dosyası bulunamadı.")
        sys.exit(0)

    # Separate files / Dosyaları ayır
    non_consolidated_sources = []
    consolidated_targets = []
    for fpath in json_files:
        filename = os.path.basename(fpath)
        if is_consolidated_filename(filename):
            consolidated_targets.append(fpath)
        else:
            non_consolidated_sources.append(fpath)

    print(f"Found {len(non_consolidated_sources)} non-consolidated source files. / {len(non_consolidated_sources)} adet birleştirilmemiş kaynak dosya bulundu.")
    print(f"Found {len(consolidated_targets)} consolidated target files. / {len(consolidated_targets)} adet birleştirilmiş hedef dosya bulundu.")

    # 2. Load all terms / Terimleri yükle
    terms_pool = []
    for fpath in json_files:
        filename = os.path.basename(fpath)
        is_cons = is_consolidated_filename(filename)
        src_lang, tgt_lang = extract_pair_from_filename(filename)
        
        print(f"Loading / Yükleniyor: {filename} ({src_lang}->{tgt_lang})...")
        try:
            with open(fpath, "rb") as f:
                loaded = orjson.loads(f.read())
            if not isinstance(loaded, list):
                print(f"  Warning: {filename} is not a JSON list. Skipping. / Uyarı: {filename} geçerli bir liste değil. Atlanıyor.")
                continue
            
            for item in loaded:
                if not isinstance(item, dict) or not item.get("term"):
                    continue
                terms_pool.append({
                    "term_obj": item,
                    "src_lang": src_lang,
                    "tgt_lang": tgt_lang,
                    "original_src_lang": src_lang,
                    "source_file": fpath,
                    "is_consolidated": is_cons
                })
        except Exception as e:
            print(f"  Error reading / Okunurken hata oluştu: {filename}: {e}")
            continue

    print(f"Loaded {len(terms_pool)} terms in total. / Toplam {len(terms_pool)} terim yüklendi.")

    # 3. Language detection / Dil tespiti
    if args.detect_languages and terms_pool:
        chain = FallbackChain(
            config,
            provider_filter=args.provider,
            model_filter=args.model
        )
        if chain.get_total_model_count() == 0:
            print("Error: No active models/providers for language detection. / Hata: Dil tespiti için aktif model/sağlayıcı bulunamadı.")
            sys.exit(1)
            
        languages_cfg = config.get("languages", {})
        allowed_langs = sorted(set(list(languages_cfg.keys()) + ["UNKNOWN"]))
        
        candidates = []
        for idx, item in enumerate(terms_pool):
            if not item["is_consolidated"]:
                first_meaning = ""
                for d in item["term_obj"].get("definitions", []):
                    if isinstance(d, dict) and d.get("meaning"):
                        first_meaning = d.get("meaning")
                        break
                candidates.append({
                    "pool_idx": idx,
                    "term": item["term_obj"]["term"],
                    "meaning_hint": first_meaning,
                    "src_lang": item["src_lang"],
                    "tgt_lang": item["tgt_lang"]
                })
                
        if args.limit > 0:
            candidates = candidates[:args.limit]
            
        if candidates:
            detected_langs = detect_term_languages(candidates, allowed_langs, chain, batch_size=args.batch_size)
            
            moved_count = 0
            for cand_idx, detected_lang in detected_langs.items():
                if detected_lang != "UNKNOWN":
                    pool_idx = candidates[cand_idx]["pool_idx"]
                    old_lang = terms_pool[pool_idx]["src_lang"]
                    if old_lang != detected_lang:
                        terms_pool[pool_idx]["src_lang"] = detected_lang
                        moved_count += 1
                        print(f"  [Rebucket] Term / Terim '{terms_pool[pool_idx]['term_obj']['term']}': {old_lang} -> {detected_lang}")
            print(f"Rebucket complete: moved {moved_count} terms out of {len(candidates)} analyzed. / "
                  f"Yeniden gruplandırma tamamlandı: Analiz edilen {len(candidates)} terimden {moved_count} tanesi taşındı.")

    # 4. Normalize & split coordinate terms / Normalize et ve koordinatları ayır
    print("Normalizing, cleaning and splitting coordinate terms... / "
          "Terimler normalize ediliyor, temizleniyor ve koordinatlar ayrıştırılıyor...")
    processed_pool = []
    deleted_terms = []

    for item in terms_pool:
        term_obj = item["term_obj"]
        term_text = (term_obj.get("term") or "").strip()
        pos = (term_obj.get("partOfSpeech") or "").strip().upper()
        
        is_empty = not term_text
        is_pure_symbol_or_num = bool(re.match(r'^[0-9\W_]+$', term_text)) if term_text else False
        
        metadata_keywords = ["projektleitung", "inhaltsverzeichnis", "alle rechte vorbehalten", "lektorat", "seite", "page"]
        is_metadata = False
        term_lower = term_text.lower()
        for kw in metadata_keywords:
            if kw in term_lower:
                if kw in ("seite", "page"):
                    if re.search(r'\b(?:seite|page)\b\s*\d+', term_lower):
                        is_metadata = True
                        break
                elif term_lower == kw or term_lower.startswith(kw + " ") or term_lower.endswith(" " + kw):
                    is_metadata = True
                    break
                    
        is_too_long = False
        if pos not in ("IDIOM", "PHRASE"):
            word_count = len(term_text.split())
            if word_count > 15 or len(term_text) > 120:
                is_too_long = True
                
        if is_empty or is_pure_symbol_or_num or is_metadata or is_too_long:
            reasons = []
            if is_empty: reasons.append("empty")
            if is_pure_symbol_or_num: reasons.append("pure_symbol_or_numeric")
            if is_metadata: reasons.append("metadata_or_page")
            if is_too_long: reasons.append("too_long")
            
            deleted_terms.append({
                "term": term_text,
                "reason": ", ".join(reasons),
                "source_file": os.path.basename(item["source_file"]),
                "src_lang": item["src_lang"],
                "tgt_lang": item["tgt_lang"]
            })
            continue

        normalized_obj = normalize_term(term_obj, src_lang=item["src_lang"])
        split_objs = split_coordinate_terms([normalized_obj])
        
        for sobj in split_objs:
            new_item = item.copy()
            new_item["term_obj"] = sobj
            processed_pool.append(new_item)

    print(f"Pool size after splitting coordinates: {len(processed_pool)} terms. / "
          f"Koordinat ayrımı sonrası havuz boyutu: {len(processed_pool)} terim.")
    if deleted_terms:
        print(f"Removed {len(deleted_terms)} invalid/noisy terms. / {len(deleted_terms)} geçersiz/gürültülü terim temizlendi.")

    # 5. Group by lang pair / Dil çiftine göre grupla
    grouped_terms = defaultdict(list)
    for item in processed_pool:
        key = (item["src_lang"], item["tgt_lang"])
        grouped_terms[key].append(item["term_obj"])

    run_logger = RunLogger(task="clean", source="bulk" if args.all else (os.path.basename(json_files[0]) if json_files else "single"))
    report = {
        "version": 2,
        "task": "clean",
        "started_at": now_iso(),
        "updated_at": now_iso(),
        "groups": [],
        "deleted_terms": deleted_terms
    }

    # 6. Merge & deduplicate / Birleştir ve tekilleştir
    for (src_lang, tgt_lang), incoming_terms in grouped_terms.items():
        cons_filename = f"sozluk_{src_lang}_{tgt_lang}.json"
        cons_path = os.path.join(merged_dir, cons_filename)
        
        existing_terms = []
        if os.path.exists(cons_path):
            print(f"Loading existing merged file / Mevcut birleştirilmiş dosya yükleniyor: {cons_filename}...")
            try:
                with open(cons_path, "rb") as f:
                    loaded = orjson.loads(f.read())
                if isinstance(loaded, list):
                    existing_terms = loaded
            except Exception as e:
                print(f"  Error reading merged file / Okuma hatası: {e}")

        merged_records = {}
        duplicates_removed = 0
        
        for item in existing_terms:
            if not isinstance(item, dict) or not item.get("term"):
                continue
            normalized = normalize_term(item, src_lang=src_lang)
            key = make_record_key(normalized)
            if key not in merged_records:
                merged_records[key] = normalized
            else:
                merge_term_objects(merged_records[key], normalized)
                duplicates_removed += 1

        for item in incoming_terms:
            key = make_record_key(item)
            if key not in merged_records:
                merged_records[key] = item
            else:
                merge_term_objects(merged_records[key], item)
                duplicates_removed += 1

        consolidated_list = list(merged_records.values())
        print(f"Consolidated / Birleştirildi {src_lang}->{tgt_lang}: {len(consolidated_list)} terms/terim "
              f"({duplicates_removed} duplicates merged/removed / mükerrer birleştirildi/elendi).")

        group_stats = {
            "language_pair": f"{src_lang}_{tgt_lang}",
            "records_before": len(existing_terms) + len(incoming_terms),
            "records_after": len(consolidated_list),
            "duplicates_removed": duplicates_removed,
            "status": "ok"
        }
        report["groups"].append(group_stats)

        if not args.dry_run:
            save_json_atomic(cons_path, consolidated_list)
            print(f"Saved / Kaydedildi: data/merged/{cons_filename}")

    report["updated_at"] = now_iso()
    summary_path = os.path.join("data", "logs", f"{run_logger.run_id}_clean_report.json")
    save_json_atomic(summary_path, report)
    run_logger.close(total_files=len(non_consolidated_sources), report_path=summary_path)
    
    print(f"Log file / Log dosyası: {run_logger.log_path}")
    print(f"Report file / Rapor dosyası: {summary_path}")
    print("\n\033[1;32mCleaning and consolidation process completed successfully! / "
          "Temizlik ve birleştirme işlemi başarıyla tamamlandı!\033[0m")


if __name__ == "__main__":
    main()
