#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
merge_translated.py – Glossa Master Unified Dictionary Merger
=============================================================
Groups all enriched/translated JSON files in data/translated/ by language pairs
and merges each group into a single file 'data/final/[SRC]_[TGT].json'.

data/translated/ klasöründeki tüm zenginleştirilmiş/çevrilmiş JSON dosyalarını
dil çiftlerine göre gruplar ve her grubu 'data/final/[SRC]_[TGT].json' dosyasında birleştirir.
"""

import os
import sys
import orjson
import re

# Prepend parent directory to sys.path to resolve imports from handlers/
# Üst dizini sys.path'e ekleyerek handlers/ modüllerini yüklemeyi sağlar
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Windows console UTF-8 support / Windows konsol UTF-8 desteği
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

TRANSLATED_DIR = os.path.join("data", "translated")
FINAL_DIR = os.path.join("data", "final")


def extract_pair_from_filename(file_name):
    """Extracts source and target language codes from file name / Dosya adından dil çiftini çıkarır."""
    matches = re.findall(r'sozluk_([a-zA-Z]{2,})_([a-zA-Z]{2,})', file_name)
    if matches:
        return matches[-1][0].upper(), matches[-1][1].upper()
    return None, None


def get_normalized_gender(item):
    """Normalizes gender or infers it from article / Cinsiyeti normalize eder veya artikelden çıkarır."""
    gender = (item.get("gender") or "").strip().upper()
    if gender in ("NULL", "NONE", "N/A", "UNDEFINED", ""):
        art = (item.get("article") or item.get("artikel") or "").strip().lower()
        if art in ("der", "le", "el", "il", "lo"):
            return "MASCULINE"
        elif art in ("die", "la"):
            return "FEMININE"
        elif art in ("das",):
            return "NEUTER"
        return ""
    return gender


def normalize_meaning(meaning):
    """Standardizes meaning text for deduplication / Eşleşme kontrolü için anlamı normalize eder."""
    if not meaning:
        return ""
    return re.sub(r'\s+', ' ', meaning.strip()).lower()


def merge_fields(target, source):
    """Merges grammatical and metadata fields / Dilbilgisel ve metaveri alanlarını birleştirir."""
    fields_to_merge = [
        "gender", "article", "plural", "genitive", "comparative", "superlative",
        "pastTense", "pastParticiple", "presentParticiple", "auxiliary",
        "conjugation", "isRegular", "isSeparable", "isReflexive", "verbPrefix",
        "pronunciation", "syllables", "level", "etymology", "declension",
        "caseMatrix", "usageNotes"
    ]
    for field in fields_to_merge:
        if not target.get(field) and source.get(field):
            target[field] = source[field]

    # Combine Synonyms/Antonyms lists / Eş/Zıt anlamlılar listesini birleştirir
    for field in ["synonyms", "antonyms"]:
        val_tgt = target.get(field)
        val_src = source.get(field)
        if val_src:
            if not val_tgt:
                target[field] = val_src
            else:
                list_tgt = [s.strip() for s in str(val_tgt).split(",") if s.strip()]
                list_src = [s.strip() for s in str(val_src).split(",") if s.strip()]
                combined = sorted(list(set(list_tgt + list_src)))
                target[field] = ", ".join(combined)

    # semanticRelations (Combine language-specific mapping / Dil bazlı ilişkileri birleştirir)
    rel_tgt = target.get("semanticRelations")
    rel_src = source.get("semanticRelations")
    if rel_src and isinstance(rel_src, dict):
        if not rel_tgt or not isinstance(rel_tgt, dict):
            target["semanticRelations"] = rel_src
        else:
            for lang, data in rel_src.items():
                if lang not in rel_tgt:
                    rel_tgt[lang] = data
                else:
                    for item_type in ["synonyms", "antonyms"]:
                        sub_tgt = rel_tgt[lang].get(item_type) or []
                        sub_src = rel_src[lang].get(item_type) or []
                        if isinstance(sub_tgt, list) and isinstance(sub_src, list):
                            rel_tgt[lang][item_type] = sorted(list(set(sub_tgt + sub_src)))

    # Combine parentTerms / Üst terimleri birleştirir
    p_tgt = target.get("parentTerms") or []
    p_src = source.get("parentTerms") or []
    if p_src:
        if not isinstance(p_tgt, list):
            p_tgt = [p_tgt] if p_tgt else []
        if not isinstance(p_src, list):
            p_src = [p_src]
        combined_parents = sorted(list(set(p_tgt + p_src)))
        target["parentTerms"] = combined_parents


def main():
    if not os.path.exists(TRANSLATED_DIR):
        print(f"ERROR: Directory '{TRANSLATED_DIR}' not found! / HATA: '{TRANSLATED_DIR}' klasörü bulunamadı!")
        sys.exit(1)

    json_files = [
        f for f in os.listdir(TRANSLATED_DIR)
        if f.endswith(".json") and not f.endswith(".tmp") and "_progress" not in f and "sozluk_" in f
    ]

    if not json_files:
        print("No files to merge found. / Birleştirilecek dosya bulunamadı.")
        sys.exit(0)

    # Group files by language pair / Dosyaları dil çiftine göre gruplayalım
    groups = {}
    for f in json_files:
        src, tgt = extract_pair_from_filename(f)
        if src and tgt:
            pair = (src, tgt)
            groups.setdefault(pair, []).append(f)

    print(f"\033[1;36m[Merger] Split {len(json_files)} files into {len(groups)} language groups: / "
          f"[Merger] {len(json_files)} dosya, {len(groups)} farklı dil grubuna ayrıldı:\033[0m")
    for pair, files in groups.items():
        print(f"  * {pair[0]} -> {pair[1]} ({len(files)} files/dosya)")

    os.makedirs(FINAL_DIR, exist_ok=True)

    for pair, files in groups.items():
        src_lang, tgt_lang = pair
        output_file_name = f"{src_lang}_{tgt_lang}.json"
        output_file_path = os.path.join(FINAL_DIR, output_file_name)
        
        print(f"\n\033[1;33m--- Merging: {src_lang} -> {tgt_lang} ➔ {output_file_name} ---\033[0m")
        
        merged_db = {}
        total_input_records = 0

        for file_name in files:
            file_path = os.path.join(TRANSLATED_DIR, file_name)
            print(f"  Reading / Okunuyor: {file_name}")

            try:
                with open(file_path, "rb") as f:
                    items = orjson.loads(f.read())
                if not isinstance(items, list):
                    print(f"    [WARNING] {file_name} does not contain a list. Skipped. / [UYARI] {file_name} geçerli bir liste içermiyor. Atlandı.")
                    continue
            except Exception as e:
                print(f"    [ERROR] Error reading {file_name}: {e} / [HATA] {file_name} okunurken hata oluştu: {e}")
                continue

            for item in items:
                term = (item.get("term") or "").strip()
                if not term:
                    continue

                total_input_records += 1
                gender = get_normalized_gender(item)
                pos = (item.get("partOfSpeech") or "").strip().upper()

                # Match key / Eşleşme anahtarı
                db_key = (term.lower(), gender, pos)

                if db_key not in merged_db:
                    merged_item = {
                        "term": term,
                        "gender": item.get("gender") or None,
                        "article": item.get("article") or None,
                        "partOfSpeech": item.get("partOfSpeech") or None,
                        "definitions": []
                    }
                    merge_fields(merged_item, item)
                    merged_db[db_key] = merged_item
                else:
                    merge_fields(merged_db[db_key], item)

                # Merge definitions / Tanımları birleştir
                definitions = item.get("definitions") or []
                for definition in definitions:
                    if isinstance(definition, str):
                        definition = {"meaning": definition}
                    if not isinstance(definition, dict):
                        continue

                    meaning = definition.get("meaning") or ""
                    if not meaning:
                        continue

                    new_def = {
                        "meaning": meaning,
                        "domain": (definition.get("domain") or "GENERAL").strip().upper(),
                        "example_source": definition.get("example_source") or None,
                        "example_target": definition.get("example_target") or None
                    }

                    # Deduplication check / Mükerrer kontrolü
                    is_duplicate = False
                    for existing_def in merged_db[db_key]["definitions"]:
                        if normalize_meaning(existing_def["meaning"]) == normalize_meaning(meaning):
                            if new_def["example_source"] and (not existing_def["example_source"] or len(new_def["example_source"]) > len(existing_def["example_source"])):
                                existing_def["example_source"] = new_def["example_source"]
                                existing_def["example_target"] = new_def["example_target"]
                            is_duplicate = True
                            break

                    if not is_duplicate:
                        merged_db[db_key]["definitions"].append(new_def)

        final_list = list(merged_db.values())

        try:
            with open(output_file_path, "wb") as f:
                f.write(orjson.dumps(final_list, option=orjson.OPT_INDENT_2))
            print(f"  \033[1;32m✓ Saved / Kaydedildi: {output_file_path} (Terms: {len(final_list)}, Defs: {sum(len(item['definitions']) for item in final_list)})\033[0m")
        except Exception as e:
            print(f"  [ERROR] Error saving output file: {e} / [HATA] Çıktı dosyası kaydedilirken hata oluştu: {e}")

    print("\n\033[1;32mAll language pairs merged successfully. / Tüm dil çiftleri başarıyla birleştirildi.\033[0m")


if __name__ == "__main__":
    main()
