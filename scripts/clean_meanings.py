#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
clean_meanings.py – Meaning Splitter & Deduplicator (LLM-free)
==============================================================
Cleans definition.meaning fields in merged dictionary JSON files:
1) Splits meanings separated by commas/semicolons.
2) Removes duplicate definitions under the same term.

Merged sözlük JSON dosyalarındaki definition.meaning alanlarını temizler:
1) Virgül / noktalı virgül ile ayrılmış anlamları ayırır.
2) Aynı term altında mükerrer olan anlamları temizler.
"""

import os
import sys
import orjson
import argparse
import re
from copy import deepcopy

# Prepend parent directory to sys.path to resolve imports from handlers/
# Üst dizini sys.path'e ekleyerek handlers/ modüllerini yüklemeyi sağlar
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Windows console UTF-8 / Windows konsolu için UTF-8 desteği
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


MERGED_DIR = os.path.join("data", "merged")

# Separators: split on commas/semicolons / Ayırıcılar: virgül ve noktalı virgüle göre ayırır
SPLIT_RE = re.compile(r'\s*[,;]\s*')
MAX_SINGLE_MEANING_WORDS = 5


def normalize_meaning(m: str) -> str:
    """Lowercase, strip, collapse whitespace — used as dedup key / Küçük harfe dönüştürür, boşlukları temizler."""
    return re.sub(r'\s+', ' ', m.strip()).lower()


def score_definition(d: dict) -> int:
    """Score a definition for quality (higher = better) / Tanımın kalitesini puanlar (yüksek = iyi)."""
    score = 0
    ex_src = (d.get("example_source") or "").strip()
    ex_tgt = (d.get("example_target") or "").strip()
    if ex_src and ex_src.lower() not in ("n/a", "-", "—", ""):
        score += len(ex_src)
    if ex_tgt and ex_tgt.lower() not in ("n/a", "-", "—", ""):
        score += len(ex_tgt) // 2
    domain = (d.get("domain") or "GENERAL").strip().upper()
    if domain != "GENERAL":
        score += 5
    return score


def split_and_dedup_definitions(definitions: list) -> tuple[list, dict]:
    """Splits comma/semicolon meanings and deduplicates duplicates / Anlamları ayırır ve mükerrerleri eler."""
    split_count = 0
    expanded = []

    # ── Phase 1: Split comma/semicolon-separated meanings / Aşama 1: Virgüllü anlamları ayır ──
    for d in definitions:
        if not isinstance(d, dict):
            expanded.append(d)
            continue

        meaning = (d.get("meaning") or "").strip()
        if not meaning:
            expanded.append(d)
            continue

        parts = SPLIT_RE.split(meaning)
        parts = [p.strip() for p in parts if p.strip()]

        should_split = (
            len(parts) > 1
            and all(len(p.split()) <= MAX_SINGLE_MEANING_WORDS for p in parts)
        )

        if should_split:
            split_count += 1
            for part in parts:
                new_def = deepcopy(d)
                new_def["meaning"] = part
                expanded.append(new_def)
        else:
            expanded.append(d)

    # ── Phase 2: Deduplicate by normalized meaning / Aşama 2: Anlamları tekilleştir ──
    seen: dict[str, int] = {}
    result = []
    dedup_count = 0

    for d in expanded:
        if not isinstance(d, dict):
            result.append(d)
            continue

        meaning = (d.get("meaning") or "").strip()
        if not meaning:
            result.append(d)
            continue

        key = normalize_meaning(meaning)

        if key in seen:
            existing_idx = seen[key]
            existing_score = score_definition(result[existing_idx])
            new_score = score_definition(d)
            if new_score > existing_score:
                result[existing_idx] = d
            dedup_count += 1
        else:
            seen[key] = len(result)
            result.append(d)

    stats = {"split_count": split_count, "dedup_count": dedup_count}
    return result, stats


def process_file(filepath: str, dry_run: bool = False) -> dict:
    """Process a single merged JSON file / Tek bir birleştirilmiş JSON dosyasını işler."""
    filename = os.path.basename(filepath)
    print(f"\n{'='*60}")
    print(f"  Processing / İşleniyor: {filename}")
    print(f"{'='*60}")

    with open(filepath, "rb") as f:
        data = orjson.loads(f.read())

    if not isinstance(data, list):
        print(f"  ⚠ Not a JSON list, skipping. / Liste değil, atlanıyor.")
        return {"file": filename, "status": "skipped"}

    total_terms = len(data)
    total_defs_before = 0
    total_defs_after = 0
    total_splits = 0
    total_dedups = 0
    terms_changed = 0

    for term_obj in data:
        if not isinstance(term_obj, dict):
            continue

        defs = term_obj.get("definitions", [])
        if not isinstance(defs, list) or not defs:
            continue

        total_defs_before += len(defs)

        cleaned, stats = split_and_dedup_definitions(defs)

        total_defs_after += len(cleaned)
        total_splits += stats["split_count"]
        total_dedups += stats["dedup_count"]

        if stats["split_count"] > 0 or stats["dedup_count"] > 0:
            terms_changed += 1
            term_name = term_obj.get("term", "?")
            article = term_obj.get("article", "")
            art_prefix = f"({article}) " if article else ""
            print(f"  {art_prefix}{term_name}: "
                  f"{len(defs)} defs → {len(cleaned)} defs "
                  f"(split: {stats['split_count']}, dedup: {stats['dedup_count']})")

        term_obj["definitions"] = cleaned

    removed = total_defs_before - total_defs_after
    print(f"\n  ── Summary for / Özet: {filename} ──")
    print(f"     Terms total / Toplam terim        : {total_terms}")
    print(f"     Terms changed / Değişen terim     : {terms_changed}")
    print(f"     Definitions before / Önceki tanım : {total_defs_before}")
    print(f"     Definitions after / Sonraki tanım  : {total_defs_after}")
    print(f"     Meanings split / Bölünen anlam    : {total_splits}")
    print(f"     Duplicates removed / Elenen kopya : {total_dedups}")
    print(f"     Net removed / Net silinen         : {removed}")

    if not dry_run and terms_changed > 0:
        tmp_path = filepath + ".tmp"
        with open(tmp_path, "wb") as f:
            f.write(orjson.dumps(data, option=orjson.OPT_INDENT_2))
        os.replace(tmp_path, filepath)
        print(f"     ✅ Saved / Kaydedildi: {filepath}")
    elif dry_run:
        print(f"     🔍 Dry-run — no file changes. / Dry-run modu — dosya kaydedilmedi.")
    else:
        print(f"     ✅ No changes needed. / Değişiklik gerekmedi.")

    return {
        "file": filename,
        "status": "ok",
        "terms_total": total_terms,
        "terms_changed": terms_changed,
        "defs_before": total_defs_before,
        "defs_after": total_defs_after,
        "splits": total_splits,
        "dedups": total_dedups,
        "net_removed": removed,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Meaning Splitter & Deduplicator for merged sözlük JSON files (LLM-free)"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", help="Specific merged JSON file name (in data/merged/)")
    group.add_argument("--all", action="store_true", help="Process all files in data/merged/")
    group.add_argument("--dry-run", action="store_true", help="Only report, do not write files")
    args = parser.parse_args()

    # Collect files / Dosyaları topla
    files = []
    if args.file:
        path = args.file
        if not os.path.exists(path):
            path = os.path.join(MERGED_DIR, args.file)
        if not os.path.exists(path):
            print(f"Error: File not found / Hata: Dosya bulunamadı: {args.file}")
            sys.exit(1)
        files.append(path)
    else:
        if not os.path.isdir(MERGED_DIR):
            print(f"Error: Directory not found / Hata: Dizin bulunamadı: {MERGED_DIR}")
            sys.exit(1)
        for f in sorted(os.listdir(MERGED_DIR)):
            if f.endswith(".json") and f.startswith("sozluk_"):
                files.append(os.path.join(MERGED_DIR, f))

    if not files:
        print("No files to process. / İşlenecek dosya bulunamadı.")
        sys.exit(0)

    print(f"\n🔧 clean_meanings.py — Meaning Splitter & Deduplicator")
    print(f"   Files to process / İşlenecek dosya sayısı: {len(files)}")
    if args.dry_run:
        print(f"   Mode / Mod: DRY-RUN (no files will be modified / dosyalar değiştirilmez)")
    print()

    all_stats = []
    for fp in files:
        stats = process_file(fp, dry_run=args.dry_run)
        all_stats.append(stats)

    # ── Grand total / Genel Toplam ──
    print(f"\n{'='*60}")
    print(f"  GRAND TOTAL / GENEL TOPLAM")
    print(f"{'='*60}")
    g_before = sum(s.get("defs_before", 0) for s in all_stats)
    g_after = sum(s.get("defs_after", 0) for s in all_stats)
    g_splits = sum(s.get("splits", 0) for s in all_stats)
    g_dedups = sum(s.get("dedups", 0) for s in all_stats)
    g_removed = sum(s.get("net_removed", 0) for s in all_stats)
    g_changed = sum(s.get("terms_changed", 0) for s in all_stats)

    print(f"  Files processed / İşlenen dosya   : {len(all_stats)}")
    print(f"  Terms changed / Değişen terim     : {g_changed}")
    print(f"  Definitions before / Önceki tanım : {g_before}")
    print(f"  Definitions after / Sonraki tanım  : {g_after}")
    print(f"  Meanings split / Bölünen anlam    : {g_splits}")
    print(f"  Duplicates removed / Elenen kopya : {g_dedups}")
    print(f"  Net removed / Net silinen         : {g_removed}")
    print(f"\n{'='*60}")
    print(f"  ✅ Done! / Tamamlandı!")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
