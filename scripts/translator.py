#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Glossa Master Translator / Glossa Ana Çevirici
=============================================
Translates existing JSON dictionary files into target languages.
Tries all provider/models using a fallback chain.
Never skips a batch until it is successfully translated.

Mevcut JSON sözlük dosyalarını hedef dillere çevirir.
Tüm provider/modelleri fallback chain ile dener.
Batch başarılı olana kadar asla geçmez.

Usage / Kullanım:
    python scripts/translator.py                                # interactive / interaktif
    python scripts/translator.py --source "X.json" --targets EN,FR
    python scripts/translator.py --source "X.json" --targets EN --limit 100
"""

import os
import sys
import orjson
import argparse
import time
import hashlib
import asyncio
import shutil
from tqdm import tqdm

# Prepend parent directory to sys.path to resolve imports from handlers/
# Üst dizini sys.path'e ekleyerek handlers/ modüllerini yüklemeyi sağlar
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Windows console UTF-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from handlers.fallback_chain import FallbackChain, parse_json_response, load_config
from handlers.run_logger import RunLogger, now_iso, load_json_with_default, save_json_atomic, safe_replace


# ─── Redis Helpers / Redis Yardımcıları ───────────────────────────────────────────

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


def get_translate_cache_key(src_lang, tgt_lang, term, domain, meaning_hint, example_source=""):
    # Dynamic namespaced Redis cache key: sözlük:{source_lang}:{target_lang}:{sha256_hash}
    input_str = f"{(term or '').strip().lower()}||{(domain or 'GENERAL').strip().upper()}||{(meaning_hint or '').strip().lower()}||{(example_source or '').strip()}"
    sha = hashlib.sha256(input_str.encode('utf-8')).hexdigest()
    return f"sözlük:{src_lang.upper()}:{tgt_lang.upper()}:{sha}"


# ─── Helpers / Yardımcılar ─────────────────────────────────────────────────────────

def extract_pair_from_filename(file_path):
    base_name = os.path.basename(file_path).replace(".json", "")
    parts = base_name.split("_")
    for index, part in enumerate(parts):
        if part.lower() == "sozluk" and index + 2 < len(parts):
            src = parts[index + 1].upper()
            return src, parts[index + 2].upper()
    return "DE", "TR"


def is_translation_done_for_pair(source_path, src_lang, target_lang):
    """Check translator progress file and compare against total definitions in source.
    Returns True if translator progress indicates the pair is already completed.
    """
    try:
        pdf_prefix = os.path.basename(source_path).split("_sozluk_")[0]
        progress_path = os.path.join("data", "progress", "translator", f"{pdf_prefix}_sozluk_{src_lang}_{target_lang}_translate_progress.json")
        if not os.path.exists(progress_path):
            return False

        with open(progress_path, "rb") as fh:
            pdata = orjson.loads(fh.read())

        status = pdata.get("status", {}) or {}
        state = (status.get("state") or "").upper()
        if state == "DONE" or status.get("done") is True:
            return True
    except Exception:
        return False
    return False


def get_jsonl_cache_path(source_path):
    return f"{source_path}.jsonl"


def ensure_jsonl_cache(source_path):
    jsonl_path = get_jsonl_cache_path(source_path)
    if os.path.exists(jsonl_path):
        return jsonl_path

    try:
        with open(source_path, "rb") as f:
            source_data = orjson.loads(f.read())
        if isinstance(source_data, list):
            with open(jsonl_path, "wb") as out:
                for item in source_data:
                    out.write(orjson.dumps(item))
                    out.write(b"\n")
            return jsonl_path
    except Exception:
        return None

    return None


def iter_source_terms(source_path):
    jsonl_path = get_jsonl_cache_path(source_path)
    if os.path.exists(jsonl_path):
        with open(jsonl_path, "rb") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    yield orjson.loads(line)
                except Exception:
                    continue
        return

    # Fallback to in-memory load if JSONL does not exist
    with open(source_path, "rb") as f:
        source_data = orjson.loads(f.read())
    if isinstance(source_data, list):
        for item in source_data:
            yield item


def get_translation_prompt(batch_items, src_lang, tgt_lang, src_name, tgt_name):
    terms_text_lines = []
    for index, item in enumerate(batch_items):
        line = f'{index + 1}. Term: "{item["term"]}" (meaning hint: {item["meaning_hint"]}, domain: {item["domain"]})'
        if item.get("example_source"):
            line += f'\n   Example: "{item["example_source"]}"'
        terms_text_lines.append(line)

    return f"""Translate these {src_name} ({src_lang}) dictionary meanings into {tgt_name} ({tgt_lang}).
For each term, write a short accurate meaning. If an example sentence exists, translate it too.

Terms:
{chr(10).join(terms_text_lines)}

Return ONLY this JSON array, preserving order:
[
  {{
    "term": "original term",
    "meaning": "{tgt_name} meaning",
    "example_target": "translated example or null"
  }}
]

Rules:
- Keep meanings concise (1-8 words where possible).
- Do not add explanations outside JSON.
"""


def load_existing_output(output_json):
    existing_data = []
    existing_lookup = {}
    if not os.path.exists(output_json):
        return existing_data, existing_lookup

    try:
        with open(output_json, "rb") as handle:
            loaded = orjson.loads(handle.read())
        if isinstance(loaded, list):
            existing_data = loaded
    except Exception:
        existing_data = []

    for item in existing_data:
        term_key = (item.get("term") or "").strip().lower()
        pos_key = (item.get("partOfSpeech") or "").strip().upper()
        if "artikel" in item:
            item["article"] = item.pop("artikel")
        art_key = (item.get("article") or "").strip().lower()
        existing_lookup[(term_key, pos_key, art_key)] = item
    return existing_data, existing_lookup


def make_def_key(term, domain, meaning_hint):
    return f"{(term or '').strip().lower()}||{(domain or 'GENERAL').strip().upper()}||{(meaning_hint or '').strip().lower()}"


def _default_translation_progress(source_path, source_lang, target_lang):
    return {
        "version": 1,
        "source_file": source_path,
        "source_lang": source_lang,
        "target_lang": target_lang,
        "total_definitions": 0,
        "translated_items": [],
        "skipped_items": [],
        "failed_batches": [],
        "stats": {
            "translated": 0,
            "skipped_existing_output": 0,
            "skipped_progress": 0,
            "batch_failures": 0,
            "json_parse_errors": 0,
        },
        "status": {
            "state": "IN_PROGRESS",
            "done": False,
            "total_definitions": 0,
            "completed_count": 0,
        },
        "started_at": now_iso(),
        "updated_at": now_iso(),
    }


def load_translation_progress(progress_path, source_path, source_lang, target_lang):
    defaults = _default_translation_progress(source_path, source_lang, target_lang)
    state = load_json_with_default(progress_path, defaults)
    if not isinstance(state.get("translated_items"), list):
        state["translated_items"] = []
    if not isinstance(state.get("skipped_items"), list):
        state["skipped_items"] = []
    if not isinstance(state.get("failed_batches"), list):
        state["failed_batches"] = []
    if not isinstance(state.get("stats"), dict):
        state["stats"] = defaults["stats"]
    if not isinstance(state.get("status"), dict):
        state["status"] = defaults["status"]
    if not isinstance(state.get("total_definitions"), int):
        state["total_definitions"] = 0
    state["source_file"] = source_path
    state["source_lang"] = source_lang
    state["target_lang"] = target_lang
    state["updated_at"] = now_iso()
    return state


def save_translation_progress(progress_path, state):
    state["translated_items"] = sorted(set(state.get("translated_items", [])))
    stats = state.get("stats", {}) or {}
    total_defs = int(state.get("total_definitions") or 0)
    completed_count = len(state["translated_items"]) + int(stats.get("skipped_existing_output", 0)) + int(stats.get("skipped_progress", 0))
    status = state.get("status", {}) or {}
    status["total_definitions"] = total_defs
    status["completed_count"] = completed_count
    if total_defs > 0 and completed_count >= total_defs:
        status["state"] = "DONE"
        status["done"] = True
    else:
        status["state"] = "IN_PROGRESS"
        status["done"] = False
    state["status"] = status
    state["updated_at"] = now_iso()
    save_json_atomic(progress_path, state)


_seen_sets = {}


def append_unique_record(items, key, reason, detail=""):
    items_id = id(items)
    if items_id not in _seen_sets:
        _seen_sets.clear()
        _seen_sets[items_id] = { (row.get("key"), row.get("reason")) for row in items if isinstance(row, dict) }
    seen = _seen_sets[items_id]
    if (key, reason) in seen:
        return
    seen.add((key, reason))
    items.append({"key": key, "reason": reason, "detail": detail, "timestamp": now_iso()})


def select_targets_interactive(config_languages, src_lang, primary_tgt):
    options = []
    for lang_code in sorted(config_languages.keys()):
        if lang_code in (src_lang, primary_tgt):
            continue
        options.append(f"{lang_code} - {config_languages.get(lang_code, lang_code)}")

    if not options:
        return []

    try:
        from handlers.selector import select_item_interactive
    except ImportError:
        print("Error: handlers/selector.py not found. Please provide --targets. / Hata: handlers/selector.py bulunamadı. Lütfen --targets argümanını sağlayın.")
        return []

    selected = []
    menu_options = ["ALL"] + options
    choice = select_item_interactive(menu_options, "Select a target language or ALL (choose DONE after selecting multiple) / Bir hedef dil veya ALL seçin (Birden fazla seçtikten sonra DONE seçin)")
    if not choice:
        return []
    if choice == "ALL":
        return [code.split(" - ")[0].strip().upper() for code in options]

    # Add the first selection
    code = choice.split(" - ")[0].strip().upper()
    selected.append(code)

    while True:
        menu = ["DONE"] + [opt for opt in options if opt.split(" - ")[0].strip().upper() not in selected]
        if len(menu) == 1: # Only DONE left
            break
        choice = select_item_interactive(menu, "Select a target language (DONE to finish) / Hedef dil seçin (Bitirmek için DONE seçin)")
        if not choice or choice == "DONE":
            break
        code = choice.split(" - ")[0].strip().upper()
        if code not in selected:
            selected.append(code)
    return selected


# ─── Async Main Loop / Asenkron Ana Döngü ─────────────────────────────────────────

async def main_async():
    parser = argparse.ArgumentParser(description="Glossa Master Translator — Fallback Chain")
    parser.add_argument("--source", help="Source JSON dictionary file path or name")
    parser.add_argument("--targets", help="Target language codes (comma separated, e.g. EN,FR)")
    parser.add_argument("--limit", type=int, default=0, help="Max definitions to translate")
    parser.add_argument("--batch-size", type=int, help="Batch size for single query")
    parser.add_argument("--provider", help="Filter by a specific provider (openrouter, cerebras, groq, google, github, sambanova, mistral)")
    parser.add_argument("--model", help="Filter by a specific model")
    parser.add_argument("--semaphore", type=int, default=10, help="Max concurrent requests (Semaphore)")
    args = parser.parse_args()

    # ── Interactive Selector Imports ──
    from handlers.selector import select_item_interactive, select_boolean_interactive, select_file_interactive

    config = load_config()

    # 1. Source JSON Selection
    merged_dir = os.path.join("data", "merged")
    translated_dir = os.path.join("data", "translated")
    os.makedirs(translated_dir, exist_ok=True)
    source_path = args.source

    if not source_path:
        all_jsons = []
        if os.path.exists(merged_dir):
            for f in os.listdir(merged_dir):
                if f.endswith(".json") and not f.endswith(".tmp") and "sozluk_" in f and "_progress" not in f:
                    all_jsons.append(os.path.join(merged_dir, f))
        if not all_jsons:
            print("ERROR: Source JSON file to translate not found / HATA: Çevrilecek kaynak JSON dosyası bulunamadı.")
            sys.exit(1)

        source_path = select_file_interactive(all_jsons, "Select the source JSON dictionary file to translate / Çevrilecek kaynak JSON sözlük dosyasını seçin")
        if not source_path:
            print("Selection cancelled / Seçim iptal edildi.")
            sys.exit(0)
        args.source = source_path
    else:
        if not os.path.exists(source_path):
            source_path = os.path.join(merged_dir, args.source)
            if not os.path.exists(source_path):
                print(f"Error: Source file '{args.source}' not found / Hata: Kaynak dosya '{args.source}' bulunamadı.")
                sys.exit(1)

    # 2. Overwrite / Skip Progress
    args.overwrite_progress = False
    if "--source" not in sys.argv and "--targets" not in sys.argv:
        skip_translated = select_boolean_interactive(
            "Should already translated definitions be skipped? / Daha önce çevrilmiş tanımlar atlansın mı (Skip already translated definitions)?",
            default=True
        )
        if not skip_translated:
            args.overwrite_progress = True

    # 3. Target Language Selection
    targets_str = args.targets
    if not targets_str:
        src_lang, primary_tgt = extract_pair_from_filename(source_path)
        target_langs = select_targets_interactive(config.get("languages", {}), src_lang, primary_tgt)
        if not target_langs:
            print("Selection cancelled or no target selected / Seçim iptal edildi ya da hedef seçilmedi.")
            sys.exit(0)
    else:
        target_langs = [t.strip().upper() for t in targets_str.split(",") if t.strip()]

    # 4. Limit
    if "--limit" not in sys.argv:
        limit_confirm = select_boolean_interactive("Do you want to set a maximum limit for translation? / Çeviri işlemi için maksimum sınır (Limit) belirlemek ister misiniz?", default=False)
        if limit_confirm:
            try:
                val = input("Enter maximum definition limit (e.g. 100) / Maksimum tanım sınırı girin (örn. 100): ").strip()
                args.limit = int(val) if val else 0
            except ValueError:
                args.limit = 0
                print("Invalid value. Limit removed / Geçersiz değer. Sınır kaldırıldı.")

    # 5. Batch Size
    if "--batch-size" not in sys.argv:
        batch_options = [
            "30 (Recommended) / 30 (Önerilen)",
            "10",
            "50",
            "Enter custom value... / Özel Değer Girin..."
        ]
        selected_batch = select_item_interactive(batch_options, "What should be the batch size sent at once? / Tek seferde gönderilecek paket boyutu (Batch Size) ne olsun?")
        if not selected_batch or selected_batch.startswith("30 (Recommended)"):
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
                print("Invalid value. Batch size set to 30 / Geçersiz değer. Paket boyutu 30 olarak belirlendi.")

    batch_size = args.batch_size if args.batch_size else 30

    if "--semaphore" not in sys.argv:
        try:
            sem_input = input("Enter max concurrent requests (Semaphore) (Default 10) / Maksimum eş zamanlı istek sayısını (Semaphore) girin (Varsayılan 10): ").strip()
            args.semaphore = int(sem_input) if sem_input else 10
        except ValueError:
            args.semaphore = 10

    # ── Redis Connection & Interactive Selector ──
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
                print("\033[1;32m[Redis] Active. Cache will be used / Aktif. Önbellek kullanılacak.\033[0m")
            else:
                print("\033[1;31m[Redis] Connection failed. Continuing file-based... / Bağlantı kurulamadı. Dosya tabanlı devam ediliyor...\033[0m")
                redis_enabled = False

    provider_filter = args.provider
    if not provider_filter:
        api_providers = config.get("api_providers", {})
        enabled_providers = [k for k, v in api_providers.items() if v.get("enabled", True)]
        if enabled_providers:
            options = ["All Providers (Full Fallback Chain) / Tüm Sağlayıcılar (Tam Yedek Zinciri)"] + enabled_providers
            try:
                selected_opt = select_item_interactive(options, "Select a Provider to translate terms with / Terimleri çevirmek için bir Sağlayıcı (Provider) seçin")
                if not selected_opt:
                    print("Selection cancelled / Seçim iptal edildi.")
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
                    print("Selection cancelled / Seçim iptal edildi.")
                    sys.exit(0)
                if not selected_opt.startswith("All Models"):
                    model_filter = selected_opt
            except ImportError:
                pass
    active_logger = None

    def fallback_event_logger(level, event, **data):
        nonlocal active_logger
        if not active_logger:
            return
        if level == "ERROR":
            active_logger.error(event, **data)
        elif level == "WARN":
            active_logger.warn(event, **data)
        else:
            active_logger.info(event, **data)

    chain = FallbackChain(
        config,
        provider_filter=provider_filter,
        model_filter=model_filter,
        event_callback=fallback_event_logger,
    )

    if chain.get_total_model_count() == 0:
        print("\033[1;31m[Translator] ERROR: No available provider/model found! / HATA: Hiç kullanılabilir provider/model bulunamadı!\033[0m")
        sys.exit(1)

    prov_order = ', '.join(dict.fromkeys(t.provider_key for t in chain.trackers))
    print(f"\033[1;36m[Translator] Fallback chain ready: {chain.get_total_model_count()} models "
          f"({chain.get_available_count()} active) / Fallback zinciri hazır: {chain.get_total_model_count()} model "
          f"({chain.get_available_count()} aktif)\033[0m")
    print(f"\033[36m[Translator] Provider order: {prov_order} / Sağlayıcı sırası: {prov_order}\033[0m")

    src_lang, primary_tgt = extract_pair_from_filename(source_path)
    total_definitions = None

    # Copy source file to data/translated/ so enricher can find everything in one place
    source_copy_dest = os.path.join(translated_dir, os.path.basename(source_path))
    if not os.path.exists(source_copy_dest):
        shutil.copy2(source_path, source_copy_dest)
        print(f"  Copied source file to: data/translated/{os.path.basename(source_path)} / Kaynak dosya şuraya kopyalandı: data/translated/{os.path.basename(source_path)}")

    languages_cfg = config.get("languages", {})
    src_name = languages_cfg.get(src_lang, src_lang)

    # ── Language Loop / Dil Döngüsü ──
    for target_lang in target_langs:
        if target_lang == primary_tgt:
            continue

        tgt_name = languages_cfg.get(target_lang, target_lang)

        # Skip this language pair early if extractor progress shows it's already done
        try:
            if is_translation_done_for_pair(source_path, src_lang, target_lang):
                print(f"  [Skip] Translator progress indicates {src_lang} -> {target_lang} already completed. Skipping / Translator ilerlemesi {src_lang} -> {target_lang} çiftinin tamamlandığını gösteriyor. Atlanıyor.")
                continue
        except Exception:
            pass
        pdf_prefix = os.path.basename(source_path).split("_sozluk_")[0]
        output_json = os.path.join(translated_dir, f"{pdf_prefix}_sozluk_{src_lang}_{target_lang}.json")
        progress_json = os.path.join("data", "progress", "translator", f"{pdf_prefix}_sozluk_{src_lang}_{target_lang}_translate_progress.json")

        active_logger = RunLogger(task="translator", source=f"{os.path.basename(source_path)}__{target_lang}")

        print(f"\n\033[1;33m{'='*60}\033[0m")
        print(f"\033[1;33m  Translating / Çevriliyor: {src_lang} ➔ {target_lang}\033[0m")
        print(f"\033[1;33m  Output / Çıktı: {os.path.basename(output_json)}\033[0m")
        print(f"\033[1;33m{'='*60}\033[0m")

        # Ensure JSONL cache for faster streaming reads on subsequent runs
        ensure_jsonl_cache(source_path)

        existing_data, existing_lookup = load_existing_output(output_json)
        progress_state = load_translation_progress(progress_json, source_path, src_lang, target_lang)
        if getattr(args, "overwrite_progress", False):
            progress_state["translated_items"] = []
            progress_state["stats"] = _default_translation_progress(source_path, src_lang, target_lang)["stats"]
            save_translation_progress(progress_json, progress_state)
            existing_data = []
            existing_lookup = {}
            if os.path.exists(output_json):
                try:
                    os.remove(output_json)
                    print(f"  [Progress] Old translation file deleted (Overwrite) / Eski çeviri dosyası silindi: {output_json}")
                except Exception as e:
                    print(f"  [Progress] Error deleting translation file / Çeviri dosyası silinirken hata: {e}")
        translated_keys = set(progress_state.get("translated_items", []))

        active_logger.info(
            "translation_started",
            source=source_path,
            target_lang=target_lang,
            output_json=output_json,
            progress_json=progress_json,
        )

        # Step 1: Flatten & Map (Bellek Haritalaması)
        memory_map = {}
        cache_keys = []
        
        total_definitions = 0
        for term_idx, term_obj in enumerate(iter_source_terms(source_path)):
            term = (term_obj.get("term") or "").strip()
            if not term:
                continue

            pos = (term_obj.get("partOfSpeech") or "").strip().upper()
            if "artikel" in term_obj:
                term_obj["article"] = term_obj.pop("artikel")
            art = (term_obj.get("article") or "").strip().lower()

            target_term_obj = existing_lookup.get((term.lower(), pos, art))
            existing_meanings = set()
            if target_term_obj:
                for definition in target_term_obj.get("definitions", []):
                    if isinstance(definition, str):
                        existing_meanings.add(definition.strip().lower())
                    elif isinstance(definition, dict):
                        existing_meanings.add((definition.get("meaning") or "").strip().lower())

            for def_idx, definition in enumerate(term_obj.get("definitions", [])):
                if isinstance(definition, str):
                    definition = {"meaning": definition}
                if not isinstance(definition, dict):
                    continue
                meaning_hint = (definition.get("meaning") or "").strip()
                domain = (definition.get("domain") or "GENERAL").strip().upper()
                example_source = (definition.get("example_source") or "").strip()
                if not meaning_hint:
                    continue
                total_definitions += 1

                def_key = make_def_key(term, domain, meaning_hint)
                if def_key in translated_keys:
                    progress_state["stats"]["skipped_progress"] = progress_state["stats"].get("skipped_progress", 0) + 1
                    append_unique_record(progress_state["skipped_items"], def_key, "already_in_progress")
                    continue

                if meaning_hint.lower() in existing_meanings:
                    translated_keys.add(def_key)
                    progress_state["translated_items"].append(def_key)
                    progress_state["stats"]["skipped_existing_output"] = progress_state["stats"].get("skipped_existing_output", 0) + 1
                    append_unique_record(progress_state["skipped_items"], def_key, "already_in_output")
                    continue

                cache_key = get_translate_cache_key(src_lang, target_lang, term, domain, meaning_hint, example_source)
                
                occurrence = {
                    "term": term,
                    "pos": pos,
                    "art": art,
                    "domain": domain,
                    "def_key": def_key,
                    "meaning_hint": meaning_hint,
                    "example_source": example_source,
                    "source_def": definition,
                    "term_obj": term_obj,
                }
                
                if cache_key not in memory_map:
                    memory_map[cache_key] = []
                    cache_keys.append(cache_key)
                memory_map[cache_key].append(occurrence)

        progress_state["total_definitions"] = total_definitions

        # Step 2: Batch lookup in Redis with MGET
        cache_hits_count = 0
        cache_miss_keys = []
        
        if redis_client and cache_keys:
            try:
                print(f"  [Redis] Querying cache (MGET) for {len(cache_keys)} definitions... / {len(cache_keys)} tanım için toplu cache (MGET) sorgulanıyor...")
                redis_results = await redis_client.mget(cache_keys)
                for key, val in zip(cache_keys, redis_results):
                     if val is not None:
                        try:
                            cached_data = orjson.loads(val)
                            translated_meaning = cached_data.get("meaning")
                            translated_example = cached_data.get("example_target", "")
                            
                            if translated_meaning:
                                for occurrence in memory_map[key]:
                                    term = occurrence["term"]
                                    pos = occurrence["pos"]
                                    art = occurrence["art"]
                                    domain = occurrence["domain"]
                                    def_key = occurrence["def_key"]
                                    term_obj = occurrence["term_obj"]
                                    definition = occurrence["source_def"]
                                    
                                    term_key = term.lower()
                                    target_term_obj = existing_lookup.get((term_key, pos, art))
                                    if not target_term_obj:
                                        target_term_obj = {
                                            "term": term,
                                            "definitions": [],
                                        }
                                        if pos:
                                            target_term_obj["partOfSpeech"] = pos
                                        if art:
                                            target_term_obj["article"] = art
                                        for k, value in term_obj.items():
                                            if k not in ("definitions", "kaynak_dosya", "artikel", "article", "domain", "partOfSpeech"):
                                                target_term_obj[k] = value
                                        existing_data.append(target_term_obj)
                                        existing_lookup[(term_key, pos, art)] = target_term_obj
                                    
                                    existing_meanings = set()
                                    for d in target_term_obj.setdefault("definitions", []):
                                        if isinstance(d, str):
                                            existing_meanings.add(d.strip().lower())
                                        elif isinstance(d, dict):
                                            existing_meanings.add((d.get("meaning") or "").strip().lower())
                                    
                                    if translated_meaning.lower() in existing_meanings:
                                        translated_keys.add(def_key)
                                        if def_key not in progress_state["translated_items"]:
                                            progress_state["translated_items"].append(def_key)
                                        continue
                                        
                                    target_term_obj["definitions"].append({
                                        "meaning": translated_meaning,
                                        "domain": domain,
                                        "example_source": definition.get("example_source") or "",
                                        "example_target": translated_example,
                                    })
                                    
                                    translated_keys.add(def_key)
                                    progress_state["translated_items"].append(def_key)
                                    progress_state["stats"]["translated"] = progress_state["stats"].get("translated", 0) + 1
                                cache_hits_count += 1
                            else:
                                cache_miss_keys.append(key)
                        except Exception as ce:
                            print(f"Redis cache parse error for {key}: {ce}")
                            cache_miss_keys.append(key)
                     else:
                        cache_miss_keys.append(key)
            except Exception as ce:
                print(f"  [Redis] MGET cache check error / MGET cache sorgu hatası: {ce}")
                cache_miss_keys = cache_keys
        else:
            cache_miss_keys = cache_keys

        if cache_hits_count > 0:
            print(f"  [Redis] {cache_hits_count} unique definitions loaded from cache / {cache_hits_count} benzersiz tanım önbellekten (cache) okunarak yüklendi.")
            tmp_path = output_json + ".tmp"
            with open(tmp_path, "wb") as handle:
                handle.write(orjson.dumps(existing_data, option=orjson.OPT_INDENT_2))
            safe_replace(tmp_path, output_json)

        # Prepare cache misses for LLM
        pending_api_items = []
        for key in cache_miss_keys:
            first_occurrence = memory_map[key][0]
            pending_api_items.append({
                "cache_key": key,
                "term": first_occurrence["term"],
                "meaning_hint": first_occurrence["meaning_hint"],
                "domain": first_occurrence["domain"],
                "example_source": first_occurrence["example_source"],
            })

        if args.limit > 0:
            pending_api_items = pending_api_items[:args.limit]

        if not pending_api_items:
            print(f"  No pending translations for {target_lang} / {target_lang} için bekleyen çeviri yok.")
            save_translation_progress(progress_json, progress_state)
            active_logger.info("translation_no_pending", target_lang=target_lang)
            active_logger.close(
                target_lang=target_lang,
                progress_json=progress_json,
                output_json=output_json,
                stats=progress_state.get("stats", {}),
            )
            print(f"  Log file / Log dosyası: {active_logger.log_path}")
            active_logger = None
            continue

        print(f"  Pending definitions / Bekleyen tanımlar: {len(pending_api_items)}")

        # Step 3: Batching / Akıllı Paketleme
        batches = [pending_api_items[i:i + batch_size] for i in range(0, len(pending_api_items), batch_size)]
        sem = asyncio.Semaphore(args.semaphore)
        save_lock = asyncio.Lock()
        
        last_save_time = time.time()
        save_interval = 15.0  # seconds
        completed_batches = 0
        total_batches = len(batches)

        # Step 4: Split-and-Retry Management / Hata/Split-and-Retry Yönetimi
        async def translate_batch_with_split(batch_items, depth=0):
            if not batch_items:
                return []
            
            api_items = [
                {
                    "term": item["term"],
                    "meaning_hint": item["meaning_hint"],
                    "domain": item["domain"],
                    "example_source": item["example_source"],
                }
                for item in batch_items
            ]
            
            prompt = get_translation_prompt(api_items, src_lang, target_lang, src_name, tgt_name)
            
            async with sem:
                resp_text = await asyncio.to_thread(
                    chain.call_with_fallback,
                    prompt,
                    context={"phase": "translate", "target_lang": target_lang, "depth": depth, "batch_size": len(batch_items)}
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
                # Save to Redis with 30-day TTL (2592000 seconds)
                if redis_client:
                    try:
                        pipe = redis_client.pipeline(transaction=False)
                        for idx, item in enumerate(batch_items):
                            res = results[idx]
                            cache_key = item["cache_key"]
                            cache_val = orjson.dumps({
                                "meaning": (res.get("meaning") or "").strip(),
                                "example_target": (res.get("example_target") or "").strip()
                            })
                            pipe.set(cache_key, cache_val, ex=2592000)
                        await pipe.execute()
                    except Exception as ce:
                        print(f"Redis pipeline write error: {ce}")
                return results
            else:
                # Split and retry
                if len(batch_items) > 1:
                    mid = len(batch_items) // 2
                    left_half = batch_items[:mid]
                    right_half = batch_items[mid:]
                    print(f"\033[1;33m  [Split] Failure/Validation failed. Splitting batch (Size: {len(batch_items)}) in two... / Hata/Doğrulama başarısız. Paket (Boyut: {len(batch_items)}) ikiye bölünüyor... Depth / Derinlik: {depth + 1}\033[0m")
                    
                    left_res = await translate_batch_with_split(left_half, depth + 1)
                    right_res = await translate_batch_with_split(right_half, depth + 1)
                    return left_res + right_res
                else:
                    # Single item fail
                    print(f"\033[1;31m  [Split-FAIL] Term '{batch_items[0]['term']}' could not be translated / Terim '{batch_items[0]['term']}' çevrilemedi.\033[0m")
                    return [{"term": batch_items[0]["term"], "meaning": "", "example_target": ""}]

        # Step 5: Concurrent Batch Execution & Lock-Protected Incremental Saving
        pbar = tqdm(total=len(batches), desc=f"Translating / Çevriliyor: {src_lang} -> {target_lang}")

        async def process_batch(b_idx, batch_items):
            nonlocal last_save_time, completed_batches
            results = await translate_batch_with_split(batch_items)
            
            async with save_lock:
                for idx, item in enumerate(batch_items):
                    if idx >= len(results):
                        break
                    res = results[idx]
                    translated_meaning = (res.get("meaning") or "").strip()
                    translated_example = (res.get("example_target") or "").strip() if res.get("example_target") else ""
                    if not translated_meaning:
                        continue

                    cache_key = item["cache_key"]
                    
                    # Apply to all original matching nested positions
                    for occurrence in memory_map[cache_key]:
                        term = occurrence["term"]
                        pos = occurrence["pos"]
                        art = occurrence["art"]
                        domain = occurrence["domain"]
                        def_key = occurrence["def_key"]
                        term_obj = occurrence["term_obj"]
                        definition = occurrence["source_def"]
                        
                        term_key = term.lower()
                        target_term_obj = existing_lookup.get((term_key, pos, art))
                        
                        if not target_term_obj:
                            target_term_obj = {
                                "term": term,
                                "definitions": [],
                            }
                            if pos:
                                target_term_obj["partOfSpeech"] = pos
                            if art:
                                target_term_obj["article"] = art
                            for k, value in term_obj.items():
                                if k not in ("definitions", "kaynak_dosya", "artikel", "article", "domain", "partOfSpeech"):
                                    target_term_obj[k] = value
                            existing_data.append(target_term_obj)
                            existing_lookup[(term_key, pos, art)] = target_term_obj
                            
                        existing_meanings = set()
                        for d in target_term_obj.get("definitions", []):
                            if isinstance(d, str):
                                existing_meanings.add(d.strip().lower())
                            elif isinstance(d, dict):
                                existing_meanings.add((d.get("meaning") or "").strip().lower())
                                
                        if translated_meaning.lower() in existing_meanings:
                            translated_keys.add(def_key)
                            if def_key not in progress_state["translated_items"]:
                                progress_state["translated_items"].append(def_key)
                            continue
                            
                        target_term_obj.setdefault("definitions", []).append({
                            "meaning": translated_meaning,
                            "domain": domain,
                            "example_source": definition.get("example_source") or "",
                            "example_target": translated_example,
                        })
                        
                        translated_keys.add(def_key)
                        progress_state["translated_items"].append(def_key)
                        progress_state["stats"]["translated"] = progress_state["stats"].get("translated", 0) + 1

                completed_batches += 1
                current_time = time.time()
                is_last_batch = (completed_batches >= total_batches)

                if is_last_batch or (current_time - last_save_time >= save_interval):
                    # Save incrementally after each batch inside lock
                    tmp_path = output_json + ".tmp"
                    with open(tmp_path, "wb") as handle:
                        handle.write(orjson.dumps(existing_data, option=orjson.OPT_INDENT_2))
                    safe_replace(tmp_path, output_json)
                    save_translation_progress(progress_json, progress_state)
                    last_save_time = current_time

            pbar.update(1)

        # Run all batches concurrently under the semaphore
        tasks = [process_batch(b_idx, batch_items) for b_idx, batch_items in enumerate(batches)]
        await asyncio.gather(*tasks)
        pbar.close()

        print(f"\n  Finished translation for {target_lang} / {target_lang} için çeviri tamamlandı.")
        active_logger.close(
            source=source_path,
            target_lang=target_lang,
            output_json=output_json,
            progress_json=progress_json,
            translated_items=len(progress_state.get("translated_items", [])),
            failed_batches=len(progress_state.get("failed_batches", [])),
            stats=progress_state.get("stats", {}),
        )
        print(f"  Log file / Log dosyası: {active_logger.log_path}")
        active_logger = None

    if redis_client:
        await redis_client.close()

    chain.print_stats()
    print("\n\033[1;32mTranslation finished / Çeviri işlemi tamamlandı.\033[0m")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
