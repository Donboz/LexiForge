#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import json
import time
import hashlib
import shutil
import asyncio
import argparse
import PyPDF2
from tqdm.asyncio import tqdm as tqdm_async

# Prepend parent directory to sys.path to resolve imports from handlers/
# Üst dizini sys.path'e ekleyerek handlers/ modüllerini yüklemeyi sağlar
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from handlers.fallback_chain import FallbackChain, parse_json_response, load_config
from handlers.run_logger import RunLogger, now_iso, load_json_with_default, save_json_atomic


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


def get_extract_cache_key(src_lang, tgt_lang, domain, page_text):
    input_str = f"{(domain or 'GENERAL').strip().upper()}||{page_text.strip()}"
    sha = hashlib.sha256(input_str.encode('utf-8')).hexdigest()
    return f"sözlük:{src_lang.upper()}:{tgt_lang.upper()}:{sha}"


# ─── Helpers / Yardımcılar ─────────────────────────────────────────────────────

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


def save_terms(output_path, new_terms, default_domain="GENERAL"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    existing_terms = []
    if os.path.exists(output_path):
        try:
            with open(output_path, "rb") as f:
                import orjson
                loaded = orjson.loads(f.read())
                if isinstance(loaded, list):
                    existing_terms = loaded
        except Exception:
            pass

    lookup = {}
    for item in existing_terms:
        term = (item.get("term") or "").strip()
        pos = (item.get("partOfSpeech") or "").strip().upper()
        if "artikel" in item:
            item["article"] = item.pop("artikel")
        gender = get_normalized_gender(item)
        
        root_dom = item.pop("domain") if "domain" in item else default_domain
        for d in item.get("definitions", []):
            if isinstance(d, dict) and "domain" not in d:
                d["domain"] = root_dom or default_domain

        key = (term.lower(), pos, gender)
        lookup[key] = item

    for item in new_terms:
        if not isinstance(item, dict) or not item.get("term"):
            continue
        
        term = (item.get("term") or "").strip()
        pos = (item.get("partOfSpeech") or "").strip().upper()
        if "artikel" in item:
            item["article"] = item.pop("artikel")
        gender = get_normalized_gender(item)

        root_dom = item.pop("domain") if "domain" in item else default_domain
        for d in item.get("definitions", []):
            if isinstance(d, dict) and "domain" not in d:
                d["domain"] = root_dom or default_domain

        key = (term.lower(), pos, gender)
        if key in lookup:
            existing_item = lookup[key]
            for field, val in item.items():
                if field in ("term", "partOfSpeech", "article", "definitions"):
                    continue
                if val is not None and val != "":
                    if existing_item.get(field) in (None, ""):
                        existing_item[field] = val
                    elif isinstance(existing_item.get(field), str) and isinstance(val, str) and len(val) > len(existing_item.get(field)):
                        existing_item[field] = val

            existing_defs = existing_item.setdefault("definitions", [])
            if not isinstance(existing_defs, list):
                existing_defs = []
                existing_item["definitions"] = existing_defs
            
            existing_meanings = { (d.get("meaning") or "").strip().lower() for d in existing_defs if isinstance(d, dict) }
            for new_def in item.get("definitions", []):
                if not isinstance(new_def, dict):
                    continue
                meaning_key = (new_def.get("meaning") or "").strip().lower()
                if meaning_key and meaning_key not in existing_meanings:
                    existing_defs.append(new_def)
                    existing_meanings.add(meaning_key)
        else:
            existing_terms.append(item)
            lookup[key] = item

    with open(output_path, "wb") as f:
        import orjson
        f.write(orjson.dumps(existing_terms, option=orjson.OPT_INDENT_2))


def _default_progress_state(pdf_name, src_lang, tgt_lang, total_pages, start_page, end_page):
    return {
        "version": 2,
        "pdf": pdf_name,
        "source_lang": src_lang,
        "target_lang": tgt_lang,
        "total_pages": total_pages,
        "range": {"start": start_page, "end": end_page},
        "processed_pages": [],
        "skipped_pages": [],
        "skipped_page_numbers": [],
        "failed_pages": [],
        "failed_page_numbers": [],
        "stats": {
            "success_pages": 0,
            "empty_pages": 0,
            "already_processed": 0,
            "extract_errors": 0,
            "json_parse_errors": 0
        },
        "status": {
            "state": "IN_PROGRESS",
            "range_done": False,
            "pdf_done": False,
            "range_progress": "0/0",
            "progress_ratio": 0.0,
        },
        "completed_at": None,
        "started_at": now_iso(),
        "updated_at": now_iso(),
    }


def load_progress(progress_path, pdf_name, src_lang, tgt_lang, total_pages, start_page, end_page):
    defaults = _default_progress_state(pdf_name, src_lang, tgt_lang, total_pages, start_page, end_page)
    loaded = load_json_with_default(progress_path, defaults)

    if loaded.get("version") != 2:
        legacy_dir = os.path.join(os.path.dirname(progress_path), "progress_legacy")
        os.makedirs(legacy_dir, exist_ok=True)
        legacy_backup_path = os.path.join(legacy_dir, os.path.basename(progress_path).replace(".json", "_v1_backup.json"))
        if os.path.exists(progress_path) and not os.path.exists(legacy_backup_path):
            try:
                shutil.copy2(progress_path, legacy_backup_path)
            except Exception:
                pass

        old_pages = loaded.get("processed_pages", []) if isinstance(loaded, dict) else []
        state = _default_progress_state(pdf_name, src_lang, tgt_lang, total_pages, start_page, end_page)
        if isinstance(old_pages, list):
            state["processed_pages"] = sorted(set([int(p) for p in old_pages if isinstance(p, int)]))
        state["migrated_from"] = "v1"
        state["legacy_backup_path"] = legacy_backup_path
        return state

    for k in ["processed_pages", "skipped_pages", "skipped_page_numbers", "failed_pages", "failed_page_numbers"]:
        if not isinstance(loaded.get(k), list):
            loaded[k] = []
    if not isinstance(loaded.get("stats"), dict):
        loaded["stats"] = defaults["stats"]
    if not isinstance(loaded.get("status"), dict):
        loaded["status"] = defaults["status"]

    loaded["pdf"] = pdf_name
    loaded["source_lang"] = src_lang
    loaded["target_lang"] = tgt_lang
    loaded["total_pages"] = total_pages
    loaded["range"] = {"start": start_page, "end": end_page}
    loaded["updated_at"] = now_iso()
    return loaded


def save_progress(progress_path, state):
    processed_pages = sorted(set(state.get("processed_pages", [])))
    state["processed_pages"] = processed_pages

    skipped_numbers = sorted(set(
        e.get("page") for e in state.get("skipped_pages", []) if isinstance(e, dict) and isinstance(e.get("page"), int)
    ))
    failed_numbers = sorted(set(
        e.get("page") for e in state.get("failed_pages", []) if isinstance(e, dict) and isinstance(e.get("page"), int)
    ))
    state["skipped_page_numbers"] = skipped_numbers
    state["failed_page_numbers"] = failed_numbers

    page_range = state.get("range", {}) if isinstance(state.get("range"), dict) else {}
    start_page = int(page_range.get("start", 1) or 1)
    end_page = int(page_range.get("end", 0) or 0)
    if end_page < start_page:
        total_range_pages = 0
        processed_in_range = 0
    else:
        total_range_pages = (end_page - start_page + 1)
        processed_in_range = len([p for p in processed_pages if isinstance(p, int) and start_page <= p <= end_page])

    range_done = (total_range_pages > 0 and processed_in_range >= total_range_pages)
    total_pages = int(state.get("total_pages", 0) or 0)
    pdf_done = (total_pages > 0 and len([p for p in processed_pages if isinstance(p, int) and 1 <= p <= total_pages]) >= total_pages)
    ratio = round((processed_in_range / total_range_pages), 4) if total_range_pages > 0 else 0.0

    state["status"] = {
        "state": "DONE" if range_done else "IN_PROGRESS",
        "range_done": range_done,
        "pdf_done": pdf_done,
        "range_progress": f"{processed_in_range}/{total_range_pages}",
        "progress_ratio": ratio,
    }
    if range_done and not state.get("completed_at"):
        state["completed_at"] = now_iso()

    state["updated_at"] = now_iso()
    try:
        save_json_atomic(progress_path, state)
    except Exception as e:
        print(f"Error saving progress: {e}")


def append_unique_page_event(items, page, reason, detail=""):
    if not hasattr(items, "_seen_set"):
        items._seen_set = { (row.get("page"), row.get("reason")) for row in items if isinstance(row, dict) }
    if (page, reason) in items._seen_set:
        return
    items._seen_set.add((page, reason))
    items.append({
        "page": page,
        "reason": reason,
        "detail": detail,
        "timestamp": now_iso(),
    })


def get_extraction_prompt(page_text, src_lang, tgt_lang, domain, src_name, tgt_name):
    eff_target = "TR" if src_lang == tgt_lang else tgt_lang
    eff_tgt_name = "Türkçe" if src_lang == tgt_lang else tgt_name

    if domain == "IDIOM":
        return f"""Sana bir {src_name} deyimler ve atasözleri sözlüğünün ham metnini veriyorum.
Metndeki deyimleri, atasözlerini ve kalıp ifadeleri (Idioms & Phrases) çıkar ve her birinin {eff_tgt_name} ({eff_target}) karşılığını/anlamını yaz.

Kurallar:
- "term": Deyim, atasözü veya kalıp ifadenin kendisi (örneğin: "jdm. den Daumen drücken").
- "partOfSpeech": Her zaman "IDIOM" veya "PHRASE" olmalıdır.
- "definitions": Her anlam için:
  - "meaning": {eff_tgt_name} ({eff_target}) karşılığı veya açıklaması.
  - "domain": "GENERAL" (veya şu listeden en uygun domain: GENERAL | MEDICAL | LEGAL | TECHNICAL | ACADEMIC | etc.).
  - "example_source": Metinde varsa kaynak dilde örnek cümle; yoksa kendin bir tane üret.
  - "example_target": Örnek cümlenin Türkçe karşılığı; örnek yoksa kendin bir tane üret.
- Sayfa başlığı, dipnot, reklam, indeks ve sayfa numarasını terim diye yazma.
- Emin olmadığın bozuk OCR satırlarını atla.

METİN:
{page_text}

SADECE JSON ARRAY VER:
[
  {{
    "term": "deyim veya kalıp ifade",
    "partOfSpeech": "IDIOM",
    "definitions": [
      {{
        "meaning": "{eff_tgt_name} anlamı/karşılığı",
        "domain": "GENERAL",
        "example_source": "kaynak dil örnek cümle veya kendin bir tane üret",
        "example_target": "çeviri karşılığı veya kendin bir tane üret"
      }}
    ]
  }}
]
"""

    return f"""Sana bir {src_name} sözlüğün ham metnini veriyorum.
Metndeki sözlük/terim girdilerini çıkar ve her birinin {eff_tgt_name} ({eff_target}) karşılığını yaz.

Kurallar:
- "term": Kaynak dildeki ({src_lang}) kelime veya terim.
- "partOfSpeech": Terimin sözcük türü (örn: NOUN, VERB, ADJECTIVE, ADVERB, PRONOUN, PREPOSITION, CONJUNCTION, etc.) varsa veya belirlenebiliyorsa.
- "gender": İsimler için artikeline göre cinsiyeti (MASCULINE, FEMININE, NEUTER, COMMON veya null).
- "article": Varsa der/die/das/le/la gibi artikel; yoksa tespit et.
- "definitions": Her anlam için:
  - "meaning": {eff_tgt_name} ({eff_target}) kısa anlamı.
  - "domain": {domain} veya şu listeden en uygun domain: GENERAL | MEDICAL | LEGAL | TECHNICAL | ACADEMIC | etc.
  - "example_source": Metinde varsa kaynak dilde örnek cümle; yoksa kendin bir tane üret.
  - "example_target": Örnek cümlenin Türkçe karşılığı; örnek yoksa kendin bir tane üret.
- Sayfa başlığı, dipnot, reklam, indeks ve sayfa numarasını terim diye yazma.
- Emin olmadığın bozuk OCR satırlarını atla.

METİN:
{page_text}

SADECE JSON ARRAY VER:
[
  {{
    "term": "kaynak kelime",
    "partOfSpeech": "NOUN | VERB | ADJECTIVE | etc. | null",
    "gender": "MASCULINE | FEMININE | NEUTER | COMMON | null",
    "article": "der/die/das/null",
    "definitions": [
      {{
        "meaning": "{eff_tgt_name} anlamı",
        "domain": "{domain}",
        "example_source": "kaynak dil örnek cümle veya kendin bir tane üret",
        "example_target": "çeviri karşılığı veya kendin bir tane üret"
      }}
    ]
  }}
]
"""


# ─── Async Main Loop / Asenkron Ana Döngü ──────────────────────────────────────

async def main_async():
    parser = argparse.ArgumentParser(description="Glossa Master Extractor — Fallback Chain")
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--file", help="PDF file name or path")
    group.add_argument("--all", action="store_true", help="Process all PDFs")
    parser.add_argument("--start-page", type=int, default=1, help="Start page (1-based)")
    parser.add_argument("--end-page", type=int, default=0, help="End page (0 = all)")
    parser.add_argument("--max-pages", type=int, default=0, help="Max pages to process")
    parser.add_argument("--provider", help="Filter by a specific provider (openrouter, cerebras, groq, google, github, sambanova, mistral)")
    parser.add_argument("--model", help="Filter by a specific model")
    parser.add_argument("--semaphore", type=int, default=8, help="Max concurrent requests (Semaphore)")
    args = parser.parse_args()

    from handlers.selector import select_item_interactive, select_boolean_interactive, select_file_interactive

    config = load_config()
    base_dir = config.get("base_dir", "")
    if not base_dir or not os.path.exists(base_dir):
        print(f"Error: Base directory '{base_dir}' does not exist. / Hata: '{base_dir}' ana dizini bulunamadı.")
        sys.exit(1)

    pdf_files = []

    if args.file:
        if os.path.exists(args.file):
            pdf_files.append(args.file)
        else:
            full_path = os.path.join(base_dir, args.file)
            if os.path.exists(full_path):
                pdf_files.append(full_path)
            else:
                print(f"Error: PDF file '{args.file}' not found. / Hata: '{args.file}' PDF dosyası bulunamadı.")
                sys.exit(1)
    elif args.all:
        for root, _, files in os.walk(base_dir):
            for f in files:
                if f.lower().endswith(".pdf"):
                    pdf_files.append(os.path.join(root, f))
    else:
        action_options = [
            "Process a single PDF / Tek bir PDF dosyasını işle", 
            "Process all PDFs / Tüm PDF dosyalarını işle"
        ]
        selected_action = select_item_interactive(action_options, "Please select an action / Lütfen yapmak istediğiniz işlemi seçin:")
        if not selected_action:
            print("Selection cancelled. / İptal edildi.")
            sys.exit(0)

        if "Process all" in selected_action or "Tüm PDF" in selected_action:
            args.all = True
            for root, _, files in os.walk(base_dir):
                for f in files:
                    if f.lower().endswith(".pdf"):
                        pdf_files.append(os.path.join(root, f))
        else:
            all_pdfs = []
            for root, _, files in os.walk(base_dir):
                for f in files:
                    if f.lower().endswith(".pdf"):
                        all_pdfs.append(os.path.join(root, f))
            if not all_pdfs:
                print("No PDF files found to process. / İşlenecek PDF dosyası bulunamadı.")
                sys.exit(1)

            selected = select_file_interactive(all_pdfs, "Select the PDF file to extract terms from / Terim çıkarılacak PDF dosyasını seçin")
            if not selected:
                print("Selection cancelled. / İptal edildi.")
                sys.exit(0)
            pdf_files.append(selected)
            args.file = selected

    args.overwrite_progress = False
    if "--file" not in sys.argv and "--all" not in sys.argv:
        skip_processed = select_boolean_interactive(
            "Should already processed pages be skipped? / Daha önce işlenmiş sayfalar atlansın mı (Skip already processed pages)?",
            default=True
        )
        if not skip_processed:
            args.overwrite_progress = True

    if "--start-page" not in sys.argv and "--end-page" not in sys.argv and "--max-pages" not in sys.argv:
        set_limits = select_boolean_interactive("Do you want to set page range and limits? / Sayfa aralığı ve maksimum sayfa sınırı belirlemek ister misiniz?", default=False)
        if set_limits:
            try:
                sp = input("Enter start page (Default 1) / Başlangıç sayfası girin (Varsayılan 1): ").strip()
                args.start_page = int(sp) if sp else 1
            except ValueError:
                args.start_page = 1

            try:
                ep = input("Enter end page (Default last page: 0) / Bitiş sayfası girin (Varsayılan son sayfa: 0): ").strip()
                args.end_page = int(ep) if ep else 0
            except ValueError:
                args.end_page = 0

            try:
                mp = input("Max pages to process (Default unlimited: 0) / Maksimum işlenecek sayfa sayısı (Limit, varsayılan limitsiz: 0): ").strip()
                args.max_pages = int(mp) if mp else 0
            except ValueError:
                args.max_pages = 0

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
                selected_opt = select_item_interactive(options, "Select a Provider to extract terms with / Terim çıkarılacak Sağlayıcıyı seçin")
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
        print("\033[1;31m[Extractor] ERROR: No available provider/model found! / HATA: Hiç kullanılabilir provider/model bulunamadı!\033[0m")
        sys.exit(1)

    print(f"\033[1;36m[Extractor] Fallback chain ready: {chain.get_total_model_count()} models ({chain.get_available_count()} active) / "
          f"Fallback zinciri hazır: {chain.get_total_model_count()} model ({chain.get_available_count()} aktif)\033[0m")
    print(f"\033[36m[Extractor] Provider order: {', '.join(dict.fromkeys(t.provider_key for t in chain.trackers))} / "
          f"Sağlayıcı sırası: {', '.join(dict.fromkeys(t.provider_key for t in chain.trackers))}\033[0m")

    if not pdf_files:
        print("No PDF files found. / PDF dosyası bulunamadı.")
        return

    # PDF processing loop / PDF işleme döngüsü
    for pdf_path in pdf_files:
        pdf_name = os.path.basename(pdf_path)
        active_logger = RunLogger(task="extractor", source=pdf_name)
        print(f"\n\033[1;33m{'='*60}\033[0m")
        print(f"\033[1;33m  Processing PDF / PDF İşleniyor: {pdf_name}\033[0m")
        print(f"\033[1;33m{'='*60}\033[0m")

        pdf_map = config.get("pdf_lang_map", {})
        mapping = pdf_map.get(pdf_name)
        if not mapping:
            for k, v in pdf_map.items():
                if os.path.basename(k) == pdf_name:
                    mapping = v
                    break
        if not mapping:
            mapping = {"source": "DE", "target": "TR", "domain": "GENERAL"}

        src_lang = mapping.get("source", "DE")
        tgt_lang = mapping.get("target", "TR")
        domain = mapping.get("domain", "GENERAL")

        languages_cfg = config.get("languages", {})
        src_name = languages_cfg.get(src_lang, src_lang)
        tgt_name = languages_cfg.get(tgt_lang, tgt_lang)

        pdf_base = os.path.splitext(pdf_name)[0]
        eff_target = "TR" if src_lang == tgt_lang else tgt_lang
        output_name = f"{pdf_base}_sozluk_{src_lang}_{eff_target}.json"
        output_path = os.path.join("data", "json", output_name)
        progress_path = os.path.join("data", "progress", "extractor", f"{pdf_base}_sozluk_{src_lang}_{eff_target}_progress.json")

        if not os.path.exists(output_path) and os.path.exists(progress_path):
            print("  Warning: output JSON missing but progress exists. Keeping progress. / Uyarı: çıktı JSON'ı eksik ama ilerleme dosyası var. İlerleme korunuyor.")

        try:
            with open(pdf_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                total_pages = len(reader.pages)
        except Exception as e:
            print(f"Error reading PDF / PDF okuma hatası {pdf_name}: {e}")
            active_logger.error("pdf_read_error", error=str(e))
            active_logger.close(error="pdf_read_error")
            active_logger = None
            continue

        start_page = max(1, args.start_page)
        end_page = total_pages
        if args.end_page > 0:
            end_page = min(args.end_page, total_pages)
        if args.max_pages > 0:
            end_page = min(end_page, start_page + args.max_pages - 1)

        progress_state = load_progress(
            progress_path,
            pdf_name,
            src_lang,
            eff_target,
            total_pages,
            start_page,
            end_page,
        )
        if getattr(args, "overwrite_progress", False):
            progress_state["processed_pages"] = []
            progress_state["stats"] = _default_progress_state(pdf_name, src_lang, eff_target, total_pages, start_page, end_page)["stats"]
            save_progress(progress_path, progress_state)
            if os.path.exists(output_path):
                try:
                    os.remove(output_path)
                    print(f"  [Progress] Old output file deleted (Overwrite) / Eski çıktı dosyası silindi: {output_path}")
                except Exception as e:
                    print(f"  [Progress] Error deleting output file / Çıktı dosyası silinirken hata: {e}")
        processed_pages = set(progress_state.get("processed_pages", []))

        active_logger.info(
            "pdf_processing_started",
            pdf=pdf_name,
            source_lang=src_lang,
            target_lang=eff_target,
            domain=domain,
            start_page=start_page,
            end_page=end_page,
            total_pages=total_pages,
            output_path=output_path,
            progress_path=progress_path,
        )

        print(f"  Lang / Dil: {src_lang}→{eff_target} | Domain: {domain} | Pages / Sayfalar: {start_page}-{end_page}/{total_pages}")

        # Flatten & Map / Sayfa metinlerini topla
        page_texts = {}
        cache_keys = []
        page_by_key = {}
        
        try:
            with open(pdf_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page_num in range(start_page, end_page + 1):
                    if page_num in processed_pages:
                        continue
                    try:
                        page_text = reader.pages[page_num - 1].extract_text() or ""
                        if page_text.strip():
                            page_texts[page_num] = page_text
                            key = get_extract_cache_key(src_lang, eff_target, domain, page_text)
                            cache_keys.append(key)
                            page_by_key[key] = (page_num, page_text)
                    except Exception:
                        pass
        except Exception as e:
            print(f"Error reading PDF / PDF okuma hatası {pdf_name}: {e}")
            active_logger.error("pdf_read_error", error=str(e))
            active_logger.close(error="pdf_read_error")
            active_logger = None
            continue

        cached_responses = {}
        if redis_client and cache_keys:
            try:
                print(f"  [Redis] Querying cache (MGET) for {len(cache_keys)} pages... / {len(cache_keys)} sayfa için toplu cache (MGET) sorgulanıyor...")
                redis_results = await redis_client.mget(cache_keys)
                for key, val in zip(cache_keys, redis_results):
                    if val is not None:
                        cached_responses[key] = val.decode('utf-8')
            except Exception as e:
                print(f"  [Redis] MGET cache check error / MGET cache sorgu hatası: {e}")

        # Process cache hits / Cache'den bulduklarımızı yükle
        if cached_responses:
            print(f"  [Redis] {len(cached_responses)} pages found in cache. Loading directly... / {len(cached_responses)} sayfa önbellekte bulundu. Doğrudan yükleniyor...")
            for key, resp_text in cached_responses.items():
                page_num, page_text = page_by_key[key]
                try:
                    terms = parse_json_response(resp_text)
                    if terms:
                        save_terms(output_path, terms, domain)
                        print(f"\033[1;32m  ✓ [Redis Cache Hit] {len(terms)} terms loaded / terim yükleniyor (Page / Sayfa {page_num})\033[0m")
                        active_logger.info("page_saved", page=page_num, terms_saved=len(terms), output_path=output_path)
                    processed_pages.add(page_num)
                    progress_state["processed_pages"] = sorted(list(processed_pages))
                    progress_state["stats"]["success_pages"] = progress_state["stats"].get("success_pages", 0) + 1
                    save_progress(progress_path, progress_state)
                except Exception as e:
                    print(f"  [Redis Cache Hit] JSON parse error / JSON ayrıştırma hatası for Page / Sayfa {page_num}: {e}")

        # Collect misses / Eksik kalan sayfaları listele
        pending_pages = []
        for page_num, page_text in page_texts.items():
            if page_num not in processed_pages:
                pending_pages.append((page_num, page_text))

        sem = asyncio.Semaphore(args.semaphore)
        file_lock = asyncio.Lock()
        
        accumulated_terms = []
        last_save_time = time.time()
        save_interval = 15.0  # seconds
        completed_pages_count = 0
        total_pending_pages = len(pending_pages)

        # Concurrent extractor with split-and-retry
        async def extract_page_with_split(page_num, page_text, depth=0):
            if not page_text.strip():
                return []
                
            prompt = get_extraction_prompt(page_text, src_lang, tgt_lang, domain, src_name, tgt_name)
            
            async with sem:
                resp_text = await asyncio.to_thread(
                    chain.call_with_fallback,
                    prompt,
                    context={"pdf": pdf_name, "page": page_num, "phase": "extract", "depth": depth}
                )

            is_valid = False
            terms = []
            if resp_text is not None:
                try:
                    terms = parse_json_response(resp_text)
                    if isinstance(terms, list):
                        is_valid = True
                except Exception:
                    pass

            if is_valid:
                if redis_client:
                    try:
                        cache_key = get_extract_cache_key(src_lang, eff_target, domain, page_text)
                        await redis_client.set(cache_key, resp_text.encode('utf-8'), ex=2592000)
                    except Exception as ce:
                        print(f"Redis cache set error: {ce}")
                return terms
            else:
                # Text splitting logic / Metin bölme mantığı
                lines = page_text.splitlines()
                if len(lines) > 5 and depth < 3:
                    mid = len(lines) // 2
                    left_text = "\n".join(lines[:mid])
                    right_text = "\n".join(lines[mid:])
                    print(f"\033[1;33m  [Split] Page {page_num} extraction failed. Splitting text in two... Depth / Derinlik: {depth+1}\033[0m")
                    
                    left_terms = await extract_page_with_split(page_num, left_text, depth + 1)
                    right_terms = await extract_page_with_split(page_num, right_text, depth + 1)
                    return left_terms + right_terms
                else:
                    print(f"\033[1;31m  [Split-FAIL] Page {page_num} extraction failed completely. / Sayfa {page_num} extraction yapılamadı.\033[0m")
                    return []

        async def run_page_extraction(page_num, page_text):
            nonlocal last_save_time, completed_pages_count
            terms = await extract_page_with_split(page_num, page_text)
            
            async with file_lock:
                if terms:
                    accumulated_terms.extend(terms)
                    print(f"\033[1;32m  ✓ Page / Sayfa {page_num} processed / işlendi ({len(terms)} terms queued / terim sıraya alındı)\033[0m")
                    active_logger.info("page_queued", page=page_num, terms_count=len(terms))
                else:
                    print(f"  Page / Sayfa {page_num}: No terms extracted / Hiç terim çıkarılamadı.")
                    active_logger.warn("page_no_terms", page=page_num)
                processed_pages.add(page_num)
                progress_state["processed_pages"] = sorted(list(processed_pages))
                progress_state["stats"]["success_pages"] = progress_state["stats"].get("success_pages", 0) + 1
                
                completed_pages_count += 1
                current_time = time.time()
                is_last_page = (completed_pages_count >= total_pending_pages)
                
                if is_last_page or (current_time - last_save_time >= save_interval):
                    if accumulated_terms:
                        save_terms(output_path, accumulated_terms, domain)
                        active_logger.info("terms_saved_batch", count=len(accumulated_terms), output_path=output_path)
                        accumulated_terms.clear()
                    save_progress(progress_path, progress_state)
                    last_save_time = current_time

            cooldown = chain.get_cooldown()
            if cooldown > 0:
                await asyncio.sleep(cooldown)

        if pending_pages:
            tasks = [run_page_extraction(p_num, p_text) for p_num, p_text in pending_pages]
            await tqdm_async.gather(*tasks, desc=f"Extracting {pdf_name}")

        save_progress(progress_path, progress_state)
        active_logger.close(
            pdf=pdf_name,
            processed_pages=len(progress_state.get("processed_pages", [])),
            skipped_pages=len(progress_state.get("skipped_pages", [])),
            failed_pages=len(progress_state.get("failed_pages", [])),
            stats=progress_state.get("stats", {}),
            progress_path=progress_path,
        )
        print(f"  Log file / Log dosyası: {active_logger.log_path}")
        active_logger = None

    if redis_client:
        await redis_client.close()

    chain.print_stats()
    print("\n\033[1;32mExtraction finished / Çıkarma işlemi tamamlandı.\033[0m")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
