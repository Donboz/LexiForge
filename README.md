# LexiForge

An optimized asynchronous script suite designed for extracting, enriching, translating, merging, and cleaning structured data from dictionary files.

---

## 🚀 Extensibility Notice (Processing Other Formats)

> [!IMPORTANT]
> Although this project was initially built and optimized for processing PDF dictionaries, its core architecture is highly modular and extensible. You can easily adapt it to process other source formats such as:
>
> - **Corpora / Raw Text Files**
> - **CSV / TSV Spreadsheets**
> - **TMX (Translation Memory Exchange)**
> - **JSON Files / API dumps**
>
> Data processing can be handled in two ways:
>
> 1. **LLM-based (Cognitive Processing):** By utilizing the provided fallback API chain for complex context-aware restructuring and validation.
> 2. **Deterministic (Simple Script Logic):** By writing clean python utilities for direct field parsing and data mapping.
>
> If you are working with an **AI Coding Agent** (e.g., Antigravity, Claude Code, etc.), you can easily direct the agent to implement new parsers or handlers within the `handlers/` and `scripts/` directories to extend support for these file formats.

---

## Core Scripts

All core executable scripts are located in the `scripts/` folder:

- `scripts/extractor.py`: Extracts terms and definitions from raw PDF text into JSON files (`data/json/*_sozluk_*.json`) [ASYNC].
- `scripts/enricher.py`: Grammatically and semantically enriches dictionary entries using LLMs [ASYNC].
- `scripts/translator.py`: Translates dictionary meanings into new target languages using LLMs [ASYNC].
- `scripts/clean.py`: Normalizes terms in merged dictionary files and deduplicates them based on language pairs [ORJSON].
- `scripts/clean_meanings.py`: Splits comma-separated meaning lists and removes duplicates [ORJSON].
- `scripts/merge_translated.py`: Merges enriched/translated language pair files into the final database [ORJSON].
- `scripts/splitter.py`: Splits large dictionary JSON files into smaller chunks [ORJSON].
- `scripts/test_providers.py`: Tests the API connection, rate limits, and latency of defined providers and models.
- `scripts/analyze_logs.py`: Analyzes fallback chain event logs to report model/provider performance, success/error rates, and latency.

---

## Asynchronous High-Performance Architecture (Async Pipeline)

Workflows are fully asynchronous to reduce costs, bypass rate limits, and make efficient use of CPU and network resources:

### 1. Concurrency Control (Semaphore = 10)

Network requests to providers are limited using `asyncio.Semaphore(10)`. A maximum of 10 batch requests are processed in parallel, allowing new ones to start as active requests finish.

### 2. Smart Batching (Batch Size = 30)

To prevent models from hitting output token limits and cutting off JSON outputs, translation and enrichment requests are divided into chunks of up to 30 items.

### 3. Visual Progress Bar (`tqdm.asyncio`)

All primary asynchronous loops are wrapped with `tqdm.asyncio`. You can monitor real-time processing speed (it/s), completion percentage, and Estimated Time of Arrival (ETA) in the terminal.

---

## Error Management & Validation (Audit & Validation)

### 1. Array Length Validation

The length of the response array returned by the LLM is strictly validated against the size of the request batch (e.g., 30 items) after each API call.

### 2. "Split-and-Retry" Error & Fallback Integration

If the array lengths do not match, or if an API/JSON parsing error occurs:

- If the batch size is greater than 1, the batch is **dynamically split into two halves**, and each half is retried asynchronously (divide & conquer).
- If the batch size is reduced to 1 and errors persist, the system triggers the fallback chain defined in `handlers/fallback_chain.py` to try the next model, or returns empty values to prevent the pipeline from blocking.

### 3. orjson Integration

To read, parse, and write large JSON files quickly without blocking the CPU, the C-based `orjson` library is integrated instead of the standard `json` library. File operations are performed directly at the UTF-8 byte level in binary mode (`wb`/`rb`).

---

## Namespaced Redis Caching Layer

An asynchronous `redis.asyncio` cache layer is integrated to speed up translation, extraction, and enrichment steps while reducing API costs.

### 1. Configuration (`config/config.json`)

```json
"redis": {
  "host": "127.0.0.1",
  "port": 6379,
  "db": 0,
  "password": null
}
```

### 2. Dynamic Namespaced Cache Schema

Cache keys are namespaced dynamically by the source (`source_lang`) and target (`target_lang`) languages. Successful operations are stored in Redis with a **30-day (2,592,000 seconds) TTL**.

- **Format:** `sözlük:{source_lang}:{target_lang}:{sha256_hash}`
- **SHA-256 Inputs:**
  - **Extractor:** `f"{domain}||{page_text}"`
  - **Enricher:** `f"{term}||{sorted_definitions_meanings_and_domains}"`
  - **Translator:** `f"{term}||{domain}||{meaning_hint}||{example_source}"`

### 3. Redis Pipeline / Bulk MGET Operations

To avoid overhead, keys for all terms/pages in a batch are compiled upfront and queried using a single asynchronous Redis `MGET` request. Hits are processed instantly from the memory map without contacting the API, and misses are routed to the LLMs.

---

## 📷 OCR & Layout Analysis (PaddleOCR)

To support scanned PDFs and complex multi-column layouts, **LexiForge** integrates **PaddleOCR** with **PyMuPDF**.

### Key Features:
- **Scanned PDF Support:** Converts PDF pages to high-resolution (300 DPI) images dynamically and reads them locally.
- **Vertical Column Heuristics:** Detects columns (e.g. 2-column dictionaries) and reads text top-to-bottom of the left column before proceeding to the right column. This keeps the dictionary layout intact for the LLM.
- **Bilingual Selection Locking:** Fully completed files and target languages are automatically detected and blocked (disabled) in interactive lists to prevent double-processing.

### Setup (Python 3.11 required):
```bash
py -3.11 -m pip install pymupdf paddlepaddle paddleocr
```

---

## Usage and CLI Orchestrator (`run.py`)

All scripts in the project can be managed through a single interactive menu in the root `run.py`:

```bash
py -3.11 run.py
```

### Menu Options:

1. `extractor`: Extracts terms from PDF (Async + Page Split-and-Retry + PaddleOCR)
2. `enricher`: Performs grammatical/semantic enrichment (Async + Split-and-Retry)
3. `translator`: Translates meanings to target languages (Async + Split-and-Retry)
4. `analyze_logs`: Analyzes provider and model performance, latency, and success rates
5. `clean`: Performs normalization and merging in the merged folder (orjson)
6. `merge`: Merges translations in the translated folder into the final database (orjson)
7. `clean_meanings`: Splits comma-separated meaning lists and deduplicates (orjson)
8. `test_redis`: Verifies connection to the Redis server
9. `quit`: Exits the menu

---

## Model & Provider Performance Analysis (`scripts/analyze_logs.py`)

Analyzes event logs generated by the fallback chain (`handlers/fallback_chain.py`):

- **General Metrics:** Total log files, total API calls, success/error rates, global success percentage, and average latency.
- **Model Rankings:** A detailed table for each model containing success rates, total requests, error counts, rate-limit frequency, and average latency.
- **Provider Summary:** Success rate and latency comparison across API providers (Google, OpenRouter, Cerebras, Groq, etc.).
- **Highlights:** Best performing model, fastest model (minimum latency), slowest model, and most unreliable model (frequent rate-limits/errors).
- **Never Ran Models:** List of configured models that were never run or logged.

---

## Progress and Log Architecture

### Progress Directory

Progress files are maintained under `data/progress/`:

- `data/progress/extractor/` → Extractor page-level progress
- `data/progress/translator/` → Translator batch/item progress
- `data/progress/clean/` → Clean script run progress

### Log Directory

Structured execution logs are stored under `data/logs/`:

- `<run_id>.jsonl` → Event stream (INFO/WARN/ERROR)
- `<run_id>_summary.json` → Run performance summary
