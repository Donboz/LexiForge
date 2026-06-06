# LexiForge Developer Guidelines (AGENTS.md)

This file contains instructions for AI coding agents and developers working on the LexiForge project.

## Commands

- **Run Interactive Menu:** `py -3.11 run.py` (or run `run.bat` configured to use Python 3.11)
- **Extract Terms (OCR + LLM):** `py -3.11 scripts/extractor.py`
- **Enrich Dictionary Entries (LLM):** `py -3.11 scripts/enricher.py`
- **Translate Dictionary (LLM):** `py -3.11 scripts/translator.py`
- **Clean Dictionary Data:** `py -3.11 scripts/clean.py`
- **Deduplicate Meanings:** `py -3.11 scripts/clean_meanings.py`
- **Merge Translated Output:** `py -3.11 scripts/merge_translated.py`
- **Test Redis Connection:** Choose Option 9 in the interactive menu

---

## Codebase Structure

- `/config/config.json`: Master configuration (models, API keys, languages, mapped PDFs)
- `/handlers/`: Core middleware:
  - `fallback_chain.py`: Resilient fallback chain for LLM calls with split-and-retry
  - `ocr_handler.py`: PaddleOCR engine with column heuristics
  - `run_logger.py`: JSONL logging and run tracking
  - `selector.py`: Interactive bilingual arrow menus with file completion locking
- `/scripts/`: Execution pipelines
- `/data/`: File outputs, progress trackers, and logs

---

## Style Guidelines

- **Asynchronous Code:** Prefer async loops (`asyncio`) wrapped in `tqdm.asyncio` for high performance. Use `asyncio.Semaphore` to manage concurrency limits.
- **JSON Processing:** Use `orjson` (binary mode) instead of standard `json` to read and write large JSON databases without blocking CPU threads.
- **Surgical Changes:** Modify only required logic. Respect comments, styling, and bilingual prints.
- **OCR Fallback:** Use PaddleOCR locally via `ocr_handler.py` as default. Keep PyPDF2 text extraction as a fallback or for custom setups.
