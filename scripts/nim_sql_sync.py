#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LexiForge NVIDIA NIM -> Glossa PostgreSQL Manual Audit, Purge & Sync Tool
========================================================================
Can be run manually at any time from either /root/LexiForge or /root/Glossa:
- Dynamically fetches & probes working models from NVIDIA NIM (/v1/models).
- Scans Glossa PostgreSQL, exported PDF, or LexiForge JSON using LexiForge's
  12-language lexical_filter engine:
    1. Identifies & purges non-word junk ("11th", "11.", "1st".."1000th", "1337",
       PDF headers "136brodeln –B ruc h", bare alphabet headers) with full JSON rollback backup.
    2. Repairs fixable OCR/digit terms ("107Cam elması" -> "Cam elması",
       "(0. PI.) Ehrenhaftigkeit" -> "Ehrenhaftigkeit"), self-echo meanings (term==meaning),
       missing POS/levels/domains/examples via rotating NVIDIA NIM keys & models.
    3. Applies updates transactionally to PostgreSQL with @@unique([text, lang, partOfSpeech]) merge safety.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time
from typing import Any
import uuid
import zlib

import psycopg2
from psycopg2.extras import Json, RealDictCursor
import requests

LEXIFORGE_ROOT = Path(__file__).resolve().parent.parent
if str(LEXIFORGE_ROOT) not in sys.path:
    sys.path.insert(0, str(LEXIFORGE_ROOT))

from handlers.lexical_filter import (
    SUPPORTED_LANGS,
    VALID_DOMAIN,
    VALID_GENDER,
    VALID_LANGS,
    VALID_LEVEL,
    VALID_POS,
    classify_headword,
    is_junk_meaning,
    is_protected_headword,
    normalize_domain,
    sanitize_article_and_gender,
)

NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
NON_CHAT_KEYWORDS = (
    "embed", "rerank", "bge", "nvclip", "guard", "shield", "parse", "ocr",
    "asr", "tts", "whisper", "riva", "flux", "sdxl", "cosmos", "vila",
    "streampetr", "paligemma", "deplot", "neva", "kosmos", "reward",
    "steerlm", "canary", "parakeet", "retriever", "translate", "safety", "moderation",
)
PREFERRED_MODELS = [
    "nvidia/nemotron-3.5-bolt-e4b",
    "nvidia/nemotron-3.5-lightning-30b-a3b",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
    "poolside/laguna-xs-2.1",
    "nvidia/nemotron-3-ultra-550b-a55b",
    "deepseek-ai/deepseek-v4.1-flash",
    "google/gemma-4-31b-it",
    "openai/gpt-oss-20b",
    "meta/llama-3.3-70b-instruct",
    "qwen/qwen2.5-72b-instruct",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_env_values(env_path: Path | None) -> tuple[str, list[str]]:
    candidates: list[Path] = []
    if env_path:
        candidates.append(env_path)
    candidates.extend([
        Path("/root/Glossa/.env"),
        LEXIFORGE_ROOT / ".env",
        Path.cwd() / ".env",
    ])

    parsed: dict[str, str] = {}
    for path in candidates:
        if path.exists():
            for raw_line in path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k, v = k.strip(), v.strip()
                if len(v) >= 2 and ((v[0] == v[-1] == '"') or (v[0] == v[-1] == "'")):
                    v = v[1:-1]
                parsed.setdefault(k, v)

    db_url = os.environ.get("DATABASE_URL") or parsed.get("DATABASE_URL", "")
    raw_keys = os.environ.get("NVIDIA_API_KEYS") or parsed.get("NVIDIA_API_KEYS") or os.environ.get("NVIDIA_API_KEY") or parsed.get("NVIDIA_API_KEY", "")
    keys: list[str] = []
    seen = set()
    for part in str(raw_keys).split(","):
        k = part.strip()
        if k and k not in seen:
            seen.add(k)
            keys.append(k)

    if not db_url:
        raise RuntimeError("DATABASE_URL not found in --env-file, /root/Glossa/.env, or environment.")
    return db_url, keys


def extract_json_payload(raw_text: str) -> Any:
    cleaned = re.sub(r"<think>.*?</think>", "", str(raw_text or ""), flags=re.DOTALL | re.IGNORECASE).strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        cleaned = fence.group(1).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        arr_s, arr_e = cleaned.find("["), cleaned.rfind("]")
        obj_s, obj_e = cleaned.find("{"), cleaned.rfind("}")
        for snippet in ([cleaned[arr_s : arr_e + 1]] if arr_s != -1 and arr_e > arr_s else []) + (
            [cleaned[obj_s : obj_e + 1]] if obj_s != -1 and obj_e > obj_s else []
        ):
            try:
                return json.loads(snippet)
            except json.JSONDecodeError:
                pass
        raise ValueError(f"Invalid JSON from LLM: {cleaned[:180]!r}")


class NimPool:
    def __init__(self, api_keys: list[str], timeout: float = 55.0):
        if not api_keys:
            raise RuntimeError("NVIDIA_API_KEYS is missing.")
        self.api_keys = api_keys
        self.timeout = timeout
        self._lock = threading.Lock()
        self._key_idx = 0
        self._key_cooldowns: dict[int, float] = {}
        self.models: list[dict[str, Any]] = []

    def fetch_and_probe(self) -> list[dict[str, Any]]:
        catalog: list[str] = []
        last_err = None
        for key in self.api_keys:
            try:
                r = requests.get(f"{NIM_BASE_URL}/models", headers={"Authorization": f"Bearer {key}"}, timeout=15)
                if r.status_code == 200:
                    catalog = sorted({m["id"] for m in r.json().get("data", []) if isinstance(m, dict) and m.get("id")})
                    break
                last_err = f"HTTP {r.status_code}"
            except Exception as exc:
                last_err = str(exc)
        if not catalog:
            raise RuntimeError(f"Failed /v1/models: {last_err}")

        cset = set(catalog)
        candidates = [m for m in PREFERRED_MODELS if m in cset]
        for m in catalog:
            low = m.lower()
            if m not in candidates and not any(b in low for b in NON_CHAT_KEYWORDS):
                if any(h in low for h in ("flash", "bolt", "lightning", "mini", "8b", "20b", "30b", "31b", "70b", "instruct")):
                    candidates.append(m)
            if len(candidates) >= 15:
                break

        def probe_one(mid: str, idx: int) -> dict[str, Any]:
            key = self.api_keys[idx % len(self.api_keys)]
            for use_kw in (True, False):
                payload: dict[str, Any] = {
                    "model": mid,
                    "messages": [{"role": "user", "content": 'Return ONLY {"ok":true}'}],
                    "temperature": 0.0,
                    "max_tokens": 24,
                }
                if use_kw:
                    payload["chat_template_kwargs"] = {"thinking": False, "enable_thinking": False}
                t0 = time.perf_counter()
                try:
                    resp = requests.post(
                        f"{NIM_BASE_URL}/chat/completions",
                        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                        json=payload,
                        timeout=14,
                    )
                    ms = round((time.perf_counter() - t0) * 1000.0, 1)
                    if resp.status_code == 200:
                        txt = resp.json().get("choices", [{}])[0].get("message", {}).get("content", "")
                        if txt and str(txt).strip():
                            return {"model": mid, "ok": True, "latency_ms": ms, "thinking_kw": use_kw, "status": "ok", "cooldown_until": 0.0}
                        return {"model": mid, "ok": False, "latency_ms": ms, "thinking_kw": use_kw, "status": "empty", "cooldown_until": 0.0}
                    if resp.status_code in (400, 422) and use_kw:
                        continue
                    return {"model": mid, "ok": False, "latency_ms": ms, "thinking_kw": use_kw, "status": f"http_{resp.status_code}", "cooldown_until": 0.0}
                except Exception as exc:
                    return {"model": mid, "ok": False, "latency_ms": 14000.0, "thinking_kw": use_kw, "status": type(exc).__name__, "cooldown_until": 0.0}
            return {"model": mid, "ok": False, "latency_ms": 0.0, "thinking_kw": False, "status": "fail", "cooldown_until": 0.0}

        results: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=6) as pool:
            futs = [pool.submit(probe_one, mid, i) for i, mid in enumerate(candidates)]
            for f in as_completed(futs):
                results.append(f.result())

        results.sort(key=lambda r: (not r["ok"], r["latency_ms"]))
        self.models = [r for r in results if r["ok"]]
        if not self.models:
            raise RuntimeError("Zero NVIDIA NIM models passed live probing.")
        return results

    def _next_key(self) -> int:
        with self._lock:
            now = time.time()
            for off in range(len(self.api_keys)):
                idx = (self._key_idx + off) % len(self.api_keys)
                if self._key_cooldowns.get(idx, 0.0) <= now:
                    self._key_idx = (idx + 1) % len(self.api_keys)
                    return idx
            idx = self._key_idx
            self._key_idx = (self._key_idx + 1) % len(self.api_keys)
            return idx

    def chat_json(self, system_prompt: str, user_prompt: str, max_tokens: int = 4200) -> tuple[Any, str]:
        with self._lock:
            ordered = sorted(self.models, key=lambda m: (m["cooldown_until"] > time.time(), m["latency_ms"]))
        attempts = 0
        last_err: Exception | None = None

        for m in ordered:
            for _ in range(len(self.api_keys)):
                attempts += 1
                kidx = self._next_key()
                key = self.api_keys[kidx]
                payload: dict[str, Any] = {
                    "model": m["model"],
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "max_tokens": max_tokens,
                }
                if m["thinking_kw"]:
                    payload["chat_template_kwargs"] = {"thinking": False, "enable_thinking": False}
                try:
                    resp = requests.post(
                        f"{NIM_BASE_URL}/chat/completions",
                        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                        json=payload,
                        timeout=self.timeout,
                    )
                    if resp.status_code == 200:
                        content = resp.json().get("choices", [{}])[0].get("message", {}).get("content", "")
                        return extract_json_payload(content), m["model"]
                    if resp.status_code == 429:
                        with self._lock:
                            self._key_cooldowns[kidx] = time.time() + 15.0
                        last_err = RuntimeError(f"429 on key#{kidx+1} ({m['model']})")
                        continue
                    with self._lock:
                        m["cooldown_until"] = time.time() + 10.0
                    last_err = RuntimeError(f"HTTP {resp.status_code} ({m['model']}): {resp.text[:160]}")
                except Exception as exc:
                    with self._lock:
                        m["cooldown_until"] = time.time() + 5.0
                    last_err = exc
                    time.sleep(min(1.5, 0.3 * attempts))
        raise RuntimeError(f"All NIM calls failed: {last_err}")


def extract_pdf_headword_candidates(pdf_path: Path, max_pages: int = 0) -> set[str]:
    """Extract suspicious numeric/ordinal/header headwords from an exported PDF."""
    tokens: set[str] = set()
    text_chunks: list[str] = []

    if shutil.which("pdftotext"):
        cmd = ["pdftotext", str(pdf_path), "-"]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if res.returncode == 0 and res.stdout:
            text_chunks.append(res.stdout)
    else:
        try:
            import pypdf  # type: ignore
            reader = pypdf.PdfReader(str(pdf_path))
            pages = reader.pages[:max_pages] if max_pages > 0 else reader.pages
            for p in pages:
                text_chunks.append(p.extract_text() or "")
        except ImportError:
            # Pure-Python compressed stream fallback for PDFKit / report PDFs
            raw = pdf_path.read_bytes()
            for match in re.finditer(rb"stream\r?\n(.*?)\r?\nendstream", raw, flags=re.DOTALL):
                stream = match.group(1)
                try:
                    dec = zlib.decompress(stream).decode("latin-1", errors="ignore")
                except Exception:
                    continue
                for lit in re.findall(r"\(([^()]{1,140})\)", dec):
                    if re.search(r"\d", lit) or len(lit.strip()) <= 2:
                        text_chunks.append(lit)

    for chunk in text_chunks:
        for line in chunk.splitlines():
            s = line.strip()
            if not s or len(s) > 120:
                continue
            head = s.split("—")[0].split(":")[0].strip()
            act, _, _ = classify_headword(head)
            if act in ("delete", "clean"):
                tokens.add(head)
    return tokens


SYSTEM_PROMPT = """You are LexiForge-SQL, an expert 12-language lexicographer (DE, TR, EN, FR, ES, IT, PT, RO, LA, RU, SR, MK).
You receive broken/noisy dictionary entries (from scraped PDFs / Admin Import) and must either FIX or mark as NON_WORD_JUNK.
Return ONLY a JSON array matching each input termId:
[
  {
    "termId": "<exact termId>",
    "action": "FIX" | "DELETE",
    "reason": "<short reason if DELETE or FIX>",
    "text": "<clean headword without page numbers like 107 or (0. PI.)>",
    "lang": "DE"|"TR"|"EN"|"FR"|"ES"|"IT"|"PT"|"RO"|"LA"|"RU"|"SR"|"MK",
    "partOfSpeech": "NOUN"|"VERB"|"ADJECTIVE"|"ADVERB"|"PRONOUN"|"PREPOSITION"|"CONJUNCTION"|"INTERJECTION"|"ARTICLE"|"NUMERAL"|"PARTICLE"|"PHRASE"|"IDIOM"|"PREFIX"|"SUFFIX",
    "gender": "MASCULINE"|"FEMININE"|"NEUTER"|"COMMON"|"AMBIGENERIC"|null,
    "article": "der"|"die"|"das"|"le"|"la"|"el"|"il"|"lo"|"o"|"a"|null,
    "plural": "<plural form or null>",
    "level": "A1"|"A2"|"B1"|"B2"|"C1"|"C2"|"UNKNOWN",
    "primaryDomain": "<exact Prisma Domain enum>",
    "contextTags": ["<2-4 lowercase context tags>"],
    "definitions": [
      {"id": "<exact def id>", "meaning": "<clean accurate translation, NEVER ordinal junk or equal to term>", "language": "<2-letter>", "domain": "<Prisma Domain enum>"}
    ],
    "examples": [
      {"sentence": "<natural sentence>", "translation": "<accurate translation>", "translationLang": "TR"|"EN", "domain": "<Domain enum>", "level": "A1".."C2"}
    ]
  }
]
Rules:
- Mark action="DELETE" ONLY for unreadable fragments with no valid headword (e.g. "11th", "11.", "1st", "1337", page headers "136brodeln –B ruc h").
- NEVER mark action="DELETE" when cleanHint is provided or valid definitions exist: instead set action="FIX" and text=cleanHint (e.g. "150daher" -> text="daher", "(0. PI.) Ehrenhaftigkeit" -> text="Ehrenhaftigkeit").
- NEVER alter or delete valid chemical locants ("1-Hexanol", "1-Pentanol", "1-Propanol", "1,2-Butadien"), dates ("11. September"), times ("1 Uhr", "12 Uhr Mittag/Mitternacht"), or standard terms ("14K", "24/7", "9/11"): keep their exact text!
- If a word has a glued page number like "107Cam elması" -> action="FIX", text="Cam elması", lang="TR", meaning="Glass cutter / Cam elması".
- Never leave ordinal/meaning==term definitions. For TR/EN: article=null, gender=null."""


def save_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload["meta"]["updatedAt"] = utc_now()
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def load_audit(path: Path) -> dict[str, Any]:
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("items"), dict):
            return data
    return {"meta": {"createdAt": utc_now(), "updatedAt": utc_now(), "modelsUsed": []}, "items": {}, "purged": {}}


def purge_junk_row_tx(conn: psycopg2.extensions.connection, term_row: dict[str, Any], reasons: list[str], dry_run: bool) -> str:
    tid = term_row["termId"]
    try:
        with conn.cursor() as cur:
            cur.execute('SELECT id FROM "Term" WHERE id = %s FOR UPDATE', (tid,))
            if not cur.fetchone():
                conn.rollback()
                return "already_deleted"
            cur.execute('UPDATE "Term" SET "parentTermId" = NULL WHERE "parentTermId" = %s', (tid,))
            cur.execute('DELETE FROM "GlossaryEntry" WHERE "termId" = %s', (tid,))
            cur.execute('DELETE FROM "Example" WHERE "termId" = %s', (tid,))
            cur.execute('DELETE FROM "Definition" WHERE "termId" = %s', (tid,))
            cur.execute('DELETE FROM "Term" WHERE id = %s', (tid,))
        if dry_run:
            conn.rollback()
            return f"dry_run_delete:{','.join(reasons)}"
        conn.commit()
        return f"deleted:{','.join(reasons)}"
    except Exception:
        conn.rollback()
        raise


def apply_fix_row_tx(conn: psycopg2.extensions.connection, term_id: str, fix: dict[str, Any], dry_run: bool) -> str:
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute('SELECT id, lang, "partOfSpeech"::text AS "partOfSpeech", level::text AS level FROM "Term" WHERE id = %s FOR UPDATE', (term_id,))
            curr = cur.fetchone()
            if not curr:
                conn.rollback()
                return "already_missing"

            lang = str(fix.get("lang") or curr["lang"]).upper()
            if lang not in VALID_LANGS:
                lang = curr["lang"]
            pos = (fix.get("partOfSpeech") or curr["partOfSpeech"])
            if pos:
                pos = str(pos).upper()
            if pos not in VALID_POS:
                pos = curr["partOfSpeech"]

            text = re.sub(r"\s+", " ", str(fix.get("text") or "")).strip()
            act, cleaned_text, _ = classify_headword(text, lang=lang, pos=pos)
            if act == "delete" or not cleaned_text:
                conn.rollback()
                return "skipped_invalid_target_text"
            text = cleaned_text

            san_obj = sanitize_article_and_gender({
                "lang": lang,
                "partOfSpeech": pos,
                "article": fix.get("article"),
                "gender": fix.get("gender"),
            }, src_lang=lang)

            level = str(fix.get("level") or curr["level"] or "UNKNOWN").upper()
            if level not in VALID_LEVEL:
                level = curr["level"] or "UNKNOWN"

            primary_domain = normalize_domain(fix.get("primaryDomain"), "GENERAL")
            context_tags = [str(t).strip().lower()[:48] for t in (fix.get("contextTags") or []) if str(t).strip()][:6]
            plural = str(fix.get("plural")).strip()[:120] if fix.get("plural") else None
            edit_note = f"lexiforge-nim: {str(fix.get('reason') or 'repaired')[:180]}"

            collision = None
            if pos:
                cur.execute(
                    'SELECT id FROM "Term" WHERE text = %s AND lang = %s AND "partOfSpeech" = %s::"PartOfSpeech" AND id <> %s FOR UPDATE',
                    (text, lang, pos, term_id),
                )
                collision = cur.fetchone()

            kept_def_ids: list[str] = []
            for d in fix.get("definitions") or []:
                did = d.get("id")
                meaning = re.sub(r"\s+", " ", str(d.get("meaning") or "")).strip()
                if not did or is_junk_meaning(meaning, text):
                    continue
                d_lang = str(d.get("language") or "TR").upper()
                if d_lang not in VALID_LANGS:
                    d_lang = "TR"
                dom = normalize_domain(d.get("domain"), primary_domain)
                if primary_domain == "GENERAL" and dom != "GENERAL":
                    primary_domain = dom
                cur.execute(
                    """
                    UPDATE "Definition"
                    SET meaning = %s, language = %s, domain = %s::"Domain",
                        "contextTags" = %s::text[], "editNote" = %s, "updatedAt" = NOW()
                    WHERE id = %s AND "termId" = %s
                    """,
                    (meaning, d_lang, dom, context_tags, edit_note, did, term_id),
                )
                if cur.rowcount > 0:
                    kept_def_ids.append(did)

            target_id = term_id
            if collision:
                target_id = collision["id"]
                cur.execute('UPDATE "Definition" SET "termId" = %s, "updatedAt" = NOW() WHERE "termId" = %s', (target_id, term_id))
                cur.execute('UPDATE "Example" SET "termId" = %s WHERE "termId" = %s', (target_id, term_id))
                cur.execute('UPDATE "GlossaryEntry" SET "termId" = %s WHERE "termId" = %s', (target_id, term_id))
                cur.execute('UPDATE "Term" SET "parentTermId" = %s WHERE "parentTermId" = %s AND id <> %s', (target_id, term_id, target_id))
                cur.execute('UPDATE "Term" SET "parentTermId" = NULL WHERE "parentTermId" = %s AND id = %s', (term_id, target_id))
                cur.execute('DELETE FROM "Term" WHERE id = %s', (term_id,))
                action = f"merged_into:{target_id}"
            else:
                action = "updated"

            cur.execute(
                """
                UPDATE "Term"
                SET text = %s, lang = %s, "partOfSpeech" = %s::"PartOfSpeech",
                    gender = %s::"Gender", article = %s, plural = %s,
                    level = %s::"LanguageLevel", "primaryDomain" = %s::"Domain",
                    "contextTags" = %s::text[], "qualityScore" = 85,
                    "enrichedAt" = NOW(), "editNote" = %s, "updatedAt" = NOW()
                WHERE id = %s
                """,
                (
                    text, lang, pos, san_obj.get("gender"), san_obj.get("article"), plural,
                    level, primary_domain, context_tags, edit_note, target_id,
                ),
            )

            first_def_id = kept_def_ids[0] if kept_def_ids else None
            for ex in (fix.get("examples") or [])[:2]:
                sent = re.sub(r"\s+", " ", str(ex.get("sentence") or "")).strip()
                tr = re.sub(r"\s+", " ", str(ex.get("translation") or "")).strip()
                if not sent or not tr or sent.lower() == tr.lower():
                    continue
                cur.execute('SELECT 1 FROM "Example" WHERE "termId" = %s AND lower(trim(sentence)) = lower(trim(%s)) LIMIT 1', (target_id, sent))
                if cur.fetchone():
                    continue
                tlang = str(ex.get("translationLang") or ("TR" if lang != "TR" else "EN")).upper()
                if tlang not in VALID_LANGS:
                    tlang = "TR"
                cur.execute(
                    """
                    INSERT INTO "Example" (id, "termId", "definitionId", sentence, translation, "translationLang",
                        domain, level, "sourceLang", "isAiGenerated", source, "createdAt")
                    VALUES (%s, %s, %s, %s, %s, %s, %s::"Domain", %s::"LanguageLevel", %s, true, 'lexiforge-nim', NOW())
                    """,
                    (str(uuid.uuid4()), target_id, first_def_id, sent, tr, tlang, primary_domain, level, lang),
                )

        if dry_run:
            conn.rollback()
            return f"dry_run_{action}"
        conn.commit()
        return action
    except Exception:
        conn.rollback()
        raise


def fetch_suspicious_db_terms(conn: psycopg2.extensions.connection, limit: int = 0) -> list[dict[str, Any]]:
    sql = """
        SELECT t.id AS "termId", t.text, t.lang, t."partOfSpeech"::text AS "partOfSpeech",
               t.gender::text AS gender, t.article, t.plural, t.level::text AS level,
               t.source, t."primaryDomain"::text AS "primaryDomain"
        FROM "Term" t
        WHERE t.text ~ '[0-9]'
           OR length(trim(t.text)) <= 1
           OR t.text ~ '^[\\(\\[\\{]|^[–—•*.,;:]'
           OR t.text ~* '^(seite|page|sayfa)\\s*[0-9]+'
        ORDER BY t.text ASC
    """
    if limit and limit > 0:
        sql += " LIMIT %s"
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(sql, (limit,) if (limit and limit > 0) else ())
        rows = [dict(r) for r in cur.fetchall()]
        if not rows:
            return []
        tids = [r["termId"] for r in rows]
        cur.execute(
            'SELECT "termId", id, meaning, language, domain::text AS domain FROM "Definition" WHERE "termId" = ANY(%s)',
            (tids,),
        )
        defs_map: dict[str, list[dict[str, Any]]] = {tid: [] for tid in tids}
        for d in cur.fetchall():
            defs_map[d["termId"]].append({
                "id": d["id"], "meaning": d["meaning"], "language": d["language"], "domain": d["domain"]
            })
        for r in rows:
            r["definitions"] = defs_map.get(r["termId"], [])
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="LexiForge NVIDIA NIM -> Glossa PostgreSQL Manual Audit & Repair CLI")
    parser.add_argument("--fetch-models", action="store_true", help="Fetch & live-probe NVIDIA NIM /v1/models and exit.")
    parser.add_argument("--source", choices=["db", "pdf", "json"], default="db", help="Input audit source (default: db).")
    parser.add_argument("--pdf-path", default="/root/Glossa/public/downloads/Glossa_Buyuk_Sozluk.pdf", help="PDF path when --source pdf.")
    parser.add_argument("--input", help="JSON path when --source json.")
    parser.add_argument("--env-file", help="Path to .env containing DATABASE_URL and NVIDIA_API_KEYS.")
    parser.add_argument("--purge-junk", action="store_true", help="Delete verified non-lexical junk (ordinals 11th/11., PDF headers) with JSON backup.")
    parser.add_argument("--limit", type=int, default=50, help="Max NIM repair terms (0 = unlimited). Default: 50.")
    parser.add_argument("--batch-size", type=int, default=5, help="Terms per NVIDIA NIM request. Default: 5.")
    parser.add_argument("--workers", type=int, default=2, help="Concurrent NIM workers. Default: 2.")
    parser.add_argument("--audit-file", default="logs/lexiforge_nim_sql_audit.json", help="Resumable JSON audit & backup log.")
    parser.add_argument("--generate-only", action="store_true", help="Generate fixes into audit JSON without SQL writes.")
    parser.add_argument("--apply-only", action="store_true", help="Apply pending items from --audit-file to PostgreSQL.")
    parser.add_argument("--dry-run", action="store_true", help="Simulate all SQL deletions/updates with rollback.")
    args = parser.parse_args()

    db_url, api_keys = load_env_values(Path(args.env_file) if args.env_file else None)
    nim = NimPool(api_keys=api_keys)

    if args.fetch_models or not args.apply_only:
        probe_res = nim.fetch_and_probe()
        print("\n=== Probed NVIDIA NIM Models ===", flush=True)
        for r in probe_res:
            print(f"  {r['model']:45s} ok={str(r['ok']):5s} {r['latency_ms']:>7}ms status={r['status']}")
        if args.fetch_models:
            return 0

    audit_path = Path(args.audit_file)
    if not audit_path.is_absolute():
        audit_path = Path.cwd() / audit_path
    audit = load_audit(audit_path)

    conn = psycopg2.connect(db_url)
    conn.autocommit = False

    try:
        if args.source == "json" and args.input and not args.apply_only:
            raw_json = json.loads(Path(args.input).read_text(encoding="utf-8"))
            if isinstance(raw_json, list):
                cleaned_out = []
                skipped = 0
                for entry in raw_json:
                    if not isinstance(entry, dict) or not entry.get("term"):
                        continue
                    act, ctext, _ = classify_headword(entry["term"], lang=entry.get("lang", "DE"), pos=entry.get("partOfSpeech"), definitions=entry.get("definitions"))
                    if act == "delete":
                        skipped += 1
                        continue
                    entry["term"] = ctext
                    sanitize_article_and_gender(entry, src_lang=entry.get("lang", "DE"))
                    cleaned_out.append(entry)
                print(f"[JSON Source] Kept {len(cleaned_out)}, dropped {skipped} junk terms.", flush=True)
            return 0

        if not args.apply_only:
            pdf_filter: set[str] | None = None
            if args.source == "pdf":
                p = Path(args.pdf_path)
                if not p.exists():
                    raise FileNotFoundError(f"PDF not found: {p}")
                print(f"[PDF Scan] Extracting suspicious headwords from {p}...", flush=True)
                pdf_filter = extract_pdf_headword_candidates(p)
                print(f"[PDF Scan] Found {len(pdf_filter)} suspicious PDF headwords.", flush=True)

            rows = fetch_suspicious_db_terms(conn, limit=0)
            if pdf_filter is not None and pdf_filter:
                rows = [r for r in rows if r["text"] in pdf_filter or classify_headword(r["text"], r["lang"], r["partOfSpeech"], r["definitions"])[0] != "keep"]

            delete_candidates: list[tuple[dict[str, Any], list[str]]] = []
            repair_candidates: list[dict[str, Any]] = []

            for r in rows:
                act, cleaned_text, reasons = classify_headword(
                    r["text"], lang=r["lang"], pos=r["partOfSpeech"], definitions=r["definitions"]
                )
                if act == "delete":
                    delete_candidates.append((r, reasons))
                elif act == "clean":
                    r["clean_hint"] = cleaned_text
                    r["clean_reasons"] = reasons
                    if r["termId"] not in audit["items"]:
                        repair_candidates.append(r)
                else:
                    # Still repair if definition has ordinal or term==meaning
                    has_bad_def = any(is_junk_meaning(d.get("meaning", ""), r["text"]) for d in r.get("definitions", []))
                    if has_bad_def and r["termId"] not in audit["items"]:
                        repair_candidates.append(r)

            print(f"[Audit Scan] Total suspicious={len(rows)} | Auto-Delete Junk={len(delete_candidates)} | NIM Repair Candidates={len(repair_candidates)}", flush=True)

            if args.purge_junk:
                purged_count = 0
                for row, reasons in delete_candidates:
                    tid = row["termId"]
                    res = purge_junk_row_tx(conn, row, reasons, dry_run=args.dry_run or args.generate_only)
                    if args.dry_run or args.generate_only:
                        print(f"  [DRY-PURGE] {row['text']!r} ({row['lang']}) -> {reasons}")
                    else:
                        audit["purged"][tid] = {"snapshot": row, "reasons": reasons, "action": res, "purgedAt": utc_now()}
                        purged_count += 1
                if not (args.dry_run or args.generate_only) and purged_count:
                    save_json_atomic(audit_path, audit)
                print(f"[Purge] Processed {len(delete_candidates)} junk rows (deleted={purged_count}, dry_run={args.dry_run or args.generate_only}).", flush=True)

            if args.limit > 0:
                repair_candidates = repair_candidates[: args.limit]

            if repair_candidates and args.dry_run and args.limit == 0:
                print(
                    f"[DRY-RUN] Previewing first 10 of {len(repair_candidates)} NIM repair candidates "
                    f"(pass --limit N to run live NIM stage on N entries):",
                    flush=True,
                )
                for sample in repair_candidates[:10]:
                    print(
                        f"  [DRY-REPAIR] {sample['text']!r} ({sample['lang']}) "
                        f"hint={sample.get('clean_hint')!r} reasons={sample.get('clean_reasons') or []}"
                    )
                repair_candidates = []

            if repair_candidates:
                batches = [repair_candidates[i : i + max(1, args.batch_size)] for i in range(0, len(repair_candidates), max(1, args.batch_size))]
                lock = threading.Lock()
                models_used = set(audit["meta"].get("modelsUsed") or [])

                def run_batch(bidx: int, batch: list[dict[str, Any]]) -> None:
                    payload = [
                        {
                            "termId": r["termId"],
                            "text": r["text"],
                            "cleanHint": r.get("clean_hint"),
                            "lang": r["lang"],
                            "partOfSpeech": r["partOfSpeech"],
                            "gender": r["gender"],
                            "article": r["article"],
                            "level": r["level"],
                            "primaryDomain": r.get("primaryDomain"),
                            "definitions": r.get("definitions", [])[:5],
                        }
                        for r in batch
                    ]
                    user_prompt = "Audit and fix each entry below. Return ONLY a JSON array:\n" + json.dumps(payload, ensure_ascii=False)
                    parsed, model_id = nim.chat_json(SYSTEM_PROMPT, user_prompt)
                    if isinstance(parsed, dict):
                        parsed = parsed.get("items") or parsed.get("results") or [parsed]
                    by_id = {str(x["termId"]): x for x in parsed if isinstance(x, dict) and x.get("termId")}

                    with lock:
                        models_used.add(model_id)
                        for r in batch:
                            tid = r["termId"]
                            fix = by_id.get(tid)
                            if fix:
                                audit["items"][tid] = {
                                    "termId": tid,
                                    "applied": False,
                                    "original": r,
                                    "proposed": fix,
                                    "model": model_id,
                                    "generatedAt": utc_now(),
                                }
                        audit["meta"]["modelsUsed"] = sorted(models_used)
                        if not args.dry_run:
                            save_json_atomic(audit_path, audit)
                        print(f"[NIM Stage 1] Batch {bidx + 1}/{len(batches)} complete via {model_id} (audit={len(audit['items'])})", flush=True)

                with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
                    for fut in as_completed([pool.submit(run_batch, i, b) for i, b in enumerate(batches)]):
                        fut.result()

        if not args.generate_only:
            pending = [(tid, item) for tid, item in audit["items"].items() if not item.get("applied")]
            print(f"[SQL Stage 2] Applying {len(pending)} fixes (dry_run={args.dry_run})...", flush=True)
            stats = {"updated": 0, "merged": 0, "deleted": 0, "dry_run": 0, "failed": 0}
            for tid, item in pending:
                orig = item.get("original") or {}
                prop = dict(item.get("proposed") or {})
                orig_text = orig.get("text") or ""
                clean_hint = orig.get("clean_hint")
                # Guard: never allow LLM to delete valid protected headwords or entries with a clean_hint
                if str(prop.get("action") or "FIX").upper() == "DELETE" and (
                    clean_hint or is_protected_headword(orig_text) or len(orig.get("definitions") or []) >= 1
                ):
                    prop["action"] = "FIX"
                    prop["text"] = clean_hint or orig_text
                    prop["reason"] = f"guarded_fix_from_delete: {prop.get('reason') or ''}".strip()
                if is_protected_headword(orig_text):
                    prop["text"] = orig_text
                elif clean_hint and not prop.get("text"):
                    prop["text"] = clean_hint
                try:
                    if str(prop.get("action") or "FIX").upper() == "DELETE":
                        res = purge_junk_row_tx(conn, item["original"], [prop.get("reason") or "llm_flagged_junk"], dry_run=args.dry_run)
                        if args.dry_run:
                            stats["dry_run"] += 1
                        else:
                            stats["deleted"] += 1
                            item["applied"] = True
                            item["applyAction"] = res
                    else:
                        res = apply_fix_row_tx(conn, tid, prop, dry_run=args.dry_run)
                        if args.dry_run:
                            stats["dry_run"] += 1
                        else:
                            item["applied"] = True
                            item["applyAction"] = res
                            if "merged_into" in res:
                                stats["merged"] += 1
                            else:
                                stats["updated"] += 1
                    if not args.dry_run:
                        item["appliedAt"] = utc_now()
                        save_json_atomic(audit_path, audit)
                except Exception as exc:
                    stats["failed"] += 1
                    item["lastError"] = str(exc)[:240]
                    if not args.dry_run:
                        save_json_atomic(audit_path, audit)
                    print(f"  [ERROR] {tid}: {exc}", flush=True)
            print(f"[SQL Stage 2] Summary: {stats}", flush=True)
            if stats["failed"] > 0:
                return 1
    finally:
        conn.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
