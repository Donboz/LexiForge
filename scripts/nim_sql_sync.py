#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LexiForge NVIDIA NIM -> Glossa PostgreSQL Audit, Repair, PDF/JSON Sync Tool
===========================================================================
Supports:
1. 12-language (DE, TR, EN, FR, ES, IT, PT, RO, LA, RU, SR, MK) language-accuracy
   and full Prisma grammar validation (--mode junk | anomalies | all | pairs).
2. True multi-client (API key x live NIM model) parallel worker pool (--workers 10).
3. Full UTF-8 PDF parsing & UPSERT sync from Glossa_Buyuk_Sozluk.pdf (--source pdf),
   JSON dictionary audit & UPSERT sync (--source json), and live PostgreSQL sync (--source db).
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import queue
import random
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from typing import Any
import uuid
import zlib

import psycopg2
from psycopg2 import errors as pg_errors
from psycopg2.extras import Json, RealDictCursor
import requests

LEXIFORGE_ROOT = Path(__file__).resolve().parent.parent
if str(LEXIFORGE_ROOT) not in sys.path:
    sys.path.insert(0, str(LEXIFORGE_ROOT))

from handlers.lexical_filter import (
    LANG_DISPLAY_TO_CODE,
    LANGUAGE_ARTICLE_RULES,
    POS_LABEL_TO_ENUM,
    SUPPORTED_LANGS,
    VALID_LANGS,
    VALID_LEVEL,
    VALID_POS,
    VALID_REGISTER,
    classify_headword,
    detect_language_anomaly,
    is_junk_meaning,
    is_protected_headword,
    normalize_domain,
    sanitize_grammar_fields,
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
    raw_keys = (
        os.environ.get("NVIDIA_API_KEYS")
        or parsed.get("NVIDIA_API_KEYS")
        or os.environ.get("NVIDIA_API_KEY")
        or parsed.get("NVIDIA_API_KEY", "")
    )
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
    """Thread-safe NVIDIA NIM pool distributing concurrency across (key x live model) channels."""

    def __init__(self, api_keys: list[str], timeout: float = 55.0):
        if not api_keys:
            raise RuntimeError("NVIDIA_API_KEYS is missing.")
        self.api_keys = list(api_keys)
        self.timeout = timeout
        self._lock = threading.Lock()
        self._local = threading.local()
        self._chan_idx = 0
        self.models: list[dict[str, Any]] = []
        self.channels: list[dict[str, Any]] = []

    def _session(self) -> requests.Session:
        sess = getattr(self._local, "session", None)
        if sess is None:
            sess = requests.Session()
            self._local.session = sess
        return sess

    def _build_channels(self) -> None:
        channels: list[dict[str, Any]] = []
        n_keys = len(self.api_keys)
        n_models = len(self.models)
        if not n_keys or not n_models:
            self.channels = []
            return
        for k_off in range(n_keys):
            for m_idx, m in enumerate(self.models):
                kidx = (m_idx + k_off) % n_keys
                channels.append({
                    "id": f"{m['model']}#k{kidx + 1}",
                    "model": m["model"],
                    "key": self.api_keys[kidx],
                    "key_idx": kidx,
                    "thinking_kw": m.get("thinking_kw", False),
                    "latency_ms": m.get("latency_ms", 1000.0),
                    "cooldown_until": 0.0,
                    "ok": 0,
                    "err": 0,
                    "rl429": 0,
                    "total_ms": 0.0,
                })
        self.channels = channels

    def pick_channel_order(self) -> list[dict[str, Any]]:
        """Return all (key x model) channels rotated round-robin, putting non-cooled channels first."""
        with self._lock:
            n = len(self.channels)
            if not n:
                return []
            start = self._chan_idx
            self._chan_idx = (self._chan_idx + 1) % n
            rotated = [self.channels[(start + i) % n] for i in range(n)]
            now = time.time()
            active = [c for c in rotated if c["cooldown_until"] <= now]
            cooled = [c for c in rotated if c["cooldown_until"] > now]
            return active + cooled

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
            if len(candidates) >= 20:
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
                        timeout=10,
                    )
                    ms = round((time.perf_counter() - t0) * 1000.0, 1)
                    if resp.status_code == 200:
                        txt = resp.json().get("choices", [{}])[0].get("message", {}).get("content", "")
                        if txt and str(txt).strip():
                            return {"model": mid, "ok": True, "latency_ms": ms, "thinking_kw": use_kw, "status": "ok"}
                        return {"model": mid, "ok": False, "latency_ms": ms, "thinking_kw": use_kw, "status": "empty"}
                    if resp.status_code in (400, 422) and use_kw:
                        continue
                    return {"model": mid, "ok": False, "latency_ms": ms, "thinking_kw": use_kw, "status": f"http_{resp.status_code}"}
                except Exception as exc:
                    return {"model": mid, "ok": False, "latency_ms": 10000.0, "thinking_kw": use_kw, "status": type(exc).__name__}
            return {"model": mid, "ok": False, "latency_ms": 0.0, "thinking_kw": False, "status": "fail"}

        results: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=10) as pool:
            futs = [pool.submit(probe_one, mid, i) for i, mid in enumerate(candidates)]
            for f in as_completed(futs):
                results.append(f.result())

        results.sort(key=lambda r: (not r["ok"], r["latency_ms"]))
        self.models = [r for r in results if r["ok"]]
        if not self.models:
            raise RuntimeError("Zero NVIDIA NIM models passed live probing.")
        self._build_channels()
        return results

    def chat_json(self, system_prompt: str, user_prompt: str, max_tokens: int = 4500) -> tuple[Any, str]:
        ordered = self.pick_channel_order()
        attempts = 0
        last_err: Exception | None = None
        sess = self._session()

        for ch in ordered:
            attempts += 1
            payload: dict[str, Any] = {
                "model": ch["model"],
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.1,
                "top_p": 0.9,
                "max_tokens": max_tokens,
            }
            if ch["thinking_kw"]:
                payload["chat_template_kwargs"] = {"thinking": False, "enable_thinking": False}
            t0 = time.perf_counter()
            try:
                resp = sess.post(
                    f"{NIM_BASE_URL}/chat/completions",
                    headers={"Authorization": f"Bearer {ch['key']}", "Content-Type": "application/json"},
                    json=payload,
                    timeout=self.timeout,
                )
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                if resp.status_code == 200:
                    content = resp.json().get("choices", [{}])[0].get("message", {}).get("content", "")
                    parsed = extract_json_payload(content)
                    if not parsed:
                        raise ValueError("Empty parsed JSON payload")
                    with self._lock:
                        ch["ok"] += 1
                        ch["total_ms"] += elapsed_ms
                    return parsed, ch["id"]
                if resp.status_code == 429:
                    with self._lock:
                        ch["rl429"] += 1
                        ch["cooldown_until"] = time.time() + 15.0
                    last_err = RuntimeError(f"429 on {ch['id']}")
                    continue
                with self._lock:
                    ch["err"] += 1
                    ch["cooldown_until"] = time.time() + 10.0
                last_err = RuntimeError(f"HTTP {resp.status_code} ({ch['id']}): {resp.text[:160]}")
            except Exception as exc:
                with self._lock:
                    ch["err"] += 1
                    ch["cooldown_until"] = time.time() + 6.0
                last_err = exc
                time.sleep(min(1.0, 0.2 * attempts))
        raise RuntimeError(f"All NIM channels failed: {last_err}")

    def format_channel_stats(self) -> str:
        lines = ["=== Multi-Client Pool Stats (Key x Model) ==="]
        active_ch = 0
        for ch in self.channels:
            if ch["ok"] == 0 and ch["err"] == 0 and ch["rl429"] == 0:
                continue
            active_ch += 1
            avg_ms = round(ch["total_ms"] / ch["ok"], 1) if ch["ok"] else 0.0
            lines.append(
                f"  {ch['id']:48s} ok={ch['ok']:<3d} 429={ch['rl429']:<2d} err={ch['err']:<2d} avg={avg_ms:>7}ms"
            )
        lines.append(f"  Active channels utilized: {active_ch}/{len(self.channels)}")
        return "\n".join(lines)


# ─── UTF-8 PDF Extraction & Entry Parsing ──────────────────────────────────────

_RE_CHAPTER_PAIR = re.compile(
    r"^\s*([A-Za-zÇĞİÖŞÜçğıöşüА-Яа-яЁё]+)\s*(?:→|->)\s*([A-Za-zÇĞİÖŞÜçğıöşüА-Яа-яЁё]+)\s*$"
)
_RE_POS_DOMAIN_LINE = re.compile(
    r"^\s*(?:\(([^)]{1,24})\)\s*)?(?:\[([^\]]+)\])?\s*$"
)
_RE_MEANING_LINE = re.compile(r"^\s*(\d+)\.\s+(.+)$")
_RE_EXAMPLE_LINE = re.compile(r"^\s*Örn:\s*(.+)$")
_RE_LETTER_DIVIDER = re.compile(r"^[A-ZÇĞİÖŞÜß#]$")


def _extract_pdf_text(pdf_path: Path, max_pages: int = 0) -> str:
    if shutil.which("pdftotext"):
        cmd = ["pdftotext", "-raw"]
        if max_pages > 0:
            cmd.extend(["-f", "1", "-l", str(max_pages)])
        cmd.extend([str(pdf_path), "-"])
        res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
        if res.returncode == 0 and res.stdout:
            return res.stdout

    try:
        import pypdf  # type: ignore
        reader = pypdf.PdfReader(str(pdf_path))
        pages = reader.pages[:max_pages] if max_pages > 0 else reader.pages
        return "\f".join((p.extract_text() or "") for p in pages)
    except ImportError:
        pass

    chunks: list[str] = []
    raw = pdf_path.read_bytes()
    for match in re.finditer(rb"stream\r?\n(.*?)\r?\nendstream", raw, flags=re.DOTALL):
        try:
            dec = zlib.decompress(match.group(1)).decode("latin-1", errors="ignore")
            chunks.extend(re.findall(r"\(([^()]{1,180})\)", dec))
        except Exception:
            continue
    return "\n".join(chunks)


def _split_pdf_article(display_term: str, src_lang: str) -> tuple[str | None, str]:
    art_set = LANGUAGE_ARTICLE_RULES.get(src_lang, {}).get("articles") or set()
    if not art_set:
        return None, display_term
    m = re.match(r"^(der|die|das|le|la|les|el|los|las|il|lo|i|gli|o|a|os|as|un)\s+(.+)$", display_term, re.IGNORECASE)
    if m and m.group(1).lower() in art_set:
        return m.group(1).lower(), m.group(2).strip()
    m_apo = re.match(r"^(l')\s*(.+)$", display_term, re.IGNORECASE)
    if m_apo and "l'" in art_set:
        return "l'", m_apo.group(2).strip()
    return None, display_term


def parse_glossa_pdf_text(
    raw_text: str,
    max_per_pair: int = 0,
    lang_pair_filter: tuple[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Parse UTF-8 extracted Glossa_Buyuk_Sozluk PDF text into structured entries grouped by (src, tgt) chapter."""
    lines = raw_text.replace("\f", "\n\x0c\n").splitlines()
    entries: list[dict[str, Any]] = []
    pair_counts: dict[tuple[str, str], int] = defaultdict(int)

    curr_pair: tuple[str, str] | None = None
    in_bolum_header = False
    skip_page_num = False
    curr: dict[str, Any] | None = None
    last_target: str | None = None  # "head" | "meaning" | "example"

    def flush_curr() -> None:
        nonlocal curr
        if not curr or not curr_pair:
            curr = None
            return
        if not curr.get("definitions"):
            curr = None
            return
        if lang_pair_filter and curr_pair != lang_pair_filter:
            curr = None
            return
        if max_per_pair > 0 and pair_counts[curr_pair] >= max_per_pair:
            curr = None
            return

        src_lang, tgt_lang = curr_pair
        art, clean_text = _split_pdf_article(curr["rawHead"].strip(), src_lang)
        dom = curr.get("domains", ["GENERAL"])[0] if curr.get("domains") else "GENERAL"
        sec_doms = [d for d in (curr.get("domains") or [])[1:] if d != dom]
        for d in curr["definitions"]:
            d["language"] = tgt_lang
            d["domain"] = dom

        pair_counts[curr_pair] += 1
        entries.append({
            "termId": f"pdf:{src_lang}:{tgt_lang}:{pair_counts[curr_pair]}",
            "inDb": False,
            "pair": f"{src_lang}->{tgt_lang}",
            "text": clean_text,
            "rawHead": curr["rawHead"].strip(),
            "lang": src_lang,
            "targetLang": tgt_lang,
            "partOfSpeech": curr.get("pos"),
            "gender": None,
            "article": art,
            "plural": None,
            "level": "UNKNOWN",
            "source": "pdf_export",
            "primaryDomain": dom,
            "secondaryDomains": sec_doms,
            "definitions": curr["definitions"],
            "examples": curr.get("examples", []),
        })
        curr = None

    n_lines = len(lines)
    i = 0
    while i < n_lines:
        raw = lines[i]
        s = raw.strip()
        i += 1

        if not s or s == "\x0c":
            continue

        if re.match(r"^Glossa Büyük Sözlük(?:\s+\d+)?$", s):
            if s == "Glossa Büyük Sözlük":
                skip_page_num = True
            continue
        if skip_page_num:
            skip_page_num = False
            if s.isdigit():
                continue

        if s == "BÖLÜM":
            flush_curr()
            in_bolum_header = True
            last_target = None
            continue

        if in_bolum_header:
            m_ch = _RE_CHAPTER_PAIR.match(s)
            if m_ch:
                k1 = m_ch.group(1).lower().replace("\u0307", "")
                k2 = m_ch.group(2).lower().replace("\u0307", "")
                sc = LANG_DISPLAY_TO_CODE.get(k1)
                tc = LANG_DISPLAY_TO_CODE.get(k2)
                if sc and tc:
                    curr_pair = (sc, tc)
                in_bolum_header = False
                continue

        if curr_pair is None:
            continue

        if s.startswith("Bu bölüm kayıtlı ve onaylı"):
            continue

        # Skip section dividers ("#", "A", "B"..) unless immediately followed by [Domain]/(pos)/1.
        if _RE_LETTER_DIVIDER.match(s):
            nxt = ""
            for k in range(i, min(n_lines, i + 4)):
                if lines[k].strip() and lines[k].strip() != "\x0c":
                    nxt = lines[k].strip()
                    break
            m_pos_nxt = _RE_POS_DOMAIN_LINE.match(nxt) if ("(" in nxt or "[" in nxt) else None
            if not _RE_MEANING_LINE.match(nxt) and not (m_pos_nxt and (m_pos_nxt.group(1) or m_pos_nxt.group(2))):
                flush_curr()
                last_target = None
                continue

        # Example line ("Örn: ...")
        m_ex = _RE_EXAMPLE_LINE.match(s)
        if m_ex and curr:
            body = m_ex.group(1).strip()
            curr.setdefault("examples", []).append({"raw": body})
            last_target = "example"
            continue

        # Meaning line ("1. ...", "2. ...")
        m_def = _RE_MEANING_LINE.match(s)
        if m_def and curr:
            idx_num = int(m_def.group(1))
            # Guard: headword starting with numbered text when curr has no definitions & line precedes [Domain]/1.
            if idx_num == 1 or curr.get("definitions"):
                curr["definitions"].append({
                    "id": None,
                    "meaning": m_def.group(2).strip(),
                    "language": curr_pair[1],
                    "domain": curr.get("domains", ["GENERAL"])[0] if curr.get("domains") else "GENERAL",
                })
                last_target = "meaning"
                continue

        # Metadata line: "(n.) [Genel]" or "[Hukuk]" or "(n.)"
        if ("(" in s or "[" in s) and curr and not curr.get("definitions"):
            m_meta = _RE_POS_DOMAIN_LINE.match(s)
            if m_meta and (m_meta.group(1) or m_meta.group(2)):
                raw_pos = (m_meta.group(1) or "").strip().lower()
                raw_doms = (m_meta.group(2) or "").strip()
                if raw_pos in POS_LABEL_TO_ENUM or raw_doms:
                    if raw_pos in POS_LABEL_TO_ENUM:
                        curr["pos"] = POS_LABEL_TO_ENUM[raw_pos]
                    if raw_doms:
                        curr["domains"] = [normalize_domain(part.strip()) for part in raw_doms.split(",") if part.strip()]
                    last_target = "meta"
                    continue

        # Continuation lines for wrapped meaning/example vs new headword
        if curr and last_target == "meaning" and not s[0].isdigit():
            # Check if upcoming line is POS/domain or "1. " (which means `s` is actually a NEW headword)
            lookahead = []
            for k in range(i, min(n_lines, i + 4)):
                cand = lines[k].strip()
                if cand and cand != "\x0c":
                    lookahead.append(cand)
            is_new_head = any(
                (_RE_MEANING_LINE.match(x) and _RE_MEANING_LINE.match(x).group(1) == "1")
                or (
                    (x.startswith("(") or x.startswith("["))
                    and _RE_POS_DOMAIN_LINE.match(x)
                    and (_RE_POS_DOMAIN_LINE.match(x).group(1) or _RE_POS_DOMAIN_LINE.match(x).group(2))
                )
                for x in lookahead[:2]
            )
            if not is_new_head and curr["definitions"]:
                curr["definitions"][-1]["meaning"] += " " + s
                continue

        if curr and last_target == "example":
            lookahead = []
            for k in range(i, min(n_lines, i + 3)):
                cand = lines[k].strip()
                if cand and cand != "\x0c":
                    lookahead.append(cand)
            is_new_head = any(
                (_RE_MEANING_LINE.match(x) and _RE_MEANING_LINE.match(x).group(1) == "1")
                or (
                    (x.startswith("(") or x.startswith("["))
                    and _RE_POS_DOMAIN_LINE.match(x)
                    and (_RE_POS_DOMAIN_LINE.match(x).group(1) or _RE_POS_DOMAIN_LINE.match(x).group(2))
                )
                for x in lookahead[:2]
            )
            if not is_new_head and curr.get("examples") and not _RE_MEANING_LINE.match(s):
                curr["examples"][-1]["raw"] += " " + s
                continue

        if curr and last_target == "head" and not curr.get("definitions"):
            curr["rawHead"] += " " + s
            continue

        # New entry starts
        flush_curr()
        if max_per_pair > 0 and pair_counts[curr_pair] >= max_per_pair and not lang_pair_filter:
            # Skip scanning rest of finished chapter quickly? Wait: next chapter changes curr_pair, so skip lines until BÖLÜM
            curr = None
            last_target = None
            continue
        curr = {"rawHead": s, "pos": None, "domains": ["GENERAL"], "definitions": [], "examples": []}
        last_target = "head"

    flush_curr()

    # Split parsed example strings "Sentence (Translation)" cleanly
    for e in entries:
        parsed_ex = []
        for ex in e.pop("examples", []):
            raw_ex = ex.get("raw", "")
            m_tr = re.match(r"^(.*)\(([^()]{2,})\)\s*$", raw_ex)
            if m_tr:
                parsed_ex.append({
                    "sentence": m_tr.group(1).strip(),
                    "translation": m_tr.group(2).strip(),
                    "translationLang": e["targetLang"],
                })
            elif raw_ex:
                parsed_ex.append({"sentence": raw_ex, "translation": "", "translationLang": e["targetLang"]})
        e["examples"] = parsed_ex
    return entries


def parse_glossa_pdf_entries(
    pdf_path: Path,
    max_per_pair: int = 0,
    lang_pair_filter: tuple[str, str] | None = None,
    max_pages: int = 0,
) -> list[dict[str, Any]]:
    raw_text = _extract_pdf_text(pdf_path, max_pages=max_pages)
    return parse_glossa_pdf_text(raw_text, max_per_pair=max_per_pair, lang_pair_filter=lang_pair_filter)


# ─── NIM Audit Prompt & Persistence ───────────────────────────────────────────

SYSTEM_PROMPT = """You are LexiForge-SQL, an expert 12-language lexicographer (DE, TR, EN, FR, ES, IT, PT, RO, LA, RU, SR, MK).
Your job is to verify EVERY dictionary entry across BOTH:
1. **Language accuracy (`langValid`, `lang`, `definitions[].language`)**: Verify the headword truly belongs to `lang` (script, orthography, characters like Cyrillic for RU/MK/SR, ğ/ş/ı for TR, ß/ä/ö/ü for DE) and each definition is in its declared `language` (not untranslated or swapped!).
2. **12-Language Grammatical & Domain Accuracy (`grammarValid`)**: Verify `partOfSpeech`, `gender`, `article`, `plural`, inflections, and 12-language fields (`script`, `transliteration` for RU/SR/MK, `lemma`, `animacy` for RU/SR nouns, `aspect` & `aspectPair` for RU/SR/MK verbs, `declensionClass`, `conjugationClass`, `definiteForm` for RO/MK, `vowelHarmony` for TR, `auxiliary`, `level`, `primaryDomain`, `secondaryDomains`, `contextTags`).

Return ONLY a JSON array matching each input `termId`:
[
  {
    "termId": "<exact termId>",
    "verdict": "CORRECT" | "NEEDS_FIX" | "DELETE",
    "action": "KEEP" | "FIX" | "DELETE",
    "langValid": true | false,
    "grammarValid": true | false,
    "issues": ["<short diagnostic tags, empty [] if CORRECT>"],
    "reason": "<concise explanation>",
    "text": "<headword>",
    "lang": "DE"|"TR"|"EN"|"FR"|"ES"|"IT"|"PT"|"RO"|"LA"|"RU"|"SR"|"MK",
    "partOfSpeech": "NOUN"|"VERB"|"ADJECTIVE"|"ADVERB"|"PRONOUN"|"PREPOSITION"|"CONJUNCTION"|"INTERJECTION"|"ARTICLE"|"NUMERAL"|"PARTICLE"|"PHRASE"|"IDIOM"|"PREFIX"|"SUFFIX",
    "gender": "MASCULINE"|"FEMININE"|"NEUTER"|"COMMON"|"AMBIGENERIC"|null,
    "article": "der"|"die"|"das"|"le"|"la"|"l'"|"les"|"el"|"los"|"las"|"il"|"lo"|"o"|"a"|"un"|null,
    "plural": "<plural form or null>",
    "level": "A1"|"A2"|"B1"|"B2"|"C1"|"C2"|"UNKNOWN",
    "script": "LATIN"|"CYRILLIC"|"LATIN_CYRILLIC",
    "transliteration": "<Latin romanization for RU/SR/MK or null>",
    "lemma": "<canonical dictionary form>",
    "animacy": "ANIMATE"|"INANIMATE"|"NOT_APPLICABLE"|null,
    "aspect": "IMPERFECTIVE"|"PERFECTIVE"|"BIASPECTUAL"|null,
    "aspectPair": "<partner aspect verb for RU/SR/MK or null>",
    "declensionClass": "<LA I-V / RU I-III / RO / SR / DE class or null>",
    "conjugationClass": "<verb conjugation class or null>",
    "definiteForm": "<RO/MK definite form or null>",
    "vowelHarmony": "<TR harmony note or null>",
    "auxiliary": "haben"|"sein"|"avoir"|"être"|"avere"|"essere"|null,
    "primaryDomain": "<Prisma Domain enum>",
    "secondaryDomains": ["<optional 0-2 Prisma Domain enums>"],
    "contextTags": ["<2-4 lowercase context tags>"],
    "definitions": [
      {"id": "<exact id or null>", "meaning": "<clean accurate translation>", "language": "<2-letter>", "domain": "<Prisma Domain enum>", "register": "NEUTRAL"|"FORMAL"|"INFORMAL"|"COLLOQUIAL"|"SLANG"|"ARCHAIC"|"LITERARY"|"TECHNICAL"}
    ],
    "examples": [
      {"sentence": "<natural sentence in source lang>", "translation": "<translation>", "translationLang": "<2-letter>", "domain": "<Domain enum>", "level": "A1".."C2"}
    ]
  }
]
Rules:
- Mark `verdict="CORRECT"` and `action="KEEP"` if headword, language, POS/gender/article, and definitions are accurate (you may still fill missing `level`, `primaryDomain`, `lemma`, `script`).
- Mark `verdict="DELETE", action="DELETE"` ONLY for pure non-lexical junk ("11th", "11.", "1st", "1337", page headers "136brodeln –B ruc h", copyright notices, standalone alphabet letters with no real meaning).
- NEVER mark `DELETE` when `cleanHint` is given or valid definitions exist: instead use `verdict="NEEDS_FIX", action="FIX"` and `text=cleanHint`.
- NEVER alter or delete valid chemical locants ("1-Hexanol", "1-Pentanol", "1-Propanol", "1,2-Butadien"), dates ("11. September"), times ("1 Uhr"), or valid compounds ("14K", "24/7", "9/11").
- For TR & EN nouns: `article=null`, `gender=null`."""


def save_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if "meta" in payload and isinstance(payload["meta"], dict):
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


def _resolve_author_id(cur) -> str:
    cur.execute('SELECT id FROM "User" ORDER BY "createdAt" ASC LIMIT 1')
    row = cur.fetchone()
    if not row:
        raise RuntimeError('No user row found in "User" table for Definition.authorId.')
    return row["id"] if isinstance(row, dict) else row[0]


def purge_junk_row_tx(conn: psycopg2.extensions.connection, term_row: dict[str, Any], reasons: list[str], dry_run: bool) -> str:
    tid = term_row["termId"]
    if str(tid).startswith(("pdf:", "json:")):
        return f"{'dry_run_' if dry_run else ''}skip_external_junk:{','.join(reasons)}"
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
            cur.execute(
                """
                SELECT id, lang, "partOfSpeech"::text AS "partOfSpeech", level::text AS level,
                       script, lemma, "primaryDomain"::text AS "primaryDomain"
                FROM "Term" WHERE id = %s FOR UPDATE
                """,
                (term_id,),
            )
            curr = cur.fetchone()
            if not curr:
                conn.rollback()
                return "already_missing"

            lang = str(fix.get("lang") or curr["lang"]).upper()
            if lang not in VALID_LANGS:
                lang = curr["lang"]
            pos = fix.get("partOfSpeech") or curr["partOfSpeech"]
            if pos:
                pos = str(pos).upper()
            if pos not in VALID_POS:
                pos = curr["partOfSpeech"]

            text = re.sub(r"\s+", " ", str(fix.get("text") or "")).strip()
            act, cleaned_text, _ = classify_headword(text, lang=lang, pos=pos)
            if act == "delete" or not cleaned_text:
                conn.rollback()
                return "skipped_invalid_target_text"

            san = sanitize_grammar_fields(
                {**fix, "text": cleaned_text, "lang": lang, "partOfSpeech": pos},
                src_lang=lang,
            )
            if san["level"] == "UNKNOWN" and curr["level"] and curr["level"] != "UNKNOWN":
                san["level"] = curr["level"]
            if san["primaryDomain"] == "GENERAL" and curr["primaryDomain"] and curr["primaryDomain"] != "GENERAL":
                san["primaryDomain"] = curr["primaryDomain"]

            edit_note = f"lexiforge-nim: {str(fix.get('reason') or 'verified')[:180]}"

            collision = None
            if pos:
                cur.execute(
                    'SELECT id FROM "Term" WHERE text = %s AND lang = %s AND "partOfSpeech" = %s::"PartOfSpeech" AND id <> %s FOR UPDATE',
                    (cleaned_text, lang, pos, term_id),
                )
                collision = cur.fetchone()

            kept_def_ids: list[str] = []
            for d in san.get("definitions") or []:
                did = d.get("id")
                meaning = re.sub(r"\s+", " ", str(d.get("meaning") or "")).strip()
                if not did or is_junk_meaning(meaning, cleaned_text):
                    continue
                d_lang = str(d.get("language") or "TR").upper()
                if d_lang not in VALID_LANGS or d_lang == lang:
                    continue
                dom = normalize_domain(d.get("domain"), san["primaryDomain"])
                reg = str(d.get("register") or "NEUTRAL").upper()
                if reg not in VALID_REGISTER:
                    reg = "NEUTRAL"
                cur.execute(
                    """
                    UPDATE "Definition"
                    SET meaning = %s, language = %s, domain = %s::"Domain",
                        register = %s::"Register",
                        "contextTags" = %s::text[], "editNote" = %s, "updatedAt" = NOW()
                    WHERE id = %s AND "termId" = %s
                    """,
                    (meaning, d_lang, dom, reg, san["contextTags"], edit_note, did, term_id),
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
                    gender = %s::"Gender", article = %s,
                    plural = COALESCE(%s, plural),
                    genitive = COALESCE(%s, genitive),
                    comparative = COALESCE(%s, comparative),
                    superlative = COALESCE(%s, superlative),
                    "pastTense" = COALESCE(%s, "pastTense"),
                    "pastParticiple" = COALESCE(%s, "pastParticiple"),
                    "presentParticiple" = COALESCE(%s, "presentParticiple"),
                    auxiliary = COALESCE(%s, auxiliary),
                    "isRegular" = COALESCE(%s, "isRegular"),
                    "isSeparable" = COALESCE(%s, "isSeparable"),
                    "isReflexive" = COALESCE(%s, "isReflexive"),
                    "verbPrefix" = COALESCE(%s, "verbPrefix"),
                    pronunciation = COALESCE(%s, pronunciation),
                    syllables = COALESCE(%s, syllables),
                    level = %s::"LanguageLevel",
                    script = %s,
                    transliteration = %s,
                    lemma = COALESCE(%s, lemma),
                    animacy = %s::"Animacy",
                    aspect = %s::"VerbAspect",
                    "aspectPair" = COALESCE(%s, "aspectPair"),
                    "declensionClass" = COALESCE(%s, "declensionClass"),
                    "conjugationClass" = COALESCE(%s, "conjugationClass"),
                    "definiteForm" = COALESCE(%s, "definiteForm"),
                    "vowelHarmony" = COALESCE(%s, "vowelHarmony"),
                    "grammarProfile" = COALESCE(%s::jsonb, "grammarProfile"),
                    "primaryDomain" = %s::"Domain",
                    "secondaryDomains" = %s::"Domain"[],
                    "contextTags" = %s::text[],
                    "qualityScore" = GREATEST("qualityScore", 85),
                    "enrichedAt" = NOW(), "editNote" = %s, "updatedAt" = NOW()
                WHERE id = %s
                """,
                (
                    cleaned_text, lang, pos, san.get("gender"), san.get("article"),
                    san.get("plural"), san.get("genitive"), san.get("comparative"), san.get("superlative"),
                    san.get("pastTense"), san.get("pastParticiple"), san.get("presentParticiple"),
                    san.get("auxiliary"), san.get("isRegular"), san.get("isSeparable"), san.get("isReflexive"),
                    san.get("verbPrefix"), san.get("pronunciation"), san.get("syllables"),
                    san["level"], san.get("script"), san.get("transliteration"), san.get("lemma"),
                    san.get("animacy"), san.get("aspect"), san.get("aspectPair"),
                    san.get("declensionClass"), san.get("conjugationClass"), san.get("definiteForm"),
                    san.get("vowelHarmony"), Json(san["grammarProfile"]) if san.get("grammarProfile") else None,
                    san["primaryDomain"], san["secondaryDomains"], san["contextTags"],
                    edit_note, target_id,
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
                    (str(uuid.uuid4()), target_id, first_def_id, sent, tr, tlang, san["primaryDomain"], san["level"], lang),
                )

        if dry_run:
            conn.rollback()
            return f"dry_run_{action}"
        conn.commit()
        return action
    except Exception:
        conn.rollback()
        raise


def upsert_external_term_tx(
    conn: psycopg2.extensions.connection,
    orig: dict[str, Any],
    fix: dict[str, Any],
    dry_run: bool,
) -> str:
    """Match or insert an external PDF/JSON term and sync definitions/grammar into PostgreSQL."""
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            lang = str(fix.get("lang") or orig.get("lang") or "DE").upper()
            if lang not in VALID_LANGS:
                lang = "DE"
            pos = fix.get("partOfSpeech") or orig.get("partOfSpeech")
            if pos:
                pos = str(pos).upper()
            if pos not in VALID_POS:
                pos = None

            text = re.sub(r"\s+", " ", str(fix.get("text") or orig.get("text") or "")).strip()
            act, cleaned_text, _ = classify_headword(text, lang=lang, pos=pos)
            if act == "delete" or not cleaned_text:
                conn.rollback()
                return "skipped_invalid_external_text"

            san = sanitize_grammar_fields({**orig, **fix, "text": cleaned_text, "lang": lang, "partOfSpeech": pos}, src_lang=lang)
            pos = san.get("partOfSpeech")

            cur.execute(
                """
                SELECT id FROM "Term"
                WHERE text = %s AND lang = %s
                  AND (%s::text IS NULL OR "partOfSpeech"::text = %s OR "partOfSpeech" IS NULL)
                ORDER BY ("partOfSpeech"::text = %s) DESC NULLS LAST
                LIMIT 1 FOR UPDATE
                """,
                (cleaned_text, lang, pos, pos, pos),
            )
            row = cur.fetchone()
            author_id = _resolve_author_id(cur)
            edit_note = f"lexiforge-sync: {str(fix.get('reason') or orig.get('source') or 'imported')[:160]}"

            if row:
                term_id = row["id"]
                mode_tag = "updated_existing"
            else:
                term_id = str(uuid.uuid4())
                mode_tag = "inserted_new"
                cur.execute(
                    """
                    INSERT INTO "Term" (
                        id, text, lang, "partOfSpeech", gender, article, plural,
                        level, script, transliteration, lemma, animacy, aspect, "aspectPair",
                        "declensionClass", "conjugationClass", "definiteForm", "vowelHarmony",
                        "grammarProfile", "primaryDomain", "secondaryDomains", "contextTags",
                        "qualityScore", "enrichedAt", source, "editNote", "createdAt", "updatedAt"
                    )
                    VALUES (
                        %s, %s, %s, %s::"PartOfSpeech", %s::"Gender", %s, %s,
                        %s::"LanguageLevel", %s, %s, %s, %s::"Animacy", %s::"VerbAspect", %s,
                        %s, %s, %s, %s, %s::jsonb, %s::"Domain", %s::"Domain"[], %s::text[],
                        85, NOW(), %s, %s, NOW(), NOW()
                    )
                    """,
                    (
                        term_id, cleaned_text, lang, pos, san.get("gender"), san.get("article"), san.get("plural"),
                        san["level"], san.get("script"), san.get("transliteration"), san.get("lemma"),
                        san.get("animacy"), san.get("aspect"), san.get("aspectPair"),
                        san.get("declensionClass"), san.get("conjugationClass"), san.get("definiteForm"),
                        san.get("vowelHarmony"), Json(san["grammarProfile"]) if san.get("grammarProfile") else None,
                        san["primaryDomain"], san["secondaryDomains"], san["contextTags"],
                        orig.get("source") or "lexiforge_sync", edit_note,
                    ),
                )

            for d in san.get("definitions") or orig.get("definitions") or []:
                meaning = re.sub(r"\s+", " ", str(d.get("meaning") or "")).strip()
                if is_junk_meaning(meaning, cleaned_text):
                    continue
                d_lang = str(d.get("language") or orig.get("targetLang") or ("TR" if lang != "TR" else "EN")).upper()
                if d_lang not in VALID_LANGS or d_lang == lang:
                    continue
                dom = normalize_domain(d.get("domain"), san["primaryDomain"])
                reg = str(d.get("register") or "NEUTRAL").upper()
                if reg not in VALID_REGISTER:
                    reg = "NEUTRAL"
                cur.execute(
                    'SELECT 1 FROM "Definition" WHERE "termId" = %s AND language = %s AND lower(trim(meaning)) = lower(trim(%s)) LIMIT 1',
                    (term_id, d_lang, meaning),
                )
                if not cur.fetchone():
                    cur.execute(
                        """
                        INSERT INTO "Definition" (
                            id, "termId", meaning, domain, register, "contextTags",
                            "authorId", language, "isApproved", "sourceCitation", "editNote", "createdAt", "updatedAt"
                        )
                        VALUES (%s, %s, %s, %s::"Domain", %s::"Register", %s::text[], %s, %s, true, 'LexiForge-NIM', %s, NOW(), NOW())
                        """,
                        (str(uuid.uuid4()), term_id, meaning, dom, reg, san["contextTags"], author_id, d_lang, edit_note),
                    )

        if dry_run:
            conn.rollback()
            return f"dry_run_{mode_tag}"
        conn.commit()
        return mode_tag
    except Exception:
        conn.rollback()
        raise


# ─── PostgreSQL Term Fetchers (--mode junk | anomalies | all | pairs) ──────────

def _attach_definitions(cur, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    tids = [r["termId"] for r in rows]
    cur.execute(
        'SELECT "termId", id, meaning, language, domain::text AS domain, register::text AS register FROM "Definition" WHERE "termId" = ANY(%s)',
        (tids,),
    )
    defs_map: dict[str, list[dict[str, Any]]] = {tid: [] for tid in tids}
    for d in cur.fetchall():
        defs_map[d["termId"]].append({
            "id": d["id"], "meaning": d["meaning"], "language": d["language"],
            "domain": d["domain"], "register": d["register"],
        })
    for r in rows:
        r["definitions"] = defs_map.get(r["termId"], [])
        tgt = r["definitions"][0]["language"] if r["definitions"] else ("TR" if r["lang"] != "TR" else "EN")
        r.setdefault("targetLang", tgt)
        r.setdefault("pair", f"{r['lang']}->{r['targetLang']}")
    return rows


_TERM_SELECT_COLS = """
    t.id AS "termId", t.text, t.lang, t."partOfSpeech"::text AS "partOfSpeech",
    t.gender::text AS gender, t.article, t.plural, t.level::text AS level,
    t.source, t."primaryDomain"::text AS "primaryDomain",
    t."secondaryDomains"::text[] AS "secondaryDomains",
    t."contextTags", t.script, t.transliteration, t.lemma,
    t.animacy::text AS animacy, t.aspect::text AS aspect, t."aspectPair",
    t."declensionClass", t."conjugationClass", t."definiteForm", t."vowelHarmony"
"""


def fetch_suspicious_db_terms(conn, limit: int = 0) -> list[dict[str, Any]]:
    sql = f"""
        SELECT {_TERM_SELECT_COLS}
        FROM "Term" t
        WHERE t.text ~ '[0-9]'
           OR length(trim(t.text)) <= 1
           OR t.text ~ '^[\\(\\[\\{{]|^[–—•*.,;:]|^©'
           OR t.text ~* '^(seite|page|sayfa)\\s*[0-9]+'
        ORDER BY t.text ASC
    """
    if limit and limit > 0:
        sql += " LIMIT %s"
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(sql, (limit,) if (limit and limit > 0) else ())
        rows = [dict(r) for r in cur.fetchall()]
        return _attach_definitions(cur, rows)


def fetch_anomaly_db_terms(
    conn,
    limit: int = 500,
    offset: int = 0,
    lang_pair_filter: tuple[str, str] | None = None,
    last_id: str | None = None,
) -> list[dict[str, Any]]:
    pair_clause = ""
    params: list[Any] = []
    keyset_clause = ""
    if last_id is not None:
        keyset_clause = " AND t.id > %s"
        params.append(last_id)
    if lang_pair_filter:
        pair_clause = ' AND t.lang = %s AND EXISTS (SELECT 1 FROM "Definition" df WHERE df."termId" = t.id AND df.language = %s)'
        params.extend([lang_pair_filter[0], lang_pair_filter[1]])

    order_sql = "ORDER BY t.id ASC" if last_id is not None else 'ORDER BY (t.text ~ \'[0-9]\' OR length(trim(t.text)) <= 1) DESC, t."qualityScore" ASC, t.id ASC'
    sql = f"""
        SELECT {_TERM_SELECT_COLS}
        FROM "Term" t
        WHERE (
            t.text ~ '[0-9]'
            OR length(trim(t.text)) <= 1
            OR t.text ~ '^[\\(\\[\\{{]|^[–—•*.,;:]|^©'
            OR (t.lang IN ('TR', 'EN') AND t.article IS NOT NULL)
            OR lower(coalesce(t.article, '')) IN ('f','m','n','s','adj','adv','v','is','pl')
            OR t."partOfSpeech" IS NULL
            OR t.level = 'UNKNOWN'
            OR t.lemma IS NULL
            OR t.script IS NULL
            OR (t.lang IN ('RU', 'MK') AND t.text ~ '[A-Za-z]' AND t.text !~ '[\u0400-\u04FF]')
            OR (t.lang NOT IN ('RU', 'SR', 'MK') AND t.text ~ '[\u0400-\u04FF]')
            OR EXISTS (
                SELECT 1 FROM "Definition" d
                WHERE d."termId" = t.id
                  AND (d.language = t.lang OR lower(trim(d.meaning)) = lower(trim(t.text)))
            )
        )
        {keyset_clause}
        {pair_clause}
        {order_sql}
    """
    if limit and limit > 0:
        if last_id is not None:
            sql += " LIMIT %s"
            params.append(limit)
        else:
            sql += " LIMIT %s OFFSET %s"
            params.extend([limit, max(0, offset)])
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(sql, tuple(params))
        rows = [dict(r) for r in cur.fetchall()]
        return _attach_definitions(cur, rows)


def fetch_all_db_terms(
    conn,
    limit: int = 500,
    offset: int = 0,
    lang_pair_filter: tuple[str, str] | None = None,
    last_id: str | None = None,
) -> list[dict[str, Any]]:
    clauses: list[str] = []
    params: list[Any] = []
    if last_id is not None:
        clauses.append("t.id > %s")
        params.append(last_id)
    if lang_pair_filter:
        clauses.append('t.lang = %s AND EXISTS (SELECT 1 FROM "Definition" df WHERE df."termId" = t.id AND df.language = %s)')
        params.extend([lang_pair_filter[0], lang_pair_filter[1]])
    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    order_sql = 'ORDER BY t.id ASC' if last_id is not None else 'ORDER BY t."qualityScore" ASC, t."updatedAt" ASC, t.id ASC'
    sql = f"""
        SELECT {_TERM_SELECT_COLS}
        FROM "Term" t
        {where}
        {order_sql}
    """
    if limit and limit > 0:
        if last_id is not None:
            sql += " LIMIT %s"
            params.append(limit)
        else:
            sql += " LIMIT %s OFFSET %s"
            params.extend([limit, max(0, offset)])
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(sql, tuple(params))
        rows = [dict(r) for r in cur.fetchall()]
        return _attach_definitions(cur, rows)


def discover_db_lang_pairs(conn, lang_pair_filter: tuple[str, str] | None = None) -> list[tuple[str, str, int]]:
    """Discover all distinct (src_lang, tgt_lang, term_count) pairs in PostgreSQL."""
    with conn.cursor() as cur:
        if lang_pair_filter:
            cur.execute(
                """
                SELECT t.lang::text AS src_lang, d.language::text AS tgt_lang, COUNT(DISTINCT t.id) AS cnt
                FROM "Term" t
                JOIN "Definition" d ON d."termId" = t.id
                WHERE t.lang = %s AND d.language = %s
                GROUP BY t.lang, d.language
                ORDER BY src_lang ASC, tgt_lang ASC
                """,
                (lang_pair_filter[0], lang_pair_filter[1]),
            )
        else:
            cur.execute(
                """
                SELECT t.lang::text AS src_lang, d.language::text AS tgt_lang, COUNT(DISTINCT t.id) AS cnt
                FROM "Term" t
                JOIN "Definition" d ON d."termId" = t.id
                GROUP BY t.lang, d.language
                ORDER BY src_lang ASC, tgt_lang ASC
                """
            )
        return [(str(r[0]), str(r[1]), int(r[2])) for r in cur.fetchall()]


def fetch_pair_chunk(
    conn,
    src_lang: str,
    tgt_lang: str,
    last_id: str = "",
    chunk_size: int = 50,
    recheck: bool = False,
) -> list[dict[str, Any]]:
    """Keyset-paginated chunk fetcher for one (src_lang, tgt_lang) pair."""
    work_filter = ""
    if not recheck:
        work_filter = """
            AND (
                t."enrichedAt" IS NULL
                OR t."qualityScore" < 85
                OR t."partOfSpeech" IS NULL
                OR t.level = 'UNKNOWN'
                OR t.lemma IS NULL
                OR t.script IS NULL
                OR t.text ~ '[0-9]'
                OR length(trim(t.text)) <= 1
                OR (t.lang IN ('TR', 'EN') AND t.article IS NOT NULL)
                OR lower(coalesce(t.article, '')) IN ('f','m','n','s','adj','adv','v','is','pl')
                OR d.language = t.lang
                OR lower(trim(d.meaning)) = lower(trim(t.text))
            )
        """
    sql = f"""
        SELECT {_TERM_SELECT_COLS},
               d.id AS "_pair_def_id", d.meaning AS "_pair_meaning",
               d.language AS "_pair_tgt_lang", d.domain::text AS "_pair_domain",
               d.register::text AS "_pair_register"
        FROM "Term" t
        JOIN LATERAL (
            SELECT id, meaning, language, domain, register
            FROM "Definition" d
            WHERE d."termId" = t.id AND d.language = %s
            ORDER BY (lower(trim(d.meaning)) = lower(trim(t.text))) DESC, d."updatedAt" ASC
            LIMIT 1
        ) d ON true
        WHERE t.lang = %s
          AND t.id > %s
          {work_filter}
        ORDER BY t.id ASC
        LIMIT %s
    """
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(sql, (tgt_lang, src_lang, last_id or "", max(1, chunk_size)))
        out: list[dict[str, Any]] = []
        for raw in cur.fetchall():
            r = dict(raw)
            def_item = {
                "id": r.pop("_pair_def_id"),
                "meaning": r.pop("_pair_meaning"),
                "language": r.pop("_pair_tgt_lang"),
                "domain": r.pop("_pair_domain"),
                "register": r.pop("_pair_register"),
            }
            r["auditKey"] = f"{r['termId']}@{tgt_lang}"
            r["inDb"] = True
            r["targetLang"] = tgt_lang
            r["pair"] = f"{src_lang}->{tgt_lang}"
            r["definitions"] = [def_item]
            out.append(r)
        return out


def fetch_stratified_lang_pair_terms(conn, per_pair: int = 3, lang_pair_filter: tuple[str, str] | None = None) -> list[dict[str, Any]]:
    """Sample representative terms from EVERY distinct (t.lang, d.language) pair in PostgreSQL."""
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        if lang_pair_filter:
            pairs = [(lang_pair_filter[0], lang_pair_filter[1])]
        else:
            cur.execute(
                """
                SELECT DISTINCT t.lang::text AS src_lang, d.language::text AS tgt_lang, COUNT(DISTINCT t.id) AS cnt
                FROM "Term" t
                JOIN "Definition" d ON d."termId" = t.id
                GROUP BY t.lang, d.language
                ORDER BY src_lang ASC, tgt_lang ASC
                """
            )
            pairs = [(r["src_lang"], r["tgt_lang"]) for r in cur.fetchall()]

        sampled: list[dict[str, Any]] = []
        for idx, (src_lang, tgt_lang) in enumerate(pairs):
            cur.execute(
                f"""
                SELECT {_TERM_SELECT_COLS},
                       d.id AS "_pair_def_id", d.meaning AS "_pair_meaning",
                       d.language AS "_pair_tgt_lang", d.domain::text AS "_pair_domain",
                       d.register::text AS "_pair_register"
                FROM "Term" t
                JOIN LATERAL (
                    SELECT id, meaning, language, domain, register
                    FROM "Definition" d
                    WHERE d."termId" = t.id AND d.language = %s
                    ORDER BY (lower(trim(d.meaning)) = lower(trim(t.text))) DESC, d."updatedAt" ASC
                    LIMIT 1
                ) d ON true
                WHERE t.lang = %s
                ORDER BY (
                    CASE
                        WHEN t.text ~ '[0-9]' OR length(trim(t.text)) <= 1 THEN 0
                        WHEN t.lang = %s OR lower(trim(d.meaning)) = lower(trim(t.text)) THEN 1
                        WHEN (t.lang IN ('TR','EN') AND t.article IS NOT NULL) THEN 2
                        WHEN t."partOfSpeech" IS NULL OR t.level = 'UNKNOWN' THEN 3
                        ELSE 4
                    END
                ) ASC, t."qualityScore" ASC, t.id ASC
                LIMIT %s
                """,
                (tgt_lang, src_lang, tgt_lang, max(1, per_pair)),
            )
            pair_rows = [dict(r) for r in cur.fetchall()]
            for pos_idx, r in enumerate(pair_rows):
                orig_tid = r["termId"]
                def_item = {
                    "id": r.pop("_pair_def_id"),
                    "meaning": r.pop("_pair_meaning"),
                    "language": r.pop("_pair_tgt_lang"),
                    "domain": r.pop("_pair_domain"),
                    "register": r.pop("_pair_register"),
                }
                # Unique audit key per (term, tgt_lang) when running multi-pair audits
                r["auditKey"] = f"{orig_tid}@{tgt_lang}"
                r["inDb"] = True
                r["targetLang"] = tgt_lang
                r["pair"] = f"{src_lang}->{tgt_lang}"
                r["definitions"] = [def_item]
                sampled.append(r)
        return sampled


# ─── Per-Language-Pair Reporting ───────────────────────────────────────────────

def _classify_pair_verdict(orig: dict[str, Any], prop: dict[str, Any]) -> tuple[str, bool, bool]:
    """Return (verdict_bucket: 'DOGRU'|'YANLIS'|'COP', lang_err: bool, grammar_err: bool)."""
    det_lang_issues = detect_language_anomaly(
        orig.get("text") or "",
        orig.get("lang") or "DE",
        article=orig.get("article"),
        definitions=orig.get("definitions"),
    )
    act_local, _, _ = classify_headword(
        orig.get("text") or "",
        lang=orig.get("lang") or "DE",
        pos=orig.get("partOfSpeech"),
        definitions=orig.get("definitions"),
    )
    action = str(prop.get("action") or "KEEP").upper()
    verdict_raw = str(prop.get("verdict") or "").upper()
    lang_valid = prop.get("langValid")
    gram_valid = prop.get("grammarValid")

    lang_err = bool(det_lang_issues) or (lang_valid is False) or (
        prop.get("lang") and str(prop.get("lang")).upper() != str(orig.get("lang") or "").upper()
    )
    grammar_err = (gram_valid is False) or (
        orig.get("partOfSpeech") is not None and prop.get("partOfSpeech") is not None
        and str(prop.get("partOfSpeech")).upper() != str(orig.get("partOfSpeech")).upper()
    ) or (
        orig.get("article") and not prop.get("article") and orig.get("lang") in {"TR", "EN"}
    )

    if action == "DELETE" or verdict_raw == "DELETE" or act_local == "delete":
        return "COP", lang_err, grammar_err
    if verdict_raw == "NEEDS_FIX" or action == "FIX" or act_local == "clean" or lang_err or grammar_err:
        return "YANLIS", lang_err, grammar_err
    return "DOGRU", lang_err, grammar_err


def _apply_stage2_guards(orig: dict[str, Any], raw_prop: dict[str, Any]) -> dict[str, Any]:
    """Enforce hard safeguards against false deletions or protected headword corruption."""
    prop = dict(raw_prop or {})
    orig_text = orig.get("text") or ""
    clean_hint = orig.get("clean_hint")
    if str(prop.get("action") or "KEEP").upper() == "DELETE" and (
        clean_hint
        or is_protected_headword(orig_text)
        or (
            orig.get("definitions")
            and classify_headword(
                orig_text,
                orig.get("lang", "DE"),
                orig.get("partOfSpeech"),
                orig.get("definitions"),
            )[0] == "keep"
        )
    ):
        prop["action"] = "FIX"
        prop["verdict"] = "NEEDS_FIX"
        prop["text"] = clean_hint or orig_text
        prop["reason"] = f"guarded_fix_from_delete: {prop.get('reason') or ''}".strip()
    if is_protected_headword(orig_text):
        prop["text"] = orig_text
    elif clean_hint and not prop.get("text"):
        prop["text"] = clean_hint
    return prop


def _is_retryable_pg_error(exc: Exception) -> bool:
    retry_types = tuple(
        t
        for t in (
            getattr(pg_errors, "DeadlockDetected", None),
            getattr(pg_errors, "LockNotAvailable", None),
            getattr(pg_errors, "SerializationFailure", None),
        )
        if t is not None
    )
    if retry_types and isinstance(exc, retry_types):
        return True
    msg = str(exc).lower()
    return "deadlock detected" in msg or "could not obtain lock" in msg or "could not serialize access" in msg


def _run_tx_with_retry(conn, fn, *args, max_retries: int = 4, **kwargs):
    """Execute a PostgreSQL transaction helper with exponential jitter on deadlock/lock contention."""
    for attempt in range(max_retries + 1):
        try:
            return fn(conn, *args, **kwargs)
        except Exception as exc:
            if not _is_retryable_pg_error(exc) or attempt >= max_retries:
                raise
            try:
                conn.rollback()
            except Exception:
                pass
            time.sleep(0.1 * (2 ** attempt) + random.uniform(0.0, 0.15))


def _record_pair_outcome(
    by_pair: dict[str, dict[str, Any]],
    totals: dict[str, int],
    orig: dict[str, Any],
    prop: dict[str, Any],
    sample_cap: int = 3,
) -> None:
    pair = orig.get("pair") or f"{orig.get('lang', '?')}->{orig.get('targetLang', '?')}"
    bucket, l_err, g_err = _classify_pair_verdict(orig, prop)
    entry = by_pair.setdefault(pair, {
        "pair": pair, "tested": 0, "dogru": 0, "yanlis": 0, "cop": 0,
        "lang_err": 0, "grammar_err": 0, "samples": [],
    })
    entry["tested"] += 1
    totals["tested"] += 1
    if bucket == "DOGRU":
        entry["dogru"] += 1
        totals["dogru"] += 1
    elif bucket == "COP":
        entry["cop"] += 1
        totals["cop"] += 1
    else:
        entry["yanlis"] += 1
        totals["yanlis"] += 1
    if l_err:
        entry["lang_err"] += 1
        totals["lang_err"] += 1
    if g_err:
        entry["grammar_err"] += 1
        totals["grammar_err"] += 1
    if len(entry["samples"]) < sample_cap and (bucket != "DOGRU" or not entry["samples"]):
        entry["samples"].append({
            "text": orig.get("text"),
            "verdict": bucket,
            "fixedText": prop.get("text"),
            "fixedLang": prop.get("lang"),
            "issues": prop.get("issues") or [],
        })


def print_pair_aggregate_summary(by_pair: dict[str, dict[str, Any]], totals: dict[str, int]) -> dict[str, Any]:
    print("\n=== Dil Çifti Doğrulama Özeti (Per-Language-Pair Report) ===")
    print(f"{'Dil Çifti':<10} | {'Test':>4} | {'DOĞRU':>5} | {'YANLIŞ':>6} | {'ÇÖP':>4} | {'DilH':>4} | {'GramH':>5} | Örnek Bulgu")
    print("-" * 96)
    for pair in sorted(by_pair):
        b = by_pair[pair]
        smp = b["samples"][0] if b["samples"] else {}
        note = f"{smp.get('text', '')!r} [{smp.get('verdict', '')}]"
        if smp.get("issues"):
            note += f" -> {', '.join(str(x) for x in smp['issues'][:2])}"
        print(
            f"{pair:<10} | {b['tested']:>4} | {b['dogru']:>5} | {b['yanlis']:>6} | "
            f"{b['cop']:>4} | {b['lang_err']:>4} | {b['grammar_err']:>5} | {note[:44]}"
        )
    print("-" * 96)
    print(
        f"{'TOPLAM':<10} | {totals['tested']:>4} | {totals['dogru']:>5} | {totals['yanlis']:>6} | "
        f"{totals['cop']:>4} | {totals['lang_err']:>4} | {totals['grammar_err']:>5} | Pairs={len(by_pair)}"
    )
    return {"totals": totals, "pairs": list(by_pair.values())}


def print_per_pair_summary(records: list[tuple[dict[str, Any], dict[str, Any]]]) -> dict[str, Any]:
    by_pair: dict[str, dict[str, Any]] = {}
    totals = {"tested": 0, "dogru": 0, "yanlis": 0, "cop": 0, "lang_err": 0, "grammar_err": 0}
    for orig, prop in records:
        _record_pair_outcome(by_pair, totals, orig, prop, sample_cap=2)
    return print_pair_aggregate_summary(by_pair, totals)


def _load_sweep_state(path: Path, reset: bool = False) -> dict[str, Any]:
    if not reset and path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("pairs"), dict):
            return data
    return {"meta": {"createdAt": utc_now(), "updatedAt": utc_now()}, "pairs": {}}


def _append_jsonl(lock: threading.Lock, path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(event, ensure_ascii=False) + "\n"
    with lock:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line)


def _build_nim_payload_item(r: dict[str, Any]) -> dict[str, Any]:
    return {
        "termId": r["auditKey"],
        "text": r["text"],
        "cleanHint": r.get("clean_hint"),
        "detectedAnomalies": r.get("clean_reasons") or [],
        "lang": r["lang"],
        "targetLang": r.get("targetLang"),
        "partOfSpeech": r.get("partOfSpeech"),
        "gender": r.get("gender"),
        "article": r.get("article"),
        "plural": r.get("plural"),
        "level": r.get("level"),
        "script": r.get("script"),
        "lemma": r.get("lemma"),
        "primaryDomain": r.get("primaryDomain"),
        "definitions": r.get("definitions", [])[:4],
    }


def _apply_sql_item_tx(
    conn,
    orig: dict[str, Any],
    raw_prop: dict[str, Any],
    dry_run: bool,
) -> tuple[str, str, dict[str, Any]]:
    """Run guarded Stage-2 SQL action on one entry; returns (stat_key, apply_action_str, guarded_prop)."""
    prop = _apply_stage2_guards(orig, raw_prop)
    orig_text = orig.get("text") or ""
    tid = orig.get("termId")
    act_upper = str(prop.get("action") or "KEEP").upper()

    if act_upper == "DELETE":
        res = _run_tx_with_retry(
            conn,
            purge_junk_row_tx,
            orig,
            [prop.get("reason") or "llm_flagged_junk"],
            dry_run=dry_run,
        )
        return ("dry_run" if dry_run else "deleted"), res, prop
    if not orig.get("inDb", True) or str(tid).startswith(("pdf:", "json:")):
        res = _run_tx_with_retry(conn, upsert_external_term_tx, orig, prop, dry_run=dry_run)
        if dry_run:
            return "dry_run", res, prop
        return ("inserted" if "inserted" in res else "updated"), res, prop
    if act_upper == "KEEP" and str(prop.get("verdict") or "").upper() == "CORRECT":
        res = _run_tx_with_retry(conn, apply_fix_row_tx, tid, {**orig, **prop, "text": orig_text}, dry_run=dry_run)
        return ("dry_run" if dry_run else "kept"), res, prop

    res = _run_tx_with_retry(conn, apply_fix_row_tx, tid, prop, dry_run=dry_run)
    if dry_run:
        return "dry_run", res, prop
    return ("merged" if "merged_into" in res else "updated"), res, prop


def run_pair_sweep_engine(
    args: argparse.Namespace,
    db_url: str,
    nim: NimPool,
    lang_pair_filter: tuple[str, str] | None,
    t_start: float,
) -> int:
    """Concurrent multi-pair wave queue: up to N workers each drain assigned (src, tgt) pairs chunk-by-chunk."""
    state_path = Path(args.state_file)
    if not state_path.is_absolute():
        state_path = Path.cwd() / state_path
    jsonl_path = Path(args.jsonl_log)
    if not jsonl_path.is_absolute():
        jsonl_path = Path.cwd() / jsonl_path

    state = _load_sweep_state(state_path, reset=args.reset_state or args.dry_run)

    init_conn = psycopg2.connect(db_url)
    init_conn.autocommit = False
    try:
        all_pairs = discover_db_lang_pairs(init_conn, lang_pair_filter=lang_pair_filter)
    finally:
        init_conn.close()

    if not all_pairs:
        print("[Sweep] No language pairs found in PostgreSQL.", flush=True)
        return 0

    pair_queue: queue.Queue[tuple[str, str, int]] = queue.Queue()
    skipped_done = 0
    for src_lang, tgt_lang, total_cnt in all_pairs:
        pkey = f"{src_lang}->{tgt_lang}"
        pstate = state["pairs"].setdefault(
            pkey,
            {
                "last_id": "",
                "total": total_cnt,
                "processed": 0,
                "updated": 0,
                "merged": 0,
                "deleted": 0,
                "kept": 0,
                "dry_run": 0,
                "failed": 0,
                "done": False,
            },
        )
        pstate["total"] = total_cnt
        if args.recheck:
            pstate["done"] = False
            pstate["last_id"] = ""
        if pstate.get("done") and not args.dry_run:
            skipped_done += 1
        else:
            pair_queue.put((src_lang, tgt_lang, total_cnt))

    if not args.dry_run:
        save_json_atomic(state_path, state)

    pending_pairs = pair_queue.qsize()
    print(
        f"[Sweep] Discovered {len(all_pairs)} pairs (queued={pending_pairs}, already_done={skipped_done}) | "
        f"workers={args.workers}, limit_per_pair={'unlimited' if args.limit == 0 else args.limit}, "
        f"chunk_size={args.chunk_size}, batch_size={args.batch_size}, dry_run={args.dry_run}",
        flush=True,
    )
    if pending_pairs == 0:
        return 0

    stop_event = threading.Event()
    prev_sigint = None
    prev_sigterm = None

    def _handle_stop(signum, _frame):
        print(f"\n[Sweep] Received signal {signum}; draining active transactions and saving checkpoint...", flush=True)
        stop_event.set()

    try:
        prev_sigint = signal.getsignal(signal.SIGINT)
        prev_sigterm = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGINT, _handle_stop)
        signal.signal(signal.SIGTERM, _handle_stop)
    except ValueError:
        pass

    state_lock = threading.Lock()
    jsonl_lock = threading.Lock()
    agg_by_pair: dict[str, dict[str, Any]] = {}
    agg_totals = {"tested": 0, "dogru": 0, "yanlis": 0, "cop": 0, "lang_err": 0, "grammar_err": 0}
    sample_items_for_report: list[dict[str, Any]] = []
    global_stats = {"updated": 0, "merged": 0, "deleted": 0, "kept": 0, "dry_run": 0, "failed": 0}

    def pair_worker(worker_num: int) -> None:
        wconn = psycopg2.connect(db_url)
        wconn.autocommit = False
        try:
            while not stop_event.is_set():
                try:
                    src_lang, tgt_lang, total_cnt = pair_queue.get_nowait()
                except queue.Empty:
                    break

                pkey = f"{src_lang}->{tgt_lang}"
                try:
                    with state_lock:
                        pstate = state["pairs"][pkey]
                        last_id = pstate.get("last_id") or ""
                    local_seen = 0

                    while not stop_event.is_set():
                        if args.limit > 0 and local_seen >= args.limit:
                            break
                        req_lim = args.chunk_size if args.limit <= 0 else min(args.chunk_size, args.limit - local_seen)
                        chunk = fetch_pair_chunk(
                            wconn, src_lang, tgt_lang, last_id=last_id, chunk_size=req_lim, recheck=args.recheck
                        )
                        if not chunk:
                            with state_lock:
                                state["pairs"][pkey]["done"] = True
                                state["pairs"][pkey]["completedAt"] = utc_now()
                                if not args.dry_run:
                                    save_json_atomic(state_path, state)
                            print(f"  [Pair DONE W#{worker_num}] {pkey} (total={total_cnt})", flush=True)
                            break

                        last_id = chunk[-1]["termId"]
                        nim_queue: list[dict[str, Any]] = []
                        chunk_purged = 0

                        for r in chunk:
                            local_seen += 1
                            act, cleaned_text, reasons = classify_headword(
                                r["text"], lang=r["lang"], pos=r.get("partOfSpeech"), definitions=r.get("definitions")
                            )
                            lang_anomalies = detect_language_anomaly(
                                r["text"], r["lang"], article=r.get("article"), definitions=r.get("definitions")
                            )
                            if act == "delete" and args.purge_junk:
                                try:
                                    res = _run_tx_with_retry(
                                        wconn, purge_junk_row_tx, r, reasons, dry_run=args.dry_run
                                    )
                                    prop_del = {"action": "DELETE", "verdict": "DELETE", "issues": reasons, "reason": ",".join(reasons)}
                                    with state_lock:
                                        _record_pair_outcome(agg_by_pair, agg_totals, r, prop_del)
                                        skey = "dry_run" if args.dry_run else "deleted"
                                        state["pairs"][pkey][skey] += 1
                                        state["pairs"][pkey]["processed"] += 1
                                        global_stats[skey] += 1
                                    chunk_purged += 1
                                    if not args.dry_run:
                                        _append_jsonl(jsonl_lock, jsonl_path, {
                                            "type": "purge", "pair": pkey, "termId": r["termId"],
                                            "text": r["text"], "reasons": reasons, "action": res, "at": utc_now(),
                                            "snapshot": r,
                                        })
                                except Exception as exc:
                                    with state_lock:
                                        state["pairs"][pkey]["failed"] += 1
                                        global_stats["failed"] += 1
                                    print(f"  [ERROR W#{worker_num} {pkey}] purge {r['termId']}: {exc}", flush=True)
                                continue

                            if act == "clean":
                                r["clean_hint"] = cleaned_text
                                r["clean_reasons"] = reasons + lang_anomalies
                            elif lang_anomalies:
                                r["clean_reasons"] = lang_anomalies
                            nim_queue.append(r)

                        chunk_fixed = 0
                        for b_idx in range(0, len(nim_queue), max(1, args.batch_size)):
                            if stop_event.is_set():
                                break
                            batch = nim_queue[b_idx : b_idx + max(1, args.batch_size)]
                            payload = [_build_nim_payload_item(x) for x in batch]
                            user_prompt = "Audit language & 12-lang grammar for each entry. Return ONLY a JSON array:\n" + json.dumps(payload, ensure_ascii=False)
                            try:
                                parsed, chan_id = nim.chat_json(SYSTEM_PROMPT, user_prompt)
                            except Exception as exc:
                                with state_lock:
                                    state["pairs"][pkey]["failed"] += len(batch)
                                    global_stats["failed"] += len(batch)
                                print(f"  [NIM ERROR W#{worker_num} {pkey}]: {exc}", flush=True)
                                continue

                            if isinstance(parsed, dict):
                                parsed = parsed.get("items") or parsed.get("results") or [parsed]
                            by_id = {str(x.get("termId")): x for x in parsed if isinstance(x, dict) and x.get("termId")}

                            for r in batch:
                                fix = by_id.get(r["auditKey"])
                                if not fix:
                                    continue
                                try:
                                    skey, apply_res, g_prop = _apply_sql_item_tx(wconn, r, fix, dry_run=args.dry_run)
                                    chunk_fixed += 1
                                    with state_lock:
                                        _record_pair_outcome(agg_by_pair, agg_totals, r, g_prop)
                                        state["pairs"][pkey][skey] += 1
                                        state["pairs"][pkey]["processed"] += 1
                                        global_stats[skey] += 1
                                        if args.report_file and len(sample_items_for_report) < 500:
                                            sample_items_for_report.append({
                                                "pair": pkey, "text": r.get("text"), "lang": r.get("lang"),
                                                "verdict": g_prop.get("verdict"), "action": g_prop.get("action"),
                                                "langValid": g_prop.get("langValid"), "grammarValid": g_prop.get("grammarValid"),
                                                "issues": g_prop.get("issues"), "fixedText": g_prop.get("text"),
                                                "fixedLang": g_prop.get("lang"), "primaryDomain": g_prop.get("primaryDomain"),
                                                "channel": chan_id, "applyAction": apply_res,
                                            })
                                    if not args.dry_run:
                                        _append_jsonl(jsonl_lock, jsonl_path, {
                                            "type": "fix", "pair": pkey, "termId": r["termId"],
                                            "text": r["text"], "channel": chan_id, "applyAction": apply_res,
                                            "proposed": g_prop, "at": utc_now(),
                                        })
                                except Exception as exc:
                                    with state_lock:
                                        state["pairs"][pkey]["failed"] += 1
                                        global_stats["failed"] += 1
                                    print(f"  [SQL ERROR W#{worker_num} {pkey}] {r['termId']}: {exc}", flush=True)

                        with state_lock:
                            state["pairs"][pkey]["last_id"] = last_id
                            if not args.dry_run:
                                save_json_atomic(state_path, state)
                        print(
                            f"  [Sweep W#{worker_num} {pkey}] chunk={len(chunk)} purged={chunk_purged} "
                            f"audited={chunk_fixed} total_seen={local_seen}",
                            flush=True,
                        )
                finally:
                    pair_queue.task_done()
        finally:
            wconn.close()

    n_workers = max(1, min(args.workers, pending_pairs))
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futs = [pool.submit(pair_worker, wid + 1) for wid in range(n_workers)]
        for fut in as_completed(futs):
            fut.result()

    if prev_sigint is not None:
        try:
            signal.signal(signal.SIGINT, prev_sigint)
            signal.signal(signal.SIGTERM, prev_sigterm)
        except ValueError:
            pass

    if not args.dry_run:
        save_json_atomic(state_path, state)

    print(f"[Sweep Stage 2] Summary: {global_stats}", flush=True)
    if agg_by_pair:
        summary = print_pair_aggregate_summary(agg_by_pair, agg_totals)
        if args.report_file:
            rep_path = Path(args.report_file)
            if not rep_path.is_absolute():
                rep_path = Path.cwd() / rep_path
            rep_path.parent.mkdir(parents=True, exist_ok=True)
            rep_path.write_text(
                json.dumps(
                    {
                        "meta": {
                            "createdAt": utc_now(),
                            "source": args.source,
                            "mode": args.mode,
                            "workers": n_workers,
                            "dryRun": args.dry_run,
                            "elapsedSec": round(time.perf_counter() - t_start, 2),
                            "channels": [{k: v for k, v in ch.items() if k != "key"} for ch in nim.channels],
                            "globalStats": global_stats,
                        },
                        "summary": summary,
                        "items": sample_items_for_report,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f"[Report] Saved sweep summary to {rep_path}", flush=True)

    if nim.channels:
        print(f"\n{nim.format_channel_stats()}", flush=True)
    print(f"[Total Elapsed] {time.perf_counter() - t_start:.2f}s", flush=True)
    return 1 if global_stats["failed"] > 0 else 0


# ─── Main CLI ──────────────────────────────────────────────────────────────────

def _parse_lang_pair(raw: str | None) -> tuple[str, str] | None:
    if not raw:
        return None
    parts = re.split(r"[-:>→/]+", raw.strip().upper())
    if len(parts) != 2 or parts[0] not in VALID_LANGS or parts[1] not in VALID_LANGS:
        raise ValueError(f"Invalid --lang-pair {raw!r}; expected format like DE-TR.")
    return parts[0], parts[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="LexiForge NVIDIA NIM -> Glossa PostgreSQL Manual Audit & Repair CLI")
    parser.add_argument("--fetch-models", action="store_true", help="Fetch & live-probe NVIDIA NIM /v1/models and exit.")
    parser.add_argument("--source", choices=["db", "pdf", "json"], default="db", help="Input audit source (default: db).")
    parser.add_argument("--mode", choices=["junk", "anomalies", "all", "pairs", "sweep"], default="anomalies", help="Audit mode (default: anomalies).")
    parser.add_argument("--per-pair", type=int, default=3, help="Sample size per language pair in --mode pairs (default: 3).")
    parser.add_argument("--chunk-size", type=int, default=50, help="Keyset chunk size for sweep / unlimited streaming (default: 50).")
    parser.add_argument("--lang-pair", help="Filter single language pair like DE-TR.")
    parser.add_argument("--offset", type=int, default=0, help="Pagination offset for --mode all|anomalies.")
    parser.add_argument("--pdf-path", default="/root/Glossa/public/downloads/Glossa_Buyuk_Sozluk.pdf", help="PDF path when --source pdf.")
    parser.add_argument("--input", help="JSON path when --source json.")
    parser.add_argument("--out-json", help="Optional cleaned JSON output file when --source json.")
    parser.add_argument("--report-file", help="Optional JSON summary report output file (safe during --dry-run).")
    parser.add_argument("--state-file", default="logs/lexiforge_sweep_state.json", help="Resumable per-pair checkpoint state file for --mode sweep.")
    parser.add_argument("--jsonl-log", default="logs/lexiforge_sweep_audit.jsonl", help="Append-only JSONL audit log for --mode sweep.")
    parser.add_argument("--recheck", action="store_true", help="Re-audit already enriched terms (qualityScore >= 85) in --mode sweep.")
    parser.add_argument("--reset-state", action="store_true", help="Reset existing --state-file before starting --mode sweep.")
    parser.add_argument("--env-file", help="Path to .env containing DATABASE_URL and NVIDIA_API_KEYS.")
    parser.add_argument("--purge-junk", action="store_true", help="Delete verified non-lexical junk with JSON backup.")
    parser.add_argument("--limit", type=int, default=50, help="Max terms (or per pair in --mode sweep); 0 = unlimited. Default: 50.")
    parser.add_argument("--batch-size", type=int, default=5, help="Terms per NVIDIA NIM request. Default: 5.")
    parser.add_argument("--workers", type=int, default=10, help="Concurrent NIM workers across (key x model) channels. Default: 10.")
    parser.add_argument("--audit-file", default="logs/lexiforge_nim_sql_audit.json", help="Resumable JSON audit & backup log.")
    parser.add_argument("--generate-only", action="store_true", help="Generate fixes into audit JSON without SQL writes.")
    parser.add_argument("--apply-only", action="store_true", help="Apply pending items from --audit-file to PostgreSQL.")
    parser.add_argument("--dry-run", action="store_true", help="Simulate all SQL operations with rollback (no DB or audit mutations).")
    args = parser.parse_args()

    t_start = time.perf_counter()
    lang_pair_filter = _parse_lang_pair(args.lang_pair)
    db_url, api_keys = load_env_values(Path(args.env_file) if args.env_file else None)
    nim = NimPool(api_keys=api_keys)

    if args.fetch_models or not args.apply_only:
        probe_res = nim.fetch_and_probe()
        print("\n=== Probed NVIDIA NIM Models ===", flush=True)
        for r in probe_res:
            print(f"  {r['model']:45s} ok={str(r['ok']):5s} {r['latency_ms']:>7}ms status={r['status']}")
        print(f"=== Built {len(nim.channels)} parallel channels ({len(nim.models)} models x {len(nim.api_keys)} keys) ===", flush=True)
        if args.fetch_models:
            return 0

    if args.mode == "sweep":
        if args.source != "db":
            raise ValueError("--mode sweep operates directly on PostgreSQL (--source db).")
        return run_pair_sweep_engine(args, db_url, nim, lang_pair_filter, t_start)

    # Unlimited keyset streaming for --source db --mode anomalies|all --limit 0 (no silent 500 cap)
    if args.source == "db" and args.mode in {"anomalies", "all"} and args.limit == 0 and not args.apply_only:
        conn_stream = psycopg2.connect(db_url)
        conn_stream.autocommit = False
        stream_by_pair: dict[str, dict[str, Any]] = {}
        stream_totals = {"tested": 0, "dogru": 0, "yanlis": 0, "cop": 0, "lang_err": 0, "grammar_err": 0}
        stream_stats = {"updated": 0, "merged": 0, "deleted": 0, "kept": 0, "dry_run": 0, "failed": 0}
        jsonl_path = Path(args.jsonl_log)
        if not jsonl_path.is_absolute():
            jsonl_path = Path.cwd() / jsonl_path
        jlock = threading.Lock()
        last_id = ""
        chunk_num = 0
        try:
            fetcher = fetch_anomaly_db_terms if args.mode == "anomalies" else fetch_all_db_terms
            while True:
                rows = fetcher(
                    conn_stream,
                    limit=max(1, args.chunk_size),
                    lang_pair_filter=lang_pair_filter,
                    last_id=last_id,
                )
                if not rows:
                    break
                chunk_num += 1
                last_id = rows[-1]["termId"]
                nim_items: list[dict[str, Any]] = []
                for r in rows:
                    r["auditKey"] = r["termId"]
                    act, cleaned_text, reasons = classify_headword(
                        r["text"], lang=r["lang"], pos=r.get("partOfSpeech"), definitions=r.get("definitions")
                    )
                    lang_anomalies = detect_language_anomaly(
                        r["text"], r["lang"], article=r.get("article"), definitions=r.get("definitions")
                    )
                    if act == "delete" and args.purge_junk:
                        res = _run_tx_with_retry(conn_stream, purge_junk_row_tx, r, reasons, dry_run=args.dry_run or args.generate_only)
                        prop_del = {"action": "DELETE", "verdict": "DELETE", "issues": reasons}
                        _record_pair_outcome(stream_by_pair, stream_totals, r, prop_del)
                        stream_stats["dry_run" if (args.dry_run or args.generate_only) else "deleted"] += 1
                        if not (args.dry_run or args.generate_only):
                            _append_jsonl(jlock, jsonl_path, {"type": "purge", "termId": r["termId"], "text": r["text"], "reasons": reasons, "action": res, "at": utc_now()})
                        continue
                    if act == "clean":
                        r["clean_hint"] = cleaned_text
                        r["clean_reasons"] = reasons + lang_anomalies
                    elif lang_anomalies:
                        r["clean_reasons"] = lang_anomalies
                    nim_items.append(r)

                batches = [nim_items[i : i + max(1, args.batch_size)] for i in range(0, len(nim_items), max(1, args.batch_size))]
                chunk_results: list[tuple[dict[str, Any], dict[str, Any], str]] = []
                rlock = threading.Lock()

                def _run_sbatch(b: list[dict[str, Any]]) -> None:
                    payload = [_build_nim_payload_item(x) for x in b]
                    parsed, cid = nim.chat_json(
                        SYSTEM_PROMPT,
                        "Audit language & 12-lang grammar for each entry. Return ONLY a JSON array:\n" + json.dumps(payload, ensure_ascii=False),
                    )
                    if isinstance(parsed, dict):
                        parsed = parsed.get("items") or parsed.get("results") or [parsed]
                    by_id = {str(x.get("termId")): x for x in parsed if isinstance(x, dict) and x.get("termId")}
                    with rlock:
                        for x in b:
                            if x["auditKey"] in by_id:
                                chunk_results.append((x, by_id[x["auditKey"]], cid))

                if batches:
                    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
                        for fut in as_completed([pool.submit(_run_sbatch, b) for b in batches]):
                            fut.result()

                if not args.generate_only:
                    for orig, fix, cid in chunk_results:
                        try:
                            skey, ares, gprop = _apply_sql_item_tx(conn_stream, orig, fix, dry_run=args.dry_run)
                            _record_pair_outcome(stream_by_pair, stream_totals, orig, gprop)
                            stream_stats[skey] += 1
                            if not args.dry_run:
                                _append_jsonl(jlock, jsonl_path, {"type": "fix", "termId": orig["termId"], "channel": cid, "applyAction": ares, "proposed": gprop, "at": utc_now()})
                        except Exception as exc:
                            stream_stats["failed"] += 1
                            print(f"  [Stream SQL ERROR] {orig['termId']}: {exc}", flush=True)
                print(f"[Stream Chunk {chunk_num}] last_id={last_id} stats={stream_stats}", flush=True)
        finally:
            conn_stream.close()
        if stream_by_pair:
            print_pair_aggregate_summary(stream_by_pair, stream_totals)
        if nim.channels:
            print(f"\n{nim.format_channel_stats()}", flush=True)
        print(f"[Total Elapsed] {time.perf_counter() - t_start:.2f}s", flush=True)
        return 1 if stream_stats["failed"] > 0 else 0

    audit_path = Path(args.audit_file)
    if not audit_path.is_absolute():
        audit_path = Path.cwd() / audit_path
    audit = {"meta": {"createdAt": utc_now(), "updatedAt": utc_now(), "modelsUsed": []}, "items": {}, "purged": {}} if args.dry_run else load_audit(audit_path)

    conn = psycopg2.connect(db_url)
    conn.autocommit = False
    verified_records: list[tuple[dict[str, Any], dict[str, Any]]] = []

    try:
        if not args.apply_only:
            rows: list[dict[str, Any]] = []
            if args.source == "json":
                if not args.input:
                    raise ValueError("--input JSON path is required when --source json.")
                raw_json = json.loads(Path(args.input).read_text(encoding="utf-8"))
                if not isinstance(raw_json, list):
                    raise ValueError("JSON input must be a list of term objects.")
                cleaned_pre: list[dict[str, Any]] = []
                for idx, entry in enumerate(raw_json):
                    if not isinstance(entry, dict):
                        continue
                    head = (entry.get("text") or entry.get("term") or "").strip()
                    if not head:
                        continue
                    src_l = (entry.get("lang") or "DE").upper()
                    defs = entry.get("definitions") or []
                    if isinstance(defs, list) and defs and isinstance(defs[0], dict):
                        tgt_l = (defs[0].get("language") or ("TR" if src_l != "TR" else "EN")).upper()
                    else:
                        tgt_l = "TR" if src_l != "TR" else "EN"
                    norm = sanitize_grammar_fields({**entry, "text": head, "lang": src_l}, src_lang=src_l)
                    norm.update({
                        "termId": f"json:{idx + 1}",
                        "auditKey": f"json:{idx + 1}",
                        "inDb": False,
                        "source": "json_import",
                        "targetLang": tgt_l,
                        "pair": f"{src_l}->{tgt_l}",
                        "definitions": defs,
                    })
                    rows.append(norm)
                    cleaned_pre.append(norm)
                if args.out_json and not args.dry_run:
                    Path(args.out_json).write_text(json.dumps(cleaned_pre, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"[JSON Source] Loaded {len(rows)} entries from {args.input}.", flush=True)

            elif args.source == "pdf":
                p = Path(args.pdf_path)
                if not p.exists():
                    raise FileNotFoundError(f"PDF not found: {p}")
                pdf_per_pair = args.per_pair if args.mode == "pairs" else 0
                print(f"[PDF Scan] Extracting UTF-8 entries from {p} (per_pair={pdf_per_pair}, pair={lang_pair_filter})...", flush=True)
                rows = parse_glossa_pdf_entries(p, max_per_pair=pdf_per_pair, lang_pair_filter=lang_pair_filter)
                # Tag rows already present in DB so existing records can be updated cleanly
                if rows:
                    with conn.cursor() as cur:
                        unique_texts = list({r["text"] for r in rows})
                        cur.execute('SELECT id, text, lang FROM "Term" WHERE text = ANY(%s)', (unique_texts,))
                        existing_map = {(t, l): tid for tid, t, l in cur.fetchall()}
                        for idx, r in enumerate(rows):
                            match_tid = existing_map.get((r["text"], r["lang"]))
                            if match_tid:
                                r["termId"] = match_tid
                                r["inDb"] = True
                            r["auditKey"] = f"{r['termId']}@{r['targetLang']}#{idx}"
                print(f"[PDF Scan] Extracted {len(rows)} structured entries across PDF chapters.", flush=True)

            else:
                if args.mode == "junk":
                    rows = fetch_suspicious_db_terms(conn, limit=0)
                elif args.mode == "anomalies":
                    rows = fetch_anomaly_db_terms(conn, limit=args.limit, offset=args.offset, lang_pair_filter=lang_pair_filter)
                elif args.mode == "all":
                    rows = fetch_all_db_terms(conn, limit=args.limit, offset=args.offset, lang_pair_filter=lang_pair_filter)
                elif args.mode == "pairs":
                    rows = fetch_stratified_lang_pair_terms(conn, per_pair=args.per_pair, lang_pair_filter=lang_pair_filter)

            delete_candidates: list[tuple[dict[str, Any], list[str]]] = []
            audit_candidates: list[dict[str, Any]] = []

            for r in rows:
                akey = r.get("auditKey") or r["termId"]
                r["auditKey"] = akey
                act, cleaned_text, reasons = classify_headword(
                    r["text"], lang=r["lang"], pos=r.get("partOfSpeech"), definitions=r.get("definitions")
                )
                lang_anomalies = detect_language_anomaly(
                    r["text"], r["lang"], article=r.get("article"), definitions=r.get("definitions")
                )
                if act == "delete" and (args.purge_junk or args.mode != "pairs"):
                    if args.purge_junk:
                        delete_candidates.append((r, reasons))
                        if args.dry_run:
                            verified_records.append((r, {"action": "DELETE", "verdict": "DELETE", "issues": reasons}))
                        continue
                if act == "clean":
                    r["clean_hint"] = cleaned_text
                    r["clean_reasons"] = reasons + lang_anomalies
                elif lang_anomalies:
                    r["clean_reasons"] = lang_anomalies
                if args.dry_run or akey not in audit["items"]:
                    audit_candidates.append(r)

            print(
                f"[Audit Scan] source={args.source} mode={args.mode} | Scanned={len(rows)} | "
                f"Auto-Delete Junk={len(delete_candidates)} | NIM Audit Candidates={len(audit_candidates)}",
                flush=True,
            )

            if args.purge_junk and delete_candidates:
                purged_count = 0
                for row, reasons in delete_candidates:
                    tid = row["termId"]
                    res = purge_junk_row_tx(conn, row, reasons, dry_run=args.dry_run or args.generate_only)
                    if args.dry_run or args.generate_only:
                        print(f"  [DRY-PURGE] {row['text']!r} ({row.get('pair') or row['lang']}) -> {reasons}")
                    else:
                        audit["purged"][tid] = {"snapshot": row, "reasons": reasons, "action": res, "purgedAt": utc_now()}
                        purged_count += 1
                if not (args.dry_run or args.generate_only) and purged_count:
                    save_json_atomic(audit_path, audit)
                print(f"[Purge] Processed {len(delete_candidates)} junk rows (deleted={purged_count}, dry_run={args.dry_run or args.generate_only}).", flush=True)

            if args.limit > 0 and args.mode != "pairs":
                audit_candidates = audit_candidates[: args.limit]
            elif args.limit > 0 and args.mode == "pairs" and args.source == "pdf":
                # Allow full pair coverage unless explicit cap > 0 conflicts? Keep all extracted per-pair unless tiny limit
                pass

            if audit_candidates:
                batches = [audit_candidates[i : i + max(1, args.batch_size)] for i in range(0, len(audit_candidates), max(1, args.batch_size))]
                lock = threading.Lock()
                models_used = set(audit["meta"].get("modelsUsed") or [])
                print(f"[NIM Stage 1] Dispatching {len(batches)} batches ({len(audit_candidates)} terms) across {max(1, args.workers)} workers...", flush=True)

                def run_batch(bidx: int, batch: list[dict[str, Any]]) -> None:
                    payload = [
                        {
                            "termId": r["auditKey"],
                            "text": r["text"],
                            "cleanHint": r.get("clean_hint"),
                            "detectedAnomalies": r.get("clean_reasons") or [],
                            "lang": r["lang"],
                            "targetLang": r.get("targetLang"),
                            "partOfSpeech": r.get("partOfSpeech"),
                            "gender": r.get("gender"),
                            "article": r.get("article"),
                            "plural": r.get("plural"),
                            "level": r.get("level"),
                            "script": r.get("script"),
                            "lemma": r.get("lemma"),
                            "primaryDomain": r.get("primaryDomain"),
                            "definitions": r.get("definitions", [])[:4],
                        }
                        for r in batch
                    ]
                    user_prompt = "Audit language & 12-lang grammar for each entry. Return ONLY a JSON array:\n" + json.dumps(payload, ensure_ascii=False)
                    parsed, chan_id = nim.chat_json(SYSTEM_PROMPT, user_prompt)
                    if isinstance(parsed, dict):
                        parsed = parsed.get("items") or parsed.get("results") or [parsed]
                    by_id = {str(x.get("termId")): x for x in parsed if isinstance(x, dict) and x.get("termId")}

                    with lock:
                        models_used.add(chan_id.split("#")[0])
                        for r in batch:
                            akey = r["auditKey"]
                            fix = by_id.get(akey)
                            if fix:
                                verified_records.append((r, fix))
                                audit["items"][akey] = {
                                    "termId": r["termId"],
                                    "auditKey": akey,
                                    "applied": False,
                                    "original": r,
                                    "proposed": fix,
                                    "channel": chan_id,
                                    "generatedAt": utc_now(),
                                }
                        audit["meta"]["modelsUsed"] = sorted(models_used)
                        if not args.dry_run:
                            save_json_atomic(audit_path, audit)
                        print(f"  [NIM Batch {bidx + 1}/{len(batches)}] {chan_id} -> {len(by_id)}/{len(batch)} items", flush=True)

                with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
                    for fut in as_completed([pool.submit(run_batch, i, b) for i, b in enumerate(batches)]):
                        fut.result()

        if not args.generate_only:
            pending = [(akey, item) for akey, item in audit["items"].items() if not item.get("applied")]
            print(f"[SQL Stage 2] Simulating/Applying {len(pending)} fixes (dry_run={args.dry_run})...", flush=True)
            stats = {"updated": 0, "merged": 0, "inserted": 0, "deleted": 0, "kept": 0, "dry_run": 0, "failed": 0}
            for akey, item in pending:
                orig = item.get("original") or {}
                prop = _apply_stage2_guards(orig, item.get("proposed") or {})
                orig_text = orig.get("text") or ""
                tid = orig.get("termId") or item.get("termId")
                item["proposed"] = prop
                for vr_idx, (vo, _vp) in enumerate(verified_records):
                    if vo.get("auditKey") == akey:
                        verified_records[vr_idx] = (vo, prop)
                        break

                try:
                    skey, res, _ = _apply_sql_item_tx(conn, orig, prop, dry_run=args.dry_run)
                    stats[skey] += 1

                    if not args.dry_run:
                        item["applied"] = True
                        item["applyAction"] = res
                        item["appliedAt"] = utc_now()
                        save_json_atomic(audit_path, audit)
                except Exception as exc:
                    stats["failed"] += 1
                    item["lastError"] = str(exc)[:240]
                    if not args.dry_run:
                        save_json_atomic(audit_path, audit)
                    print(f"  [ERROR] {akey}: {exc}", flush=True)
            print(f"[SQL Stage 2] Summary: {stats}", flush=True)
            if stats["failed"] > 0:
                return 1

        if verified_records:
            summary = print_per_pair_summary(verified_records)
            if args.report_file:
                rep_path = Path(args.report_file)
                if not rep_path.is_absolute():
                    rep_path = Path.cwd() / rep_path
                rep_path.parent.mkdir(parents=True, exist_ok=True)
                summary_payload = {
                    "meta": {
                        "createdAt": utc_now(),
                        "source": args.source,
                        "mode": args.mode,
                        "workers": args.workers,
                        "dryRun": args.dry_run,
                        "elapsedSec": round(time.perf_counter() - t_start, 2),
                        "channels": [
                            {k: v for k, v in ch.items() if k != "key"} for ch in nim.channels
                        ],
                    },
                    "summary": summary,
                    "items": [
                        {
                            "pair": o.get("pair"),
                            "text": o.get("text"),
                            "lang": o.get("lang"),
                            "verdict": p.get("verdict"),
                            "action": p.get("action"),
                            "langValid": p.get("langValid"),
                            "grammarValid": p.get("grammarValid"),
                            "issues": p.get("issues"),
                            "fixedText": p.get("text"),
                            "fixedLang": p.get("lang"),
                            "primaryDomain": p.get("primaryDomain"),
                            "reason": p.get("reason"),
                        }
                        for o, p in verified_records
                    ],
                }
                rep_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"[Report] Saved per-pair summary to {rep_path}", flush=True)

        if nim.channels:
            print(f"\n{nim.format_channel_stats()}", flush=True)
            print(f"[Total Elapsed] {time.perf_counter() - t_start:.2f}s", flush=True)
    finally:
        conn.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
