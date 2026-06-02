#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run Logger Module / Çalıştırma Loglama Modülü
==========================================
Provides structured file logging and atomic JSON saving for workflow scripts.
İş akışı betikleri için yapılandırılmış dosya loglaması ve atomik JSON kaydetme sağlar.
"""
import orjson
import os
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional


def now_iso() -> str:
    """Returns current UTC ISO timestamp / Mevcut UTC ISO zaman damgasını döner."""
    return datetime.now(timezone.utc).isoformat()


def _safe_name(value: str) -> str:
    """Sanitizes names for safe file creation / Güvenli dosya oluşturma için isimleri temizler."""
    banned = '<>:"/\\|?*'
    out = "".join("_" if c in banned else c for c in value)
    return out.strip().replace(" ", "_")


def safe_replace(src: str, dst: str, max_retries: int = 10, delay: float = 0.1):
    """
    Safely replaces file, retrying on Windows permission block.
    Windows izin engellemelerine karşı yeniden deneyerek dosyayı güvenle değiştirir.
    """
    for i in range(max_retries):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if i == max_retries - 1:
                raise
            time.sleep(delay)


def save_json_atomic(path: str, payload: Dict[str, Any]):
    """
    Saves JSON atomically using temporary file and safe replace.
    Geçici dosya ve güvenli değiştirme kullanarak JSON'u atomik olarak kaydeder.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(orjson.dumps(payload, option=orjson.OPT_INDENT_2 | orjson.OPT_NON_STR_KEYS))
    safe_replace(tmp, path)


def load_json_with_default(path: str, default: Dict[str, Any]) -> Dict[str, Any]:
    """
    Loads JSON with fallback default dict if file missing or corrupt.
    Dosya yoksa veya bozuksa yedek varsayılan sözlükle JSON'u yükler.
    """
    if not os.path.exists(path):
        return dict(default)
    try:
        with open(path, "rb") as f:
            loaded = orjson.loads(f.read())
        if isinstance(loaded, dict):
            merged = dict(default)
            merged.update(loaded)
            return merged
    except Exception:
        pass
    return dict(default)


class RunLogger:
    """
    Structured event logger logging to both JSONL files and summaries.
    Hem JSONL dosyalarına hem de özetlere log yazan yapılandırılmış olay loglayıcı.
    """
    def __init__(self, task: str, source: str = "", log_dir: str = os.path.join("data", "logs")):
        self.task = task
        self.source = source or "global"
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_source = _safe_name(self.source)
        self.run_id = f"{task}_{safe_source}_{ts}"
        self.log_path = os.path.join(self.log_dir, f"{self.run_id}.jsonl")
        self.summary_path = os.path.join(self.log_dir, f"{self.run_id}_summary.json")
        self._lock = threading.Lock()
        self._counts = {"INFO": 0, "WARN": 0, "ERROR": 0}
        self._started_at = now_iso()

        self.info("run_started", message="Run initialized / Çalıştırma başlatıldı", source=self.source)

    def _write(self, level: str, event: str, **data):
        row = {
            "ts": now_iso(),
            "level": level,
            "event": event,
            "task": self.task,
            "source": self.source,
            "run_id": self.run_id,
            "data": data,
        }
        with self._lock:
            self._counts[level] = self._counts.get(level, 0) + 1
            with open(self.log_path, "ab") as f:
                f.write(orjson.dumps(row) + b"\n")

    def info(self, event: str, **data):
        """Logs INFO level event / INFO düzeyinde olay günlükler."""
        self._write("INFO", event, **data)

    def warn(self, event: str, **data):
        """Logs WARN level event / WARN düzeyinde olay günlükler."""
        self._write("WARN", event, **data)

    def error(self, event: str, **data):
        """Logs ERROR level event / ERROR düzeyinde olay günlükler."""
        self._write("ERROR", event, **data)

    def close(self, **extra_summary):
        """
        Finalizes logging and writes summary file.
        Loglamayı sonlandırır ve özet dosyasını yazar.
        """
        self.info("run_finished", message="Run completed / Çalıştırma tamamlandı")
        summary = {
            "run_id": self.run_id,
            "task": self.task,
            "source": self.source,
            "started_at": self._started_at,
            "finished_at": now_iso(),
            "log_path": self.log_path,
            "counts": self._counts,
        }
        if extra_summary:
            summary["summary"] = extra_summary
        save_json_atomic(self.summary_path, summary)
