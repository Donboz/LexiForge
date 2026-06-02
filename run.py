#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Glossa Orchestrator / Glossa Orkestratör
=======================================
Main entry point to run various extraction, enrichment, translation, and clean up scripts.

Çeşitli çıkarma, zenginleştirme, çeviri ve temizleme scriptlerini çalıştırmak için ana giriş noktası.
"""

import os
import sys
import subprocess

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from handlers.selector import select_item_interactive

CHOICES = [
    "1. Extractor / Veri Çıkarıcı (extractor)",
    "2. Enricher / Zenginleştirici (enricher)",
    "3. Translator / Çevirici (translator)",
    "4. Analyze Logs / Log Analizi (analyze_logs)",
    "5. Clean Raw Text / Metin Temizleme (clean)",
    "6. Merge Translated / Çevirileri Birleştir (merge)",
    "7. Clean Meanings / Anlamları Temizle (clean_meanings)",
    "8. Splitter / Ayırıcı (splitter)",
    "9. Test Redis Connection / Redis Bağlantı Testi (test_redis)",
    "10. Quit / Çıkış (quit)",
]


def run_command(cmd_args):
    print("Running / Çalıştırılıyor:", " ".join(cmd_args))
    proc = subprocess.run(cmd_args)
    return proc.returncode


def main():
    choice = select_item_interactive(CHOICES, "Select a task to run / Çalıştırılacak görevi seçin")
    if not choice or "quit" in choice:
        print("Cancelled / İptal edildi.")
        return

    if "test_redis" in choice:
        try:
            import redis
            from handlers.fallback_chain import load_config
            cfg = load_config()
            rcfg = cfg.get("redis", {})
            host = rcfg.get("host", "127.0.0.1")
            port = rcfg.get("port", 6379)
            db = rcfg.get("db", 0)
            password = rcfg.get("password", None)
            
            print(f"Connecting to Redis at / Redis bağlantısı kuruluyor: {host}:{port}...")
            r = redis.Redis(host=host, port=port, db=db, password=password, protocol=2, socket_timeout=2.0)
            r.ping()
            print("\033[1;32m[Redis] Connection successful! Redis is running and ready. / Redis bağlantısı başarılı! Redis çalışıyor ve hazır.\033[0m")
        except ImportError:
            print("\033[1;31m[Redis] Python 'redis' library is not installed. / Python 'redis' kütüphanesi kurulu değil.\033[0m")
        except Exception as e:
            print(f"\033[1;31m[Redis] Connection failed / Redis bağlantısı başarısız oldu: {e}\033[0m")
        return

    # Map choice to script
    mapping = {
        "extractor": [sys.executable, os.path.join("scripts", "extractor.py")],
        "enricher": [sys.executable, os.path.join("scripts", "enricher.py")],
        "translator": [sys.executable, os.path.join("scripts", "translator.py")],
        "analyze_logs": [sys.executable, os.path.join("scripts", "analyze_logs.py")],
        "clean": [sys.executable, os.path.join("scripts", "clean.py")],
        "merge": [sys.executable, os.path.join("scripts", "merge_translated.py")],
        "clean_meanings": [sys.executable, os.path.join("scripts", "clean_meanings.py")],
        "splitter": [sys.executable, os.path.join("scripts", "splitter.py")],
    }

    key = None
    if "extractor" in choice:
        key = "extractor"
    elif "enricher" in choice:
        key = "enricher"
    elif "translator" in choice:
        key = "translator"
    elif "analyze_logs" in choice:
        key = "analyze_logs"
    elif "clean_meanings" in choice:
        key = "clean_meanings"
    elif "clean" in choice:
        key = "clean"
    elif "merge" in choice:
        key = "merge"
    elif "splitter" in choice:
        key = "splitter"

    cmd = mapping.get(key) if key else None
    if not cmd:
        print("Unknown choice / Bilinmeyen seçim")
        return

    rc = run_command(cmd)
    print("Return code / Çıkış kodu:", rc)


if __name__ == "__main__":
    main()
