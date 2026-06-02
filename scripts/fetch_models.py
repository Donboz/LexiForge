#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Glossa Models Fetcher / Glossa Model Güncelleyici
===============================================
Queries active providers in config.json to fetch all text-capable models,
estimates/retrieves their context length, sorts them descending, and saves
the structured results to config/models.json.

config.json'daki aktif sağlayıcıları sorgulayarak metin işleme özellikli
tüm modelleri çeker, context uzunluklarını bulur/tahmin eder, büyükten küçüğe
sıralar ve yapılandırılmış sonucu config/models.json dosyasına kaydeder.
"""

import os
import sys
import json
import time
import urllib.request
from typing import List, Dict, Any, Set

# Prepend parent directory to sys.path to resolve imports from handlers/
# Üst dizini sys.path'e ekleyerek handlers/ modüllerini yüklemeyi sağlar
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Windows console UTF-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from handlers.fallback_chain import load_config


def is_text_model(model_id: str) -> bool:
    mid = model_id.lower()
    # Exclude embeddings, audio, vision-only, or moderation models
    exclude_keywords = [
        "embed", "moderation", "whisper", "tts", "speech", "audio", "similarity", "rerank", "bge-"
    ]
    for kw in exclude_keywords:
        if kw in mid:
            return False
    return True


def is_valid_key(api_key: str) -> bool:
    if not api_key:
        return False
    # Check for placeholder strings
    key_upper = api_key.upper()
    if "YOUR_" in key_upper or "API_KEY" in key_upper or "TOKEN" in key_upper:
        return False
    return True


# ─── Query Functions / Sorgu Fonksiyonları ────────────────────────────────────────

def fetch_google(api_key: str) -> List[Dict[str, Any]]:
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
    req = urllib.request.Request(url, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10.0) as r:
        res = json.loads(r.read().decode("utf-8"))
        models = res.get("models", [])
        out = []
        for m in models:
            name = m.get("name", "")
            if name.startswith("models/"):
                model_id = name.split("/")[-1]
            else:
                model_id = name
            
            # Check if it supports text-based generation
            if "generateContent" in m.get("supportedGenerationMethods", []):
                out.append({
                    "id": model_id,
                    "context_length": int(m.get("inputTokenLimit", 0))
                })
        return out


def fetch_openai_compat(base_url: str, api_key: str, provider_key: str) -> List[Dict[str, Any]]:
    # Correct slash formatting in base_url
    url = base_url
    if not url.endswith("/models"):
        if url.endswith("/v1"):
            url = f"{url}/models"
        elif url.endswith("/v1/"):
            url = f"{url}models"
        else:
            url = f"{url.rstrip('/')}/models"
            
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    )
    with urllib.request.urlopen(req, timeout=10.0) as r:
        res = json.loads(r.read().decode("utf-8"))
        data_list = res if isinstance(res, list) else res.get("data", [])
        out = []
        for m in data_list:
            model_id = m.get("id")
            if not model_id or not is_text_model(model_id):
                continue
            
            # Extract context length if returned by API
            context = 0
            if "context_length" in m:
                context = int(m.get("context_length") or 0)
            elif "providers" in m: # Hugging Face Router API format
                providers = m.get("providers", [])
                if providers:
                    context = int(providers[0].get("context_length") or 0)
                
            out.append({
                "id": model_id,
                "context_length": context
            })
        return out


def main():
    print("Loading config.json... / config.json yükleniyor...")
    config = load_config()
    providers = config.get("api_providers", {})
    
    # Provider mapping config details
    openai_compat_configs = {
        "github": "https://models.inference.ai.azure.com",
        "deepseek": "https://api.deepseek.com/v1",
        "huggingface": "https://router.huggingface.co/v1",
        "openrouter": "https://openrouter.ai/api/v1",
        "opencode": "https://opencode.ai/zen/v1",
        "cerebras": "https://api.cerebras.ai/v1",
        "groq": "https://api.groq.com/openai/v1",
        "sambanova": "https://api.sambanova.ai/v1",
        "mistral": "https://api.mistral.ai/v1",
    }
    
    models_db = {}
    
    for provider, pcfg in providers.items():
        if not pcfg.get("enabled", True):
            print(f"Skipping disabled provider / Sağlayıcı aktif değil, atlanıyor: {provider}")
            continue
            
        api_key = pcfg.get("api_key")
        if not is_valid_key(api_key):
            print(f"Skipping provider with no valid API key / Geçerli API anahtarı bulunamadı, atlanıyor: {provider}")
            continue
            
        print(f"Fetching models for / Modeller çekiliyor: {provider}...")
        provider_models = []
        try:
            if provider == "google":
                provider_models = fetch_google(api_key)
            elif provider in openai_compat_configs:
                base_url = openai_compat_configs[provider]
                provider_models = fetch_openai_compat(base_url, api_key, provider)
            else:
                print(f"Unknown provider type / Bilinmeyen sağlayıcı türü: {provider}")
                continue
                
            if provider_models:
                # Sort models by context length descending, then by name alphabetically
                sorted_models = sorted(provider_models, key=lambda x: (-int(x.get("context_length") or 0), x.get("id")))
                
                # Extract only IDs for models.json output
                models_db[provider] = {
                    "models": [m["id"] for m in sorted_models]
                }
                
                print(f"  [SUCCESS] Found {len(sorted_models)} text models for {provider} / {provider} için {len(sorted_models)} adet metin modeli bulundu.")
                for m in sorted_models[:5]:
                    print(f"    - {m['id']} (Context: {m['context_length']})")
                if len(sorted_models) > 5:
                    print(f"    - ... and {len(sorted_models)-5} more / ve {len(sorted_models)-5} tane daha.")
            else:
                print(f"  [Warning] No text models found for / Hiç model bulunamadı: {provider}")
        except Exception as e:
            print(f"  [FAILED] Error fetching models for {provider} / {provider} modelleri alınırken hata: {e}")
            
    if models_db:
        output_path = os.path.join("config", "models.json")
        print(f"\nWriting results to / Sonuçlar şuraya yazılıyor: {output_path}...")
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(models_db, f, indent=2, ensure_ascii=False)
            print("\033[1;32m[SUCCESS] models.json successfully updated! / models.json başarıyla güncellendi!\033[0m")
        except Exception as e:
            print(f"\033[1;31m[ERROR] Failed to save models.json / models.json dosyasına yazma başarısız oldu: {e}\033[0m")
    else:
        print("\n\033[1;31m[Warning] No models fetched from any provider. models.json was not updated. / Hiçbir sağlayıcıdan model çekilemedi. models.json güncellenmedi.\033[0m")


if __name__ == "__main__":
    main()
