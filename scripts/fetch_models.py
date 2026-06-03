#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Glossa Models Fetcher / Glossa Model Güncelleyici
===============================================
Queries active providers in config.json to fetch all text-capable models,
estimates/retrieves their context length, sorts them descending, and saves
the structured results to separate JSON files inside config/models/ directory.

config.json'daki aktif sağlayıcıları sorgulayarak metin işleme özellikli
tüm modelleri çeker, context uzunluklarını bulur/tahmin eder, büyükten küçüğe
sıralar ve yapılandırılmış sonucu config/models/ dizini altındaki ayrı JSON dosyalarına kaydeder.

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
    # Exclude embeddings, audio, vision, image, and text-to-speech models
    exclude_keywords = [
        "embed", "moderation", "whisper", "tts", "speech", "audio", "similarity", "rerank", "bge-",
        "image", "clip", "lyria", "video"
    ]
    for kw in exclude_keywords:
        if kw in mid:
            return False
    return True


def is_valid_key(api_key: str) -> bool:
    if not api_key:
        return False
    key_upper = api_key.upper()
    if "YOUR_" in key_upper or "API_KEY" in key_upper or "TOKEN" in key_upper:
        return False
    return True


# ─── Query Functions / Sorgu Fonksiyonları ────────────────────────────────────────

def fetch_google(api_key: str) -> List[Dict[str, Any]]:
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
    req = urllib.request.Request(
        url,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
    )
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
            
            # Check if it supports text-based generation and is a text model
            if "generateContent" in m.get("supportedGenerationMethods", []) and is_text_model(model_id):
                out.append({
                    "id": model_id,
                    "context": int(m.get("inputTokenLimit", 0))
                })
        return out


def find_context_value(d: Dict[str, Any]) -> int:
    for k, v in d.items():
        k_lower = k.lower()
        if "context_length" in k_lower or "context_lenght" in k_lower or "context_window" in k_lower or "context_windows" in k_lower:
            try:
                if v is not None:
                    return int(v)
            except (ValueError, TypeError):
                pass
    return 0


def fetch_openai_compat(base_url: str, api_key: str) -> List[Dict[str, Any]]:
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
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
    )
    with urllib.request.urlopen(req, timeout=10.0) as r:
        res = json.loads(r.read().decode("utf-8"))
        data_list = res if isinstance(res, list) else res.get("data", [])
        out = []
        for m in data_list:
            model_id = m.get("id")
            # For GitHub Models, use the friendly name field (e.g. Meta-Llama-3.1-8B-Instruct)
            if "inference.ai.azure.com" in base_url and m.get("name"):
                model_id = m.get("name")
                
            if not model_id or not is_text_model(model_id):
                continue
            
            context = find_context_value(m)
            if not context and "providers" in m:  # Hugging Face Router API format
                providers = m.get("providers", [])
                if isinstance(providers, list):
                    for provider_obj in providers:
                        if isinstance(provider_obj, dict):
                            val = find_context_value(provider_obj)
                            if val:
                                context = val
                                break
                
            out.append({
                "id": model_id,
                "context": context,
                "pricing": m.get("pricing"),
                "providers": m.get("providers")
            })
        return out


def main():
    print("Loading config.json... / config.json yükleniyor...")
    config = load_config()
    providers = config.get("api_providers", {})
    
    # Delete old models.json if it exists
    old_models_path = os.path.join("config", "models.json")
    if os.path.exists(old_models_path):
        try:
            os.remove(old_models_path)
            print("Removed old monolithic models.json / Eski tek parça models.json silindi.")
        except Exception as e:
            print(f"Warning: Could not remove old models.json: {e}")

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
        "nvidia": "https://integrate.api.nvidia.com/v1",
    }
    
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
                provider_models = fetch_openai_compat(base_url, api_key)
            else:
                print(f"Unknown provider type / Bilinmeyen sağlayıcı türü: {provider}")
                continue
                
            if provider_models:
                free_list = []
                paid_list = []
                
                # Fetch provider rate limit from config
                rate_limit_val = pcfg.get("rate_limit_rpm")
                rate_limit_str = f"{rate_limit_val} RPM" if rate_limit_val else "N/A"
                
                for m in provider_models:
                    model_id = m["id"]
                    context = int(m.get("context") or 0)
                    
                    # ─── Dynamic Categorization Logic / Dinamik Sınıflandırma Mantığı ───
                    is_free = False
                    if provider == "google":
                        # Gemini flash and lite are free, pro is paid
                        is_free = not ("pro" in model_id.lower())
                    elif provider == "openrouter":
                        pricing = m.get("pricing") or {}
                        prompt_price = float(pricing.get("prompt", 0) or 0)
                        completion_price = float(pricing.get("completion", 0) or 0)
                        is_free = (prompt_price == 0.0 and completion_price == 0.0) or model_id.endswith(":free")
                    elif provider == "huggingface":
                        providers_list = m.get("providers") or []
                        is_free = True
                        if providers_list:
                            pricing = providers_list[0].get("pricing") or {}
                            input_price = float(pricing.get("input", 0) or 0)
                            output_price = float(pricing.get("output", 0) or 0)
                            if input_price > 0.0 or output_price > 0.0:
                                 is_free = False
                    elif provider in ("github", "groq", "cerebras", "sambanova"):
                        is_free = True
                    elif provider == "opencode":
                        is_free = ("free" in model_id.lower())
                    elif provider == "mistral":
                        is_free = not ("large" in model_id.lower() or "medium" in model_id.lower() or "codestral" in model_id.lower())
                    elif provider == "deepseek" or provider == "nvidia":
                        is_free = False
                        
                    # Structure output objects
                    if is_free:
                        free_list.append({
                            "id": model_id,
                            "context": context,
                            "rate_limit": rate_limit_str
                        })
                    else:
                        paid_list.append({
                            "id": model_id,
                            "context": context
                        })
                
                # Sort free and paid lists by context descending, then by name alphabetically
                sorted_free = sorted(free_list, key=lambda x: (-x["context"], x["id"]))
                sorted_paid = sorted(paid_list, key=lambda x: (-x["context"], x["id"]))
                
                provider_db = {
                    "free": sorted_free,
                    "paid": sorted_paid
                }
                
                # Write to separate JSON file per provider inside config/models/
                models_dir = os.path.join("config", "models")
                os.makedirs(models_dir, exist_ok=True)
                output_path = os.path.join(models_dir, f"{provider}.json")
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(provider_db, f, indent=2, ensure_ascii=False)
                
                print(f"  [SUCCESS] Wrote {len(sorted_free)} free and {len(sorted_paid)} paid models to config/models/{provider}.json")
            else:
                print(f"  [Warning] No text models found for / Hiç model bulunamadı: {provider}")
        except Exception as e:
            print(f"  [FAILED] Error fetching models for {provider} / {provider} modelleri alınırken hata: {e}")
            
    print("\nModel fetching and split-saving completed. / Model çekme ve ayrı kaydetme tamamlandı.")


if __name__ == "__main__":
    main()
