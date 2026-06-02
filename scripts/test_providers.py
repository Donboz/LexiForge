#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import json
import time

# Prepend parent directory to sys.path to resolve imports from handlers/
# Üst dizini sys.path'e ekleyerek handlers/ modüllerini yüklemeyi sağlar
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def load_config():
    config_path = os.path.join("config", "config.json")
    if not os.path.exists(config_path):
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)

def test_openrouter(api_key, model):
    print("Testing / Test ediliyor: OpenRouter...")
    from openai import OpenAI
    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1"
    )
    start = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Say exactly 'OK'"}],
        temperature=0.1,
        max_tokens=150
    )
    latency = time.time() - start
    content = response.choices[0].message.content
    if not content or not content.strip():
        raise ValueError("OpenRouter returned None content / OpenRouter boş içerik döndürdü")
    text = content.strip()
    print(f"  [SUCCESS / BAŞARILI] OpenRouter: '{text}' (took/sürdü {latency:.2f}s)")
    return text

def test_cerebras(api_key, model):
    print("Testing / Test ediliyor: Cerebras...")
    from cerebras.cloud.sdk import Cerebras
    client = Cerebras(api_key=api_key)
    start = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Say exactly 'OK'"}],
        temperature=0.1,
        max_tokens=150
    )
    latency = time.time() - start
    content = response.choices[0].message.content
    if not content or not content.strip():
        raise ValueError("Cerebras returned None content / Cerebras boş içerik döndürdü")
    text = content.strip()
    print(f"  [SUCCESS / BAŞARILI] Cerebras: '{text}' (took/sürdü {latency:.2f}s)")
    return text

def test_openai_compat(provider_name, api_key, model, base_url):
    print(f"Testing / Test ediliyor: {provider_name}...")
    from openai import OpenAI
    client = OpenAI(
        api_key=api_key,
        base_url=base_url
    )
    start = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Say exactly 'OK'"}],
        temperature=0.1,
        max_tokens=150
    )
    latency = time.time() - start
    content = response.choices[0].message.content
    if not content or not content.strip():
        raise ValueError(f"{provider_name} returned None content / {provider_name} boş içerik döndürdü")
    text = content.strip()
    print(f"  [SUCCESS / BAŞARILI] {provider_name}: '{text}' (took/sürdü {latency:.2f}s)")
    return text

def test_groq(api_key, model):
    print("Testing / Test ediliyor: Groq...")
    from groq import Groq
    client = Groq(api_key=api_key)
    start = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Say exactly 'OK'"}],
        temperature=0.1,
        max_tokens=150
    )
    latency = time.time() - start
    content = response.choices[0].message.content
    if not content or not content.strip():
        raise ValueError("Groq returned None content / Groq boş içerik döndürdü")
    text = content.strip()
    print(f"  [SUCCESS / BAŞARILI] Groq: '{text}' (took/sürdü {latency:.2f}s)")
    return text

def test_mistral(api_key, model):
    return test_openai_compat("Mistral", api_key, model, "https://api.mistral.ai/v1")



def test_google(api_key, model):
    print("Testing / Test ediliyor: Google...")
    from google import genai
    client = genai.Client(api_key=api_key)
    start = time.time()
    response = client.models.generate_content(
        model=model,
        contents="Say exactly 'OK'",
        config={'temperature': 0.1, 'max_output_tokens': 100}
    )
    latency = time.time() - start
    content = response.text
    if not content or not content.strip():
        raise ValueError("Google returned None content / Google boş içerik döndürdü")
    text = content.strip()
    print(f"  [SUCCESS / BAŞARILI] Google: '{text}' (took/sürdü {latency:.2f}s)")
    return text

def test_github(api_key, model):
    print("Testing / Test ediliyor: GitHub...")
    return test_openai_compat("GitHub", api_key, model, "https://models.inference.ai.azure.com")

def test_opencode(api_key, model):
    print("Testing / Test ediliyor: OpenCode...")
    return test_openai_compat("OpenCode", api_key, model, "https://opencode.ai/zen/v1")

def main():
    config = load_config()
    providers = config.get("api_providers", {})
    
    results = {}
    
    # 0.1 Google
    google_cfg = providers.get("google", {})
    if google_cfg.get("enabled", True):
        api_key = google_cfg.get("api_key")
        model = google_cfg.get("models", ["gemini-2.5-flash"])[0]
        if api_key:
            try:
                test_google(api_key, model)
                results["Google"] = "Success / Başarılı"
            except Exception as e:
                print(f"  [FAILED / BAŞARISIZ] Google: {e}")
                results["Google"] = f"Failed / Başarısız ({e})"
        else:
            print("Google API key missing / Google API anahtarı eksik.")
            results["Google"] = "Missing Key / Anahtar Eksik"

    # 0.2 GitHub
    github_cfg = providers.get("github", {})
    if github_cfg.get("enabled", True):
        api_key = github_cfg.get("api_key")
        model = github_cfg.get("models", ["Meta-Llama-3.1-8B-Instruct"])[0]
        if api_key:
            try:
                test_github(api_key, model)
                results["GitHub"] = "Success / Başarılı"
            except Exception as e:
                print(f"  [FAILED / BAŞARISIZ] GitHub: {e}")
                results["GitHub"] = f"Failed / Başarısız ({e})"
        else:
            print("GitHub API key missing / GitHub API anahtarı eksik.")
            results["GitHub"] = "Missing Key / Anahtar Eksik"

    # 1. OpenRouter
    openrouter_cfg = providers.get("openrouter", {})
    if openrouter_cfg.get("enabled", True):
        api_key = openrouter_cfg.get("api_key")
        model = openrouter_cfg.get("models", ["openrouter/owl-alpha"])[0]
        if api_key:
            try:
                test_openrouter(api_key, model)
                results["OpenRouter"] = "Success / Başarılı"
            except Exception as e:
                print(f"  [FAILED / BAŞARISIZ] OpenRouter: {e}")
                results["OpenRouter"] = f"Failed / Başarısız ({e})"
        else:
            print("OpenRouter API key missing / OpenRouter API anahtarı eksik.")
            results["OpenRouter"] = "Missing Key / Anahtar Eksik"

    # 2. Cerebras
    cerebras_cfg = providers.get("cerebras", {})
    if cerebras_cfg.get("enabled", True):
        api_key = cerebras_cfg.get("api_key")
        model = cerebras_cfg.get("models", ["llama3.1-8b"])[0]
        if api_key:
            try:
                test_cerebras(api_key, model)
                results["Cerebras"] = "Success / Başarılı"
            except Exception as e:
                print(f"  [FAILED / BAŞARISIZ] Cerebras: {e}")
                results["Cerebras"] = f"Failed / Başarısız ({e})"
        else:
            print("Cerebras API key missing / Cerebras API anahtarı eksik.")
            results["Cerebras"] = "Missing Key / Anahtar Eksik"

    # 3. Groq
    groq_cfg = providers.get("groq", {})
    if groq_cfg.get("enabled", True):
        api_key = groq_cfg.get("api_key")
        model = groq_cfg.get("models", ["llama-3.3-70b-versatile"])[0]
        if api_key:
            try:
                test_groq(api_key, model)
                results["Groq"] = "Success / Başarılı"
            except Exception as e:
                print(f"  [FAILED / BAŞARISIZ] Groq: {e}")
                results["Groq"] = f"Failed / Başarısız ({e})"
        else:
            print("Groq API key missing / Groq API anahtarı eksik.")
            results["Groq"] = "Missing Key / Anahtar Eksik"

    # 5. SambaNova
    sambanova_cfg = providers.get("sambanova", {})
    if sambanova_cfg.get("enabled", True):
        api_key = sambanova_cfg.get("api_key")
        model = sambanova_cfg.get("models", ["Meta-Llama-3.3-70B-Instruct"])[0]
        if api_key:
            try:
                test_openai_compat("SambaNova", api_key, model, "https://api.sambanova.ai/v1")
                results["SambaNova"] = "Success / Başarılı"
            except Exception as e:
                print(f"  [FAILED / BAŞARISIZ] SambaNova: {e}")
                results["SambaNova"] = f"Failed / Başarısız ({e})"
        else:
            print("SambaNova API key missing / SambaNova API anahtarı eksik.")
            results["SambaNova"] = "Missing Key / Anahtar Eksik"

    # 6. Mistral
    mistral_cfg = providers.get("mistral", {})
    if mistral_cfg.get("enabled", True):
        api_key = mistral_cfg.get("api_key")
        model = mistral_cfg.get("models", ["pixtral-12b"])[0]
        if api_key:
            try:
                test_mistral(api_key, model)
                results["Mistral"] = "Success / Başarılı"
            except Exception as e:
                print(f"  [FAILED / BAŞARISIZ] Mistral: {e}")
                results["Mistral"] = f"Failed / Başarısız ({e})"
        else:
            print("Mistral API key missing / Mistral API anahtarı eksik.")
            results["Mistral"] = "Missing Key / Anahtar Eksik"

    # 7. OpenCode
    opencode_cfg = providers.get("opencode", {})
    if opencode_cfg.get("enabled", True):
        api_key = opencode_cfg.get("api_key")
        model = opencode_cfg.get("models", ["nemotron-3-super-free"])[0]
        if api_key:
            try:
                test_opencode(api_key, model)
                results["OpenCode"] = "Success / Başarılı"
            except Exception as e:
                print(f"  [FAILED / BAŞARISIZ] OpenCode: {e}")
                results["OpenCode"] = f"Failed / Başarısız ({e})"
        else:
            print("OpenCode API key missing / OpenCode API anahtarı eksik.")
            results["OpenCode"] = "Missing Key / Anahtar Eksik"

    print("\n=== TEST RESULTS / TEST SONUÇLARI ===")
    for provider, res in results.items():
        print(f"{provider}: {res}")

if __name__ == "__main__":
    main()
