# Configuration Template / Yapılandırma Şablonu (`config.json`)

This document provides a template and explanation for the `config/config.json` configuration file used by the Glossa Dictionary Miner. 

Bu belge, Glossa Dictionary Miner tarafından kullanılan `config/config.json` yapılandırma dosyası için bir şablon ve açıklama sunar.

---

## 📄 Example Configuration Template / Örnek Yapılandırma Şablonu

Create a file named `config.json` inside the `config/` directory with the following structure:
`config/` dizini içerisinde aşağıdaki yapıya sahip `config.json` adında bir dosya oluşturun:

```json
{
  "base_dir": "C:\\Path\\To\\Your\\Dictionaries",
  "redis": {
    "host": "127.0.0.1",
    "port": 6379,
    "db": 0,
    "password": null
  },
  "api_providers": {
    "google": {
      "enabled": true,
      "api_key": "YOUR_GEMINI_API_KEY",
      "models": [
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "gemini-2.0-flash"
      ],
      "rate_limit_rpm": 15,
      "max_concurrent": 2,
      "_comment": "Google Gemini API via google-genai SDK."
    },
    "github": {
      "enabled": false,
      "api_key": "YOUR_GITHUB_TOKEN",
      "models": [
        "Meta-Llama-3.1-405B-Instruct", 
        "Meta-Llama-3.1-8B-Instruct"
      ],
      "rate_limit_rpm": 15,
      "max_concurrent": 2
    },
    "deepseek": {
      "enabled": true,
      "api_key": "YOUR_DEEPSEEK_API_KEY",
      "models": [
        "deepseek-v4-flash",
        "deepseek-v4-pro"
      ],
      "rate_limit_rpm": 60,
      "max_concurrent": 4,
      "_comment": "DeepSeek official API."
    },
    "huggingface": {
      "enabled": false,
      "api_key": "YOUR_HUGGINGFACE_TOKEN",
      "models": [
        "Qwen/Qwen2.5-7B-Instruct",
        "meta-llama/Llama-3.2-3B-Instruct",
        "meta-llama/Llama-3.1-8B-Instruct",
        "google/gemma-2-9b-it"
      ],
      "rate_limit_rpm": 20,
      "max_concurrent": 2,
      "_comment": "Hugging Face Serverless Inference API (Free tier)"
    },
    "openrouter": {
      "enabled": true,
      "api_key": "YOUR_OPENROUTER_API_KEY",
      "models": [
        "qwen/qwen3-coder:free",
        "meta-llama/llama-3.3-70b-instruct:free"
      ],
      "rate_limit_rpm": 20,
      "max_concurrent": 2
    },
    "cerebras": {
      "enabled": false,
      "api_key": "YOUR_CEREBRAS_API_KEY",
      "models": [
        "llama3.1-8b"
      ],
      "rate_limit_rpm": 30,
      "max_concurrent": 3
    },
    "groq": {
      "enabled": true,
      "api_key": "YOUR_GROQ_API_KEY",
      "models": [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant"
      ],
      "rate_limit_rpm": 30,
      "max_concurrent": 2
    },
    "sambanova": {
      "enabled": false,
      "api_key": "YOUR_SAMBANOVA_API_KEY",
      "models": [
        "Meta-Llama-3.3-70B-Instruct"
      ],
      "rate_limit_rpm": 30,
      "max_concurrent": 2
    },
    "mistral": {
      "enabled": false,
      "api_key": "YOUR_MISTRAL_API_KEY",
      "models": [
        "codestral-latest",
        "open-mistral-nemo"
      ],
      "rate_limit_rpm": 30,
      "max_concurrent": 2
    }
  },
  "languages": {
    "DE": "Almanca",
    "TR": "Türkçe",
    "EN": "İngilizce",
    "FR": "Fransızca",
    "RU": "Rusça",
    "IT": "İtalyanca",
    "ES": "İspanyolca"
  },
  "fallback": {
    "max_retries_per_page": 0,
    "provider_order": [
      "google",
      "github",
      "deepseek",
      "openrouter",
      "opencode",
      "cerebras",
      "groq",
      "sambanova",
      "mistral",
      "huggingface"
    ],
    "retry_delay_sec": 10,
    "max_total_failures_before_abort": 0,
    "_comment": "max_retries_per_page=0 and max_total_failures_before_abort=0 means infinite retries until page succeeds."
  },
  "pdf_lang_map": {
    "German-Turkish Law Dictionary.pdf": {
      "source": "DE",
      "target": "TR",
      "domain": "LEGAL"
    },
    "English-Turkish Dictionary.pdf": {
      "source": "EN",
      "target": "TR",
      "domain": "GENERAL",
      "provider": "deepseek",
      "model": "deepseek-v4-flash"
    }
  }
}
```

---

## 🔍 Section Explanations / Alan Açıklamaları

### 1. `base_dir`
* **EN**: Absolute path to the folder containing your source dictionary files (e.g. PDFs). Use double backslashes (`\\`) on Windows.
* **TR**: Kaynak sözlük dosyalarınızın (örn. PDF'ler) bulunduğu klasörün mutlak yolu. Windows'ta çift eğik çizgi (`\\`) kullanın.

### 2. `redis`
* **EN**: Redis connection details. If Redis is not used, this section will be ignored, and file-based execution is run.
* **TR**: Redis bağlantı detayları. Redis kullanılmadığında bu alan yoksayılır ve dosya tabanlı çalışma yürütülür.

### 3. `api_providers`
* **EN**: List of API endpoints, keys, models list, concurrency, and rate-limits. Set `"enabled": true` to include a provider in the active fallback chain.
  * **DeepSeek**: Connects to `https://api.deepseek.com` endpoint using OpenAI-compatible queries.
  * **Hugging Face**: Connects to the Serverless Inference API at `https://api-inference.huggingface.co` without downloading any model weights locally.
* **TR**: API adresleri, anahtarlar, modeller listesi, eşzamanlılık ve istek sınırları listesi. Bir sağlayıcıyı aktif fallback zincirine dahil etmek için `"enabled": true` yapın.
  * **DeepSeek**: OpenAI uyumlu sorgular kullanarak `https://api.deepseek.com` adresine bağlanır.
  * **Hugging Face**: Yerel diskinize herhangi bir model indirmeden doğrudan `https://api-inference.huggingface.co` adresindeki Serverless Inference API'ye bağlanır.

### 4. `languages`
* **EN**: Key-value pairs mapping language codes (e.g. `"DE"`) to their full names (e.g. `"Almanca"`).
* **TR**: Dil kodlarını (örn. `"DE"`) tam adlarıyla (örn. `"Almanca"`) eşleştiren anahtar-değer çiftleri.

### 5. `fallback`
* **EN**: Controls the behavior of the API failover architecture.
  - `provider_order`: The priority sequence when attempting model requests.
  - `retry_delay_sec`: Time to wait before retrying a failed query.
* **TR**: API yedekleme mimarisinin davranışını kontrol eder.
  - `provider_order`: Model istekleri denenirken izlenecek öncelik sırası.
  - `retry_delay_sec`: Başarısız bir sorguyu yeniden denemeden önce beklenecek süre.

### 6. `pdf_lang_map`
* **EN**: Maps dictionary filenames to their respective configurations:
  - `source`: Source language code.
  - `target`: Target language code.
  - `domain`: Subject domain (`GENERAL`, `LEGAL`, `MEDICAL`, `IDIOM`, `TECHNICAL`, etc.).
  - `provider` / `model` (Optional): Forces a specific LLM instead of using the fallback chain.
* **TR**: Sözlük dosya isimlerini kendi yapılandırmalarıyla eşleştirir:
  - `source`: Kaynak dil kodu.
  - `target`: Hedef dil kodu.
  - `domain`: Konu alanı (`GENERAL`, `LEGAL`, `MEDICAL`, `IDIOM`, `TECHNICAL` vb.).
  - `provider` / `model` (İsteğe bağlı): Fallback zinciri kullanmak yerine belirli bir LLM'i zorunlu kılar.
