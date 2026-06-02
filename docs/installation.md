# Installation Guide / Kurulum Kılavuzu

This document details the environment setup, packages installation, Redis configurations, and API keys management for the Glossa Dictionary Miner.

Bu belge, Glossa Dictionary Miner için ortam kurulumunu, paketlerin yüklenmesini, Redis konfigürasyonlarını ve API anahtarlarının yönetimini ayrıntılı olarak açıklamaktadır.

---

## 📋 Prerequisites / Gereksinimler

- **Python 3.10+** (Recommended / Önerilen)
- **Redis Server** (Optional but highly recommended for caching / Önbellekleme için isteğe bağlı ancak şiddetle önerilir)
- **API Keys / API Anahtarları** (At least one of the supported providers, e.g. OpenRouter, Groq, Google Gemini, Cerebras, Sambanova, Mistral, GitHub / Desteklenen sağlayıcılardan en az biri)

---

## 🛠️ Step 1: Environment Setup / Adım 1: Ortam Kurulumu

### 1. Clone the repository / Repoyu klonlayın
```bash
git clone <repository-url>
cd scraper
```

### 2. Create a Virtual Environment / Sanal Ortam Oluşturun

**Windows (CMD/PowerShell):**
```powershell
python -m venv venv
venv\Scripts\activate
```

**Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies / Bağımlılıkları yükleyin
```bash
pip install -r requirements.txt
```
*(If `requirements.txt` does not exist, install the core packages manually / Eğer `requirements.txt` mevcut değilse, ana paketleri manuel olarak yükleyin:)*
```bash
pip install PyPDF2 orjson tqdm redis websockets requests aiohttp
```

---

## 🧠 Step 2: Redis Server Setup / Adım 2: Redis Sunucusu Kurulumu

To use the namespaced cache layer, a running Redis instance is required.

Önbellek katmanını kullanabilmek için çalışan bir Redis sunucusu gereklidir.

### Windows Installation / Windows Kurulumu
1. Download **Redis-x64** from GitHub releases (e.g., [tporadowski/redis](https://github.com/tporadowski/redis/releases)) or use **WSL** (Windows Subsystem for Linux).
   GitHub releases üzerinden Windows için Redis indirin veya WSL kullanın.
2. Run the installer or extract and start:
   Redis sunucusunu başlatmak için:
   ```cmd
   redis-server.exe
   ```

### Linux Installation / Linux Kurulumu
Install via your system's package manager:
Sistem paket yöneticiniz ile kurulum yapın:
```bash
sudo apt update
sudo apt install redis-server
sudo systemctl enable redis-server
sudo systemctl start redis-server
```

---

## 🔑 Step 3: API Keys & Configuration / Adım 3: API Anahtarları ve Yapılandırma

The application configuration is managed under `config/config.json` and `config/models.json`.

Uygulama yapılandırması `config/config.json` ve `config/models.json` dosyaları altından yönetilir.

1. **API Credentials / API Kimlik Bilgileri**:
   Open `config/config.json` and add your API keys under the appropriate provider section:
   `config/config.json` dosyasını açarak API anahtarlarınızı ilgili sağlayıcı (provider) bölümüne ekleyin:
   ```json
   "api_providers": {
     "openrouter": {
       "api_key": "YOUR_OPENROUTER_API_KEY",
       "enabled": true
     },
     "google": {
       "api_key": "YOUR_GEMINI_API_KEY",
       "enabled": true
     }
   }
   ```
2. **Redis settings / Redis Ayarları**:
   If your Redis server is running on a custom port or requires a password, update the `"redis"` block in `config/config.json`.
   Redis sunucunuz farklı bir portta çalışıyorsa ya da şifre gerektiriyorsa `config/config.json` içerisindeki `"redis"` bloğunu güncelleyin.

---

## 🏃 Step 4: Running the Orchestrator / Adım 4: Orkestratörü Çalıştırma

Start the interactive CLI menu:

İnteraktif CLI menüsünü başlatın:

```bash
python run.py
```

Choose Option `9` first to verify that your Redis connection is active and working. Then, proceed with data extraction, translation, or enrichment!

Öncelikle Redis bağlantınızın aktif ve çalışır durumda olduğunu doğrulamak için `9` numaralı seçeneği (Test Redis Connection) seçin. Ardından veri çıkarma, çeviri veya zenginleştirme adımlarına geçebilirsiniz!
