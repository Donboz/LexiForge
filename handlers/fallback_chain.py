#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Glossa Fallback Chain Engine / Glossa Yedek Zincir Motoru
=========================================================
Reads all providers and models from config.json.
If an API call fails, sequentially retries next model → next provider.
Pages are only marked "processed" when a call succeeds.

config.json'dan tüm sağlayıcı ve modelleri okur.
Bir API çağrısı başarısız olursa sırayla sonraki model → sonraki sağlayıcı dener.
İşlem ancak başarılı olduğunda başarı olarak işaretlenir.
"""

import os
import time
import json
from enum import Enum
from typing import Optional, Dict, List, Any, Tuple


# ─── State Enum / Durum Enum ──────────────────────────────────────────────────

class ModelState(Enum):
    READY = "READY"
    COOLDOWN = "COOLDOWN"
    DISABLED = "DISABLED"
    EXHAUSTED = "EXHAUSTED"


# ─── Provider Query Functions / Sağlayıcı Sorgu Fonksiyonları ─────────────────

def query_openai_compat(prompt: str, api_key: str, model: str,
                        base_url: str, temperature: float = 0.2,
                        max_tokens: int = 4096) -> str:
    """OpenAI-compatible API query / OpenAI uyumlu API sorgusu."""
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=0.9
    )
    content = response.choices[0].message.content
    if not content or not content.strip():
        raise ValueError(f"API returned empty content for model {model} / API {model} modeli için boş yanıt döndürdü")
    return content.strip()


def query_openrouter(prompt: str, api_key: str, model: str,
                      temperature: float = 0.2, max_tokens: int = 4096) -> str:
    """OpenRouter API query / OpenRouter API sorgusu."""
    return query_openai_compat(prompt, api_key, model,
                               "https://openrouter.ai/api/v1",
                               temperature, max_tokens)

def query_cerebras(prompt: str, api_key: str, model: str,
                    temperature: float = 0.2, max_tokens: int = 4096) -> str:
    """Cerebras API query using official Cerebras SDK / Cerebras resmi SDK kullanarak Cerebras API sorgusu."""
    from cerebras.cloud.sdk import Cerebras
    client = Cerebras(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens
    )
    content = response.choices[0].message.content
    if not content or not content.strip():
        raise ValueError(f"Cerebras API returned empty content for model {model} / Cerebras API {model} modeli için boş yanıt döndürdü")
    return content.strip()


def query_groq(prompt: str, api_key: str, model: str,
               temperature: float = 0.2, max_tokens: int = 4096) -> str:
    """Groq API query using official Groq SDK / Groq resmi SDK kullanarak Groq API sorgusu."""
    from groq import Groq
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens
    )
    content = response.choices[0].message.content
    if not content or not content.strip():
        raise ValueError(f"Groq API returned empty content for model {model} / Groq API {model} modeli için boş yanıt döndürdü")
    return content.strip()


def query_sambanova(prompt: str, api_key: str, model: str,
                    temperature: float = 0.2, max_tokens: int = 4096) -> str:
    """SambaNova API query / SambaNova API sorgusu."""
    return query_openai_compat(prompt, api_key, model,
                               "https://api.sambanova.ai/v1",
                               temperature, max_tokens)


def query_mistral(prompt: str, api_key: str, model: str,
                  temperature: float = 0.2, max_tokens: int = 4096) -> str:
    """Mistral API query using official mistralai SDK / Mistral resmi SDK kullanarak Mistral API sorgusu."""
    from mistralai.client import Mistral
    client = Mistral(api_key=api_key)
    response = client.chat.complete(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens
    )
    content = response.choices[0].message.content
    if not content or not content.strip():
        raise ValueError(f"Mistral API returned empty content for model {model} / Mistral API {model} modeli için boş yanıt döndürdü")
    return content.strip()


def query_google(prompt: str, api_key: str, model: str,
                 temperature: float = 0.2, max_tokens: int = 4096) -> str:
    """Google Gemini API query using official google-genai SDK / Google resmi google-genai SDK kullanarak Google Gemini API sorgusu."""
    from google import genai
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config={
            'temperature': temperature,
            'max_output_tokens': max_tokens
        }
    )
    content = response.text
    if not content or not content.strip():
        raise ValueError(f"Google API returned empty content for model {model} / Google API {model} modeli için boş yanıt döndürdü")
    return content.strip()


def query_github(prompt: str, api_key: str, model: str,
                 temperature: float = 0.2, max_tokens: int = 4096) -> str:
    """GitHub Models API query / GitHub Models API sorgusu."""
    return query_openai_compat(prompt, api_key, model,
                               "https://models.inference.ai.azure.com",
                               temperature, max_tokens)


def query_opencode(prompt: str, api_key: str, model: str,
                   temperature: float = 0.2, max_tokens: int = 4096) -> str:
    """OpenCode API query / OpenCode API sorgusu."""
    return query_openai_compat(prompt, api_key, model,
                               "https://opencode.ai/zen/v1",
                               temperature, max_tokens)


def query_deepseek(prompt: str, api_key: str, model: str,
                   temperature: float = 0.2, max_tokens: int = 4096) -> str:
    """DeepSeek API query / DeepSeek API sorgusu."""
    return query_openai_compat(prompt, api_key, model,
                               "https://api.deepseek.com/v1",
                               temperature, max_tokens)


def query_huggingface(prompt: str, api_key: str, model: str,
                      temperature: float = 0.2, max_tokens: int = 4096) -> str:
    """Hugging Face Router API query / Hugging Face Router API sorgusu."""
    return query_openai_compat(prompt, api_key, model,
                               "https://router.huggingface.co/v1",
                               temperature, max_tokens)


# Provider key → query function mapping / Sağlayıcı anahtarı → sorgu fonksiyonu eşleşmesi
PROVIDER_QUERY_FNS = {
    "openrouter": query_openrouter,
    "opencode": query_opencode,
    "cerebras": query_cerebras,
    "groq": query_groq,
    "sambanova": query_sambanova,
    "mistral": query_mistral,
    "google": query_google,
    "github": query_github,
    "deepseek": query_deepseek,
    "huggingface": query_huggingface,
}


# ─── Model Tracker / Model Takipçisi ──────────────────────────────────────────

class ModelTracker:
    """Tracks state and metrics of a (provider, model) pair / Bir (sağlayıcı, model) çiftinin durumunu ve metriklerini takip eder."""

    def __init__(self, provider_key: str, model_name: str, api_key: str):
        self.provider_key = provider_key
        self.model_name = model_name
        self.api_key = api_key
        self.state = ModelState.READY
        self.consecutive_failures = 0
        self.consecutive_rate_limits = 0
        self.cooldown_until = 0.0  # timestamp
        self.total_failures = 0
        self.total_successes = 0
        self.rate_limit_ban_rounds = 0

    def is_available(self) -> bool:
        """Is the model currently available? / Model şu anda kullanılabilir mi?"""
        if self.state == ModelState.DISABLED:
            return False
        if self.rate_limit_ban_rounds > 0:
            return False
        if self.state in (ModelState.COOLDOWN, ModelState.EXHAUSTED):
            if time.time() >= self.cooldown_until:
                self.state = ModelState.READY
                self.consecutive_rate_limits = 0
                return True
            return False
        return True

    def report_success(self):
        """Update state after a successful call / Başarılı çağrı sonrası durumu güncelle."""
        self.state = ModelState.READY
        self.consecutive_failures = 0
        self.consecutive_rate_limits = 0
        self.total_successes += 1
        self.rate_limit_ban_rounds = 0

    def report_failure(self, cooldown_sec: float = 2.0, max_consecutive: int = 3):
        """Update state after a failed call / Başarısız çağrı sonrası durumu güncelle."""
        self.consecutive_failures += 1
        self.total_failures += 1
        self.state = ModelState.COOLDOWN
        self.cooldown_until = time.time() + cooldown_sec

    def report_rate_limit(self, cooldown_sec: float = 10.0, max_rate_limits: int = 2):
        """Update state on rate limit hit / Rate limit durumunda durumu güncelle."""
        self.consecutive_failures += 1
        self.total_failures += 1
        self.consecutive_rate_limits += 1
        self.rate_limit_ban_rounds = 4  # skip for 3 subsequent calls / sonraki 3 arama boyunca atla
        
        if self.consecutive_rate_limits >= max_rate_limits:
            self.state = ModelState.EXHAUSTED
            self.cooldown_until = time.time() + 30.0  # 30 seconds cooldown / 30 saniye cooldown
            print(f"\033[1;31m  [Fallback] [EXHAUSTED] {self.provider_key}/{self.model_name} EXHAUSTED (30s cooldown & 3 rounds skip) / TÜKENDİ (30sn bekleme & 3 tur atlanacak)\033[0m")
        else:
            self.state = ModelState.COOLDOWN
            self.cooldown_until = time.time() + cooldown_sec
            print(f"\033[1;33m  [Fallback] [WARNING] {self.provider_key}/{self.model_name} rate-limited, will skip for 3 rounds. / rate-limit yedi, 3 tur boyunca kullanılmayacak.\033[0m")

    def reset(self):
        """Make the model retriable again after a full loop / Tam bir tur sonrası modeli tekrar denenebilir yap."""
        if self.state != ModelState.DISABLED:
            self.state = ModelState.READY
            self.consecutive_failures = 0
            self.consecutive_rate_limits = 0

    def __repr__(self):
        return f"<ModelTracker {self.provider_key}/{self.model_name} state={self.state.value} ban_rounds={self.rate_limit_ban_rounds}>"


# ─── Fallback Chain / Yedek Zinciri ───────────────────────────────────────────

class FallbackChain:
    """
    Loads all enabled providers and models from config.json.
    Calls them sequentially until a prompt call succeeds.
    """

    def __init__(self, config: dict, provider_filter: Optional[str] = None, model_filter: Optional[str] = None,
                 event_callback=None):
        self.config = config
        self.trackers: List[ModelTracker] = []
        self.stats = {
            "total_calls": 0,
            "total_successes": 0,
            "total_failures": 0,
            "provider_stats": {},
        }
        self.event_callback = event_callback

        # Fallback config / Fallback ayarları
        fallback_cfg = config.get("fallback", {})
        self.max_retries_per_page = fallback_cfg.get("max_retries_per_page", 0)  # 0 = unlimited / sınırsız
        self.max_total_failures_before_abort = fallback_cfg.get("max_total_failures_before_abort", 10)

        # Hardcoded optimal cooldown settings / Sabitlenmiş optimal bekleme ayarları
        self.between_requests_sec = 2.0
        self.between_models_sec = 2.0
        self.retry_delay_sec = 5.0
        self.full_pool_exhausted_sec = 10.0

        # Provider order / Sağlayıcı sırası
        provider_order = fallback_cfg.get("provider_order", None)
        api_providers = config.get("api_providers", {})

        if provider_filter:
            ordered_keys = [provider_filter]
        elif provider_order:
            ordered_keys = [k for k in provider_order if k in api_providers]
            for k in api_providers:
                if k not in ordered_keys:
                    ordered_keys.append(k)
        else:
            ordered_keys = list(api_providers.keys())

        # Create trackers / Tracker'ları oluştur
        for provider_key in ordered_keys:
            pcfg = api_providers.get(provider_key, {})
            if not provider_filter and not pcfg.get("enabled", True):
                continue
            api_key = pcfg.get("api_key", "")
            if not api_key:
                env_prefixes = [f"GLOSSA_{provider_key.upper()}_API_KEY",
                                f"{provider_key.upper()}_API_KEY"]
                for env_name in env_prefixes:
                    val = os.environ.get(env_name)
                    if val:
                        api_key = val
                        break
            if not api_key:
                continue

            models = pcfg.get("models", [])
            if not models:
                continue

            for model_name in models:
                if model_filter and model_filter.lower() not in ("all", "all models (fallback chain)", "all models (fallback chain) / tüm modeller (yedek zincir)") and model_name.lower() != model_filter.lower():
                    continue
                tracker = ModelTracker(provider_key, model_name, api_key)
                self.trackers.append(tracker)

            # Initialize stats / İstatistik başlangıcı
            self.stats["provider_stats"][provider_key] = {
                "successes": 0,
                "failures": 0,
                "models_tried": 0,
            }

        if not self.trackers:
            print("\033[1;31m[FallbackChain] ERROR: No available provider/model found! / HATA: Hiç kullanılabilir provider/model bulunamadı!\033[0m")

    def _emit_event(self, level: str, event: str, **data):
        if not self.event_callback:
            return
        try:
            self.event_callback(level=level, event=event, **data)
        except Exception:
            pass

    def _get_query_fn(self, provider_key: str):
        """Returns query function according to provider key / Sağlayıcı anahtarına göre sorgu fonksiyonunu döner."""
        fn = PROVIDER_QUERY_FNS.get(provider_key)
        if not fn:
            raise ValueError(f"Unknown provider / Bilinmeyen sağlayıcı: {provider_key}")
        return fn

    def _get_available_trackers(self) -> List[ModelTracker]:
        """Returns currently available trackers / Şu anda kullanılabilir tracker'ları döner."""
        return [t for t in self.trackers if t.is_available()]

    def _reset_exhausted(self):
        """Resets all EXHAUSTED trackers back to READY / Tüm EXHAUSTED tracker'ları READY'ye döndür."""
        for t in self.trackers:
            t.reset()

    def call_with_fallback(self, prompt: str, temperature: float = 0.2,
                           max_tokens: int = 4096, context: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Sequentially tries prompt across all provider/model chain.
        Continues until successful.
        """
        self.stats["total_calls"] += 1

        # Decrement rate limit ban rounds / Rate limit engel turlarını azalt
        for tracker in self.trackers:
            if tracker.rate_limit_ban_rounds > 0:
                tracker.rate_limit_ban_rounds -= 1

        retry_round = 0
        while True:
            retry_round += 1
            self._emit_event("INFO", "fallback_round_started", retry_round=retry_round, context=context or {})

            # Sınır kontrolü
            if self.max_retries_per_page > 0 and retry_round > self.max_retries_per_page:
                print(f"\033[1;31m  [Fallback] FAILED: Tried {self.max_retries_per_page} rounds, no model worked. / "
                      f"BAŞARISIZ: {self.max_retries_per_page} tur denendi, hiçbir model çalışmadı.\033[0m")
                self._emit_event("ERROR", "fallback_rounds_exhausted",
                                  retry_round=retry_round,
                                  max_retries_per_page=self.max_retries_per_page,
                                  context=context or {})
                return None

            if retry_round > 1:
                wait_sec = self.retry_delay_sec if len(self.trackers) == 1 else self.full_pool_exhausted_sec
                print(f"\033[1;33m  [Fallback] Round {retry_round} — resetting all models, waiting {wait_sec}s... / "
                      f"Tur {retry_round} — tüm modeller sıfırlanıyor, {wait_sec}s bekleniyor...\033[0m")
                self._emit_event("WARN", "fallback_full_pool_reset",
                                  retry_round=retry_round,
                                  wait_sec=wait_sec,
                                  context=context or {})
                self._reset_exhausted()
                time.sleep(wait_sec)

            for tracker in self.trackers:
                if not tracker.is_available():
                    continue

                query_fn = self._get_query_fn(tracker.provider_key)
                provider_display = f"{tracker.provider_key}/{tracker.model_name}"

                try:
                    print(f"\033[36m  [Fallback] Attempting / Deneniyor: {provider_display}\033[0m")
                    self._emit_event("INFO", "fallback_model_attempt",
                                     provider=tracker.provider_key,
                                     model=tracker.model_name,
                                     context=context or {})
                    result = query_fn(
                        prompt,
                        tracker.api_key,
                        tracker.model_name,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    # Success!
                    tracker.report_success()
                    self.stats["total_successes"] += 1
                    self.stats["provider_stats"][tracker.provider_key]["successes"] += 1
                    print(f"\033[1;32m  [Fallback] [OK] Success / Başarılı: {provider_display}\033[0m")
                    self._emit_event("INFO", "fallback_model_success",
                                     provider=tracker.provider_key,
                                     model=tracker.model_name,
                                     retry_round=retry_round,
                                     context=context or {})
                    return result

                except Exception as e:
                    error_msg = str(e)
                    self.stats["total_failures"] += 1
                    self.stats["provider_stats"][tracker.provider_key]["failures"] += 1
                    self.stats["provider_stats"][tracker.provider_key]["models_tried"] += 1

                    # Rate limit detection / Rate limit tespiti
                    is_rate_limit = any(kw in error_msg.lower()
                                        for kw in ["rate limit", "429", "too many requests",
                                                    "quota", "resource_exhausted"])
                    if is_rate_limit:
                        tracker.report_rate_limit(cooldown_sec=120.0, max_rate_limits=2)
                        cooldown_remaining = max(0.0, tracker.cooldown_until - time.time())
                        self._emit_event("WARN", "fallback_model_rate_limited",
                                         provider=tracker.provider_key,
                                         model=tracker.model_name,
                                         consecutive_rate_limits=tracker.consecutive_rate_limits,
                                         tracker_state=tracker.state.value,
                                         cooldown_sec=round(cooldown_remaining, 2),
                                         error=error_msg[:500],
                                         context=context or {})
                    else:
                        print(f"\033[1;31m  [Fallback] [ERROR] Error / Hata: {provider_display}: "
                               f"{error_msg[:120]}\033[0m")
                        tracker.report_failure(
                            cooldown_sec=self.between_models_sec,
                            max_consecutive=2
                        )
                        cooldown_remaining = max(0.0, tracker.cooldown_until - time.time())
                        self._emit_event("ERROR", "fallback_model_failed",
                                         provider=tracker.provider_key,
                                         model=tracker.model_name,
                                         tracker_state=tracker.state.value,
                                         cooldown_sec=round(cooldown_remaining, 2),
                                         error=error_msg[:500],
                                         context=context or {})

                    if tracker.state == ModelState.EXHAUSTED:
                        cooldown_remaining = max(0.0, tracker.cooldown_until - time.time())
                        self._emit_event("WARN", "fallback_model_exhausted",
                                         provider=tracker.provider_key,
                                         model=tracker.model_name,
                                         cooldown_sec=round(cooldown_remaining, 2),
                                         context=context or {})

                    time.sleep(self.between_models_sec)

    def get_cooldown(self) -> float:
        """Returns request delay / İstekler arası gecikmeyi döner."""
        return self.between_requests_sec

    def print_stats(self):
        """Prints execution statistics / Çalışma istatistiklerini yazdırır."""
        print(f"\n\033[1;33m{'=' * 50}\033[0m")
        print(f"\033[1;33m  FALLBACK CHAIN STATISTICS / FALLBACK CHAIN İSTATİSTİKLERİ\033[0m")
        print(f"\033[1;33m{'=' * 50}\033[0m")
        print(f"  Total calls / Toplam çağrı        : {self.stats['total_calls']}")
        print(f"  Total successes / Toplam başarı   : {self.stats['total_successes']}")
        print(f"  Total failures / Toplam başarısız : {self.stats['total_failures']}")
        print()
        for pkey, pstats in self.stats["provider_stats"].items():
            print(f"  [{pkey}]")
            print(f"    Successes / Başarılı   : {pstats['successes']}")
            print(f"    Failures / Başarısız   : {pstats['failures']}")
        print(f"\033[1;33m{'=' * 50}\033[0m\n")

    def get_available_count(self) -> int:
        """Returns count of available models / Kullanılabilir model sayısını döner."""
        return len(self._get_available_trackers())

    def get_total_model_count(self) -> int:
        """Returns total model count / Toplam model sayısını döner."""
        return len(self.trackers)


# ─── JSON Parse Helper / JSON Çözümleme Yardımcısı ────────────────────────────

def parse_json_response(text: str):
    """Parses JSON from API text response, sanitizing markdown fences / API yanıtından JSON parse eder, markdown çitlerini temizler."""
    clean = text.strip()
    if clean.startswith("```"):
        clean = clean.replace("```json", "").replace("```JSON", "").replace("```", "").strip()

    start_arr = clean.find("[")
    start_obj = clean.find("{")

    if start_arr != -1 and (start_obj == -1 or start_arr < start_obj):
        end = clean.rfind("]") + 1
        clean = clean[start_arr:end]
    elif start_obj != -1:
        end = clean.rfind("}") + 1
        clean = clean[start_obj:end]

    try:
        return json.loads(clean)
    except Exception:
        try:
            from json_repair import repair_json
            return json.loads(repair_json(clean))
        except ImportError:
            raise


# ─── Config Loader Helper / Ayar Yükleme Yardımcısı ───────────────────────────

def load_config(config_path: str = "config/config.json") -> dict:
    """Loads config.json configuration / config.json dosyasını yükler."""
    if not os.path.exists(config_path):
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)
