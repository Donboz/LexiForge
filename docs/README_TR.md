# LexiForge (Türkçe Dokümantasyon)

Sözlük dosyalarından veri çıkarma, zenginleştirme, çeviri, birleştirme ve temizleme işlemleri için optimize edilmiş asenkron script seti.

---

## Genişletilebilirlik Notu (Diğer Formatları İşleme)

> [!IMPORTANT]
> LexiForge başlangıçta PDF formatındaki sözlükleri işlemek için tasarlanmış ve optimize edilmiş olsa da, modüler ve genişletilebilir bir mimariye sahiptir. Sistemi aşağıdakiler gibi diğer kaynak veri formatlarını işleyecek şekilde kolayca uyarlayabilirsiniz:
>
> - **Derlemler (Corpus) / Ham Metin Dosyaları**
> - **CSV / TSV Tabloları**
> - **TMX (Çeviri Belleği / Translation Memory Exchange)**
> - **JSON Dosyaları / API Verileri**
>
> Veri işleme adımları iki şekilde gerçekleştirilebilir:
>
> 1. **LLM Tabanlı (Bilişsel İşleme):** Bağlam duyarlı yeniden yapılandırma ve doğrulama işlemleri için sağlanan yedekli API zinciri (fallback chain) kullanılarak.
> 2. **Deterministik (Basit Script Mantığı):** Doğrudan alan ayrıştırma ve veri eşleştirme işlemleri için basit Python araçları yazılarak.
>
> Eğer bir **AI Kodlama Ajanı** (örn. Antigravity, Claude Code vb.) kullanıyorsanız, ajana `handlers/` ve `scripts/` dizinleri altında yeni ayrıştırıcılar (parsers) veya işleyiciler (handlers) eklemesini söyleyerek bu dosya formatlarına kolayca destek kazandırabilirsiniz.

---

## Ana Scriptler

Tüm çalıştırılabilir scriptler `scripts/` klasöründe yer almaktadır:

- `scripts/extractor.py`: PDF ham metninden terimleri ve anlamları çıkarır (`data/json/*_sozluk_*.json`) [ASENKRON].
- `scripts/enricher.py`: Sözlük kayıtlarını dilbilgisel ve anlamsal olarak LLM kullanarak zenginleştirir [ASENKRON].
- `scripts/translator.py`: Sözlük anlamlarını LLM kullanarak yeni hedef dillere çevirir [ASENKRON].
- `scripts/clean.py`: Birleştirilmiş sözlük dosyalarındaki terimleri normalize eder ve dil çiftlerine göre tekilleştirir [ORJSON].
- `scripts/clean_meanings.py`: Virgülle ayrılmış anlam listelerini böler ve mükerrer kayıtları temizler [ORJSON].
- `scripts/merge_translated.py`: Çevrilmiş/zenginleştirilmiş dil çifti dosyalarını nihai veritabanında birleştirir [ORJSON].
- `scripts/splitter.py`: Büyük JSON dosyalarını daha küçük parçalara böler [ORJSON].
- `scripts/test_providers.py`: Tanımlı API sağlayıcılarının ve modellerinin bağlantı durumunu, limitlerini ve yanıt sürelerini test eder.
- `scripts/analyze_logs.py`: Fallback zinciri olay günlüklerini okuyarak model/sağlayıcı performansı, başarı oranları ve gecikme analizleri üretir.

---

## Asenkron Yüksek Performans Mimarisi (Async Pipeline)

Maliyetleri azaltmak, rate-limit engellerini aşmak ve CPU/Ağ kaynaklarını en verimli şekilde kullanmak için iş akışları tamamen asenkron tasarlanmıştır:

### 1. Eşzamanlılık Kontrolü (Semaphore = 10)

Sağlayıcılara yapılacak eşzamanlı istek sayısı `asyncio.Semaphore(10)` ile sınırlandırılmıştır. Aynı anda en fazla 10 paket paralel olarak ağa sürülür; bitenlerin yerine yenileri alınır.

### 2. Akıllı Paketleme (Batch Size = 30)

Modellerin çıktı token sınırlarına takılıp JSON yanıtlarını yarıda kesmesini önlemek amacıyla, LLM'e gönderilecek çeviri ve zenginleştirme istekleri maksimum 30'arlı paketlere (chunk) bölünür.

### 3. Görsel İlerleme Çubuğu (`tqdm.asyncio`)

Tüm ana asenkron iş döngüleri `tqdm.asyncio` ile sarmalanmıştır. Terminalde anlık işlem hızı (it/s), tamamlanan yüzde ve Tahmini Bitiş Süresi (ETA) anlık olarak takip edilebilir.

---

## Hata Yönetimi & Akıllı Denetimler (Audit & Validation)

### 1. Dizi Uzunluğu Doğrulaması (Array Length Validation)

LLM/Provider paketinden dönen yanıtın (çevrilmiş/zenginleştirilmiş liste) uzunluğu ile gönderilen paketteki (30'lu chunk) terim uzunluğunun eşit olup olmadığı her istek sonrasında sıkı bir şekilde doğrulanır.

### 2. Hata & Fallback ile "Split-and-Retry" Entegrasyonu

Eğer dizi uzunlukları eşleşmezse, API hatası alınırsa veya JSON parse hatası oluşursa:

- Paket büyüklüğü 1'den büyükse, paket **dinamik olarak ikiye bölünür (split)** ve her yarım asenkron olarak yeniden denenir (divide & conquer).
- Paket büyüklüğü 1'e kadar indiyse ve hâlâ hata alınıyorsa, `handlers/fallback_chain.py` üzerindeki yedek modele geçme (fallback zinciri) tetiklenir veya o terim için boş veri döndürülerek sürecin tıkanması önlenir.

### 3. orjson Entegrasyonu

Büyük sözlük JSON dosyalarını CPU'yu kilitlemeden jet hızıyla yüklemek, işlemek ve kaydetmek için standart `json` kütüphanesi yerine C-hızındaki `orjson` kütüphanesi entegre edilmiştir. Dosya işlemleri doğrudan UTF-8 byte düzeyinde binary modda (`wb`/`rb`) gerçekleştirilir.

---

## Namespaced Redis Önbellek Katmanı (Redis Cache)

Çeviri, çıkarma ve zenginleştirme adımlarını hızlandırmak ve API maliyetlerini azaltmak amacıyla `redis.asyncio` asenkron önbellek katmanı entegre edilmiştir.

### 1. Konfigürasyon (`config/config.json`)

```json
"redis": {
  "host": "127.0.0.1",
  "port": 6379,
  "db": 0,
  "password": null
}
```

### 2. Dinamik Namespaced Önbellek Şeması

Önbellek anahtarları kaynak (`source_lang`) ve hedef (`target_lang`) dillere göre dinamik olarak ayrışır. Değerler başarıyla tamamlandığında **30 gün (2.592.000 saniye) TTL** ile Redis'e kaydedilir.

- **Format:** `sözlük:{source_lang}:{target_lang}:{sha256_hash}`
- **SHA-256 Girdileri:**
  - **Extractor:** `f"{domain}||{page_text}"`
  - **Enricher:** `f"{term}||{sorted_definitions_meanings_and_domains}"`
  - **Translator:** `f"{term}||{domain}||{meaning_hint}||{example_source}"`

### 3. Redis Pipeline / Bulk MGET Kontrolü

Sorgular tek tek yapılmaz; işlem yapılacak tüm terimlerin/sayfaların anahtar listesi önceden çıkarılarak tek bir asenkron Redis `MGET` sorgusu ile sorgulanır. Sonuçlar `cache_hits` ve `cache_misses` olarak ayrılır. Cache hit olanlar API'ye gitmeden bellek haritası üzerinden doğrudan nihai veritabanına yeniden inşa (rebuild) edilir.

---

## 📷 OCR ve Sütun Analizi (PaddleOCR)

Taranmış veya çok sütunlu PDF sözlükleri için **LexiForge**, **PaddleOCR** ve **PyMuPDF** entegrasyonuna sahiptir.

### Öne Çıkan Özellikler:
- **Taranmış PDF Desteği:** Sayfaları dinamik olarak 300 DPI görsellere çevirerek yerel olarak okur.
- **Dikey Sütun Sıralama (Heuristic):** 2 veya 3 sütunlu sözlüklerde metinlerin soldan sağa karışmasını önler, her sütunu yukarıdan aşağıya sırayla işler.
- **Seçim Kilitleri (UI):** İşlemi tamamlanmış PDF'ler, JSON'lar veya hedef diller menülerde kilitlenir (seçilemez hale gelir).

### Kurulum (Python 3.11 gereklidir):
```bash
py -3.11 -m pip install pymupdf paddlepaddle paddleocr
```

---

## 🛠️ Kullanım ve CLI Orchestrator (`run.py`)

Projedeki tüm scriptler `run.py` içerisindeki tek bir interaktif menü üzerinden kolayca yönetilebilir.

```bash
py -3.11 run.py
```

### Menü Seçenekleri:

1. `extractor`: PDF'den terim çıkarır (Asenkron + Sayfa Split-and-Retry + PaddleOCR)
2. `enricher`: Dilbilgisel/anlamsal zenginleştirme yapar (Asenkron + Split-and-Retry)
3. `translator`: Anlamları hedef dillere çevirir (Asenkron + Split-and-Retry)
4. `analyze_logs`: Modellerin ve API sağlayıcılarının performans, latency ve başarı analizini yapar
5. `clean`: Merged klasöründe normalizasyon ve birleştirme yapar (orjson)
6. `merge`: Translated klasöründeki çevirileri final veritabanında birleştirir (orjson)
7. `clean_meanings`: Virgülle ayrılmış anlamları böler ve tekilleştirir (orjson)
8. `test_redis`: Redis bağlantısını doğrular
9. `quit`: Menüden çıkar

---

## Model & Provider Performans Analizi (`scripts/analyze_logs.py`)

Geliştirilen yedekli LLM yapısının (`handlers/fallback_chain.py`) ürettiği olay günlüklerini (event logs) okuyarak detaylı bir performans analizi sunar:

- **Genel Metrikler:** Toplam log dosya sayısı, toplam API çağrısı, başarı/hata oranları, global başarı yüzdesi ve ortalama gecikme süresi (latency).
- **Model Sıralamaları (Performance Rankings):** Her model için başarı oranı, toplam çağrı, hata sayısı, rate-limit sıklığı ve ortalama yanıt süresi (avg latency) tablosu.
- **Sağlayıcı Özeti (Provider Summary):** API sağlayıcılarının genel başarı oranı ve gecikme karşılaştırması.
- **Highlights (Öne Çıkanlar):** En güvenilir model, en hızlı model (minimum latency), en yavaş model ve en başarısız (rate-limit / hata alan) model tespiti.
- **Kullanılmayan Modeller (Never Ran):** Config dosyalarında tanımlanmış fakat hiç çalıştırılmamış (loglanmamış) modellerin sağlayıcı bazlı listesi.

---

## Progress ve Log Mimarisi

### Progress Klasörü

Tüm ilerleme dosyaları `data/progress/` altında tutulur:

- `data/progress/extractor/` → extractor sayfa bazlı ilerleme
- `data/progress/translator/` → translator batch/öğe bazlı ilerleme
- `data/progress/clean/` → temizlik işlemi ilerlemesi

### Log Klasörü

Yapılandırılmış çalışma günlükleri `data/logs/` altında tutulur:

- `<run_id>.jsonl` → olay akışı (INFO/WARN/ERROR)
- `<run_id>_summary.json` → çalışma performansı özeti
