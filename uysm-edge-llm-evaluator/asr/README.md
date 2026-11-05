# ATBK-NG ASR-LLM Pipeline - Local PC Sürümü

Askeri saha operasyon talimatnamesi için Türkçe konuşma tanıma, doküman Q&A ve TTS sistemi.

## Özellikler

-  **TTS ile Soru Üretimi**: 15 askeri soru için .wav ses dosyaları
-  **Whisper ASR**: Türkçe konuşma tanıma (tiny, base, small, medium modelleri)
-  **Qwen LLM**: Doküman tabanlı soru-cevap (0.5B, 1.5B, 3B modelleri)
-  **Otomatik TTS Cevap**: LLM cevaplarını otomatik ses dosyasına çevirme
-  **Benchmark Analizi**: Model performans karşılaştırması
-  **GPU/CPU Desteği**: Otomatik cihaz algılama
-  **Modüler Çalışma**: İstediğiniz adımları seçerek çalıştırın

##  Gereksinimler

### Minimum Sistem
- Python 3.8+
- 16GB RAM
- 50GB Disk (model cache için)

### GPU (Önerilen)
- NVIDIA GPU (CUDA 11.8+)
- 8GB+ VRAM (RTX 3060 veya üzeri)
- 16GB VRAM için whisper-medium + Qwen-1.5B önerilir

### CPU Modu
- CPU modunda çalışabilir ancak 10-20x daha yavaş
- RAM kullanımı daha yüksek

##  Kurulum

### 1. Repository'yi İndirin
```bash
git clone https://github.com/cem8kaya/tr-asr-llm-pipeline.git
cd atbk-asr-llm-pipeline
```

### 2. Sanal Ortam Oluşturun (Önerilen)
```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# veya
venv\Scripts\activate  # Windows
```

### 3. Bağımlılıkları Yükleyin
```bash
pip install -r requirements.txt
```

### 4. FFmpeg Kurulumu (TTS için gerekli)

**Ubuntu/Debian:**
```bash
sudo apt-get install ffmpeg
```

**Windows:**
- [FFmpeg'i indirin](https://ffmpeg.org/download.html)
- PATH'e ekleyin

**macOS:**
```bash
brew install ffmpeg
```

## Kullanım

### Tüm Pipeline'ı Çalıştır
```bash
python atbk_asr_llm_local.py
```

### Sadece Belirli Adımları Çalıştır
```bash
# Sadece 1. ve 3. adımları çalıştır
python atbk_asr_llm_local.py --steps 1,3

# Benchmark'ları atla
python atbk_asr_llm_local.py --skip-benchmark
```

### Adımlar:
1. **TTS Soru Üretimi** - 15 soru için ses dosyaları oluştur
2. **Whisper Benchmark** - 3-4 Whisper modelini karşılaştır
3. **Tam Transkripsiyon** - Tüm soruları transkribe et
4. **Qwen Benchmark** - 2-3 Qwen modelini karşılaştır
5. **Q&A Üretimi** - Tüm soruları cevapla ve TTS ile kaydet
6. **Doküman Analizi** - 5 farklı analiz promptu çalıştır
7. **Rapor Oluştur** - Final benchmark raporu

## Çıktı Dosyaları

Pipeline çalıştıktan sonra `ATBK_Pipeline/` dizininde:

```
ATBK_Pipeline/
├── transcripts/
│   └── all_transcriptions.json
├── reports/
│   ├── benchmark_report.txt      # 📊 Ana rapor
│   ├── benchmark_results.json
│   ├── whisper_benchmark.json
│   ├── qwen_benchmark.json
│   ├── all_qa_results.json
│   └── document_analysis.json
└── data/
    ├── question_files.json
    ├── saha_operasyon_talimatnamesi.txt
    └── audio/
        ├── questions/          # 15 soru .wav dosyası
        ├── answers/            # 15 cevap .wav dosyası
        └── benchmark/          # benchmark ses örnekleri
```

## ⚙️ Konfigürasyon

`atbk_asr_llm_local.py` dosyasında `Config` sınıfını düzenleyebilirsiniz:

```python
class Config:
    # Test edilecek modeller
    WHISPER_MODELS = [
        "openai/whisper-tiny",
        "openai/whisper-base",
        "openai/whisper-small",
        "openai/whisper-medium",
        "openai/whisper-large-v2",
        "openai/whisper-large-v3",
    ]
    
    QWEN_MODELS = [
        "Qwen/Qwen2.5-0.5B-Instruct",
        "Qwen/Qwen2.5-1.5B-Instruct",
        # "Qwen/Qwen2.5-3B-Instruct",  # Yüksek VRAM
    ]
    
    # Production için seçilen modeller
    SELECTED_WHISPER = "openai/whisper-small"
    SELECTED_QWEN = "Qwen/Qwen2.5-1.5B-Instruct"
    
    # 4-bit quantization (GPU için)
    USE_4BIT = True
```

## Model Önerileri

### VRAM'e Göre Model Seçimi

| VRAM | Whisper | Qwen | Toplam Süre (15 soru) |
|------|---------|------|----------------------|
| 6GB  | tiny    | 0.5B | ~5 dakika |
| 8GB  | small   | 0.5B | ~8 dakika |
| 12GB | small   | 1.5B | ~12 dakika |
| 16GB | medium  | 1.5B | ~15 dakika |
| 24GB | medium  | 3B   | ~20 dakika |

### Hız vs Kalite

**Hızlı Test (Minimum Kaynak):**
```python
SELECTED_WHISPER = "openai/whisper-tiny"
SELECTED_QWEN = "Qwen/Qwen2.5-0.5B-Instruct"
```

**Dengeli (Önerilen):**
```python
SELECTED_WHISPER = "openai/whisper-small"
SELECTED_QWEN = "Qwen/Qwen2.5-1.5B-Instruct"
```

**Maksimum Kalite:**
```python
SELECTED_WHISPER = "openai/whisper-medium"
SELECTED_QWEN = "Qwen/Qwen2.5-3B-Instruct"
```

## Sorun Giderme

### CUDA Out of Memory
```bash
# Daha küçük modeller kullanın
SELECTED_WHISPER = "openai/whisper-tiny"
SELECTED_QWEN = "Qwen/Qwen2.5-0.5B-Instruct"
```

### CPU Modunda Çok Yavaş
- Model boyutunu küçültün
- Sadece gerekli adımları çalıştırın
- Benchmark'ları atlayın: `--skip-benchmark`

### FFmpeg Bulunamadı
```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# Windows - PATH'e ekleyin
# macOS
brew install ffmpeg
```

### bitsandbytes CUDA Hatası
```bash
# CPU modunda quantization'ı devre dışı bırakın
USE_4BIT = False
```

## Örnek Çıktı

```
╔══════════════════════════════════════════════════════════════════════╗
║      ATBK-NG ASR-LLM PIPELINE - BENCHMARK RAPORU                     ║
║      Tarih: 2026-02-09 14:30:45                                      ║
║      Cihaz: CUDA                                                     ║
╚══════════════════════════════════════════════════════════════════════╝

1. WHISPER MODEL BENCHMARK SONUÇLARI
======================================================================

┌─────────────────┬──────────────┬─────────────────┬─────────────────┐
│ Model           │ Yükleme      │ Ort. Çıkarım    │ VRAM            │
├─────────────────┼──────────────┼─────────────────┼─────────────────┤
│ whisper-tiny    │ 3.24s        │ 0.52s           │ 1.12GB          │
│ whisper-base    │ 3.89s        │ 0.68s           │ 1.48GB          │
│ whisper-small   │ 5.12s        │ 1.23s           │ 2.31GB          │
└─────────────────┴──────────────┴─────────────────┴─────────────────┘

...
```

## ATBK-NG Entegrasyon

Jetson AGX Xavier/Orin için öneriler:
- `whisper-small` + `Qwen-0.5B` kombinasyonu
- 4-bit quantization aktif
- Batch processing modu
- LoRa/UHF mesh network desteği

## Loglar

Tüm çıktılar hem console'da hem de `atbk_pipeline.log` dosyasında:
```bash
tail -f atbk_pipeline.log
```

## Katkıda Bulunma

Issues ve pull request'ler memnuniyetle karşılanır.

## Lisans

[Lisans bilgisi eklenecek]

## 📧 İletişim

Sorularınız için: [cem8kaya@gmail.com]
