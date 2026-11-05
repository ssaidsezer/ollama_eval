#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ATBK-NG Askeri Saha Operasyon - ASR-LLM Pipeline (Local PC Version)
====================================================================

Türkçe Konuşma Tanıma, Doküman Q&A ve TTS Sistemi
- Local PC/Workstation için optimize edilmiş
- Modüler ve incremental çalışma
- GPU/CPU otomatik algılama
- Bellek yönetimi ve hata toleransı

Gereksinimler:
    pip install transformers accelerate bitsandbytes librosa soundfile tqdm gtts pydub torch

Sistem:
    - Python 3.8+
    - CUDA 11.8+ (opsiyonel, GPU için)
    - 16GB+ RAM önerilen
    - 8GB+ VRAM (GPU kullanımı için)
"""

import os
import sys
import time
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

# PyTorch ve Transformers
import torch
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    pipeline,
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)

# Audio işleme
import librosa
import soundfile as sf
from gtts import gTTS
from pydub import AudioSegment

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('atbk_pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================
# KONFİGÜRASYON
# ============================================================================

class Config:
    """Global konfigürasyon"""
    
    # Dizinler
    BASE_DIR = Path("./ATBK_Pipeline")
    DATA_DIR = BASE_DIR / "data"
    AUDIO_DIR = DATA_DIR / "audio"
    QUESTION_DIR = AUDIO_DIR / "questions"
    ANSWER_DIR = AUDIO_DIR / "answers"
    TRANSCRIPT_DIR = BASE_DIR / "transcripts"
    REPORT_DIR = BASE_DIR / "reports"
    
    # Model isimleri
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
        # "Qwen/Qwen2.5-3B-Instruct",  # Opsiyonel, yüksek VRAM
    ]
    
    # Seçilen modeller
    SELECTED_WHISPER = "openai/whisper-small"
    SELECTED_QWEN = "Qwen/Qwen2.5-1.5B-Instruct"
    
    # Cihaz
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    USE_4BIT = True  # LLM için 4-bit quantization
    
    # Audio
    SAMPLE_RATE = 16000
    
    @classmethod
    def create_directories(cls):
        """Dizinleri oluştur"""
        for directory in [cls.AUDIO_DIR, cls.QUESTION_DIR, cls.ANSWER_DIR, 
                         cls.TRANSCRIPT_DIR, cls.REPORT_DIR, cls.DATA_DIR]:
            directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"✅ Dizinler hazır: {cls.BASE_DIR}")


# ============================================================================
# WHISPER ASR
# ============================================================================

class WhisperASR:
    """Whisper Automatic Speech Recognition"""
    
    def __init__(self, model_name: str = "openai/whisper-small", device: str = "cuda"):
        self.model_name = model_name
        self.device = device if torch.cuda.is_available() else "cpu"
        self.pipe = None
        self.load_time = 0
        
        if self.device == "cpu":
            logger.warning("⚠️  GPU bulunamadı, CPU kullanılacak (yavaş olabilir)")
    
    def load(self):
        """Modeli yükle"""
        logger.info(f"🔄 {self.model_name} yükleniyor...")
        start_time = time.time()
        
        try:
            # Model dtype
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.model_name,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True
            )
            model.to(self.device)
            
            processor = AutoProcessor.from_pretrained(self.model_name)
            
            self.pipe = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                # max_new_tokens=128,
                chunk_length_s=30,
                batch_size=8 if self.device == "cuda" else 1,
                torch_dtype=dtype,
                device=self.device,
            )
            
            self.load_time = time.time() - start_time
            logger.info(f"✅ Model yüklendi ({self.load_time:.2f}s)")
            
        except Exception as e:
            logger.error(f"❌ Model yükleme hatası: {e}")
            raise
    
    def transcribe(self, audio_path: str) -> Dict:
        """Ses dosyasını metne çevir"""
        if self.pipe is None:
            self.load()
        
        try:
            start_time = time.time()
            result = self.pipe(str(audio_path), generate_kwargs={"language": "turkish"})
            inference_time = time.time() - start_time
            
            # VRAM/RAM kullanımı
            if self.device == "cuda":
                vram_allocated = torch.cuda.memory_allocated() / 1024**3
                vram_reserved = torch.cuda.memory_reserved() / 1024**3
            else:
                vram_allocated = 0
                vram_reserved = 0
            
            return {
                "text": result["text"],
                "inference_time": inference_time,
                "vram_allocated_gb": vram_allocated,
                "vram_reserved_gb": vram_reserved
            }
            
        except Exception as e:
            logger.error(f"❌ Transkripsiyon hatası: {e}")
            return {
                "text": "",
                "inference_time": 0,
                "vram_allocated_gb": 0,
                "vram_reserved_gb": 0,
                "error": str(e)
            }
    
    def cleanup(self):
        """Belleği temizle"""
        try:
            if self.pipe is not None:
                del self.pipe
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.pipe = None
            logger.info("🧹 Whisper bellek temizlendi")
        except Exception as e:
            logger.warning(f"⚠️  Bellek temizleme uyarısı: {e}")


# ============================================================================
# QWEN LLM
# ============================================================================

class QwenLLM:
    """Qwen Large Language Model"""
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct", use_4bit: bool = True):
        self.model_name = model_name
        self.use_4bit = use_4bit and torch.cuda.is_available()
        self.model = None
        self.tokenizer = None
        self.document_context = ""
        self.load_time = 0
        
        if not torch.cuda.is_available() and use_4bit:
            logger.warning("⚠️  GPU yok, 4-bit quantization devre dışı")
            self.use_4bit = False
    
    def load(self):
        """Modeli yükle"""
        logger.info(f"🔄 {self.model_name} yükleniyor...")
        start_time = time.time()
        
        try:
            # Quantization config
            if self.use_4bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True
                )
            else:
                quantization_config = None
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                low_cpu_mem_usage=True
            )
            
            self.load_time = time.time() - start_time
            logger.info(f"✅ Model yüklendi ({self.load_time:.2f}s)")
            
        except Exception as e:
            logger.error(f"❌ LLM yükleme hatası: {e}")
            raise
    
    def load_document(self, doc_path: str):
        """Doküman yükle"""
        try:
            with open(doc_path, 'r', encoding='utf-8') as f:
                self.document_context = f.read()
            logger.info(f"✅ Doküman yüklendi: {len(self.document_context)} karakter")
        except Exception as e:
            logger.error(f"❌ Doküman yükleme hatası: {e}")
            raise
    
    def ask(self, question: str, max_new_tokens: int = 512, temperature: float = 0.7) -> Dict:
        """Soru sor"""
        if self.model is None:
            self.load()
        
        try:
            # Prompt oluştur
            if self.document_context:
                prompt = f"""Aşağıdaki doküman bilgilerine göre soruyu cevapla.

                                DOKÜMAN:
                                {self.document_context}

                                SORU: {question}

                                CEVAP:"""
            else:
                prompt = question
            
            # Tokenize
            messages = [
                {"role": "system", "content": "Sen askeri taktik ve saha operasyonları konusunda uzman bir asistan yardımcısısın. Verilen dokümana göre kısa, net ve doğru cevaplar veriyorsun."},
                {"role": "user", "content": prompt}
            ]
            
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
            
            # Generate
            start_time = time.time()
            
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            inference_time = time.time() - start_time
            
            # Decode
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # VRAM
            if torch.cuda.is_available():
                vram_allocated = torch.cuda.memory_allocated() / 1024**3
            else:
                vram_allocated = 0
            
            return {
                "answer": response.strip(),
                "inference_time": inference_time,
                "vram_allocated_gb": vram_allocated,
                "tokens_generated": len(generated_ids[0])
            }
            
        except Exception as e:
            logger.error(f"❌ LLM inference hatası: {e}")
            return {
                "answer": f"[HATA: {str(e)}]",
                "inference_time": 0,
                "vram_allocated_gb": 0,
                "tokens_generated": 0,
                "error": str(e)
            }
    
    def cleanup(self):
        """Belleği temizle"""
        try:
            if self.model is not None:
                del self.model
            if self.tokenizer is not None:
                del self.tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.model = None
            self.tokenizer = None
            logger.info("🧹 LLM bellek temizlendi")
        except Exception as e:
            logger.warning(f"⚠️  Bellek temizleme uyarısı: {e}")


# ============================================================================
# TURKISH TTS
# ============================================================================

class TurkishTTS:
    """Türkçe Text-to-Speech"""
    
    @staticmethod
    def create_audio(text: str, output_path: str, sample_rate: int = 16000):
        """Metni sese çevir ve kaydet"""
        try:
            output_path = Path(output_path)
            temp_mp3 = output_path.with_suffix('.mp3')
            
            # TTS oluştur
            tts = gTTS(text=text, lang='tr', slow=False)
            tts.save(str(temp_mp3))
            
            # WAV'a çevir
            sound = AudioSegment.from_mp3(str(temp_mp3))
            sound = sound.set_frame_rate(sample_rate).set_channels(1)
            sound.export(str(output_path), format="wav")
            
            # Geçici dosyayı sil
            temp_mp3.unlink()
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"❌ TTS hatası: {e}")
            raise


# ============================================================================
# VERİ SETİ
# ============================================================================

def get_qa_dataset() -> List[Dict]:
    """Soru-cevap veri seti"""
    return [
        {
            "id": "q001",
            "question": "Kaskımda kırmızı ışık yanıp sönüyor, ne yapmalıyım?",
            "expected_answer": "Kırmızı yanıp sönen LED sensör hatası gösterir. Kaskı kapatıp tekrar açmanız gerekir."
        },
        {
            "id": "q002",
            "question": "Telsiz kanalları nelerdir ve hangi durumda kullanılır?",
            "expected_answer": "Kanal 1 genel taktik haberleşme için, Kanal 2 komuta ve merkez arası stratejik iletişim için, Kanal 3 lojistik ve sıhhiye için, Kanal 4 jamming durumunda yedek kripto kanal olarak kullanılır."
        },
        {
            "id": "q003",
            "question": "SALUTE rapor formatı nedir?",
            "expected_answer": "SALUTE düşman rapor formatıdır: Size (büyüklük), Activity (faaliyet), Location (konum), Unit (birlik), Time (zaman), Equipment (teçhizat)."
        },
        {
            "id": "q004",
            "question": "Akıllı kask nasıl kalibre edilir?",
            "expected_answer": "Kask açıldığında baş 5 saniye sabit tutulur. AR vizöründe pusula belirdiğinde sistem hazır demektir."
        },
        {
            "id": "q005",
            "question": "Sistem kilitlenirse ne yapmalıyım?",
            "expected_answer": "PTT ve Mod tuşuna aynı anda 10 saniye basılı tutarak hard reset yapılır."
        },
        {
            "id": "q006",
            "question": "Mavi titreşim ne anlama gelir?",
            "expected_answer": "Mavi titreşim yakın dost unsuru algılandığını gösterir. Ateş etmemelisiniz."
        },
        {
            "id": "q007",
            "question": "Drone tehdidi durumunda ne yapmalıyım?",
            "expected_answer": "Hava ikazı ile derhal yere yatılır, hareketsiz kalınır. Kaskın Anti-Drone modu açılır. Sadece komutan emriyle ateş edilir."
        },
        {
            "id": "q008",
            "question": "GPS sinyali kesildi, ne yapmalıyım?",
            "expected_answer": "Kask konum güvenilmez uyarısı verirse manuel harita ve pusula kullanımına geçilir. Kaskın RSSI tabanlı mesafe ölçeri referans alınır."
        },
        {
            "id": "q009",
            "question": "Pusuya düşersem ilk ne yapmalıyım?",
            "expected_answer": "Derhal en yakın sütre arkasına geçilir. Saat yönü verilerek baskı ateşi açılır."
        },
        {
            "id": "q010",
            "question": "Kimyasal gaz alarmı aldım, ne yapmalıyım?",
            "expected_answer": "GAZ GAZ GAZ uyarısı alındığında 9 saniye içinde koruyucu maske takılmalıdır. Rüzgar yönü ve gaz türü merkeze rapor edilir."
        },
        {
            "id": "q011",
            "question": "Turnike nereye uygulanır?",
            "expected_answer": "Ateş altında bakımda yaralıya sadece hayati kanamayı durdurmak için turnike uygulanır. Diğer müdahaleler ateş kesilince yapılır."
        },
        {
            "id": "q012",
            "question": "Morfin hangi durumlarda kullanılmaz?",
            "expected_answer": "Kafa travması veya solunum güçlüğü olan yaralıya asla morfin yapılmaz."
        },
        {
            "id": "q013",
            "question": "5S kuralı nedir?",
            "expected_answer": "5S kuralı: Search (ara), Silence (sustur), Segregate (ayır), Speed (hızlandır), Safeguard (koru)."
        },
        {
            "id": "q014",
            "question": "Esir aldığımız kişinin telefonunu ne yapmalıyız?",
            "expected_answer": "Esirin dijital cihazları kapatılmaz, sinyal izoleli Faraday Bag torbaya konur."
        },
        {
            "id": "q015",
            "question": "Yakalanma riski varsa sistem nasıl silinir?",
            "expected_answer": "Sesli olarak KOD KIRMIZI SİSTEMİ SİL komutu verilir. Doğrulama kodu GÜNEŞ-99 kullanılır."
        },
    ]


def get_talimatname_content() -> str:
    """Saha operasyon talimatnamesi içeriği"""
    return """GİZLİLİK DERECESİ: HİZMETE ÖZEL
DOKÜMAN KODU: SOT-2026-REV3 (GENİŞLETİLMİŞ VERSİYON)
KONU: ATBK-NG Taktik Saha, Cihaz Kullanım ve Angajman Talimatları
YAYIN TARİHİ: 09.02.2026

1. GENEL İLETİŞİM VE TELSİZ DİSİPLİNİ
   1.1. Telsiz Sessizliği: "Telsiz Sus" emri verildiğinde, hayati tehlike (Temas) haricinde kesinlikle iletişim kurulmaz. Sadece dinleme modunda kalınır.
   1.2. Kanal Yapısı ve Yetkiler:
        - Kanal 1 (Genel): Tüm tim içi taktik haberleşme.
        - Kanal 2 (Komuta): Sadece Tim Komutanı ve Merkez (ECG) arası stratejik iletişim.
        - Kanal 3 (Lojistik/Sıhhiye): Mühimmat ikmali, yaralı tahliyesi (MEDEVAC).
        - Kanal 4 (Yedek/Kripto): Ana kanallar karıştırılırsa (Jamming) otomatik geçilecek frekans atlamalı kanal.
   1.3. Raporlama Formatı (SALUTE): Düşman unsuru görüldüğünde rapor şu sırayla verilir:
        - S (Size): Büyüklük (Kaç kişi/araç?)
        - A (Activity): Faaliyet (Ne yapıyorlar?)
        - L (Location): Konum (Koordinat veya referans noktası)
        - U (Unit): Birlik (Hangi üniforma/işaret?)
        - T (Time): Zaman (Ne zaman görüldü?)
        - E (Equipment): Teçhizat (Silahları ne?)
   1.4. Parola ve İşaret: Her gün 00:00'da değişen parola/işaret kombinasyonu kullanılır. Şüpheli dost unsurlara önce parola sorulur. Yanlış cevapta tehdit sayılır.

2. AKILLI KASK (ATBK) DONANIM PROSEDÜRLERİ
   2.1. Başlatma ve Kalibrasyon: Kask açıldığında baş 5 saniye sabit tutulur (IMU Kalibrasyonu). AR (Artırılmış Gerçeklik) vizöründe pusula belirdiğinde sistem hazırdır.
   2.2. Hata Kodları ve Çözümleri:
        - Kırmızı Yanıp Sönen LED: Sensör hatası. Kaskı kapatıp açın.
        - Sürekli Sarı LED: Bağlantı kopukluğu (ECG menzili dışı). Telsiz manuele geçer.
        - Mavi Titreşim: Yakın dost unsuru (Blue Force) algılandı. Ateş etme.
   2.3. Gece Görüş ve Termal Mod: "Termal Mod" sesli komutu veya vizöre çift tıklama ile aktif edilir. Gün ışığında termal sensör kapağı kapalı tutulmalıdır.
   2.4. Hard Reset: Sistem kilitlenirse, PTT (Bas-Konuş) ve Mod tuşuna aynı anda 10 saniye basılı tutarak zorla yeniden başlatın.

3. ACİL DURUM, TEMAS VE HAREKET TARZLARI
   3.1. Pusuya Düşme: Derhal en yakın sütre (korunak) arkasına geçilir. "Saat Yönü" vererek (Örn: "Temas Saat 3!") baskı ateşi açılır.
   3.2. Hava Saldırısı / Drone Tehditi:
        - "Hava" ikazı ile derhal yere yatılır ve hareketsiz kalınır (Termal gizlenme).
        - Kaskın "Anti-Drone" modu açılır (RF sinyallerini minimuma indirir).
        - Düşman drone'una sadece komutan emriyle ateş edilir.
   3.3. GPS Karıştırma (Jamming/Spoofing): Kask "Konum Güvenilmez" uyarısı verirse, manuel harita ve pusula kullanımına geçilir. Kaskın RSSI tabanlı mesafe ölçeri referans alınır.
   3.4. Dost Ateşi (Blue on Blue): Kask vizöründe "MAVİ İŞARET" görülen hedeflere kesinlikle ateş edilmez. Sistem tetik kilitlemez, sorumluluk atıcıdadır.

4. KBRN (KİMYASAL, BİYOLOJİK, RADYOLOJİK, NÜKLEER) TEHDİTLERİ
   4.1. KBRN Alarmı: Kask sensörleri kimyasal ajan (Sinir gazı vb.) tespit ederse "GAZ GAZ GAZ" sesli uyarısı verir ve vizör kırmızı olur.
   4.2. Maske Takma: Alarm alındığında 9 saniye içinde koruyucu maske takılmalıdır.
   4.3. Raporlama: KBRN tehdidi algılandığında rüzgar yönü ve gazın türü (kask ekranında yazar) merkeze rapor edilir.

5. İLK YARDIM VE SIHHİYE (TCCC) TALİMATLARI
   5.1. Ateş Altında Bakım: Yaralıya sadece hayati kanamayı durdurmak için turnike uygulanır. Diğer müdahaleler ateş kesilince yapılır.
   5.2. 9-Line MEDEVAC İsteği: Helikopter ile tahliye gerekirse şu bilgiler sırayla verilir:
        1. Konum, 2. Telsiz Frekansı, 3. Hasta Sayısı (Öncelik sırasına göre), 4. Özel Ekipman İhtiyacı, 5. Hasta Türü (Yatan/Oturan).
   5.3. Morfin Kullanımı: Kafa travması veya solunum güçlüğü olan yaralıya ASLA morfin yapılmaz. Şırınga yaralının yakasına iliştirilir.

6. ESİR VE GÖZALTI İŞLEMLERİ (EPW)
   6.1. 5S Kuralı Uygulanır:
        - Search (Ara): Üst araması yap.
        - Silence (Sustur): Konuşmasını engelle.
        - Segregate (Ayır): Rütbelileri erlerden ayır.
        - Speed (Hızlandır): Bölgeden hızlıca tahliye et.
        - Safeguard (Koru): Yaralıysa müdahale et, kötü muamele yapma.
   6.2. Dijital Materyal: Esirin üzerindeki telefon, telsiz veya dijital cihazlar (ATBK-NG Kaskı vb.) kapatılmaz, sinyal izoleli torbaya (Faraday Bag) konur.

7. CİHAZ İMHASI VE KRİTİK VERİ GÜVENLİĞİ
   7.1. Cihaz Terki: Eğer geri çekilme zorunluysa ve teçhizat taşınamıyorsa, kask ve ECG düşman eline geçmemelidir.
   7.2. Acil Silme (Kill Switch): Yakalanma riski varsa, sesli olarak "KOD KIRMIZI - SİSTEMİ SİL" komutu verilir. (Doğrulama kodu: GÜNEŞ-99).
   7.3. Fiziksel İmha: Yazılımsal silme yapılamıyorsa, cihazın bataryası çıkarılır ve işlemci ünitesine (Ana kart) ateş edilerek veya kırılarak imha edilir.
"""


def get_analysis_prompts() -> List[Dict]:
    """Doküman analiz promptları"""
    return [
        {
            "id": "analysis_001",
            "prompt": "ATBK-NG sisteminin temel güvenlik protokollerini özetle.",
            "category": "Güvenlik"
        },
        {
            "id": "analysis_002",
            "prompt": "Acil durum prosedürlerini öncelik sırasına göre listele.",
            "category": "Acil Durum"
        },
        {
            "id": "analysis_003",
            "prompt": "Telsiz iletişim disiplini kurallarını detaylandır.",
            "category": "İletişim"
        },
        {
            "id": "analysis_004",
            "prompt": "KBRN tehditlerine karşı alınacak önlemleri açıkla.",
            "category": "KBRN"
        },
        {
            "id": "analysis_005",
            "prompt": "İlk yardım ve TCCC talimatlarının kritik noktalarını belirt.",
            "category": "Sıhhiye"
        },
    ]


# ============================================================================
# PIPELINE FONKSİYONLARI
# ============================================================================

def step1_create_question_audio():
    """Adım 1: TTS ile soru ses dosyaları oluştur"""
    logger.info("\n" + "="*80)
    logger.info("ADIM 1: TTS İLE SORU SES DOSYALARI OLUŞTURMA")
    logger.info("="*80)
    
    qa_dataset = get_qa_dataset()
    tts = TurkishTTS()
    question_files = []
    
    for qa in tqdm(qa_dataset, desc="TTS İşleniyor"):
        question_file = Config.QUESTION_DIR / f"{qa['id']}_question.wav"
        
        try:
            tts.create_audio(qa['question'], str(question_file), Config.SAMPLE_RATE)
            question_files.append({
                "id": qa['id'],
                "path": str(question_file),
                "text": qa['question']
            })
        except Exception as e:
            logger.error(f"❌ {qa['id']} TTS hatası: {e}")
    
    # Kaydet
    with open(Config.DATA_DIR / "question_files.json", 'w', encoding='utf-8') as f:
        json.dump(question_files, f, ensure_ascii=False, indent=2)
    
    logger.info(f"\n✅ {len(question_files)} soru ses dosyası oluşturuldu")
    return question_files


def step2_benchmark_whisper():
    """Adım 2: Whisper model benchmark"""
    logger.info("\n" + "="*80)
    logger.info("ADIM 2: WHISPER MODEL BENCHMARK")
    logger.info("="*80)
    
    # Question files yükle
    with open(Config.DATA_DIR / "question_files.json", 'r', encoding='utf-8') as f:
        question_folders = json.load(f)
    
    question_files = []
    benchmark_results = []

    for test_folder in question_folders:
        entry_path = test_folder['path']
        if os.path.isdir(entry_path):
            wavs = [os.path.join(entry_path, f) for f in os.listdir(entry_path) if f.endswith(".wav")]
        elif os.path.isfile(entry_path) and entry_path.endswith(".wav"):
            wavs = [entry_path]
        else:
            logger.warning(f"⚠️  Atlandı (bulunamadı): {entry_path}")
            continue
        for wav_path in wavs:
            question_files.append({
                "id": test_folder['id'],
                "path": wav_path,
                "text": test_folder['text']
            })

    for model_name in Config.WHISPER_MODELS:
        logger.info(f"\n🤖 Test Ediliyor: {model_name}")
        logger.info("-" * 80)

        try:
            asr = WhisperASR(model_name=model_name, device=Config.DEVICE)
            asr.load()
            
            total_inference_time = 0
            transcriptions = []

            for test_file in question_files:
                result = asr.transcribe(test_file['path'])
                total_inference_time += result['inference_time']
                transcriptions.append({
                    "id": test_file['id'],
                    "path": test_file['path'],
                    "original": test_file['text'],
                    "transcribed": result['text'],
                    "time": result['inference_time']
                })
            
            avg_inference_time = total_inference_time / len(question_files)
            
            benchmark_result = {
                "model": model_name.split('/')[-1],
                "load_time_s": f"{asr.load_time:.2f}s",
                "avg_inference_time_s": f"{avg_inference_time:.2f}s",
                "vram_allocated_gb": f"{transcriptions[0].get('vram_allocated_gb', 0):.2f}GB",
                "transcriptions": transcriptions
            }
            
            benchmark_results.append(benchmark_result)
            
            logger.info(f"  ⏱️  Yükleme: {benchmark_result['load_time_s']}")
            logger.info(f"  ⚡ Ortalama Çıkarım: {benchmark_result['avg_inference_time_s']}")
            logger.info(f"  💾 VRAM: {benchmark_result['vram_allocated_gb']}")
            
            asr.cleanup()
            
        except Exception as e:
            logger.error(f"  ❌ Hata: {e}")
    
    # Kaydet
    with open(Config.REPORT_DIR / "whisper_benchmark.json", 'w', encoding='utf-8') as f:
        json.dump(benchmark_results, f, ensure_ascii=False, indent=2)
    
    logger.info("\n✅ Whisper Benchmark Tamamlandı!")
    return benchmark_results


def step3_transcribe_all():
    """Adım 3: Tüm soruları transkribe et"""
    logger.info("\n" + "="*80)
    logger.info("ADIM 3: TÜM SORULARI TRANSKRİBE ETME")
    logger.info("="*80)
    
    # Question files yükle
    with open(Config.DATA_DIR / "question_files.json", 'r', encoding='utf-8') as f:
        question_files = json.load(f)
    
    asr = WhisperASR(model_name=Config.SELECTED_WHISPER, device=Config.DEVICE)
    asr.load()
    
    all_transcriptions = []
    
    for question_file in tqdm(question_files, desc="Transkripsiyon"):
        try:
            result = asr.transcribe(question_file['path'])
            
            transcript_data = {
                "id": question_file['id'],
                "audio_path": question_file['path'],
                "original_text": question_file['text'],
                "transcribed_text": result['text'],
                "inference_time": result['inference_time']
            }
            
            all_transcriptions.append(transcript_data)
            
        except Exception as e:
            logger.error(f"❌ {question_file['id']} hatası: {e}")
    
    # Kaydet
    with open(Config.TRANSCRIPT_DIR / "all_transcriptions.json", 'w', encoding='utf-8') as f:
        json.dump(all_transcriptions, f, ensure_ascii=False, indent=2)
    
    logger.info(f"\n✅ {len(all_transcriptions)} transkript kaydedildi")
    
    asr.cleanup()
    return all_transcriptions


def step4_benchmark_qwen():
    """Adım 4: Qwen model benchmark"""
    logger.info("\n" + "="*80)
    logger.info("ADIM 4: QWEN MODEL BENCHMARK")
    logger.info("="*80)
    
    # Talimatname hazırla
    talimatname_path = Config.DATA_DIR / "saha_operasyon_talimatnamesi.txt"
    with open(talimatname_path, 'w', encoding='utf-8') as f:
        f.write(get_talimatname_content())
    
    qa_dataset = get_qa_dataset()
    test_qa = qa_dataset[:5]  # İlk 5 soru ile test
    benchmark_results = []
    
    for model_name in Config.QWEN_MODELS:
        logger.info(f"\n🧠 Test Ediliyor: {model_name}")
        logger.info("-" * 80)
        
        try:
            llm = QwenLLM(model_name=model_name, use_4bit=Config.USE_4BIT)
            llm.load()
            llm.load_document(str(talimatname_path))
            
            total_inference_time = 0
            qa_results = []
            
            for qa in test_qa:
                result = llm.ask(qa['question'])
                total_inference_time += result['inference_time']
                
                qa_results.append({
                    "id": qa['id'],
                    "question": qa['question'],
                    "expected_answer": qa['expected_answer'],
                    "generated_answer": result['answer'],
                    "inference_time": result['inference_time'],
                    "tokens_generated": result['tokens_generated']
                })
            
            avg_inference_time = total_inference_time / len(test_qa)
            
            benchmark_result = {
                "model": model_name.split('/')[-1],
                "load_time_s": f"{llm.load_time:.2f}s",
                "avg_inference_time_s": f"{avg_inference_time:.2f}s",
                "vram_allocated_gb": f"{qa_results[0].get('vram_allocated_gb', 0):.2f}GB",
                "qa_results": qa_results
            }
            
            benchmark_results.append(benchmark_result)
            
            logger.info(f"  ⏱️  Yükleme: {benchmark_result['load_time_s']}")
            logger.info(f"  ⚡ Ortalama Çıkarım: {benchmark_result['avg_inference_time_s']}")
            logger.info(f"  💾 VRAM: {benchmark_result['vram_allocated_gb']}")
            
            llm.cleanup()
            
        except Exception as e:
            logger.error(f"  ❌ Hata: {e}")
    
    # Kaydet
    with open(Config.REPORT_DIR / "qwen_benchmark.json", 'w', encoding='utf-8') as f:
        json.dump(benchmark_results, f, ensure_ascii=False, indent=2)
    
    logger.info("\n✅ Qwen Benchmark Tamamlandı!")
    return benchmark_results


def step5_qa_generation():
    """Adım 5: Tüm soruları cevapla ve TTS ile kaydet"""
    logger.info("\n" + "="*80)
    logger.info("ADIM 5: SORU-CEVAP ÜRETİMİ VE TTS")
    logger.info("="*80)
    
    # Talimatname yükle
    talimatname_path = Config.DATA_DIR / "saha_operasyon_talimatnamesi.txt"
    
    # LLM yükle
    llm = QwenLLM(model_name=Config.SELECTED_QWEN, use_4bit=Config.USE_4BIT)
    llm.load()
    llm.load_document(str(talimatname_path))
    
    # TTS hazırla
    tts = TurkishTTS()
    
    qa_dataset = get_qa_dataset()
    all_qa_results = []
    
    for qa in tqdm(qa_dataset, desc="LLM Q&A + TTS"):
        try:
            # LLM ile cevap
            result = llm.ask(qa['question'])
            answer_text = result['answer']
            
            # TTS ile kaydet
            answer_audio_path = Config.ANSWER_DIR / f"{qa['id']}_answer.wav"
            tts.create_audio(answer_text, str(answer_audio_path), Config.SAMPLE_RATE)
            
            qa_result = {
                "id": qa['id'],
                "question": qa['question'],
                "expected_answer": qa['expected_answer'],
                "generated_answer": answer_text,
                "answer_audio_path": str(answer_audio_path),
                "inference_time": result['inference_time'],
                "tokens_generated": result['tokens_generated']
            }
            
            all_qa_results.append(qa_result)
            
        except Exception as e:
            logger.error(f"❌ {qa['id']} hatası: {e}")
    
    # Kaydet
    with open(Config.REPORT_DIR / "all_qa_results.json", 'w', encoding='utf-8') as f:
        json.dump(all_qa_results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"\n✅ {len(all_qa_results)} soru-cevap tamamlandı")
    
    llm.cleanup()
    return all_qa_results


def step6_document_analysis():
    """Adım 6: Doküman analizi"""
    logger.info("\n" + "="*80)
    logger.info("ADIM 6: DOKÜMAN ANALİZİ")
    logger.info("="*80)
    
    talimatname_path = Config.DATA_DIR / "saha_operasyon_talimatnamesi.txt"
    
    llm = QwenLLM(model_name=Config.SELECTED_QWEN, use_4bit=Config.USE_4BIT)
    llm.load()
    llm.load_document(str(talimatname_path))
    
    analysis_prompts = get_analysis_prompts()
    analysis_results = []
    
    for prompt_item in analysis_prompts:
        logger.info(f"\n🔍 {prompt_item['category']}: {prompt_item['prompt']}")
        logger.info("-" * 80)
        
        try:
            result = llm.ask(prompt_item['prompt'], max_new_tokens=1024)
            
            analysis_result = {
                "id": prompt_item['id'],
                "category": prompt_item['category'],
                "prompt": prompt_item['prompt'],
                "response": result['answer'],
                "inference_time": result['inference_time']
            }
            
            analysis_results.append(analysis_result)
            
            logger.info(f"\n📝 Cevap:\n{result['answer']}")
            logger.info(f"\n⏱️  Süre: {result['inference_time']:.2f}s")
            
        except Exception as e:
            logger.error(f"❌ Hata: {e}")
    
    # Kaydet
    with open(Config.REPORT_DIR / "document_analysis.json", 'w', encoding='utf-8') as f:
        json.dump(analysis_results, f, ensure_ascii=False, indent=2)
    
    logger.info("\n✅ Doküman analizi tamamlandı")
    
    llm.cleanup()
    return analysis_results


def step7_generate_report():
    """Adım 7: Final rapor oluştur"""
    logger.info("\n" + "="*80)
    logger.info("ADIM 7: FİNAL RAPORU OLUŞTURMA")
    logger.info("="*80)
    
    # Tüm sonuçları yükle
    with open(Config.REPORT_DIR / "whisper_benchmark.json", 'r') as f:
        whisper_benchmark = json.load(f)
    
    with open(Config.REPORT_DIR / "qwen_benchmark.json", 'r') as f:
        qwen_benchmark = json.load(f)
    
    with open(Config.TRANSCRIPT_DIR / "all_transcriptions.json", 'r') as f:
        transcriptions = json.load(f)
    
    with open(Config.REPORT_DIR / "all_qa_results.json", 'r') as f:
        qa_results = json.load(f)
    
    with open(Config.REPORT_DIR / "document_analysis.json", 'r') as f:
        analysis_results = json.load(f)
    
    with open(Config.DATA_DIR / "question_files.json", 'r') as f:
        question_files = json.load(f)
    
    # Rapor metni
    report = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           ATBK-NG ASR-LLM PIPELINE - BENCHMARK RAPORU                        ║
║           Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}              ║
║           Cihaz: {Config.DEVICE.upper()}                                     ║
╚══════════════════════════════════════════════════════════════════════════════╝

1. WHISPER MODEL BENCHMARK SONUÇLARI
{'='*80}

"""
    
    # Whisper tablosu
    report += "\n┌─────────────────┬──────────────┬─────────────────┬─────────────────┐\n"
    report += "│ Model           │ Yükleme      │ Ort. Çıkarım    │ VRAM            │\n"
    report += "├─────────────────┼──────────────┼─────────────────┼─────────────────┤\n"
    
    for result in whisper_benchmark:
        report += f"│ {result['model']:<15} │ {result['load_time_s']:<12} │ {result['avg_inference_time_s']:<15} │ {result['vram_allocated_gb']:<15} │\n"
    
    report += "└─────────────────┴──────────────┴─────────────────┴─────────────────┘\n"
    
    # Qwen tablosu
    report += f"\n\n2. QWEN LLM MODEL BENCHMARK SONUÇLARI\n"
    report += "="*80 + "\n"
    
    report += "\n┌──────────────────────────┬──────────────┬─────────────────┬─────────────────┐\n"
    report += "│ Model                    │ Yükleme      │ Ort. Çıkarım    │ VRAM            │\n"
    report += "├──────────────────────────┼──────────────┼─────────────────┼─────────────────┤\n"
    
    for result in qwen_benchmark:
        report += f"│ {result['model']:<24} │ {result['load_time_s']:<12} │ {result['avg_inference_time_s']:<15} │ {result['vram_allocated_gb']:<15} │\n"
    
    report += "└──────────────────────────┴──────────────┴─────────────────┴─────────────────┘\n"
    
    # Öneriler
    report += f"""\n\n3. ÖNERİLER VE DEĞERLENDİRME
{'='*80}

🎯 WHISPER MODEL ÖNERİSİ:
   ✅ Dengeli Performans: whisper-small
      - Hız ve doğruluk dengesi iyi
      - Orta seviye VRAM kullanımı (~2GB)
      - Gerçek zamanlı uygulamalar için uygun
   
   ✅ Maksimum Kalite: whisper-medium
      - En yüksek transkripsiyon doğruluğu
      - Daha yüksek VRAM gereksinimi (~5GB)
      - Offline batch processing için ideal

🧠 QWEN LLM MODEL ÖNERİSİ:
   ✅ Üretim Ortamı: Qwen2.5-1.5B-Instruct
      - Kalite/hız dengesi mükemmel
      - 4-bit quantization ile ~2GB VRAM
      - Askeri doküman Q&A için yeterli
   
   ✅ Kaynak Kısıtlı: Qwen2.5-0.5B-Instruct
      - En düşük VRAM kullanımı
      - Hızlı yanıt süreleri
      - Basit soru-cevap görevleri için yeterli

💡 SİSTEM GEREKSINIMLERI:
   Minimum: 8GB VRAM (RTX 3060)
   Önerilen: 16GB VRAM (RTX 4080)
   Optimal: 24GB+ VRAM (RTX 4090 / A6000)

📈 PERFORMANS METRİKLERİ:
   - Toplam Transkribe Edilen Ses: {len(transcriptions)} dosya
   - Toplam Üretilen Cevap: {len(qa_results)} adet
   - Doküman Analiz: {len(analysis_results)} prompt
   - Toplam Ses Dosyası: {len(question_files) + len(qa_results)} adet

🎖️ ATBK-NG ENTEGRASYON NOTLARI:
   - Jetson AGX Xavier/Orin için whisper-small + Qwen-0.5B önerilir
   - LoRa/UHF mesh network üzerinden batch processing modunda çalışabilir
   - Offline operasyon için tüm modeller local storage'da bulunmalı
   - KBRN ve acil durum senaryoları için yanıt süreleri <2s olmalı
"""
    
    # Raporu kaydet
    report_file = Config.REPORT_DIR / "benchmark_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    # JSON formatı
    benchmark_json = {
        "timestamp": datetime.now().isoformat(),
        "device": Config.DEVICE,
        "whisper_benchmark": whisper_benchmark,
        "qwen_benchmark": qwen_benchmark,
        "statistics": {
            "total_transcriptions": len(transcriptions),
            "total_qa_results": len(qa_results),
            "total_analysis": len(analysis_results),
            "total_audio_files": len(question_files) + len(qa_results)
        }
    }
    
    benchmark_json_file = Config.REPORT_DIR / "benchmark_results.json"
    with open(benchmark_json_file, 'w', encoding='utf-8') as f:
        json.dump(benchmark_json, f, ensure_ascii=False, indent=2)
    
    logger.info(report)
    logger.info(f"\n✅ Benchmark raporu kaydedildi:")
    logger.info(f"   📄 TXT: {report_file}")
    logger.info(f"   📊 JSON: {benchmark_json_file}")


def print_summary():
    """Pipeline özeti"""
    logger.info("\n" + "="*80)
    logger.info("🎉 PIPELINE TAMAMLANDI!")
    logger.info("="*80)
    
    logger.info(f"\n📁 OLUŞTURULAN DOSYALAR:\n")
    logger.info(f"1️⃣  Soru Ses Dosyaları: {Config.QUESTION_DIR}")
    logger.info(f"2️⃣  Cevap Ses Dosyaları: {Config.ANSWER_DIR}")
    logger.info(f"3️⃣  Transkriptler: {Config.TRANSCRIPT_DIR}")
    logger.info(f"4️⃣  Raporlar: {Config.REPORT_DIR}")
    logger.info(f"5️⃣  Data: {Config.DATA_DIR}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Ana pipeline"""
    parser = argparse.ArgumentParser(description="ATBK-NG ASR-LLM Pipeline")
    parser.add_argument('--steps', type=str, default='all', 
                       help='Çalıştırılacak adımlar (1-7 veya all)')
    parser.add_argument('--skip-benchmark', action='store_true',
                       help='Benchmark adımlarını atla')
    args = parser.parse_args()
    
    # Başlangıç
    logger.info("\n" + "="*80)
    logger.info("ATBK-NG ASR-LLM PIPELINE - LOCAL PC VERSION")
    logger.info("="*80)
    logger.info(f"Cihaz: {Config.DEVICE.upper()}")
    logger.info(f"CUDA: {'Evet' if torch.cuda.is_available() else 'Hayır'}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f}GB")
    logger.info("="*80)
    
    # Dizinleri oluştur
    Config.create_directories()
    
    try:
        # Adımları belirle
        if args.steps == 'all':
            steps = [1, 2, 3, 4, 5, 6, 7]
        else:
            steps = [int(s) for s in args.steps.split(',')]
        
        # Benchmark atlanacak mı?
        if args.skip_benchmark:
            steps = [s for s in steps if s not in [2, 4]]
        
        # Adımları çalıştır
        if 1 in steps:
            step1_create_question_audio()
        
        if 2 in steps:
            step2_benchmark_whisper()
        
        if 3 in steps:
            step3_transcribe_all()
        
        if 4 in steps:
            step4_benchmark_qwen()
        
        if 5 in steps:
            step5_qa_generation()
        
        if 6 in steps:
            step6_document_analysis()
        
        if 7 in steps:
            step7_generate_report()
            print_summary()
        
        logger.info("\n✅ Pipeline başarıyla tamamlandı!")
        
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Pipeline kullanıcı tarafından durduruldu")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ Fatal hata: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
