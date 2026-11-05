"""STT ve TTS için uzak FastAPI endpoint'lerine HTTP istemcisi.

Her iki model de uzak sunucuda (Docker container) çalışır.
Bu modül sadece ses byte'larını /stt'ye, metni /tts'ye POST eder.

Yapılandırma:
    VOICE_API_URL  — FastAPI sunucusunun adresi (varsayılan: http://localhost:8000)
"""

from __future__ import annotations

import os
from typing import List, Tuple

import requests

VOICE_API_URL = os.environ.get("VOICE_API_URL", "").strip()

TTS_MODEL = os.environ.get("TTS_MODEL")

def get_downloaded_tts_models() -> List[str]:
    """Uzak sunucuda hâlihazırda indirilmiş ve hazır olan HuggingFace TTS modellerini listeler."""
    try:
        url = f"{VOICE_API_URL.rstrip('/')}/models"
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        return data.get("downloaded_models", [])
    except Exception:
        return []

# MMS-TTS-TUR her zaman 16 kHz çıktı üretir
_TTS_SAMPLE_RATE = 16_000


def synthesize_speech(
    text: str, 
    model: str = TTS_MODEL, 
    speaker_id: str | None = None, 
    voice_preset: str | None = None
) -> Tuple[bytes, int, float]:
    """Metni uzak modele WAV'a dönüştürür. Yeni parametreler (speaker_id, preset) eklendi."""
    url = f"{VOICE_API_URL.rstrip('/')}/tts"
    payload = {
        "text": text, 
        "model": model,
        "speaker_id": speaker_id,
        "voice_preset": voice_preset
    }
    resp = requests.post(url, json=payload, timeout=180)
    resp.raise_for_status()
    duration_sec = float(resp.headers.get("X-Audio-Duration", "0.0"))
    return resp.content, _TTS_SAMPLE_RATE, duration_sec

