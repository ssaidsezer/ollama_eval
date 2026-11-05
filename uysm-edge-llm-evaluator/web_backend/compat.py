"""Shared helpers for Ollama/Qdrant HTTP calls and collection naming (HTTP + pure logic)."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import List, Tuple

import requests
from dotenv import load_dotenv

from rag.config import DEFAULT_COLLECTION_NAME, QDRANT_URL

WORKSPACE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(WORKSPACE_DIR / ".env", override=True)

OLLAMA_ENV_MISSING_MSG = (
    "OLLAMA_BASE_URL ortam değişkeni tanımlı değil."
)


def _qdrant_base_url() -> str:
    return os.environ.get("QDRANT_URL", QDRANT_URL)


def get_ollama_base_url() -> str:
    base_url = os.environ.get("OLLAMA_BASE_URL", "").strip()
    if not base_url:
        return ""
    if not base_url.startswith("http"):
        base_url = f"http://{base_url}"
    return base_url.rstrip("/")


def get_monitor_base_url() -> str:
    monitor_url = os.environ.get("MONITOR_URL", "").strip()
    if not monitor_url:
        return ""
    if not monitor_url.startswith("http"):
        monitor_url = f"http://{monitor_url}"
    return monitor_url.rstrip("/")


def collection_name_full(
    base: str, embed_model: str, chunk_size: int, chunk_overlap: int
) -> str:
    safe = embed_model.replace(":", "_").replace("/", "_").replace(".", "_")
    return f"{base}_{safe}_{chunk_size}c_{chunk_overlap}ov"


def smart_collection_name_full(
    base: str,
    embed_model: str,
    parent_size: int,
    child_size: int,
    child_overlap: int,
) -> str:
    safe = embed_model.replace(":", "_").replace("/", "_").replace(".", "_")
    return f"{base}_{safe}_{parent_size}p_{child_size}c_{child_overlap}ov"


def collection_options_for_embed(
    collections: List[str],
    embed_model: str | None,
) -> tuple[List[str], List[str]]:
    if not embed_model:
        return [], []
    safe_embed = embed_model.replace(":", "_").replace("/", "_").replace(".", "_")
    classic_cols = sorted(
        c
        for c in collections
        if safe_embed in c and not c.endswith("_children") and not c.endswith("_parents")
    )
    smart_bases = sorted(
        {
            c[: -len("_children")]
            for c in collections
            if c.endswith("_children")
            and safe_embed in c
            and f"{c[: -len('_children')]}_parents" in collections
        }
    )
    return classic_cols, smart_bases


def is_embedding_model(host: str, model_name: str) -> bool:
    resp = requests.post(
        host.rstrip("/") + "/api/show",
        json={"name": model_name},
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json() or {}

    capabilities = data.get("capabilities") or []
    if capabilities:
        return "embedding" in capabilities

    name_lower = model_name.lower()
    if any(kw in name_lower for kw in ("embed", "bge", "e5", "gte", "rerank")):
        return True

    template = (data.get("template") or "").strip()
    if not template:
        return True

    return False


def list_ollama_models() -> tuple[List[str], str, float, int]:
    host = get_ollama_base_url()
    if not host:
        return [], (
            "OLLAMA_BASE_URL ortam değişkeni tanımlı değil. "
            "Lütfen .env dosyasına uzak sunucu adresini ekleyin "
            "(örn: OLLAMA_BASE_URL=http://192.168.1.151:11434)."
        ), 0.0, 0

    url = host + "/api/tags"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json() or {}
    except (requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.HTTPError) as e:
        return [], (
            f"Uzak Ollama sunucusuna ({host}) bağlanılamadı: {e}"
        ), 0.0, 0

    all_names: List[str] = []
    for item in data.get("models", []):
        name = item.get("name")
        if isinstance(name, str):
            all_names.append(name)

    t0 = time.time()
    models: List[str] = []
    filtered_count = 0
    for name in all_names:
        if is_embedding_model(host, name):
            filtered_count += 1
        else:
            models.append(name)
    elapsed = round(time.time() - t0, 2)

    return models, "", elapsed, filtered_count


def list_embedding_models() -> tuple[List[str], str]:
    host = get_ollama_base_url()
    if not host:
        return [], OLLAMA_ENV_MISSING_MSG

    url = host + "/api/tags"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json() or {}
    except (requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.HTTPError) as e:
        return [], f"Uzak Ollama sunucusuna ({host}) bağlanılamadı: {e}"

    all_names: List[str] = [
        item.get("name")
        for item in data.get("models", [])
        if isinstance(item.get("name"), str)
    ]

    embed_models = [name for name in all_names if is_embedding_model(host, name)]
    return embed_models, ""


def ensure_tmp_dir() -> Path:
    tmp_dir = WORKSPACE_DIR / "tmp"
    tmp_dir.mkdir(exist_ok=True)
    return tmp_dir


def pull_ollama_model(model_name: str) -> tuple[bool, str]:
    host = get_ollama_base_url()
    if not host:
        return False, OLLAMA_ENV_MISSING_MSG
    try:
        resp = requests.post(
            host + "/api/pull",
            json={"name": model_name, "stream": False},
            timeout=300,
        )
        resp.raise_for_status()
        return True, f"'{model_name}' başarıyla pull edildi."
    except Exception as e:
        return False, f"Pull sırasında hata: {e}"


def delete_ollama_model(model_name: str) -> tuple[bool, str]:
    host = get_ollama_base_url()
    if not host:
        return False, OLLAMA_ENV_MISSING_MSG
    try:
        resp = requests.delete(host + "/api/delete", json={"name": model_name}, timeout=30)
        resp.raise_for_status()
        return True, f"'{model_name}' başarıyla silindi."
    except Exception as e:
        return False, f"Silme sırasında hata: {e}"


def list_qdrant_collections() -> tuple[List[str], str]:
    qdrant_url = _qdrant_base_url()
    try:
        resp = requests.get(f"{qdrant_url}/collections", timeout=10)
        resp.raise_for_status()
        data = resp.json() or {}
        collections = [c["name"] for c in data.get("result", {}).get("collections", [])]
        return collections, ""
    except (requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.HTTPError) as e:
        return [], f"Qdrant koleksiyonları listelenemedi: {e}"


def delete_qdrant_collection(collection_name: str) -> tuple[bool, str]:
    qdrant_url = _qdrant_base_url()
    try:
        resp = requests.delete(f"{qdrant_url}/collections/{collection_name}", timeout=30)
        resp.raise_for_status()
        return True, f"'{collection_name}' koleksiyonu başarıyla silindi."
    except (requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.HTTPError) as e:
        return False, f"Koleksiyon silme sırasında hata: {e}"


def list_all_ollama_model_names_raw() -> tuple[List[str], str]:
    host = get_ollama_base_url()
    if not host:
        return [], "OLLAMA_BASE_URL tanımlı değil."
    try:
        resp = requests.get(host + "/api/tags", timeout=10)
        resp.raise_for_status()
        names = [
            item["name"]
            for item in (resp.json() or {}).get("models", [])
            if isinstance(item.get("name"), str)
        ]
        return names, ""
    except (requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.HTTPError) as e:
        return [], str(e)


def get_monitor_stats() -> tuple[dict | None, str]:
    monitor_url = get_monitor_base_url()
    if not monitor_url:
        return None, "MONITOR_URL tanımlı değil."

    try:
        resp = requests.get(f"{monitor_url}/stats", timeout=3)
        if resp.status_code == 200:
            raw = resp.json() or {}
            return _normalize_monitor_stats(raw), ""
        return None, "Sistem bilgisi alınamadı"
    except Exception:
        return None, f"Monitor erişilemiyor ({monitor_url})"


def _get_num(payload: dict, *keys: str) -> float | int | None:
    for key in keys:
        val = payload.get(key)
        if val is None:
            continue
        if isinstance(val, bool):
            continue
        if isinstance(val, (int, float)):
            return val
        if isinstance(val, str):
            s = val.strip().replace(",", ".")
            if not s:
                continue
            try:
                return float(s)
            except ValueError:
                continue
    return None


def _normalize_monitor_stats(raw: dict) -> dict:
    cpu_usage = _get_num(raw, "cpu_usage", "cpu_percent", "cpu")
    gpu_usage = _get_num(raw, "gpu_usage", "gpu_percent", "gpu")
    vram_used = _get_num(raw, "vram_used", "gpu_memory_used", "gpu_vram_used")
    vram_total = _get_num(raw, "vram_total", "gpu_memory_total", "gpu_vram_total")
    ram_used = _get_num(raw, "ram_used", "memory_used", "mem_used")
    ram_total = _get_num(raw, "ram_total", "memory_total", "mem_total")
    storage_used = _get_num(raw, "storage_used", "disk_used")
    storage_total = _get_num(raw, "storage_total", "disk_total")
    disk_free = _get_num(raw, "disk_free", "storage_free")

    if storage_used is None and storage_total is not None and disk_free is not None:
        storage_used = float(storage_total) - float(disk_free)

    if storage_used is not None:
        storage_used = float(storage_used) * 1024.0
    if storage_total is not None:
        storage_total = float(storage_total) * 1024.0

    return {
        "cpu_usage": cpu_usage if cpu_usage is not None else 0.0,
        "gpu_usage": gpu_usage if gpu_usage is not None else 0.0,
        "vram_used": vram_used,
        "vram_total": vram_total,
        "ram_used": ram_used,
        "ram_total": ram_total,
        "storage_used": storage_used,
        "storage_total": storage_total,
    }
