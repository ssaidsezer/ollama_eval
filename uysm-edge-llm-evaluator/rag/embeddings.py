from __future__ import annotations

from typing import List

import requests

from .config import OLLAMA_BASE_URL, OLLAMA_EMBED_MODEL


def get_embeddings(
    texts: List[str],
    model: str = OLLAMA_EMBED_MODEL,
    base_url: str = OLLAMA_BASE_URL,
    timeout: int = 1200,
) -> List[List[float]]:
    """
    Uzak Ollama sunucusundan embedding al.
    Önce yeni /api/embed, 404 alırsa eski /api/embeddings endpoint'ini dener.
    """
    if not model or not str(model).strip():
        raise ValueError("Embedding modeli seçilmedi.")

    if not base_url:
        raise ValueError(
            "OLLAMA_BASE_URL ortam değişkeni tanımlı değil. "
            "Lütfen .env dosyasına uzak sunucu adresini ekleyin "
            "(örn: OLLAMA_BASE_URL=http://192.168.1.151:11434)."
        )

    base = base_url.rstrip("/")

    resp = requests.post(
        f"{base}/api/embed",
        json={"model": model, "input": texts},
        timeout=timeout,
    )
    if resp.status_code != 404:
        resp.raise_for_status()
        try:
            data = resp.json()
        except ValueError as e:
            raise RuntimeError(
                f"Ollama /api/embed JSON dışı yanıt döndü. "
                f"URL={base}/api/embed, Status={resp.status_code}, "
                f"Content-Type={resp.headers.get('Content-Type')!r}, "
                f"Body={resp.text[:500]!r}, Model={model}"
            ) from e
        embeddings = data.get("embeddings")
        if not embeddings:
            raise RuntimeError(
                f"Ollama /api/embed 'embeddings' alanı boş. Model: {model}, Yanıt: {data}"
            )
        return embeddings

    all_embeddings: List[List[float]] = []
    for text in texts:
        resp = requests.post(
            f"{base}/api/embeddings",
            json={"model": model, "prompt": text},
            timeout=timeout,
        )
        resp.raise_for_status()
        try:
            data = resp.json()
        except ValueError as e:
            raise RuntimeError(
                f"Ollama /api/embeddings JSON dışı yanıt döndü. "
                f"URL={base}/api/embeddings, Status={resp.status_code}, "
                f"Content-Type={resp.headers.get('Content-Type')!r}, "
                f"Body={resp.text[:500]!r}, Model={model}"
            ) from e
        embedding = data.get("embedding")
        if not embedding:
            raise RuntimeError(
                f"Ollama embedding yanıtında 'embedding' alanı bulunamadı. "
                f"Model: {model}, Yanıt: {data}"
            )
        all_embeddings.append(embedding)

    return all_embeddings
