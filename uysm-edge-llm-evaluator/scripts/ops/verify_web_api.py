#!/usr/bin/env python3
"""Smoke-test FastAPI after `python -m uvicorn web_backend.main:app --port 8000`."""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request


def main() -> int:
    base = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000").rstrip("/")
    paths = (
        "/api/health",
        "/api/config",
        "/api/models/ollama",
        "/api/models/embeddings",
        "/api/model-profiles",
        "/api/model-profiles/default-templates",
    )
    for path in paths:
        url = f"{base}{path}"
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                body = resp.read().decode("utf-8", "replace")
                print(f"OK {path} -> {resp.status}")
                try:
                    obj = json.loads(body)
                    preview = json.dumps(obj, ensure_ascii=False)[:180]
                    print(f"   {preview}...")
                except json.JSONDecodeError:
                    print(f"   {body[:180]}")
        except urllib.error.URLError as e:
            print(f"FAIL {path}: {e}", file=sys.stderr)
            return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
