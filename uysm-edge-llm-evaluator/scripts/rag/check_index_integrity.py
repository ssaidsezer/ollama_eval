#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rag.config import (  # noqa: E402
    OLLAMA_BASE_URL,
    OLLAMA_EMBED_MODEL,
    QDRANT_URL,
    SMART_BOUNDARY_LLM_MODEL,
)
from rag.qdrant_store import get_qdrant_client, inspect_parent_child_integrity  # noqa: E402
from rag.smart_indexing import index_pdfs_smart  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check smart RAG parent-child referential integrity."
    )
    parser.add_argument("--base-collection", required=True)
    parser.add_argument("--source", help="Limit the read-only check to one source payload value.")
    parser.add_argument("--json", action="store_true", help="Print compact JSON.")
    parser.add_argument(
        "--rebuild-source",
        type=Path,
        help="Mutating operation: rebuild exactly this PDF source before checking it.",
    )
    parser.add_argument("--qdrant-url", default=QDRANT_URL)
    parser.add_argument("--ollama-base-url", default=OLLAMA_BASE_URL)
    parser.add_argument("--embed-model", default=OLLAMA_EMBED_MODEL)
    parser.add_argument("--boundary-model", default=SMART_BOUNDARY_LLM_MODEL)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.source
    if args.rebuild_source is not None:
        source_path = args.rebuild_source.resolve()
        if not source_path.is_file():
            raise SystemExit(f"Rebuild source does not exist: {source_path}")
        index_pdfs_smart(
            [str(source_path)],
            base_collection=args.base_collection,
            boundary_llm_model=args.boundary_model,
            embed_model=args.embed_model,
            qdrant_url=args.qdrant_url,
            ollama_base_url=args.ollama_base_url,
        )
        source = str(source_path)

    client = get_qdrant_client(args.qdrant_url)
    report = inspect_parent_child_integrity(
        client,
        args.base_collection,
        source=source,
    )
    if args.json:
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
