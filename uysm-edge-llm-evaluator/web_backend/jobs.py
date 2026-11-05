"""Background jobs with polling (indexing, CSV pipeline)."""

from __future__ import annotations

import logging
import threading
import traceback
import uuid
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str, int, int, float], None]


class JobStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: Dict[str, Dict[str, Any]] = {}

    def create(self) -> str:
        jid = str(uuid.uuid4())
        with self._lock:
            self._jobs[jid] = {
                "status": "pending",
                "progress": None,
                "result": None,
                "error": None,
            }
        return jid

    def _update(self, jid: str, **kwargs: Any) -> None:
        with self._lock:
            if jid in self._jobs:
                self._jobs[jid].update(kwargs)

    def set_running(self, jid: str) -> None:
        self._update(jid, status="running")

    def set_progress(self, jid: str, phase: str, current: int, total: int, elapsed: float) -> None:
        self._update(
            jid,
            progress={
                "phase": phase,
                "current": current,
                "total": total,
                "elapsed_sec": elapsed,
            },
        )

    def complete(self, jid: str, result: Any) -> None:
        self._update(jid, status="completed", result=result, progress=None)

    def fail(self, jid: str, error: str) -> None:
        self._update(jid, status="failed", error=error, progress=None)

    def get(self, jid: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return dict(self._jobs[jid]) if jid in self._jobs else None

    def run_async(self, jid: str, fn: Callable[[ProgressCallback], Any]) -> None:
        store = self

        def progress(phase: str, current: int, total: int, elapsed: float) -> None:
            store.set_progress(jid, phase, current, total, elapsed)

        def runner() -> None:
            store.set_running(jid)
            try:
                result = fn(progress)
                store.complete(jid, result)
            except Exception as e:
                logger.error("Job %s failed: %s\n%s", jid, e, traceback.format_exc())
                store.fail(jid, str(e))

        threading.Thread(target=runner, daemon=True).start()


job_store = JobStore()
