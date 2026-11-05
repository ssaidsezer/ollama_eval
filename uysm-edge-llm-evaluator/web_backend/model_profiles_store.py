"""Persistent model profile store (JSON file + POSIX file lock)."""

from __future__ import annotations

import json
import os
import portalocker
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

DATA_DIR = Path(__file__).resolve().parent / "data"
PROFILES_FILE = DATA_DIR / "model_profiles.json"
LOCK_FILE = DATA_DIR / "model_profiles.json.lock"

MAX_PROMPT_CHARS = 100_000

_ROOT = {"version": 1, "profiles": []}


def _ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _with_lock_write(mutator: Any) -> Any:
    """Run mutator(state: dict) -> Any under exclusive lock; mutator may modify state in place."""
    _ensure_data_dir()
    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOCK_FILE, "a+", encoding="utf-8") as lock_fp:
        portalocker.lock(lock_fp, portalocker.LOCK_EX)
        try:
            if PROFILES_FILE.is_file():
                raw = PROFILES_FILE.read_text(encoding="utf-8").strip()
                state = json.loads(raw) if raw else dict(_ROOT)
            else:
                state = dict(_ROOT)
            if "profiles" not in state:
                state["profiles"] = []
            out = mutator(state)
            _atomic_write(PROFILES_FILE, json.dumps(state, ensure_ascii=False, indent=2))
            return out
        finally:
            portalocker.unlock(lock_fp)


def _with_lock_read(reader: Any) -> Any:
    _ensure_data_dir()
    with open(LOCK_FILE, "a+", encoding="utf-8") as lock_fp:
        portalocker.lock(lock_fp, portalocker.LOCK_SH)
        try:
            if not PROFILES_FILE.is_file():
                return reader(dict(_ROOT))
            raw = PROFILES_FILE.read_text(encoding="utf-8").strip()
            state = json.loads(raw) if raw else dict(_ROOT)
            return reader(state)
        finally:
            portalocker.unlock(lock_fp)


def _validate_params(params: Dict[str, Any]) -> None:
    allowed_keys = {
        "num_ctx",
        "temperature",
        "top_p",
        "repeat_penalty",
        "num_predict",
        "think",
        "speaker_id",
        "voice_preset",
        "tts_model",
    }
    for k in params:
        if k not in allowed_keys:
            raise ValueError(f"Geçersiz parametre anahtarı: {k}")
    if "num_ctx" in params and params["num_ctx"] is not None:
        v = int(params["num_ctx"])
        if v < 256 or v > 1_000_000:
            raise ValueError("num_ctx 256 ile 1000000 arasında olmalı.")
    if "temperature" in params and params["temperature"] is not None:
        t = float(params["temperature"])
        if not 0 <= t <= 2:
            raise ValueError("temperature 0-2 arası olmalı.")
    if "top_p" in params and params["top_p"] is not None:
        tp = float(params["top_p"])
        if not 0 < tp <= 1:
            raise ValueError("top_p (0,1] aralığında olmalı.")
    if "repeat_penalty" in params and params["repeat_penalty"] is not None:
        rp = float(params["repeat_penalty"])
        if rp < 0.5 or rp > 2.5:
            raise ValueError("repeat_penalty 0.5-2.5 arası önerilir.")
    if "num_predict" in params and params["num_predict"] is not None:
        np_ = int(params["num_predict"])
        if np_ < 1 or np_ > 1_000_000:
            raise ValueError("num_predict geçersiz.")
    if "think" in params and params["think"] is not None and not isinstance(params["think"], bool):
        raise ValueError("think boolean olmalı.")


def _validate_profile_dict(p: Dict[str, Any], *, existing_ids: Optional[set] = None) -> None:
    model_name = (p.get("model_name") or "").strip()
    if not model_name:
        raise ValueError("model_name gerekli.")
    name = (p.get("name") or "").strip()
    if not name:
        raise ValueError("Profil adı (name) gerekli.")
    sp = p.get("system_prompt")
    if sp is not None:
        if not isinstance(sp, str):
            raise ValueError("system_prompt string olmalı.")
        if len(sp) > MAX_PROMPT_CHARS:
            raise ValueError(f"system_prompt en fazla {MAX_PROMPT_CHARS} karakter.")
    params = p.get("params") or {}
    if not isinstance(params, dict):
        raise ValueError("params bir nesne olmalı.")
    _validate_params(params)
    pid = p.get("id")
    if existing_ids is not None and pid in existing_ids:
        raise ValueError(f"Yinelenen id: {pid}")


def list_profiles() -> List[Dict[str, Any]]:
    def reader(state: dict) -> List[Dict[str, Any]]:
        return list(state.get("profiles", []))

    return _with_lock_read(reader)


def get_profile(profile_id: str) -> Optional[Dict[str, Any]]:
    pid = (profile_id or "").strip()
    if not pid:
        return None

    for p in list_profiles():
        if p.get("id") == pid:
            return dict(p)
    return None


def get_default_profile(model_name: str) -> Optional[Dict[str, Any]]:
    m = (model_name or "").strip()
    if not m:
        return None
    for p in list_profiles():
        if p.get("model_name") == m and p.get("is_default"):
            return dict(p)
    return None


def _clear_other_defaults(state: dict, model_name: str, keep_id: str) -> None:
    for p in state["profiles"]:
        if (
            p.get("model_name") == model_name
            and p.get("is_default")
            and p.get("id") != keep_id
        ):
            p["is_default"] = False


def create_profile(
    *,
    name: str,
    model_name: str,
    system_prompt: Optional[str],
    params: Dict[str, Any],
    is_default: bool,
) -> Dict[str, Any]:
    params = dict(params or {})
    now = datetime.now(timezone.utc).isoformat()
    new_id = str(uuid.uuid4())

    def mutator(state: dict) -> Dict[str, Any]:
        prof = {
            "id": new_id,
            "name": name.strip(),
            "model_name": model_name.strip(),
            "system_prompt": system_prompt,
            "params": params,
            "is_default": bool(is_default),
            "version": 1,
            "updated_at": now,
        }
        ids = {p.get("id") for p in state["profiles"]}
        _validate_profile_dict(prof, existing_ids=ids)
        if is_default:
            _clear_other_defaults(state, model_name.strip(), new_id)
        state["profiles"].append(prof)
        return prof

    return _with_lock_write(mutator)


def update_profile(
    profile_id: str,
    *,
    name: Optional[str] = None,
    model_name: Optional[str] = None,
    system_prompt: Optional[str] = None,
    system_prompt_set_null: bool = False,
    params: Optional[Dict[str, Any]] = None,
    is_default: Optional[bool] = None,
) -> Optional[Dict[str, Any]]:
    pid = (profile_id or "").strip()
    if not pid:
        return None

    def mutator(state: dict) -> Optional[Dict[str, Any]]:
        found: Optional[Dict[str, Any]] = None
        for p in state["profiles"]:
            if p.get("id") == pid:
                found = p
                break
        if not found:
            return None
        if name is not None:
            found["name"] = name.strip()
        if model_name is not None:
            found["model_name"] = model_name.strip()
        if system_prompt_set_null:
            found["system_prompt"] = None
        elif system_prompt is not None:
            found["system_prompt"] = system_prompt
        if params is not None:
            found["params"] = dict(params)
        if is_default is not None:
            found["is_default"] = bool(is_default)
        found["version"] = int(found.get("version", 1)) + 1
        found["updated_at"] = datetime.now(timezone.utc).isoformat()

        merged = dict(found)
        _validate_profile_dict(merged, existing_ids=set())
        if found.get("is_default"):
            _clear_other_defaults(
                state,
                str(found["model_name"]),
                str(found["id"]),
            )
        return dict(found)

    return _with_lock_write(mutator)


def delete_profile(profile_id: str) -> bool:
    pid = (profile_id or "").strip()
    if not pid:
        return False

    def mutator(state: dict) -> bool:
        before = len(state["profiles"])
        state["profiles"] = [p for p in state["profiles"] if p.get("id") != pid]
        return len(state["profiles"]) < before

    return _with_lock_write(mutator)


def clone_profile(profile_id: str) -> Optional[Dict[str, Any]]:
    src = get_profile(profile_id)
    if not src:
        return None
    return create_profile(
        name=f"{src['name']} (kopya)",
        model_name=src["model_name"],
        system_prompt=src.get("system_prompt"),
        params=dict(src.get("params") or {}),
        is_default=False,
    )


def resolve_ollama_options_from_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Build Ollama /api/chat `options` dict from profile params (request overrides win when merging)."""
    out: Dict[str, Any] = {}
    if not params:
        return out
    mapping = [
        ("num_ctx", "num_ctx"),
        ("temperature", "temperature"),
        ("top_p", "top_p"),
        ("repeat_penalty", "repeat_penalty"),
        ("num_predict", "num_predict"),
    ]
    for src, dst in mapping:
        if src in params and params[src] is not None:
            out[dst] = params[src]
    return out


def merge_profile_params(profile: Optional[Dict[str, Any]], overrides: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    base = dict((profile or {}).get("params") or {})
    if not overrides:
        return base
    merged = {**base, **overrides}
    _validate_params(merged)
    return merged


def _normalized_model_key(name: Optional[str]) -> str:
    return (name or "").strip().lower()


def _profile_matches_model(profile: Dict[str, Any], expected_model: str) -> bool:
    """True when profile.model_name matches expected_model (case-insensitive, trimmed)."""
    if not (expected_model or "").strip():
        return False
    return _normalized_model_key(profile.get("model_name")) == _normalized_model_key(expected_model)


def resolve_qa_for_model(
    model_name: str,
    *,
    use_saved_defaults: bool = True,
    profile_id_override: Optional[str] = None,
    param_overrides: Optional[Dict[str, Any]] = None,
) -> tuple[Optional[str], Dict[str, Any], bool, Dict[str, Any]]:
    """Returns (system_prompt, ollama_options, think, trace_meta). trace_meta may be empty."""
    profile: Optional[Dict[str, Any]] = None
    trace: Dict[str, Any] = {}
    if profile_id_override:
        p = get_profile(profile_id_override)
        if p:
            if _profile_matches_model(p, model_name):
                profile = p
            else:
                trace["qa_profile_override_ignored"] = True
                trace["qa_profile_override_reason"] = "profile_model_mismatch"
        else:
            trace["qa_profile_override_ignored"] = True
            trace["qa_profile_override_reason"] = "profile_not_found"
    if profile is None and use_saved_defaults:
        profile = get_default_profile(model_name)
    params = merge_profile_params(profile, param_overrides)
    system = (profile or {}).get("system_prompt")
    if system is not None and not str(system).strip():
        system = None
    options = resolve_ollama_options_from_params(params)
    think_val = params.get("think")
    think = bool(think_val) if think_val is not None else False
    if profile:
        trace["qa_profile_id"] = profile.get("id")
        trace["qa_profile_version"] = profile.get("version")
    return system, options, think, trace


def resolve_eval_profile(
    *,
    eval_backend: str,
    eval_model_name: str,
    local_eval_model_name: Optional[str],
    use_saved_defaults: bool = True,
    profile_id_override: Optional[str] = None,
    param_overrides: Optional[Dict[str, Any]] = None,
) -> tuple[Optional[str], Dict[str, Any], bool, Dict[str, Any], Dict[str, Any]]:
    """Returns (system_prompt, ollama_options, think, openai_extra, trace_meta)."""
    external_eval = (eval_backend or "").strip() in {"DeepSeek", "OpenAI"}
    model_key = (eval_model_name or "").strip() if external_eval else (local_eval_model_name or "").strip()
    profile: Optional[Dict[str, Any]] = None
    trace: Dict[str, Any] = {}
    if profile_id_override:
        p = get_profile(profile_id_override)
        if p:
            profile_model = _normalized_model_key(p.get("model_name"))
            matches_eval_profile = profile_model == "eval"
            matches_runtime_model = _profile_matches_model(p, model_key) if model_key else False
            if matches_eval_profile or matches_runtime_model:
                profile = p
            else:
                trace["eval_profile_override_ignored"] = True
                trace["eval_profile_override_reason"] = "profile_model_mismatch"
        else:
            trace["eval_profile_override_ignored"] = True
            trace["eval_profile_override_reason"] = "profile_not_found"
    if profile is None and use_saved_defaults and model_key:
        profile = get_default_profile(model_key)
    params = merge_profile_params(profile, param_overrides)
    system = (profile or {}).get("system_prompt")
    if system is not None and not str(system).strip():
        system = None
    o_options = resolve_ollama_options_from_params(params)
    think_val = params.get("think")
    think = bool(think_val) if think_val is not None else False
    openai_extra: Dict[str, Any] = {}
    if params.get("temperature") is not None:
        openai_extra["temperature"] = float(params["temperature"])
    if params.get("top_p") is not None:
        openai_extra["top_p"] = float(params["top_p"])
    if profile:
        trace["eval_profile_id"] = profile.get("id")
        trace["eval_profile_version"] = profile.get("version")
    return system, o_options, think, openai_extra, trace


def resolve_tts_profile(
    *,
    profile_id_override: Optional[str] = None,
    model_name_hint: Optional[str] = None,
    use_saved_defaults: bool = True,
    param_overrides: Optional[Dict[str, Any]] = None,
) -> tuple[Optional[str], Optional[str], Optional[str], Dict[str, Any], Dict[str, Any]]:
    """
    Returns (tts_model, speaker_id, voice_preset, merged_params, trace).
    Preset 'Varsayılan' is treated as no preset downstream.
    """
    profile: Optional[Dict[str, Any]] = None
    if profile_id_override:
        p = get_profile(profile_id_override)
        if p:
            profile = p
    if profile is None and use_saved_defaults and (model_name_hint or "").strip():
        profile = get_default_profile(model_name_hint.strip())
    params = merge_profile_params(profile, param_overrides)
    tts_model = params.get("tts_model")
    if isinstance(tts_model, str) and tts_model.strip():
        tts_model = tts_model.strip()
    else:
        tts_model = None
    sp = params.get("speaker_id")
    speaker_id = str(sp).strip() if sp is not None and str(sp).strip() else None
    vp = params.get("voice_preset")
    voice_preset = str(vp).strip() if vp is not None and str(vp).strip() else None
    trace: Dict[str, Any] = {}
    if profile:
        trace["tts_profile_id"] = profile.get("id")
        trace["tts_profile_version"] = profile.get("version")
    return tts_model, speaker_id, voice_preset, params, trace
