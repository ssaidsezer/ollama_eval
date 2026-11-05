"""FastAPI application for RAG indexing, evaluation, and management APIs."""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
from typing import Any, Callable, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response

from pydantic import BaseModel, Field

from pipeline import (
    DEEPSEEK_EVAL_MODEL_NAME,
    EVAL_MODEL_NAME,
    OPENAI_EVAL_MODEL_NAME,
    QA_OLLAMA_MODEL,
    get_default_eval_model,
    get_eval_api_key,
    get_eval_api_key_env_var,
    get_external_eval_client,
    is_external_eval_backend,
    run_full_pipeline,
    unload_model,
    write_results_to_csv,
)
from prompts import DEFAULT_EVAL_SYSTEM_PROMPT
from rag.config import (
    DEFAULT_COLLECTION_NAME,
    QDRANT_URL,
    SMART_BOUNDARY_LLM_MODEL,
    SMART_CHILD_OVERLAP,
    SMART_CHILD_SIZE,
    SMART_PARENT_BLOCK_SIZE,
)
from rag.indexing import index_pdfs
from rag.smart_indexing import index_pdfs_smart

from web_backend.analysis_service import analyze_export_csv
from web_backend.asr_service import get_reports as asr_get_reports
from web_backend.asr_service import resolve_audio_path as asr_resolve_audio_path
from web_backend.asr_service import run_pipeline_job as asr_run_pipeline_job
from web_backend.chat_service import run_chat_eval
from web_backend.compat import (
    collection_name_full,
    collection_options_for_embed,
    delete_ollama_model,
    delete_qdrant_collection,
    ensure_tmp_dir,
    get_monitor_base_url,
    get_monitor_stats,
    get_ollama_base_url,
    list_all_ollama_model_names_raw,
    list_embedding_models,
    list_ollama_models,
    list_qdrant_collections,
    pull_ollama_model,
    smart_collection_name_full,
)
from web_backend.jobs import job_store
from web_backend.model_profiles_store import (
    clone_profile,
    create_profile,
    delete_profile,
    list_profiles,
    resolve_eval_profile,
    resolve_qa_for_model,
    resolve_tts_profile,
    update_profile,
)
from web_backend.schemas import (
    AnalysisResponse,
    AppConfigResponse,
    ChatEvalRequest,
    ChatEvalResponse,
    CollectionsResponse,
    ConnectionStatusResponse,
    DeleteModelRequest,
    EmbeddingModelsResponse,
    HealthResponse,
    IndexJobStartResponse,
    JobProgress,
    JobStatusResponse,
    ModelProfileCreate,
    ModelProfileOut,
    ModelProfileUpdate,
    ModelProfilesListResponse,
    OllamaModelsResponse,
    PullModelRequest,
    SimpleMessageResponse,
    TTSModelsResponse,
    TTSRequest,
)

WORKSPACE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(WORKSPACE_DIR / ".env", override=True)

app = FastAPI(title="RAG Pipeline API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get(
        "CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    ).split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _to_profile_out(raw: dict) -> ModelProfileOut:
    return ModelProfileOut(
        id=str(raw["id"]),
        name=str(raw["name"]),
        model_name=str(raw["model_name"]),
        system_prompt=raw.get("system_prompt"),
        params=dict(raw.get("params") or {}),
        is_default=bool(raw.get("is_default")),
        version=int(raw.get("version", 1)),
        updated_at=str(raw.get("updated_at", "")),
    )


@app.get("/api/model-profiles", response_model=ModelProfilesListResponse)
def api_list_model_profiles() -> ModelProfilesListResponse:
    profiles = [_to_profile_out(p) for p in list_profiles()]
    return ModelProfilesListResponse(profiles=profiles)


@app.post("/api/model-profiles", response_model=ModelProfileOut)
def api_create_model_profile(body: ModelProfileCreate) -> ModelProfileOut:
    try:
        params = body.params.model_dump(exclude_none=True)
        created = create_profile(
            name=body.name,
            model_name=body.model_name,
            system_prompt=body.system_prompt,
            params=params,
            is_default=body.is_default,
        )
        return _to_profile_out(created)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.put("/api/model-profiles/{profile_id}", response_model=ModelProfileOut)
def api_update_model_profile(profile_id: str, body: ModelProfileUpdate) -> ModelProfileOut:
    try:
        params_dict = None
        if body.params is not None:
            params_dict = body.params.model_dump(exclude_none=True)
        updated = update_profile(
            profile_id,
            name=body.name,
            model_name=body.model_name,
            system_prompt=body.system_prompt if not body.system_prompt_clear else None,
            system_prompt_set_null=body.system_prompt_clear,
            params=params_dict,
            is_default=body.is_default,
        )
        if not updated:
            raise HTTPException(status_code=404, detail="Profil bulunamadı.")
        return _to_profile_out(updated)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete("/api/model-profiles/{profile_id}", response_model=SimpleMessageResponse)
def api_delete_model_profile(profile_id: str) -> SimpleMessageResponse:
    ok = delete_profile(profile_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Profil bulunamadı.")
    return SimpleMessageResponse(success=True, message="Silindi.")


@app.get("/api/model-profiles/default-templates")
def api_model_profile_default_templates() -> dict:
    return {
        "eval_system": DEFAULT_EVAL_SYSTEM_PROMPT,
        "note": "QA için system_prompt boşsa yalnızca yerleşik kullanıcı mesajı şablonu kullanılır.",
    }


@app.post("/api/model-profiles/{profile_id}/clone", response_model=ModelProfileOut)
def api_clone_model_profile(profile_id: str) -> ModelProfileOut:
    try:
        cloned = clone_profile(profile_id)
        if not cloned:
            raise HTTPException(status_code=404, detail="Profil bulunamadı.")
        return _to_profile_out(cloned)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _json_safe_rows(rows: List[dict]) -> List[dict]:
    out: List[dict] = []
    for r in rows:
        row = {}
        for k, v in r.items():
            if v is None or isinstance(v, (str, int, float, bool)):
                row[k] = v
            elif hasattr(v, "item"):  # numpy scalar
                row[k] = v.item()
            else:
                row[k] = str(v)
        out.append(row)
    return out


def _parse_json_form_field(
    raw_json: str,
    *,
    field_name: str,
    default: Any,
    expected_type: type,
    require_type_always: bool = True,
) -> Any:
    value = json.loads(raw_json) if raw_json else default
    if require_type_always or value:
        if not isinstance(value, expected_type):
            exp_name = "nesne" if expected_type is dict else "liste"
            raise ValueError(f"{field_name} {exp_name} olmalı")
    return value


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse()


@app.get("/api/config", response_model=AppConfigResponse)
def config() -> AppConfigResponse:
    monitor_url = get_monitor_base_url()
    return AppConfigResponse(
        qdrant_url=os.environ.get("QDRANT_URL", QDRANT_URL),
        default_collection_name=os.environ.get("QDRANT_COLLECTION", DEFAULT_COLLECTION_NAME),
        ollama_base_url=get_ollama_base_url(),
        indexing_ollama_base_url=os.environ.get(
            "INDEXING_OLLAMA_BASE_URL",
            "http://192.168.1.151:11434",
        ),
        monitor_url=monitor_url,
        qa_ollama_model=QA_OLLAMA_MODEL,
        eval_model_name=EVAL_MODEL_NAME,
        deepseek_eval_model_name=DEEPSEEK_EVAL_MODEL_NAME or EVAL_MODEL_NAME or "",
        openai_eval_model_name=OPENAI_EVAL_MODEL_NAME or "gpt-5.4-mini",
        chunk_size_default=int(os.environ.get("CHUNK_SIZE", "1000")),
        chunk_overlap_default=int(os.environ.get("CHUNK_OVERLAP", "200")),
        smart_parent_size=SMART_PARENT_BLOCK_SIZE,
        smart_child_size=SMART_CHILD_SIZE,
        smart_child_overlap=SMART_CHILD_OVERLAP,
        smart_boundary_llm_model=SMART_BOUNDARY_LLM_MODEL,
    )


@app.get("/api/connection-status", response_model=ConnectionStatusResponse)
def connection_status() -> ConnectionStatusResponse:
    q_url = os.environ.get("QDRANT_URL", QDRANT_URL)
    q_ok, q_msg = True, f"Qdrant bağlı ({q_url})"
    try:
        import requests

        r = requests.get(f"{q_url}/collections", timeout=5)
        r.raise_for_status()
    except Exception:
        q_ok = False
        q_msg = f"Qdrant erişilemiyor ({q_url})"

    o_base = get_ollama_base_url()
    o_ok, o_msg = False, "Ollama tanımlı değil"
    if o_base:
        try:
            import requests

            r = requests.get(f"{o_base}/api/tags", timeout=5)
            r.raise_for_status()
            o_ok = True
            o_msg = f"Ollama bağlı ({o_base})"
        except Exception:
            o_msg = f"Ollama erişilemiyor ({o_base})"

    stats, m_err = get_monitor_stats()
    return ConnectionStatusResponse(
        qdrant_ok=q_ok,
        qdrant_message=q_msg,
        ollama_ok=o_ok,
        ollama_message=o_msg,
        monitor=stats,
        monitor_error=m_err if not stats else None,
    )


@app.get("/api/models/ollama", response_model=OllamaModelsResponse)
def api_ollama_models() -> OllamaModelsResponse:
    models, err, elapsed, fcount = list_ollama_models()
    return OllamaModelsResponse(
        models=sorted({m for m in models if m}),
        error=err,
        filter_elapsed_seconds=elapsed,
        filtered_embedding_count=fcount,
    )


@app.get("/api/models/embeddings", response_model=EmbeddingModelsResponse)
def api_embed_models() -> EmbeddingModelsResponse:
    models, err = list_embedding_models()
    return EmbeddingModelsResponse(models=models, error=err)


@app.get("/api/qdrant/collections", response_model=CollectionsResponse)
def api_collections() -> CollectionsResponse:
    cols, err = list_qdrant_collections()
    return CollectionsResponse(collections=cols, error=err)


@app.get("/api/collection-options")
def api_collection_options(embed_model: str) -> dict:
    cols, err = list_qdrant_collections()
    if err:
        return {"classic": [], "smart_bases": [], "error": err}
    classic, smart = collection_options_for_embed(cols, embed_model)
    return {"classic": classic, "smart_bases": smart, "error": ""}


@app.delete("/api/qdrant/collections/{name}", response_model=SimpleMessageResponse)
def api_delete_collection(name: str) -> SimpleMessageResponse:
    ok, msg = delete_qdrant_collection(name)
    return SimpleMessageResponse(success=ok, message=msg)


@app.post("/api/ollama/models/pull", response_model=SimpleMessageResponse)
def api_pull(req: PullModelRequest) -> SimpleMessageResponse:
    ok, msg = pull_ollama_model(req.name.strip())
    return SimpleMessageResponse(success=ok, message=msg)


@app.delete("/api/ollama/models", response_model=SimpleMessageResponse)
def api_delete_model(req: DeleteModelRequest) -> SimpleMessageResponse:
    ok, msg = delete_ollama_model(req.name.strip())
    return SimpleMessageResponse(success=ok, message=msg)


@app.get("/api/ollama/models/all-raw")
def api_all_ollama_raw() -> dict:
    names, err = list_all_ollama_model_names_raw()
    return {"models": names, "error": err}


# --- Index job ---


def _run_index_job(
    pdf_paths: List[str],
    *,
    use_smart: bool,
    collection_base: str,
    qdrant_url: str,
    embed_model: str,
    chunk_size: int,
    chunk_overlap: int,
    smart_parent_size: int,
    smart_child_size: int,
    smart_child_overlap: int,
    boundary_llm_model: str,
    ollama_base_url: str,
    progress_cb: Any,
) -> dict:
    if use_smart:
        index_name = smart_collection_name_full(
            collection_base,
            embed_model,
            smart_parent_size,
            smart_child_size,
            smart_child_overlap,
        )
        return index_pdfs_smart(
            pdf_paths,
            base_collection=index_name,
            parent_size=smart_parent_size,
            child_size=smart_child_size,
            child_overlap=smart_child_overlap,
            boundary_llm_model=boundary_llm_model,
            qdrant_url=qdrant_url,
            embed_model=embed_model,
            ollama_base_url=ollama_base_url,
            progress_callback=progress_cb,
        )

    index_name = collection_name_full(collection_base, embed_model, chunk_size, chunk_overlap)
    return index_pdfs(
        pdf_paths,
        collection_name=index_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        qdrant_url=qdrant_url,
        embed_model=embed_model,
        ollama_base_url=ollama_base_url,
        progress_callback=progress_cb,
    )


@app.post("/api/jobs/index", response_model=IndexJobStartResponse)
async def start_index_job(
    files: List[UploadFile] = File(...),
    use_smart: bool = Form(False),
    embed_model: str = Form(...),
    chunk_size: int = Form(1000),
    chunk_overlap: int = Form(200),
    smart_parent_size: int = Form(SMART_PARENT_BLOCK_SIZE),
    smart_child_size: int = Form(SMART_CHILD_SIZE),
    smart_child_overlap: int = Form(SMART_CHILD_OVERLAP),
    boundary_llm_model: str = Form(SMART_BOUNDARY_LLM_MODEL),
    index_ollama_base_url: str = Form(""),
) -> IndexJobStartResponse:
    if not files or not any(f.filename for f in files):
        raise HTTPException(status_code=400, detail="İndekslenecek PDF bulunamadı.")

    tmp = ensure_tmp_dir()
    paths: List[str] = []
    for up in files:
        if not up.filename or not up.filename.lower().endswith(".pdf"):
            continue
        dest = tmp / up.filename
        content = await up.read()
        dest.write_bytes(content)
        paths.append(str(dest))

    if not paths:
        raise HTTPException(status_code=400, detail="Geçerli PDF dosyası yok.")

    collection_base = os.environ.get("QDRANT_COLLECTION", DEFAULT_COLLECTION_NAME)
    qdrant_url = os.environ.get("QDRANT_URL", QDRANT_URL)
    ollama_base_url = index_ollama_base_url.strip() or get_ollama_base_url()

    jid = job_store.create()

    def run(progress: Callable[..., Any]) -> dict:
        return _run_index_job(
            paths,
            use_smart=use_smart,
            collection_base=collection_base,
            qdrant_url=qdrant_url,
            embed_model=embed_model,
            chunk_size=int(chunk_size),
            chunk_overlap=int(chunk_overlap),
            smart_parent_size=int(smart_parent_size),
            smart_child_size=int(smart_child_size),
            smart_child_overlap=int(smart_child_overlap),
            boundary_llm_model=boundary_llm_model,
            ollama_base_url=ollama_base_url,
            progress_cb=progress,
        )

    job_store.run_async(jid, run)
    return IndexJobStartResponse(job_id=jid)


@app.post("/api/jobs/csv-pipeline", response_model=IndexJobStartResponse)
async def start_csv_job(
    csv_file: Optional[UploadFile] = File(None),
    use_sample: bool = Form(False),
    eval_enabled: bool = Form(True),
    eval_backend: str = Form("DeepSeek"),
    eval_model_name: str = Form(""),
    local_eval_model_name: Optional[str] = Form(None),
    csv_question_col: str = Form("question"),
    csv_answer_col: str = Form("answer"),
    csv_embed_model: Optional[str] = Form(None),
    csv_collection_name: str = Form(""),
    csv_smart_rag: bool = Form(False),
    csv_retrieval_mode: str = Form("vector"),
    csv_score_threshold: float = Form(0.55),
    rag_mode: str = Form("rag"),
    k: int = Form(5),
    thinking_enabled: bool = Form(False),
    qa_models_json: str = Form("[]"),
    openai_api_key: Optional[str] = Form(None),
    use_saved_qa_defaults: bool = Form(True),
    qa_profile_by_model_json: str = Form("{}"),
    qa_param_overrides_json: str = Form("{}"),
    use_saved_eval_defaults: bool = Form(True),
    eval_profile_id: Optional[str] = Form(None),
    eval_param_overrides_json: str = Form("{}"),
) -> IndexJobStartResponse:
    """qa_models_json: JSON array of model names."""

    try:
        qa_models: List[str] = json.loads(qa_models_json) if qa_models_json else []
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="qa_models_json geçersiz JSON.") from exc
    try:
        qa_profile_map: dict = _parse_json_form_field(
            qa_profile_by_model_json,
            field_name="qa_profile_by_model",
            default={},
            expected_type=dict,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"qa_profile_by_model_json: {exc}") from exc
    try:
        qa_param_overrides: dict = _parse_json_form_field(
            qa_param_overrides_json,
            field_name="qa_param_overrides",
            default={},
            expected_type=dict,
            require_type_always=False,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"qa_param_overrides_json: {exc}") from exc
    try:
        eval_param_overrides: dict = _parse_json_form_field(
            eval_param_overrides_json,
            field_name="eval_param_overrides",
            default={},
            expected_type=dict,
            require_type_always=False,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"eval_param_overrides_json: {exc}") from exc

    tmp = ensure_tmp_dir()
    sample_path = WORKSPACE_DIR / "sample_rag_input.csv"
    csv_path: Optional[Path] = None
    errors_pre: List[str] = []

    if csv_file and csv_file.filename:
        dest = tmp / "uploaded_input.csv"
        dest.write_bytes(await csv_file.read())
        csv_path = dest
    elif use_sample and sample_path.exists():
        csv_path = sample_path
    else:
        errors_pre.append("CSV seçilmedi.")

    client = None
    if csv_path is not None and eval_enabled and is_external_eval_backend(eval_backend):
        key = get_eval_api_key(eval_backend, request_key=openai_api_key)
        if not key:
            env_var = get_eval_api_key_env_var(eval_backend)
            errors_pre.append(f"{eval_backend} değerlendirme motoru seçili. {env_var} gerekli.")
            csv_path = None
        else:
            client = get_external_eval_client(eval_backend, api_key=key)

    if rag_mode != "no_rag" and not (csv_collection_name or "").strip() and not errors_pre:
        errors_pre.append("Seçili retrieval tipi için geçerli bir koleksiyon gerekli.")
        csv_path = None

    qdrant_url = os.environ.get("QDRANT_URL", QDRANT_URL)
    ollama_models, _, _, _ = list_ollama_models()
    all_m = sorted({m for m in ollama_models if m})

    jid = job_store.create()

    def run(progress: Callable[..., Any]) -> dict:
        rows: List[dict] = []
        errors: List[str] = list(errors_pre)

        if csv_path is None:
            return {"rows": [], "errors": errors}

        eval_sys, eval_oopts, eval_think_prof, eval_openai_extras, eval_tr = resolve_eval_profile(
            eval_backend=eval_backend,
            eval_model_name=eval_model_name or get_default_eval_model(eval_backend),
            local_eval_model_name=local_eval_model_name,
            use_saved_defaults=use_saved_eval_defaults,
            profile_id_override=(eval_profile_id or "").strip() or None,
            param_overrides=eval_param_overrides or None,
        )

        if not qa_models:
            raise HTTPException(
                status_code=400,
                detail="Benchmark için en az bir model seçilmeli. qa_models_json alanı boş veya geçersiz.",
            )
        _models = qa_models
        n_models = len(_models)
        for idx, qa_model in enumerate(_models):
            try:
                pid_ov = qa_profile_map.get(qa_model)
                if isinstance(pid_ov, str):
                    pid_ov = pid_ov.strip() or None
                else:
                    pid_ov = None
                qa_sys, qa_opts, qa_think_prof, qa_tr = resolve_qa_for_model(
                    qa_model,
                    use_saved_defaults=use_saved_qa_defaults,
                    profile_id_override=pid_ov,
                    param_overrides=qa_param_overrides or None,
                )
                qa_think = bool(thinking_enabled or qa_think_prof)
                trace = {**qa_tr, **eval_tr}
                model_rows = run_full_pipeline(
                    csv_path=str(csv_path),
                    collection_name=csv_collection_name,
                    qdrant_url=qdrant_url,
                    eval_model=eval_model_name or get_default_eval_model(eval_backend),
                    k=int(k),
                    openai_client=client,
                    eval_backend=eval_backend if is_external_eval_backend(eval_backend) else "ollama",
                    eval_local_model=local_eval_model_name,
                    qa_model=qa_model,
                    rag_mode=rag_mode,
                    eval_enabled=eval_enabled,
                    question_col=csv_question_col,
                    answer_col=csv_answer_col,
                    embed_model=csv_embed_model or "",
                    smart_chunking=csv_smart_rag,
                    score_threshold=float(csv_score_threshold),
                    retrieval_mode=csv_retrieval_mode,
                    qa_system_prompt=qa_sys,
                    qa_ollama_options=qa_opts or None,
                    qa_think=qa_think,
                    eval_system_prompt=eval_sys,
                    eval_ollama_options=eval_oopts or None,
                    eval_think=eval_think_prof,
                    eval_openai_extras=eval_openai_extras or None,
                    profile_trace=trace,
                )
                rows.extend(model_rows)
            except Exception as exc:
                errors.append(f"{qa_model} için pipeline hatası: {exc}")
            finally:
                unload_model(qa_model)
            progress(
                "csv_model",
                idx + 1,
                n_models,
                0.0,
            )

        buf = io.StringIO()
        write_results_to_csv(rows, buf)
        return {
            "rows": _json_safe_rows(rows),
            "errors": errors,
            "csv_text": buf.getvalue(),
        }

    job_store.run_async(jid, run)
    return IndexJobStartResponse(job_id=jid)


@app.get("/api/jobs/{job_id}", response_model=JobStatusResponse)
def get_job(job_id: str) -> JobStatusResponse:
    data = job_store.get(job_id)
    if not data:
        raise HTTPException(status_code=404, detail="Job bulunamadı.")

    prog = data.get("progress")
    progress_model = None
    if prog:
        progress_model = JobProgress(
            phase=str(prog.get("phase", "")),
            current=int(prog.get("current", 0)),
            total=int(prog.get("total", 0)),
            elapsed_sec=float(prog.get("elapsed_sec", 0.0)),
        )

    return JobStatusResponse(
        id=job_id,
        status=data["status"],
        progress=progress_model,
        result=data.get("result"),
        error=data.get("error"),
    )


@app.post("/api/chat/eval", response_model=ChatEvalResponse)
def chat_eval(req: ChatEvalRequest) -> ChatEvalResponse:
    models, err, _, _ = list_ollama_models()
    if err:
        return ChatEvalResponse(errors=[err])
    all_models = sorted({m for m in models if m})
    try:
        return run_chat_eval(req, all_models)
    except ValueError as exc:
        return ChatEvalResponse(errors=[str(exc)])
    except Exception as exc:
        return ChatEvalResponse(
            errors=[f"Manual chat eval sırasında beklenmeyen bir hata oluştu: {exc}"]
        )


@app.get("/api/voice/models", response_model=TTSModelsResponse)
def voice_models() -> TTSModelsResponse:
    from voice_utils import get_downloaded_tts_models

    return TTSModelsResponse(downloaded_models=get_downloaded_tts_models())


@app.post("/api/voice/tts")
def voice_tts(req: TTSRequest) -> Response:
    from voice_utils import synthesize_speech

    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Metin boş.")
    pm, p_sp, p_vp, _merged, _tr = resolve_tts_profile(
        profile_id_override=(req.tts_profile_id or "").strip() or None,
        model_name_hint=(req.model or "").strip() or None,
        use_saved_defaults=req.use_saved_tts_defaults,
        param_overrides=req.tts_param_overrides,
    )
    model_use = (req.model or "").strip() or (pm or "").strip() or os.getenv("TTS_MODEL")
    speaker_final = req.speaker_id
    if speaker_final is not None and str(speaker_final).strip() == "":
        speaker_final = None
    if speaker_final is None:
        speaker_final = p_sp

    if req.voice_preset and req.voice_preset != "Varsayılan":
        preset_final: Optional[str] = req.voice_preset
    elif p_vp and str(p_vp).strip() and p_vp != "Varsayılan":
        preset_final = str(p_vp).strip()
    else:
        preset_final = None
    try:
        wav_bytes, _sr, duration = synthesize_speech(
            text,
            model=model_use,
            speaker_id=speaker_final,
            voice_preset=preset_final,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={"X-Audio-Duration": str(duration)},
    )


@app.post("/api/analysis", response_model=AnalysisResponse)
async def analysis(file: UploadFile = File(...)) -> AnalysisResponse:
    content = await file.read()
    return analyze_export_csv(content)


# --- ASR Benchmark ---

class AsrRunRequest(BaseModel):
    steps: Optional[List[int]] = Field(default=None, description="Çalıştırılacak adım numaraları (1-7)")
    skip_benchmark: bool = False


@app.get("/api/asr/reports")
def asr_reports() -> dict:
    return asr_get_reports()


@app.get("/api/asr/audio")
def asr_audio(path: str = Query(...)) -> FileResponse:
    resolved = asr_resolve_audio_path(path)
    if not resolved:
        raise HTTPException(status_code=404, detail="Ses dosyası bulunamadı.")
    return FileResponse(str(resolved), media_type="audio/wav")


@app.post("/api/asr/run", response_model=IndexJobStartResponse)
def asr_run(req: AsrRunRequest) -> IndexJobStartResponse:
    jid = job_store.create()

    def run(progress: Callable[..., Any]) -> dict:
        return asr_run_pipeline_job(
            progress,
            steps=req.steps,
            skip_benchmark=req.skip_benchmark,
        )

    job_store.run_async(jid, run)
    return IndexJobStartResponse(job_id=jid)
