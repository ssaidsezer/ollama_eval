from __future__ import annotations

from pathlib import Path

from scripts.evaluation import batch_eval, run_llamacpp_inference_eval, run_rag_ablation
from scripts.experiments import run_speculative_ablation
from scripts.qa import build_balanced_qa, gen_qa_from_images, gen_qa_render, validate_rag_blind_test
from scripts.rag import check_index_integrity


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_active_scripts_resolve_the_repository_root() -> None:
    modules = (
        batch_eval,
        run_llamacpp_inference_eval,
        run_rag_ablation,
        run_speculative_ablation,
        build_balanced_qa,
        gen_qa_from_images,
        gen_qa_render,
        validate_rag_blind_test,
        check_index_integrity,
    )
    assert all(module.REPO_ROOT == REPO_ROOT for module in modules)


def test_balanced_builder_retains_legacy_100_defaults() -> None:
    args = build_balanced_qa.parse_args([])
    assert args.per_book == 45
    assert args.seed == 42
    assert args.out == REPO_ROOT / "data/qa/final_qa_100.csv"


def test_balanced_builder_accepts_canonical_150_parameters() -> None:
    args = build_balanced_qa.parse_args(
        ["--per-book", "70", "--out", "data/qa/final_qa_150.csv", "--seed", "42"]
    )
    assert args.per_book == 70
    assert args.seed == 42
    assert args.out == Path("data/qa/final_qa_150.csv")
