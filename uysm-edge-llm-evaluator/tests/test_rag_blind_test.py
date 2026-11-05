from __future__ import annotations

import json
import csv
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.qa import validate_rag_blind_test as validator


@pytest.fixture
def valid_rows() -> list[dict[str, str]]:
    per_source_difficulties = ["easy"] * 20 + ["medium"] * 20 + ["hard"] * 20
    per_source_overlaps = ["high"] * 20 + ["medium"] * 20 + ["low"] * 20
    difficulties = per_source_difficulties * 2
    overlaps = per_source_overlaps * 2
    per_source_question_types = (
        ["direct"] * 15 + ["scenario"] * 21 + ["procedure"] * 12
        + ["comparison"] * 6 + ["cause_effect"] * 6
    )
    per_source_scopes = (
        ["single_chunk"] * 33 + ["same_page_multi_chunk"] * 13
        + ["multi_page"] * 8 + ["none"] * 6
    )
    question_types = per_source_question_types * 2
    scopes = per_source_scopes * 2
    assert {len(difficulties), len(overlaps), len(question_types), len(scopes)} == {120}

    source_topics = {
        "taktik_muharebe_yarali_bakimi_el_kitabi.pdf": [
            "bleeding_control", "airway", "respiration", "circulation_shock",
            "medication", "burns_hypothermia", "evacuation_documentation",
            "head_eye_injury", "fracture_immobilization", "assessment_monitoring",
        ],
        "askeri_egitim_kitabi.pdf": [
            "weapon_safety", "ammunition", "handgun", "rifle", "machine_gun",
            "rocket_grenade", "marksmanship", "maintenance_malfunction",
            "anti_aircraft", "weapon_identification",
        ],
    }
    rows = []
    for index in range(120):
        source = (
            "taktik_muharebe_yarali_bakimi_el_kitabi.pdf"
            if index < 60 else "askeri_egitim_kitabi.pdf"
        )
        source_index = index % 60
        answerability = "unanswerable" if scopes[index] == "none" else "answerable"
        prefix = "tccc" if index < 60 else "askeri"
        rows.append({
            "question_id": f"{prefix}_{source_index + 1:03d}",
            "source_document": source,
            "topic": source_topics[source][source_index % 10],
            "question_type": question_types[index],
            "difficulty": difficulties[index],
            "retrieval_scope": scopes[index],
            "lexical_overlap": overlaps[index],
            "answerability": answerability,
        })
    return rows


def test_projection_exposes_only_id_and_question() -> None:
    rows = [{
        "question_id": "tccc_001",
        "question": "Kaynakta hangi işlem öneriliyor?",
        "reference_answer": "Kaynak cevabı",
    }]

    assert validator.build_questions_rows(rows) == [{
        "question_id": "tccc_001",
        "question": "Kaynakta hangi işlem öneriliyor?",
    }]


def test_parse_json_list_rejects_non_list_values() -> None:
    with pytest.raises(ValueError, match="required_facts"):
        validator.parse_json_list({"required_facts": "{}"}, "required_facts")


def test_distribution_validator_reports_wrong_source_count(valid_rows) -> None:
    issues = validator.validate_distribution(valid_rows[:-1])
    assert any(issue.code == "row_count" for issue in issues)
    assert any(issue.code == "source_distribution" for issue in issues)


def test_distribution_validator_accepts_exact_contract(valid_rows) -> None:
    assert validator.validate_distribution(valid_rows) == []


@pytest.fixture
def base_answerable_row() -> dict[str, str]:
    return {
        "question_id": "tccc_001",
        "source_document": "taktik_muharebe_yarali_bakimi_el_kitabi.pdf",
        "pdf_page_index": "2",
        "printed_page_label": "",
        "acceptable_page_indices": "[2]",
        "gold_evidence": '[{"pdf_page_index":2,"printed_page_label":"","text":"gerçek kaynak metni"}]',
        "question": "Kaynağa göre işlem nedir?",
        "reference_answer": "Gerçek kaynak işlemi uygulanır.",
        "required_facts": '["gerçek kaynak işlemi"]',
        "topic": "assessment_monitoring",
        "question_type": "direct",
        "difficulty": "easy",
        "difficulty_reason": "Ayırt edici terim ve cevap aynı pasajdadır.",
        "retrieval_scope": "single_chunk",
        "lexical_overlap": "high",
        "criticality": "medium",
        "answerability": "answerable",
        "notes": "reviewed=source; overlap=keep; distinct target",
    }


@pytest.fixture
def base_unanswerable_row(base_answerable_row) -> dict[str, str]:
    row = dict(base_answerable_row)
    row.update({
        "pdf_page_index": "",
        "acceptable_page_indices": "[]",
        "gold_evidence": "[]",
        "reference_answer": "Bu bilgi sağlanan iki belgede yer almıyor.",
        "required_facts": '["iki belgede bilgi bulunmadığını belirtir"]',
        "retrieval_scope": "none",
        "answerability": "unanswerable",
    })
    return row


def test_answerable_evidence_must_exist_on_declared_page(base_answerable_row) -> None:
    row = dict(base_answerable_row)
    row["gold_evidence"] = json.dumps([
        {"pdf_page_index": 2, "printed_page_label": "", "text": "olmayan kanıt"}
    ], ensure_ascii=False)

    issues = validator.validate_rows(
        [row],
        {row["source_document"]: ["ilk", "gerçek kaynak metni"]},
        [],
        enforce_distribution=False,
    )

    assert any(issue.code == "evidence_not_found" for issue in issues)


def test_unanswerable_row_has_no_gold_page_or_evidence(base_unanswerable_row) -> None:
    issues = validator.validate_rows(
        [base_unanswerable_row], {}, [], enforce_distribution=False
    )
    assert issues == []


def test_legacy_answer_column_rejects_reused_answer_target(base_answerable_row) -> None:
    row = dict(base_answerable_row)
    row["gold_evidence"] = json.dumps([
        {"pdf_page_index": 2, "printed_page_label": "", "text": "gerçek kaynak işlemi"}
    ], ensure_ascii=False)
    issues = validator.validate_rows(
        [row],
        {row["source_document"]: ["ilk", "gerçek kaynak işlemi"]},
        [{"answer": row["reference_answer"]}],
        enforce_distribution=False,
    )

    assert any(issue.code == "legacy_answer_reuse" for issue in issues)


def test_legacy_reference_answer_fallback_rejects_reused_answer_target(base_answerable_row) -> None:
    row = dict(base_answerable_row)
    row["gold_evidence"] = json.dumps([
        {"pdf_page_index": 2, "printed_page_label": "", "text": "gerçek kaynak işlemi"}
    ], ensure_ascii=False)
    issues = validator.validate_rows(
        [row],
        {row["source_document"]: ["ilk", "gerçek kaynak işlemi"]},
        [{"answer": "", "reference_answer": row["reference_answer"]}],
        enforce_distribution=False,
    )

    assert any(issue.code == "legacy_answer_reuse" for issue in issues)


def test_answerable_fact_and_answer_supported_by_evidence(base_answerable_row) -> None:
    row = dict(base_answerable_row)
    row["gold_evidence"] = json.dumps([
        {"pdf_page_index": 2, "printed_page_label": "", "text": "gerçek kaynak işlemi"}
    ], ensure_ascii=False)

    issues = validator.validate_rows(
        [row],
        {row["source_document"]: ["ilk", "gerçek kaynak işlemi"]},
        [],
        enforce_distribution=False,
    )

    assert issues == []


def test_answerable_row_rejects_required_fact_absent_from_evidence(base_answerable_row) -> None:
    row = dict(base_answerable_row)
    row["required_facts"] = '["desteklenmeyen atomik gerçek"]'
    row["gold_evidence"] = json.dumps([
        {"pdf_page_index": 2, "printed_page_label": "", "text": "gerçek kaynak işlemi"}
    ], ensure_ascii=False)

    issues = validator.validate_rows(
        [row],
        {row["source_document"]: ["ilk", "gerçek kaynak işlemi"]},
        [],
        enforce_distribution=False,
    )

    assert any(issue.code == "required_fact_not_in_evidence" for issue in issues)


def test_answerable_row_rejects_required_fact_absent_from_reference_answer(base_answerable_row) -> None:
    row = dict(base_answerable_row)
    row["reference_answer"] = "Başka bir ifade kullanılır."
    row["gold_evidence"] = json.dumps([
        {"pdf_page_index": 2, "printed_page_label": "", "text": "gerçek kaynak işlemi"}
    ], ensure_ascii=False)

    issues = validator.validate_rows(
        [row],
        {row["source_document"]: ["ilk", "gerçek kaynak işlemi"]},
        [],
        enforce_distribution=False,
    )

    assert any(issue.code == "required_fact_not_in_reference_answer" for issue in issues)


def test_overlap_candidate_requires_machine_readable_review_disposition(base_answerable_row) -> None:
    row = dict(base_answerable_row)
    row.update({
        "question": "Turnike ne zaman gevşetilir?",
        "gold_evidence": '[{"pdf_page_index":2,"printed_page_label":"","text":"gerçek kaynak işlemi"}]',
        "notes": "reviewed=source; overlap=keep",
    })
    issues = validator.validate_rows(
        [row],
        {row["source_document"]: ["ilk", "gerçek kaynak işlemi"]},
        [{"question": "Turnikeyi hangi zamanda gevşetmeliyim?"}],
        enforce_distribution=False,
    )

    assert any(issue.code == "overlap_review_required" for issue in issues)


def test_overlap_candidate_accepts_machine_readable_review_disposition(base_answerable_row) -> None:
    row = dict(base_answerable_row)
    row.update({
        "question": "Turnike ne zaman gevşetilir?",
        "gold_evidence": '[{"pdf_page_index":2,"printed_page_label":"","text":"gerçek kaynak işlemi"}]',
        "notes": "reviewed=source; overlap=keep; reason=ayrı hedef",
    })
    issues = validator.validate_rows(
        [row],
        {row["source_document"]: ["ilk", "gerçek kaynak işlemi"]},
        [{"question": "Turnikeyi hangi zamanda gevşetmeliyim?"}],
        enforce_distribution=False,
    )

    assert not any(issue.code == "overlap_review_required" for issue in issues)


def test_legacy_answer_paraphrase_requires_overlap_review_disposition(
    base_answerable_row,
) -> None:
    row = dict(base_answerable_row)
    row.update({
        "question": "Telsiz haberleşmesinde kod kelimeler nasıl seçilir?",
        "reference_answer": "Güvenli nakil öncesinde yaralının sabitlenmesi gerekir.",
        "required_facts": '["güvenli nakil öncesinde yaralının sabitlenmesi gerekir"]',
        "gold_evidence": '[{"pdf_page_index":2,"printed_page_label":"","text":"güvenli nakil öncesinde yaralının sabitlenmesi gerekir"}]',
        "notes": "reviewed=source; overlap=keep",
    })
    legacy = [{
        "question": "Bir silahın emniyet mandalı ne zaman kullanılır?",
        "answer": "Yaralının güvenli taşınması için önce sabitlenmesi gerekir.",
    }]

    candidates = validator.find_overlap_candidates([row], legacy, threshold=0.35)
    issues = validator.validate_rows(
        [row],
        {row["source_document"]: ["ilk", "güvenli nakil öncesinde yaralının sabitlenmesi gerekir"]},
        legacy,
        enforce_distribution=False,
    )

    assert candidates[0]["comparison_set"] == "legacy"
    assert any(issue.code == "overlap_review_required" for issue in issues)


def test_row_validator_rejects_an_unsupported_source_when_counts_are_disabled(
    base_answerable_row,
) -> None:
    row = dict(base_answerable_row)
    row["source_document"] = "unsupported.pdf"

    issues = validator.validate_rows([row], {}, [], enforce_distribution=False)

    assert any(issue.code == "source_document" for issue in issues)


def test_overlap_report_flags_reused_answer_target() -> None:
    current = [{"question_id": "tccc_001", "question": "Turnike ne zaman gevşetilir?"}]
    legacy = [{"question": "Turnikeyi hangi zamanda gevşetmeliyim?"}]

    matches = validator.find_overlap_candidates(current, legacy, threshold=0.35)

    assert matches[0]["question_id"] == "tccc_001"
    assert matches[0]["comparison_set"] == "legacy"


def test_load_rows_returns_csv_rows(tmp_path: Path) -> None:
    path = tmp_path / "rows.csv"
    path.write_text("question_id,question\ntccc_001,Soru?\n", encoding="utf-8")

    assert validator.load_rows(path) == [{"question_id": "tccc_001", "question": "Soru?"}]


def test_load_source_pages_uses_physical_pages(monkeypatch, tmp_path: Path) -> None:
    for name in validator.SOURCE_DOCUMENTS:
        (tmp_path / name).touch()
    monkeypatch.setattr(validator, "_extract_pages_text", lambda path: ["ilk sayfa", "ikinci sayfa"])

    assert validator.load_source_pages(tmp_path) == {
        name: ["ilk sayfa", "ikinci sayfa"] for name in validator.SOURCE_DOCUMENTS
    }


def test_report_includes_distribution_evidence_and_overlap_disposition(
    base_answerable_row,
) -> None:
    row = dict(base_answerable_row)
    row["notes"] = "reviewed=source; overlap=keep"
    report = validator.build_validation_report(
        [row],
        [{
            "question_id": row["question_id"], "max_score": 0.5, "comparison_set": "legacy",
            "legacy_neighbors": [{"question_id": "legacy_001", "question": "Eski soru", "score": 0.5}],
            "new_set_neighbors": [],
        }],
    )

    assert "## Distribution" in report
    assert "## Evidence checks" in report
    assert "reviewed=source; overlap=keep" in report
    assert "### tccc_001 legacy neighbors" in report


def test_cli_does_not_modify_derivatives_after_validation_error(
    monkeypatch, tmp_path: Path, base_unanswerable_row,
) -> None:
    gold = tmp_path / "gold.csv"
    legacy = tmp_path / "legacy.csv"
    questions_output = tmp_path / "questions.csv"
    report_output = tmp_path / "report.md"
    with gold.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=validator.GOLD_FIELDS)
        writer.writeheader()
        writer.writerow(base_unanswerable_row)
    legacy.write_text("question\nEski soru?\n", encoding="utf-8")
    questions_output.write_text("unchanged questions", encoding="utf-8")
    report_output.write_text("unchanged report", encoding="utf-8")
    monkeypatch.setattr(validator, "load_source_pages", lambda pdf_dir: {})

    assert validator.main([
        "--gold", str(gold), "--legacy", str(legacy),
        "--questions-output", str(questions_output), "--report-output", str(report_output),
    ]) == 1
    assert questions_output.read_text(encoding="utf-8") == "unchanged questions"
    assert report_output.read_text(encoding="utf-8") == "unchanged report"


def test_module_cli_executes_after_all_helpers_are_defined(monkeypatch, tmp_path: Path) -> None:
    gold = tmp_path / "gold.csv"
    legacy = tmp_path / "legacy.csv"
    gold.write_text("question_id\ninvalid\n", encoding="utf-8")
    legacy.write_text("question\nEski soru?\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [
        "validate_rag_blind_test", "--gold", str(gold), "--legacy", str(legacy), "--check-only",
    ])
    result = subprocess.run(
        [sys.executable, "-m", "scripts.qa.validate_rag_blind_test", *sys.argv[1:]],
        cwd=Path(__file__).parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    assert "row_count" in result.stdout
    assert "NameError" not in result.stderr


def test_script_cli_can_import_repo_modules_directly() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/qa/validate_rag_blind_test.py", "--help"],
        cwd=Path(__file__).parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Validate and project the RAG blind test set" in result.stdout
    assert "ModuleNotFoundError" not in result.stderr
