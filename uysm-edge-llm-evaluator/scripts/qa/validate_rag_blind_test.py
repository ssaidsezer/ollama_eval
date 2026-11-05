from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import sys
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from types import MappingProxyType

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rag.pdf_text import _decode_pdf_text, _extract_pages_text


GOLD_FIELDS = (
    "question_id", "source_document", "pdf_page_index", "printed_page_label",
    "acceptable_page_indices", "gold_evidence", "question", "reference_answer",
    "required_facts", "topic", "question_type", "difficulty",
    "difficulty_reason", "retrieval_scope", "lexical_overlap", "criticality",
    "answerability", "notes",
)

TCCC_DOCUMENT = "taktik_muharebe_yarali_bakimi_el_kitabi.pdf"
MILITARY_DOCUMENT = "askeri_egitim_kitabi.pdf"
SOURCE_DOCUMENTS = frozenset((TCCC_DOCUMENT, MILITARY_DOCUMENT))
SOURCE_TOPICS = MappingProxyType({
    TCCC_DOCUMENT: frozenset((
        "bleeding_control", "airway", "respiration", "circulation_shock",
        "medication", "burns_hypothermia", "evacuation_documentation",
        "head_eye_injury", "fracture_immobilization", "assessment_monitoring",
    )),
    MILITARY_DOCUMENT: frozenset((
        "weapon_safety", "ammunition", "handgun", "rifle", "machine_gun",
        "rocket_grenade", "marksmanship", "maintenance_malfunction",
        "anti_aircraft", "weapon_identification",
    )),
})
QUESTION_TYPES = frozenset(("direct", "scenario", "procedure", "comparison", "cause_effect"))
DIFFICULTIES = frozenset(("easy", "medium", "hard"))
RETRIEVAL_SCOPES = frozenset(("single_chunk", "same_page_multi_chunk", "multi_page", "none"))
LEXICAL_OVERLAPS = frozenset(("high", "medium", "low"))
ANSWERABILITIES = frozenset(("answerable", "unanswerable"))
CRITICALITIES = frozenset(("low", "medium", "high"))

_ID_PATTERNS = MappingProxyType({
    TCCC_DOCUMENT: re.compile(r"^tccc_\d{3}$"),
    MILITARY_DOCUMENT: re.compile(r"^askeri_\d{3}$"),
})


@dataclass(frozen=True)
class Issue:
    code: str
    question_id: str
    message: str


def _counter_issue(
    code: str, actual: Counter[str], expected: Counter[str], message: str
) -> list[Issue]:
    if actual == expected:
        return []
    return [Issue(code, "", f"{message}: expected {dict(expected)}, got {dict(actual)}")]


def validate_distribution(rows: Sequence[Mapping[str, str]]) -> list[Issue]:
    """Return all violations of the benchmark's fixed distribution contract."""
    issues: list[Issue] = []
    if len(rows) != 120:
        issues.append(Issue("row_count", "", f"expected 120 rows, got {len(rows)}"))

    sources = Counter(row.get("source_document", "") for row in rows)
    issues.extend(_counter_issue(
        "source_distribution", sources,
        Counter({TCCC_DOCUMENT: 60, MILITARY_DOCUMENT: 60}),
        "source distribution mismatch",
    ))
    issues.extend(_counter_issue(
        "answerability_distribution",
        Counter(row.get("answerability", "") for row in rows),
        Counter({"answerable": 108, "unanswerable": 12}),
        "answerability distribution mismatch",
    ))
    issues.extend(_counter_issue(
        "difficulty_distribution",
        Counter(row.get("difficulty", "") for row in rows),
        Counter({"easy": 40, "medium": 40, "hard": 40}),
        "difficulty distribution mismatch",
    ))
    issues.extend(_counter_issue(
        "lexical_overlap_distribution",
        Counter(row.get("lexical_overlap", "") for row in rows),
        Counter({"high": 40, "medium": 40, "low": 40}),
        "lexical overlap distribution mismatch",
    ))
    issues.extend(_counter_issue(
        "question_type_distribution",
        Counter(row.get("question_type", "") for row in rows),
        Counter({"direct": 30, "scenario": 42, "procedure": 24, "comparison": 12, "cause_effect": 12}),
        "question type distribution mismatch",
    ))
    issues.extend(_counter_issue(
        "retrieval_scope_distribution",
        Counter(row.get("retrieval_scope", "") for row in rows),
        Counter({"single_chunk": 66, "same_page_multi_chunk": 26, "multi_page": 16, "none": 12}),
        "retrieval scope distribution mismatch",
    ))

    seen_ids: set[str] = set()
    for row in rows:
        question_id = row.get("question_id", "")
        source = row.get("source_document", "")
        if question_id in seen_ids:
            issues.append(Issue("duplicate_question_id", question_id, "question_id is not unique"))
        seen_ids.add(question_id)
        pattern = _ID_PATTERNS.get(source)
        if pattern is None or not pattern.fullmatch(question_id):
            issues.append(Issue("id_format", question_id, "question_id does not match its source document"))
        if source not in SOURCE_DOCUMENTS:
            issues.append(Issue("source_document", question_id, f"unsupported source document: {source}"))
        for field, allowed in (
            ("question_type", QUESTION_TYPES), ("difficulty", DIFFICULTIES),
            ("retrieval_scope", RETRIEVAL_SCOPES), ("lexical_overlap", LEXICAL_OVERLAPS),
            ("answerability", ANSWERABILITIES),
        ):
            if row.get(field, "") not in allowed:
                issues.append(Issue("invalid_value", question_id, f"{field} is not allowed"))
        if source in SOURCE_TOPICS and row.get("topic", "") not in SOURCE_TOPICS[source]:
            issues.append(Issue("topic", question_id, "topic is not allowed for source document"))

    for source in SOURCE_DOCUMENTS:
        source_rows = [row for row in rows if row.get("source_document") == source]
        issues.extend(_counter_issue(
            "source_difficulty_distribution",
            Counter(row.get("difficulty", "") for row in source_rows),
            Counter({"easy": 20, "medium": 20, "hard": 20}),
            f"difficulty distribution mismatch for {source}",
        ))
        issues.extend(_counter_issue(
            "source_lexical_overlap_distribution",
            Counter(row.get("lexical_overlap", "") for row in source_rows),
            Counter({"high": 20, "medium": 20, "low": 20}),
            f"lexical overlap distribution mismatch for {source}",
        ))
        for topic, count in Counter(row.get("topic", "") for row in source_rows).items():
            if count > 10:
                issues.append(Issue("topic_limit", "", f"{source} has {count} items for topic {topic}"))
    return issues


def _parse_row_list(
    row: Mapping[str, str], field: str, issues: list[Issue]
) -> list[object] | None:
    try:
        return parse_json_list(row, field)
    except (KeyError, TypeError, json.JSONDecodeError, ValueError) as exc:
        issues.append(Issue("invalid_json", row.get("question_id", ""), f"{field}: {exc}"))
        return None


def _is_page_index(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def _page_in_range(page_index: int, pages: Sequence[str]) -> bool:
    return 1 <= page_index <= len(pages)


def _source_limited_abstention(answer: str) -> bool:
    normalized = normalize_text(answer)
    cites_documents = any(term in normalized for term in ("belge", "doküman", "dokuman", "kaynak"))
    expresses_absence = any(term in normalized for term in ("yer alm", "bulunm", "içerm", "icer", "yeterli bilgi yok"))
    return cites_documents and expresses_absence


def _legacy_answer_target(row: Mapping[str, str]) -> str:
    return row.get("answer", "") or row.get("reference_answer", "")


def _has_overlap_review_disposition(notes: str) -> bool:
    fields: dict[str, str] = {}
    for part in notes.split(";"):
        key, separator, value = part.strip().partition("=")
        if separator:
            fields[key.strip()] = value.strip()
    return (
        fields.get("reviewed") == "source"
        and bool(fields.get("overlap"))
        and bool(fields.get("reason"))
    )


def validate_rows(
    rows: Sequence[Mapping[str, str]],
    page_texts: Mapping[str, Sequence[str]],
    legacy_rows: Sequence[Mapping[str, str]],
    *,
    enforce_distribution: bool = True,
) -> list[Issue]:
    """Validate row schema, source evidence, and exact duplicate targets."""
    issues = validate_distribution(rows) if enforce_distribution else []
    seen_questions: dict[str, str] = {}
    seen_answers: dict[str, str] = {}
    legacy_answers = {
        normalize_text(_legacy_answer_target(row))
        for row in legacy_rows
        if _legacy_answer_target(row).strip()
    }
    always_required = (
        "question_id", "source_document", "question", "reference_answer",
        "required_facts", "topic", "question_type", "difficulty",
        "difficulty_reason", "retrieval_scope", "lexical_overlap", "criticality",
        "answerability", "notes", "acceptable_page_indices", "gold_evidence",
    )
    for row in rows:
        question_id = row.get("question_id", "")
        source = row.get("source_document", "")
        missing_fields = set(GOLD_FIELDS) - set(row)
        extra_fields = set(row) - set(GOLD_FIELDS)
        if missing_fields:
            issues.append(Issue("schema_fields", question_id, f"missing fields: {sorted(missing_fields)}"))
        if extra_fields:
            issues.append(Issue("schema_fields", question_id, f"unexpected fields: {sorted(extra_fields)}"))
        for field in always_required:
            if not str(row.get(field, "")).strip():
                issues.append(Issue("missing_field", question_id, f"{field} is required"))
        if row.get("criticality", "") not in CRITICALITIES:
            issues.append(Issue("invalid_value", question_id, "criticality is not allowed"))
        if source not in SOURCE_DOCUMENTS:
            issues.append(Issue("source_document", question_id, f"unsupported source document: {source}"))
        elif row.get("topic", "") not in SOURCE_TOPICS[source]:
            issues.append(Issue("topic", question_id, "topic is not allowed for source document"))
        question = normalize_text(row.get("question", ""))
        answer = normalize_text(row.get("reference_answer", ""))
        if question:
            previous = seen_questions.setdefault(question, question_id)
            if previous != question_id:
                issues.append(Issue("duplicate_question", question_id, f"duplicates question_id {previous}"))
        if answer:
            previous = seen_answers.setdefault(answer, question_id)
            if previous != question_id:
                issues.append(Issue("duplicate_reference_answer", question_id, f"duplicates question_id {previous}"))
            if answer in legacy_answers:
                issues.append(Issue("legacy_answer_reuse", question_id, "reference answer duplicates legacy target"))

        acceptable_pages = _parse_row_list(row, "acceptable_page_indices", issues)
        evidence = _parse_row_list(row, "gold_evidence", issues)
        required_facts = _parse_row_list(row, "required_facts", issues)
        if required_facts is not None:
            if not required_facts or any(not isinstance(fact, str) or not fact.strip() for fact in required_facts):
                issues.append(Issue("required_facts", question_id, "required_facts must contain non-empty strings"))

        answerability = row.get("answerability", "")
        if answerability == "unanswerable":
            if row.get("pdf_page_index", "") or row.get("printed_page_label", ""):
                issues.append(Issue("unanswerable_page", question_id, "unanswerable rows cannot declare a page"))
            if acceptable_pages is not None and acceptable_pages:
                issues.append(Issue("unanswerable_pages", question_id, "unanswerable rows require [] acceptable pages"))
            if evidence is not None and evidence:
                issues.append(Issue("unanswerable_evidence", question_id, "unanswerable rows require [] evidence"))
            if row.get("retrieval_scope", "") != "none":
                issues.append(Issue("unanswerable_scope", question_id, "unanswerable rows require retrieval_scope=none"))
            if not _source_limited_abstention(row.get("reference_answer", "")):
                issues.append(Issue("unanswerable_answer", question_id, "reference answer must limit abstention to supplied documents"))
            continue

        if answerability != "answerable":
            continue
        pages = page_texts.get(source)
        if pages is None:
            issues.append(Issue("source_pages_missing", question_id, f"no extracted pages for {source}"))
            continue
        primary_raw = row.get("pdf_page_index", "")
        try:
            primary_page = int(primary_raw)
        except (TypeError, ValueError):
            primary_page = 0
        if not _is_page_index(primary_page) or not _page_in_range(primary_page, pages):
            issues.append(Issue("primary_page", question_id, "pdf_page_index must be a physical source page"))
        parsed_acceptable: set[int] = set()
        if acceptable_pages is not None:
            if not acceptable_pages or any(not _is_page_index(page) for page in acceptable_pages):
                issues.append(Issue("acceptable_pages", question_id, "acceptable_page_indices must contain positive integer pages"))
            else:
                parsed_acceptable = set(acceptable_pages)
                for page in parsed_acceptable:
                    if not _page_in_range(page, pages):
                        issues.append(Issue("acceptable_page_range", question_id, f"page {page} is outside source range"))
                if primary_page not in parsed_acceptable:
                    issues.append(Issue("primary_page_accepted", question_id, "primary page must be acceptable"))
        if evidence is None:
            continue
        if not evidence:
            issues.append(Issue("missing_evidence", question_id, "answerable rows require evidence"))
            continue
        evidence_texts: list[str] = []
        for evidence_item in evidence:
            if not isinstance(evidence_item, Mapping):
                issues.append(Issue("evidence_schema", question_id, "each evidence item must be an object"))
                continue
            evidence_page = evidence_item.get("pdf_page_index")
            evidence_text = evidence_item.get("text")
            if not _is_page_index(evidence_page) or not isinstance(evidence_text, str) or not evidence_text.strip():
                issues.append(Issue("evidence_schema", question_id, "evidence requires positive pdf_page_index and text"))
                continue
            if evidence_page not in parsed_acceptable:
                issues.append(Issue("evidence_page", question_id, "evidence page must be acceptable"))
            if not _page_in_range(evidence_page, pages):
                issues.append(Issue("evidence_page_range", question_id, f"evidence page {evidence_page} is outside source range"))
                continue
            if normalize_text(evidence_text) not in normalize_text(pages[evidence_page - 1]):
                issues.append(Issue("evidence_not_found", question_id, "evidence text is absent from declared page"))
            evidence_texts.append(evidence_text)
        if required_facts is not None:
            evidence_text = normalize_text(" ".join(evidence_texts))
            reference_answer = normalize_text(row.get("reference_answer", ""))
            for fact in required_facts:
                if not isinstance(fact, str) or not fact.strip():
                    continue
                normalized_fact = normalize_text(fact)
                if normalized_fact not in evidence_text:
                    issues.append(Issue(
                        "required_fact_not_in_evidence", question_id,
                        f"required fact is absent from concatenated evidence: {fact}",
                    ))
                if normalized_fact not in reference_answer:
                    issues.append(Issue(
                        "required_fact_not_in_reference_answer", question_id,
                        f"required fact is absent from reference answer: {fact}",
                    ))
    for overlap in find_overlap_candidates(rows, legacy_rows):
        question_id = str(overlap.get("question_id", ""))
        row = next((row for row in rows if row.get("question_id", "") == question_id), None)
        if row is None or not _has_overlap_review_disposition(row.get("notes", "")):
            issues.append(Issue(
                "overlap_review_required", question_id,
                "overlap candidate requires reviewed=source, overlap=<disposition>, and non-empty reason",
            ))
    return issues


_TOKEN_RE = re.compile(r"[^\wçğıöşüÇĞİÖŞÜ]+", re.UNICODE)
_TURKISH_STOP_WORDS = frozenset((
    "acaba", "ama", "ancak", "bir", "bu", "da", "de", "gibi", "hangi", "için",
    "ile", "ise", "mi", "mı", "mu", "mü", "nasıl", "ne", "neden", "ve", "veya",
    "ya", "zaman", "yer", "olarak", "olan", "olarak", "göre", "kadar",
))
_TURKISH_SUFFIXES = (
    "lerinizden", "larımızdan", "lerimizden", "larınızdan", "lerinin", "larının",
    "leriniz", "larımız", "lerimiz", "larınız", "lerinin", "larının", "meden",
    "madan", "meye", "maya", "meyi", "mayı", "meli", "malı", "leri", "ları",
    "ler", "lar", "nin", "nın", "nun", "nün", "den", "dan", "ten", "tan",
    "de", "da", "te", "ta", "yi", "yı", "yu", "yü", "i", "ı", "u", "ü",
)


def _turkish_tokens(value: str) -> set[str]:
    tokens: set[str] = set()
    for token in _TOKEN_RE.sub(" ", normalize_text(value)).split():
        if token in _TURKISH_STOP_WORDS:
            continue
        for suffix in _TURKISH_SUFFIXES:
            if token.endswith(suffix) and len(token) > len(suffix) + 2:
                token = token[:-len(suffix)]
                break
        if token and token not in _TURKISH_STOP_WORDS:
            tokens.add(token)
    return tokens


def _overlap_score(left: Mapping[str, str], right: Mapping[str, str]) -> float:
    left_question = normalize_text(left.get("question", ""))
    right_question = normalize_text(right.get("question", ""))
    left_tokens = _turkish_tokens(left_question)
    right_tokens = _turkish_tokens(right_question)
    jaccard = len(left_tokens & right_tokens) / len(left_tokens | right_tokens) if left_tokens or right_tokens else 0.0
    question_sequence = SequenceMatcher(None, left_question, right_question).ratio()
    left_answer = normalize_text(_legacy_answer_target(left))
    right_answer = normalize_text(_legacy_answer_target(right))
    answer_sequence = SequenceMatcher(None, left_answer, right_answer).ratio() if left_answer and right_answer else 0.0
    return max(jaccard, question_sequence, answer_sequence)


def _overlap_neighbor(row: Mapping[str, str], score: float, fallback_id: str) -> dict[str, object]:
    return {
        "question_id": row.get("question_id", fallback_id),
        "question": row.get("question", ""),
        "score": round(score, 4),
    }


def find_overlap_candidates(
    rows: Sequence[Mapping[str, str]],
    legacy_rows: Sequence[Mapping[str, str]],
    *,
    threshold: float = 0.35,
) -> list[dict[str, object]]:
    """Return manual-review candidates from legacy and within-set neighbors."""
    candidates: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        legacy_neighbors = sorted(
            (_overlap_neighbor(other, _overlap_score(row, other), f"legacy_{other_index + 1:03d}")
             for other_index, other in enumerate(legacy_rows)),
            key=lambda neighbor: (-float(neighbor["score"]), str(neighbor["question"]), str(neighbor["question_id"])),
        )[:5]
        new_set_neighbors = sorted(
            (_overlap_neighbor(other, _overlap_score(row, other), f"new_{other_index + 1:03d}")
             for other_index, other in enumerate(rows) if other_index != index),
            key=lambda neighbor: (-float(neighbor["score"]), str(neighbor["question"]), str(neighbor["question_id"])),
        )[:5]
        legacy_best = float(legacy_neighbors[0]["score"]) if legacy_neighbors else 0.0
        new_best = float(new_set_neighbors[0]["score"]) if new_set_neighbors else 0.0
        max_score = max(legacy_best, new_best)
        if max_score < threshold:
            continue
        comparison_set = "legacy" if legacy_best >= new_best else "new_set"
        candidates.append({
            "question_id": row.get("question_id", ""),
            "question": row.get("question", ""),
            "max_score": round(max_score, 4),
            "comparison_set": comparison_set,
            "legacy_neighbors": legacy_neighbors,
            "new_set_neighbors": new_set_neighbors,
        })
    return candidates


def load_rows(path: Path) -> list[dict[str, str]]:
    """Load UTF-8 CSV rows without converting annotation values."""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return [dict(row) for row in reader]


def load_source_pages(pdf_dir: Path) -> dict[str, list[str]]:
    """Extract decoded physical PDF page text for the two allowed sources."""
    pages_by_source: dict[str, list[str]] = {}
    for source in sorted(SOURCE_DOCUMENTS):
        pdf_path = pdf_dir / source
        if not pdf_path.is_file():
            raise FileNotFoundError(f"missing source PDF: {pdf_path}")
        pages_by_source[source] = [
            _decode_pdf_text(page) for page in _extract_pages_text(str(pdf_path))
        ]
    return pages_by_source


def _markdown_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _markdown_counter(title: str, counts: Counter[str]) -> list[str]:
    lines = [f"### {title}", "", "| Value | Count |", "| --- | ---: |"]
    lines.extend(f"| {_markdown_cell(value)} | {count} |" for value, count in sorted(counts.items()))
    return lines


def build_validation_report(
    rows: Sequence[Mapping[str, str]], overlap_rows: Sequence[Mapping[str, object]]
) -> str:
    """Build the deterministic Markdown companion to the question projection."""
    lines = ["# RAG blind test validation report", "", "## Distribution", ""]
    for title, field in (
        ("Source documents", "source_document"),
        ("Answerability", "answerability"),
        ("Difficulty", "difficulty"),
        ("Lexical overlap", "lexical_overlap"),
        ("Question type", "question_type"),
        ("Retrieval scope", "retrieval_scope"),
    ):
        lines.extend(_markdown_counter(title, Counter(row.get(field, "") for row in rows)))
        lines.append("")
    lines.extend(["## Topic counts", "", "| Source document | Topic | Count |", "| --- | --- | ---: |"])
    topic_counts = Counter((row.get("source_document", ""), row.get("topic", "")) for row in rows)
    lines.extend(
        f"| {_markdown_cell(source)} | {_markdown_cell(topic)} | {count} |"
        for (source, topic), count in sorted(topic_counts.items())
    )
    answerable = [row for row in rows if row.get("answerability") == "answerable"]
    evidence_count = 0
    for row in answerable:
        try:
            evidence_count += len(parse_json_list(row, "gold_evidence"))
        except (KeyError, TypeError, json.JSONDecodeError, ValueError):
            pass
    lines.extend([
        "", "## Evidence checks", "", "| Metric | Count |", "| --- | ---: |",
        f"| Answerable items | {len(answerable)} |",
        f"| Unanswerable items | {sum(row.get('answerability') == 'unanswerable' for row in rows)} |",
        f"| Declared evidence excerpts | {evidence_count} |",
        "", "## Overlap candidates", "",
    ])
    notes_by_id = {row.get("question_id", ""): row.get("notes", "") for row in rows}
    if not overlap_rows:
        lines.append("No candidates met the review threshold.")
    else:
        lines.extend([
            "| Question ID | Max score | Comparison set | Review disposition |",
            "| --- | ---: | --- | --- |",
        ])
        for overlap in overlap_rows:
            question_id = str(overlap.get("question_id", ""))
            lines.append(
                f"| {_markdown_cell(question_id)} | {_markdown_cell(overlap.get('max_score', ''))} | "
                f"{_markdown_cell(overlap.get('comparison_set', ''))} | {_markdown_cell(notes_by_id.get(question_id, ''))} |"
            )
        for overlap in overlap_rows:
            question_id = str(overlap.get("question_id", ""))
            for label, field in (("legacy", "legacy_neighbors"), ("new-set", "new_set_neighbors")):
                lines.extend([
                    "", f"### {question_id} {label} neighbors", "",
                    "| Question ID | Question | Score |", "| --- | --- | ---: |",
                ])
                neighbors = overlap.get(field, [])
                if not isinstance(neighbors, Sequence) or isinstance(neighbors, (str, bytes)):
                    neighbors = []
                if not neighbors:
                    lines.append("| — | No neighbor | — |")
                    continue
                for neighbor in neighbors:
                    if not isinstance(neighbor, Mapping):
                        continue
                    lines.append(
                        f"| {_markdown_cell(neighbor.get('question_id', ''))} | "
                        f"{_markdown_cell(neighbor.get('question', ''))} | {_markdown_cell(neighbor.get('score', ''))} |"
                    )
    return "\n".join(lines) + "\n"


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def _questions_csv(rows: Sequence[Mapping[str, str]]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=("question_id", "question"))
    writer.writeheader()
    writer.writerows(build_questions_rows(rows))
    return output.getvalue()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate and project the RAG blind test set.")
    parser.add_argument("--gold", type=Path, default=Path("data/csv/rag_blind_test_gold.csv"))
    parser.add_argument("--legacy", type=Path, default=Path("data/qa/old/question_answers_100.csv"))
    parser.add_argument("--questions-output", type=Path, default=Path("data/csv/rag_blind_test_questions.csv"))
    parser.add_argument("--report-output", type=Path, default=Path("reports/rag_blind_test_validation.md"))
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        rows = load_rows(args.gold)
        legacy_rows = load_rows(args.legacy)
        pages = load_source_pages(Path("data/pdf"))
    except (OSError, ValueError, csv.Error) as exc:
        print(f"validation setup error: {exc}")
        return 2
    issues = validate_rows(rows, pages, legacy_rows)
    if issues:
        for issue in issues:
            question = f" [{issue.question_id}]" if issue.question_id else ""
            print(f"{issue.code}{question}: {issue.message}")
        return 1
    overlaps = find_overlap_candidates(rows, legacy_rows)
    if args.check_only:
        return 0
    _atomic_write_text(args.questions_output, _questions_csv(rows))
    _atomic_write_text(args.report_output, build_validation_report(rows, overlaps))
    return 0


def normalize_text(value: str) -> str:
    return " ".join(value.casefold().split())


def parse_json_list(row: Mapping[str, str], field: str) -> list[object]:
    value = json.loads(row[field])
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a JSON array")
    return value


def build_questions_rows(rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    return [
        {"question_id": row["question_id"], "question": row["question"]}
        for row in rows
    ]


if __name__ == "__main__":
    raise SystemExit(main())
