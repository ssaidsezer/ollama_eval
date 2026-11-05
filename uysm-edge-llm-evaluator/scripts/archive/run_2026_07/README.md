# July 2026 experiment archive

These scripts are preserved to reproduce and explain the July 2026 run. Their
hardcoded experiment paths, host/IP assumptions, and result interpretations are
intentional provenance and must not be treated as current defaults.

| Script | Inputs/results | Produced artifacts |
|---|---|---|
| `benchmark_rag_retrieval.py` | live July RAG collections | `data/results/rag_methods_gemma_q4_20260722_150/retrieval_timing/summary.json` and timing rows |
| `generate_kv_ablation_report.py` | `data/results/kv_ablation_20260721_150/` | `output/pdf/uysm_kv_cache_ablation_150_balanced_v2_report.pdf` |
| `generate_final_experiment_reports.py` | July Classic Vector/BM25, Smart RAG, and speculative result folders | `output/pdf/uysm_classic_vector_rag_kv_cache_report.pdf`, `output/pdf/uysm_gemma_q4_rag_methods_report.pdf`, and `output/pdf/uysm_speculative_decoding_report.pdf` |

The report generators share result parsing with
`scripts/reporting/report_utils.py`; report-specific conclusions remain frozen
in this archive.
