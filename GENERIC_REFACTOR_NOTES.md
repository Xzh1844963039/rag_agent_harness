# Generic Agentic RAG Refactor Notes

## Why this refactor is needed

The uploaded code still contains several overfitting risks:

1. `AgenticRAG.rewrite_query()` hardcodes the current thesis topic, Qwen2.5-Math, QLoRA/DoRA, and math500.
2. `generate_answer()`, `audit_answer()`, `revise_answer()`, and `AgenticJudge` contain hard-coded rules for GSM8K, GPT-4, Claude, and model architecture claims.
3. `src/rag/query_thesis.py` and `src/eval/evaluate_retrieval.py` duplicate thesis-specific query-rewrite prompts.
4. `build_index.py` still contains page-range and section-title rules designed for this single thesis. This is acceptable for the current prototype, but it should be generalized before multi-paper ingestion.

## What this package changes

- Adds `configs/corpus_profile.yaml`.
- Makes `AgenticRAG` use corpus profile instead of hardcoded thesis facts.
- Makes answer generation and audit use generic entity-mismatch rules instead of specific math500/GSM8K/Qwen/GPT-4 rules.
- Adds `AgenticJudge(use_gold_hints=False)` as the default to avoid evaluation leakage.
- Keeps eval JSONL gold fields only for offline retrieval metrics and optional benchmark analysis.

## Files included

- `configs/corpus_profile.yaml`
- `src/agentic/agentic_rag.py`
- `src/eval/agentic_judge.py`
- `src/eval/evaluate_agentic_rag.py`
- `src/rag/query_thesis.py`

## Commands

Hard eval, online-style judge:

```powershell
python src\eval\evaluate_agentic_rag.py `
  --eval_file data\eval\eval_set_hard.jsonl `
  --output_dir outputs\agentic_rag_hard_generic `
  --run_name agentic_hard_eval_generic
```

Standard eval, online-style judge:

```powershell
python src\eval\evaluate_agentic_rag.py `
  --eval_file data\eval\eval_set.jsonl `
  --output_dir outputs\agentic_rag_generic `
  --run_name agentic_eval_generic
```

Optional offline benchmark judge with gold hints:

```powershell
python src\eval\evaluate_agentic_rag.py `
  --eval_file data\eval\eval_set_hard.jsonl `
  --output_dir outputs\agentic_rag_hard_goldhint `
  --run_name agentic_hard_eval_goldhint `
  --use_gold_hints
```

## Important next step

`build_index.py` should be generalized later:
- remove hardcoded page ranges,
- infer sections from headings more generically,
- add `doc_id`, `title`, `authors`, `year`, and `paper_id`,
- support multiple PDFs under `data/raw_docs/`.
