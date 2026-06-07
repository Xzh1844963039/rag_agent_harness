#VERSION.md

# Project Freeze: Corpus-aware Agentic RAG v0.3

This is the current stable display version of `rag_agent_harness`.

## Frozen version name

`v0.3-corpus-aware-agentic-rag`

## Current project status

The project has been upgraded from a thesis-specific RAG demo to a corpus-aware Agentic RAG framework.

Current supported workflow:

```text
PDF / parsed pages
-> generic multi-document index construction
-> corpus-aware query rewrite
-> retrieval
-> evidence sufficiency check
-> retry retrieval when evidence is weak
-> neighbor context expansion
-> grounded answer generation
-> citation audit
-> answer revision
-> retrieval and hard Agentic RAG evaluation
```

## Current evaluation snapshot

Retrieval evaluation on `data/eval/eval_set.jsonl`:

```text
page_hit@8      = 0.9583
section_hit@8   = 0.8333
keyword_recall@8 = 0.7368
```

Hard Agentic RAG evaluation on `data/eval/eval_set_hard.jsonl`:

```text
avg_overall_score                     = 4.8667 / 5
advanced_judge.avg_overall_score       = 4.8667 / 5
advanced_judge.total_critical_mismatch = 0
verdict_counts                         = 14 excellent, 1 partial
```

## Recommended freeze command

Run this after replacing the files and confirming everything still works:

```powershell
git add .
git commit -m "Freeze corpus-aware Agentic RAG demo v0.3"
git tag v0.3-corpus-aware-agentic-rag
```

If you do not want to create a tag yet, just make the commit.
