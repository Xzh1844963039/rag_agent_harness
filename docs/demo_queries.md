#docs/demo_queries.md
# Demo Queries

This file contains three recommended demo queries for project presentation, README screenshots, and interview explanation.

## Demo 1: Method Understanding

### Command

```powershell
python src\rag\query_thesis.py "What is the Teacher-Student-Controller framework?"
```

### What this demo shows

This demo shows that the system can retrieve method-related chunks and explain the main framework in a grounded way.

### Talking point

This is the basic document QA capability. It proves that the system can locate and summarize the core method, but it is still not the strongest part of the project.

## Demo 2: Cross-section Reasoning

### Command

```powershell
python src\rag\query_thesis.py "How is the Teacher-Student-Controller framework connected to the final math500 strict improvements?"
```

### What this demo shows

This demo shows cross-section reasoning. The answer should connect method evidence with result evidence instead of only summarizing one section.

### Talking point

This is where Agentic RAG is more useful than simple top-k retrieval. The system needs to combine method, experiment, and result evidence.

## Demo 3: Unsupported Claim Control

### Command

```powershell
python src\rag\query_thesis.py "Does the thesis prove that local CoT repair works for all LLMs and all reasoning benchmarks?"
```

### What this demo shows

This demo shows hallucination control. The answer should reject the overgeneralized claim and explain that the available evidence only supports the specific experimental setting in the thesis.

### Expected answer style

The answer should say something close to:

```text
The thesis does not prove that local CoT repair works for all LLMs and all reasoning benchmarks. The retrieved sources support the method and results under the thesis's own experimental setting, but they do not establish universal generalization.
```

### Talking point

This is the best demo for interview or project presentation because it shows that the system is not only retrieving evidence, but also checking whether a claim is actually supported.

## Optional Debug Command

To inspect the rewritten retrieval query:

```powershell
python src\rag\query_thesis.py "Does the thesis prove that local CoT repair works for all LLMs and all reasoning benchmarks?" --show_rewritten_query
```

## Recommended Evaluation Commands

```powershell
python src\eval\evaluate_retrieval.py `
  --eval_file data\eval\eval_set.jsonl `
  --output_dir outputs\agentic_rag
```

```powershell
python src\eval\evaluate_agentic_rag.py `
  --eval_file data\eval\eval_set_hard.jsonl `
  --output_dir outputs\agentic_rag_hard_generic `
  --run_name agentic_rag_hard_generic_v3
```
