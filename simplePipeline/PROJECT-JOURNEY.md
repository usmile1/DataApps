# Building a Multi-Model Classification Pipeline from Scratch

**What:** A local data classification system that scans documents for PII using a layered pipeline — regex, NER, a fine-tuned small language model, and a general LLM — each layer progressively more capable and expensive.

---

## Architecture

```
                         ┌─────────────────────┐
                         │   Document Source    │
                         │  (local files, S3,   │
                         │   Google Drive...)   │
                         └─────────┬───────────┘
                                   │
                         ┌─────────▼───────────┐
                         │  Connector Layer     │
                         │  discover() + fetch()│
                         └─────────┬───────────┘
                                   │
                         ┌─────────▼───────────┐
                         │  Feature Computation │
                         │  file type, entropy, │
                         │  headers, PII vocab  │
                         └─────────┬───────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              │                    │                     │
    ┌─────────▼──────────┐        │                     │
    │  Layer 1: Regex    │        │                     │
    │  <1ms per doc      │        │                     │
    │  SSN, email, CC,   │      features              features
    │  phone patterns    │      shared                shared
    │                    │      across                across
    └─────────┬──────────┘      all layers            all layers
              │                    │                     │
    ┌─────────▼──────────┐        │                     │
    │  Layer 2: NER      │        │                     │
    │  ~35ms per doc     │◄───────┘                     │
    │  SpaCy en_core_web │                              │
    │  person names,     │                              │
    │  organizations     │                              │
    └─────────┬──────────┘                              │
              │                                         │
    ┌─────────▼──────────┐                              │
    │  Layer 4: SLM      │                              │
    │  ~500ms per line   │◄─────────────────────────────┘
    │  secret-scanner    │
    │  (Llama 3.2 3B)    │   ┌──────────────────┐
    │  passwords, API    │   │  RAG Vector Store │
    │  keys, tokens      │   │  past classifi-   │
    │  ALWAYS RUNS       │   │  cations as       │
    └─────────┬──────────┘   │  few-shot context │
              │              └────────┬─────────┘
    ┌─────────▼──────────┐           │
    │  Layer 3: LLM      │◄──────────┘
    │  ~30-50s per doc   │
    │  gpt-oss:20b       │
    │  GATED: only when  │
    │  confidence < 0.80 │
    └─────────┬──────────┘
              │
    ┌─────────▼──────────┐
    │  SQLite Catalog    │
    │  entity graph,     │    ┌──────────────────┐
    │  features, scan    │───▶│  CLI + HTML      │
    │  metrics, runs     │    │  Reports         │
    └────────────────────┘    └──────────────────┘
```

## Benchmark Results

10 test documents, 89 labeled PII entities, 22 labeled non-PII (hard negatives).

| Metric | Without SLM | With SLM | Delta |
|--------|:-----------:|:--------:|:-----:|
| **Precision** | 0.11 | 0.36 | +0.25 |
| **Recall** | 0.01 | 0.33 | **+0.31** |
| **F1** | 0.02 | 0.34 | **+0.32** |
| True Positives | 1 | 29 | +28 |
| False Negatives | 88 | 60 | -28 |
| False Positives | 8 | 52 | +44 |

**Per-file recall improvement:**

| File | Without SLM | With SLM |
|------|:----------:|:--------:|
| test_source_code.py | 11% | **56%** |
| test_misspelled.txt | 0% | **50%** |
| test_multilingual.txt | 0% | **50%** |
| test_obfuscated_ssn.txt | 0% | **25%** |
| test_password_dump.txt | 0% | **22%** |
| test_slack_export.json | 0% | **17%** |
| test_medical_notes.txt | 0% | **9%** |

**New entity types discovered by SLM:** `api_key` (5), `password` (3), `jwt_secret` (1), `private_key` (1), `client_secret` (1) — all categories with 0% recall before.

---

**The cost pyramid.** The core architectural bet: don't send everything to an LLM. Regex runs in <1ms and catches SSNs, emails, credit cards, phones. SpaCy NER adds ~35ms for person names and contextual entities. Only documents with ambiguous findings escalate to the LLM at 30-50 seconds. Most documents never need the expensive layer.

**The data model debate.** Early on, we drew a hard line between *models* (declared facts about identity and location) and *features* (derived observations from content analysis). Everything became an entity — files, PII findings, directories — with features attached. This unified graph in SQLite made querying, training data generation, and risk scoring work at any level without special cases.

**Config-driven pipeline with a "bag" pattern.** Steps declare what they require and produce. The pipeline validates the dependency graph before running. Steps are loaded dynamically from a YAML config — swap a model, reorder layers, add a new step, all without touching orchestration code. We called the orchestrator "preschool-level simple" on purpose.

**RAG changed the LLM's accuracy.** When we added retrieval-augmented generation — embedding past classifications and injecting similar examples into the LLM prompt — the model stopped second-guessing obvious patterns. Prior classifications became few-shot context, turning the pipeline's history into a feedback loop.

**The secrets gap exposed a structural limit.** Benchmarking revealed that regex, NER, and the general LLM had **0% recall** on passwords, API keys, and tokens. No surprise — there's no regex pattern for an arbitrary password. The answer wasn't a bigger model; it was a *specialized* one.

**Fine-tuning a 3B model in 2 minutes.** We generated 180 synthetic training examples (secrets in context with hard negatives), ran QLoRA fine-tuning on a Llama 3.2 3B model using Apple Silicon — 1.3M trainable parameters, 2 minutes of training, 88% F1. The base model scored 0% (couldn't even parse the output format). We exported to GGUF, loaded it into Ollama as `secret-scanner`, and wired it into the pipeline as a new layer.

**Result: recall went from 1% to 33%.** The SLM found 28 true positives that every other layer missed — passwords, API keys, JWT secrets, private keys. At ~500ms per candidate line, it sits comfortably in the cost pyramid between NER and the general LLM. The natural next step: a fleet of specialized SLMs routed by document features, with the general LLM as a true last resort.

---

## Tools, Models & Libraries

| Category | Tool / Model | What we used it for |
|----------|-------------|-------------------|
| **Pipeline orchestration** | Python 3.10, PyYAML | Config-driven step loading, dynamic class imports |
| **Layer 1 — Regex** | Python `re` stdlib | Fast pattern matching for SSNs, emails, credit cards, phones |
| **Layer 2 — NER** | SpaCy `en_core_web_sm` | Named entity recognition (person names, orgs, locations) |
| **Layer 3 — General LLM** | Ollama + `gpt-oss:20b` / `llama3.2:latest` | Ambiguous entity validation, open-ended PII discovery |
| **Layer 4 — Secret scanner SLM** | Ollama + `secret-scanner` (Llama 3.2 3B, Q8_0) | Password, API key, token, private key detection |
| **SLM fine-tuning** | `mlx-lm` 0.30.7 (QLoRA) | LoRA rank-8 fine-tune on Apple Silicon M3 Max |
| **SLM base model** | `mlx-community/Llama-3.2-3B-Instruct-4bit` | 2GB quantized base for fine-tuning |
| **Model export** | `mlx-lm` fuse + `llama.cpp` convert | MLX adapters → fused model → GGUF for Ollama serving |
| **RAG embeddings** | Ollama + `nomic-embed-text` | Vector embeddings for similar-classification retrieval |
| **Vector store** | Custom SQLite-backed store | Cosine similarity search over past classification vectors |
| **Data catalog** | SQLite (stdlib) | Entity graph, features, scan metrics, run history |
| **Benchmarking** | Custom `evaluate.py` | Precision/recall/F1 against labeled test set, per-file breakdown |
| **Reporting** | Custom CLI + HTML report | Per-document findings, layer latency, risk assessment |
| **Version control** | Git + GitHub | Private repo `usmile1/DataApps` |
| **AI assistant** | Claude Code (Claude Opus) | Architecture design, implementation, code review |
