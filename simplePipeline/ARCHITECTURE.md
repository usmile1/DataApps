# SimplePipeline: System Architecture

## Overview

A multi-model data classification pipeline that scans documents for PII (Personally Identifiable Information) using a layered cost pyramid: cheap/fast methods first, expensive/accurate methods only when needed.

```
                    +-----------------+
                    |    Connector    |  Discovers & fetches documents
                    |  (filesystem)   |  Abstracted for swappable sources
                    +--------+--------+
                             |
                             v
                    +--------+--------+
                    | Feature Computer|  Computes document-level signals
                    |  (features-v1)  |  Used by all downstream layers
                    +--------+--------+
                             |
                    =========|=========  Pipeline (config-driven)
                    |        v        |
                    | +------+------+ |
                    | | Layer 1     | |  Regex pattern matching
                    | | Regex       | |  ~0.1ms per doc, always runs
                    | +------+------+ |
                    |        |        |
                    |   conf >= 0.9?  |  YES → entity resolved
                    |        | NO     |
                    |        v        |
                    | +------+------+ |
                    | | Layer 2     | |  SpaCy NER (local, ~30ms)
                    | | NER         | |  Adds contextual confidence
                    | +------+------+ |
                    |        |        |
                    |   conf >= 0.8?  |  YES → entity resolved
                    |        | NO     |
                    |        v        |
                    | +------+------+ |
                    | | Layer 3     | |  LLM via Ollama (~30-50s)
                    | | LLM         | |  Full doc + features + questions
                    | +------+------+ |
                    |                 |
                    ==================
                             |
                             v
                    +--------+--------+
                    |    Catalog      |  SQLite entity store
                    |  (entity graph) |  Entities → Features → Relationships
                    +--------+--------+
                             |
                             v
                    +--------+--------+
                    |    Report       |  CLI + HTML output
                    +-----------------+
```

## Core Design Principles

### 1. Cost Pyramid
Use the lightest-weight method possible without sacrificing accuracy. Regex resolves ~50% of entities instantly. NER resolves another ~30%. The LLM only handles the ~20% that are genuinely ambiguous. This is a common pattern in production data classification systems.

### 2. Everything is an Entity
Files, PII findings, and (eventually) directories are all entities in a unified graph. Each entity has features. Entities link to other entities via relationships (file *contains* SSN). This allows consistent querying, training data generation, and risk scoring at any level of the hierarchy.

```
Entity: employees.csv (type: file)
  Features: digit_density=0.29, entropy=5.16, header_pii_score=0.67
  └── Entity: 123-45-*789 (type: ssn)
        Features: confidence=0.95, line=2, classified_by=layer1
  └── Entity: alice@acme.com (type: email)
        Features: confidence=1.0, line=2, classified_by=layer1
```

### 3. Config-Driven Pipeline
Steps are declared in `pipeline/config.yaml` and loaded dynamically. Adding a new classification layer means writing a class and adding one line to config. `main.py` doesn't change.

```yaml
steps:
  - name: features
    class: features.step.FeatureComputationStep
  - name: regex
    class: pipeline.layer1_step.RegexClassificationStep
  - name: ner
    class: pipeline.layer2_step.NERClassificationStep
  - name: llm
    class: pipeline.layer3_step.LLMClassificationStep
    model: gpt-oss:20b
```

### 4. Context Bag Pattern
Each document gets a `PipelineContext` — a typed bag that steps read from and write to. Each step declares `requires` and `produces`, validated at pipeline construction time. This enables flexible step ordering and clear data flow without tight coupling.

### 5. Separation of Concerns
```
connector/
  models.py         ← Data shapes only. No logic, no I/O.
  base.py           ← Abstract interface. WHAT, not HOW.
  filesystem.py     ← Implementation. Only place that touches disk.

features/
  compute.py        ← Feature logic. No pipeline knowledge.
  step.py           ← Thin adapter to pipeline interface.

pipeline/
  context.py        ← Bag + step interface.
  pipeline.py       ← Orchestrator. Reads config, validates, runs.
  models.py         ← Entity dataclass used by all layers.
  layer1_regex.py   ← Detection logic. No pipeline knowledge.
  layer1_step.py    ← Thin adapter.
  layer2_ner.py     ← Detection logic.
  layer2_step.py    ← Thin adapter + gating logic.
  layer3_llm.py     ← Detection logic.
  layer3_step.py    ← Thin adapter + gating logic.

store/
  catalog.py        ← Unified entity graph in SQLite.

rag/
  embed.py          ← Ollama embedding client (nomic-embed-text).
  store.py          ← DIY vector store (SQLite + numpy cosine sim).

slm/
  generate_training_data.py  ← Synthetic training data generator (180 examples).
  training_data_train.jsonl  ← 153 chat-format training examples.
  training_data_val.jsonl    ← 27 chat-format validation examples.
  lora_config.yaml           ← QLoRA hyperparameters for mlx-lm.
  evaluate_slm.py            ← Inference + metrics (base vs fine-tuned).
  data/                      ← Symlinks for mlx-lm (train.jsonl, valid.jsonl).
  adapters/                  ← LoRA weight checkpoints (generated, not in git).

report/
  cli.py            ← Console report from catalog.
  html.py           ← HTML report from catalog.

test_docs/
  test_labels.json  ← Ground truth (111 entities, 89 PII, 22 not-PII).
  *.txt/py/csv/json ← 10 labeled test documents.

build_vectors.py    ← Populate RAG vector DB from catalog.
evaluate.py         ← Compare runs against ground truth labels.
run_benchmark.sh    ← Full benchmark: train → vectors → test → evaluate.
run_model_comparison.sh ← Multi-model comparison benchmark.
```

Each detection layer (regex, NER, LLM) has two files: the core logic (no framework dependencies) and a thin step adapter. The core logic can be tested independently, reused outside the pipeline, or swapped without touching the pipeline.

## Data Flow

```
1. Connector.discover()  →  List[DocumentMetadata]
2. For each document:
   a. Connector.fetch()  →  RawDocument
   b. PipelineContext created with "document" key
   c. FeatureComputer  →  context["features"]  (DocumentFeatures)
   d. RegexClassifier  →  context["entities"]   (List[Entity])
   e. NERClassifier    →  context["entities"]   (updated, if any < 0.9 conf)
   f. LLMClassifier    →  context["entities"]   (updated, if any < 0.8 conf)
   g. Store file entity + features + child entities in Catalog
   h. Store scan metrics (latency per layer)
3. Generate report from Catalog
```

## Confidence Scoring

Confidence (0.0–1.0) is the currency of the pipeline. It determines routing.

**Layer 1 (Regex) base confidence:**
| Signal | Score |
|--------|-------|
| Pattern match (e.g., SSN format) | 0.50 base |
| PII-like header names | +0.30 |
| Structured file (CSV/JSON) | +0.10 |
| Expected path context (hr/) | +0.05 |
| Luhn-valid credit card | +0.25 |
| Email with valid TLD | 0.90 base |

**Layer 2 (NER) adjustments:**
| Signal | Adjustment |
|--------|-----------|
| PERSON entity nearby | +0.15 |
| ORG entity, no PERSON | -0.10 |
| PERSON/MONEY near credit card | +0.10 |

**Layer 3 (LLM):**
Full document context. LLM provides its own confidence assessment. Can also discover entities missed by regex and NER (obfuscated PII, spelled-out numbers, etc.).

## Entity Store (Catalog) Schema

```
entities
  id TEXT PRIMARY KEY        -- e.g., "employees.csv" or "employees.csv:ssn:0"
  entity_type TEXT           -- "file", "ssn", "email", "person_name", etc.
  name TEXT                  -- display name (redacted for PII)
  created_at TEXT
  scan_id TEXT

entity_features
  entity_id TEXT FK          -- links to entities.id
  feature_name TEXT          -- e.g., "confidence", "digit_density"
  feature_value TEXT
  pipeline_version TEXT
  computed_at TEXT

entity_relationships
  parent_id TEXT FK          -- file entity
  child_id TEXT FK           -- PII entity
  relationship_type TEXT     -- "contains"

scan_metrics
  scan_id TEXT
  entity_id TEXT FK
  layer TEXT                 -- "features", "regex", "ner", "llm"
  latency_ms REAL
  entities_found INTEGER
  escalated BOOLEAN

runs
  scan_id TEXT PRIMARY KEY   -- links to entities.scan_id
  slug TEXT                  -- human-readable run name (e.g., "llama3.1-rag")
  description TEXT           -- what this run is testing
  pipeline_version TEXT      -- e.g., "features→regex→ner→llm(llama3.1:8b)+rag"
  config_snapshot TEXT       -- full YAML config as JSON
  dataset TEXT               -- source directory (e.g., "test_docs")
  created_at TEXT
```

## Risk Model (Context-Based)

The same PII type has different risk depending on business context:

| Location | PII Type | Risk | Reasoning |
|----------|----------|------|-----------|
| hr/w2_forms.csv | SSN | Expected | HR systems hold SSNs by design |
| public/shared_doc.txt | SSN | High | Public directory + SSN = data governance failure |
| employees.csv | Email | High | Root directory, no access controls implied |

Classification alone isn't enough — you need to understand *where* data lives and *who* can access it to properly assess risk.

## Planned Enhancements

### E1: Concurrent Processing & Queue Architecture

Currently the pipeline processes documents sequentially — while waiting 30-50s for the LLM on one document, every other document sits idle. The fix is to separate the pipeline into phases:

```
Phase 1 (fast, parallel):   All docs → regex + NER         (~200ms total)
Phase 2 (slow, queued):     Ambiguous docs → LLM queue     (concurrent workers)
```

Implementation approach:
- **Producer/consumer queue** between Phase 1 and Phase 2, using Python `asyncio.Queue` or `multiprocessing.Queue`
- Phase 1 runs synchronously (it's fast enough), produces a list of documents needing LLM
- Phase 2 workers pull from the queue and call the LLM concurrently
- This mirrors the Kafka-based fan-out pattern used in production data classification systems

### E2: Two-Tier Processing (Hot Path / Warm Path)

Optimize GPU cost by splitting into two processing tiers:

```
Hot path (real-time):
  regex → NER → LLM (only for ambiguous)
  Purpose: fast classification for incoming data
  SLA: seconds

Warm path (batch):
  ALL documents → LLM (full analysis)
  Purpose: comprehensive audit, catch what hot path missed
  SLA: hours/days
  Runs on reserved GPU instances to keep utilization high
```

The economics: reserved GPU instances are cheap per-hour but expensive if idle. The hot path has spiky demand. The warm path fills idle GPU time with batch re-processing — effectively free at the margin. If the warm path finds something the hot path missed, it triggers retroactive review.

This is analogous to how Stripe handles fraud detection: fast rules engine for real-time, batch ML pipeline re-scores overnight.

### E3: RAG-Enhanced Classification ✅ IMPLEMENTED

Retrieval-Augmented Generation adds few-shot context to the LLM layer by retrieving similar past classifications from a vector store.

**Architecture:**
```
Training catalog (sample_docs)
        |
        v
  build_vectors.py  →  Embed entity snippets + classification decisions
        |                using Ollama nomic-embed-text (768-dim vectors)
        v
  rag_vectors.db    →  DIY vector store (SQLite + numpy cosine similarity)
        |
        v
  Pipeline LLM step →  For each ambiguous entity:
        |                1. Embed the snippet
        |                2. Search for similar past classifications
        |                3. Include matches (with similarity scores) in LLM prompt
        |                4. LLM reasons with both document context AND historical examples
        v
  "Here's a similar case where '445-29-8831' near header
   'Soical Securty Nubmer' was classified as SSN with 0.95 confidence"
```

**Key design decisions:**
- **What we embed:** Entity snippet + classification decision (type, confidence, layer). Not full documents — too coarse.
- **Similarity threshold:** Only include matches above 0.50 cosine similarity. Weak matches (0.50-0.75) and strong matches (0.75+) are labeled as such so the LLM can weigh them accordingly.
- **DIY vector store over ChromaDB/FAISS:** At our scale (~200 vectors), brute-force cosine similarity is <1ms. Keeps the implementation transparent for learning.
- **Separate training/test databases:** Training knowledge base from sample_docs, test evaluation from test_docs. Never contaminate.

**Benchmark results (4-way model comparison):**

| Model | RAG | Precision | Recall | F1 |
|---|---|---|---|---|
| llama3.2 (3B) | No | 0.56 | 0.06 | 0.10 |
| llama3.2 (3B) | Yes | 0.07 | 0.01 | 0.02 |
| llama3.1 (8B) | No | 0.36 | 0.06 | 0.10 |
| **llama3.1 (8B)** | **Yes** | **0.40** | **0.38** | **0.39** |

RAG significantly improved recall for the 8B model (0.06 → 0.38) while the 3B model was too small to follow the complex prompt with RAG examples. The biggest gains were on multilingual docs (71% recall) and obfuscated SSNs (62%).

**Persistent blind spots** (0% recall across all configurations):
- Spelled-out numbers ("five five five, eight six seven...")
- Passwords and API keys (no regex patterns, no training examples)
- These are candidates for specialized SLMs (see E11).

### E4: Streaming Data Support

The pipeline currently processes files from a directory. Production data flows through streams:

```
Kafka topic → Connector → Pipeline → Store
```

The connector abstraction already supports this — write a `KafkaConnector` (or `KinesisConnector`, `PubSubConnector`) that yields messages as `RawDocument` objects. The pipeline processes them identically.

Key considerations for streaming:
- Backpressure handling when LLM is slow
- Exactly-once processing semantics
- Windowed aggregation for reporting (entities found per minute/hour)
- Dead letter queue for documents that fail classification

### E5: Discovery as a Pipeline Step

Currently discovery lives in `main.py`. For full configurability, it should be a pipeline step:

```yaml
pipeline:
  steps:
    - name: discover
      class: connector.discover_step.DiscoveryStep
      source: filesystem
      root_dir: sample_docs
    - name: features
      class: features.step.FeatureComputationStep
    ...
```

This requires the pipeline to understand fan-out — one discovery step produces N documents, and subsequent steps run for each. Two-phase model: source phase (produces documents) and process phase (runs per-document).

### E6: Pre-Classification Quarantine

Act on files before full classification completes:

```
Discovery → Quick risk score (file type + path + size heuristic)
  → High risk? Quarantine immediately, classify later
  → Low risk? Classify in normal pipeline
```

This addresses the scenario where a file in `public/` with PII-like headers should be locked down *before* spending 50 seconds on LLM analysis. The quarantine decision uses document-level features only (no content scanning needed).

### E7: Per-Entity Features

Currently features are document-level. For training and fine-grained classification, entities need their own features:

```
Entity: "123-45-6789" (SSN candidate)
  Features:
    context_window: "Customer SSN: 123-45-6789, verified on..."
    nearby_person: "Margaret Rivera"
    column_name: "ssn"
    structural_position: "row 3, column 2"
    character_entropy: 3.2
```

This enables training a per-entity classifier where the input features are the context around a specific match, not just document-level signals. The entity store schema already supports this — entity_features is a key-value store that works at any level.

### E8: Learned Vocabularies

Replace the hard-coded `pii_vocabulary.yaml` with embedding-based similarity:

```
Current:  header "ssn" matches vocabulary term "ssn" (exact/substring)
Future:   header "sin" → embedding → cosine similarity with "ssn" cluster → match
```

The curated vocabulary becomes training data for a small embedding model. Handles variations the vocabulary never anticipated: "sin" (Canadian), "nino" (UK), "ss#", "soc_sec_no", etc.

### E9: Adversarial / Obfuscation Detection

People trying to evade PII detection might:
- Spell out numbers: "one two three dash four five dash six seven eight nine"
- Use substitutions: "SSN: l23-4S-67B9" (letter/number swaps)
- Encode data: base64, ROT13, or custom encoding
- Split across fields: first 5 digits in column A, last 4 in column B

The LLM layer already catches some of this. Dedicated enhancements:
- Pre-processing step that normalizes common obfuscation patterns
- Entropy analysis to flag suspicious high-entropy sections
- Cross-field correlation for split-value detection
- Training data generation with synthetic obfuscated PII for model fine-tuning

### E10: Schema Evolution & Model Registry

**Schema evolution:** Migrate from Python dataclasses to protobuf or Avro with a schema registry (e.g., Confluent Schema Registry). This matters when the system becomes multi-service and multi-language. The registry enforces backward compatibility — you can't deploy a breaking schema change.

**Model registry:** Track versions of each classification layer. When a regex pattern changes or NER is retrained, store the version. Re-scan documents with the new version and compare results between versions (A/B testing for classification accuracy). The `pipeline_version` field in features is the seed of this.

### E11: Specialized Small Language Models (SLMs) — PARTIALLY IMPLEMENTED

Instead of one large generalist LLM, run multiple smaller specialist models in parallel, each tuned for a specific PII category:

```
                          ┌─────────────────────┐
                          │  Secret Scanner SLM  │  Passwords, API keys, tokens  ✅ TRAINED
   Ambiguous entities ──>─┤  NER-Enhanced SLM    │  Names, addresses in messy text
                          │  Format Recognizer   │  International IDs (CPF, NINO, SIN)
                          └─────────────────────┘
                                    │
                                    v
                          Merge + deduplicate results
```

**Economics:**
- Small models (1-3B) are 10-50x faster than a 20B model
- Running 3 SLMs in parallel takes the same wall-clock time as running 1
- Each SLM has a narrower job, so it can be smaller and still accurate
- Can run on smaller/cheaper GPU instances

**Current gap this addresses:** The benchmark shows 0% recall on passwords, API keys, and secrets across all configurations. A secret-scanning SLM trained on credential patterns (GitHub secret scanning datasets, TruffleHog patterns) would catch what general-purpose models miss entirely.

**Secret Scanner SLM — Training Results:**

Fine-tuned llama3.2 (3B, 4-bit quantized) via QLoRA on Apple Silicon (M3 Max) using mlx-lm.

Training data: 180 synthetic chat-format examples (153 train / 27 val) covering passwords, API keys, JWT secrets, tokens, private keys, connection strings, webhook secrets, plus negative examples (non-secret config values, paths, URLs, build artifacts).

Hyperparameters:
- LoRA rank 8 on q_proj + v_proj (1.3M trainable params / 3212M total = 0.041%)
- 400 iterations at batch_size 1, lr 2e-5, mask_prompt=true
- Training time: ~2 minutes on M3 Max, peak memory 2.6GB

Loss curve (monotonically decreasing, no overfitting):
```
Iter    Train Loss    Val Loss
  1         —          3.812
 50       1.206        1.217
100       0.968        0.941
200       0.631        0.671
300       0.487        0.501
400       0.430        0.419
```

Evaluation on 27 held-out validation examples (base model → fine-tuned):

| Metric     | Base Model | Fine-Tuned | Delta   |
|------------|-----------|------------|---------|
| Accuracy   | 0.0%      | 85.2%      | +85.2%  |
| Precision  | 0.0%      | 83.3%      | +83.3%  |
| Recall     | 0.0%      | 93.8%      | +93.8%  |
| F1         | 0.0%      | 88.2%      | +88.2%  |
| Parse Errs | 27        | 0          | -27     |

Base model produced 0 parseable JSON responses (27/27 parse errors). Fine-tuned model produced valid JSON for all 27 examples.

Per-type recall (fine-tuned): password 100%, api_key 100%, jwt_secret 100%, token 100%, private_key 100%, connection_string 100%. The 4 errors: 3 false positives (non-secrets flagged as secrets) and 1 missed webhook_secret. For a security tool, this bias toward false positives is acceptable.

**What remains (not yet implemented):**
- Pipeline integration: `pipeline/layer4_slm.py` step adapter, slotted before the general LLM
- GGUF export: `mlx_lm.fuse --export-gguf` for Ollama-compatible model
- Full benchmark: re-run 4-way comparison with SLM layer added
- Additional SLMs for other categories (NER-enhanced, format recognizer)

## Benchmarking & Evaluation

The pipeline supports reproducible benchmarking via run slugs and an evaluation harness.

**Run infrastructure:**
- `--slug` names each run for comparison (e.g., "llama3.1-rag", "baseline-norag")
- `--clean` opt-in to wipe database (default: append runs)
- `--rag VECTOR_DB` enables RAG retrieval for the LLM layer
- `--model` overrides LLM model from config
- Pipeline version derived from config (e.g., "features→regex→ner→llm(llama3.1:8b)+rag")
- Full config snapshot stored per run for reproducibility

**Evaluation:**
- Ground truth labels in `test_docs/test_labels.json` (111 entities, 89 PII, 22 not-PII)
- `evaluate.py` compares pipeline detections against labels
- Reports precision, recall, F1 per run with per-file breakdown
- Delta comparison across runs (FP/FN changes)
- Configurable confidence threshold

**Scripts:**
- `run_benchmark.sh` — train → vectors → test no-RAG → test RAG → evaluate
- `run_model_comparison.sh` — compare two models, each with/without RAG

## Tech Stack

- Python 3.10+
- SpaCy (en_core_web_sm) for NER
- Ollama for local LLM inference (llama3.1:8b, llama3.2, gpt-oss:20b)
- Ollama nomic-embed-text for RAG embeddings (768-dim vectors)
- NumPy for vector similarity (cosine similarity)
- SQLite for entity store + vector store
- mlx-lm for LoRA fine-tuning on Apple Silicon (QLoRA, 4-bit)
- PyYAML for configuration
- No frameworks — intentionally simple and readable
