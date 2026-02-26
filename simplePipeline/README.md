# SimplePipeline

A multi-model data classification pipeline that scans documents for PII (Personally Identifiable Information) using a layered architecture. Each layer is progressively more capable and expensive — cheap/fast methods handle the easy cases, and only ambiguous findings escalate to more powerful (and slower) models.

> **Note:** This is a learning project, not a production system. Built to explore multi-model routing, confidence scoring, SLM fine-tuning, and ML pipeline architecture patterns hands-on.

## Architecture

```
  Documents  →  Connector  →  Feature Computation
                                      │
                              ┌───────▼────────┐
                              │ Layer 1: Regex  │  <1ms
                              │ SSN, email, CC, │
                              │ phone patterns  │
                              └───────┬────────┘
                              ┌───────▼────────┐
                              │ Layer 2: NER    │  ~35ms
                              │ SpaCy names,    │
                              │ organizations   │
                              └───────┬────────┘
                              ┌───────▼────────┐
                              │ Layer 4: SLM    │  ~500ms/line
                              │ Fine-tuned 3B   │
                              │ secret scanner  │
                              └───────┬────────┘
                              ┌───────▼────────┐
                              │ Layer 3: LLM    │  ~30-50s
                              │ General model   │
                              │ (gated: <0.80)  │
                              └───────┬────────┘
                                      │
                              SQLite Catalog → Reports
```

## Key Design Patterns

- **Cost pyramid**: Regex resolves most entities in <1ms. Only ambiguous cases escalate to NER, then LLM. The SLM always runs (discovery-focused) but is cheap at ~500ms per candidate line.
- **Shared feature computation**: One module computes features (file type, entropy, headers, PII vocabulary scoring) used by all layers.
- **Config-driven pipeline**: Steps are declared in YAML with dependency injection. Swap models, reorder layers, or add new steps without touching orchestration code.
- **Specialized SLM**: A fine-tuned Llama 3.2 3B model detects secrets (passwords, API keys, tokens) that regex/NER are structurally unable to catch. Trained with QLoRA in 2 minutes on synthetic data, achieving 88% F1.
- **RAG-augmented LLM**: Past classifications are embedded and retrieved as few-shot context for the general LLM layer.
- **Unified entity model**: Everything is an entity (files, PII findings, directories) with features attached, stored in a SQLite graph.

## Benchmark Results

Tested on 10 documents with 89 labeled PII entities:

| Metric | Without SLM | With SLM | Delta |
|--------|:-----------:|:--------:|:-----:|
| Precision | 0.11 | 0.36 | +0.25 |
| Recall | 0.01 | 0.33 | +0.31 |
| F1 | 0.02 | 0.34 | +0.32 |

The SLM found 28 true positives that every other layer missed — passwords, API keys, JWT secrets, and private keys.

## Quick Start

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai) running locally

### Install

```bash
cd simplePipeline
pip install spacy pyyaml requests
python -m spacy download en_core_web_sm
```

### Run

```bash
# Scan sample documents
python main.py

# Scan specific directory with a named run
python main.py --docs test_docs --db results.db --slug my-run --description "Test run"

# Use a different LLM model
python main.py --model llama3.2:latest

# Evaluate against ground truth labels
python evaluate.py --db results.db --labels test_docs/test_labels.json --details
```

### SLM Setup

The secret-scanner SLM requires a fine-tuned model in Ollama. See `slm/Modelfile` for the model definition. If you have the GGUF weights:

```bash
cd slm
ollama create secret-scanner -f Modelfile
```

## Project Structure

```
connector/          Abstraction for reading from different sources
features/           Shared feature computation (used by all layers)
pipeline/
  layer1_regex.py   Fast pattern matching
  layer2_ner.py     SpaCy NER entity detection
  layer_slm.py      Fine-tuned secret detection SLM
  layer3_llm.py     General LLM validation (Ollama)
  config.yaml       Pipeline step configuration
store/              SQLite-backed entity catalog
report/             CLI and HTML reporting
rag/                Vector store for retrieval-augmented classification
slm/                Training data generation, fine-tuning, evaluation
main.py             Pipeline orchestrator
evaluate.py         Benchmark against labeled test data
```

## Further Reading

- [PROJECT-JOURNEY.md](PROJECT-JOURNEY.md) — narrative summary of the build process and key decisions
- [ARCHITECTURE.md](ARCHITECTURE.md) — system design and planned enhancements
- [PLAN.md](PLAN.md) — original phased build plan
