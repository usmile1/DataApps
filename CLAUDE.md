# SimplePipeline: Data Classification Pipeline

## Project Overview

A hands-on learning project building a multi-model data classification pipeline. Explores ML pipeline patterns including multi-model routing, feature computation, confidence scoring, and fine-tuning specialized small language models.

## What We're Building

A local data classification pipeline that:
1. Scans sample documents containing planted PII (SSNs, emails, credit cards, phone numbers)
2. Runs them through a 3-layer multi-model classification pipeline (Regex → SpaCy NER → Claude API)
3. Routes documents through layers based on confidence scoring (cheap/fast first, expensive/accurate only when needed)
4. Stores all results and features in SQLite as a mini data catalog
5. Produces a CLI report of findings with per-layer latency and accuracy metrics

## Architecture

```
sample_docs/                    → Source documents with planted PII
connector/                      → Abstraction for reading from different sources
features/                       → Shared feature computation module (used by all layers)
pipeline/
  ├── layer1_regex.py          → Fast pattern matching (SSN, email, CC, phone regex)
  ├── layer2_ner.py            → SpaCy NER entity detection
  ├── layer3_llm.py            → Claude API validation for ambiguous cases
  └── router.py                → Confidence-based routing logic between layers
store/                          → SQLite-backed results store (mini feature store / data catalog)
report/                         → CLI reporting on findings, latency, accuracy
main.py                         → Pipeline orchestrator
```

## Key Design Principles

- **Multi-model routing**: Layer 1 (regex) is fast and cheap. Only escalate to Layer 2 (NER) when regex confidence is low. Only escalate to Layer 3 (LLM) when NER confidence is also low. Use the lightest-weight method possible without sacrificing accuracy.
- **Shared feature computation**: One module computes features (file type, text length, entropy, header analysis, etc.) used by all layers. This is the "feature store" pattern — same computation logic everywhere.
- **Everything gets logged**: Every classification decision, confidence score, layer used, and latency measurement goes to SQLite. This is the data catalog / audit trail.
- **Connector abstraction**: The file reader is behind an interface so it could be swapped for S3, Google Drive, etc.

## Tech Stack

- Python 3.10+
- SpaCy with `en_core_web_sm` model for NER
- Anthropic Python SDK for Claude API (Layer 3 validation)
- SQLite (stdlib) for results storage
- No frameworks — keep it simple and readable

## Build Order

Follow PLAN.md for the phased build approach. Build each phase completely and test it before moving to the next.

## IMPORTANT: Collaborative Learning Mode

This is a learning project. Follow these rules:

1. **Work through steps one at a time.** Do one sub-step, explain what it does, run it and see the output.
2. **Explain WHY, not just WHAT.** Connect design choices to production ML pipeline patterns.
3. **Make it runnable at every step.** No long stretches of coding before the first test.
4. **Pause for exploration.** After each layer is built, experiment — change thresholds, add edge cases, break things intentionally — before moving on.
5. **Call out the tradeoffs.** When making a design choice, explain what alternatives exist and why we're choosing this one.

## Style Guidelines

- Clear, readable Python with type hints
- Docstrings on all public classes and functions
- Each module should be runnable independently for testing
- Print timing/latency for every classification to make the cost pyramid visible
- Use dataclasses for structured data (Document, Feature, ClassificationResult, etc.)

## Environment Setup

```bash
pip install spacy anthropic
python -m spacy download en_core_web_sm
```

The Anthropic API key should be set as ANTHROPIC_API_KEY environment variable.
