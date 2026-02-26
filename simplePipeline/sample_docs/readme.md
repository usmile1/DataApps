# DataProcessor v3.2

Internal data processing pipeline for quarterly report generation.

## Overview

This system ingests raw event data from our analytics platform, applies
transformation rules, and generates aggregated reports for the leadership team.

## Architecture

The pipeline consists of three stages:
1. **Ingestion** — Reads from the event stream and buffers into batches
2. **Transformation** — Applies business rules and computes derived metrics
3. **Output** — Writes aggregated results to the reporting database

## Running Locally

```bash
python main.py --config config.yaml --env development
```

## Team

Maintained by the Data Engineering team. See the wiki for on-call rotation.
