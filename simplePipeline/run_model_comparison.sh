#!/usr/bin/env bash
#
# Compare two LLM models: llama3.2 (3B) vs llama3.1:8b
# Runs each model with and without RAG on the test set.
#
# Produces 4 runs in test_results.db:
#   llama3.2-norag, llama3.2-rag, llama3.1-norag, llama3.1-rag
#
# Training/vectors are rebuilt once with llama3.2 (fast).
#

set -euo pipefail

TRAINING_DB="training.db"
VECTOR_DB="rag_vectors.db"
TEST_DB="test_results.db"

echo "============================================"
echo "  Model Comparison Benchmark"
echo "============================================"
echo ""

# ── Training + vectors (once, with llama3.2 for speed) ───────────
echo "── Training run (sample_docs, llama3.2) ─────────────────────"
echo ""
python main.py \
    --docs sample_docs \
    --db "$TRAINING_DB" \
    --slug "training" \
    --description "Training data for RAG vectors" \
    --model llama3.2:latest \
    --clean
echo ""

echo "── Building RAG vectors ─────────────────────────────────────"
echo ""
python build_vectors.py \
    --db "$TRAINING_DB" \
    --vector-db "$VECTOR_DB"
echo ""

# ── Clean test DB for fresh comparison ───────────────────────────
rm -f "$TEST_DB"

# ── llama3.2 (3B) without RAG ───────────────────────────────────
echo "── llama3.2 (3B) WITHOUT RAG ────────────────────────────────"
echo ""
python main.py \
    --docs test_docs \
    --db "$TEST_DB" \
    --slug "llama3.2-norag" \
    --description "Llama 3.2 3B, no RAG" \
    --model llama3.2:latest
echo ""

# ── llama3.2 (3B) with RAG ──────────────────────────────────────
echo "── llama3.2 (3B) WITH RAG ───────────────────────────────────"
echo ""
python main.py \
    --docs test_docs \
    --db "$TEST_DB" \
    --slug "llama3.2-rag" \
    --description "Llama 3.2 3B, with RAG" \
    --model llama3.2:latest \
    --rag "$VECTOR_DB"
echo ""

# ── llama3.1:8b without RAG ─────────────────────────────────────
echo "── llama3.1:8b WITHOUT RAG ──────────────────────────────────"
echo ""
python main.py \
    --docs test_docs \
    --db "$TEST_DB" \
    --slug "llama3.1-norag" \
    --description "Llama 3.1 8B, no RAG" \
    --model llama3.1:8b
echo ""

# ── llama3.1:8b with RAG ────────────────────────────────────────
echo "── llama3.1:8b WITH RAG ─────────────────────────────────────"
echo ""
python main.py \
    --docs test_docs \
    --db "$TEST_DB" \
    --slug "llama3.1-rag" \
    --description "Llama 3.1 8B, with RAG" \
    --model llama3.1:8b \
    --rag "$VECTOR_DB"
echo ""

# ── Evaluate all 4 runs ─────────────────────────────────────────
echo "── Evaluation ───────────────────────────────────────────────"
echo ""
python evaluate.py \
    --db "$TEST_DB" \
    --labels test_docs/test_labels.json \
    --details

echo ""
echo "============================================"
echo "  Comparison complete — 4 runs in $TEST_DB"
echo "============================================"
