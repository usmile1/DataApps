#!/usr/bin/env bash
#
# Full benchmark pipeline: train → build vectors → test without RAG → test with RAG → evaluate
#
# Steps 1-2 use --clean to rebuild training data from scratch.
# Steps 3-4 append to the test DB (default behavior) so runs accumulate.
# To start test results fresh, delete test_results.db manually.
#
# Usage:
#   ./run_benchmark.sh                    # default: gpt-oss:20b
#   ./run_benchmark.sh llama3.2:latest    # override LLM model
#

set -euo pipefail

MODEL="${1:-gpt-oss:20b}"
TRAINING_DB="training.db"
VECTOR_DB="rag_vectors.db"
TEST_DB="test_results.db"

echo "============================================"
echo "  Benchmark Pipeline"
echo "  Model: $MODEL"
echo "============================================"
echo ""

# ── Step 1: Run pipeline on training data (sample_docs) ──────────
# --clean: wipes training.db so vectors are built from a clean slate.
echo "── Step 1/4: Training run (sample_docs) ─────────────────────"
echo ""
python main.py \
    --docs sample_docs \
    --db "$TRAINING_DB" \
    --slug "training" \
    --description "Training run for RAG vector building" \
    --model "$MODEL" \
    --clean
echo ""

# ── Step 2: Build RAG vector DB from training results ────────────
# build_vectors.py always recreates the vector DB.
echo "── Step 2/4: Building RAG vectors ───────────────────────────"
echo ""
python build_vectors.py \
    --db "$TRAINING_DB" \
    --vector-db "$VECTOR_DB"
echo ""

# ── Step 3: Test WITHOUT RAG (baseline) ──────────────────────────
# No --clean: appends to test_results.db.
echo "── Step 3/4: Test run WITHOUT RAG ───────────────────────────"
echo ""
python main.py \
    --docs test_docs \
    --db "$TEST_DB" \
    --slug "baseline-norag" \
    --description "Baseline, no RAG, model=$MODEL" \
    --model "$MODEL"
echo ""

# ── Step 4: Test WITH RAG ────────────────────────────────────────
# No --clean: appends to same test_results.db.
echo "── Step 4/4: Test run WITH RAG ──────────────────────────────"
echo ""
python main.py \
    --docs test_docs \
    --db "$TEST_DB" \
    --slug "baseline-rag" \
    --description "With RAG retrieval, model=$MODEL" \
    --model "$MODEL" \
    --rag "$VECTOR_DB"
echo ""

# ── Step 5: Evaluate ─────────────────────────────────────────────
echo "── Step 5/5: Evaluation ─────────────────────────────────────"
echo ""
python evaluate.py \
    --db "$TEST_DB" \
    --labels test_docs/test_labels.json \
    --details
echo ""
echo "============================================"
echo "  Benchmark complete"
echo ""
echo "  Training catalog:  $TRAINING_DB"
echo "  RAG vectors:       $VECTOR_DB"
echo "  Test results:      $TEST_DB"
echo ""
echo "  To re-evaluate with a different threshold:"
echo "    python evaluate.py --db $TEST_DB --labels test_docs/test_labels.json --threshold 0.60 --details"
echo "============================================"
