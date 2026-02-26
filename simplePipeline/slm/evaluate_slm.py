"""
Evaluate the fine-tuned SLM on held-out validation examples.

Runs inference on every validation example with both:
  1. Base model (no adapters) — baseline
  2. Base model + LoRA adapters — fine-tuned

Computes precision, recall, F1 for is_secret classification,
broken down by secret type.

Usage:
    python simplePipeline/slm/evaluate_slm.py \
        --model mlx-community/Llama-3.2-3B-Instruct-4bit \
        --adapter-path simplePipeline/slm/adapters \
        --data simplePipeline/slm/data/valid.jsonl
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import mlx.core as mx
from mlx_lm import generate, load


@dataclass
class Prediction:
    """A single model prediction paired with ground truth."""
    expected_is_secret: bool
    expected_type: str
    predicted_is_secret: Optional[bool] = None
    predicted_type: Optional[str] = None
    predicted_confidence: Optional[float] = None
    parse_error: bool = False
    raw_response: str = ""
    latency_ms: float = 0.0


@dataclass
class Metrics:
    """Aggregate classification metrics."""
    tp: int = 0  # true positives (correctly predicted secret)
    fp: int = 0  # false positives (predicted secret, actually not)
    tn: int = 0  # true negatives (correctly predicted not secret)
    fn: int = 0  # false negatives (predicted not secret, actually is)
    parse_errors: int = 0
    total_latency_ms: float = 0.0

    @property
    def total(self) -> int:
        return self.tp + self.fp + self.tn + self.fn + self.parse_errors

    @property
    def accuracy(self) -> float:
        correct = self.tp + self.tn
        total = self.total
        return correct / total if total > 0 else 0.0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / self.total if self.total > 0 else 0.0


def load_examples(data_path: str) -> list[dict]:
    """Load chat-format JSONL examples."""
    examples = []
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def parse_response(raw: str) -> dict:
    """Parse JSON response from model output.

    The model should return JSON like:
        {"is_secret": true, "type": "password", "confidence": 0.95, "reasoning": "..."}

    Handles common failure modes:
    - Extra text before/after JSON
    - Missing fields
    """
    raw = raw.strip()

    # Try to extract JSON from the response
    # Look for the first { and last }
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(raw[start:end + 1])
        except json.JSONDecodeError:
            pass

    # Try parsing the whole thing
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def run_inference(
    model,
    tokenizer,
    examples: list[dict],
    label: str,
    max_tokens: int = 200,
) -> list[Prediction]:
    """Run inference on all examples and return predictions."""
    predictions = []
    total = len(examples)

    for i, example in enumerate(examples):
        messages = example["messages"]
        user_msg = messages[0]  # the user prompt
        ground_truth = json.loads(messages[1]["content"])  # expected response

        # Build prompt using chat template
        prompt = tokenizer.apply_chat_template(
            [user_msg],
            tokenize=False,
            add_generation_prompt=True,
        )

        # Generate
        start_time = time.perf_counter()
        response = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            verbose=False,
        )
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Parse response
        parsed = parse_response(response)

        pred = Prediction(
            expected_is_secret=ground_truth["is_secret"],
            expected_type=ground_truth.get("type", "none"),
            raw_response=response,
            latency_ms=elapsed_ms,
        )

        if "is_secret" not in parsed:
            pred.parse_error = True
        else:
            pred.predicted_is_secret = parsed["is_secret"]
            pred.predicted_type = parsed.get("type", "none")
            pred.predicted_confidence = parsed.get("confidence")

        predictions.append(pred)
        status = "OK" if not pred.parse_error else "PARSE_ERROR"
        print(f"  [{label}] {i+1}/{total} — {status} ({elapsed_ms:.0f}ms)", flush=True)

    return predictions


def compute_metrics(predictions: list[Prediction]) -> Metrics:
    """Compute aggregate metrics from predictions."""
    m = Metrics()
    for p in predictions:
        m.total_latency_ms += p.latency_ms

        if p.parse_error:
            m.parse_errors += 1
            # Count parse errors as wrong: if expected secret, it's a FN; if not, it's a FP
            if p.expected_is_secret:
                m.fn += 1
            else:
                m.fp += 1
            continue

        if p.expected_is_secret and p.predicted_is_secret:
            m.tp += 1
        elif p.expected_is_secret and not p.predicted_is_secret:
            m.fn += 1
        elif not p.expected_is_secret and p.predicted_is_secret:
            m.fp += 1
        else:
            m.tn += 1

    return m


def compute_per_type_metrics(predictions: list[Prediction]) -> dict[str, Metrics]:
    """Compute metrics broken down by expected secret type."""
    by_type: dict[str, list[Prediction]] = defaultdict(list)
    for p in predictions:
        by_type[p.expected_type].append(p)

    return {t: compute_metrics(preds) for t, preds in sorted(by_type.items())}


def print_comparison(base_metrics: Metrics, ft_metrics: Metrics, base_preds: list[Prediction], ft_preds: list[Prediction]):
    """Print a comparison table between base and fine-tuned models."""
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)

    # Overall comparison
    print(f"\n{'Metric':<20} {'Base Model':>15} {'Fine-Tuned':>15} {'Delta':>10}")
    print("-" * 60)
    for name, base_val, ft_val in [
        ("Accuracy", base_metrics.accuracy, ft_metrics.accuracy),
        ("Precision", base_metrics.precision, ft_metrics.precision),
        ("Recall", base_metrics.recall, ft_metrics.recall),
        ("F1", base_metrics.f1, ft_metrics.f1),
    ]:
        delta = ft_val - base_val
        sign = "+" if delta >= 0 else ""
        print(f"{name:<20} {base_val:>14.1%} {ft_val:>14.1%} {sign}{delta:>8.1%}")

    print(f"{'Parse Errors':<20} {base_metrics.parse_errors:>15} {ft_metrics.parse_errors:>15}")
    print(f"{'Avg Latency':<20} {base_metrics.avg_latency_ms:>12.0f}ms {ft_metrics.avg_latency_ms:>12.0f}ms")

    # Confusion matrix for fine-tuned
    print(f"\nFine-tuned confusion matrix:")
    print(f"  TP={ft_metrics.tp}  FP={ft_metrics.fp}")
    print(f"  FN={ft_metrics.fn}  TN={ft_metrics.tn}")

    # Per-type breakdown for fine-tuned
    ft_by_type = compute_per_type_metrics(ft_preds)
    base_by_type = compute_per_type_metrics(base_preds)

    print(f"\n{'Type':<20} {'Base Recall':>12} {'FT Recall':>12} {'FT Prec':>10} {'FT F1':>10} {'Count':>6}")
    print("-" * 70)
    for t, ft_m in ft_by_type.items():
        base_m = base_by_type.get(t, Metrics())
        count = ft_m.total
        # For single-type slices, recall = (tp / (tp+fn))
        print(f"{t:<20} {base_m.recall:>11.1%} {ft_m.recall:>11.1%} {ft_m.precision:>9.1%} {ft_m.f1:>9.1%} {count:>6}")

    # Show misclassifications
    misses = [p for p in ft_preds if p.parse_error or p.predicted_is_secret != p.expected_is_secret]
    if misses:
        print(f"\nFine-tuned misclassifications ({len(misses)}):")
        for p in misses:
            label = "SECRET" if p.expected_is_secret else "NOT SECRET"
            if p.parse_error:
                print(f"  [{label}] type={p.expected_type} — PARSE ERROR")
                print(f"    Raw: {p.raw_response[:120]}")
            else:
                pred_label = "secret" if p.predicted_is_secret else "not secret"
                print(f"  [{label}] type={p.expected_type} — predicted {pred_label} (type={p.predicted_type})")
    else:
        print("\nNo misclassifications!")


def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned secret detection SLM")
    parser.add_argument(
        "--model",
        type=str,
        default="mlx-community/Llama-3.2-3B-Instruct-4bit",
        help="Base model path or HuggingFace repo",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default="simplePipeline/slm/adapters",
        help="Path to LoRA adapter weights",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="simplePipeline/slm/data/valid.jsonl",
        help="Path to validation JSONL file",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=200,
        help="Maximum tokens to generate per example",
    )
    parser.add_argument(
        "--skip-base",
        action="store_true",
        help="Skip base model evaluation (faster, just test fine-tuned)",
    )
    args = parser.parse_args()

    # Load examples
    examples = load_examples(args.data)
    print(f"Loaded {len(examples)} validation examples from {args.data}")

    # --- Base model (no adapters) ---
    base_preds = []
    base_metrics = Metrics()
    if not args.skip_base:
        print(f"\nLoading base model: {args.model}")
        base_model, base_tokenizer = load(args.model)
        print("Running base model inference...")
        base_preds = run_inference(base_model, base_tokenizer, examples, "base", args.max_tokens)
        base_metrics = compute_metrics(base_preds)

        # Free base model memory
        del base_model
        mx.clear_cache()
        print(f"Base model: accuracy={base_metrics.accuracy:.1%}, F1={base_metrics.f1:.1%}")

    # --- Fine-tuned model (with adapters) ---
    adapter_path = Path(args.adapter_path)
    if not adapter_path.exists():
        print(f"\nAdapter path {adapter_path} does not exist. Run training first.")
        sys.exit(1)

    print(f"\nLoading fine-tuned model: {args.model} + {args.adapter_path}")
    ft_model, ft_tokenizer = load(args.model, adapter_path=str(adapter_path))
    print("Running fine-tuned model inference...")
    ft_preds = run_inference(ft_model, ft_tokenizer, examples, "fine-tuned", args.max_tokens)
    ft_metrics = compute_metrics(ft_preds)

    # Print results
    if args.skip_base:
        # Just show fine-tuned results
        print(f"\n{'='*70}")
        print("FINE-TUNED MODEL RESULTS")
        print(f"{'='*70}")
        print(f"  Accuracy:  {ft_metrics.accuracy:.1%}")
        print(f"  Precision: {ft_metrics.precision:.1%}")
        print(f"  Recall:    {ft_metrics.recall:.1%}")
        print(f"  F1:        {ft_metrics.f1:.1%}")
        print(f"  Parse Err: {ft_metrics.parse_errors}")
        print(f"  Avg Lat:   {ft_metrics.avg_latency_ms:.0f}ms")
        print(f"\n  TP={ft_metrics.tp}  FP={ft_metrics.fp}")
        print(f"  FN={ft_metrics.fn}  TN={ft_metrics.tn}")

        ft_by_type = compute_per_type_metrics(ft_preds)
        print(f"\n{'Type':<20} {'Recall':>10} {'Precision':>10} {'F1':>10} {'Count':>6}")
        print("-" * 56)
        for t, m in ft_by_type.items():
            print(f"{t:<20} {m.recall:>9.1%} {m.precision:>9.1%} {m.f1:>9.1%} {m.total:>6}")

        misses = [p for p in ft_preds if p.parse_error or p.predicted_is_secret != p.expected_is_secret]
        if misses:
            print(f"\nMisclassifications ({len(misses)}):")
            for p in misses:
                label = "SECRET" if p.expected_is_secret else "NOT SECRET"
                if p.parse_error:
                    print(f"  [{label}] type={p.expected_type} — PARSE ERROR")
                    print(f"    Raw: {p.raw_response[:120]}")
                else:
                    pred_label = "secret" if p.predicted_is_secret else "not secret"
                    print(f"  [{label}] type={p.expected_type} — predicted {pred_label}")
    else:
        print_comparison(base_metrics, ft_metrics, base_preds, ft_preds)


if __name__ == "__main__":
    main()
