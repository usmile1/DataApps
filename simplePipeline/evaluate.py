"""Evaluate pipeline runs against ground truth labels.

Compares detected entities from one or more runs against the labeled
test data. Reports precision, recall, F1, and detailed misses/false alarms.

Usage:
    # Evaluate all runs in a test database
    python evaluate.py --db test_results.db --labels test_docs/test_labels.json

    # Evaluate a specific run
    python evaluate.py --db test_results.db --labels test_docs/test_labels.json \
                       --slug baseline-norag
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from store.catalog import Catalog


@dataclass
class GroundTruthEntity:
    """A labeled entity from the test set."""
    file: str
    text: str
    line: int
    entity_type: str
    is_pii: bool
    note: str


@dataclass
class DetectedEntity:
    """An entity detected by the pipeline."""
    file: str
    entity_type: str
    matched_text_redacted: str
    line_number: int
    confidence: float
    classified_by_layer: str
    pattern_name: str


@dataclass
class MatchResult:
    """Result of matching a ground truth entity against detections."""
    ground_truth: GroundTruthEntity
    detected: Optional[DetectedEntity] = None
    category: str = ""  # TP, FP, FN, TN


@dataclass
class RunMetrics:
    """Aggregate metrics for a single run."""
    slug: str
    pipeline_version: str
    true_positives: List[MatchResult] = field(default_factory=list)
    false_positives: List[DetectedEntity] = field(default_factory=list)
    false_negatives: List[MatchResult] = field(default_factory=list)
    true_negatives: List[MatchResult] = field(default_factory=list)

    @property
    def tp(self) -> int:
        return len(self.true_positives)

    @property
    def fp(self) -> int:
        return len(self.false_positives)

    @property
    def fn(self) -> int:
        return len(self.false_negatives)

    @property
    def tn(self) -> int:
        return len(self.true_negatives)

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


# ---------------------------------------------------------------------------
# Matching logic
# ---------------------------------------------------------------------------

LINE_TOLERANCE = 3  # ground truth and detected lines can differ by this much


def _text_matches(ground_truth_text: str, redacted_text: str) -> bool:
    """Check if redacted text plausibly matches the ground truth.

    Redacted format is like "44***31" (first 2 + *** + last 2 chars).
    Short texts are just "***".
    """
    if redacted_text == "***":
        # Can't match on redacted short text, rely on line matching
        return True
    gt = ground_truth_text
    if len(gt) <= 6:
        return True  # too short to verify, trust line match
    # Check first 2 and last 2 chars
    return gt[:2] == redacted_text[:2] and gt[-2:] == redacted_text[-2:]


def _match_detection(gt: GroundTruthEntity,
                     detections: List[DetectedEntity]) -> Optional[DetectedEntity]:
    """Find the best matching detection for a ground truth entity."""
    candidates = []
    for d in detections:
        if d.file != gt.file:
            continue
        line_dist = abs(d.line_number - gt.line)
        if line_dist > LINE_TOLERANCE:
            continue
        if _text_matches(gt.text, d.matched_text_redacted):
            candidates.append((line_dist, d))

    if not candidates:
        return None
    # Return closest line match
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def load_ground_truth(labels_path: str) -> List[GroundTruthEntity]:
    """Load ground truth entities from the labels JSON file."""
    with open(labels_path, "r") as f:
        data = json.load(f)

    entities = []
    for filename, file_data in data.items():
        if filename.startswith("_"):
            continue
        for e in file_data["entities"]:
            entities.append(GroundTruthEntity(
                file=filename,
                text=e["text"],
                line=e["line"],
                entity_type=e["entity_type"],
                is_pii=e["is_pii"],
                note=e.get("note", ""),
            ))
    return entities


def load_detections(catalog: Catalog, scan_id: str) -> List[DetectedEntity]:
    """Load detected entities from the catalog for a specific scan.

    Queries all non-file entities directly by scan_id, extracting the
    source filename from the entity ID (format: 'filename:type:index').
    This avoids depending on parent-child relationships which can break
    when multiple runs share a database (INSERT OR REPLACE on file entities).
    """
    detections = []

    # Get all entities for this scan, skip 'file' type
    all_entities = catalog._conn.execute(
        "SELECT id, entity_type FROM entities "
        "WHERE scan_id = ? AND entity_type != 'file'",
        (scan_id,)
    ).fetchall()

    for ent in all_entities:
        entity_id = ent["id"]
        entity_type = ent["entity_type"]

        # Extract source filename from entity ID: "filename:type:index"
        parts = entity_id.rsplit(":", 2)
        if len(parts) >= 3:
            file_name = parts[0]
        else:
            continue

        features = catalog.get_features(entity_id)
        if not features:
            continue

        detections.append(DetectedEntity(
            file=file_name,
            entity_type=entity_type,
            matched_text_redacted=features.get("matched_text", "***"),
            line_number=int(features.get("line_number", 0)),
            confidence=float(features.get("confidence", 0)),
            classified_by_layer=features.get("classified_by_layer", ""),
            pattern_name=features.get("pattern_name", ""),
        ))

    return detections


def evaluate_run(ground_truth: List[GroundTruthEntity],
                 detections: List[DetectedEntity],
                 slug: str, pipeline_version: str,
                 confidence_threshold: float = 0.40) -> RunMetrics:
    """Compare detections against ground truth for a single run."""
    metrics = RunMetrics(slug=slug, pipeline_version=pipeline_version)

    # Track which detections got matched (to find false positives)
    matched_detections = set()

    # Filter detections by confidence threshold
    active_detections = [d for d in detections if d.confidence >= confidence_threshold]

    for gt in ground_truth:
        match = _match_detection(gt, active_detections)

        if gt.is_pii:
            if match:
                # True positive: PII correctly detected
                metrics.true_positives.append(
                    MatchResult(ground_truth=gt, detected=match, category="TP")
                )
                matched_detections.add(id(match))
            else:
                # False negative: PII missed
                metrics.false_negatives.append(
                    MatchResult(ground_truth=gt, category="FN")
                )
        else:
            if match:
                # False positive: not-PII incorrectly flagged
                metrics.false_positives.append(match)
                matched_detections.add(id(match))
            else:
                # True negative: not-PII correctly ignored
                metrics.true_negatives.append(
                    MatchResult(ground_truth=gt, category="TN")
                )

    # Also count detections that don't match ANY ground truth entity
    # (things the pipeline found that we didn't even label)
    for d in active_detections:
        if id(d) not in matched_detections:
            metrics.false_positives.append(d)

    return metrics


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_summary(all_metrics: List[RunMetrics]) -> None:
    """Print a comparison table across all runs."""
    print()
    print("=" * 72)
    print("  EVALUATION RESULTS")
    print("=" * 72)

    # Summary table
    print()
    header = f"{'Slug':<22} {'Version':<30} {'P':>5} {'R':>5} {'F1':>5} {'TP':>4} {'FP':>4} {'FN':>4} {'TN':>4}"
    print(header)
    print("-" * len(header))

    for m in all_metrics:
        # Truncate version string if too long
        ver = m.pipeline_version
        if len(ver) > 28:
            ver = ver[:25] + "..."
        print(
            f"{m.slug:<22} {ver:<30} "
            f"{m.precision:5.2f} {m.recall:5.2f} {m.f1:5.2f} "
            f"{m.tp:4d} {m.fp:4d} {m.fn:4d} {m.tn:4d}"
        )

    print()


def print_details(metrics: RunMetrics) -> None:
    """Print detailed misses and false alarms for a single run."""
    print(f"── {metrics.slug} ({'─' * (55 - len(metrics.slug))})")
    print(f"   Pipeline: {metrics.pipeline_version}")
    print()

    if metrics.false_negatives:
        print(f"   MISSED (False Negatives): {metrics.fn}")
        for mr in metrics.false_negatives:
            gt = mr.ground_truth
            print(f"     {gt.file}:{gt.line}  \"{gt.text}\"")
            print(f"       type={gt.entity_type}  {gt.note}")
        print()

    if metrics.false_positives:
        print(f"   FALSE ALARMS (False Positives): {metrics.fp}")
        for d in metrics.false_positives:
            if isinstance(d, DetectedEntity):
                print(
                    f"     {d.file}:{d.line_number}  \"{d.matched_text_redacted}\" "
                    f"({d.entity_type}, conf={d.confidence:.2f})"
                )
        print()

    if not metrics.false_negatives and not metrics.false_positives:
        print("   Perfect score!")
        print()


def print_per_file(all_metrics: List[RunMetrics],
                   ground_truth: List[GroundTruthEntity]) -> None:
    """Print per-file recall breakdown across runs."""
    files = sorted(set(gt.file for gt in ground_truth))

    print("── Per-File Recall " + "─" * 53)
    print()

    header = f"  {'File':<30}"
    for m in all_metrics:
        slug = m.slug if len(m.slug) <= 18 else m.slug[:15] + "..."
        header += f" {slug:>18}"
    print(header)
    print("  " + "-" * (30 + 19 * len(all_metrics)))

    for f in files:
        file_gt_pii = [gt for gt in ground_truth if gt.file == f and gt.is_pii]
        total_pii = len(file_gt_pii)
        if total_pii == 0:
            continue

        row = f"  {f:<30}"
        for m in all_metrics:
            found = sum(
                1 for mr in m.true_positives
                if mr.ground_truth.file == f
            )
            row += f" {found:>3}/{total_pii:<3} ({found/total_pii*100:4.0f}%)"
        print(row)

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate pipeline runs against ground truth labels."
    )
    parser.add_argument(
        "--db", required=True,
        help="Test results database (e.g. test_results.db)"
    )
    parser.add_argument(
        "--labels", required=True,
        help="Ground truth labels JSON (e.g. test_docs/test_labels.json)"
    )
    parser.add_argument(
        "--slug", default=None,
        help="Evaluate only this run slug (default: all runs)"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.40,
        help="Confidence threshold for counting a detection (default: 0.40)"
    )
    parser.add_argument(
        "--details", action="store_true",
        help="Show detailed misses and false alarms per run"
    )
    args = parser.parse_args()

    # Load ground truth
    ground_truth = load_ground_truth(args.labels)
    pii_count = sum(1 for gt in ground_truth if gt.is_pii)
    not_pii_count = sum(1 for gt in ground_truth if not gt.is_pii)
    print(f"Ground truth: {len(ground_truth)} entities "
          f"({pii_count} PII, {not_pii_count} not-PII)")
    print(f"Confidence threshold: {args.threshold}")

    # Load runs
    catalog = Catalog(args.db)
    runs = catalog.get_all_runs()

    if args.slug:
        runs = [r for r in runs if r["slug"] == args.slug]

    if not runs:
        print("No matching runs found.")
        catalog.close()
        sys.exit(1)

    # Evaluate each run
    all_metrics = []
    for run in runs:
        detections = load_detections(catalog, run["scan_id"])
        metrics = evaluate_run(
            ground_truth, detections,
            slug=run["slug"],
            pipeline_version=run["pipeline_version"],
            confidence_threshold=args.threshold,
        )
        all_metrics.append(metrics)

    # Report
    print_summary(all_metrics)

    if args.details:
        for m in all_metrics:
            print_details(m)

    if len(all_metrics) > 1:
        print_per_file(all_metrics, ground_truth)

        # Delta summary
        base = all_metrics[0]
        for comp in all_metrics[1:]:
            dp = comp.precision - base.precision
            dr = comp.recall - base.recall
            df = comp.f1 - base.f1
            print(f"  Delta ({comp.slug} vs {base.slug}):")
            print(f"    Precision: {dp:+.2f}  Recall: {dr:+.2f}  F1: {df:+.2f}")
            print(f"    FP change: {comp.fp - base.fp:+d}  "
                  f"FN change: {comp.fn - base.fn:+d}")
            print()

    catalog.close()


if __name__ == "__main__":
    main()
