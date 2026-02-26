"""CLI report generator.

Reads from the catalog and prints a formatted summary of scan results.
"""

from typing import Dict, List
from store.catalog import Catalog


def print_report(catalog: Catalog, scan_id: str) -> None:
    """Print the full scan report."""
    _print_header(scan_id)
    _print_scan_summary(catalog, scan_id)
    _print_document_findings(catalog, scan_id)
    _print_layer_performance(catalog, scan_id)
    _print_risk_report(catalog, scan_id)


def _print_header(scan_id: str) -> None:
    print()
    print("=" * 78)
    print("  SIMPLEPIPELINE SCAN REPORT")
    print(f"  Scan ID: {scan_id}")
    print("=" * 78)


def _print_scan_summary(catalog: Catalog, scan_id: str) -> None:
    print()
    print("--- SCAN SUMMARY ---")
    print()

    counts = catalog.get_entity_count_by_type(scan_id)
    file_count = counts.pop("file", 0)
    total_findings = sum(counts.values())

    print(f"  Documents scanned:  {file_count}")
    print(f"  Total PII findings: {total_findings}")
    print()

    if counts:
        print("  Findings by type:")
        for entity_type, count in sorted(counts.items(), key=lambda x: -x[1]):
            print(f"    {entity_type:<16} {count}")

    # Escalation stats from metrics
    metrics = catalog.get_scan_metrics(scan_id)
    # Threshold: if latency > 1ms, the layer actually did work (not just gate check)
    ner_runs = [m for m in metrics if m["layer"] == "ner" and m["latency_ms"] > 1.0]
    llm_runs = [m for m in metrics if m["layer"] == "llm" and m["latency_ms"] > 1.0]
    print()
    print(f"  Escalated to NER:   {len(ner_runs)} / {file_count} documents")
    print(f"  Escalated to LLM:   {len(llm_runs)} / {file_count} documents")


def _print_document_findings(catalog: Catalog, scan_id: str) -> None:
    print()
    print("--- PER-DOCUMENT FINDINGS ---")
    print()
    print(f"  {'Document':<30} {'Findings':>8}  {'Max Conf':>8}  {'Layers Used'}")
    print(f"  {'-'*30} {'-'*8}  {'-'*8}  {'-'*20}")

    files = catalog.get_all_entities_by_type("file", scan_id)
    for f in sorted(files, key=lambda x: x["name"]):
        children = catalog.get_children(f["id"])
        if not children:
            print(f"  {f['name']:<30} {'0':>8}  {'--':>8}  regex only")
            continue

        # Get confidence from child features
        max_conf = 0.0
        layers_used = set()
        for child in children:
            feats = catalog.get_features(child["id"])
            conf = float(feats.get("confidence", 0))
            max_conf = max(max_conf, conf)
            layer = feats.get("classified_by_layer", "?")
            layers_used.add(f"L{layer}")

        layers_str = ", ".join(sorted(layers_used))
        print(f"  {f['name']:<30} {len(children):>8}  {max_conf:>8.2f}  {layers_str}")


def _print_layer_performance(catalog: Catalog, scan_id: str) -> None:
    print()
    print("--- LAYER PERFORMANCE ---")
    print()

    metrics = catalog.get_scan_metrics(scan_id)

    # Group by layer
    by_layer: Dict[str, List[Dict]] = {}
    for m in metrics:
        by_layer.setdefault(m["layer"], []).append(m)

    print(f"  {'Layer':<12} {'Calls':>6} {'Skipped':>8} {'Avg (ms)':>10} {'Max (ms)':>10} {'Total (ms)':>12}")
    print(f"  {'-'*12} {'-'*6} {'-'*8} {'-'*10} {'-'*10} {'-'*12}")

    for layer in ["features", "regex", "ner", "llm"]:
        if layer not in by_layer:
            continue
        layer_metrics = by_layer[layer]
        ran = [m for m in layer_metrics if m["latency_ms"] > 0]
        skipped = len(layer_metrics) - len(ran)

        if ran:
            latencies = [m["latency_ms"] for m in ran]
            avg = sum(latencies) / len(latencies)
            max_lat = max(latencies)
            total = sum(latencies)
        else:
            avg = max_lat = total = 0

        print(f"  {layer:<12} {len(ran):>6} {skipped:>8} {avg:>10.1f} {max_lat:>10.1f} {total:>12.1f}")


def _print_risk_report(catalog: Catalog, scan_id: str) -> None:
    print()
    print("--- RISK REPORT ---")
    print()

    files = catalog.get_all_entities_by_type("file", scan_id)

    # Group findings by risk level
    high_risk = []  # PII in unexpected locations
    expected = []   # PII in expected locations
    clean = []      # No PII

    for f in sorted(files, key=lambda x: x["name"]):
        feats = catalog.get_features(f["id"])
        children = catalog.get_children(f["id"])
        path_context = feats.get("path_context", "")

        if not children:
            clean.append(f["name"])
            continue

        # PII in public/shared locations = high risk
        # PII in hr/finance = expected
        if path_context in ("public", "shared", "temp"):
            high_risk.append((f["name"], len(children), path_context))
        elif path_context in ("hr", "finance", "payroll"):
            expected.append((f["name"], len(children), path_context))
        else:
            # Default: any PII finding is medium risk
            high_risk.append((f["name"], len(children), path_context or "(root)"))

    if high_risk:
        print("  HIGH RISK — PII in unexpected or unprotected locations:")
        for name, count, ctx in high_risk:
            print(f"    !! {name:<30} {count} findings  (context: {ctx})")
        print()

    if expected:
        print("  EXPECTED — PII in appropriate locations:")
        for name, count, ctx in expected:
            print(f"    ok {name:<30} {count} findings  (context: {ctx})")
        print()

    if clean:
        print("  CLEAN — No PII detected:")
        for name in clean:
            print(f"       {name}")
        print()

    # Business context insight
    if high_risk and expected:
        print("  ** CONTEXT INSIGHT: Same PII types appear in both protected (hr/)")
        print("     and unprotected locations. Review access controls for high-risk files.")

    print()
    print("=" * 78)
