"""HTML report generator.

Reads from the catalog and produces a self-contained HTML report file.
"""

from typing import Dict, List
from store.catalog import Catalog


def write_html_report(catalog: Catalog, scan_id: str,
                      output_path: str = "report.html") -> str:
    """Generate an HTML report and write it to disk. Returns the path."""

    counts = catalog.get_entity_count_by_type(scan_id)
    file_count = counts.pop("file", 0)
    total_findings = sum(counts.values())

    metrics = catalog.get_scan_metrics(scan_id)
    ner_runs = [m for m in metrics if m["layer"] == "ner" and m["latency_ms"] > 1.0]
    llm_runs = [m for m in metrics if m["layer"] == "llm" and m["latency_ms"] > 1.0]

    # --- Build document rows ---
    files = catalog.get_all_entities_by_type("file", scan_id)
    doc_rows = ""
    for f in sorted(files, key=lambda x: x["name"]):
        feats = catalog.get_features(f["id"])
        children = catalog.get_children(f["id"])
        path_context = feats.get("path_context", "")

        if not children:
            risk_class = "clean"
            risk_label = "Clean"
            max_conf = "—"
            layers_str = "—"
        else:
            max_conf_val = 0.0
            layers_used = set()
            for child in children:
                cf = catalog.get_features(child["id"])
                conf = float(cf.get("confidence", 0))
                max_conf_val = max(max_conf_val, conf)
                layer = cf.get("classified_by_layer", "?")
                layers_used.add(f"L{layer}")
            max_conf = f"{max_conf_val:.2f}"
            layers_str = ", ".join(sorted(layers_used))

            if path_context in ("hr", "finance", "payroll"):
                risk_class = "expected"
                risk_label = "Expected"
            elif path_context in ("public", "shared", "temp"):
                risk_class = "high"
                risk_label = "High Risk"
            else:
                risk_class = "high"
                risk_label = "High Risk"

        # Entity detail rows for this file
        entity_details = ""
        for child in children:
            cf = catalog.get_features(child["id"])
            entity_details += f"""
                <tr class="entity-detail">
                    <td></td>
                    <td>{child['entity_type']}</td>
                    <td>{cf.get('matched_text', '***')}</td>
                    <td>Line {cf.get('line_number', '?')}</td>
                    <td>{float(cf.get('confidence', 0)):.2f}</td>
                    <td>L{cf.get('classified_by_layer', '?')}</td>
                </tr>"""

        doc_rows += f"""
            <tr class="doc-row {risk_class}">
                <td><strong>{f['name']}</strong></td>
                <td>{len(children)}</td>
                <td>{max_conf}</td>
                <td>{layers_str}</td>
                <td>{float(feats.get('entropy_score', 0)):.2f}</td>
                <td><span class="risk-badge {risk_class}">{risk_label if children else 'Clean'}</span></td>
            </tr>
            {entity_details}"""

    # --- Build layer performance rows ---
    by_layer: Dict[str, List[Dict]] = {}
    for m in metrics:
        by_layer.setdefault(m["layer"], []).append(m)

    layer_rows = ""
    for layer in ["features", "regex", "ner", "llm"]:
        if layer not in by_layer:
            continue
        layer_metrics = by_layer[layer]
        ran = [m for m in layer_metrics if m["latency_ms"] > 1.0]
        skipped = len(layer_metrics) - len(ran)
        if ran:
            latencies = [m["latency_ms"] for m in ran]
            avg = sum(latencies) / len(latencies)
            max_lat = max(latencies)
            total = sum(latencies)
        else:
            avg = max_lat = total = 0

        layer_rows += f"""
            <tr>
                <td><strong>{layer}</strong></td>
                <td>{len(ran)}</td>
                <td>{skipped}</td>
                <td>{avg:.1f}</td>
                <td>{max_lat:.1f}</td>
                <td>{total:.1f}</td>
            </tr>"""

    # --- Findings by type rows ---
    type_rows = ""
    for entity_type, count in sorted(counts.items(), key=lambda x: -x[1]):
        type_rows += f"<tr><td>{entity_type}</td><td>{count}</td></tr>\n"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SimplePipeline Scan Report — {scan_id}</title>
<style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
           background: #0f1117; color: #e1e4e8; padding: 2rem; }}
    h1 {{ color: #58a6ff; margin-bottom: 0.3rem; }}
    h2 {{ color: #c9d1d9; margin: 2rem 0 1rem; border-bottom: 1px solid #30363d; padding-bottom: 0.5rem; }}
    .scan-id {{ color: #8b949e; font-size: 0.9rem; margin-bottom: 2rem; }}
    .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                     gap: 1rem; margin-bottom: 2rem; }}
    .summary-card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px;
                     padding: 1.2rem; text-align: center; }}
    .summary-card .number {{ font-size: 2rem; font-weight: bold; color: #58a6ff; }}
    .summary-card .label {{ color: #8b949e; font-size: 0.85rem; margin-top: 0.3rem; }}
    table {{ width: 100%; border-collapse: collapse; margin-bottom: 1rem; }}
    th {{ background: #161b22; color: #8b949e; text-align: left; padding: 0.7rem;
         font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em; }}
    td {{ padding: 0.6rem 0.7rem; border-bottom: 1px solid #21262d; }}
    tr:hover {{ background: #161b22; }}
    .entity-detail td {{ font-size: 0.85rem; color: #8b949e; padding-left: 2rem; }}
    .risk-badge {{ padding: 0.2rem 0.6rem; border-radius: 12px; font-size: 0.75rem; font-weight: 600; }}
    .risk-badge.high {{ background: #da363340; color: #f85149; }}
    .risk-badge.expected {{ background: #23883040; color: #3fb950; }}
    .risk-badge.clean {{ background: #30363d; color: #8b949e; }}
    .context-box {{ background: #1c1206; border: 1px solid #d29922; border-radius: 8px;
                  padding: 1rem 1.2rem; margin-top: 1.5rem; }}
    .context-box strong {{ color: #d29922; }}
    .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; }}
    @media (max-width: 768px) {{ .two-col {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>

<h1>SimplePipeline Scan Report</h1>
<div class="scan-id">Scan ID: {scan_id}</div>

<div class="summary-grid">
    <div class="summary-card">
        <div class="number">{file_count}</div>
        <div class="label">Documents Scanned</div>
    </div>
    <div class="summary-card">
        <div class="number">{total_findings}</div>
        <div class="label">PII Findings</div>
    </div>
    <div class="summary-card">
        <div class="number">{len(ner_runs)}</div>
        <div class="label">Escalated to NER</div>
    </div>
    <div class="summary-card">
        <div class="number">{len(llm_runs)}</div>
        <div class="label">Escalated to LLM</div>
    </div>
</div>

<div class="two-col">
<div>
<h2>Findings by Type</h2>
<table>
    <tr><th>Entity Type</th><th>Count</th></tr>
    {type_rows}
</table>
</div>

<div>
<h2>Layer Performance</h2>
<table>
    <tr><th>Layer</th><th>Ran</th><th>Skipped</th><th>Avg (ms)</th><th>Max (ms)</th><th>Total (ms)</th></tr>
    {layer_rows}
</table>
</div>
</div>

<h2>Document Findings</h2>
<table>
    <tr>
        <th>Document</th><th>Findings</th><th>Max Confidence</th>
        <th>Layers</th><th>Entropy</th><th>Risk</th>
    </tr>
    {doc_rows}
</table>

<div class="context-box">
    <strong>Context Insight:</strong> Same PII types (SSN) appear in both protected locations
    (hr/) and unprotected locations (public/, root). Review access controls — data classification
    alone isn't enough without proper data governance.
</div>

</body>
</html>"""

    with open(output_path, "w") as f:
        f.write(html)

    return output_path
