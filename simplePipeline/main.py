"""SimplePipeline: Data Classification Pipeline.

Entry point that discovers documents, runs them through the
configured pipeline, stores results in the catalog, and prints a report.

Usage examples:
    # Default: scan sample_docs, write to catalog.db
    python main.py

    # Scan test docs into a separate database with a run slug
    python main.py --docs test_docs --db test_catalog.db --slug baseline-v1 \
                   --description "Baseline run, no RAG, gpt-oss:20b"

    # Same test docs, different model
    python main.py --docs test_docs --db test_catalog.db --slug llama-v1 \
                   --description "Llama 3.2 comparison" --model llama3.2:latest

    # Fresh start (wipes existing db)
    python main.py --docs sample_docs --db training.db --slug training --clean
"""

import argparse
import os
import yaml

from connector.filesystem import FilesystemConnector
from pipeline.context import PipelineContext
from pipeline.pipeline import Pipeline, _import_class
from store.catalog import Catalog
from report.cli import print_report
from report.html import write_html_report


def _redact(text: str) -> str:
    """Redact matched text for console output — show first/last 2 chars."""
    if len(text) > 6:
        return text[:2] + "***" + text[-2:]
    return "***"


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the data classification pipeline."
    )
    parser.add_argument(
        "--docs", default=None,
        help="Document directory to scan (overrides config.yaml root_dir)"
    )
    parser.add_argument(
        "--db", default="catalog.db",
        help="SQLite database path (default: catalog.db)"
    )
    parser.add_argument(
        "--slug", default=None,
        help="Short name for this run (e.g. 'baseline-v1', 'rag-experiment')"
    )
    parser.add_argument(
        "--description", default="",
        help="Free-text description of what this run is testing"
    )
    parser.add_argument(
        "--model", default=None,
        help="Override the LLM model in config (e.g. 'llama3.2:latest')"
    )
    parser.add_argument(
        "--clean", action="store_true",
        help="Delete existing database before running (default: append)"
    )
    parser.add_argument(
        "--rag", default=None, metavar="VECTOR_DB",
        help="Path to RAG vector database (enables retrieval-augmented LLM calls)"
    )
    parser.add_argument(
        "--config", default="pipeline/config.yaml",
        help="Path to pipeline config (default: pipeline/config.yaml)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Override docs directory if specified
    if args.docs:
        config["pipeline"]["connector"]["root_dir"] = args.docs

    # Override LLM model if specified
    if args.model:
        for step in config["pipeline"]["steps"]:
            if step["name"] == "llm":
                step["model"] = args.model

    # Derive pipeline version from (possibly modified) config
    pipeline_version = Pipeline.derive_version(config)
    if args.rag:
        pipeline_version += "+rag"
    docs_dir = config["pipeline"]["connector"]["root_dir"]

    # Set up connector
    connector_config = config["pipeline"]["connector"]
    connector_class = _import_class(connector_config["class"])
    connector = connector_class(connector_config["root_dir"])

    # Load RAG vector store if specified
    vector_store = None
    if args.rag:
        if not os.path.exists(args.rag):
            print(f"Error: RAG vector database '{args.rag}' not found.")
            print("Run build_vectors.py first to create it.")
            return
        from rag.store import VectorStore
        vector_store = VectorStore(args.rag)
        print(f"RAG: {args.rag} ({vector_store.count()} vectors)")

    # Build pipeline from config dict
    pipeline = _build_pipeline(config, vector_store=vector_store)

    # Set up catalog
    db_path = args.db
    if args.clean and os.path.exists(db_path):
        os.remove(db_path)
    catalog = Catalog(db_path)
    scan_id = catalog.new_scan_id()

    # Generate slug if not provided
    slug = args.slug or scan_id

    # Record this run
    catalog.create_run(
        scan_id=scan_id,
        slug=slug,
        description=args.description,
        pipeline_version=pipeline_version,
        config=config,
        dataset=docs_dir,
    )

    # Discover and process all documents
    documents = connector.discover()
    print(f"Run: {slug}")
    print(f"Pipeline: {pipeline_version}")
    print(f"Dataset: {docs_dir} ({len(documents)} documents)")
    print(f"Database: {db_path}")
    print(f"Scan ID: {scan_id}")
    print()

    for metadata in sorted(documents, key=lambda d: d.id):
        doc = connector.fetch(metadata.id)

        # Fresh context per document
        context = PipelineContext()
        context.set("document", doc)

        # Run all pipeline steps
        timings = pipeline.run(context)

        # --- Store file entity ---
        file_entity_id = metadata.id
        catalog.store_entity(file_entity_id, "file", metadata.id, scan_id)

        # Store file-level features
        if context.has("features"):
            feat = context.get("features")
            catalog.store_features(file_entity_id, {
                "file_type": feat.file_type,
                "text_length": feat.text_length,
                "line_count": feat.line_count,
                "has_structured_headers": feat.has_structured_headers,
                "header_names": ",".join(feat.header_names),
                "header_pii_similarity_score": feat.header_pii_similarity_score,
                "digit_density": feat.digit_density,
                "path_context": feat.path_context,
                "entropy_score": feat.entropy_score,
            }, pipeline_version=feat.pipeline_version)

        # Store scan metrics per layer
        for layer_name, latency_ms in timings.items():
            entities_at_layer = 0
            escalated = False
            if context.has("entities"):
                entities_at_layer = len([
                    e for e in context.get("entities")
                    if str(e.classified_by_layer) == str(
                        {"features": 0, "regex": 1, "ner": 2, "slm": 4, "llm": 3}.get(layer_name, -1)
                    )
                ])
            if layer_name in ("ner", "slm", "llm") and latency_ms > 1.0:
                escalated = True

            catalog.store_scan_metric(
                scan_id, file_entity_id, layer_name,
                latency_ms, entities_at_layer, escalated
            )

        # --- Store PII entities as children of the file ---
        if context.has("entities"):
            entities = context.get("entities")
            for i, e in enumerate(entities):
                entity_id = f"{file_entity_id}:{e.entity_type}:{i}"
                catalog.store_entity(entity_id, e.entity_type, _redact(e.matched_text), scan_id)
                catalog.store_features(entity_id, {
                    "matched_text": _redact(e.matched_text),
                    "match_length": e.match_length,
                    "line_number": e.line_number,
                    "char_offset": e.char_offset,
                    "position": e.position,
                    "confidence": e.confidence,
                    "pattern_name": e.pattern_name,
                    "classified_by_layer": e.classified_by_layer,
                    "layer1_confidence": e.layer1_confidence or "",
                    "layer2_confidence": e.layer2_confidence or "",
                    "layer3_confidence": e.layer3_confidence or "",
                    "layer4_confidence": e.layer4_confidence or "",
                })
                catalog.store_relationship(file_entity_id, entity_id)

        # Progress output
        finding_count = len(context.get("entities")) if context.has("entities") else 0
        timing_str = "  ".join(f"{k}: {v:.1f}ms" for k, v in timings.items())
        print(f"  {metadata.id:<30} {finding_count} findings  [{timing_str}]")

    # --- Print the report ---
    print_report(catalog, scan_id)
    html_path = write_html_report(catalog, scan_id)
    print(f"\nHTML report: {html_path}")
    if vector_store:
        vector_store.close()
    catalog.close()
    print(f"Results stored in {db_path}")


def _build_pipeline(config: dict, vector_store=None) -> Pipeline:
    """Build a Pipeline directly from a config dict.

    If a vector_store is provided, it gets passed to the LLM step
    so it can retrieve similar past classifications for few-shot context.
    """
    steps = []
    for step_config in config["pipeline"]["steps"]:
        step_class = _import_class(step_config["class"])
        kwargs = {k: v for k, v in step_config.items()
                  if k not in ("name", "class")}
        # Inject vector store into the LLM step
        if step_config["name"] == "llm" and vector_store is not None:
            kwargs["vector_store"] = vector_store
        steps.append(step_class(**kwargs))
    return Pipeline(steps)


if __name__ == "__main__":
    main()
