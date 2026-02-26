"""Build the RAG vector database from a catalog's classified entities.

Reads all PII entity classifications from a catalog database,
embeds each one (snippet + classification decision), and stores
the vectors for later retrieval during pipeline runs.

Usage:
    python build_vectors.py --db catalog.db --vector-db rag_vectors.db
"""

import argparse
import os
import sys

from store.catalog import Catalog
from rag.embed import embed
from rag.store import VectorStore, STRONG_THRESHOLD, WEAK_THRESHOLD


def build_embed_text(entity_type: str, matched_text: str,
                     confidence: float, classified_by_layer: str,
                     pattern_name: str) -> str:
    """Build the text string that gets embedded.

    This is the key design choice: we embed the entity snippet
    together with the classification decision, so retrieval finds
    similar things WITH their answers attached.
    """
    layer_name = {0: "features", 1: "regex", 2: "ner", 3: "llm"}.get(
        int(classified_by_layer), f"layer-{classified_by_layer}"
    )
    return (
        f"{matched_text} → {entity_type}, "
        f"confidence {confidence:.2f}, detected by: {layer_name}/{pattern_name}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Build RAG vector database from classified entities."
    )
    parser.add_argument(
        "--db", required=True,
        help="Source catalog database (e.g. catalog.db)"
    )
    parser.add_argument(
        "--vector-db", required=True,
        help="Output vector database path (e.g. rag_vectors.db)"
    )
    parser.add_argument(
        "--scan-id", default=None,
        help="Only use entities from a specific scan (default: all scans)"
    )
    parser.add_argument(
        "--min-confidence", type=float, default=0.50,
        help="Only embed entities with confidence >= this value (default: 0.50)"
    )
    args = parser.parse_args()

    if not os.path.exists(args.db):
        print(f"Error: catalog database '{args.db}' not found.")
        sys.exit(1)

    # Fresh vector DB
    if os.path.exists(args.vector_db):
        os.remove(args.vector_db)

    catalog = Catalog(args.db)
    vector_store = VectorStore(args.vector_db)

    # Get all entity types we care about (PII types, not 'file')
    pii_types = ["ssn", "email", "credit_card", "phone", "person_name",
                 "PERSON", "unknown", "other"]

    total = 0
    skipped = 0

    for entity_type in pii_types:
        entities = catalog.get_all_entities_by_type(
            entity_type, scan_id=args.scan_id
        )

        for entity in entities:
            entity_id = entity["id"]
            features = catalog.get_features(entity_id)

            confidence = float(features.get("confidence", 0))
            if confidence < args.min_confidence:
                skipped += 1
                continue

            matched_text = features.get("matched_text", "")
            classified_by_layer = features.get("classified_by_layer", "0")
            pattern_name = features.get("pattern_name", "")

            # Get parent file for document context
            # Entity IDs are like "filename:entity_type:index"
            file_id = entity_id.rsplit(":", 2)[0] if ":" in entity_id else entity_id
            file_entity = catalog.get_entity(file_id)
            doc_context = file_id
            if file_entity:
                file_features = catalog.get_features(file_id)
                file_type = file_features.get("file_type", "")
                if file_type:
                    doc_context = f"{file_id} ({file_type})"

            # Build the text to embed
            embed_text = build_embed_text(
                entity_type, matched_text, confidence,
                classified_by_layer, pattern_name
            )

            # Embed and store
            try:
                vector = embed(embed_text)
            except Exception as e:
                print(f"  Warning: failed to embed {entity_id}: {e}")
                skipped += 1
                continue

            vector_store.add(
                entity_type=entity_type,
                matched_text=matched_text,
                confidence=confidence,
                classified_by_layer=classified_by_layer,
                pattern_name=pattern_name,
                document_context=doc_context,
                embed_text=embed_text,
                embedding=vector,
            )
            total += 1

            if total % 10 == 0:
                print(f"  Embedded {total} entities...")

    print(f"\nDone. Embedded {total} entities, skipped {skipped}.")
    print(f"Vector database: {args.vector_db}")
    print(f"Similarity thresholds: strong={STRONG_THRESHOLD}, weak={WEAK_THRESHOLD}")

    vector_store.close()
    catalog.close()


if __name__ == "__main__":
    main()
