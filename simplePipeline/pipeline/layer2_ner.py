"""Layer 2: SpaCy NER-based contextual classification.

Runs Named Entity Recognition on the full document to:
1. Adjust confidence on ambiguous regex findings using context
   (e.g., SSN near a PERSON name → confidence boost)
2. Discover new PII that regex missed entirely
   (e.g., person names, organization names)

Only invoked when a document has entities below the confidence
threshold after Layer 1. Runs on the full document when called —
SpaCy is local and cheap (~20ms), so the cost funnel is about
skipping documents entirely, not limiting input.
"""

import spacy
from typing import Dict, List, Tuple

from connector.models import RawDocument
from features.compute import DocumentFeatures
from pipeline.models import Entity


# How close (in characters) a NER entity needs to be to a regex
# finding to count as "nearby" context
PROXIMITY_WINDOW = 150

# SpaCy entity labels we care about for PII context
PII_RELEVANT_LABELS = {"PERSON", "ORG", "GPE", "DATE", "MONEY", "CARDINAL"}


class NERClassifier:
    """Uses SpaCy NER to add contextual confidence to entity findings."""

    def __init__(self) -> None:
        self._nlp = spacy.load("en_core_web_sm")

    def classify(self, doc: RawDocument, features: DocumentFeatures,
                 existing_entities: List[Entity]) -> List[Entity]:
        """Run NER on the full document. Adjust existing entities and find new ones.

        Args:
            doc: The raw document to analyze.
            features: Pre-computed document features.
            existing_entities: Entities found by Layer 1 (regex).

        Returns:
            Updated list of entities with adjusted confidence scores
            and any newly discovered entities.
        """
        spacy_doc = self._nlp(doc.content)
        ner_entities = [
            (ent.text, ent.label_, ent.start_char, ent.end_char)
            for ent in spacy_doc.ents
            if ent.label_ in PII_RELEVANT_LABELS
        ]

        # Step 1: Adjust confidence on existing regex findings
        updated = self._adjust_existing(existing_entities, ner_entities)

        # Step 2: Discover new PII that regex missed (person names, etc.)
        new_entities = self._discover_new(
            doc, ner_entities, existing_entities
        )

        return updated + new_entities

    def _adjust_existing(self, entities: List[Entity],
                         ner_entities: List[Tuple]) -> List[Entity]:
        """Adjust confidence on regex findings based on NER context.

        The logic:
        - SSN/phone near a PERSON → likely real PII, boost confidence
        - SSN/phone with NO person context nearby → no boost, might decrease
        - Entity near ORG/GPE → could be business data, slight penalty
        """
        updated = []
        for entity in entities:
            nearby = self._find_nearby_ner(entity.position, ner_entities)
            nearby_labels = {label for _, label, _, _ in nearby}
            nearby_texts = {text for text, _, _, _ in nearby}

            adjustment = 0.0
            reason = "no_ner_context"

            if entity.entity_type in ("ssn", "phone"):
                if "PERSON" in nearby_labels:
                    adjustment = 0.15
                    reason = f"person_nearby:{','.join(nearby_texts & _persons(nearby))}"
                elif "ORG" in nearby_labels and "PERSON" not in nearby_labels:
                    adjustment = -0.10
                    reason = "org_context_no_person"

            elif entity.entity_type == "credit_card":
                if "PERSON" in nearby_labels or "MONEY" in nearby_labels:
                    adjustment = 0.10
                    reason = "person_or_money_nearby"

            new_confidence = max(0.0, min(1.0, entity.confidence + adjustment))

            # Create updated entity with Layer 2 confidence
            updated_entity = Entity(
                document_id=entity.document_id,
                entity_type=entity.entity_type,
                matched_text=entity.matched_text,
                match_length=entity.match_length,
                line_number=entity.line_number,
                char_offset=entity.char_offset,
                position=entity.position,
                confidence=new_confidence,
                pattern_name=entity.pattern_name,
                classified_by_layer=2,
                layer1_confidence=entity.layer1_confidence,
                layer2_confidence=new_confidence,
                layer3_confidence=entity.layer3_confidence,
            )
            updated.append(updated_entity)

        return updated

    def _discover_new(self, doc: RawDocument,
                      ner_entities: List[Tuple],
                      existing_entities: List[Entity]) -> List[Entity]:
        """Find PII that regex missed — primarily person names."""
        existing_positions = {
            (e.position, e.match_length) for e in existing_entities
        }

        new_entities = []
        for text, label, start, end in ner_entities:
            # Skip if this overlaps with an existing regex finding
            if any(start < pos + length and pos < end
                   for pos, length in existing_positions):
                continue

            # We care about PERSON entities as new PII discoveries
            if label == "PERSON" and len(text.split()) >= 2:
                line_number, char_offset = _compute_line_and_offset(
                    doc.content, start
                )
                new_entities.append(Entity(
                    document_id=doc.metadata.id,
                    entity_type="person_name",
                    matched_text=text,
                    match_length=end - start,
                    line_number=line_number,
                    char_offset=char_offset,
                    position=start,
                    confidence=0.60,
                    pattern_name="spacy_person",
                    classified_by_layer=2,
                    layer1_confidence=None,
                    layer2_confidence=0.60,
                ))

        return new_entities

    def _find_nearby_ner(self, position: int,
                         ner_entities: List[Tuple]) -> List[Tuple]:
        """Find NER entities within PROXIMITY_WINDOW of a position."""
        nearby = []
        for text, label, start, end in ner_entities:
            distance = min(abs(start - position), abs(end - position))
            if distance <= PROXIMITY_WINDOW:
                nearby.append((text, label, start, end))
        return nearby


def _persons(nearby: List[Tuple]) -> set:
    """Extract text of PERSON entities from nearby list."""
    return {text for text, label, _, _ in nearby if label == "PERSON"}


def _compute_line_and_offset(text: str, position: int) -> Tuple[int, int]:
    """Convert absolute position to (line_number, char_offset)."""
    lines_before = text[:position].count("\n")
    line_start = text.rfind("\n", 0, position) + 1
    return lines_before + 1, position - line_start
