"""Pipeline step adapter for the NER classification layer.

Contains the gating logic: only runs NER on documents that have
entities below the confidence threshold after Layer 1.
"""

from typing import List

from pipeline.context import PipelineContext, PipelineStep
from pipeline.layer2_ner import NERClassifier

# If all entities are at or above this, skip NER entirely
ESCALATION_THRESHOLD = 0.90


class NERClassificationStep(PipelineStep):
    """Runs SpaCy NER for contextual PII validation."""

    @property
    def name(self) -> str:
        return "ner"

    @property
    def requires(self) -> List[str]:
        return ["document", "features", "entities"]

    @property
    def produces(self) -> List[str]:
        return ["entities"]  # replaces the entity list with updated version

    def __init__(self) -> None:
        self._classifier = NERClassifier()

    def run(self, context: PipelineContext) -> None:
        entities = context.get("entities")

        # Gate: skip if no entities need escalation
        needs_escalation = any(
            e.confidence < ESCALATION_THRESHOLD for e in entities
        )
        if not needs_escalation:
            return

        doc = context.get("document")
        features = context.get("features")
        updated = self._classifier.classify(doc, features, entities)
        context.set("entities", updated)
