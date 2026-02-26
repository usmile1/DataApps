"""Pipeline step adapter for the LLM classification layer.

Contains gating logic: only calls the LLM when a document has
entities still below confidence threshold after regex + NER.
"""

from typing import List

from pipeline.context import PipelineContext, PipelineStep
from pipeline.layer3_llm import LLMClassifier

# Only invoke LLM if any entity is still below this
ESCALATION_THRESHOLD = 0.80


class LLMClassificationStep(PipelineStep):
    """Runs LLM validation for ambiguous PII findings."""

    @property
    def name(self) -> str:
        return "llm"

    @property
    def requires(self) -> List[str]:
        return ["document", "features", "entities"]

    @property
    def produces(self) -> List[str]:
        return ["entities"]

    def __init__(self, model: str = "gpt-oss:20b",
                 vector_store=None) -> None:
        self._classifier = LLMClassifier(model=model,
                                          vector_store=vector_store)

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
