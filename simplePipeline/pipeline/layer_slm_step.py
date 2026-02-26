"""Pipeline step adapter for the SLM secret detection layer.

No gating — always runs on every document. The SLM's purpose is
discovery of secrets that regex and NER are structurally unable to
detect (no patterns exist for arbitrary passwords/keys). Gating
on low-confidence entities would skip documents entirely.
"""

from typing import List

from pipeline.context import PipelineContext, PipelineStep
from pipeline.layer_slm import SLMClassifier


class SLMClassificationStep(PipelineStep):
    """Runs the fine-tuned SLM for secret/credential detection."""

    @property
    def name(self) -> str:
        return "slm"

    @property
    def requires(self) -> List[str]:
        return ["document", "features", "entities"]

    @property
    def produces(self) -> List[str]:
        return ["entities"]

    def __init__(self, model: str = "secret-scanner") -> None:
        self._classifier = SLMClassifier(model=model)

    def run(self, context: PipelineContext) -> None:
        doc = context.get("document")
        features = context.get("features")
        entities = context.get("entities")
        updated = self._classifier.classify(doc, features, entities)
        context.set("entities", updated)
