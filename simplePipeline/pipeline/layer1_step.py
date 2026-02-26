"""Pipeline step adapter for the regex classification layer."""

from typing import List

from pipeline.context import PipelineContext, PipelineStep
from pipeline.layer1_regex import RegexClassifier


class RegexClassificationStep(PipelineStep):
    """Runs regex PII detection and adds entities to the context."""

    @property
    def name(self) -> str:
        return "regex"

    @property
    def requires(self) -> List[str]:
        return ["document", "features"]

    @property
    def produces(self) -> List[str]:
        return ["entities"]

    def __init__(self) -> None:
        self._classifier = RegexClassifier()

    def run(self, context: PipelineContext) -> None:
        doc = context.get("document")
        features = context.get("features")
        entities = self._classifier.classify(doc, features)
        context.set("entities", entities)
