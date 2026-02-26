"""Feature computation as a pipeline step.

Wraps FeatureComputer to conform to the PipelineStep interface.
Keeps compute.py clean — it doesn't need to know about the pipeline.
"""

from typing import List

from features.compute import FeatureComputer
from pipeline.context import PipelineContext, PipelineStep


class FeatureComputationStep(PipelineStep):
    """Computes document features and adds them to the context."""

    @property
    def name(self) -> str:
        return "features"

    @property
    def requires(self) -> List[str]:
        return ["document"]

    @property
    def produces(self) -> List[str]:
        return ["features"]

    def __init__(self) -> None:
        self._computer = FeatureComputer()

    def run(self, context: PipelineContext) -> None:
        doc = context.get("document")
        features = self._computer.compute(doc)
        context.set("features", features)
