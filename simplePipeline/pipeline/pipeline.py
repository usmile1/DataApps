"""Pipeline orchestrator.

Reads a config file, instantiates steps, validates the data flow,
and runs each step in sequence. The 'preschool orchestrator.'
"""

import importlib
import time
from typing import Dict, List

import yaml

from pipeline.context import PipelineContext, PipelineStep


class Pipeline:
    """Configurable, step-based document processing pipeline."""

    def __init__(self, steps: List[PipelineStep]) -> None:
        self._steps = steps
        self._validate_flow()

    def _validate_flow(self) -> None:
        """Check that every step's requirements are met by prior steps.

        Fails fast at construction time, not at runtime.
        """
        available_keys = {"document"}  # connector always provides this
        for step in self._steps:
            missing = set(step.requires) - available_keys
            if missing:
                raise ValueError(
                    f"Step '{step.name}' requires {missing}, but only "
                    f"{available_keys} are available at that point. "
                    f"Check step ordering in config."
                )
            available_keys.update(step.produces)

    def run(self, context: PipelineContext) -> Dict[str, float]:
        """Run all steps in sequence. Returns timing info per step."""
        timings = {}
        for step in self._steps:
            start = time.perf_counter()
            step.run(context)
            elapsed_ms = (time.perf_counter() - start) * 1000
            timings[step.name] = elapsed_ms
        return timings

    @classmethod
    def from_config(cls, config_path: str) -> "Pipeline":
        """Build a Pipeline from a YAML config file."""
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        steps = []
        for step_config in config["pipeline"]["steps"]:
            step_class = _import_class(step_config["class"])
            # Pass any extra config keys as kwargs to the step constructor
            kwargs = {k: v for k, v in step_config.items()
                      if k not in ("name", "class")}
            steps.append(step_class(**kwargs))

        return cls(steps)

    @staticmethod
    def derive_version(config: dict) -> str:
        """Derive a pipeline version string from the config.

        Captures the step names and any model parameters so that
        different configurations produce different version strings.
        """
        parts = []
        for step in config["pipeline"]["steps"]:
            name = step["name"]
            # Include model name if present
            model = step.get("model", "")
            if model:
                parts.append(f"{name}({model})")
            else:
                parts.append(name)
        return "→".join(parts)


def _import_class(dotted_path: str):
    """Import a class from a dotted path like 'features.step.FeatureComputationStep'."""
    module_path, class_name = dotted_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)
