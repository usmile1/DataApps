"""Pipeline context (the 'bag') and step interface.

The context is a flexible container that steps read from and write to.
Each step declares what it requires and produces, so the pipeline can
validate the data flow before running.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Set


class PipelineContext:
    """Typed bag that accumulates results as steps run.

    Values are stored by key (e.g., 'features', 'regex_results').
    The actual values are typed objects (dataclasses), not raw dicts.
    """

    def __init__(self) -> None:
        self._data: Dict[str, Any] = {}

    def set(self, key: str, value: Any) -> None:
        """Add or update a value in the context."""
        self._data[key] = value

    def get(self, key: str) -> Any:
        """Retrieve a value. Raises KeyError if not present."""
        if key not in self._data:
            raise KeyError(
                f"'{key}' not found in context. "
                f"Available keys: {list(self._data.keys())}"
            )
        return self._data[key]

    def has(self, key: str) -> bool:
        """Check whether a key exists in the context."""
        return key in self._data

    def keys(self) -> Set[str]:
        """Return all keys currently in the context."""
        return set(self._data.keys())


class PipelineStep(ABC):
    """Interface for a pluggable pipeline step.

    Each step declares:
      - requires: keys that must exist in the context before this step runs
      - produces: keys that this step will add to the context

    The pipeline validates these before execution, so a misconfigured
    pipeline fails fast with a clear error rather than silently producing
    wrong results.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable step name for logging and config."""
        ...

    @property
    @abstractmethod
    def requires(self) -> List[str]:
        """Context keys this step needs to run."""
        ...

    @property
    @abstractmethod
    def produces(self) -> List[str]:
        """Context keys this step will add."""
        ...

    @abstractmethod
    def run(self, context: PipelineContext) -> None:
        """Execute the step, reading from and writing to the context."""
        ...
