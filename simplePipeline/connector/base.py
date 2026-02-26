"""Abstract connector interface.

Defines WHAT a connector does, not HOW. All downstream code
should depend on this interface, never on a specific implementation.
"""

from abc import ABC, abstractmethod
from typing import List

from connector.models import DocumentMetadata, RawDocument


class Connector(ABC):
    """Interface for discovering and fetching documents from any source."""

    @abstractmethod
    def discover(self) -> List[DocumentMetadata]:
        """List all available documents in the source.

        Returns metadata only — no content is loaded.
        """
        ...

    @abstractmethod
    def fetch(self, doc_id: str) -> RawDocument:
        """Retrieve a document's content by its ID.

        Args:
            doc_id: The document identifier (as returned by discover()).

        Raises:
            KeyError: If no document with the given ID exists.
        """
        ...
