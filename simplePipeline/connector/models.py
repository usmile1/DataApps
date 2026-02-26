"""Data classes that define the core document types used across the pipeline."""

from dataclasses import dataclass
from datetime import datetime


@dataclass
class DocumentMetadata:
    """Describes a document without loading its content.

    Used for discovery, listing, routing, and logging — anywhere
    you need to talk about a document without reading it into memory.
    """

    id: str  # relative path, e.g. "hr/w2_forms.csv"
    path: str  # absolute path on disk
    file_type: str  # extension: csv, txt, json, yaml, md
    size_bytes: int
    last_modified: datetime


@dataclass
class RawDocument:
    """A document's metadata plus its text content.

    Composition over inheritance — metadata can be passed around
    independently without dragging the full content along.
    """

    metadata: DocumentMetadata
    content: str
