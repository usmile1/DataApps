"""Filesystem connector implementation.

This is the only module that knows how to read local files.
All filesystem-specific logic (path walking, file reading) stays here.
"""

import mimetypes
import os
from datetime import datetime, timezone
from typing import List

from connector.base import Connector
from connector.models import DocumentMetadata, RawDocument


class FilesystemConnector(Connector):
    """Discovers and fetches documents from a local directory."""

    # Files matching these patterns are skipped during discovery
    DEFAULT_EXCLUDES = {"test_labels.json"}

    def __init__(self, root_dir: str, exclude: List[str] = None) -> None:
        self._root = os.path.abspath(root_dir)
        if not os.path.isdir(self._root):
            raise ValueError(f"Not a directory: {self._root}")
        self._exclude = set(exclude) if exclude else self.DEFAULT_EXCLUDES

    def discover(self) -> List[DocumentMetadata]:
        """Recursively walk root_dir and return metadata for all files."""
        documents = []
        for dirpath, _, filenames in os.walk(self._root):
            for filename in filenames:
                if filename in self._exclude:
                    continue
                abs_path = os.path.join(dirpath, filename)
                rel_path = os.path.relpath(abs_path, self._root)
                stat = os.stat(abs_path)

                documents.append(DocumentMetadata(
                    id=rel_path,
                    path=abs_path,
                    file_type=self._detect_file_type(abs_path),
                    size_bytes=stat.st_size,
                    last_modified=datetime.fromtimestamp(
                        stat.st_mtime, tz=timezone.utc
                    ),
                ))
        return documents

    def fetch(self, doc_id: str) -> RawDocument:
        """Read a document's content by its relative path ID."""
        abs_path = os.path.join(self._root, doc_id)
        if not os.path.isfile(abs_path):
            raise KeyError(f"Document not found: {doc_id}")

        metadata = DocumentMetadata(
            id=doc_id,
            path=abs_path,
            file_type=self._detect_file_type(abs_path),
            size_bytes=os.path.getsize(abs_path),
            last_modified=datetime.fromtimestamp(
                os.stat(abs_path).st_mtime, tz=timezone.utc
            ),
        )

        with open(abs_path, "r", encoding="utf-8") as f:
            content = f.read()

        return RawDocument(metadata=metadata, content=content)

    @staticmethod
    def _detect_file_type(path: str) -> str:
        """Detect file type via mimetypes, fall back to extension."""
        mime_type, _ = mimetypes.guess_type(path)
        if mime_type:
            # e.g. "text/csv" → "csv", "application/json" → "json"
            subtype = mime_type.split("/")[-1]
            # mimetypes returns "plain" for .txt — normalize to "txt"
            if subtype == "plain":
                return "txt"
            return subtype
        # Fallback: use extension without the dot
        _, ext = os.path.splitext(path)
        return ext.lstrip(".").lower() if ext else "unknown"
