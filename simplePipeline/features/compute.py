"""Shared feature computation module.

Computes document-level features used by all pipeline layers.
This is the 'feature store' pattern — compute once, use everywhere.
"""

import csv
import io
import json
import math
import os
from collections import Counter
from dataclasses import dataclass, field
from typing import List

import yaml

from connector.models import RawDocument


@dataclass
class DocumentFeatures:
    """Computed features for a single document.

    Every pipeline layer receives the same features — ensuring
    consistency and avoiding redundant computation.
    """

    file_type: str
    text_length: int
    line_count: int
    has_structured_headers: bool
    header_names: List[str]
    header_pii_similarity_score: float  # 0.0-1.0
    digit_density: float  # ratio of digits to total chars
    path_context: str  # parent directory name, e.g. "hr", "public"
    entropy_score: float  # Shannon entropy of the text
    pipeline_version: str  # e.g. "features-v1" — tracks which logic produced these


class FeatureComputer:
    """Computes DocumentFeatures from a RawDocument.

    Loads PII vocabulary once at init, reuses for all documents.
    """

    VERSION = "features-v1"

    def __init__(self, vocab_path: str = None) -> None:
        if vocab_path is None:
            vocab_path = os.path.join(
                os.path.dirname(__file__), "pii_vocabulary.yaml"
            )
        with open(vocab_path, "r") as f:
            self._pii_vocab: List[str] = yaml.safe_load(f)

    def compute(self, doc: RawDocument) -> DocumentFeatures:
        """Compute all features for a document."""
        headers = self._extract_headers(doc)

        return DocumentFeatures(
            file_type=doc.metadata.file_type,
            text_length=len(doc.content),
            line_count=doc.content.count("\n") + 1,
            has_structured_headers=len(headers) > 0,
            header_names=headers,
            header_pii_similarity_score=self._score_headers(headers),
            digit_density=self._compute_digit_density(doc.content),
            path_context=self._extract_path_context(doc.metadata.id),
            entropy_score=self._compute_entropy(doc.content),
            pipeline_version=self.VERSION,
        )

    def _extract_headers(self, doc: RawDocument) -> List[str]:
        """Extract column names or key names from structured documents."""
        file_type = doc.metadata.file_type

        if file_type == "csv":
            reader = csv.reader(io.StringIO(doc.content))
            try:
                first_row = next(reader)
                # Heuristic: if no cell looks numeric, it's a header row
                if not all(cell.replace(".", "").replace("-", "").isdigit()
                           for cell in first_row if cell.strip()):
                    return [h.strip().lower() for h in first_row]
            except StopIteration:
                pass
            return []

        if file_type == "json":
            try:
                data = json.loads(doc.content)
                return self._extract_json_keys(data)
            except json.JSONDecodeError:
                return []

        if file_type == "yaml":
            try:
                data = yaml.safe_load(doc.content)
                if isinstance(data, dict):
                    return [str(k).lower() for k in data.keys()]
            except yaml.YAMLError:
                return []

        return []

    def _extract_json_keys(self, data, depth: int = 0) -> List[str]:
        """Recursively extract keys from JSON, up to 2 levels deep."""
        if depth > 2:
            return []
        keys = []
        if isinstance(data, dict):
            for k, v in data.items():
                keys.append(str(k).lower())
                keys.extend(self._extract_json_keys(v, depth + 1))
        elif isinstance(data, list) and data:
            # Sample first element only
            keys.extend(self._extract_json_keys(data[0], depth + 1))
        return keys

    def _score_headers(self, headers: List[str]) -> float:
        """Score how PII-like the headers are, 0.0 to 1.0.

        Compares each header against the PII vocabulary using
        substring matching. Returns the fraction of headers that
        match at least one vocab term.
        """
        if not headers:
            return 0.0

        matches = 0
        for header in headers:
            normalized = header.lower().replace(" ", "_").replace("-", "_")
            for term in self._pii_vocab:
                if term in normalized or normalized in term:
                    matches += 1
                    break

        return matches / len(headers)

    @staticmethod
    def _compute_digit_density(text: str) -> float:
        """Ratio of digit characters to total characters."""
        if not text:
            return 0.0
        digit_count = sum(1 for c in text if c.isdigit())
        return digit_count / len(text)

    @staticmethod
    def _extract_path_context(doc_id: str) -> str:
        """Extract the parent directory name as business context.

        'hr/w2_forms.csv' → 'hr'
        'employees.csv' → '' (root level)
        """
        dirname = os.path.dirname(doc_id)
        return dirname if dirname else ""

    @staticmethod
    def _compute_entropy(text: str) -> float:
        """Shannon entropy of the text.

        Higher entropy = more randomness/variety in characters.
        Encrypted data, hashes, and dense numeric data (like SSNs)
        tend to have higher entropy than natural prose.
        """
        if not text:
            return 0.0
        counts = Counter(text)
        length = len(text)
        return -sum(
            (count / length) * math.log2(count / length)
            for count in counts.values()
        )
