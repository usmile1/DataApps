"""Layer 1: Regex-based PII detection.

Fast, cheap pattern matching. The first line of defense.
High-confidence matches stop here. Low-confidence matches
escalate to Layer 2 (NER) for contextual validation.
"""

import re
from typing import List, Tuple

from connector.models import RawDocument
from features.compute import DocumentFeatures
from pipeline.models import Entity


# --- Pattern definitions ---
# Each pattern: (entity_type, pattern_name, compiled_regex)

PATTERNS: List[Tuple[str, str, re.Pattern]] = [
    # SSN: dashed format
    ("ssn", "ssn_dashed", re.compile(r"\b(\d{3}-\d{2}-\d{4})\b")),

    # Email
    ("email", "email_standard", re.compile(
        r"\b([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b"
    )),

    # Credit card: various formats (13-19 digits, optional dashes/spaces)
    ("credit_card", "cc_with_separators", re.compile(
        r"\b(\d{4}[-\s]\d{4}[-\s]\d{4}[-\s]\d{4})\b"
    )),
    ("credit_card", "cc_contiguous", re.compile(
        r"\b(\d{13,19})\b"
    )),

    # Phone: US formats
    ("phone", "phone_parens", re.compile(
        r"\((\d{3})\)\s*(\d{3})-(\d{4})"
    )),
    ("phone", "phone_dashed", re.compile(
        r"\b(\d{3}-\d{3}-\d{4})\b"
    )),
    ("phone", "phone_tollfree", re.compile(
        r"\b(1-800-\d{3}-\d{4})\b"
    )),
]

# PII-suggestive header terms — if a match is near a column with
# one of these names, confidence gets a boost
PII_HEADER_TERMS = {"ssn", "social_security", "social_security_number",
                    "email", "email_address", "phone", "phone_number",
                    "card_number", "credit_card", "cc_number"}


def luhn_check(number_str: str) -> bool:
    """Validate a credit card number using the Luhn algorithm.

    Strips spaces and dashes, then checks the checksum.
    Returns False for numbers that are clearly not credit cards.
    """
    digits = [int(d) for d in number_str if d.isdigit()]
    if len(digits) < 13 or len(digits) > 19:
        return False

    # Luhn: double every second digit from the right
    checksum = 0
    for i, digit in enumerate(reversed(digits)):
        if i % 2 == 1:
            doubled = digit * 2
            checksum += doubled - 9 if doubled > 9 else doubled
        else:
            checksum += digit

    return checksum % 10 == 0


def _compute_line_and_offset(text: str, position: int) -> Tuple[int, int]:
    """Convert an absolute position to (line_number, char_offset).

    Line numbers are 1-based. Char offsets are 0-based within the line.
    """
    lines_before = text[:position].count("\n")
    line_start = text.rfind("\n", 0, position) + 1
    return lines_before + 1, position - line_start


def _base_confidence(entity_type: str, pattern_name: str,
                     matched_text: str) -> float:
    """Assign base confidence from the pattern match alone."""
    if entity_type == "ssn":
        return 0.50

    if entity_type == "email":
        # Emails with valid-looking TLDs are high confidence
        return 0.90

    if entity_type == "credit_card":
        base = 0.60
        if luhn_check(matched_text):
            base += 0.25  # Luhn-valid is a strong signal, not a hard filter
        return base

    if entity_type == "phone":
        if pattern_name == "phone_parens":
            return 0.70  # (555) 234-5678 is a strong phone signal
        if pattern_name == "phone_tollfree":
            return 0.80
        return 0.50  # dashed format is ambiguous

    return 0.40


def _boost_confidence(base: float, entity_type: str,
                      features: DocumentFeatures,
                      line_number: int) -> float:
    """Adjust confidence using document-level features.

    This is where the feature computation pays off — same pattern
    match gets different confidence depending on context.
    """
    confidence = base

    # Boost: header names suggest PII
    if features.has_structured_headers:
        matching_headers = PII_HEADER_TERMS & set(features.header_names)
        if matching_headers:
            confidence += 0.30

    # Boost: structured file (CSV/JSON) — patterns are more intentional
    if features.file_type in ("csv", "json"):
        confidence += 0.10

    # Small boost: path context suggests PII is expected here
    if features.path_context in ("hr", "finance", "payroll"):
        confidence += 0.05

    return min(confidence, 1.0)


class RegexClassifier:
    """Runs regex patterns against a document and produces Entity results."""

    def classify(self, doc: RawDocument,
                 features: DocumentFeatures) -> List[Entity]:
        """Find all regex matches and score their confidence."""
        entities = []
        text = doc.content

        for entity_type, pattern_name, pattern in PATTERNS:
            for match in pattern.finditer(text):
                matched_text = match.group(0)
                position = match.start()

                line_number, char_offset = _compute_line_and_offset(
                    text, position
                )

                base = _base_confidence(entity_type, pattern_name, matched_text)
                confidence = _boost_confidence(
                    base, entity_type, features, line_number
                )

                entities.append(Entity(
                    document_id=doc.metadata.id,
                    entity_type=entity_type,
                    matched_text=matched_text,
                    match_length=len(matched_text),
                    line_number=line_number,
                    char_offset=char_offset,
                    position=position,
                    confidence=confidence,
                    pattern_name=pattern_name,
                    classified_by_layer=1,
                    layer1_confidence=confidence,
                ))

        return entities
