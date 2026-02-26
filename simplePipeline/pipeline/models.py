"""Data classes for pipeline results.

Defines the entity model that all classification layers produce.
Kept separate from layer logic so every layer outputs the same type.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Entity:
    """A single PII finding at a specific location in a document.

    Carries enough information to:
    - Trace back to the source document and exact position
    - Understand what was found and how confident we are
    - Know which pipeline layer made the decision
    - Perform redaction (position + match_length)
    """

    document_id: str       # which document this came from
    entity_type: str       # ssn, email, credit_card, phone, person, etc.
    matched_text: str      # the actual matched value (for redaction)
    match_length: int      # length of the match (for redaction offset math)

    # Location — precise enough for redaction
    line_number: int       # 1-based
    char_offset: int       # 0-based offset from start of line
    position: int          # 0-based offset from start of document

    # Classification
    confidence: float      # 0.0-1.0, current confidence after all layers so far
    pattern_name: str      # which pattern or model matched (e.g. "ssn_dashed")
    classified_by_layer: int = 0  # which layer made the final call (1=regex, 2=ner, 3=llm)

    # Confidence history — tracks how confidence evolved through layers
    layer1_confidence: Optional[float] = None
    layer2_confidence: Optional[float] = None
    layer3_confidence: Optional[float] = None
    layer4_confidence: Optional[float] = None  # SLM layer
