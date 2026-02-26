"""Layer 3: LLM-based PII validation and discovery.

The most expensive and capable layer. Only invoked when a document
still has ambiguous entities after regex + NER. When called, gets
the full document + all features + specific questions about
ambiguous entities, plus an open-ended ask for anything missed.

Uses Ollama for local model inference. Model is configurable.
"""

import json
import re
import requests
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

from connector.models import RawDocument
from features.compute import DocumentFeatures
from pipeline.models import Entity

if TYPE_CHECKING:
    from rag.store import VectorStore


OLLAMA_API = "http://localhost:11434/api/generate"


def _build_rag_section(ambiguous_entities: List[Entity],
                       vector_store: "VectorStore") -> str:
    """Retrieve similar past classifications for ambiguous entities.

    For each ambiguous entity, queries the vector store and formats
    matching examples with similarity scores for the LLM prompt.
    """
    from rag.embed import embed
    from rag.store import STRONG_THRESHOLD

    sections = []
    for e in ambiguous_entities:
        query_text = f"{e.matched_text} → {e.entity_type}"
        try:
            query_vec = embed(query_text)
        except Exception:
            continue

        results = vector_store.search(query_vec, top_k=3)
        if not results:
            continue

        lines = [f'  For "{e.matched_text}" ({e.entity_type}):']
        for r in results:
            strength = "Strong" if r.similarity >= STRONG_THRESHOLD else "Weak"
            lines.append(
                f"    [{strength} match, similarity {r.similarity:.2f}] "
                f'"{r.matched_text}" → {r.entity_type}, '
                f"confidence {r.confidence:.2f}, "
                f"detected by: {r.classified_by_layer}/{r.pattern_name} "
                f"(from {r.document_context})"
            )
        sections.append("\n".join(lines))

    if not sections:
        return ""

    return (
        "SIMILAR PAST CLASSIFICATIONS (from training data):\n"
        + "\n".join(sections) + "\n\n"
        "Use these examples to inform your assessment. "
        "Stronger matches (higher similarity) are more relevant.\n\n"
    )


def _build_prompt(doc: RawDocument, features: DocumentFeatures,
                  ambiguous_entities: List[Entity],
                  all_entities: List[Entity],
                  rag_context: str = "") -> str:
    """Build the prompt sent to the LLM.

    Includes:
    - Full document text
    - Computed features as context
    - Specific questions about ambiguous entities
    - Open-ended ask for anything we missed
    """
    # Summarize what we already know
    confident_summary = ""
    confident = [e for e in all_entities if e not in ambiguous_entities]
    if confident:
        confident_summary = "Already confirmed with high confidence:\n"
        for e in confident:
            confident_summary += f"  - {e.entity_type} at line {e.line_number} (confidence {e.confidence:.2f})\n"

    ambiguous_summary = ""
    for i, e in enumerate(ambiguous_entities, 1):
        ambiguous_summary += (
            f"  {i}. \"{e.matched_text}\" at line {e.line_number} — "
            f"detected as {e.entity_type} by {e.pattern_name}, "
            f"current confidence {e.confidence:.2f}\n"
        )

    return f"""You are a data classification expert analyzing a document for PII (Personally Identifiable Information).

DOCUMENT PATH: {doc.metadata.id}
FILE TYPE: {features.file_type}
DIRECTORY CONTEXT: {features.path_context or '(root)'}
HEADER PII SCORE: {features.header_pii_similarity_score:.2f}

--- DOCUMENT CONTENT ---
{doc.content}
--- END DOCUMENT ---

{confident_summary}
{rag_context}The following entities are AMBIGUOUS and need your assessment:
{ambiguous_summary}
For each ambiguous entity, respond with:
- entity_number: (the number from the list above)
- is_pii: true or false
- entity_type: the PII type (ssn, phone, email, credit_card, person_name, or other)
- confidence: your confidence 0.0-1.0
- reasoning: brief explanation

ALSO: Are there any PII items in this document that were NOT listed above? This includes obfuscated PII (numbers spelled out, common letter substitutions, encoded values, etc). If you find any, list them with the same format plus the matched_text and line_number.

Respond ONLY with valid JSON in this format:
{{
  "assessments": [
    {{"entity_number": 1, "is_pii": true, "entity_type": "ssn", "confidence": 0.95, "reasoning": "..."}},
  ],
  "new_discoveries": [
    {{"matched_text": "...", "line_number": 1, "entity_type": "...", "confidence": 0.8, "reasoning": "..."}}
  ]
}}"""


def _parse_response(raw_response: str) -> Optional[Dict]:
    """Extract JSON from LLM response, handling markdown code blocks."""
    # Try to find JSON in code blocks first
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_response, re.DOTALL)
    if json_match:
        text = json_match.group(1)
    else:
        # Try to find raw JSON
        json_match = re.search(r"\{.*\}", raw_response, re.DOTALL)
        if json_match:
            text = json_match.group(0)
        else:
            return None

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


class LLMClassifier:
    """Uses a local LLM via Ollama to validate ambiguous PII findings."""

    def __init__(self, model: str = "gpt-oss:20b",
                 vector_store: Optional["VectorStore"] = None) -> None:
        self._model = model
        self._vector_store = vector_store
        self.total_calls = 0
        self.total_tokens = 0

    def classify(self, doc: RawDocument, features: DocumentFeatures,
                 entities: List[Entity]) -> List[Entity]:
        """Send ambiguous entities to the LLM for validation.

        Args:
            doc: Full document (LLM gets everything).
            features: Computed features for context.
            entities: All entities from prior layers.

        Returns:
            Updated entity list with LLM-adjusted confidence + any new finds.
        """
        ambiguous = [e for e in entities if e.confidence < 0.80]
        confident = [e for e in entities if e.confidence >= 0.80]

        if not ambiguous:
            return entities

        # Build RAG context if vector store is available
        rag_context = ""
        if self._vector_store:
            rag_context = _build_rag_section(ambiguous, self._vector_store)

        prompt = _build_prompt(doc, features, ambiguous, entities,
                               rag_context=rag_context)
        response = self._call_ollama(prompt)

        if response is None:
            # LLM call failed — return entities unchanged
            return entities

        parsed = _parse_response(response)
        if parsed is None:
            # Couldn't parse response — return unchanged
            return entities

        # Update ambiguous entities with LLM assessments
        updated_ambiguous = self._apply_assessments(
            ambiguous, parsed.get("assessments", [])
        )

        # Add any new discoveries
        new_entities = self._apply_discoveries(
            doc, parsed.get("new_discoveries", [])
        )

        return confident + updated_ambiguous + new_entities

    def _call_ollama(self, prompt: str) -> Optional[str]:
        """Call Ollama API and return the full response text."""
        try:
            resp = requests.post(OLLAMA_API, json={
                "model": self._model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,  # Low temp for consistent classification
                },
            }, timeout=180)
            resp.raise_for_status()
            result = resp.json()
            self.total_calls += 1
            self.total_tokens += result.get("eval_count", 0)
            return result.get("response", "")
        except (requests.RequestException, KeyError) as e:
            print(f"    [LLM WARNING] Ollama call failed: {e}")
            return None

    def _apply_assessments(self, ambiguous: List[Entity],
                           assessments: List[Dict]) -> List[Entity]:
        """Update ambiguous entities based on LLM assessments."""
        # Index assessments by entity_number
        assessment_map = {}
        for a in assessments:
            num = a.get("entity_number")
            if num is not None:
                assessment_map[num] = a

        updated = []
        for i, entity in enumerate(ambiguous, 1):
            assessment = assessment_map.get(i)
            if assessment:
                llm_confidence = float(assessment.get("confidence", entity.confidence))
                # If LLM says not PII, drop confidence
                if not assessment.get("is_pii", True):
                    llm_confidence = min(llm_confidence, 0.20)
            else:
                llm_confidence = entity.confidence

            updated.append(Entity(
                document_id=entity.document_id,
                entity_type=entity.entity_type,
                matched_text=entity.matched_text,
                match_length=entity.match_length,
                line_number=entity.line_number,
                char_offset=entity.char_offset,
                position=entity.position,
                confidence=llm_confidence,
                pattern_name=entity.pattern_name,
                classified_by_layer=3,
                layer1_confidence=entity.layer1_confidence,
                layer2_confidence=entity.layer2_confidence,
                layer3_confidence=llm_confidence,
            ))

        return updated

    def _apply_discoveries(self, doc: RawDocument,
                           discoveries: List[Dict]) -> List[Entity]:
        """Create entities from new LLM discoveries."""
        new_entities = []
        for d in discoveries:
            matched_text = d.get("matched_text", "")
            line_number = d.get("line_number", 0)
            if not matched_text:
                continue

            # Find position in document
            position = doc.content.find(matched_text)
            if position == -1:
                position = 0

            new_entities.append(Entity(
                document_id=doc.metadata.id,
                entity_type=d.get("entity_type", "unknown"),
                matched_text=matched_text,
                match_length=len(matched_text),
                line_number=line_number,
                char_offset=0,
                position=position,
                confidence=float(d.get("confidence", 0.50)),
                pattern_name="llm_discovery",
                classified_by_layer=3,
                layer1_confidence=None,
                layer2_confidence=None,
                layer3_confidence=float(d.get("confidence", 0.50)),
            ))

        return new_entities
