"""Layer 4: SLM-based secret/credential detection.

A fine-tuned small language model (Llama 3.2 3B) that detects passwords,
API keys, tokens, JWT secrets, private keys, connection strings, and
webhook secrets — categories that regex and NER are structurally unable
to catch because there are no fixed patterns for arbitrary secrets.

Uses the Ollama chat API (not generate) because the model was trained
with chat-format messages. Scans individual candidate lines with
surrounding context, matching the exact prompt format from training.
"""

import json
import math
import re
import requests
from collections import Counter
from typing import Dict, List, Optional, Tuple

from connector.models import RawDocument
from features.compute import DocumentFeatures
from pipeline.models import Entity


OLLAMA_CHAT_API = "http://localhost:11434/api/chat"

# Keywords that suggest a line might contain a secret
SECRET_KEYWORDS = {
    "password", "passwd", "pwd", "pass",
    "secret", "key", "token", "api_key", "apikey",
    "credential", "auth", "private",
    "access_key", "secret_key",
    "connection_string", "conn_str", "database_url", "db_url",
    "jwt", "signing",
    "BEGIN RSA", "BEGIN PRIVATE", "BEGIN EC PRIVATE",
}

# Assignment-like patterns: key=value, key: value, key = "value"
ASSIGNMENT_RE = re.compile(
    r"""(?:=|:|=>)\s*["']?[^\s"']{8,}""",
    re.IGNORECASE,
)


def _line_entropy(line: str) -> float:
    """Shannon entropy of a string — high entropy suggests random secrets."""
    if not line:
        return 0.0
    counts = Counter(line)
    length = len(line)
    return -sum(
        (c / length) * math.log2(c / length) for c in counts.values()
    )


def _file_type_to_context(file_type: str, path: str) -> str:
    """Map document file_type to the context_type labels used in training."""
    ft = file_type.lower()
    pl = path.lower()

    if ft == "py" or pl.endswith(".py"):
        return "python source code"
    elif ft == "yaml" or ft == "yml":
        return "yaml configuration"
    elif ft == "json":
        return "json data"
    elif ft == "env" or ".env" in pl:
        return "environment variables"
    elif ft == "toml":
        return "toml configuration"
    elif ft in ("sh", "bash", "zsh"):
        return "shell script"
    elif ft in ("js", "ts"):
        return "javascript source code"
    elif ft == "csv":
        return "csv data"
    elif ft == "xml":
        return "xml configuration"
    elif ft == "txt":
        # Infer from path context
        if "audit" in pl or "security" in pl:
            return "security audit report"
        if "password" in pl or "credential" in pl:
            return "credential dump"
        return "text file"
    else:
        return "configuration file"


class SLMClassifier:
    """Uses a fine-tuned SLM via Ollama to detect secrets and credentials."""

    def __init__(self, model: str = "secret-scanner") -> None:
        self._model = model
        self.total_calls = 0

    def classify(self, doc: RawDocument, features: DocumentFeatures,
                 entities: List[Entity]) -> List[Entity]:
        """Scan document lines for secrets using the fine-tuned SLM.

        Args:
            doc: Full document.
            features: Computed features for context type mapping.
            entities: All entities from prior layers.

        Returns:
            Updated entity list with any new secret discoveries appended.
        """
        lines = doc.content.splitlines()
        candidates = self._find_candidate_lines(lines)

        if not candidates:
            return entities

        context_type = _file_type_to_context(
            features.file_type, doc.metadata.id
        )

        new_entities = []
        for line_idx in candidates:
            result = self._classify_line(lines, line_idx, context_type)
            if result is None:
                continue

            is_secret = result.get("is_secret", False)
            confidence = float(result.get("confidence", 0.0))

            if is_secret and confidence > 0.50:
                entity = self._make_entity(
                    doc, lines, line_idx, result
                )
                if entity and not self._is_duplicate(entity, entities + new_entities):
                    new_entities.append(entity)

        return entities + new_entities

    def _find_candidate_lines(self, lines: List[str]) -> List[int]:
        """Pre-filter lines that could plausibly contain secrets.

        Heuristics:
        1. Line contains a secret-related keyword near an assignment
        2. Line has a high-entropy value after an assignment operator
        3. Line contains PEM header markers
        """
        candidates = []
        for i, line in enumerate(lines):
            lower = line.lower().strip()
            if not lower or lower.startswith("#") and len(lower) < 5:
                continue

            # Check for keyword + assignment pattern
            has_keyword = any(kw in lower for kw in SECRET_KEYWORDS)
            has_assignment = ASSIGNMENT_RE.search(line) is not None

            if has_keyword and has_assignment:
                candidates.append(i)
                continue

            # Check for high-entropy values after assignment
            if has_assignment:
                # Extract the value portion after = or :
                match = re.search(r'[=:]\s*["\']?(.{12,})', line)
                if match:
                    value = match.group(1).strip("\"' ")
                    if _line_entropy(value) > 4.0:
                        candidates.append(i)
                        continue

            # PEM markers
            if "BEGIN" in line and "PRIVATE" in line:
                candidates.append(i)

        return candidates

    def _classify_line(self, lines: List[str], target_idx: int,
                       context_type: str) -> Optional[Dict]:
        """Build prompt for a single line and call the SLM."""
        # Build context window: 2 lines before, target, 2 lines after
        start = max(0, target_idx - 2)
        end = min(len(lines), target_idx + 3)
        context_lines = lines[start:end]
        target_in_context = target_idx - start

        prompt = self._build_prompt(context_lines, target_in_context,
                                    context_type)
        response = self._call_ollama(prompt)
        if response is None:
            return None

        return self._parse_response(response)

    def _build_prompt(self, context_lines: List[str],
                      target_idx: int, context_type: str) -> str:
        """Build the exact prompt format used during training.

        Highlighted line gets >>> prefix, context lines get 2-space indent.
        """
        formatted = []
        for i, line in enumerate(context_lines):
            if i == target_idx:
                formatted.append(f">>> {line}")
            else:
                formatted.append(f"  {line}")

        snippet = "\n".join(formatted)
        return (
            "Classify whether the highlighted line contains a secret, "
            "credential, or sensitive key.\n\n"
            f"Context type: {context_type}\n\n"
            f"{snippet}"
        )

    def _call_ollama(self, prompt: str) -> Optional[str]:
        """Call Ollama chat API and return the assistant response."""
        try:
            resp = requests.post(OLLAMA_CHAT_API, json={
                "model": self._model,
                "messages": [
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
                "options": {
                    "temperature": 0.1,
                },
            }, timeout=30)
            resp.raise_for_status()
            result = resp.json()
            self.total_calls += 1
            return result.get("message", {}).get("content", "")
        except (requests.RequestException, KeyError) as e:
            print(f"    [SLM WARNING] Ollama call failed: {e}")
            return None

    def _parse_response(self, raw: str) -> Optional[Dict]:
        """Parse JSON response from SLM output.

        Expected format:
            {"is_secret": true, "type": "password", "confidence": 0.95,
             "reasoning": "..."}
        """
        raw = raw.strip()
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(raw[start:end + 1])
            except json.JSONDecodeError:
                pass
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    def _make_entity(self, doc: RawDocument, lines: List[str],
                     line_idx: int, result: Dict) -> Optional[Entity]:
        """Create an Entity from an SLM classification result."""
        line = lines[line_idx]
        secret_type = result.get("type", "secret")
        confidence = float(result.get("confidence", 0.70))

        # Try to extract the secret value from the line
        matched_text = self._extract_value(line)
        if not matched_text:
            matched_text = line.strip()

        # Calculate position in document
        position = sum(len(l) + 1 for l in lines[:line_idx])
        char_offset = len(line) - len(line.lstrip())

        return Entity(
            document_id=doc.metadata.id,
            entity_type=secret_type,
            matched_text=matched_text,
            match_length=len(matched_text),
            line_number=line_idx + 1,  # 1-based
            char_offset=char_offset,
            position=position + char_offset,
            confidence=confidence,
            pattern_name="slm_discovery",
            classified_by_layer=4,
            layer1_confidence=None,
            layer2_confidence=None,
            layer3_confidence=None,
            layer4_confidence=confidence,
        )

    def _extract_value(self, line: str) -> str:
        """Extract the secret value portion from an assignment line.

        Looks for the value after =, :, or => operators.
        """
        # Match: key = "value" or key: value or key=value
        match = re.search(
            r'[=:]\s*["\']?([^\s"\'#,;}{)]+(?:\s+[^\s"\'#,;}{)]+)*)',
            line
        )
        if match:
            value = match.group(1).strip("\"' ")
            if len(value) >= 4:
                return value

        return ""

    def _is_duplicate(self, new: Entity,
                      existing: List[Entity]) -> bool:
        """Check if an entity at this location already exists."""
        for e in existing:
            if e.document_id != new.document_id:
                continue
            # Same line — consider it a duplicate
            if abs(e.line_number - new.line_number) <= 1:
                return True
        return False
