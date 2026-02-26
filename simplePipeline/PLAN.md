# Build Plan: SimplePipeline

## Phase 1: Sample Data & Connector (~20 min)

### 1a: Create sample documents
Create 8-10 sample files in `sample_docs/` with a mix of:

**Obvious PII (should be caught by regex):**
- `employees.csv` — CSV with columns: name, email, ssn, phone. 5-10 rows of fake data.
- `customer_notes.txt` — Free text with embedded SSNs ("Customer SSN: 123-45-6789") and email addresses.
- `payment_log.json` — JSON with credit card numbers in a "card_number" field.

**Ambiguous PII (regex might catch, NER adds confidence):**
- `meeting_notes.txt` — Free text mentioning people by name, with phone numbers embedded in sentences naturally ("Call John at 555-867-5309 to confirm").
- `config.yaml` — Config file with a 9-digit number that looks like an SSN but is actually an API ID.

**Business context matters:**
- `hr/w2_forms.csv` — SSNs in an HR directory (expected, lower risk based on business context).
- `public/shared_doc.txt` — SSNs in a public-facing directory (unexpected, high risk).

**Clean files (no PII — tests false positive rate):**
- `readme.md` — Project documentation, no PII.
- `quarterly_metrics.csv` — Revenue numbers, percentages. No PII but lots of numbers that naive regex might flag.

### 1b: Build the connector
Create `connector/base.py` with an abstract `Connector` interface:
- `discover() -> List[DocumentMetadata]` — list available documents
- `fetch(doc_id) -> RawDocument` — retrieve document content

Create `connector/filesystem.py` implementing the interface for local files.

Define data classes:
- `DocumentMetadata`: id, path, file_type, size_bytes, last_modified
- `RawDocument`: metadata + raw text content

### 1c: Verify
Run the connector against `sample_docs/`, print discovered documents.

---

## Phase 2: Feature Computation + Regex Layer (~30 min)

### 2a: Shared feature computation module
Create `features/compute.py` with a `FeatureComputer` class:

Input: `RawDocument`
Output: `DocumentFeatures` dataclass containing:
- `file_type` (str): csv, txt, json, yaml, md
- `text_length` (int): character count
- `line_count` (int)
- `has_structured_headers` (bool): does it have CSV headers or JSON keys
- `header_names` (List[str]): extracted column/key names
- `header_pii_similarity_score` (float): how much do headers look like PII field names (compare against a vocabulary: "ssn", "social_security", "email", "credit_card", "phone", "dob", etc.)
- `digit_density` (float): ratio of digit characters to total characters
- `path_context` (str): directory name (e.g., "hr", "public") for business context
- `entropy_score` (float): Shannon entropy of the text (high entropy = more random/sensitive data)

### 2b: Regex classification layer
Create `pipeline/layer1_regex.py`:

Patterns to detect:
- SSN: `\d{3}-\d{2}-\d{4}` and `\d{9}` (with context check)
- Email: standard email regex
- Credit card: 13-19 digit numbers, optional dashes/spaces, Luhn check
- Phone: various US formats

For each match, produce:
- `entity_type` (str): ssn, email, credit_card, phone
- `matched_text` (str): the matched value (redacted in output)
- `pattern_name` (str): which regex matched
- `position` (int): character offset
- `line_number` (int)
- `confidence` (float): 0.0-1.0
  - High confidence (0.9+): SSN in a column called "ssn", email with @ and valid TLD
  - Medium confidence (0.5-0.9): 9-digit number without clear context
  - Low confidence (<0.5): pattern match but likely false positive (e.g., zip+4 matching SSN pattern)

### 2c: Verify
Run feature computation + regex layer against all sample docs. Print results table showing: document, entities found, confidence scores. Check: does it catch the obvious PII? Does it flag the ambiguous cases with lower confidence? Does it false-positive on the clean files?

---

## Phase 3: SpaCy NER + Claude Validation (~45 min)

### 3a: SpaCy NER layer
Create `pipeline/layer2_ner.py`:

Use SpaCy's `en_core_web_sm` to:
- Run NER on the document text
- Extract PERSON, ORG, GPE, DATE, MONEY, CARDINAL entities
- Cross-reference with regex findings: if NER confirms a PERSON entity near a detected SSN, increase confidence
- Detect additional PII that regex missed: names, addresses, dates of birth
- Produce enriched entity classifications with updated confidence scores

Confidence adjustment logic:
- Regex found SSN + NER found PERSON nearby → confidence boost (+0.15)
- Regex found 9-digit number + NER found no person context → confidence stays low
- NER found PERSON entities not caught by regex → new finding with medium confidence

### 3b: Claude API validation layer
Create `pipeline/layer3_llm.py`:

For entities with confidence still below threshold (e.g., < 0.8 after Layer 2):
- Send the surrounding text context (±200 chars) to Claude claude-sonnet-4-20250514
- Ask Claude to classify: "Is this likely PII? What type? How confident are you?"
- Parse Claude's response into a structured classification
- Update confidence scores based on Claude's assessment

Keep the prompt focused and efficient — we're paying per token. Include the entity type hypothesis and surrounding context only.

Implement a simple rate limiter / cost tracker to avoid runaway API calls.

### 3c: Router logic
Create `pipeline/router.py`:

The routing decision tree:
```
For each document:
  1. Run Layer 1 (regex) on full document
  2. For each finding:
     - If confidence >= 0.9 → ACCEPT, no escalation
     - If confidence >= 0.5 → escalate to Layer 2 (NER)
     - If confidence < 0.5 → escalate to Layer 2 (NER)
  3. After Layer 2:
     - If confidence >= 0.8 → ACCEPT
     - If confidence < 0.8 → escalate to Layer 3 (Claude)
  4. After Layer 3:
     - Accept Claude's classification as final
```

Track which layer made the final decision for each entity.

### 3d: Verify
Run the full pipeline against all sample docs. Print results showing:
- Which layer classified each entity
- Confidence scores at each layer
- Latency per layer (this should dramatically show the cost pyramid: regex ~0ms, NER ~10ms, Claude ~500ms)
- Did the ambiguous cases correctly escalate? Did the obvious cases stay at Layer 1?

---

## Phase 4: Results Store + Reporting (~30 min)

### 4a: SQLite results store
Create `store/catalog.py`:

Tables:
```sql
documents:
  id TEXT PRIMARY KEY,
  path TEXT,
  file_type TEXT,
  size_bytes INTEGER,
  scanned_at TIMESTAMP,
  total_entities_found INTEGER,
  risk_score FLOAT

features:
  document_id TEXT,
  feature_name TEXT,
  feature_value TEXT,
  computed_at TIMESTAMP,
  pipeline_version TEXT

entities:
  id TEXT PRIMARY KEY,
  document_id TEXT,
  entity_type TEXT,
  matched_text_redacted TEXT,
  confidence FLOAT,
  classified_by_layer INTEGER,
  layer1_confidence FLOAT,
  layer2_confidence FLOAT,
  layer3_confidence FLOAT,
  position INTEGER,
  line_number INTEGER,
  surrounding_context TEXT,
  classified_at TIMESTAMP

scan_metrics:
  scan_id TEXT,
  document_id TEXT,
  layer INTEGER,
  latency_ms FLOAT,
  entities_found INTEGER,
  escalated_to_next BOOLEAN
```

### 4b: CLI report
Create `report/cli.py`:

Print a formatted summary:
1. **Scan summary**: X documents scanned, Y entities found, Z escalated to NER, W escalated to Claude
2. **Per-document findings**: table with document name, entity count, highest risk, classification layers used
3. **Layer performance**: avg latency per layer, % of entities resolved at each layer, cost estimate
4. **Risk report**: entities sorted by confidence, flagging any in "public" or unexpected directories
5. **Business context**: highlight the hr/w2 vs public/shared_doc contrast (same data, different risk based on location)

### 4c: Wire it all together
Create `main.py` that:
1. Discovers documents via connector
2. Computes features
3. Runs classification pipeline with routing
4. Stores everything in SQLite
5. Prints the report

### 4d: Verify
Run the full pipeline end-to-end. Review the report. The key things to see:
- Regex handles ~60-70% of classifications alone (the easy ones)
- NER resolves another ~20% (adding context to ambiguous cases)
- Claude only gets called for ~10% (the truly ambiguous ones)
- Latency pyramid is clearly visible in the metrics
- Business context (hr/ vs public/) affects risk scoring

---

## Stretch Goals (If Time Permits)

### S1: Async queue simulation
Add a simple queue (Python `queue.Queue` or `asyncio.Queue`) between the connector and the classifier. Process documents concurrently. This simulates the Kafka-based fan-out pattern.

### S2: Business context scoring
Enhance the feature computer to score "expected vs. unexpected" PII based on directory path and file naming. SSNs in `hr/` = expected (low alert). SSNs in `public/` = unexpected (high alert). Same data, different risk based on business context.

### S3: Model versioning simulation
Add a `pipeline_version` field to all results. Change a regex pattern, re-scan, and compare results between versions. This simulates the model registry / A/B testing concept.

### S4: Remediation stub
Add a simple remediation action: for high-risk findings, generate a "remediation recommendation" (redact, move, revoke access). Don't execute it — just log what the system would do. This simulates the Temporal workflow concept.
