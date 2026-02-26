#!/usr/bin/env python3
"""Generate LoRA training data for secret-detection SLM.

Produces chat-format JSONL training examples combining:
  1. Seed examples extracted from existing test documents
  2. Synthetic positives (passwords, API keys, connection strings, private keys, etc.)
  3. Synthetic negatives / hard negatives (hashes, UUIDs, placeholders, part numbers)

Usage:
    python slm/generate_training_data.py \
        --output slm/training_data \
        --count 180 \
        --positive-ratio 0.60 \
        --seed 42 \
        --val-split 0.15
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import string
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Example:
    """A single training example before formatting to chat JSONL."""
    context_lines: list[str]          # surrounding lines (the snippet)
    target_line_idx: int              # which line in context_lines is the target
    context_type: str                 # e.g. "python source code", "yaml config"
    is_secret: bool
    secret_type: str                  # e.g. "password", "api_key", "none"
    confidence: float
    reasoning: str
    source: str = "synthetic"         # "seed" or "synthetic"


# ---------------------------------------------------------------------------
# Context type labels
# ---------------------------------------------------------------------------

CONTEXT_TYPES = [
    "python source code",
    "javascript source code",
    "yaml configuration",
    "toml configuration",
    "json configuration",
    "environment file",
    "shell script",
    "docker compose file",
    "terraform configuration",
    "properties file",
    "slack chat log",
    "security audit report",
    "CI/CD pipeline config",
    "kubernetes manifest",
    "nginx configuration",
]


# ---------------------------------------------------------------------------
# Value pools — used by synthetic generators
# ---------------------------------------------------------------------------

PASSWORD_VALUES = [
    "Tr0ub4dor&3", "P@ssw0rd123!", "Summer2024!", "Monkey123!",
    "r00tP@ss!2024", "Kj8#mNp2$vX9", "Admin$ecure99", "Welcome1!",
    "Ch@ngeM3N0w!", "Qwerty!2024", "D@tabase#1", "Pr0duction!Key",
    "s3cretP@55", "Myp@ssword1", "L3tM3In!2024", "Sup3rS3cret!",
    "N0tTh3P@ss!", "B@ckd00r2024", "T3stPa$$123", "R3allyStr0ng!",
]

API_KEY_PREFIXES = {
    "openai": ("sk-proj-", 32, "OpenAI API key"),
    "stripe_live": ("sk_live_", 24, "Stripe live secret key"),
    "stripe_test": ("sk_test_", 24, "Stripe test secret key"),
    "github_pat": ("ghp_", 36, "GitHub personal access token"),
    "github_fine": ("github_pat_", 30, "GitHub fine-grained PAT"),
    "aws_access": ("AKIA", 16, "AWS access key ID"),
    "slack_bot": ("xoxb-", 48, "Slack bot token"),
    "slack_user": ("xoxp-", 48, "Slack user token"),
    "sendgrid": ("SG.", 48, "SendGrid API key"),
    "google_oauth": ("GOCSPX-", 20, "Google OAuth client secret"),
    "twilio": ("SK", 32, "Twilio API key"),
    "datadog": ("dd", 32, "Datadog API key"),
    "npm": ("npm_", 36, "npm access token"),
    "pypi": ("pypi-AgEIcHlwaS5vcmc", 40, "PyPI API token"),
    "vercel": ("vercel_", 24, "Vercel access token"),
}

VARIABLE_NAMES_PASSWORD = [
    "DB_PASSWORD", "db_password", "dbPassword", "DATABASE_PASSWORD",
    "MYSQL_ROOT_PASSWORD", "POSTGRES_PASSWORD", "password", "passwd",
    "secret", "admin_password", "root_password", "user_password",
    "app_secret", "session_secret", "REDIS_PASSWORD", "AUTH_PASSWORD",
    "smtp_password", "mail_password", "ldap_password", "ftp_password",
]

VARIABLE_NAMES_KEY = [
    "API_KEY", "api_key", "apiKey", "SECRET_KEY", "secret_key",
    "secretKey", "ACCESS_TOKEN", "access_token", "AUTH_TOKEN",
    "PRIVATE_KEY", "private_key", "CLIENT_SECRET", "client_secret",
    "WEBHOOK_SECRET", "webhook_secret", "SIGNING_KEY", "signing_key",
    "ENCRYPTION_KEY", "encryption_key", "MASTER_KEY", "master_key",
    "SERVICE_KEY", "SERVICE_ACCOUNT_KEY", "JWT_SECRET", "jwt_secret",
]

# Connection string templates — {user}, {password}, {host}, {port}, {db}
CONN_STRING_TEMPLATES = [
    "postgresql://{user}:{password}@{host}:{port}/{db}",
    "postgres://{user}:{password}@{host}:{port}/{db}",
    "mysql://{user}:{password}@{host}:{port}/{db}",
    "mongodb://{user}:{password}@{host}:{port}/{db}?authSource=admin",
    "mongodb+srv://{user}:{password}@{host}/{db}?retryWrites=true",
    "redis://:{password}@{host}:{port}/0",
    "amqp://{user}:{password}@{host}:{port}/",
]

CONN_HOSTS = [
    "db-prod.internal.acme.com", "localhost", "10.0.1.42",
    "prod-db.us-east-1.rds.amazonaws.com", "mongo-cluster.acme.io",
    "redis-cache.internal", "db.example.com", "192.168.1.100",
]

CONN_USERS = ["admin", "deploy_user", "app_service", "root", "dbadmin", "webapp"]
CONN_DBS = ["production", "maindb", "appdb", "userdata", "analytics", "orders"]

# PEM key headers
PEM_HEADERS = [
    ("-----BEGIN RSA PRIVATE KEY-----", "-----END RSA PRIVATE KEY-----", "RSA private key"),
    ("-----BEGIN EC PRIVATE KEY-----", "-----END EC PRIVATE KEY-----", "EC private key"),
    ("-----BEGIN OPENSSH PRIVATE KEY-----", "-----END OPENSSH PRIVATE KEY-----", "SSH private key"),
    ("-----BEGIN PRIVATE KEY-----", "-----END PRIVATE KEY-----", "PKCS#8 private key"),
]

# Hard negative value pools
SHA256_HASHES = [
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "a591a6d40bf420404a011733cfb7b190d62c65bf0bcda32b57b277d9ad9f146e",
    "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824",
    "d7a8fbb307d7809469ca9abcb0082e4f8d5651e46d3cdb762d02d0bf37c9e592",
]

MD5_HASHES = [
    "d41d8cd98f00b204e9800998ecf8427e",
    "098f6bcd4621d373cade4e832627b4f6",
    "5d41402abc4b2a76b9719d911017c592",
    "e99a18c428cb38d5f260853678922e03",
]

GIT_COMMIT_HASHES = [
    "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0",
    "f47ac10b58cc4372a5670e02b2c3d479f2d1e8a3",
    "3b18e512dba79e4c8300dd08aeb37f8e728b8dad",
]

UUIDS = [
    "550e8400-e29b-41d4-a716-446655440000",
    "6ba7b810-9dad-11d1-80b4-00c04fd430c8",
    "f47ac10b-58cc-4372-a567-0e02b2c3d479",
    "a8098c1a-f86e-11da-bd1a-00112444be1e",
    "123e4567-e89b-12d3-a456-426614174000",
]

PLACEHOLDER_VALUES = [
    "<YOUR_API_KEY>", "<your-api-key-here>", "YOUR_API_KEY_HERE",
    "your_password_here", "CHANGEME", "changeme", "TODO_REPLACE",
    "xxx-xxx-xxxx", "REPLACE_ME", "INSERT_TOKEN_HERE",
    "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
    "ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
    "${API_KEY}", "${SECRET_KEY}", "$DB_PASSWORD",
    "os.environ['API_KEY']", "process.env.API_KEY",
]

REDACTED_VALUES = [
    "****", "***-**-****", "XXXX-XXXX-XXXX-XXXX", "[REDACTED]",
    "●●●●●●●●●●", "sk-...redacted...", "*" * 20, "••••••••",
]

VERSION_STRINGS = [
    "v2.4.1", "1.0.0-beta.3", "3.14.159", "2024.09.1",
    "v1.2.3-rc1", "0.99.7", "22.04.1", "10.15.7",
    "v3.0.0-alpha", "5.4.3.2",
]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def random_hex(length: int, rng: random.Random) -> str:
    """Generate a random hex string."""
    return "".join(rng.choices("0123456789abcdef", k=length))


def random_alnum(length: int, rng: random.Random) -> str:
    """Generate a random alphanumeric string."""
    return "".join(rng.choices(string.ascii_letters + string.digits, k=length))


def random_base64_chars(length: int, rng: random.Random) -> str:
    """Generate a random base64-ish string."""
    chars = string.ascii_letters + string.digits + "+/="
    return "".join(rng.choices(chars, k=length))


def format_assignment(var_name: str, value: str, style: str, rng: random.Random) -> str:
    """Format a variable assignment in a given language style."""
    if style == "python":
        quote = rng.choice(['"', "'"])
        return f'{var_name} = {quote}{value}{quote}'
    elif style == "env":
        return f"{var_name}={value}"
    elif style == "yaml":
        return f"{var_name}: {value}"
    elif style == "json":
        return f'"{var_name}": "{value}"'
    elif style == "js":
        kw = rng.choice(["const", "let", "var"])
        return f'{kw} {var_name} = "{value}";'
    elif style == "toml":
        return f'{var_name} = "{value}"'
    elif style == "shell":
        return f'export {var_name}="{value}"'
    else:
        return f'{var_name} = "{value}"'


def make_surrounding_lines(rng: random.Random, context_type: str, n_before: int, n_after: int) -> tuple[list[str], list[str]]:
    """Generate plausible surrounding lines for a given context type."""
    filler_python = [
        "import os", "import sys", "from pathlib import Path",
        "# Configuration module", "class Config:", "    pass",
        "def get_config():", "    return {}", "# TODO: refactor",
        "logger = logging.getLogger(__name__)", "DB_HOST = \"localhost\"",
        "DB_PORT = 5432", "DEBUG = False", "TIMEOUT = 30",
        "MAX_RETRIES = 3", "# Database settings",
    ]
    filler_env = [
        "NODE_ENV=production", "PORT=8080", "HOST=0.0.0.0",
        "LOG_LEVEL=info", "DEBUG=false", "# Application config",
        "TZ=UTC", "WORKERS=4", "MAX_CONNECTIONS=100",
        "CACHE_TTL=3600", "ENABLE_METRICS=true",
    ]
    filler_yaml = [
        "version: '3.8'", "services:", "  app:", "    image: myapp:latest",
        "    ports:", '      - "8080:8080"', "    environment:",
        "      NODE_ENV: production", "      LOG_LEVEL: info",
        "    restart: always", "    # service config",
        "    timeout: 30", "    replicas: 3",
    ]
    filler_json = [
        "{", "}", '  "name": "my-app",', '  "version": "1.0.0",',
        '  "port": 8080,', '  "debug": false,', '  "host": "0.0.0.0",',
        '  "timeout": 30,', '  "retries": 3,',
    ]
    filler_shell = [
        "#!/bin/bash", "set -euo pipefail", "echo \"Starting deployment...\"",
        "cd /opt/app", "source .env", "# Deploy script",
        "docker compose up -d", "echo \"Done.\"",
    ]
    filler_generic = [
        "# Configuration", "# Settings", "---",
        "# NOTE: update before deploy", "# See documentation",
        "enabled: true", "region: us-east-1",
    ]

    if "python" in context_type:
        pool = filler_python
    elif "environment" in context_type or "env" in context_type:
        pool = filler_env
    elif "yaml" in context_type or "docker" in context_type or "kubernetes" in context_type:
        pool = filler_yaml
    elif "json" in context_type:
        pool = filler_json
    elif "shell" in context_type:
        pool = filler_shell
    else:
        pool = filler_generic

    before = [rng.choice(pool) for _ in range(n_before)]
    after = [rng.choice(pool) for _ in range(n_after)]
    return before, after


def build_user_prompt(context_lines: list[str], target_idx: int, context_type: str) -> str:
    """Build the user prompt with >>> marker on the target line."""
    lines = []
    for i, line in enumerate(context_lines):
        if i == target_idx:
            lines.append(f">>> {line}")
        else:
            lines.append(f"  {line}")

    snippet = "\n".join(lines)
    return (
        "Classify whether the highlighted line contains a secret, credential, or sensitive key.\n\n"
        f"Context type: {context_type}\n\n"
        f"{snippet}"
    )


def build_assistant_response(is_secret: bool, secret_type: str, confidence: float, reasoning: str) -> str:
    """Build the structured JSON assistant response."""
    resp = {
        "is_secret": is_secret,
        "type": secret_type,
        "confidence": round(confidence, 2),
        "reasoning": reasoning,
    }
    return json.dumps(resp)


def example_to_jsonl(ex: Example) -> dict:
    """Convert an Example to a chat-format JSONL dict."""
    user_msg = build_user_prompt(ex.context_lines, ex.target_line_idx, ex.context_type)
    asst_msg = build_assistant_response(ex.is_secret, ex.secret_type, ex.confidence, ex.reasoning)
    return {
        "messages": [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": asst_msg},
        ]
    }


# ---------------------------------------------------------------------------
# Seed example extraction
# ---------------------------------------------------------------------------

def load_seed_examples(test_docs_dir: Path) -> list[Example]:
    """Extract seed examples from existing test documents.

    Focuses on secrets (passwords, API keys, JWT, private keys) and
    a set of non-secret hard negatives from the same documents.
    """
    seeds: list[Example] = []

    # --- test_source_code.py: passwords and API keys ---
    src_path = test_docs_dir / "test_source_code.py"
    if src_path.exists():
        src_lines = src_path.read_text().splitlines()

        # Each (line_idx_0based, is_secret, type, confidence, reasoning)
        source_targets = [
            (9, True, "password", 0.95, "Variable named DB_PASSWORD assigned a high-entropy string literal"),
            (10, True, "api_key", 0.93, "Variable named API_KEY assigned string with sk-proj- prefix (OpenAI key format)"),
            (11, True, "api_key", 0.95, "Variable named STRIPE_SECRET assigned string with sk_live_ prefix (Stripe live key)"),
            (14, True, "password", 0.90, "Connection string contains embedded password after colon in userinfo segment"),
            (18, True, "password", 0.88, "Environment variable fallback contains a hardcoded password string"),
            (31, True, "api_key", 0.94, "AWS access key ID with AKIA prefix assigned to AWS_ACCESS_KEY variable"),
            (32, True, "api_key", 0.94, "AWS secret access key — high-entropy string assigned to AWS_SECRET_KEY"),
            # Non-secret seeds from same file
            (7, False, "none", 0.05, "Database hostname string — not a secret, just a server address"),
            (23, False, "none", 0.08, "Test identifier with invalid 000 prefix — not a real SSN or secret"),
        ]

        for line_idx, is_secret, stype, conf, reasoning in source_targets:
            start = max(0, line_idx - 2)
            end = min(len(src_lines), line_idx + 3)
            context = src_lines[start:end]
            target_in_context = line_idx - start
            seeds.append(Example(
                context_lines=context,
                target_line_idx=target_in_context,
                context_type="python source code",
                is_secret=is_secret,
                secret_type=stype,
                confidence=conf,
                reasoning=reasoning,
                source="seed",
            ))

    # --- test_password_dump.txt: audit report with credentials ---
    dump_path = test_docs_dir / "test_password_dump.txt"
    if dump_path.exists():
        dump_lines = dump_path.read_text().splitlines()

        dump_targets = [
            (8, True, "password", 0.94, "Database password in connection string from security audit finding"),
            (13, True, "api_key", 0.96, "GitHub personal access token with ghp_ prefix"),
            (19, True, "api_key", 0.95, "Stripe live secret key with sk_live_ prefix"),
            (24, True, "password", 0.93, "MySQL root password in docker-compose environment variable"),
            (29, True, "private_key", 0.97, "RSA private key PEM header — indicates exposed private key material"),
            (35, True, "api_key", 0.94, "AWS access key ID with AKIA prefix in Terraform variables"),
            (36, True, "api_key", 0.94, "AWS secret access key — high-entropy string in Terraform"),
            (41, True, "jwt_secret", 0.91, "JWT signing secret assigned to JWT_SECRET variable"),
            (46, True, "api_key", 0.92, "Google OAuth client secret with GOCSPX- prefix"),
            # Non-secret from same file
            (49, False, "none", 0.05, "Summary line describing scan results — not a secret"),
        ]

        for line_idx, is_secret, stype, conf, reasoning in dump_targets:
            # 0-indexed: line_idx is 0-based here, file lines from read are 1-based
            li = line_idx - 1  # convert to 0-based index
            start = max(0, li - 2)
            end = min(len(dump_lines), li + 3)
            context = dump_lines[start:end]
            target_in_context = li - start
            seeds.append(Example(
                context_lines=context,
                target_line_idx=target_in_context,
                context_type="security audit report",
                is_secret=is_secret,
                secret_type=stype,
                confidence=conf,
                reasoning=reasoning,
                source="seed",
            ))

    # --- test_slack_export.json: password shared in chat ---
    slack_path = test_docs_dir / "test_slack_export.json"
    if slack_path.exists():
        slack_lines = slack_path.read_text().splitlines()
        # Line 7 (0-based 6): password "Monkey123!"
        li = 6
        start = max(0, li - 2)
        end = min(len(slack_lines), li + 3)
        seeds.append(Example(
            context_lines=slack_lines[start:end],
            target_line_idx=li - start,
            context_type="slack chat log",
            is_secret=True,
            secret_type="password",
            confidence=0.88,
            reasoning="User sharing a password in a chat message — plaintext credential",
            source="seed",
        ))

    # --- Non-secret seeds from test_part_numbers.csv ---
    parts_path = test_docs_dir / "test_part_numbers.csv"
    if parts_path.exists():
        parts_lines = parts_path.read_text().splitlines()
        parts_targets = [
            (1, False, "none", 0.06, "Industrial part number in CSV catalog — matches SSN format but context is parts inventory"),
            (4, False, "none", 0.04, "Circuit board part number matching credit card format — context is manufacturing catalog"),
        ]
        for li, is_secret, stype, conf, reasoning in parts_targets:
            start = max(0, li - 1)
            end = min(len(parts_lines), li + 2)
            seeds.append(Example(
                context_lines=parts_lines[start:end],
                target_line_idx=li - start,
                context_type="csv inventory data",
                is_secret=is_secret,
                secret_type=stype,
                confidence=conf,
                reasoning=reasoning,
                source="seed",
            ))

    # --- Non-secret seeds from test_legal_disclaimer.txt ---
    legal_path = test_docs_dir / "test_legal_disclaimer.txt"
    if legal_path.exists():
        legal_lines = legal_path.read_text().splitlines()
        # Line 3 (0-based 2): document reference 483-29-1847
        li = 2
        start = max(0, li - 1)
        end = min(len(legal_lines), li + 2)
        seeds.append(Example(
            context_lines=legal_lines[start:end],
            target_line_idx=li - start,
            context_type="legal document",
            is_secret=False,
            secret_type="none",
            confidence=0.03,
            reasoning="Document reference number in a legal agreement — not a secret or credential",
            source="seed",
        ))

    return seeds


# ---------------------------------------------------------------------------
# Synthetic positive generators
# ---------------------------------------------------------------------------

def gen_password_examples(count: int, rng: random.Random) -> list[Example]:
    """Generate synthetic password examples."""
    examples = []
    styles = ["python", "env", "yaml", "json", "js", "shell", "toml"]

    for _ in range(count):
        password = rng.choice(PASSWORD_VALUES)
        var = rng.choice(VARIABLE_NAMES_PASSWORD)
        style = rng.choice(styles)
        ctx_type = {
            "python": "python source code",
            "env": "environment file",
            "yaml": "yaml configuration",
            "json": "json configuration",
            "js": "javascript source code",
            "shell": "shell script",
            "toml": "toml configuration",
        }[style]

        target_line = format_assignment(var, password, style, rng)

        # Sometimes wrap in a connection string instead
        if rng.random() < 0.15:
            tmpl = rng.choice(CONN_STRING_TEMPLATES)
            conn_str = tmpl.format(
                user=rng.choice(CONN_USERS),
                password=password,
                host=rng.choice(CONN_HOSTS),
                port=rng.choice([5432, 3306, 27017, 6379, 5672]),
                db=rng.choice(CONN_DBS),
            )
            target_line = format_assignment(
                rng.choice(["DATABASE_URL", "DSN", "CONNECTION_STRING", "DB_URI"]),
                conn_str, style, rng,
            )

        n_before = rng.randint(0, 3)
        n_after = rng.randint(0, 3)
        before, after = make_surrounding_lines(rng, ctx_type, n_before, n_after)

        context = before + [target_line] + after
        target_idx = len(before)

        conf = rng.uniform(0.85, 0.98) if rng.random() > 0.2 else rng.uniform(0.65, 0.82)

        reasoning_templates = [
            f"Variable named {var} assigned a string literal containing a password",
            f"Hardcoded password value in {ctx_type} — high-entropy string with special characters",
            f"Password assigned to {var} — should be loaded from a secret manager, not hardcoded",
            f"Credential string assigned to configuration variable {var}",
        ]

        examples.append(Example(
            context_lines=context,
            target_line_idx=target_idx,
            context_type=ctx_type,
            is_secret=True,
            secret_type="password",
            confidence=round(conf, 2),
            reasoning=rng.choice(reasoning_templates),
            source="synthetic",
        ))

    return examples


def gen_api_key_examples(count: int, rng: random.Random) -> list[Example]:
    """Generate synthetic API key examples."""
    examples = []
    styles = ["python", "env", "yaml", "json", "js", "shell"]

    for _ in range(count):
        key_type = rng.choice(list(API_KEY_PREFIXES.keys()))
        prefix, suffix_len, description = API_KEY_PREFIXES[key_type]

        # Generate a realistic-looking key value
        value = prefix + random_alnum(suffix_len, rng)

        var = rng.choice(VARIABLE_NAMES_KEY)
        style = rng.choice(styles)
        ctx_type = {
            "python": "python source code",
            "env": "environment file",
            "yaml": "yaml configuration",
            "json": "json configuration",
            "js": "javascript source code",
            "shell": "shell script",
        }[style]

        target_line = format_assignment(var, value, style, rng)

        n_before = rng.randint(0, 3)
        n_after = rng.randint(0, 3)
        before, after = make_surrounding_lines(rng, ctx_type, n_before, n_after)

        context = before + [target_line] + after
        target_idx = len(before)

        conf = rng.uniform(0.88, 0.98) if rng.random() > 0.15 else rng.uniform(0.70, 0.85)

        reasoning_templates = [
            f"{description} — recognized by {prefix} prefix pattern",
            f"API key with {prefix} prefix assigned to {var} — {description}",
            f"High-entropy token with known service prefix ({prefix}) in {ctx_type}",
            f"Credential token: {description} found in configuration",
        ]

        examples.append(Example(
            context_lines=context,
            target_line_idx=target_idx,
            context_type=ctx_type,
            is_secret=True,
            secret_type="api_key",
            confidence=round(conf, 2),
            reasoning=rng.choice(reasoning_templates),
            source="synthetic",
        ))

    return examples


def gen_connection_string_examples(count: int, rng: random.Random) -> list[Example]:
    """Generate synthetic connection string examples with embedded passwords."""
    examples = []
    styles = ["python", "env", "yaml", "json"]

    for _ in range(count):
        tmpl = rng.choice(CONN_STRING_TEMPLATES)
        password = rng.choice(PASSWORD_VALUES)
        conn_str = tmpl.format(
            user=rng.choice(CONN_USERS),
            password=password,
            host=rng.choice(CONN_HOSTS),
            port=rng.choice([5432, 3306, 27017, 6379, 5672]),
            db=rng.choice(CONN_DBS),
        )

        var = rng.choice(["DATABASE_URL", "DSN", "CONN_STRING", "DB_URI",
                          "SQLALCHEMY_DATABASE_URI", "MONGO_URI", "REDIS_URL",
                          "CELERY_BROKER_URL", "AMQP_URL"])
        style = rng.choice(styles)
        ctx_type = {
            "python": "python source code",
            "env": "environment file",
            "yaml": "yaml configuration",
            "json": "json configuration",
        }[style]

        target_line = format_assignment(var, conn_str, style, rng)

        n_before = rng.randint(1, 3)
        n_after = rng.randint(0, 2)
        before, after = make_surrounding_lines(rng, ctx_type, n_before, n_after)

        context = before + [target_line] + after
        target_idx = len(before)

        conf = rng.uniform(0.88, 0.96)

        examples.append(Example(
            context_lines=context,
            target_line_idx=target_idx,
            context_type=ctx_type,
            is_secret=True,
            secret_type="connection_string",
            confidence=round(conf, 2),
            reasoning="Connection string with embedded password in userinfo segment — credentials should not be hardcoded in URIs",
            source="synthetic",
        ))

    return examples


def gen_private_key_examples(count: int, rng: random.Random) -> list[Example]:
    """Generate synthetic private key examples."""
    examples = []

    for _ in range(count):
        header, footer, desc = rng.choice(PEM_HEADERS)
        # Generate fake key material (a few lines of base64)
        key_lines = [header]
        for _ in range(rng.randint(1, 3)):
            key_lines.append(random_base64_chars(64, rng))
        key_lines.append("...")

        # Pick which line to highlight (the header)
        target_idx = 0
        n_before = rng.randint(0, 2)
        before_lines = []
        if n_before > 0:
            file_comments = [
                "# Private key for production TLS",
                "# DO NOT COMMIT THIS FILE",
                f"# {desc}",
                "-----",
                f"File: .ssh/id_rsa",
            ]
            before_lines = [rng.choice(file_comments) for _ in range(n_before)]

        context = before_lines + key_lines
        target_idx = len(before_lines)

        conf = rng.uniform(0.93, 0.98)

        examples.append(Example(
            context_lines=context,
            target_line_idx=target_idx,
            context_type=rng.choice(["security audit report", "shell script", "yaml configuration"]),
            is_secret=True,
            secret_type="private_key",
            confidence=round(conf, 2),
            reasoning=f"{desc} PEM header detected — private key material should never be stored in code or config",
            source="synthetic",
        ))

    return examples


def gen_jwt_webhook_examples(count: int, rng: random.Random) -> list[Example]:
    """Generate JWT secrets and webhook signing tokens."""
    examples = []
    styles = ["python", "env", "yaml", "json", "js"]

    jwt_values = [
        f"my-super-secret-jwt-key-{rng.randint(1000, 9999)}",
        random_alnum(48, rng),
        random_hex(64, rng),
        f"whsec_{random_alnum(32, rng)}",
        f"whsec_{''.join(rng.choices(string.ascii_letters + string.digits, k=40))}",
    ]

    jwt_vars = [
        "JWT_SECRET", "JWT_SIGNING_KEY", "TOKEN_SECRET", "WEBHOOK_SECRET",
        "WEBHOOK_SIGNING_KEY", "HMAC_SECRET", "SIGNING_SECRET",
        "jwt_secret_key", "webhookSecret", "STRIPE_WEBHOOK_SECRET",
        "GITHUB_WEBHOOK_SECRET", "SLACK_SIGNING_SECRET",
    ]

    for _ in range(count):
        value = rng.choice(jwt_values) if rng.random() > 0.3 else random_alnum(rng.randint(24, 64), rng)
        var = rng.choice(jwt_vars)
        style = rng.choice(styles)
        ctx_type = {
            "python": "python source code",
            "env": "environment file",
            "yaml": "yaml configuration",
            "json": "json configuration",
            "js": "javascript source code",
        }[style]

        target_line = format_assignment(var, value, style, rng)

        n_before = rng.randint(0, 3)
        n_after = rng.randint(0, 2)
        before, after = make_surrounding_lines(rng, ctx_type, n_before, n_after)

        context = before + [target_line] + after
        target_idx = len(before)

        stype = "jwt_secret" if "jwt" in var.lower() or "token" in var.lower() or "signing" in var.lower() else "webhook_secret"
        conf = rng.uniform(0.82, 0.95)

        examples.append(Example(
            context_lines=context,
            target_line_idx=target_idx,
            context_type=ctx_type,
            is_secret=True,
            secret_type=stype,
            confidence=round(conf, 2),
            reasoning=f"Signing key or verification secret assigned to {var} — used for cryptographic operations",
            source="synthetic",
        ))

    return examples


def gen_generic_token_examples(count: int, rng: random.Random) -> list[Example]:
    """Generate generic bearer tokens, session secrets, etc."""
    examples = []
    styles = ["python", "env", "yaml", "json", "js", "shell"]

    token_vars = [
        "BEARER_TOKEN", "bearer_token", "SESSION_SECRET", "session_secret",
        "ACCESS_TOKEN", "access_token", "REFRESH_TOKEN", "refresh_token",
        "AUTH_TOKEN", "ADMIN_TOKEN", "INTERNAL_API_TOKEN", "SERVICE_TOKEN",
        "DEPLOY_TOKEN", "CI_TOKEN", "REGISTRY_TOKEN",
    ]

    for _ in range(count):
        # Generate a plausible token value
        token_formats = [
            lambda: random_alnum(rng.randint(32, 64), rng),
            lambda: random_hex(rng.randint(32, 64), rng),
            lambda: f"Bearer {random_alnum(40, rng)}",
            lambda: random_base64_chars(rng.randint(40, 80), rng),
        ]
        value = rng.choice(token_formats)()

        var = rng.choice(token_vars)
        style = rng.choice(styles)
        ctx_type = {
            "python": "python source code",
            "env": "environment file",
            "yaml": "yaml configuration",
            "json": "json configuration",
            "js": "javascript source code",
            "shell": "shell script",
        }[style]

        target_line = format_assignment(var, value, style, rng)

        n_before = rng.randint(0, 3)
        n_after = rng.randint(0, 2)
        before, after = make_surrounding_lines(rng, ctx_type, n_before, n_after)

        context = before + [target_line] + after
        target_idx = len(before)

        conf = rng.uniform(0.78, 0.94)

        examples.append(Example(
            context_lines=context,
            target_line_idx=target_idx,
            context_type=ctx_type,
            is_secret=True,
            secret_type="token",
            confidence=round(conf, 2),
            reasoning=f"Authentication/authorization token assigned to {var} — high-entropy value that grants access",
            source="synthetic",
        ))

    return examples


# ---------------------------------------------------------------------------
# Synthetic negative generators
# ---------------------------------------------------------------------------

def gen_hash_negatives(count: int, rng: random.Random) -> list[Example]:
    """Generate hash/checksum negatives — high entropy but not secrets."""
    examples = []
    styles = ["python", "env", "yaml", "shell"]

    hash_vars = [
        "CHECKSUM", "SHA256", "sha256_hash", "MD5", "md5sum",
        "CONTENT_HASH", "file_hash", "INTEGRITY_HASH", "GIT_COMMIT",
        "git_sha", "COMMIT_SHA", "IMAGE_DIGEST",
    ]

    for _ in range(count):
        if rng.random() < 0.4:
            value = rng.choice(SHA256_HASHES) if rng.random() > 0.3 else random_hex(64, rng)
            hash_type = "SHA-256"
        elif rng.random() < 0.6:
            value = rng.choice(MD5_HASHES) if rng.random() > 0.3 else random_hex(32, rng)
            hash_type = "MD5"
        else:
            value = rng.choice(GIT_COMMIT_HASHES) if rng.random() > 0.3 else random_hex(40, rng)
            hash_type = "git commit"

        var = rng.choice(hash_vars)
        style = rng.choice(styles)
        ctx_type = {
            "python": "python source code",
            "env": "environment file",
            "yaml": "yaml configuration",
            "shell": "shell script",
        }[style]

        target_line = format_assignment(var, value, style, rng)

        n_before = rng.randint(0, 2)
        n_after = rng.randint(0, 2)
        before, after = make_surrounding_lines(rng, ctx_type, n_before, n_after)

        context = before + [target_line] + after
        target_idx = len(before)

        conf = rng.uniform(0.03, 0.15) if rng.random() > 0.25 else rng.uniform(0.15, 0.30)

        examples.append(Example(
            context_lines=context,
            target_line_idx=target_idx,
            context_type=ctx_type,
            is_secret=False,
            secret_type="none",
            confidence=round(conf, 2),
            reasoning=f"{hash_type} hash/checksum — high entropy but used for integrity verification, not authentication",
            source="synthetic",
        ))

    return examples


def gen_uuid_negatives(count: int, rng: random.Random) -> list[Example]:
    """Generate UUID/trace ID negatives."""
    examples = []
    styles = ["python", "env", "yaml", "json"]

    uuid_vars = [
        "REQUEST_ID", "request_id", "TRACE_ID", "trace_id",
        "CORRELATION_ID", "correlation_id", "SESSION_ID",
        "TRANSACTION_ID", "INSTANCE_ID", "DEVICE_ID",
    ]

    for _ in range(count):
        value = rng.choice(UUIDS) if rng.random() > 0.4 else (
            f"{random_hex(8, rng)}-{random_hex(4, rng)}-{random_hex(4, rng)}-"
            f"{random_hex(4, rng)}-{random_hex(12, rng)}"
        )

        var = rng.choice(uuid_vars)
        style = rng.choice(styles)
        ctx_type = {
            "python": "python source code",
            "env": "environment file",
            "yaml": "yaml configuration",
            "json": "json configuration",
        }[style]

        target_line = format_assignment(var, value, style, rng)

        n_before = rng.randint(0, 2)
        n_after = rng.randint(0, 2)
        before, after = make_surrounding_lines(rng, ctx_type, n_before, n_after)

        context = before + [target_line] + after
        target_idx = len(before)

        conf = rng.uniform(0.02, 0.12)

        examples.append(Example(
            context_lines=context,
            target_line_idx=target_idx,
            context_type=ctx_type,
            is_secret=False,
            secret_type="none",
            confidence=round(conf, 2),
            reasoning=f"UUID/trace identifier used for request tracking — not a secret or credential",
            source="synthetic",
        ))

    return examples


def gen_placeholder_negatives(count: int, rng: random.Random) -> list[Example]:
    """Generate placeholder/template negatives — look like secrets but aren't real."""
    examples = []
    styles = ["python", "env", "yaml", "json", "js"]

    for _ in range(count):
        value = rng.choice(PLACEHOLDER_VALUES)
        var = rng.choice(VARIABLE_NAMES_KEY + VARIABLE_NAMES_PASSWORD)
        style = rng.choice(styles)
        ctx_type = {
            "python": "python source code",
            "env": "environment file",
            "yaml": "yaml configuration",
            "json": "json configuration",
            "js": "javascript source code",
        }[style]

        target_line = format_assignment(var, value, style, rng)

        n_before = rng.randint(0, 3)
        n_after = rng.randint(0, 2)
        before, after = make_surrounding_lines(rng, ctx_type, n_before, n_after)

        # Sometimes add a comment indicating it's a placeholder
        if rng.random() < 0.3:
            comments = [
                "# Replace with your actual key",
                "# TODO: set this in CI/CD",
                "# Placeholder — will be injected at deploy time",
            ]
            before.append(rng.choice(comments))

        context = before + [target_line] + after
        target_idx = len(before)

        conf = rng.uniform(0.05, 0.20) if rng.random() > 0.3 else rng.uniform(0.20, 0.40)

        examples.append(Example(
            context_lines=context,
            target_line_idx=target_idx,
            context_type=ctx_type,
            is_secret=False,
            secret_type="none",
            confidence=round(conf, 2),
            reasoning=f"Placeholder or template value — variable references an environment variable or uses a sentinel value, not a real credential",
            source="synthetic",
        ))

    return examples


def gen_redacted_negatives(count: int, rng: random.Random) -> list[Example]:
    """Generate redacted/masked value negatives."""
    examples = []

    for _ in range(count):
        value = rng.choice(REDACTED_VALUES)
        var = rng.choice(VARIABLE_NAMES_KEY[:6] + VARIABLE_NAMES_PASSWORD[:6])
        style = rng.choice(["python", "env", "yaml", "json"])
        ctx_type = {
            "python": "python source code",
            "env": "environment file",
            "yaml": "yaml configuration",
            "json": "json configuration",
        }[style]

        target_line = format_assignment(var, value, style, rng)

        n_before = rng.randint(0, 2)
        n_after = rng.randint(0, 2)
        before, after = make_surrounding_lines(rng, ctx_type, n_before, n_after)

        context = before + [target_line] + after
        target_idx = len(before)

        examples.append(Example(
            context_lines=context,
            target_line_idx=target_idx,
            context_type=ctx_type,
            is_secret=False,
            secret_type="none",
            confidence=round(rng.uniform(0.02, 0.10), 2),
            reasoning="Redacted or masked value — the actual secret has been removed and replaced with placeholder characters",
            source="synthetic",
        ))

    return examples


def gen_part_number_negatives(count: int, rng: random.Random) -> list[Example]:
    """Generate part number/SKU negatives that look like SSNs or credit cards."""
    examples = []

    for _ in range(count):
        if rng.random() < 0.6:
            # SSN-format part number
            value = f"{rng.randint(100,999)}-{rng.randint(10,99)}-{rng.randint(1000,9999)}"
            note = "Part number in SSN format"
        else:
            # CC-format part number
            value = f"{rng.randint(1000,9999)}-{rng.randint(1000,9999)}-{rng.randint(1000,9999)}-{rng.randint(1000,9999)}"
            note = "Part number in credit card format"

        desc = rng.choice([
            "Hydraulic Valve Assembly", "Bearing Housing Type C",
            "Compressor Shaft Seal Kit", "Circuit Board Rev D",
            "Power Supply Module 240V", "Gasket Set - Engine",
            "Thermocouple Probe", "Actuator Assembly",
        ])
        price = f"{rng.uniform(5, 500):.2f}"

        target_line = f"{value},{desc},{rng.randint(10, 500)},{price},WH-{rng.choice(['East', 'West', 'North', 'South'])}"

        header = "part_number,description,quantity,unit_price,warehouse_location"
        other_parts = [
            f"{rng.randint(100,999)}-{rng.randint(10,99)}-{rng.randint(1000,9999)},{rng.choice(['Valve Kit', 'Seal Ring', 'Bracket Assy'])},{rng.randint(10,300)},{rng.uniform(5,200):.2f},WH-East"
            for _ in range(rng.randint(1, 3))
        ]

        context = [header] + other_parts[:rng.randint(0, 2)] + [target_line] + other_parts[rng.randint(0, 1):]
        target_idx = len(context) - 1 - len(other_parts[rng.randint(0, 1):])
        # Simpler: just find it
        target_idx = context.index(target_line)

        examples.append(Example(
            context_lines=context,
            target_line_idx=target_idx,
            context_type="csv inventory data",
            is_secret=False,
            secret_type="none",
            confidence=round(rng.uniform(0.03, 0.12), 2),
            reasoning=f"{note} — numeric pattern matches PII format but context is manufacturing/inventory catalog",
            source="synthetic",
        ))

    return examples


def gen_encoded_negatives(count: int, rng: random.Random) -> list[Example]:
    """Generate base64-encoded or URL-encoded non-sensitive values."""
    examples = []
    styles = ["python", "env", "yaml", "json"]

    encoded_vars = [
        "CONFIG_B64", "ENCODED_CONFIG", "BASE64_PAYLOAD", "ICON_DATA",
        "CERT_DATA", "CA_BUNDLE", "LOGO_BASE64", "ENCODED_SETTINGS",
    ]

    for _ in range(count):
        value = random_base64_chars(rng.randint(40, 80), rng)
        var = rng.choice(encoded_vars)
        style = rng.choice(styles)
        ctx_type = {
            "python": "python source code",
            "env": "environment file",
            "yaml": "yaml configuration",
            "json": "json configuration",
        }[style]

        target_line = format_assignment(var, value, style, rng)

        n_before = rng.randint(0, 2)
        n_after = rng.randint(0, 2)
        before, after = make_surrounding_lines(rng, ctx_type, n_before, n_after)

        # Sometimes add a comment clarifying it's config
        if rng.random() < 0.4:
            before.append(rng.choice([
                "# Base64-encoded configuration blob",
                "# CA certificate bundle",
                "# Encoded icon data",
            ]))

        context = before + [target_line] + after
        target_idx = len(before)

        conf = rng.uniform(0.10, 0.35) if rng.random() > 0.3 else rng.uniform(0.35, 0.50)

        examples.append(Example(
            context_lines=context,
            target_line_idx=target_idx,
            context_type=ctx_type,
            is_secret=False,
            secret_type="none",
            confidence=round(conf, 2),
            reasoning="Base64-encoded data blob — high entropy but variable name and context indicate non-sensitive configuration or asset data",
            source="synthetic",
        ))

    return examples


def gen_numeric_config_negatives(count: int, rng: random.Random) -> list[Example]:
    """Generate numeric configuration values (ports, timeouts, etc.)."""
    examples = []
    styles = ["python", "env", "yaml", "json"]

    configs = [
        ("PORT", lambda: str(rng.choice([3000, 5432, 8080, 8443, 9090, 27017])), "Port number"),
        ("TIMEOUT", lambda: str(rng.randint(5, 300)), "Timeout value in seconds"),
        ("MAX_RETRIES", lambda: str(rng.randint(1, 10)), "Retry count"),
        ("BATCH_SIZE", lambda: str(rng.choice([32, 64, 128, 256, 512, 1024])), "Batch size"),
        ("WORKERS", lambda: str(rng.randint(1, 16)), "Worker count"),
        ("MAX_CONNECTIONS", lambda: str(rng.choice([10, 25, 50, 100, 200])), "Connection pool size"),
        ("CACHE_TTL", lambda: str(rng.choice([60, 300, 600, 3600, 86400])), "Cache TTL in seconds"),
    ]

    for _ in range(count):
        var, val_fn, desc = rng.choice(configs)
        value = val_fn()
        style = rng.choice(styles)
        ctx_type = {
            "python": "python source code",
            "env": "environment file",
            "yaml": "yaml configuration",
            "json": "json configuration",
        }[style]

        target_line = format_assignment(var, value, style, rng)

        n_before = rng.randint(0, 3)
        n_after = rng.randint(0, 2)
        before, after = make_surrounding_lines(rng, ctx_type, n_before, n_after)

        context = before + [target_line] + after
        target_idx = len(before)

        examples.append(Example(
            context_lines=context,
            target_line_idx=target_idx,
            context_type=ctx_type,
            is_secret=False,
            secret_type="none",
            confidence=round(rng.uniform(0.01, 0.08), 2),
            reasoning=f"{desc} — numeric configuration parameter, not a secret",
            source="synthetic",
        ))

    return examples


def gen_doc_example_negatives(count: int, rng: random.Random) -> list[Example]:
    """Generate documentation/instructional examples showing credentials."""
    examples = []

    doc_snippets = [
        (
            ["# How to configure your API key:", "# 1. Go to https://platform.openai.com/api-keys",
             "# 2. Click 'Create new secret key'", "# 3. Set it in your environment:"],
            'export OPENAI_API_KEY="sk-proj-your-key-here"',
            "Documentation example showing how to set an environment variable — uses placeholder value, not a real key",
        ),
        (
            ["## Authentication", "", "Set your credentials in `.env`:"],
            'API_KEY=your_api_key_here',
            "README/documentation instructing users to set their own key — placeholder text, not an actual secret",
        ),
        (
            ["# Example connection string (replace with your credentials):", "# See docs at https://docs.example.com/db"],
            'DATABASE_URL="postgresql://user:password@localhost:5432/mydb"',
            "Example connection string in documentation with generic placeholder credentials (user/password)",
        ),
        (
            ["// Example: authenticating with the API", "// Replace <YOUR_TOKEN> with your actual token"],
            'const token = "<YOUR_TOKEN>";',
            "Code example in documentation with a placeholder token — instructional, not a real credential",
        ),
        (
            ["# Default test configuration", "# These are NOT production credentials"],
            'TEST_PASSWORD = "test123"',
            "Test configuration with trivial/obvious test password — intentionally non-secret default for CI/testing",
        ),
        (
            ["'''", "Example usage:", ""],
            "    curl -H 'Authorization: Bearer YOUR_TOKEN_HERE' https://api.example.com/data",
            "Documentation curl example with placeholder bearer token — instructional text, not a real token",
        ),
        (
            ["# Stripe integration quickstart", "# Test mode keys start with sk_test_"],
            "STRIPE_KEY=sk_test_xxxxxxxxxxxxxxxxxxxx",
            "Documentation showing Stripe test key format with x-placeholder — not a real key",
        ),
    ]

    for _ in range(count):
        before_lines, target_line, reasoning = rng.choice(doc_snippets)
        after_lines = rng.choice([
            [],
            ["", "# Then run: python app.py"],
            ["# After setting the above, restart the service"],
        ])

        context = before_lines + [target_line] + after_lines
        target_idx = len(before_lines)

        examples.append(Example(
            context_lines=context,
            target_line_idx=target_idx,
            context_type=rng.choice(["python source code", "shell script", "environment file", "javascript source code"]),
            is_secret=False,
            secret_type="none",
            confidence=round(rng.uniform(0.08, 0.25), 2),
            reasoning=reasoning,
            source="synthetic",
        ))

    return examples


def gen_version_string_negatives(count: int, rng: random.Random) -> list[Example]:
    """Generate version string negatives."""
    examples = []
    styles = ["python", "env", "yaml", "json"]

    version_vars = [
        "VERSION", "APP_VERSION", "version", "API_VERSION",
        "SCHEMA_VERSION", "BUILD_VERSION", "RELEASE",
    ]

    for _ in range(count):
        value = rng.choice(VERSION_STRINGS)
        var = rng.choice(version_vars)
        style = rng.choice(styles)
        ctx_type = {
            "python": "python source code",
            "env": "environment file",
            "yaml": "yaml configuration",
            "json": "json configuration",
        }[style]

        target_line = format_assignment(var, value, style, rng)

        n_before = rng.randint(0, 2)
        n_after = rng.randint(0, 2)
        before, after = make_surrounding_lines(rng, ctx_type, n_before, n_after)

        context = before + [target_line] + after
        target_idx = len(before)

        examples.append(Example(
            context_lines=context,
            target_line_idx=target_idx,
            context_type=ctx_type,
            is_secret=False,
            secret_type="none",
            confidence=round(rng.uniform(0.01, 0.05), 2),
            reasoning="Version string — semantic version identifier, not a secret or credential",
            source="synthetic",
        ))

    return examples


# ---------------------------------------------------------------------------
# Stratified train/val split
# ---------------------------------------------------------------------------

def stratified_split(
    examples: list[Example],
    val_ratio: float,
    rng: random.Random,
) -> tuple[list[Example], list[Example]]:
    """Split examples into train/val sets, stratified by (is_secret, secret_type).

    Ensures every (is_secret, secret_type) combination appears in both sets.
    """
    # Group by stratification key
    groups: dict[tuple[bool, str], list[Example]] = defaultdict(list)
    for ex in examples:
        key = (ex.is_secret, ex.secret_type)
        groups[key].append(ex)

    train, val = [], []

    for key, group in sorted(groups.items()):
        rng.shuffle(group)
        n_val = max(1, round(len(group) * val_ratio))  # at least 1 per group
        # But don't take more than half if group is tiny
        if n_val >= len(group):
            n_val = max(1, len(group) // 2)

        val.extend(group[:n_val])
        train.extend(group[n_val:])

    # Final shuffle within each set
    rng.shuffle(train)
    rng.shuffle(val)

    return train, val


# ---------------------------------------------------------------------------
# Main generation pipeline
# ---------------------------------------------------------------------------

def generate_all_examples(
    count: int,
    positive_ratio: float,
    seed: int,
    test_docs_dir: Path,
) -> list[Example]:
    """Generate the full set of training examples."""
    rng = random.Random(seed)

    n_positive = round(count * positive_ratio)
    n_negative = count - n_positive

    # --- 1. Seed examples ---
    seeds = load_seed_examples(test_docs_dir)
    seed_positive = [s for s in seeds if s.is_secret]
    seed_negative = [s for s in seeds if not s.is_secret]

    # --- 2. Synthetic positives ---
    # Target counts for each type (will be adjusted to hit n_positive total)
    remaining_positive = n_positive - len(seed_positive)
    if remaining_positive < 0:
        remaining_positive = 0

    # Distribute across types with approximate ratios
    n_passwords = round(remaining_positive * 0.22)
    n_api_keys = round(remaining_positive * 0.28)
    n_conn_strings = round(remaining_positive * 0.10)
    n_private_keys = round(remaining_positive * 0.08)
    n_jwt_webhook = round(remaining_positive * 0.16)
    n_generic_tokens = remaining_positive - n_passwords - n_api_keys - n_conn_strings - n_private_keys - n_jwt_webhook

    syn_positives = (
        gen_password_examples(n_passwords, rng)
        + gen_api_key_examples(n_api_keys, rng)
        + gen_connection_string_examples(n_conn_strings, rng)
        + gen_private_key_examples(n_private_keys, rng)
        + gen_jwt_webhook_examples(n_jwt_webhook, rng)
        + gen_generic_token_examples(n_generic_tokens, rng)
    )

    # --- 3. Synthetic negatives ---
    remaining_negative = n_negative - len(seed_negative)
    if remaining_negative < 0:
        remaining_negative = 0

    n_hashes = round(remaining_negative * 0.17)
    n_uuids = round(remaining_negative * 0.11)
    n_placeholders = round(remaining_negative * 0.14)
    n_redacted = round(remaining_negative * 0.07)
    n_parts = round(remaining_negative * 0.14)
    n_encoded = round(remaining_negative * 0.11)
    n_numeric = round(remaining_negative * 0.10)
    n_docs = round(remaining_negative * 0.10)
    n_versions = remaining_negative - n_hashes - n_uuids - n_placeholders - n_redacted - n_parts - n_encoded - n_numeric - n_docs

    syn_negatives = (
        gen_hash_negatives(n_hashes, rng)
        + gen_uuid_negatives(n_uuids, rng)
        + gen_placeholder_negatives(n_placeholders, rng)
        + gen_redacted_negatives(n_redacted, rng)
        + gen_part_number_negatives(n_parts, rng)
        + gen_encoded_negatives(n_encoded, rng)
        + gen_numeric_config_negatives(n_numeric, rng)
        + gen_doc_example_negatives(n_docs, rng)
        + gen_version_string_negatives(n_versions, rng)
    )

    # --- Combine ---
    all_examples = seed_positive + seed_negative + syn_positives + syn_negatives
    rng.shuffle(all_examples)

    return all_examples


def print_stats(examples: list[Example], label: str) -> None:
    """Print distribution statistics for a set of examples."""
    total = len(examples)
    positives = [e for e in examples if e.is_secret]
    negatives = [e for e in examples if not e.is_secret]

    print(f"\n{'=' * 60}")
    print(f"  {label}: {total} examples")
    print(f"{'=' * 60}")
    print(f"  Positives: {len(positives)} ({len(positives)/total*100:.1f}%)")
    print(f"  Negatives: {len(negatives)} ({len(negatives)/total*100:.1f}%)")

    # Positive breakdown by type
    type_counts: dict[str, int] = defaultdict(int)
    for e in examples:
        key = e.secret_type if e.is_secret else f"neg:{e.secret_type}"
        type_counts[key] += 1

    print(f"\n  Positive types:")
    for t in sorted(type_counts):
        if not t.startswith("neg:"):
            print(f"    {t:25s} {type_counts[t]:4d}")

    print(f"\n  Negative types:")
    for t in sorted(type_counts):
        if t.startswith("neg:"):
            print(f"    {t:25s} {type_counts[t]:4d}")

    # Source breakdown
    seed_count = sum(1 for e in examples if e.source == "seed")
    syn_count = sum(1 for e in examples if e.source == "synthetic")
    print(f"\n  Sources: seed={seed_count}, synthetic={syn_count}")

    # Confidence distribution
    confs = [e.confidence for e in examples]
    print(f"  Confidence: min={min(confs):.2f}, max={max(confs):.2f}, "
          f"mean={sum(confs)/len(confs):.2f}")


def validate_jsonl(examples: list[dict]) -> list[str]:
    """Validate that all JSONL entries are well-formed."""
    errors = []
    for i, entry in enumerate(examples):
        if "messages" not in entry:
            errors.append(f"Line {i}: missing 'messages' key")
            continue
        msgs = entry["messages"]
        if len(msgs) != 2:
            errors.append(f"Line {i}: expected 2 messages, got {len(msgs)}")
            continue
        if msgs[0]["role"] != "user":
            errors.append(f"Line {i}: first message role should be 'user'")
        if msgs[1]["role"] != "assistant":
            errors.append(f"Line {i}: second message role should be 'assistant'")
        # Validate assistant response is valid JSON
        try:
            parsed = json.loads(msgs[1]["content"])
            if "is_secret" not in parsed:
                errors.append(f"Line {i}: assistant response missing 'is_secret'")
            if "type" not in parsed:
                errors.append(f"Line {i}: assistant response missing 'type'")
            if "confidence" not in parsed:
                errors.append(f"Line {i}: assistant response missing 'confidence'")
        except json.JSONDecodeError as e:
            errors.append(f"Line {i}: assistant response is not valid JSON: {e}")

    return errors


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate LoRA training data for secret-detection SLM",
    )
    parser.add_argument(
        "--output", type=str, default="slm/training_data",
        help="Output path prefix (produces <prefix>_train.jsonl and <prefix>_val.jsonl)",
    )
    parser.add_argument(
        "--count", type=int, default=180,
        help="Total number of examples to generate",
    )
    parser.add_argument(
        "--positive-ratio", type=float, default=0.60,
        help="Fraction of examples that are positive (secrets)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--val-split", type=float, default=0.15,
        help="Fraction of examples for validation set",
    )
    parser.add_argument(
        "--test-docs-dir", type=str, default=None,
        help="Path to test_docs directory (auto-detected if not specified)",
    )

    args = parser.parse_args()

    # Find test docs directory
    if args.test_docs_dir:
        test_docs_dir = Path(args.test_docs_dir)
    else:
        # Try relative to script location, then relative to cwd
        candidates = [
            Path(__file__).parent.parent / "test_docs",
            Path("simplePipeline") / "test_docs",
            Path("test_docs"),
        ]
        test_docs_dir = None
        for c in candidates:
            if c.exists():
                test_docs_dir = c
                break
        if test_docs_dir is None:
            print("ERROR: Could not find test_docs directory. Use --test-docs-dir to specify.", file=sys.stderr)
            sys.exit(1)

    print(f"Using test docs from: {test_docs_dir.resolve()}")

    # Generate
    rng = random.Random(args.seed)
    all_examples = generate_all_examples(
        count=args.count,
        positive_ratio=args.positive_ratio,
        seed=args.seed,
        test_docs_dir=test_docs_dir,
    )

    print(f"\nGenerated {len(all_examples)} total examples")

    # Convert to JSONL format
    jsonl_entries = [example_to_jsonl(ex) for ex in all_examples]

    # Validate
    errors = validate_jsonl(jsonl_entries)
    if errors:
        print(f"\nValidation errors ({len(errors)}):")
        for err in errors[:10]:
            print(f"  {err}")
        sys.exit(1)
    else:
        print("All examples pass JSONL validation.")

    # Stratified split
    train_examples, val_examples = stratified_split(all_examples, args.val_split, rng)
    train_jsonl = [example_to_jsonl(ex) for ex in train_examples]
    val_jsonl = [example_to_jsonl(ex) for ex in val_examples]

    # Check no overlap (by user prompt content)
    train_prompts = {entry["messages"][0]["content"] for entry in train_jsonl}
    val_prompts = {entry["messages"][0]["content"] for entry in val_jsonl}
    overlap = train_prompts & val_prompts
    if overlap:
        print(f"\nWARNING: {len(overlap)} overlapping examples between train and val!")
    else:
        print("No train/val overlap detected.")

    # Write output files
    output_prefix = args.output
    # Create output directory if needed
    output_dir = Path(output_prefix).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = f"{output_prefix}_train.jsonl"
    val_path = f"{output_prefix}_val.jsonl"

    with open(train_path, "w") as f:
        for entry in train_jsonl:
            f.write(json.dumps(entry) + "\n")

    with open(val_path, "w") as f:
        for entry in val_jsonl:
            f.write(json.dumps(entry) + "\n")

    print(f"\nWrote {len(train_jsonl)} training examples to {train_path}")
    print(f"Wrote {len(val_jsonl)} validation examples to {val_path}")

    # Print stats
    print_stats(train_examples, "TRAINING SET")
    print_stats(val_examples, "VALIDATION SET")
    print_stats(all_examples, "COMBINED")


if __name__ == "__main__":
    main()
