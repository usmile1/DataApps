"""Unified entity store.

Everything is an entity — files, PII findings, directories.
Entities have features. Entities can contain other entities.
This is a simplified graph model backed by SQLite.
"""

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


class Catalog:
    """SQLite-backed entity catalog."""

    def __init__(self, db_path: str = "catalog.db") -> None:
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._create_tables()

    def _create_tables(self) -> None:
        """Create the schema if it doesn't exist."""
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS entities (
                id TEXT PRIMARY KEY,
                entity_type TEXT NOT NULL,
                name TEXT NOT NULL,
                created_at TEXT NOT NULL,
                scan_id TEXT
            );

            CREATE TABLE IF NOT EXISTS entity_features (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entity_id TEXT NOT NULL,
                feature_name TEXT NOT NULL,
                feature_value TEXT,
                pipeline_version TEXT,
                computed_at TEXT NOT NULL,
                FOREIGN KEY (entity_id) REFERENCES entities(id)
            );

            CREATE TABLE IF NOT EXISTS entity_relationships (
                parent_id TEXT NOT NULL,
                child_id TEXT NOT NULL,
                relationship_type TEXT NOT NULL DEFAULT 'contains',
                PRIMARY KEY (parent_id, child_id),
                FOREIGN KEY (parent_id) REFERENCES entities(id),
                FOREIGN KEY (child_id) REFERENCES entities(id)
            );

            CREATE TABLE IF NOT EXISTS scan_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scan_id TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                layer TEXT NOT NULL,
                latency_ms REAL,
                entities_found INTEGER DEFAULT 0,
                escalated BOOLEAN DEFAULT 0,
                scanned_at TEXT NOT NULL,
                FOREIGN KEY (entity_id) REFERENCES entities(id)
            );

            CREATE TABLE IF NOT EXISTS runs (
                scan_id TEXT PRIMARY KEY,
                slug TEXT NOT NULL,
                description TEXT,
                pipeline_version TEXT NOT NULL,
                config_snapshot TEXT,
                dataset TEXT,
                created_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_runs_slug
                ON runs(slug);
            CREATE INDEX IF NOT EXISTS idx_features_entity
                ON entity_features(entity_id);
            CREATE INDEX IF NOT EXISTS idx_relationships_parent
                ON entity_relationships(parent_id);
            CREATE INDEX IF NOT EXISTS idx_relationships_child
                ON entity_relationships(child_id);
            CREATE INDEX IF NOT EXISTS idx_metrics_scan
                ON scan_metrics(scan_id);
        """)
        self._conn.commit()

    def new_scan_id(self) -> str:
        """Generate a unique scan ID for this pipeline run."""
        return f"scan-{uuid.uuid4().hex[:8]}"

    def create_run(self, scan_id: str, slug: str, description: str,
                   pipeline_version: str, config: Dict,
                   dataset: str) -> None:
        """Record a pipeline run with its configuration."""
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "INSERT INTO runs "
            "(scan_id, slug, description, pipeline_version, config_snapshot, dataset, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (scan_id, slug, description, pipeline_version,
             json.dumps(config, indent=2), dataset, now)
        )
        self._conn.commit()

    def get_run(self, scan_id: str) -> Optional[Dict]:
        """Get a run by scan_id."""
        row = self._conn.execute(
            "SELECT * FROM runs WHERE scan_id = ?", (scan_id,)
        ).fetchone()
        return dict(row) if row else None

    def get_runs_by_slug(self, slug: str) -> List[Dict]:
        """Get all runs matching a slug pattern."""
        rows = self._conn.execute(
            "SELECT * FROM runs WHERE slug LIKE ? ORDER BY created_at",
            (f"%{slug}%",)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_all_runs(self) -> List[Dict]:
        """Get all runs, ordered by creation time."""
        rows = self._conn.execute(
            "SELECT * FROM runs ORDER BY created_at"
        ).fetchall()
        return [dict(r) for r in rows]

    def store_entity(self, entity_id: str, entity_type: str,
                     name: str, scan_id: str) -> str:
        """Insert an entity. Returns the entity ID."""
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "INSERT OR REPLACE INTO entities (id, entity_type, name, created_at, scan_id) "
            "VALUES (?, ?, ?, ?, ?)",
            (entity_id, entity_type, name, now, scan_id)
        )
        self._conn.commit()
        return entity_id

    def store_features(self, entity_id: str,
                       features: Dict[str, Any],
                       pipeline_version: str = "") -> None:
        """Store multiple features for an entity."""
        now = datetime.now(timezone.utc).isoformat()
        rows = [
            (entity_id, name, str(value), pipeline_version, now)
            for name, value in features.items()
        ]
        self._conn.executemany(
            "INSERT INTO entity_features "
            "(entity_id, feature_name, feature_value, pipeline_version, computed_at) "
            "VALUES (?, ?, ?, ?, ?)",
            rows
        )
        self._conn.commit()

    def store_relationship(self, parent_id: str, child_id: str,
                           relationship_type: str = "contains") -> None:
        """Link a child entity to a parent entity."""
        self._conn.execute(
            "INSERT OR REPLACE INTO entity_relationships "
            "(parent_id, child_id, relationship_type) VALUES (?, ?, ?)",
            (parent_id, child_id, relationship_type)
        )
        self._conn.commit()

    def store_scan_metric(self, scan_id: str, entity_id: str,
                          layer: str, latency_ms: float,
                          entities_found: int = 0,
                          escalated: bool = False) -> None:
        """Record a layer's performance for a specific entity in a scan."""
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "INSERT INTO scan_metrics "
            "(scan_id, entity_id, layer, latency_ms, entities_found, escalated, scanned_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (scan_id, entity_id, layer, latency_ms, entities_found, escalated, now)
        )
        self._conn.commit()

    # --- Query methods ---

    def get_entity(self, entity_id: str) -> Optional[Dict]:
        """Retrieve an entity by ID."""
        row = self._conn.execute(
            "SELECT * FROM entities WHERE id = ?", (entity_id,)
        ).fetchone()
        return dict(row) if row else None

    def get_features(self, entity_id: str) -> Dict[str, str]:
        """Get all features for an entity as a dict."""
        rows = self._conn.execute(
            "SELECT feature_name, feature_value FROM entity_features "
            "WHERE entity_id = ?", (entity_id,)
        ).fetchall()
        return {row["feature_name"]: row["feature_value"] for row in rows}

    def get_children(self, parent_id: str,
                     entity_type: Optional[str] = None) -> List[Dict]:
        """Get child entities of a parent, optionally filtered by type."""
        if entity_type:
            rows = self._conn.execute(
                "SELECT e.* FROM entities e "
                "JOIN entity_relationships r ON e.id = r.child_id "
                "WHERE r.parent_id = ? AND e.entity_type = ?",
                (parent_id, entity_type)
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT e.* FROM entities e "
                "JOIN entity_relationships r ON e.id = r.child_id "
                "WHERE r.parent_id = ?", (parent_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    def get_scan_metrics(self, scan_id: str) -> List[Dict]:
        """Get all metrics for a scan run."""
        rows = self._conn.execute(
            "SELECT * FROM scan_metrics WHERE scan_id = ? ORDER BY entity_id, layer",
            (scan_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_all_entities_by_type(self, entity_type: str,
                                 scan_id: Optional[str] = None) -> List[Dict]:
        """Get all entities of a given type, optionally for a specific scan."""
        if scan_id:
            rows = self._conn.execute(
                "SELECT * FROM entities WHERE entity_type = ? AND scan_id = ?",
                (entity_type, scan_id)
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM entities WHERE entity_type = ?",
                (entity_type,)
            ).fetchall()
        return [dict(r) for r in rows]

    def get_entity_count_by_type(self, scan_id: str) -> Dict[str, int]:
        """Count entities by type for a given scan."""
        rows = self._conn.execute(
            "SELECT entity_type, COUNT(*) as count FROM entities "
            "WHERE scan_id = ? GROUP BY entity_type",
            (scan_id,)
        ).fetchall()
        return {row["entity_type"]: row["count"] for row in rows}

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
