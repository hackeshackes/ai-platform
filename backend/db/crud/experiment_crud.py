"""
Experiment CRUD operations for AI Platform v5.
"""

from typing import Optional, List
import sqlite3
from datetime import datetime

from ..database import db
from ..models import Experiment


class ExperimentCRUD:
    """CRUD operations for Experiment model."""

    @staticmethod
    def create(experiment: Experiment) -> Experiment:
        """Create a new experiment."""
        conn = db.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO experiments (name, config, project_id, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (experiment.name, experiment.config, experiment.project_id, experiment.status, datetime.now().isoformat(), datetime.now().isoformat())
        )
        experiment.id = cursor.lastrowid
        db.commit()
        return experiment

    @staticmethod
    def get_by_id(experiment_id: int) -> Optional[Experiment]:
        """Get experiment by ID."""
        conn = db.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM experiments WHERE id = ?", (experiment_id,))
        row = cursor.fetchone()
        return Experiment.from_row(row) if row else None

    @staticmethod
    def get_by_project(project_id: int, limit: int = 100, offset: int = 0) -> List[Experiment]:
        """Get experiments by project ID."""
        conn = db.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM experiments WHERE project_id = ? ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (project_id, limit, offset)
        )
        return [Experiment.from_row(row) for row in cursor.fetchall()]

    @staticmethod
    def get_by_status(status: str, limit: int = 100, offset: int = 0) -> List[Experiment]:
        """Get experiments by status."""
        conn = db.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM experiments WHERE status = ? ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (status, limit, offset)
        )
        return [Experiment.from_row(row) for row in cursor.fetchall()]

    @staticmethod
    def get_all(limit: int = 100, offset: int = 0) -> List[Experiment]:
        """Get all experiments."""
        conn = db.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM experiments ORDER BY created_at DESC LIMIT ? OFFSET ?", (limit, offset))
        return [Experiment.from_row(row) for row in cursor.fetchall()]

    @staticmethod
    def update(experiment: Experiment) -> Experiment:
        """Update experiment."""
        conn = db.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE experiments SET name = ?, config = ?, status = ?, updated_at = ?
            WHERE id = ?
            """,
            (experiment.name, experiment.config, experiment.status, datetime.now().isoformat(), experiment.id)
        )
        db.commit()
        return experiment

    @staticmethod
    def update_status(experiment_id: int, status: str) -> Optional[Experiment]:
        """Update experiment status."""
        conn = db.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE experiments SET status = ?, updated_at = ?
            WHERE id = ?
            """,
            (status, datetime.now().isoformat(), experiment_id)
        )
        db.commit()
        if cursor.rowcount > 0:
            return ExperimentCRUD.get_by_id(experiment_id)
        return None

    @staticmethod
    def delete(experiment_id: int) -> bool:
        """Delete experiment by ID."""
        conn = db.get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM experiments WHERE id = ?", (experiment_id,))
        deleted = cursor.rowcount > 0
        db.commit()
        return deleted

    @staticmethod
    def count() -> int:
        """Count total experiments."""
        conn = db.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM experiments")
        return cursor.fetchone()[0]

    @staticmethod
    def count_by_project(project_id: int) -> int:
        """Count experiments by project."""
        conn = db.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM experiments WHERE project_id = ?", (project_id,))
        return cursor.fetchone()[0]

    @staticmethod
    def count_by_status(status: str) -> int:
        """Count experiments by status."""
        conn = db.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM experiments WHERE status = ?", (status,))
        return cursor.fetchone()[0]


# Singleton instance
experiment_crud = ExperimentCRUD()
