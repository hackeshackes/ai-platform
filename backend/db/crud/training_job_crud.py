"""
TrainingJob CRUD operations for AI Platform v5.
"""

from typing import Optional, List
import sqlite3
from datetime import datetime

from ..database import db
from ..models import TrainingJob


class TrainingJobCRUD:
    """CRUD operations for TrainingJob model."""

    @staticmethod
    def create(job: TrainingJob) -> TrainingJob:
        """Create a new training job."""
        conn = db.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO training_jobs (name, config, experiment_id, status, metrics, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (job.name, job.config, job.experiment_id, job.status, job.metrics, datetime.now().isoformat(), datetime.now().isoformat())
        )
        job.id = cursor.lastrowid
        db.commit()
        return job

    @staticmethod
    def get_by_id(job_id: int) -> Optional[TrainingJob]:
        """Get training job by ID."""
        conn = db.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM training_jobs WHERE id = ?", (job_id,))
        row = cursor.fetchone()
        return TrainingJob.from_row(row) if row else None

    @staticmethod
    def get_by_experiment(experiment_id: int, limit: int = 100, offset: int = 0) -> List[TrainingJob]:
        """Get training jobs by experiment ID."""
        conn = db.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM training_jobs WHERE experiment_id = ? ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (experiment_id, limit, offset)
        )
        return [TrainingJob.from_row(row) for row in cursor.fetchall()]

    @staticmethod
    def get_by_status(status: str, limit: int = 100, offset: int = 0) -> List[TrainingJob]:
        """Get training jobs by status."""
        conn = db.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM training_jobs WHERE status = ? ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (status, limit, offset)
        )
        return [TrainingJob.from_row(row) for row in cursor.fetchall()]

    @staticmethod
    def get_all(limit: int = 100, offset: int = 0) -> List[TrainingJob]:
        """Get all training jobs."""
        conn = db.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM training_jobs ORDER BY created_at DESC LIMIT ? OFFSET ?", (limit, offset))
        return [TrainingJob.from_row(row) for row in cursor.fetchall()]

    @staticmethod
    def update(job: TrainingJob) -> TrainingJob:
        """Update training job."""
        conn = db.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE training_jobs SET name = ?, config = ?, status = ?, metrics = ?, updated_at = ?
            WHERE id = ?
            """,
            (job.name, job.config, job.status, job.metrics, datetime.now().isoformat(), job.id)
        )
        db.commit()
        return job

    @staticmethod
    def update_status(job_id: int, status: str) -> Optional[TrainingJob]:
        """Update training job status."""
        conn = db.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE training_jobs SET status = ?, updated_at = ?
            WHERE id = ?
            """,
            (status, datetime.now().isoformat(), job_id)
        )
        db.commit()
        if cursor.rowcount > 0:
            return TrainingJobCRUD.get_by_id(job_id)
        return None

    @staticmethod
    def update_metrics(job_id: int, metrics: str) -> Optional[TrainingJob]:
        """Update training job metrics."""
        conn = db.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE training_jobs SET metrics = ?, updated_at = ?
            WHERE id = ?
            """,
            (metrics, datetime.now().isoformat(), job_id)
        )
        db.commit()
        if cursor.rowcount > 0:
            return TrainingJobCRUD.get_by_id(job_id)
        return None

    @staticmethod
    def delete(job_id: int) -> bool:
        """Delete training job by ID."""
        conn = db.get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM training_jobs WHERE id = ?", (job_id,))
        deleted = cursor.rowcount > 0
        db.commit()
        return deleted

    @staticmethod
    def count() -> int:
        """Count total training jobs."""
        conn = db.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM training_jobs")
        return cursor.fetchone()[0]

    @staticmethod
    def count_by_experiment(experiment_id: int) -> int:
        """Count training jobs by experiment."""
        conn = db.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM training_jobs WHERE experiment_id = ?", (experiment_id,))
        return cursor.fetchone()[0]

    @staticmethod
    def count_by_status(status: str) -> int:
        """Count training jobs by status."""
        conn = db.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM training_jobs WHERE status = ?", (status,))
        return cursor.fetchone()[0]


# Singleton instance
training_job_crud = TrainingJobCRUD()
