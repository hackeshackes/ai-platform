"""
Project CRUD operations for AI Platform v5.
"""

from typing import Optional, List
import sqlite3
from datetime import datetime

from ..database import db
from ..models import Project


class ProjectCRUD:
    """CRUD operations for Project model."""

    @staticmethod
    def create(project: Project) -> Project:
        """Create a new project."""
        conn = db.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO projects (name, description, owner_id, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (project.name, project.description, project.owner_id, datetime.now().isoformat(), datetime.now().isoformat())
        )
        project.id = cursor.lastrowid
        db.commit()
        return project

    @staticmethod
    def get_by_id(project_id: int) -> Optional[Project]:
        """Get project by ID."""
        conn = db.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM projects WHERE id = ?", (project_id,))
        row = cursor.fetchone()
        return Project.from_row(row) if row else None

    @staticmethod
    def get_by_owner(owner_id: int, limit: int = 100, offset: int = 0) -> List[Project]:
        """Get projects by owner ID."""
        conn = db.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM projects WHERE owner_id = ? ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (owner_id, limit, offset)
        )
        return [Project.from_row(row) for row in cursor.fetchall()]

    @staticmethod
    def get_all(limit: int = 100, offset: int = 0) -> List[Project]:
        """Get all projects."""
        conn = db.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM projects ORDER BY created_at DESC LIMIT ? OFFSET ?", (limit, offset))
        return [Project.from_row(row) for row in cursor.fetchall()]

    @staticmethod
    def update(project: Project) -> Project:
        """Update project."""
        conn = db.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE projects SET name = ?, description = ?, updated_at = ?
            WHERE id = ?
            """,
            (project.name, project.description, datetime.now().isoformat(), project.id)
        )
        db.commit()
        return project

    @staticmethod
    def delete(project_id: int) -> bool:
        """Delete project by ID."""
        conn = db.get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM projects WHERE id = ?", (project_id,))
        deleted = cursor.rowcount > 0
        db.commit()
        return deleted

    @staticmethod
    def count() -> int:
        """Count total projects."""
        conn = db.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM projects")
        return cursor.fetchone()[0]

    @staticmethod
    def count_by_owner(owner_id: int) -> int:
        """Count projects by owner."""
        conn = db.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM projects WHERE owner_id = ?", (owner_id,))
        return cursor.fetchone()[0]


# Singleton instance
project_crud = ProjectCRUD()
