"""
User CRUD operations for AI Platform v5.
"""

from typing import Optional, List
import sqlite3
from datetime import datetime

from ..database import db
from ..models import User


class UserCRUD:
    """CRUD operations for User model."""

    @staticmethod
    def create(user: User) -> User:
        """Create a new user."""
        conn = db.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO users (email, name, password_hash, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (user.email, user.name, user.password_hash, datetime.now().isoformat())
        )
        user.id = cursor.lastrowid
        db.commit()
        return user

    @staticmethod
    def get_by_id(user_id: int) -> Optional[User]:
        """Get user by ID."""
        conn = db.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
        row = cursor.fetchone()
        return User.from_row(row) if row else None

    @staticmethod
    def get_by_email(email: str) -> Optional[User]:
        """Get user by email."""
        conn = db.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
        row = cursor.fetchone()
        return User.from_row(row) if row else None

    @staticmethod
    def get_all(limit: int = 100, offset: int = 0) -> List[User]:
        """Get all users."""
        conn = db.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users ORDER BY created_at DESC LIMIT ? OFFSET ?", (limit, offset))
        return [User.from_row(row) for row in cursor.fetchall()]

    @staticmethod
    def update(user: User) -> User:
        """Update user."""
        conn = db.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE users SET email = ?, name = ?, password_hash = ?
            WHERE id = ?
            """,
            (user.email, user.name, user.password_hash, user.id)
        )
        db.commit()
        return user

    @staticmethod
    def delete(user_id: int) -> bool:
        """Delete user by ID."""
        conn = db.get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
        deleted = cursor.rowcount > 0
        db.commit()
        return deleted

    @staticmethod
    def count() -> int:
        """Count total users."""
        conn = db.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM users")
        return cursor.fetchone()[0]


# Singleton instance
user_crud = UserCRUD()
