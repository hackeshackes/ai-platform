"""
Database connection module for AI Platform v5.
Uses SQLite as the database and Prisma-style async patterns.
"""

import sqlite3
import asyncio
from pathlib import Path
from typing import Any, Optional
from contextlib import asynccontextmanager

DATABASE_PATH = Path(__file__).parent.parent.parent / "data" / "ai_platform.db"


class Database:
    """Database connection manager."""

    def __init__(self, db_path: Path = DATABASE_PATH):
        self.db_path = db_path
        self._connection: Optional[sqlite3.Connection] = None
        self._local = type('Local', (), {})()  # Simple object for storage

    def _ensure_db_exists(self):
        """Ensure the database directory and file exist."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.db_path.exists():
            self.db_path.touch()

    def get_connection(self) -> sqlite3.Connection:
        """Get or create a database connection."""
        if not hasattr(self._local, 'connection') or self._local.connection is None:
            self._ensure_db_exists()
            self._local.connection = sqlite3.connect(str(self.db_path))
            self._local.connection.row_factory = sqlite3.Row
            self._local.connection.execute("PRAGMA foreign_keys = ON")
        return self._local.connection

    def close_connection(self):
        """Close the current connection."""
        if hasattr(self._local, 'connection') and self._local.connection:
            self._local.connection.close()
            self._local.connection = None

    def commit(self):
        """Commit the current transaction."""
        if hasattr(self._local, 'connection') and self._local.connection:
            self._local.connection.commit()

    def rollback(self):
        """Rollback the current transaction."""
        if hasattr(self._local, 'connection') and self._local.connection:
            self._local.connection.rollback()


# Global database instance
db = Database()


def init_database():
    """Initialize database tables."""
    conn = db.get_connection()
    cursor = conn.cursor()

    # Users table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            password_hash TEXT,
            provider TEXT DEFAULT 'local',
            provider_id TEXT,
            is_active INTEGER DEFAULT 1,
            role TEXT DEFAULT 'user',
            created_at TEXT,
            updated_at TEXT
        )
    """)

    # Projects table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS projects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            description TEXT,
            owner_id INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (owner_id) REFERENCES users(id) ON DELETE CASCADE
        )
    """)

    # Experiments table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS experiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            config TEXT,  -- JSON string
            project_id INTEGER NOT NULL,
            status TEXT DEFAULT 'pending',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
        )
    """)

    # Training jobs table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS training_jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            config TEXT,  -- JSON string
            experiment_id INTEGER NOT NULL,
            status TEXT DEFAULT 'pending',
            metrics TEXT,  -- JSON string
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (experiment_id) REFERENCES experiments(id) ON DELETE CASCADE
        )
    """)

    conn.commit()
    print("Database initialized successfully.")


if __name__ == "__main__":
    init_database()
