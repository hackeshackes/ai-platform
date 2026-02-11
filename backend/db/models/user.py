"""
User model for AI Platform v5.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class User:
    """User data model."""
    id: Optional[int] = None
    email: str = ""
    name: str = ""
    password_hash: str = ""
    created_at: Optional[datetime] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        data = {
            "id": self.id,
            "email": self.email,
            "name": self.name,
            "password_hash": self.password_hash,
        }
        if self.created_at:
            data["created_at"] = self.created_at.isoformat() if isinstance(self.created_at, datetime) else self.created_at
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "User":
        """Create from dictionary."""
        created_at = data.get("created_at")
        if created_at and isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        return cls(
            id=data.get("id"),
            email=data.get("email", ""),
            name=data.get("name", ""),
            password_hash=data.get("password_hash", ""),
            created_at=created_at,
        )

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "User":
        """Create from sqlite3.Row."""
        return cls(
            id=row["id"],
            email=row["email"],
            name=row["name"],
            password_hash=row["password_hash"],
            created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else None,
        )
