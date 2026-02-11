"""
Experiment model for AI Platform v5.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class Experiment:
    """Experiment data model."""
    id: Optional[int] = None
    name: str = ""
    config: Optional[str] = ""  # JSON string
    project_id: Optional[int] = None
    status: str = "pending"
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        data = {
            "id": self.id,
            "name": self.name,
            "config": self.config,
            "project_id": self.project_id,
            "status": self.status,
        }
        if self.created_at:
            data["created_at"] = self.created_at.isoformat() if isinstance(self.created_at, datetime) else self.created_at
        if self.updated_at:
            data["updated_at"] = self.updated_at.isoformat() if isinstance(self.updated_at, datetime) else self.updated_at
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "Experiment":
        """Create from dictionary."""
        created_at = data.get("created_at")
        updated_at = data.get("updated_at")
        if created_at and isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        if updated_at and isinstance(updated_at, str):
            updated_at = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
        return cls(
            id=data.get("id"),
            name=data.get("name", ""),
            config=data.get("config"),
            project_id=data.get("project_id"),
            status=data.get("status", "pending"),
            created_at=created_at,
            updated_at=updated_at,
        )

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Experiment":
        """Create from sqlite3.Row."""
        return cls(
            id=row["id"],
            name=row["name"],
            config=row["config"],
            project_id=row["project_id"],
            status=row["status"],
            created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else None,
            updated_at=datetime.fromisoformat(row["updated_at"]) if row["updated_at"] else None,
        )
