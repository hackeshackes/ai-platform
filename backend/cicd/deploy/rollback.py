"""
回滚机制 - Phase 2
"""
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
from uuid import uuid4

class RollbackStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class RollbackRecord:
    rollback_id: str
    deployment_id: str
    from_version: str
    to_version: str
    reason: str
    status: RollbackStatus
    created_at: datetime
    completed_at: Optional[datetime] = None
    error: Optional[str] = None

class RollbackManager:
    """回滚管理器"""
    
    def __init__(self):
        self.rollback_queue: List[RollbackRecord] = []
        self.rollback_history: Dict[str, RollbackRecord] = {}
    
    async def create_rollback(
        self,
        deployment_id: str,
        from_version: str,
        to_version: str,
        reason: str = "Manual rollback"
    ) -> RollbackRecord:
        rollback = RollbackRecord(
            rollback_id=str(uuid4()),
            deployment_id=deployment_id,
            from_version=from_version,
            to_version=to_version,
            reason=reason,
            status=RollbackStatus.PENDING,
            created_at=datetime.utcnow()
        )
        self.rollback_queue.append(rollback)
        return rollback
    
    async def execute_rollback(self, rollback_id: str) -> RollbackRecord:
        rollback = next((r for r in self.rollback_queue if r.rollback_id == rollback_id), None)
        if not rollback:
            raise ValueError(f"Rollback {rollback_id} not found")
        
        rollback.status = RollbackStatus.IN_PROGRESS
        rollback.status = RollbackStatus.COMPLETED
        rollback.completed_at = datetime.utcnow()
        
        self.rollback_history[rollback_id] = rollback
        self.rollback_queue = [r for r in self.rollback_queue if r.rollback_id != rollback_id]
        
        return rollback
    
    async def quick_rollback(self, deployment_id: str) -> RollbackRecord:
        return await self.create_rollback(
            deployment_id=deployment_id,
            from_version="current",
            to_version="previous",
            reason="Quick rollback"
        )
    
    def get_rollback_queue(self) -> List[Dict]:
        return [{"rollback_id": r.rollback_id, "to_version": r.to_version, "status": r.status.value} for r in self.rollback_queue]
    
    def get_history(self, limit: int = 10) -> List[Dict]:
        records = list(self.rollback_history.values())[-limit:]
        return [{"rollback_id": r.rollback_id, "to_version": r.to_version, "status": r.status.value} for r in records]

rollback_manager = RollbackManager()
