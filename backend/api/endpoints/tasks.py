"""Task management endpoints"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from fastapi import APIRouter, HTTPException
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime, timedelta
import uuid

router = APIRouter()

fake_tasks_db = {
    1: {
        "id": 1,
        "name": "training-task-1",
        "type": "training",
        "project_id": 1,
        "experiment_id": 1,
        "user_id": 1,
        "status": "completed",
        "progress": 100.0,
        "created_at": (datetime.utcnow() - timedelta(hours=6)).isoformat(),
    }
}

class TaskCreate(BaseModel):
    name: str
    type: str
    project_id: Optional[int] = None
    experiment_id: Optional[int] = None
    config: Optional[dict] = None

@router.get("")
async def list_tasks(status: Optional[str] = None, skip: int = 0, limit: int = 100):
    tasks = list(fake_tasks_db.values())
    if status:
        tasks = [t for t in tasks if t["status"] == status]
    return {"total": len(tasks), "tasks": tasks[skip:skip+limit]}

@router.get("/{task_id}")
async def get_task(task_id: int):
    if task_id not in fake_tasks_db:
        raise HTTPException(status_code=404, detail="Task not found")
    return fake_tasks_db[task_id]

@router.post("", response_model=dict, status_code=201)
async def create_task(data: TaskCreate):
    task_id = len(fake_tasks_db) + 1
    new_task = {"id": task_id, **data.dict(), "status": "pending", "progress": 0}
    fake_tasks_db[task_id] = new_task
    return new_task

@router.delete("/{task_id}", status_code=204)
async def delete_task(task_id: int):
    if task_id not in fake_tasks_db:
        raise HTTPException(status_code=404, detail="Task not found")
    del fake_tasks_db[task_id]
