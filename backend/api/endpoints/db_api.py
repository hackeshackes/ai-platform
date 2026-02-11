"""
Database API endpoints for AI Platform v5.
Provides RESTful API for CRUD operations on all models.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from db.database import db, init_database
from db.models import User, Project, Experiment, TrainingJob
from db.crud import user_crud, project_crud, experiment_crud, training_job_crud

router = APIRouter(prefix="/api/v1/db", tags=["database"])


# ============== Health Check ==============

@router.get("/health")
async def db_health():
    """Check database health."""
    try:
        conn = db.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        return {"status": "healthy", "database": "SQLite"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/init")
async def initialize_database():
    """Initialize database tables."""
    try:
        init_database()
        return {"status": "initialized"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============== User Endpoints ==============

@router.post("/users")
async def create_user(user: User):
    """Create a new user."""
    existing = user_crud.get_by_email(user.email)
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    created = user_crud.create(user)
    return created.to_dict()


@router.get("/users")
async def get_users(
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0)
):
    """Get all users."""
    users = user_crud.get_all(limit=limit, offset=offset)
    return {"items": [u.to_dict() for u in users], "total": user_crud.count()}


@router.get("/users/{user_id}")
async def get_user(user_id: int):
    """Get user by ID."""
    user = user_crud.get_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user.to_dict()


@router.put("/users/{user_id}")
async def update_user(user_id: int, user: User):
    """Update user."""
    user.id = user_id
    updated = user_crud.update(user)
    return updated.to_dict()


@router.delete("/users/{user_id}")
async def delete_user(user_id: int):
    """Delete user."""
    if not user_crud.delete(user_id):
        raise HTTPException(status_code=404, detail="User not found")
    return {"deleted": True}


# ============== Project Endpoints ==============

@router.post("/projects")
async def create_project(project: Project):
    """Create a new project."""
    created = project_crud.create(project)
    return created.to_dict()


@router.get("/projects")
async def get_projects(
    owner_id: Optional[int] = None,
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0)
):
    """Get all projects, optionally filtered by owner."""
    if owner_id:
        projects = project_crud.get_by_owner(owner_id, limit=limit, offset=offset)
        total = project_crud.count_by_owner(owner_id)
    else:
        projects = project_crud.get_all(limit=limit, offset=offset)
        total = project_crud.count()
    return {"items": [p.to_dict() for p in projects], "total": total}


@router.get("/projects/{project_id}")
async def get_project(project_id: int):
    """Get project by ID."""
    project = project_crud.get_by_id(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project.to_dict()


@router.put("/projects/{project_id}")
async def update_project(project_id: int, project: Project):
    """Update project."""
    project.id = project_id
    updated = project_crud.update(project)
    return updated.to_dict()


@router.delete("/projects/{project_id}")
async def delete_project(project_id: int):
    """Delete project."""
    if not project_crud.delete(project_id):
        raise HTTPException(status_code=404, detail="Project not found")
    return {"deleted": True}


# ============== Experiment Endpoints ==============

@router.post("/experiments")
async def create_experiment(experiment: Experiment):
    """Create a new experiment."""
    created = experiment_crud.create(experiment)
    return created.to_dict()


@router.get("/experiments")
async def get_experiments(
    project_id: Optional[int] = None,
    status: Optional[str] = None,
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0)
):
    """Get all experiments, optionally filtered."""
    if project_id:
        experiments = experiment_crud.get_by_project(project_id, limit=limit, offset=offset)
        total = experiment_crud.count_by_project(project_id)
    elif status:
        experiments = experiment_crud.get_by_status(status, limit=limit, offset=offset)
        total = experiment_crud.count_by_status(status)
    else:
        experiments = experiment_crud.get_all(limit=limit, offset=offset)
        total = experiment_crud.count()
    return {"items": [e.to_dict() for e in experiments], "total": total}


@router.get("/experiments/{experiment_id}")
async def get_experiment(experiment_id: int):
    """Get experiment by ID."""
    experiment = experiment_crud.get_by_id(experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return experiment.to_dict()


@router.put("/experiments/{experiment_id}")
async def update_experiment(experiment_id: int, experiment: Experiment):
    """Update experiment."""
    experiment.id = experiment_id
    updated = experiment_crud.update(experiment)
    return updated.to_dict()


@router.patch("/experiments/{experiment_id}/status")
async def update_experiment_status(experiment_id: int, status: str):
    """Update experiment status."""
    updated = experiment_crud.update_status(experiment_id, status)
    if not updated:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return updated.to_dict()


@router.delete("/experiments/{experiment_id}")
async def delete_experiment(experiment_id: int):
    """Delete experiment."""
    if not experiment_crud.delete(experiment_id):
        raise HTTPException(status_code=404, detail="Experiment not found")
    return {"deleted": True}


# ============== Training Job Endpoints ==============

@router.post("/training-jobs")
async def create_training_job(job: TrainingJob):
    """Create a new training job."""
    created = training_job_crud.create(job)
    return created.to_dict()


@router.get("/training-jobs")
async def get_training_jobs(
    experiment_id: Optional[int] = None,
    status: Optional[str] = None,
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0)
):
    """Get all training jobs, optionally filtered."""
    if experiment_id:
        jobs = training_job_crud.get_by_experiment(experiment_id, limit=limit, offset=offset)
        total = training_job_crud.count_by_experiment(experiment_id)
    elif status:
        jobs = training_job_crud.get_by_status(status, limit=limit, offset=offset)
        total = training_job_crud.count_by_status(status)
    else:
        jobs = training_job_crud.get_all(limit=limit, offset=offset)
        total = training_job_crud.count()
    return {"items": [j.to_dict() for j in jobs], "total": total}


@router.get("/training-jobs/{job_id}")
async def get_training_job(job_id: int):
    """Get training job by ID."""
    job = training_job_crud.get_by_id(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")
    return job.to_dict()


@router.put("/training-jobs/{job_id}")
async def update_training_job(job_id: int, job: TrainingJob):
    """Update training job."""
    job.id = job_id
    updated = training_job_crud.update(job)
    return updated.to_dict()


@router.patch("/training-jobs/{job_id}/status")
async def update_training_job_status(job_id: int, status: str):
    """Update training job status."""
    updated = training_job_crud.update_status(job_id, status)
    if not updated:
        raise HTTPException(status_code=404, detail="Training job not found")
    return updated.to_dict()


@router.patch("/training-jobs/{job_id}/metrics")
async def update_training_job_metrics(job_id: int, metrics: str):
    """Update training job metrics."""
    updated = training_job_crud.update_metrics(job_id, metrics)
    if not updated:
        raise HTTPException(status_code=404, detail="Training job not found")
    return updated.to_dict()


@router.delete("/training-jobs/{job_id}")
async def delete_training_job(job_id: int):
    """Delete training job."""
    if not training_job_crud.delete(job_id):
        raise HTTPException(status_code=404, detail="Training job not found")
    return {"deleted": True}


# ============== Statistics Endpoints ==============

@router.get("/stats")
async def get_database_stats():
    """Get database statistics."""
    return {
        "users": user_crud.count(),
        "projects": project_crud.count(),
        "experiments": experiment_crud.count(),
        "training_jobs": training_job_crud.count(),
    }
