"""
数据库管理API端点
"""
from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
import sys
import os

# 添加backend到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.database.pool import get_pool, PostgresPool, execute_read, execute_write

router = APIRouter(prefix="/db", tags=["Database"])


class DatabaseStatusResponse(BaseModel):
    """数据库状态响应"""
    type: str
    pool: Dict[str, Any]
    tables: Optional[List[str]] = None
    health: Dict[str, Any]


class QueryRequest(BaseModel):
    """查询请求"""
    query: str
    params: Optional[List[Any]] = None


class QueryResponse(BaseModel):
    """查询响应"""
    results: List[Dict[str, Any]]
    row_count: int


@router.get("/status", response_model=DatabaseStatusResponse)
async def db_status():
    """
    获取数据库状态
    """
    try:
        pool = get_pool()
        stats = pool.get_stats()
        
        return {
            "type": "postgresql",
            "pool": stats,
            "tables": None,  # 可以添加表列表查询
            "health": stats.get("health", {})
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def db_health():
    """
    数据库健康检查
    """
    try:
        pool = get_pool()
        health = pool.health_check()
        
        main_healthy = health.get("main_pool", {}).get("healthy", False)
        read_healthy = health.get("read_pool", {}).get("healthy", False)
        
        if main_healthy or read_healthy:
            return {
                "status": "healthy",
                "main_pool": main_healthy,
                "read_pool": read_healthy
            }
        else:
            raise HTTPException(status_code=503, detail="Database unhealthy")
    
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@router.get("/stats")
async def db_stats():
    """
    获取数据库统计信息
    """
    try:
        pool = get_pool()
        stats = pool.get_stats()
        
        return {
            "config": stats.get("config", {}),
            "initialized": stats.get("initialized", False)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query")
async def execute_query(request: QueryRequest):
    """
    执行SQL查询（只读）
    """
    try:
        results = execute_read(request.query, tuple(request.params) if request.params else None)
        
        # 转换为字典列表
        rows = []
        for row in results:
            if isinstance(row, dict):
                rows.append(row)
            else:
                # 如果是tuple，转换为字典
                # 注意：这里需要知道列名，生产环境应使用列名
                rows.append({"row": row})
        
        return {
            "results": rows,
            "row_count": len(rows)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/tables")
async def list_tables():
    """
    列出所有表
    """
    try:
        query = """
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            ORDER BY table_name
        """
        results = execute_read(query)
        
        tables = [row[0] for row in results]
        
        return {"tables": tables}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tables/{table_name}")
async def get_table_info(table_name: str):
    """
    获取表信息
    """
    try:
        # 获取列信息
        query = """
            SELECT column_name, data_type, is_nullable, column_default
            FROM information_schema.columns
            WHERE table_name = %s
            ORDER BY ordinal_position
        """
        results = execute_read(query, (table_name,))
        
        columns = []
        for row in results:
            columns.append({
                "name": row[0],
                "type": row[1],
                "nullable": row[2] == "YES",
                "default": row[3]
            })
        
        # 获取行数
        count_query = f"SELECT COUNT(*) FROM \"{table_name}\""
        count_result = execute_read(count_query)
        row_count = count_result[0][0] if count_result else 0
        
        return {
            "name": table_name,
            "columns": columns,
            "row_count": row_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 业务表查询 ====================

@router.get("/users")
async def list_users(limit: int = 10, offset: int = 0):
    """
    列出用户
    """
    try:
        query = """
            SELECT id, username, email, full_name, is_active, created_at
            FROM users
            ORDER BY created_at DESC
            LIMIT %s OFFSET %s
        """
        results = execute_read(query, (limit, offset))
        
        users = []
        for row in results:
            users.append({
                "id": row[0],
                "username": row[1],
                "email": row[2],
                "full_name": row[3],
                "is_active": row[4],
                "created_at": row[5]
            })
        
        return {"users": users}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/projects")
async def list_projects(limit: int = 10, offset: int = 0):
    """
    列出项目
    """
    try:
        query = """
            SELECT id, name, description, owner_id, status, created_at
            FROM projects
            ORDER BY created_at DESC
            LIMIT %s OFFSET %s
        """
        results = execute_read(query, (limit, offset))
        
        projects = []
        for row in results:
            projects.append({
                "id": row[0],
                "name": row[1],
                "description": row[2],
                "owner_id": row[3],
                "status": row[4],
                "created_at": row[5]
            })
        
        return {"projects": projects}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/agents")
async def list_agents(limit: int = 10, offset: int = 0):
    """
    列出Agents
    """
    try:
        query = """
            SELECT id, name, type, project_id, status, created_at
            FROM agents
            ORDER BY created_at DESC
            LIMIT %s OFFSET %s
        """
        results = execute_read(query, (limit, offset))
        
        agents = []
        for row in results:
            agents.append({
                "id": row[0],
                "name": row[1],
                "type": row[2],
                "project_id": row[3],
                "status": row[4],
                "created_at": row[5]
            })
        
        return {"agents": agents}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/datasets")
async def list_datasets(limit: int = 10, offset: int = 0):
    """
    列出数据集
    """
    try:
        query = """
            SELECT id, name, project_id, data_type, size_bytes, row_count, created_at
            FROM datasets
            ORDER BY created_at DESC
            LIMIT %s OFFSET %s
        """
        results = execute_read(query, (limit, offset))
        
        datasets = []
        for row in results:
            datasets.append({
                "id": row[0],
                "name": row[1],
                "project_id": row[2],
                "data_type": row[3],
                "size_bytes": row[4],
                "row_count": row[5],
                "created_at": row[6]
            })
        
        return {"datasets": datasets}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models")
async def list_models(limit: int = 10, offset: int = 0):
    """
    列出模型
    """
    try:
        query = """
            SELECT id, name, version, model_type, project_id, status, created_at
            FROM models
            ORDER BY created_at DESC
            LIMIT %s OFFSET %s
        """
        results = execute_read(query, (limit, offset))
        
        models = []
        for row in results:
            models.append({
                "id": row[0],
                "name": row[1],
                "version": row[2],
                "model_type": row[3],
                "project_id": row[4],
                "status": row[5],
                "created_at": row[6]
            })
        
        return {"models": models}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/experiments")
async def list_experiments(limit: int = 10, offset: int = 0):
    """
    列出实验
    """
    try:
        query = """
            SELECT id, name, project_id, model_id, status, started_at, created_at
            FROM experiments
            ORDER BY created_at DESC
            LIMIT %s OFFSET %s
        """
        results = execute_read(query, (limit, offset))
        
        experiments = []
        for row in results:
            experiments.append({
                "id": row[0],
                "name": row[1],
                "project_id": row[2],
                "model_id": row[3],
                "status": row[4],
                "started_at": row[5],
                "created_at": row[6]
            })
        
        return {"experiments": experiments}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tasks")
async def list_tasks(
    limit: int = 10, 
    offset: int = 0,
    status: Optional[str] = None,
    project_id: Optional[int] = None
):
    """
    列出任务
    """
    try:
        base_query = """
            FROM tasks
            WHERE 1=1
        """
        params = []
        
        if status:
            base_query += " AND status = %s"
            params.append(status)
        
        if project_id:
            base_query += " AND project_id = %s"
            params.append(project_id)
        
        # 查询数据
        query = f"""
            SELECT id, name, task_type, project_id, agent_id, status, priority, created_at
            {base_query}
            ORDER BY priority DESC, created_at DESC
            LIMIT %s OFFSET %s
        """
        params.extend([limit, offset])
        
        results = execute_read(query, tuple(params))
        
        tasks = []
        for row in results:
            tasks.append({
                "id": row[0],
                "name": row[1],
                "task_type": row[2],
                "project_id": row[3],
                "agent_id": row[4],
                "status": row[5],
                "priority": row[6],
                "created_at": row[7]
            })
        
        return {"tasks": tasks}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/audit-logs")
async def list_audit_logs(limit: int = 10, offset: int = 0):
    """
    列出审计日志
    """
    try:
        query = """
            SELECT id, user_id, action, resource_type, resource_id, ip_address, created_at
            FROM audit_logs
            ORDER BY created_at DESC
            LIMIT %s OFFSET %s
        """
        results = execute_read(query, (limit, offset))
        
        logs = []
        for row in results:
            logs.append({
                "id": row[0],
                "user_id": row[1],
                "action": row[2],
                "resource_type": row[3],
                "resource_id": row[4],
                "ip_address": row[5],
                "created_at": row[6]
            })
        
        return {"audit_logs": logs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
