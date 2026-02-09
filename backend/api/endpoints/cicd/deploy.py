"""
CI/CD API端点 v2.0 Phase 2
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from fastapi import APIRouter, HTTPException, Depends
from typing import List, Optional
from pydantic import BaseModel

from cicd.deploy.blue_green import blue_green_deployer
from cicd.deploy.rollback import rollback_manager
from cicd.quality.gates import quality_gates
from api.endpoints.auth import get_current_user

router = APIRouter()

class DeployRequest(BaseModel):
    version: str
    environment: str = "production"

class RollbackRequest(BaseModel):
    deployment_id: str
    to_version: Optional[str] = None
    reason: Optional[str] = None

@router.post("/deploy")
async def deploy(request: DeployRequest, current_user = Depends(get_current_user)):
    """执行蓝绿部署"""
    deployment = await blue_green_deployer.deploy(request.version, request.environment)
    return {"deployment_id": deployment.deployment_id, "status": deployment.status.value}

@router.get("/deploy/status/{deployment_id}")
async def get_deploy_status(deployment_id: str, current_user = Depends(get_current_user)):
    """获取部署状态"""
    try:
        return await blue_green_deployer.get_status(deployment_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.post("/rollback")
async def create_rollback(request: RollbackRequest, current_user = Depends(get_current_user)):
    """创建回滚"""
    rollback = await rollback_manager.create_rollback(
        deployment_id=request.deployment_id,
        from_version="current",
        to_version=request.to_version or "previous",
        reason=request.reason or "Manual rollback"
    )
    result = await rollback_manager.execute_rollback(rollback.rollback_id)
    return {"rollback_id": result.rollback_id, "status": result.status.value}

@router.post("/rollback/quick/{deployment_id}")
async def quick_rollback(deployment_id: str, current_user = Depends(get_current_user)):
    """快速回滚"""
    result = await rollback_manager.quick_rollback(deployment_id)
    return {"rollback_id": result.rollback_id, "status": result.status.value}

@router.get("/rollback/queue")
async def get_rollback_queue(current_user = Depends(get_current_user)):
    """获取回滚队列"""
    return {"queue": rollback_manager.get_rollback_queue()}

@router.get("/rollback/history")
async def get_rollback_history(limit: int = 10, current_user = Depends(get_current_user)):
    """获取回滚历史"""
    return {"history": rollback_manager.get_history(limit)}

@router.get("/quality/gates")
async def list_quality_gates(current_user = Depends(get_current_user)):
    """列出质量门禁"""
    return {"gates": [{"gate_id": g.gate_id, "name": g.name, "threshold": g.threshold, "enabled": g.enabled} for g in quality_gates.gates.values()]}

@router.get("/quality/check/{gate_id}")
async def check_quality_gate(gate_id: str, current_user = Depends(get_current_user)):
    """检查质量门禁"""
    result = await quality_gates.check(gate_id)
    return {"gate_id": result.gate_id, "status": result.status.value, "value": result.value}

@router.get("/quality/check-all")
async def check_all_quality_gates(current_user = Depends(get_current_user)):
    """检查所有质量门禁"""
    results = await quality_gates.check_all()
    summary = quality_gates.get_summary(results)
    return summary
