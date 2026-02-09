"""
Cloud Integration API端点 v2.4
"""
from fastapi import APIRouter, HTTPException
from typing import List, Optional, Dict, Any

# 直接导入模块
import importlib.util
import os

backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
module_path = os.path.join(backend_dir, 'cloud/engine.py')

spec = importlib.util.spec_from_file_location("cloud_module", module_path)
module = importlib.util.module_from_spec(spec)

try:
    spec.loader.exec_module(module)
    cloud_engine = module.cloud_integration_engine
    CloudProvider = module.CloudProvider
except Exception as e:
    print(f"Failed to import cloud module: {e}")
    cloud_engine = None
    CloudProvider = None

router = APIRouter()

from pydantic import BaseModel

class RegisterCredentialModel(BaseModel):
    provider: str
    name: str
    config: Optional[Dict] = None

class SyncToLocalModel(BaseModel):
    bucket: str
    key: str
    local_path: str

class PushImageModel(BaseModel):
    image_name: str
    tag: str = "latest"

# ==================== 凭证管理 ====================

@router.get("/credentials")
async def list_credentials(provider: Optional[str] = None):
    """列出凭证"""
    cprovider = CloudProvider(provider) if provider else None
    credentials = cloud_engine.list_credentials(provider=cprovider)
    
    return {
        "total": len(credentials),
        "credentials": [
            {
                "credential_id": c.credential_id,
                "provider": c.provider.value,
                "name": c.name,
                "created_at": c.created_at.isoformat()
            }
            for c in credentials
        ]
    }

@router.post("/credentials")
async def register_credential(request: RegisterCredentialModel):
    """注册凭证"""
    try:
        provider = CloudProvider(request.provider)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid provider: {request.provider}")
    
    credential = cloud_engine.register_credential(
        provider=provider,
        name=request.name,
        config=request.config or {}
    )
    
    return {
        "credential_id": credential.credential_id,
        "provider": credential.provider.value,
        "name": credential.name,
        "message": "Credential registered"
    }

@router.get("/credentials/{credential_id}")
async def get_credential(credential_id: str):
    """获取凭证"""
    credential = cloud_engine.get_credential(credential_id)
    if not credential:
        raise HTTPException(status_code=404, detail="Credential not found")
    
    return {
        "credential_id": credential.credential_id,
        "provider": credential.provider.value,
        "name": credential.name,
        "config_keys": list(credential.config.keys())
    }

@router.post("/credentials/{credential_id}/validate")
async def validate_credential(credential_id: str):
    """验证凭证"""
    try:
        result = cloud_engine.validate_credential(credential_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.delete("/credentials/{credential_id}")
async def delete_credential(credential_id: str):
    """删除凭证"""
    result = cloud_engine.delete_credential(credential_id)
    if not result:
        raise HTTPException(status_code=404, detail="Credential not found")
    return {"message": "Credential deleted"}

# ==================== AWS集成 ====================

@router.get("/aws/s3/buckets")
async def list_s3_buckets(credential_id: str):
    """列出S3桶"""
    try:
        buckets = cloud_engine.list_s3_buckets(credential_id)
        return {"buckets": buckets}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/aws/s3/sync")
async def sync_s3_to_local(credential_id: str, request: SyncToLocalModel):
    """同步S3到本地"""
    try:
        job = cloud_engine.sync_s3_to_local(
            credential_id=credential_id,
            bucket=request.bucket,
            key=request.key,
            local_path=request.local_path
        )
        return {
            "job_id": job.job_id,
            "status": job.status,
            "message": "Sync job created"
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/aws/s3/upload")
async def upload_to_s3(
    credential_id: str,
    local_path: str,
    bucket: str,
    key: str
):
    """上传到S3"""
    try:
        result = cloud_engine.upload_to_s3(
            credential_id=credential_id,
            local_path=local_path,
            bucket=bucket,
            key=key
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

# ==================== GCP集成 ====================

@router.get("/gcs/buckets")
async def list_gcs_buckets(credential_id: str):
    """列出GCS桶"""
    try:
        buckets = cloud_engine.list_gcs_buckets(credential_id)
        return {"buckets": buckets}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/gcs/sync")
async def sync_gcs_to_local(
    credential_id: str,
    bucket: str,
    key: str,
    local_path: str
):
    """同步GCS到本地"""
    try:
        job = cloud_engine.sync_gcs_to_local(
            credential_id=credential_id,
            bucket=bucket,
            key=key,
            local_path=local_path
        )
        return {
            "job_id": job.job_id,
            "status": job.status
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

# ==================== Azure集成 ====================

@router.get("/azure/containers")
async def list_azure_blobs(credential_id: str, container: str):
    """列出Azure Blob"""
    try:
        blobs = cloud_engine.list_azure_blobs(credential_id, container)
        return {"blobs": blobs}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

# ==================== 容器仓库 ====================

@router.get("/registries")
async def list_registries(provider: Optional[str] = None):
    """列出容器仓库"""
    cprovider = CloudProvider(provider) if provider else None
    registries = cloud_engine.list_container_registries(provider=cprovider)
    
    return {
        "total": len(registries),
        "registries": [
            {
                "registry_id": r.registry_id,
                "provider": r.provider.value,
                "name": r.name,
                "url": r.url
            }
            for r in registries
        ]
    }

@router.post("/registries")
async def register_registry(
    provider: str,
    name: str,
    url: str
):
    """注册容器仓库"""
    try:
        cprovider = CloudProvider(provider)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid provider: {provider}")
    
    registry = cloud_engine.register_container_registry(
        provider=cprovider,
        name=name,
        url=url
    )
    
    return {
        "registry_id": registry.registry_id,
        "name": registry.name,
        "message": "Registry registered"
    }

@router.post("/registries/{registry_id}/push")
async def push_image(registry_id: str, request: PushImageModel):
    """推送镜像"""
    try:
        result = cloud_engine.push_container_image(
            registry_id=registry_id,
            image_name=request.image_name,
            tag=request.tag
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/registries/{registry_id}/pull")
async def pull_image(registry_id: str, request: PushImageModel):
    """拉取镜像"""
    try:
        result = cloud_engine.pull_container_image(
            registry_id=registry_id,
            image_name=request.image_name,
            tag=request.tag
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

# ==================== 同步任务 ====================

@router.get("/sync-jobs")
async def list_sync_jobs(status: Optional[str] = None):
    """列出同步任务"""
    jobs = cloud_engine.list_sync_jobs(status=status)
    
    return {
        "total": len(jobs),
        "jobs": [
            {
                "job_id": j.job_id,
                "source": j.source_id,
                "destination": j.destination_id,
                "status": j.status,
                "progress": j.progress
            }
            for j in jobs
        ]
    }

@router.get("/sync-jobs/{job_id}")
async def get_sync_job(job_id: str):
    """获取同步任务"""
    job = cloud_engine.get_sync_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return {
        "job_id": job.job_id,
        "source_type": job.source_type.value,
        "source_id": job.source_id,
        "destination_type": job.destination_type.value,
        "destination_id": job.destination_id,
        "status": job.status,
        "progress": job.progress
    }

# ==================== 统计信息 ====================

@router.get("/summary")
async def get_summary():
    """获取统计"""
    return cloud_engine.get_summary()

@router.get("/health")
async def cloud_health():
    """健康检查"""
    return {
        "status": "healthy",
        "credentials": len(cloud_engine.credentials),
        "registries": len(cloud_engine.container_registries)
    }
