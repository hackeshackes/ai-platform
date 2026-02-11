"""
Multi-Cloud API Endpoints - AI Platform v4

多云部署API端点
"""
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
import uuid
import json

from cloud.base import (
    CloudProvider,
    CloudCredential,
    CloudResource,
    create_cloud_client
)

router = APIRouter()

# 存储凭证的内存数据库
_credentials_db: Dict[str, CloudCredential] = {}
_credentials_mapping: Dict[str, str] = {}  # credential_id -> provider


# ============ 请求/响应模型 ============

class CredentialCreateRequest(BaseModel):
    """创建凭证请求"""
    provider: str = Field(..., description="云提供商: aws, gcp, azure, aliyun")
    access_key_id: str = Field(..., description="访问密钥ID")
    secret_access_key: str = Field(..., description="密钥")
    region: str = Field(default="us-east-1", description="区域")
    project_id: Optional[str] = Field(None, description="GCP项目ID")
    tenant_id: Optional[str] = Field(None, description="Azure租户ID")
    subscription_id: Optional[str] = Field(None, description="Azure订阅ID")
    name: Optional[str] = Field(None, description="凭证名称")
    description: Optional[str] = Field(None, description="描述")


class CredentialResponse(BaseModel):
    """凭证响应"""
    credential_id: str
    provider: str
    name: Optional[str]
    region: str
    created_at: datetime
    validated: bool
    message: str


class ProviderInfo(BaseModel):
    """云提供商信息"""
    id: str
    name: str
    description: str
    regions: List[str]
    instance_types: List[Dict[str, str]]
    features: List[str]


class DeploymentRequest(BaseModel):
    """部署请求"""
    name: str = Field(..., min_length=1, max_length=100)
    provider: str = Field(..., description="目标云提供商")
    credential_id: str = Field(..., description="凭证ID")
    config: Dict[str, Any] = Field(..., description="部署配置")
    regions: List[str] = Field(default=["us-east-1"], description="部署区域")
    replicas: int = Field(default=1, ge=1, le=10, description="副本数量")


class DeploymentResponse(BaseModel):
    """部署响应"""
    deployment_id: str
    name: str
    provider: str
    status: str
    instances: List[Dict[str, Any]]
    total_cost_per_hour: float
    created_at: datetime
    completed_at: Optional[datetime]


class CostAnalysisRequest(BaseModel):
    """成本分析请求"""
    provider: Optional[str] = Field(None, description="云提供商")
    credential_id: Optional[str] = Field(None, description="凭证ID")
    instance_type: Optional[str] = Field(None, description="实例类型")
    hours: int = Field(default=24, description="小时数")
    region: Optional[str] = Field(None, description="区域")


class CostAnalysisResponse(BaseModel):
    """成本分析响应"""
    total_cost: float
    breakdown: List[Dict[str, Any]]
    recommendations: List[str]
    comparison: Optional[Dict[str, Dict[str, float]]]


# ============ 凭证管理端点 ============

@router.post("/cloud/credentials", response_model=CredentialResponse, tags=["Cloud Credentials"])
async def create_credential(request: CredentialCreateRequest):
    """
    创建云凭证
    
    添加云服务商的访问凭证，用于后续的部署和管理操作。
    
    - **provider**: 云提供商 (aws/gcp/azure/aliyun)
    - **access_key_id**: 访问密钥ID
    - **secret_access_key**: 密钥
    - **region**: 默认区域
    - **project_id**: GCP项目ID（仅GCP需要）
    - **tenant_id**: Azure租户ID（仅Azure需要）
    - **subscription_id**: Azure订阅ID（仅Azure需要）
    """
    # 验证提供商
    try:
        provider_enum = CloudProvider(request.provider.lower())
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported provider: {request.provider}. Supported: aws, gcp, azure, aliyun"
        )
    
    # 创建凭证
    credential = CloudCredential(
        provider=provider_enum,
        access_key_id=request.access_key_id,
        secret_access_key=request.secret_access_key,
        region=request.region,
        project_id=request.project_id,
        tenant_id=request.tenant_id,
        subscription_id=request.subscription_id
    )
    
    # 验证凭证
    try:
        client = create_cloud_client(credential)
        is_valid = client.validate_credentials()
    except Exception as e:
        is_valid = False
        validation_message = str(e)
    else:
        validation_message = "Credentials validated successfully" if is_valid else "Validation failed"
    
    # 生成凭证ID
    credential_id = str(uuid.uuid4())
    
    # 保存凭证
    _credentials_db[credential_id] = credential
    _credentials_mapping[credential_id] = request.provider.lower()
    
    return CredentialResponse(
        credential_id=credential_id,
        provider=request.provider.lower(),
        name=request.name,
        region=request.region,
        created_at=datetime.utcnow(),
        validated=is_valid,
        message=validation_message
    )


@router.get("/cloud/credentials", response_model=List[CredentialResponse], tags=["Cloud Credentials"])
async def list_credentials():
    """
    列出所有云凭证
    
    返回已添加的所有云凭证信息（不包含敏感信息）。
    """
    credentials = []
    for credential_id, credential in _credentials_db.items():
        provider = _credentials_mapping.get(credential_id, 'unknown')
        credentials.append(CredentialResponse(
            credential_id=credential_id,
            provider=provider,
            name=None,
            region=credential.region,
            created_at=datetime.utcnow(),
            validated=True,
            message="Stored credential"
        ))
    return credentials


@router.delete("/cloud/credentials/{credential_id}", tags=["Cloud Credentials"])
async def delete_credential(credential_id: str):
    """
    删除云凭证
    
    删除指定的云凭证。
    """
    if credential_id not in _credentials_db:
        raise HTTPException(status_code=404, detail="Credential not found")
    
    del _credentials_db[credential_id]
    if credential_id in _credentials_mapping:
        del _credentials_mapping[credential_id]
    
    return {"message": "Credential deleted successfully"}


# ============ 云提供商端点 ============

@router.get("/cloud/providers", response_model=List[ProviderInfo], tags=["Cloud Providers"])
async def list_providers():
    """
    列出支持的云提供商
    
    返回所有支持的云服务商及其信息。
    """
    providers = [
        ProviderInfo(
            id="aws",
            name="Amazon Web Services",
            description="AWS - 全球领先的云服务平台",
            regions=[
                "us-east-1", "us-east-2", "us-west-1", "us-west-2",
                "eu-west-1", "eu-west-2", "eu-central-1",
                "ap-northeast-1", "ap-northeast-2", "ap-southeast-1", "ap-southeast-2",
                "cn-north-1", "cn-south-1"
            ],
            instance_types=[
                {"id": "t3.micro", "name": "t3.micro - 2 vCPU, 1 GiB RAM"},
                {"id": "t3.medium", "name": "t3.medium - 2 vCPU, 4 GiB RAM"},
                {"id": "m5.large", "name": "m5.large - 2 vCPU, 8 GiB RAM"},
                {"id": "m5.xlarge", "name": "m5.xlarge - 4 vCPU, 16 GiB RAM"},
                {"id": "c5.large", "name": "c5.large - 2 vCPU, 4 GiB RAM"},
                {"id": "p3.2xlarge", "name": "p3.2xlarge - 8 vCPU, 61 GiB RAM (GPU)"},
            ],
            features=["EC2", "S3", "RDS", "Lambda", "EKS", "SageMaker"]
        ),
        ProviderInfo(
            id="gcp",
            name="Google Cloud Platform",
            description="GCP - Google云服务平台",
            regions=[
                "us-central1", "us-east1", "us-west1", "us-west2",
                "europe-west1", "europe-west2", "europe-west3",
                "asia-east1", "asia-northeast1", "asia-south1",
                "australia-southeast1"
            ],
            instance_types=[
                {"id": "e2-micro", "name": "e2-micro - 2 vCPU, 1 GiB RAM"},
                {"id": "e2-medium", "name": "e2-medium - 2 vCPU, 4 GiB RAM"},
                {"id": "n1-standard-1", "name": "n1-standard-1 - 1 vCPU, 3.75 GiB RAM"},
                {"id": "n1-standard-2", "name": "n1-standard-2 - 2 vCPU, 7.5 GiB RAM"},
                {"id": "n1-highmem-2", "name": "n1-highmem-2 - 2 vCPU, 13 GiB RAM"},
                {"id": "a2-highgpu-1g", "name": "a2-highgpu-1g - 12 vCPU, 140 GiB RAM + 1 GPU"},
            ],
            features=["Compute Engine", "Cloud Storage", "BigQuery", "Cloud Run", "GKE", "Vertex AI"]
        ),
        ProviderInfo(
            id="azure",
            name="Microsoft Azure",
            description="Azure - 微软云服务平台",
            regions=[
                "eastus", "westus", "centralus", "southeastasia",
                "northeurope", "westeurope", "japaneast", "australiaeast"
            ],
            instance_types=[
                {"id": "Standard_B1s", "name": "Standard_B1s - 1 vCPU, 1 GiB RAM"},
                {"id": "Standard_B2s", "name": "Standard_B2s - 2 vCPU, 4 GiB RAM"},
                {"id": "Standard_D2s_v3", "name": "Standard_D2s_v3 - 2 vCPU, 8 GiB RAM"},
                {"id": "Standard_D4s_v3", "name": "Standard_D4s_v3 - 4 vCPU, 16 GiB RAM"},
                {"id": "Standard_F2s_v2", "name": "Standard_F2s_v2 - 2 vCPU, 4 GiB RAM"},
                {"id": "Standard_NC4as_T4_v3", "name": "Standard_NC4as_T4_v3 - 4 vCPU, 28 GiB RAM + 1 GPU"},
            ],
            features=["Virtual Machines", "Blob Storage", "Azure SQL", "Azure Functions", "AKS", "Azure ML"]
        ),
        ProviderInfo(
            id="aliyun",
            name="Alibaba Cloud (阿里云)",
            description="阿里云 - 中国领先的云服务平台",
            regions=[
                "cn-hangzhou", "cn-shanghai", "cn-beijing", "cn-shenzhen",
                "cn-hongkong", "ap-southeast-1", "ap-southeast-2",
                "us-west-1", "us-east-1", "eu-central-1"
            ],
            instance_types=[
                {"id": "ecs.t5-lc2m1.small", "name": "ecs.t5-lc2m1.small - 1 vCPU, 1 GiB RAM"},
                {"id": "ecs.c5.large", "name": "ecs.c5.large - 2 vCPU, 4 GiB RAM"},
                {"id": "ecs.g5.large", "name": "ecs.g5.large - 2 vCPU, 8 GiB RAM"},
                {"id": "ecs.r5.large", "name": "ecs.r5.large - 2 vCPU, 16 GiB RAM"},
                {"id": "ecs.i3.2xlarge", "name": "ecs.i3.2xlarge - 8 vCPU, 32 GiB RAM"},
                {"id": "ecs.gn6v-c8g1.2xlarge", "name": "ecs.gn6v-c8g1.2xlarge - 8 vCPU, 32 GiB RAM + 1 GPU"},
            ],
            features=["ECS", "OSS", "RDS", "Function Compute", "ACK", "PAI"]
        )
    ]
    
    return providers


@router.get("/cloud/providers/{provider}/info", tags=["Cloud Providers"])
async def get_provider_info(provider: str):
    """
    获取特定云提供商详情
    """
    providers = await list_providers()
    for p in providers:
        if p.id == provider.lower():
            return p
    raise HTTPException(status_code=404, detail=f"Provider {provider} not found")


# ============ 跨云部署端点 ============

@router.post("/cloud/deploy", response_model=DeploymentResponse, tags=["Deployment"])
async def deploy_across_clouds(request: DeploymentRequest):
    """
    跨云部署
    
    使用指定的云凭证在目标云平台部署资源。
    
    - **name**: 部署名称
    - **provider**: 目标云提供商
    - **credential_id**: 凭证ID
    - **config**: 部署配置（实例类型、镜像等）
    - **regions**: 部署区域列表
    - **replicas**: 副本数量
    """
    # 验证凭证
    if request.credential_id not in _credentials_db:
        raise HTTPException(status_code=404, detail="Credential not found")
    
    credential = _credentials_db[request.credential_id]
    provider = _credentials_mapping.get(request.credential_id)
    
    if provider != request.provider.lower():
        raise HTTPException(
            status_code=400,
            detail=f"Credential provider ({provider}) does not match deployment provider ({request.provider})"
        )
    
    # 创建云客户端
    try:
        client = create_cloud_client(credential)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to create client: {str(e)}")
    
    # 验证凭证
    if not client.validate_credentials():
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # 创建实例
    instances = []
    total_cost = 0.0
    
    for region in request.regions:
        for i in range(request.replicas):
            instance_name = f"{request.name}-{region}-{i}"
            config = {
                **request.config,
                'name': instance_name
            }
            
            try:
                instance = client.create_instance(config)
                instances.append({
                    'instance_id': instance.resource_id,
                    'name': instance.name,
                    'status': instance.status,
                    'region': region,
                    'cost_per_hour': instance.cost_per_hour
                })
                total_cost += instance.cost_per_hour
            except Exception as e:
                instances.append({
                    'name': instance_name,
                    'status': 'failed',
                    'error': str(e)
                })
    
    deployment_id = str(uuid.uuid4())
    
    return DeploymentResponse(
        deployment_id=deployment_id,
        name=request.name,
        provider=request.provider.lower(),
        status='completed' if all(i.get('status') == 'running' for i in instances) else 'partial',
        instances=instances,
        total_cost_per_hour=total_cost,
        created_at=datetime.utcnow(),
        completed_at=datetime.utcnow()
    )


@router.get("/cloud/deployments/{deployment_id}", tags=["Deployment"])
async def get_deployment_status(deployment_id: str):
    """
    获取部署状态
    """
    # 简化的部署状态查询
    return {
        "deployment_id": deployment_id,
        "status": "running",
        "message": "Deployment is active"
    }


# ============ 成本分析端点 ============

@router.get("/cloud/cost", response_model=CostAnalysisResponse, tags=["Cost Analysis"])
async def analyze_cost(
    provider: Optional[str] = None,
    instance_type: Optional[str] = None,
    hours: int = 24,
    region: Optional[str] = None
):
    """
    成本分析
    
    分析不同云提供商的资源成本。
    
    - **provider**: 特定云提供商
    - **instance_type**: 实例类型
    - **hours**: 分析时长（小时）
    - **region**: 区域
    """
    breakdown = []
    recommendations = []
    
    # 各提供商的默认实例类型和价格
    providers_info = {
        'aws': {
            'default_instance': 'm5.large',
            'price_per_hour': 0.115,
            'regions': {'us-east-1': 0.115, 'eu-west-1': 0.124, 'ap-northeast-1': 0.131}
        },
        'gcp': {
            'default_instance': 'n1-standard-1',
            'price_per_hour': 0.0475,
            'regions': {'us-central1': 0.0475, 'europe-west1': 0.0515, 'asia-northeast1': 0.0555}
        },
        'azure': {
            'default_instance': 'Standard_D2s_v3',
            'price_per_hour': 0.096,
            'regions': {'eastus': 0.096, 'westeurope': 0.108, 'southeastasia': 0.108}
        },
        'aliyun': {
            'default_instance': 'ecs.g5.large',
            'price_per_hour': 0.84,
            'regions': {'cn-hangzhou': 0.84, 'cn-beijing': 0.84, 'ap-southeast-1': 0.92}
        }
    }
    
    # 计算指定提供商的成本
    if provider:
        if provider.lower() not in providers_info:
            raise HTTPException(status_code=400, detail=f"Unknown provider: {provider}")
        
        info = providers_info[provider.lower()]
        price = info['price_per_hour']
        
        if region and region in info['regions']:
            price = info['regions'][region]
        
        breakdown.append({
            'provider': provider.lower(),
            'instance_type': instance_type or info['default_instance'],
            'price_per_hour': price,
            'total_cost': price * hours
        })
    else:
        # 计算所有提供商的成本
        for p_name, info in providers_info.items():
            p_region = region if region in info['regions'] else list(info['regions'].keys())[0]
            price = info['regions'][p_region]
            
            breakdown.append({
                'provider': p_name,
                'instance_type': instance_type or info['default_instance'],
                'price_per_hour': price,
                'total_cost': price * hours
            })
    
    # 生成成本比较
    if len(breakdown) > 1:
        sorted_by_cost = sorted(breakdown, key=lambda x: x['total_cost'])
        cheapest = sorted_by_cost[0]
        
        comparison = {}
        for item in breakdown:
            comparison[item['provider']] = {
                'price_per_hour': item['price_per_hour'],
                'total_cost': item['total_cost'],
                'vs_cheapest': item['total_cost'] - cheapest['total_cost']
            }
        
        # 生成建议
        recommendations = [
            f"Cheapest option: {cheapest['provider']} at ${cheapest['price_per_hour']}/hour",
            f"Savings potential: ${sorted_by_cost[-1]['total_cost'] - cheapest['total_cost']} for {hours} hours"
        ]
        
        if cheapest['provider'] == 'gcp':
            recommendations.append("GCP offers sustained use discounts for long-running instances")
        elif cheapest['provider'] == 'aws':
            recommendations.append("Consider AWS Spot Instances for up to 90% savings")
        elif cheapest['provider'] == 'azure':
            recommendations.append("Azure Reserved Instances can save up to 72% for 1-3 year commitments")
        elif cheapest['provider'] == 'aliyun':
            recommendations.append("阿里云提供抢占式实例，可节省高达90%成本")
    else:
        comparison = None
        recommendations = [
            f"Estimated cost for {hours} hours: ${breakdown[0]['total_cost']:.2f}",
            f"Monthly estimated cost: ${breakdown[0]['total_cost'] * 30:.2f}"
        ]
    
    total_cost = sum(item['total_cost'] for item in breakdown)
    
    return CostAnalysisResponse(
        total_cost=total_cost,
        breakdown=breakdown,
        recommendations=recommendations,
        comparison=comparison
    )


@router.post("/cloud/cost", response_model=CostAnalysisResponse, tags=["Cost Analysis"])
async def analyze_cost_detailed(request: CostAnalysisRequest):
    """
    详细成本分析（POST版本）
    
    提供更详细的成本分析请求参数。
    """
    return await analyze_cost(
        provider=request.provider,
        instance_type=request.instance_type,
        hours=request.hours,
        region=request.region
    )


# ============ 资源管理端点 ============

@router.get("/cloud/resources", response_model=List[Dict[str, Any]], tags=["Resources"])
async def list_resources(credential_id: str, region: Optional[str] = None):
    """
    列出云资源
    
    列出指定凭证下的所有云资源。
    """
    if credential_id not in _credentials_db:
        raise HTTPException(status_code=404, detail="Credential not found")
    
    credential = _credentials_db[credential_id]
    
    try:
        client = create_cloud_client(credential)
        
        if not client.validate_credentials():
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        instances = client.list_instances()
        
        resources = []
        for instance in instances:
            resources.append({
                'resource_id': instance.resource_id,
                'resource_type': instance.resource_type,
                'name': instance.name,
                'status': instance.status,
                'region': instance.region,
                'cost_per_hour': instance.cost_per_hour,
                'specifications': instance.specifications
            })
        
        return resources
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list resources: {str(e)}")


@router.get("/cloud/resources/{resource_id}", tags=["Resources"])
async def get_resource(credential_id: str, resource_id: str):
    """
    获取资源详情
    """
    if credential_id not in _credentials_db:
        raise HTTPException(status_code=404, detail="Credential not found")
    
    credential = _credentials_db[credential_id]
    
    try:
        client = create_cloud_client(credential)
        instance = client.get_instance(resource_id)
        
        return {
            'resource_id': instance.resource_id,
            'resource_type': instance.resource_type,
            'name': instance.name,
            'status': instance.status,
            'region': instance.region,
            'cost_per_hour': instance.cost_per_hour,
            'specifications': instance.specifications,
            'created_at': instance.created_at
        }
        
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Resource not found: {str(e)}")
