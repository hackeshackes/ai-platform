"""
Cloud Provider Base Client - Multi-Cloud Support

多云提供商基础客户端
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import boto3
from botocore.exceptions import ClientError


class CloudProvider(Enum):
    """云提供商枚举"""
    AWS = "aws"
    GCP = "gcp"
    AZURE = "azure"
    ALIYUN = "aliyun"


@dataclass
class CloudCredential:
    """云凭证"""
    provider: CloudProvider
    access_key_id: str
    secret_access_key: str
    region: str = "us-east-1"
    project_id: Optional[str] = None  # For GCP
    tenant_id: Optional[str] = None  # For Azure
    subscription_id: Optional[str] = None  # For Azure


@dataclass
class CloudResource:
    """云资源"""
    resource_id: str
    resource_type: str
    name: str
    status: str
    region: str
    provider: CloudProvider
    specifications: Dict[str, Any]
    cost_per_hour: float
    created_at: str


class CloudClientBase(ABC):
    """云客户端基类"""
    
    PROVIDER: CloudProvider
    
    def __init__(self, credential: CloudCredential):
        """
        初始化云客户端
        
        Args:
            credential: 云凭证
        """
        self.credential = credential
    
    @abstractmethod
    def validate_credentials(self) -> bool:
        """验证凭证是否有效"""
        pass
    
    @abstractmethod
    def list_instances(self) -> List[CloudResource]:
        """列出所有实例"""
        pass
    
    @abstractmethod
    def create_instance(self, config: Dict[str, Any]) -> CloudResource:
        """创建实例"""
        pass
    
    @abstractmethod
    def delete_instance(self, instance_id: str) -> bool:
        """删除实例"""
        pass
    
    @abstractmethod
    def get_instance_status(self, instance_id: str) -> str:
        """获取实例状态"""
        pass
    
    @abstractmethod
    def get_cost_estimate(self, instance_type: str, hours: int) -> float:
        """获取成本估算"""
        pass
    
    @abstractmethod
    def list_regions(self) -> List[str]:
        """列出可用区域"""
        pass
    
    def get_provider_name(self) -> str:
        """获取提供商名称"""
        return self.PROVIDER.value


def create_cloud_client(credential: CloudCredential) -> CloudClientBase:
    """
    创建云客户端
    
    Args:
        credential: 云凭证
        
    Returns:
        云客户端实例
    """
    if credential.provider == CloudProvider.AWS:
        from cloud.aws.client import AWSClient
        return AWSClient(credential)
    elif credential.provider == CloudProvider.GCP:
        from cloud.gcp.client import GCPClient
        return GCPClient(credential)
    elif credential.provider == CloudProvider.AZURE:
        from cloud.azure.client import AzureClient
        return AzureClient(credential)
    elif credential.provider == CloudProvider.ALIYUN:
        from cloud.aliyun.client import AliyunClient
        return AliyunClient(credential)
    else:
        raise ValueError(f"Unsupported cloud provider: {credential.provider}")
