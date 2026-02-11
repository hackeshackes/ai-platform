"""
GCP Cloud Client - Multi-Cloud Support

Google Cloud Platform客户端实现
"""
from typing import Dict, List, Any, Optional
from datetime import datetime
from google.cloud import compute_v1
from google.cloud import resource_manager
from google.oauth2 import service_account
from googleapiclient import discovery
from googleapiclient.errors import HttpError


from cloud.base import (
    CloudClientBase,
    CloudCredential,
    CloudResource,
    CloudProvider
)


class GCPClient(CloudClientBase):
    """GCP云客户端"""
    
    PROVIDER = CloudProvider.GCP
    
    def __init__(self, credential: CloudCredential):
        """
        初始化GCP客户端
        
        Args:
            credential: GCP凭证
        """
        super().__init__(credential)
        
        # 创建服务账号凭据
        self.credentials = service_account.Credentials.from_service_account_info({
            'type': 'service_account',
            'project_id': credential.project_id,
            'private_key_id': credential.access_key_id,  # Using access_key_id as private_key_id
            'private_key': credential.secret_access_key,  # Using secret_access_key as private_key
            'client_email': f'service-account@{credential.project_id}.iam.gserviceaccount.com',
            'client_id': '123456789',
        })
        
        # 初始化客户端
        self.compute = compute_v1.InstancesClient(credentials=self.credentials)
        self.zones_client = compute_v1.ZonesClient(credentials=self.credentials)
        self.project_id = credential.project_id
        self.region = credential.region
    
    def validate_credentials(self) -> bool:
        """验证GCP凭证"""
        try:
            # 尝试列出项目信息
            cloud_resource_client = resource_manager.Client()
            project = cloud_resource_client.fetch_project(self.project_id)
            return project.project_id == self.project_id
        except Exception:
            return False
    
    def list_instances(self, zone: str = "us-central1-a") -> List[CloudResource]:
        """列出GCP Compute Engine实例"""
        instances = []
        try:
            request = compute_v1.AggregatedListInstancesRequest(
                project=self.project_id,
            )
            
            response = self.compute.aggregated_list(request=request)
            
            for zone_response in response.items:
                zone_name = zone_responsezone.split('/')[-1]
                
                for instance in zone_response.instances:
                    instance_type = instance.machine_type.split('/')[-1]
                    cost = self._get_instance_price(instance_type, zone_name)
                    
                    instances.append(CloudResource(
                        resource_id=instance.id,
                        resource_type='compute_engine',
                        name=instance.name,
                        status=instance.status.lower(),
                        region=zone_name,
                        provider=CloudProvider.GCP,
                        specifications={
                            'machine_type': instance_type,
                            'zone': zone_name,
                            'networkInterfaces': len(instance.network_interfaces),
                            'disks': len(instance.disks),
                            'can_ip_forward': instance.can_ip_forward,
                            'metadata': dict(instance.metadata.items()) if instance.metadata else {},
                        },
                        cost_per_hour=cost,
                        created_at=str(instance.creation_timestamp)
                    ))
        except Exception as e:
            print(f"Error listing instances: {e}")
        
        return instances
    
    def create_instance(self, config: Dict[str, Any], zone: str = "us-central1-a") -> CloudResource:
        """
        创建GCP Compute Engine实例
        
        Args:
            config: 实例配置
                - name: 实例名称
                - machine_type: 机器类型
                - source_image: 镜像URL
                - network_interface: 网络配置
                - disk: 启动磁盘配置
            zone: 区域
        """
        try:
            # 配置实例
            instance_config = {
                'name': config['name'],
                'machine_type': f"zones/{zone}/machineTypes/{config['machine_type']}",
                'disks': [
                    compute_v1.AttachedDisk(
                        initialize_params=compute_v1.AttachedDiskInitializeParams(
                            source_image=config.get('source_image', 'projects/debian-cloud/global/images/family/debian-11'),
                            disk_size_gb=config.get('disk_size_gb', 20),
                            disk_type=config.get('disk_type', 'pd-balanced'),
                        ),
                        auto_delete=True,
                        boot=True,
                    )
                ],
                'network_interfaces': [
                    compute_v1.NetworkInterface(
                        network=config.get('network', 'global/networks/default'),
                        access_configs=[
                            compute_v1.AccessConfig(
                                name='External NAT',
                                type='ONE_TO_ONE_NAT',
                            )
                        ]
                    )
                ],
                'service_accounts': [
                    compute_v1.ServiceAccount(
                        email=config.get('service_account_email', 'default'),
                        scopes=['https://www.googleapis.com/auth/cloud-platform']
                    )
                ],
                'metadata': {
                    'items': [
                        {'key': 'startup-script', 'value': config.get('startup_script', '')}
                    ]
                }
            }
            
            request = compute_v1.InsertInstanceRequest(
                project=self.project_id,
                zone=zone,
                instance_resource=compute_v1.Instance(**instance_config),
            )
            
            operation = self.compute.insert(request=request)
            # 等待操作完成
            while operation.status != 'DONE':
                operation = self.compute.wait(operation_ref=operation.self_link)
            
            return self.get_instance(config['name'], zone)
            
        except Exception as e:
            raise Exception(f"Failed to create instance: {e}")
    
    def delete_instance(self, instance_name: str, zone: str = "us-central1-a") -> bool:
        """删除GCP实例"""
        try:
            request = compute_v1.DeleteInstanceRequest(
                project=self.project_id,
                zone=zone,
                instance=instance_name,
            )
            
            operation = self.compute.delete(request=request)
            # 等待操作完成
            while operation.status != 'DONE':
                operation = self.compute.wait(operation_ref=operation.self_link)
            
            return True
        except Exception:
            return False
    
    def get_instance_status(self, instance_name: str, zone: str = "us-central1-a") -> str:
        """获取实例状态"""
        try:
            request = compute_v1.GetInstanceRequest(
                project=self.project_id,
                zone=zone,
                instance=instance_name,
            )
            instance = self.compute.get(request=request)
            return instance.status.lower()
        except Exception:
            return 'unknown'
    
    def get_instance(self, instance_name: str, zone: str = "us-central1-a") -> CloudResource:
        """获取单个实例详情"""
        try:
            request = compute_v1.GetInstanceRequest(
                project=self.project_id,
                zone=zone,
                instance=instance_name,
            )
            instance = self.compute.get(request=request)
            
            machine_type = instance.machine_type.split('/')[-1]
            cost = self._get_instance_price(machine_type, zone)
            
            return CloudResource(
                resource_id=str(instance.id),
                resource_type='compute_engine',
                name=instance.name,
                status=instance.status.lower(),
                region=zone,
                provider=CloudProvider.GCP,
                specifications={
                    'machine_type': machine_type,
                    'zone': zone,
                    'networkInterfaces': len(instance.network_interfaces),
                    'disks': len(instance.disks),
                    'cpu_platform': instance.cpu_platform,
                },
                cost_per_hour=cost,
                created_at=str(instance.creation_timestamp)
            )
        except Exception as e:
            raise Exception(f"Failed to get instance: {e}")
    
    def get_cost_estimate(self, machine_type: str, hours: int, zone: str = "us-central1-a") -> float:
        """获取GCP成本估算"""
        price = self._get_instance_price(machine_type, zone)
        return price * hours
    
    def _get_instance_price(self, machine_type: str, zone: str) -> float:
        """获取实例价格（每小时）"""
        # GCP定价（简化版，实际应使用Cloud Billing API）
        zone_prices = {
            'us-central1': {'f1-micro': 0.0062, 'e2-micro': 0.0096, 'e2-small': 0.0192, 'e2-medium': 0.0384},
            'us-east1': {'n1-standard-1': 0.0475, 'n1-standard-2': 0.095, 'n1-standard-4': 0.19},
            'europe-west1': {'n1-standard-1': 0.0515, 'n1-standard-2': 0.103, 'n1-standard-4': 0.206},
            'asia-northeast1': {'n1-standard-1': 0.0555, 'n1-standard-2': 0.111, 'n1-standard-4': 0.222},
        }
        
        region = '-'.join(zone.split('-')[:2])
        zone_data = zone_prices.get(region, zone_prices['us-central1'])
        
        # 通用类型定价
        default_prices = {
            'f1-micro': 0.0062,
            'e2-micro': 0.0096,
            'e2-small': 0.0192,
            'e2-medium': 0.0384,
            'n1-standard-1': 0.0475,
            'n1-standard-2': 0.095,
            'n1-standard-4': 0.19,
            'n1-standard-8': 0.38,
            'n1-highmem-2': 0.118,
            'n1-highmem-4': 0.236,
            'n1-highmem-8': 0.472,
            'n1-highcpu-2': 0.063,
            'n1-highcpu-4': 0.126,
            'n1-highcpu-8': 0.252,
            'a2-highgpu-1g': 3.67,
            'a2-ultragpu-8g': 29.38,
        }
        
        return zone_data.get(machine_type, default_prices.get(machine_type, 0.1))
    
    def list_regions(self) -> List[str]:
        """列出所有GCP区域"""
        regions = [
            'us-central1', 'us-east1', 'us-east4', 'us-west1', 'us-west2',
            'europe-west1', 'europe-west2', 'europe-west3', 'europe-west4',
            'asia-east1', 'asia-east2', 'asia-northeast1', 'asia-northeast2', 'asia-south1',
            'australia-southeast1'
        ]
        return regions
    
    def list_machine_types(self, zone: str = "us-central1-a") -> List[Dict[str, str]]:
        """列出机器类型"""
        return [
            {'id': 'f1-micro', 'name': 'f1-micro - 1 vCPU, 0.6 GiB RAM', 'family': 'General Purpose'},
            {'id': 'e2-micro', 'name': 'e2-micro - 2 vCPU, 1 GiB RAM', 'family': 'General Purpose'},
            {'id': 'e2-small', 'name': 'e2-small - 2 vCPU, 2 GiB RAM', 'family': 'General Purpose'},
            {'id': 'e2-medium', 'name': 'e2-medium - 2 vCPU, 4 GiB RAM', 'family': 'General Purpose'},
            {'id': 'n1-standard-1', 'name': 'n1-standard-1 - 1 vCPU, 3.75 GiB RAM', 'family': 'General Purpose'},
            {'id': 'n1-standard-2', 'name': 'n1-standard-2 - 2 vCPU, 7.5 GiB RAM', 'family': 'General Purpose'},
            {'id': 'n1-standard-4', 'name': 'n1-standard-4 - 4 vCPU, 15 GiB RAM', 'family': 'General Purpose'},
            {'id': 'n1-highmem-2', 'name': 'n1-highmem-2 - 2 vCPU, 13 GiB RAM', 'family': 'Memory Optimized'},
            {'id': 'n1-highcpu-2', 'name': 'n1-highcpu-2 - 2 vCPU, 1.8 GiB RAM', 'family': 'Compute Optimized'},
            {'id': 'a2-highgpu-1g', 'name': 'a2-highgpu-1g - 12 vCPU, 140 GiB RAM + 1 GPU', 'family': 'GPU Accelerated'},
        ]
    
    def list_images(self) -> List[Dict[str, str]]:
        """列出常用镜像"""
        return [
            {'family': 'debian-11', 'name': 'Debian 11', 'project': 'debian-cloud'},
            {'family': 'ubuntu-2204-lts', 'name': 'Ubuntu 22.04 LTS', 'project': 'ubuntu-os-cloud'},
            {'family': 'ubuntu-2004-lts', 'name': 'Ubuntu 20.04 LTS', 'project': 'ubuntu-os-cloud'},
            {'family': 'centos-7', 'name': 'CentOS 7', 'project': 'centos-cloud'},
            {'family': 'cos-109', 'name': 'Container-Optimized OS', 'project': 'cos-cloud'},
            {'family': 'windows-2022', 'name': 'Windows Server 2022', 'project': 'windows-cloud'},
        ]
