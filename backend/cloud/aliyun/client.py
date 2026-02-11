"""
Aliyun Cloud Client - Multi-Cloud Support

阿里云客户端实现
"""
from typing import Dict, List, Any, Optional
from datetime import datetime
from alibabacloud_ecs20140526.client import Client as EcsClient
from alibabacloud_ecs20140526.models import (
    RunInstancesRequest,
    DescribeInstancesRequest,
    DeleteInstanceRequest,
    StopInstanceRequest,
    StartInstanceRequest
)
from alibabacloud_tea_openapi.models import Config
from alibabacloud_tea_util.models import RuntimeOptions


from cloud.base import (
    CloudClientBase,
    CloudCredential,
    CloudResource,
    CloudProvider
)


class AliyunClient(CloudClientBase):
    """阿里云客户端"""
    
    PROVIDER = CloudProvider.ALIYUN
    
    def __init__(self, credential: CloudCredential):
        """
        初始化阿里云客户端
        
        Args:
            credential: 阿里云凭证
        """
        super().__init__(credential)
        
        # 创建阿里云配置
        config = Config(
            access_key_id=credential.access_key_id,
            access_key_secret=credential.secret_access_key,
            endpoint=f'ecs.{credential.region}.aliyuncs.com'
        )
        
        # 初始化ECS客户端
        self.ecs_client = EcsClient(config)
        self.region_id = credential.region
    
    def validate_credentials(self) -> bool:
        """验证阿里云凭证"""
        try:
            request = DescribeInstancesRequest()
            request.accept_format='json'
            self.ecs_client.describe_instances(request)
            return True
        except Exception:
            return False
    
    def list_instances(self) -> List[CloudResource]:
        """列出所有ECS实例"""
        instances = []
        try:
            request = DescribeInstancesRequest(region_id=self.region_id)
            request.accept_format='json'
            
            response = self.ecs_client.describe_instances(request)
            
            if response.body and response.body.instances:
                for instance in response.body.instances.instance:
                    instance_type = instance.instance_type_id
                    cost = self._get_instance_price(instance_type)
                    
                    # 获取状态映射
                    status = self._map_status(instance.status)
                    
                    instances.append(CloudResource(
                        resource_id=instance.instance_id,
                        resource_type='ecs',
                        name=instance.instance_name,
                        status=status,
                        region=self.region_id,
                        provider=CloudProvider.ALIYUN,
                        specifications={
                            'instance_type': instance_type,
                            'zone_id': instance.zone_id,
                            'vpc_id': instance.vpc_attributes.vpc_id,
                            'private_ip': instance.vpc_attributes.private_ip_address,
                            'public_ip': instance.public_ip_address.ip_address[0] if instance.public_ip_address.ip_address else None,
                            'cpu': instance.cpu,
                            'memory': instance.memory,
                            'instance_charge_type': instance.instance_charge_type,
                        },
                        cost_per_hour=cost,
                        created_at=str(instance.creation_time)
                    ))
        except Exception as e:
            print(f"Error listing instances: {e}")
        
        return instances
    
    def create_instance(self, config: Dict[str, Any]) -> CloudResource:
        """
        创建ECS实例
        
        Args:
            config: 实例配置
                - instance_type: 实例规格
                - image_id: 镜像ID
                - instance_name: 实例名称
                - security_group_id: 安全组ID
                - v_switch_id: 交换机ID
                - internet_max_bandwidth_out: 公网带宽
        """
        try:
            request = RunInstancesRequest(
                region_id=self.region_id,
                instance_type=config['instance_type'],
                image_id=config['image_id'],
                instance_name=config.get('instance_name', 'aliyun-instance'),
                security_group_id=config['security_group_id'],
                vswitch_id=config.get('v_switch_id'),
                internet_max_bandwidth_out=config.get('internet_max_bandwidth_out', 5),
                internet_max_bandwidth_in=config.get('internet_max_bandwidth_in', 100),
                password=config.get('password'),
                amount=config.get('count', 1),
            )
            
            response = self.ecs_client.run_instances(request)
            
            if response.body and response.body.instance_id_sets:
                instance_id = response.body.instance_id_sets.instance_id_set[0]
                
                # 等待实例运行
                self._wait_for_instance(instance_id, 'Running')
                
                return self.get_instance(instance_id)
            else:
                raise Exception("Failed to create instance")
                
        except Exception as e:
            raise Exception(f"Failed to create instance: {e}")
    
    def delete_instance(self, instance_id: str) -> bool:
        """删除ECS实例"""
        try:
            request = DeleteInstanceRequest(
                region_id=self.region_id,
                instance_id=instance_id,
                force=True
            )
            
            self.ecs_client.delete_instance(request)
            return True
        except Exception:
            return False
    
    def get_instance_status(self, instance_id: str) -> str:
        """获取实例状态"""
        try:
            request = DescribeInstancesRequest(
                region_id=self.region_id,
                instance_ids=[instance_id],
                accept_format='json'
            )
            
            response = self.ecs_client.describe_instances(request)
            
            if response.body and response.body.instances:
                status = response.body.instances.instance[0].status
                return self._map_status(status)
            return 'unknown'
        except Exception:
            return 'unknown'
    
    def get_instance(self, instance_id: str) -> CloudResource:
        """获取单个实例详情"""
        try:
            request = DescribeInstancesRequest(
                region_id=self.region_id,
                instance_ids=[instance_id],
                accept_format='json'
            )
            
            response = self.ecs_client.describe_instances(request)
            
            if response.body and response.body.instances:
                instance = response.body.instances.instance[0]
                instance_type = instance.instance_type_id
                cost = self._get_instance_price(instance_type)
                
                return CloudResource(
                    resource_id=instance.instance_id,
                    resource_type='ecs',
                    name=instance.instance_name,
                    status=self._map_status(instance.status),
                    region=self.region_id,
                    provider=CloudProvider.ALIYUN,
                    specifications={
                        'instance_type': instance_type,
                        'zone_id': instance.zone_id,
                        'cpu': instance.cpu,
                        'memory': instance.memory,
                        'private_ip': instance.vpc_attributes.private_ip_address,
                        'public_ip': instance.public_ip_address.ip_address[0] if instance.public_ip_address.ip_address else None,
                    },
                    cost_per_hour=cost,
                    created_at=str(instance.creation_time)
                )
            else:
                raise Exception(f"Instance {instance_id} not found")
        except Exception as e:
            raise Exception(f"Failed to get instance: {e}")
    
    def get_cost_estimate(self, instance_type: str, hours: int) -> float:
        """获取阿里云ECS成本估算"""
        price = self._get_instance_price(instance_type)
        return price * hours
    
    def _get_instance_price(self, instance_type: str) -> float:
        """获取实例价格（每小时）"""
        # 阿里云ECS定价（简化版）
        prices = {
            'ecs.t5-lc2m1.nano': 0.012,
            'ecs.t5-lc2m1.small': 0.024,
            'ecs.c5.large': 0.63,
            'ecs.c5.xlarge': 1.26,
            'ecs.c5.2xlarge': 2.52,
            'ecs.g5.large': 0.84,
            'ecs.g5.xlarge': 1.68,
            'ecs.g5.2xlarge': 3.36,
            'ecs.r5.large': 1.05,
            'ecs.r5.xlarge': 2.1,
            'ecs.r5.2xlarge': 4.2,
            'ecs.i3.2xlarge': 4.5,
            'ecs.i3.4xlarge': 9.0,
            'ecs.gn6i-c4g1.xlarge': 11.0,
            'ecs.gn6i-c8g1.2xlarge': 18.0,
            'ecs.gn6v-c8g1.2xlarge': 23.0,
        }
        return prices.get(instance_type, 0.5)
    
    def _map_status(self, status: str) -> str:
        """映射阿里云状态到标准状态"""
        status_map = {
            'Pending': 'pending',
            'Running': 'running',
            'Starting': 'starting',
            'Stopping': 'stopping',
            'Stopped': 'stopped',
            'Deleted': 'deleted',
        }
        return status_map.get(status, status.lower())
    
    def _wait_for_instance(self, instance_id: str, target_status: str, timeout: int = 300):
        """等待实例状态变化"""
        import time
        elapsed = 0
        while elapsed < timeout:
            current_status = self.get_instance_status(instance_id)
            if current_status.lower() == target_status.lower():
                return
            time.sleep(5)
            elapsed += 5
        raise Exception(f"Timeout waiting for instance {instance_id} to be {target_status}")
    
    def list_regions(self) -> List[str]:
        """列出所有阿里云区域"""
        return [
            'cn-hangzhou', 'cn-shanghai', 'cn-beijing', 'cn-shenzhen',
            'cn-hongkong', 'ap-southeast-1', 'ap-southeast-2',
            'ap-southeast-3', 'ap-southeast-5', 'ap-northeast-1',
            'us-west-1', 'us-east-1', 'eu-central-1', 'eu-west-1'
        ]
    
    def list_instance_types(self) -> List[Dict[str, str]]:
        """列出实例规格"""
        return [
            {'id': 'ecs.t5-lc2m1.nano', 'name': 'ecs.t5-lc2m1.nano - 1 vCPU, 0.5 GiB RAM', 'family': 'Burstable'},
            {'id': 'ecs.t5-lc2m1.small', 'name': 'ecs.t5-lc2m1.small - 1 vCPU, 1 GiB RAM', 'family': 'Burstable'},
            {'id': 'ecs.c5.large', 'name': 'ecs.c5.large - 2 vCPU, 4 GiB RAM', 'family': 'Compute Optimized'},
            {'id': 'ecs.c5.xlarge', 'name': 'ecs.c5.xlarge - 4 vCPU, 8 GiB RAM', 'family': 'Compute Optimized'},
            {'id': 'ecs.g5.large', 'name': 'ecs.g5.large - 2 vCPU, 8 GiB RAM', 'family': 'General Purpose'},
            {'id': 'ecs.g5.xlarge', 'name': 'ecs.g5.xlarge - 4 vCPU, 16 GiB RAM', 'family': 'General Purpose'},
            {'id': 'ecs.r5.large', 'name': 'ecs.r5.large - 2 vCPU, 16 GiB RAM', 'family': 'Memory Optimized'},
            {'id': 'ecs.r5.xlarge', 'name': 'ecs.r5.xlarge - 4 vCPU, 32 GiB RAM', 'family': 'Memory Optimized'},
            {'id': 'ecs.i3.2xlarge', 'name': 'ecs.i3.2xlarge - 8 vCPU, 32 GiB RAM', 'family': 'I/O Optimized'},
            {'id': 'ecs.gn6v-c8g1.2xlarge', 'name': 'ecs.gn6v-c8g1.2xlarge - 8 vCPU, 32 GiB RAM + 1 GPU', 'family': 'GPU Accelerated'},
        ]
    
    def list_images(self) -> List[Dict[str, str]]:
        """列出常用镜像"""
        return [
            {'image_id': 'm-bp13pXXXXXXXXXXXXXXXXX', 'name': 'Alibaba Cloud Linux 3', 'type': 'linux'},
            {'image_id': 'm-bp1XXXXXXXXXXXXXXXXX', 'name': 'CentOS 7.9', 'type': 'linux'},
            {'image_id': 'm-bp1XXXXXXXXXXXXXXXXX', 'name': 'Ubuntu 22.04 LTS', 'type': 'linux'},
            {'image_id': 'm-bp1XXXXXXXXXXXXXXXXX', 'name': 'Ubuntu 20.04 LTS', 'type': 'linux'},
            {'image_id': 'm-bp1XXXXXXXXXXXXXXXXX', 'name': 'Debian 11', 'type': 'linux'},
            {'image_id': 'm-bp1XXXXXXXXXXXXXXXXX', 'name': 'Windows Server 2022', 'type': 'windows'},
            {'image_id': 'm-bp1XXXXXXXXXXXXXXXXX', 'name': 'Windows Server 2019', 'type': 'windows'},
        ]
