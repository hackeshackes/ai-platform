"""
AWS Cloud Client - Multi-Cloud Support

AWS云客户端实现
"""
from typing import Dict, List, Any, Optional
from datetime import datetime
import boto3
from botocore.exceptions import ClientError, BotoCoreError

from cloud.base import (
    CloudClientBase,
    CloudCredential,
    CloudResource,
    CloudProvider
)


class AWSClient(CloudClientBase):
    """AWS云客户端"""
    
    PROVIDER = CloudProvider.AWS
    
    def __init__(self, credential: CloudCredential):
        """
        初始化AWS客户端
        
        Args:
            credential: AWS凭证
        """
        super().__init__(credential)
        self.ec2 = boto3.client(
            'ec2',
            aws_access_key_id=credential.access_key_id,
            aws_secret_access_key=credential.secret_access_key,
            region_name=credential.region
        )
        self.pricing = boto3.client('pricing', region_name='us-east-1')
        self.resource_groups = boto3.client('resource-groups', region_name=credential.region)
    
    def validate_credentials(self) -> bool:
        """验证AWS凭证"""
        try:
            self.ec2.describe_vpcs()
            return True
        except (ClientError, BotoCoreError):
            return False
    
    def list_instances(self) -> List[CloudResource]:
        """列出所有EC2实例"""
        instances = []
        try:
            response = self.ec2.describe_instances(
                Filters=[
                    {'Name': 'instance-state-name', 'Values': ['running', 'pending', 'stopped', 'stopping']}
                ]
            )
            
            for reservation in response['Reservations']:
                for instance in reservation['Instances']:
                    instance_type = instance.get('InstanceType', 'unknown')
                    cost = self._get_instance_price(instance_type, self.credential.region)
                    
                    instances.append(CloudResource(
                        resource_id=instance['InstanceId'],
                        resource_type='ec2',
                        name=instance.get('Tags', {}).get('Name', instance['InstanceId']),
                        status=instance['State']['Name'],
                        region=self.credential.region,
                        provider=CloudProvider.AWS,
                        specifications={
                            'instance_type': instance_type,
                            'vpc_id': instance.get('VpcId'),
                            'subnet_id': instance.get('SubnetId'),
                            'key_name': instance.get('KeyName'),
                            'security_groups': [sg['GroupId'] for sg in instance.get('SecurityGroups', [])],
                            'public_ip': instance.get('PublicIpAddress'),
                            'private_ip': instance.get('PrivateIpAddress'),
                        },
                        cost_per_hour=cost,
                        created_at=str(instance['LaunchTime'])
                    ))
        except ClientError as e:
            print(f"Error listing instances: {e}")
        
        return instances
    
    def create_instance(self, config: Dict[str, Any]) -> CloudResource:
        """
        创建EC2实例
        
        Args:
            config: 实例配置
                - image_id: AMI ID
                - instance_type: 实例类型
                - key_name: SSH密钥对名称
                - security_group_ids: 安全组ID列表
                - subnet_id: 子网ID
                - user_data: 启动脚本
                - count: 实例数量
        """
        try:
            launch_config = {
                'ImageId': config['image_id'],
                'InstanceType': config['instance_type'],
                'KeyName': config['key_name'],
                'MaxCount': config.get('count', 1),
                'MinCount': 1,
                'SecurityGroupIds': config.get('security_group_ids', []),
                'SubnetId': config.get('subnet_id'),
                'UserData': config.get('user_data'),
                'TagSpecifications': [
                    {
                        'ResourceType': 'instance',
                        'Tags': [
                            {'Key': 'Name', 'Value': config.get('name', 'aws-instance')}
                        ]
                    }
                ]
            }
            
            response = self.ec2.run_instances(**launch_config)
            instance_id = response['Instances'][0]['InstanceId']
            
            # Wait for instance to be running
            waiter = self.ec2.get_waiter('instance_running')
            waiter.wait(InstanceIds=[instance_id])
            
            return self.get_instance(instance_id)
            
        except ClientError as e:
            raise Exception(f"Failed to create instance: {e}")
    
    def delete_instance(self, instance_id: str) -> bool:
        """删除EC2实例"""
        try:
            self.ec2.terminate_instances(InstanceIds=[instance_id])
            waiter = self.ec2.get_waiter('instance_terminated')
            waiter.wait(InstanceIds=[instance_id])
            return True
        except ClientError:
            return False
    
    def get_instance_status(self, instance_id: str) -> str:
        """获取实例状态"""
        try:
            response = self.ec2.describe_instances(InstanceIds=[instance_id])
            return response['Reservations'][0]['Instances'][0]['State']['Name']
        except ClientError:
            return 'unknown'
    
    def get_instance(self, instance_id: str) -> CloudResource:
        """获取单个实例详情"""
        try:
            response = self.ec2.describe_instances(InstanceIds=[instance_id])
            instance = response['Reservations'][0]['Instances'][0]
            instance_type = instance.get('InstanceType', 'unknown')
            cost = self._get_instance_price(instance_type, self.credential.region)
            
            return CloudResource(
                resource_id=instance['InstanceId'],
                resource_type='ec2',
                name=instance.get('Tags', {}).get('Name', instance['InstanceId']),
                status=instance['State']['Name'],
                region=self.credential.region,
                provider=CloudProvider.AWS,
                specifications={
                    'instance_type': instance_type,
                    'vpc_id': instance.get('VpcId'),
                    'subnet_id': instance.get('SubnetId'),
                    'key_name': instance.get('KeyName'),
                    'public_ip': instance.get('PublicIpAddress'),
                    'private_ip': instance.get('PrivateIpAddress'),
                },
                cost_per_hour=cost,
                created_at=str(instance['LaunchTime'])
            )
        except ClientError as e:
            raise Exception(f"Failed to get instance: {e}")
    
    def get_cost_estimate(self, instance_type: str, hours: int) -> float:
        """获取EC2成本估算"""
        price = self._get_instance_price(instance_type, self.credential.region)
        return price * hours
    
    def _get_instance_price(self, instance_type: str, region: str) -> float:
        """获取实例价格（每小时）"""
        try:
            response = self.pricing.get_products(
                ServiceCode='AmazonEC2',
                Filters=[
                    {'Type': 'TERM_MATCH', 'Field': 'instanceType', 'Value': instance_type},
                    {'Type': 'TERM_MATCH', 'Field': 'location', 'Value': self._get_region_name(region)},
                    {'Type': 'TERM_MATCH', 'Field': 'tenancy', 'Value': 'Shared'},
                    {'Type': 'TERM_MATCH', 'Field': 'operatingSystem', 'Value': 'Linux'},
                    {'Type': 'TERM_MATCH', 'Field': 'preInstalledSw', 'Value': 'NA'},
                    {'Type': 'TERM_MATCH', 'Field': 'capacitystatus', 'Value': 'Used'},
                ],
                MaxResults=1
            )
            
            if response['PriceList']:
                import json
                price_data = json.loads(response['PriceList'][0])
                terms = price_data.get('terms', {}).get('OnDemand', {})
                if terms:
                    first_term = list(terms.values())[0]
                    price_dimensions = first_term.get('priceDimensions', {})
                    first_dimension = list(price_dimensions.values())[0]
                    price_per_unit = float(first_dimension['pricePerUnit']['USD'])
                    return price_per_unit
                    
        except Exception as e:
            print(f"Error getting price: {e}")
        
        # 默认价格（如果无法获取）
        default_prices = {
            't2.micro': 0.0116,
            't2.small': 0.0232,
            't2.medium': 0.0464,
            't3.micro': 0.0104,
            't3.small': 0.0208,
            't3.medium': 0.0416,
            'm5.large': 0.115,
            'm5.xlarge': 0.23,
            'c5.large': 0.085,
            'c5.xlarge': 0.17,
        }
        return default_prices.get(instance_type, 0.1)
    
    def _get_region_name(self, region_code: str) -> str:
        """获取区域显示名称"""
        region_names = {
            'us-east-1': 'US East (N. Virginia)',
            'us-west-2': 'US West (Oregon)',
            'eu-west-1': 'EU (Ireland)',
            'ap-northeast-1': 'Asia Pacific (Tokyo)',
            'cn-north-1': 'China (Beijing)',
        }
        return region_names.get(region_code, region_code)
    
    def list_regions(self) -> List[str]:
        """列出所有AWS区域"""
        try:
            response = self.ec2.describe_regions()
            return [r['RegionName'] for r in response['Regions']]
        except ClientError:
            return ['us-east-1', 'us-west-2', 'eu-west-1']
    
    def list_instance_types(self) -> List[Dict[str, str]]:
        """列出常用实例类型"""
        return [
            {'id': 't3.micro', 'name': 't3.micro - 2 vCPU, 1 GiB RAM', 'family': 'General Purpose'},
            {'id': 't3.small', 'name': 't3.small - 2 vCPU, 2 GiB RAM', 'family': 'General Purpose'},
            {'id': 't3.medium', 'name': 't3.medium - 2 vCPU, 4 GiB RAM', 'family': 'General Purpose'},
            {'id': 'm5.large', 'name': 'm5.large - 2 vCPU, 8 GiB RAM', 'family': 'General Purpose'},
            {'id': 'm5.xlarge', 'name': 'm5.xlarge - 4 vCPU, 16 GiB RAM', 'family': 'General Purpose'},
            {'id': 'c5.large', 'name': 'c5.large - 2 vCPU, 4 GiB RAM', 'family': 'Compute Optimized'},
            {'id': 'c5.xlarge', 'name': 'c5.xlarge - 4 vCPU, 8 GiB RAM', 'family': 'Compute Optimized'},
            {'id': 'p3.2xlarge', 'name': 'p3.2xlarge - 8 vCPU, 61 GiB RAM', 'family': 'GPU Accelerated'},
            {'id': 'g4dn.xlarge', 'name': 'g4dn.xlarge - 4 vCPU, 16 GiB RAM', 'family': 'GPU Accelerated'},
        ]
    
    def list_amis(self) -> List[Dict[str, str]]:
        """列出常用AMI"""
        return [
            {'id': 'ami-0c55b159cbfafe1f0', 'name': 'Amazon Linux 2', 'description': 'Amazon Linux 2'},
            {'id': 'ami-0ff8a18407b2f80a3', 'name': 'Ubuntu 22.04 LTS', 'description': 'Ubuntu Server 22.04 LTS'},
            {'id': 'ami-0b2d6e7d5c8b944d1', 'name': 'Ubuntu 20.04 LTS', 'description': 'Ubuntu Server 20.04 LTS'},
            {'id': 'ami-0b5e70916d3395216', 'name': 'CentOS 7', 'description': 'CentOS 7'},
            {'id': 'ami-0d9858aa2d0f7080b', 'name': 'Windows Server 2022', 'description': 'Windows Server 2022 Base'},
        ]
