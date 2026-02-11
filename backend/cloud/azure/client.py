"""
Azure Cloud Client - Multi-Cloud Support

Microsoft Azure客户端实现
"""
from typing import Dict, List, Any, Optional
from datetime import datetime
from azure.identity import ClientSecretCredential
from azure.mgmt.compute import ComputeManagementClient
from azure.mgmt.network import NetworkManagementClient
from azure.mgmt.resource import ResourceManagementClient
from azure.mgmt.compute.models import VirtualMachine
from azure.core.exceptions import AzureError


from cloud.base import (
    CloudClientBase,
    CloudCredential,
    CloudResource,
    CloudProvider
)


class AzureClient(CloudClientBase):
    """Azure云客户端"""
    
    PROVIDER = CloudProvider.AZURE
    
    def __init__(self, credential: CloudCredential):
        """
        初始化Azure客户端
        
        Args:
            credential: Azure凭证
        """
        super().__init__(credential)
        
        # 创建凭据
        self.credentials = ClientSecretCredential(
            tenant_id=credential.tenant_id,
            client_id=credential.access_key_id,
            client_secret=credential.secret_access_key
        )
        
        # 初始化客户端
        self.compute_client = ComputeManagementClient(
            self.credentials,
            credential.subscription_id
        )
        self.network_client = NetworkManagementClient(
            self.credentials,
            credential.subscription_id
        )
        self.resource_client = ResourceManagementClient(
            self.credentials,
            credential.subscription_id
        )
        self.subscription_id = credential.subscription_id
        self.resource_group = credential.region  # Using region as resource group name
    
    def validate_credentials(self) -> bool:
        """验证Azure凭证"""
        try:
            # 尝试获取订阅信息
            self.compute_client.subscriptions.list()
            return True
        except AzureError:
            return False
    
    def list_instances(self, resource_group: Optional[str] = None) -> List[CloudResource]:
        """列出所有Azure虚拟机"""
        instances = []
        rg = resource_group or self.resource_group
        
        try:
            # 获取资源组中的所有虚拟机
            vms = self.compute_client.virtual_machines.list(rg)
            
            for vm in vms:
                instance_details = self.compute_client.virtual_machines.get(
                    rg, vm.name, expand='instanceView'
                )
                
                vm_size = instance_details.hardware_profile.vm_size
                cost = self._get_vm_price(vm_size)
                
                # 获取状态
                status = 'unknown'
                if instance_details.instance_view.statuses:
                    status = instance_details.instance_view.statuses[-1].code.lower()
                    if 'running' in status:
                        status = 'running'
                    elif 'stopped' in status:
                        status = 'stopped'
                
                instances.append(CloudResource(
                    resource_id=vm.id,
                    resource_type='virtual_machine',
                    name=vm.name,
                    status=status,
                    region=self._get_location_display_name(vm.location),
                    provider=CloudProvider.AZURE,
                    specifications={
                        'vm_size': vm_size,
                        'resource_group': rg,
                        'location': vm.location,
                        'os_type': instance_details.storage_profile.os_disk.os_type.value,
                        'network_interfaces': len(instance_details.network_profile.network_interfaces),
                        'data_disks': len(instance_details.storage_profile.data_disks),
                    },
                    cost_per_hour=cost,
                    created_at=str(vm.id.split('/')[-1])[:10]  # Simplified date extraction
                ))
        except AzureError as e:
            print(f"Error listing instances: {e}")
        
        return instances
    
    def create_instance(self, config: Dict[str, Any]) -> CloudResource:
        """
        创建Azure虚拟机
        
        Args:
            config: 实例配置
                - name: VM名称
                - vm_size: VM大小
                - admin_username: 管理员用户名
                - admin_password: 管理员密码
                - image_reference: 镜像引用
                - virtual_network: 虚拟网络名称
                - subnet_name: 子网名称
                - resource_group: 资源组
        """
        try:
            rg = config.get('resource_group', self.resource_group)
            location = config.get('location', 'eastus')
            
            # 确保资源组存在
            if not self._resource_group_exists(rg):
                self.resource_client.resource_groups.create_or_update(
                    rg,
                    {'location': location}
                )
            
            # 创建虚拟网络和子网（如果不存在）
            vnet_name = config.get('virtual_network', f"{rg}-vnet")
            subnet_name = config.get('subnet_name', 'default')
            
            self._create_network_components(rg, vnet_name, subnet_name, location)
            
            # 创建网络接口
            nic_name = f"{config['name']}-nic"
            nic = self.network_client.network_interfaces.begin_create_or_update(
                rg,
                nic_name,
                {
                    'location': location,
                    'ip_configurations': [{
                        'name': f"{config['name']}-ipconfig",
                        'subnet': {
                            'id': f"/subscriptions/{self.subscription_id}/resourceGroups/{rg}/providers/Microsoft.Network/virtualNetworks/{vnet_name}/subnets/{subnet_name}"
                        }
                    }]
                }
            ).result()
            
            # 创建VM
            vm_parameters = {
                'location': location,
                'hardware_profile': {
                    'vm_size': config['vm_size']
                },
                'storage_profile': {
                    'image_reference': config.get('image_reference', {
                        'publisher': 'Canonical',
                        'offer': 'UbuntuServer',
                        'sku': '22.04-lts',
                        'version': 'latest'
                    }),
                    'os_disk': {
                        'name': f"{config['name']}-osdisk",
                        'disk_size_gb': config.get('disk_size_gb', 30),
                        'managed_disk': {
                            'storage_account_type': 'Standard_LRS'
                        },
                        'create_option': 'FromImage'
                    }
                },
                'network_profile': {
                    'network_interfaces': [{
                        'id': nic.id
                    }]
                },
                'os_profile': {
                    'computer_name': config['name'],
                    'admin_username': config['admin_username'],
                    'admin_password': config['admin_password']
                }
            }
            
            poller = self.compute_client.virtual_machines.begin_create_or_update(
                rg, config['name'], vm_parameters
            )
            vm = poller.result()
            
            return self.get_instance(config['name'], rg)
            
        except AzureError as e:
            raise Exception(f"Failed to create instance: {e}")
    
    def delete_instance(self, vm_name: str, resource_group: Optional[str] = None) -> bool:
        """删除Azure虚拟机"""
        rg = resource_group or self.resource_group
        try:
            poller = self.compute_client.virtual_machines.begin_delete(rg, vm_name)
            poller.result()
            return True
        except AzureError:
            return False
    
    def get_instance_status(self, vm_name: str, resource_group: Optional[str] = None) -> str:
        """获取VM状态"""
        rg = resource_group or self.resource_group
        try:
            vm = self.compute_client.virtual_machines.get(rg, vm_name, expand='instanceView')
            status = vm.instance_view.statuses[-1].code.lower()
            
            if 'running' in status:
                return 'running'
            elif 'stopped' in status:
                return 'stopped'
            elif 'deallocated' in status:
                return 'deallocated'
            else:
                return status
        except AzureError:
            return 'unknown'
    
    def get_instance(self, vm_name: str, resource_group: Optional[str] = None) -> CloudResource:
        """获取单个VM详情"""
        rg = resource_group or self.resource_group
        try:
            vm = self.compute_client.virtual_machines.get(rg, vm_name, expand='instanceView')
            vm_size = vm.hardware_profile.vm_size
            cost = self._get_vm_price(vm_size)
            
            status = vm.instance_view.statuses[-1].code.lower()
            if 'running' in status:
                status = 'running'
            elif 'stopped' in status:
                status = 'stopped'
            
            return CloudResource(
                resource_id=vm.id,
                resource_type='virtual_machine',
                name=vm.name,
                status=status,
                region=self._get_location_display_name(vm.location),
                provider=CloudProvider.AZURE,
                specifications={
                    'vm_size': vm_size,
                    'resource_group': rg,
                    'location': vm.location,
                    'os_type': vm.storage_profile.os_disk.os_type.value,
                    'computer_name': vm.os_profile.computer_name,
                },
                cost_per_hour=cost,
                created_at=str(vm.id.split('/')[-1])[:10]
            )
        except AzureError as e:
            raise Exception(f"Failed to get instance: {e}")
    
    def get_cost_estimate(self, vm_size: str, hours: int) -> float:
        """获取Azure VM成本估算"""
        price = self._get_vm_price(vm_size)
        return price * hours
    
    def _get_vm_price(self, vm_size: str) -> float:
        """获取VM价格（每小时）"""
        # Azure VM定价（简化版）
        prices = {
            'Standard_B1s': 0.0104,
            'Standard_B2s': 0.0416,
            'Standard_B4ms': 0.166,
            'Standard_D2s_v3': 0.096,
            'Standard_D4s_v3': 0.192,
            'Standard_D8s_v3': 0.384,
            'Standard_D2s_v4': 0.096,
            'Standard_D4s_v4': 0.192,
            'Standard_D8s_v4': 0.384,
            'Standard_F2s_v2': 0.084,
            'Standard_F4s_v2': 0.168,
            'Standard_F8s_v2': 0.336,
            'Standard_NC4as_T4_v3': 0.612,
            'Standard_NC8as_T4_v3': 1.224,
            'Standard_NC24rs_v3': 6.48,
        }
        return prices.get(vm_size, 0.1)
    
    def _get_location_display_name(self, location: str) -> str:
        """获取位置显示名称"""
        locations = {
            'eastus': 'East US',
            'westus': 'West US',
            'centralus': 'Central US',
            'eastasia': 'East Asia',
            'southeastasia': 'Southeast Asia',
            'northeurope': 'North Europe',
            'westeurope': 'West Europe',
            'japaneast': 'Japan East',
        }
        return locations.get(location, location)
    
    def _resource_group_exists(self, resource_group: str) -> bool:
        """检查资源组是否存在"""
        try:
            self.resource_client.resource_groups.get(resource_group)
            return True
        except AzureError:
            return False
    
    def _create_network_components(self, rg: str, vnet_name: str, subnet_name: str, location: str):
        """创建网络组件"""
        try:
            # 创建虚拟网络
            vnet_poller = self.network_client.virtual_networks.begin_create_or_update(
                rg, vnet_name,
                {
                    'location': location,
                    'address_space': {'address_prefixes': ['10.0.0.0/16']}
                }
            )
            vnet_poller.result()
            
            # 创建子网
            subnet_poller = self.network_client.subnets.begin_create_or_update(
                rg, vnet_name, subnet_name,
                {'address_prefix': '10.0.0.0/24'}
            )
            subnet_poller.result()
        except AzureError:
            pass  # 组件可能已存在
    
    def list_regions(self) -> List[str]:
        """列出所有Azure区域"""
        return [
            'eastus', 'westus', 'centralus', 'eastasia', 'southeastasia',
            'northeurope', 'westeurope', 'japaneast', 'japanwest',
            'australiaeast', 'australiasoutheast', 'brazilsouth',
            'southcentralus', 'northcentralus', 'eastus2', 'westus2'
        ]
    
    def list_vm_sizes(self, location: str = 'eastus') -> List[Dict[str, str]]:
        """列出VM大小"""
        return [
            {'id': 'Standard_B1s', 'name': 'Standard_B1s - 1 vCPU, 1 GiB RAM', 'family': 'General Purpose'},
            {'id': 'Standard_B2s', 'name': 'Standard_B2s - 2 vCPU, 4 GiB RAM', 'family': 'General Purpose'},
            {'id': 'Standard_B4ms', 'name': 'Standard_B4ms - 4 vCPU, 16 GiB RAM', 'family': 'General Purpose'},
            {'id': 'Standard_D2s_v3', 'name': 'Standard_D2s_v3 - 2 vCPU, 8 GiB RAM', 'family': 'General Purpose'},
            {'id': 'Standard_D4s_v3', 'name': 'Standard_D4s_v3 - 4 vCPU, 16 GiB RAM', 'family': 'General Purpose'},
            {'id': 'Standard_F2s_v2', 'name': 'Standard_F2s_v2 - 2 vCPU, 4 GiB RAM', 'family': 'Compute Optimized'},
            {'id': 'Standard_F4s_v2', 'name': 'Standard_F4s_v2 - 4 vCPU, 8 GiB RAM', 'family': 'Compute Optimized'},
            {'id': 'Standard_NC4as_T4_v3', 'name': 'Standard_NC4as_T4_v3 - 4 vCPU, 28 GiB RAM + 1 GPU', 'family': 'GPU Accelerated'},
        ]
    
    def list_images(self) -> List[Dict[str, str]]:
        """列出常用镜像"""
        return [
            {'publisher': 'Canonical', 'offer': 'UbuntuServer', 'sku': '22.04-lts', 'name': 'Ubuntu 22.04 LTS'},
            {'publisher': 'Canonical', 'offer': 'UbuntuServer', 'sku': '20.04-lts', 'name': 'Ubuntu 20.04 LTS'},
            {'publisher': 'MicrosoftWindowsServer', 'offer': 'WindowsServer', 'sku': '2022-datacenter', 'name': 'Windows Server 2022'},
            {'publisher': 'MicrosoftWindowsServer', 'offer': 'WindowsServer', 'sku': '2019-datacenter', 'name': 'Windows Server 2019'},
            {'publisher': 'RedHat', 'offer': 'RHEL', 'sku': '82gen2', 'name': 'Red Hat Enterprise Linux 8'},
            {'publisher': 'openlogic', 'offer': 'CentOS', 'sku': '8_4', 'name': 'CentOS 8.4'},
        ]
