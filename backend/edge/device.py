"""
边缘AI部署 - 设备管理
管理边缘设备连接、状态监控、部署和通信
"""

import os
import json
import logging
import asyncio
import uuid
from pathlib import Path
from typing import Optional, Dict, Any, List, AsyncGenerator
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)

# 条件导入aiohttp
try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    logger.warning("aiohttp not available, async device communication disabled")


class DeviceStatus(Enum):
    ONLINE = "online"
    OFFLINE = "offline"
    BUSY = "busy"
    UPDATING = "updating"
    ERROR = "error"


class DeviceType(Enum):
    JETSON_NANO = "jetson_nano"
    JETSON_XAVIER = "jetson_xavier"
    JETSON_ORIN = "jetson_orin"
    RASPBERRY_PI = "raspberry_pi"
    EDGE_TPU = "edge_tpu"
    INTEL_NCS2 = "intel_ncs2"
    GENERIC_X86 = "generic_x86"
    ANDROID = "android"
    IOS = "ios"


@dataclass
class DeviceInfo:
    """设备信息"""
    device_id: str
    device_name: str
    device_type: DeviceType
    status: DeviceStatus
    ip_address: Optional[str] = None
    mac_address: Optional[str] = None
    os_version: Optional[str] = None
    cpu_cores: int = 0
    memory_total: int = 0  # GB
    storage_total: int = 0  # GB
    gpu_info: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "device_id": self.device_id,
            "device_name": self.device_name,
            "device_type": self.device_type.value,
            "status": self.status.value,
            "ip_address": self.ip_address,
            "os_version": self.os_version,
            "cpu_cores": self.cpu_cores,
            "memory_total": self.memory_total,
            "storage_total": self.storage_total,
            "gpu_info": self.gpu_info,
            "created_at": self.created_at.isoformat(),
            "last_seen": self.last_seen.isoformat(),
            "tags": self.tags,
            "metadata": self.metadata
        }


@dataclass
class DeviceMetrics:
    """设备性能指标"""
    device_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    cpu_usage: float = 0.0  # percentage
    memory_usage: float = 0.0  # percentage
    storage_usage: float = 0.0  # percentage
    gpu_usage: float = 0.0  # percentage
    temperature: float = 0.0  # celsius
    network_in: int = 0  # bytes
    network_out: int = 0  # bytes
    active_inferences: int = 0
    queue_length: int = 0


class DeviceManager:
    """边缘设备管理器"""
    
    def __init__(self, db_path: str = "/tmp/edge_devices.json"):
        self.db_path = Path(db_path)
        self._devices: Dict[str, DeviceInfo] = {}
        self._metrics_history: Dict[str, List[DeviceMetrics]] = {}
        self._load_devices()
    
    def _load_devices(self):
        """从文件加载设备列表"""
        if self.db_path.exists():
            try:
                with open(self.db_path, 'r') as f:
                    data = json.load(f)
                    for item in data:
                        item['created_at'] = datetime.fromisoformat(item['created_at'])
                        item['last_seen'] = datetime.fromisoformat(item['last_seen'])
                        self._devices[item['device_id']] = DeviceInfo(**item)
            except Exception as e:
                logger.error(f"Failed to load devices: {e}")
    
    def _save_devices(self):
        """保存设备列表到文件"""
        data = [device.to_dict() for device in self._devices.values()]
        with open(self.db_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def register_device(
        self,
        device_name: str,
        device_type: DeviceType,
        ip_address: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> DeviceInfo:
        """注册新设备"""
        device_id = str(uuid.uuid4())[:8]
        
        device = DeviceInfo(
            device_id=device_id,
            device_name=device_name,
            device_type=device_type,
            status=DeviceStatus.OFFLINE,
            ip_address=ip_address,
            metadata=metadata or {}
        )
        
        self._devices[device_id] = device
        self._metrics_history[device_id] = []
        self._save_devices()
        
        logger.info(f"Registered device: {device_id} - {device_name}")
        return device
    
    def unregister_device(self, device_id: str) -> bool:
        """注销设备"""
        if device_id in self._devices:
            del self._devices[device_id]
            if device_id in self._metrics_history:
                del self._metrics_history[device_id]
            self._save_devices()
            return True
        return False
    
    def get_device(self, device_id: str) -> Optional[DeviceInfo]:
        """获取设备信息"""
        return self._devices.get(device_id)
    
    def list_devices(
        self,
        status: Optional[DeviceStatus] = None,
        device_type: Optional[DeviceType] = None
    ) -> List[DeviceInfo]:
        """列出设备"""
        devices = list(self._devices.values())
        
        if status:
            devices = [d for d in devices if d.status == status]
        if device_type:
            devices = [d for d in devices if d.device_type == device_type]
        
        return devices
    
    def update_device_status(
        self,
        device_id: str,
        status: DeviceStatus,
        metrics: Optional[DeviceMetrics] = None
    ) -> bool:
        """更新设备状态"""
        if device_id not in self._devices:
            return False
        
        device = self._devices[device_id]
        device.status = status
        device.last_seen = datetime.now()
        
        if metrics:
            device.cpu_cores = getattr(metrics, 'cpu_cores', device.cpu_cores)
            if device_id not in self._metrics_history:
                self._metrics_history[device_id] = []
            self._metrics_history[device_id].append(metrics)
            self._metrics_history[device_id] = self._metrics_history[device_id][-100:]
        
        self._save_devices()
        return True
    
    def get_device_metrics(
        self,
        device_id: str,
        limit: int = 100
    ) -> List[DeviceMetrics]:
        """获取设备指标历史"""
        return self._metrics_history.get(device_id, [])[-limit:]
    
    def get_device_stats(self) -> Dict[str, Any]:
        """获取设备统计信息"""
        devices = list(self._devices.values())
        
        status_counts = {}
        type_counts = {}
        
        for device in devices:
            status_counts[device.status.value] = status_counts.get(device.status.value, 0) + 1
            type_counts[device.device_type.value] = type_counts.get(device.device_type.value, 0) + 1
        
        return {
            "total_devices": len(devices),
            "online_count": status_counts.get(DeviceStatus.ONLINE.value, 0),
            "offline_count": status_counts.get(DeviceStatus.OFFLINE.value, 0),
            "status_distribution": status_counts,
            "type_distribution": type_counts
        }


class DeviceCommunicator:
    """设备通信器 - 用于与边缘设备交互"""
    
    def __init__(self, device_manager: DeviceManager):
        self.device_manager = device_manager
        self._connections: Dict[str, Any] = {}
    
    async def connect_to_device(
        self,
        device_id: str,
        endpoint: str,
        api_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """连接到设备"""
        if not AIOHTTP_AVAILABLE:
            return {"status": "warning", "message": "aiohttp not available, using mock connection"}
        
        device = self.device_manager.get_device(device_id)
        if not device:
            return {"status": "error", "message": "Device not found"}
        
        try:
            async with aiohttp.ClientSession() as session:
                headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
                
                async with session.get(
                    f"{endpoint}/api/v1/health",
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        self._connections[device_id] = {"endpoint": endpoint, "session": session}
                        self.device_manager.update_device_status(device_id, DeviceStatus.ONLINE)
                        return {"status": "connected", "device_info": data}
                    else:
                        return {"status": "error", "message": f"Connection failed: {response.status}"}
                        
        except Exception as e:
            logger.error(f"Failed to connect to device {device_id}: {e}")
            self.device_manager.update_device_status(device_id, DeviceStatus.OFFLINE)
            return {"status": "error", "message": str(e)}
    
    async def deploy_model(
        self,
        device_id: str,
        model_path: str,
        endpoint: str,
        api_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """部署模型到设备"""
        if not AIOHTTP_AVAILABLE:
            return {"status": "warning", "message": "aiohttp not available, deployment simulated"}
        
        if not os.path.exists(model_path):
            return {"status": "error", "message": "Model file not found"}
        
        try:
            async with aiohttp.ClientSession() as session:
                headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
                
                with open(model_path, 'rb') as f:
                    model_data = f.read()
                
                form_data = aiohttp.FormData()
                form_data.add_field('model', model_data, filename=os.path.basename(model_path))
                
                async with session.post(
                    f"{endpoint}/api/v1/models/deploy",
                    data=form_data,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=300)
                ) as response:
                    return await response.json()
                    
        except Exception as e:
            logger.error(f"Deployment to device {device_id} failed: {e}")
            return {"status": "error", "message": str(e)}
    
    async def run_inference_on_device(
        self,
        device_id: str,
        endpoint: str,
        input_data: Dict[str, Any],
        api_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """在设备上运行推理"""
        if not AIOHTTP_AVAILABLE:
            return {"status": "warning", "message": "aiohttp not available, inference simulated"}
        
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                } if api_key else {"Content-Type": "application/json"}
                
                async with session.post(
                    f"{endpoint}/api/v1/inference",
                    json=input_data,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        return {"status": "error", "message": f"Inference failed: {response.status}"}
                        
        except Exception as e:
            logger.error(f"Inference on device {device_id} failed: {e}")
            return {"status": "error", "message": str(e)}


# 便捷函数
def create_device_manager(db_path: str = "/tmp/edge_devices.json") -> DeviceManager:
    """创建设备管理器"""
    return DeviceManager(db_path=db_path)
