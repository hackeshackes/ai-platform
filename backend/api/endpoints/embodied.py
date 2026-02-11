"""
Embodied AI API Endpoints - v1.0

具身AI集成层API端点

提供设备管理、控制、实时流等功能
"""
import asyncio
import json
from datetime import datetime
from typing import Optional, Dict, Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, Query, Depends
from pydantic import BaseModel, Field

from embodied.devices.models import (
    Device,
    DeviceType,
    DeviceStatus,
    CommunicationProtocol,
    DeviceRegistration,
    DeviceCommand,
    DeviceResponse,
    MotionCommand,
    SensorReading,
    TrajectoryPoint,
)
from embodied.devices.device_manager import DeviceManager
from embodied.control.motion_control import MotionController, MotionType
from embodied.control.sensor_processor import SensorProcessor, SensorType
from embodied.control.trajectory import TrajectoryPlanner, CartesianPose
from embodied.iot.smart_home import SmartHomeManager, DeviceCategory
from embodied.iot.industrial import IndustrialManager, DeviceClass, IndustrialProtocol


# 全局实例
device_manager = DeviceManager(simulation_mode=True)
motion_controller = MotionController(simulation_mode=True)
sensor_processor = SensorProcessor(simulation_mode=True)
trajectory_planner = TrajectoryPlanner(simulation_mode=True)
smart_home_manager = SmartHomeManager(simulation_mode=True)
industrial_manager = IndustrialManager(simulation_mode=True)


router = APIRouter(prefix="/embodied", tags=["Embodied AI"])


# ==================== 请求/响应模型 ====================

class DeviceControlRequest(BaseModel):
    """设备控制请求"""
    action: str = Field(..., description="控制动作")
    params: Dict[str, Any] = Field(default_factory=dict, description="动作参数")


class MoveCommandRequest(BaseModel):
    """移动命令请求"""
    motion_type: str = Field(default="joint", description="运动类型")
    position: Dict[str, float] = Field(default_factory=dict, description="目标位置")
    velocity: float = Field(default=0.5, description="速度")
    blocking: bool = Field(default=False, description="是否阻塞")


class TrajectoryRequest(BaseModel):
    """轨迹请求"""
    start_pose: Dict[str, float] = Field(default_factory=dict, description="起始位置")
    end_pose: Dict[str, float] = Field(default_factory=dict, description="结束位置")
    num_points: int = Field(default=100, description="轨迹点数")
    motion_type: str = Field(default="linear", description="轨迹类型")


class SmartHomeDeviceRequest(BaseModel):
    """智能家居设备添加请求"""
    name: str = Field(..., description="设备名称")
    category: str = Field(..., description="设备类别")
    room: str = Field(default=None, description="所在房间")


class IndustrialDeviceRequest(BaseModel):
    """工业设备添加请求"""
    name: str = Field(..., description="设备名称")
    device_class: str = Field(..., description="设备类别")
    protocol: str = Field(..., description="通信协议")


# ==================== 设备管理端点 ====================

@router.post("/devices/register", response_model=Device)
async def register_device(registration: DeviceRegistration):
    """
    注册新设备
    
    注册一个设备到设备管理器
    """
    device = await device_manager.register_device(registration)
    return device


@router.delete("/devices/{device_id}")
async def unregister_device(device_id: str):
    """
    注销设备
    
    从设备管理器中移除设备
    """
    success = await device_manager.unregister_device(device_id)
    if not success:
        raise HTTPException(status_code=404, detail="Device not found")
    return {"status": "unregistered", "device_id": device_id}


@router.get("/devices", response_model=list)
async def list_devices(
    device_type: Optional[DeviceType] = Query(None, description="设备类型过滤"),
    status: Optional[DeviceStatus] = Query(None, description="状态过滤"),
    protocol: Optional[CommunicationProtocol] = Query(None, description="协议过滤")
):
    """
    列出设备
    
    返回符合过滤条件的设备列表
    """
    devices = await device_manager.list_devices(
        device_type=device_type,
        status=status,
        protocol=protocol
    )
    return devices


@router.get("/devices/{device_id}", response_model=Device)
async def get_device(device_id: str):
    """
    获取设备详情
    """
    device = await device_manager.get_device(device_id)
    if not device:
        raise HTTPException(status_code=404, detail="Device not found")
    return device


@router.post("/devices/{device_id}/control", response_model=DeviceResponse)
async def control_device(
    device_id: str,
    request: DeviceControlRequest
):
    """
    控制设备
    
    向设备发送控制命令
    """
    command = DeviceCommand(
        device_id=device_id,
        action=request.action,
        params=request.params
    )
    
    # 验证命令
    valid = await device_manager.validate_command(device_id, request.action, request.params)
    if not valid:
        raise HTTPException(status_code=400, detail="Invalid command")
    
    response = await device_manager.send_command(command)
    return response


@router.post("/devices/{device_id}/command", response_model=DeviceResponse)
async def send_command(
    device_id: str,
    command: DeviceCommand
):
    """
    发送设备命令
    
    发送原始命令到设备
    """
    command.device_id = device_id
    response = await device_manager.send_command(command)
    return response


@router.put("/devices/{device_id}/status")
async def update_device_status(
    device_id: str,
    status: DeviceStatus
):
    """
    更新设备状态
    """
    success = await device_manager.update_device_status(device_id, status)
    if not success:
        raise HTTPException(status_code=404, detail="Device not found")
    return {"status": "updated", "device_id": device_id}


@router.post("/devices/{device_id}/heartbeat")
async def heartbeat(device_id: str):
    """
    设备心跳
    """
    success = await device_manager.heartbeat(device_id)
    if not success:
        raise HTTPException(status_code=404, detail="Device not found")
    return {"status": "ok", "device_id": device_id}


@router.get("/devices/{device_id}/capabilities")
async def get_capabilities(device_id: str):
    """
    获取设备能力列表
    """
    capabilities = await device_manager.get_device_capabilities(device_id)
    if capabilities is None:
        raise HTTPException(status_code=404, detail="Device not found")
    return {"device_id": device_id, "capabilities": capabilities}


# ==================== 机器人控制端点 ====================

@router.post("/robots/{device_id}/move", response_model=dict)
async def move_robot(
    device_id: str,
    request: MoveCommandRequest
):
    """
    控制机器人移动
    
    向机器人发送移动命令
    """
    motion_cmd = MotionCommand(
        device_id=device_id,
        motion_type=MotionType(request.motion_type),
        target=request.position,
        velocity=request.velocity,
        blocking=request.blocking
    )
    
    motion_id = await motion_controller.submit_motion(device_id, motion_cmd)
    
    return {
        "motion_id": motion_id,
        "status": "submitted",
        "command": request.dict()
    }


@router.get("/robots/{device_id}/status")
async def get_robot_status(device_id: str):
    """
    获取机器人运动状态
    """
    state = await motion_controller.get_motion_state(device_id)
    return state


@router.post("/robots/{device_id}/stop")
async def stop_robot(device_id: str):
    """
    停止机器人运动
    """
    success = await motion_controller.stop_motion(device_id)
    return {"status": "stopped" if success else "error"}


@router.post("/robots/{device_id}/pause")
async def pause_robot(device_id: str):
    """
    暂停机器人运动
    """
    success = await motion_controller.pause_motion(device_id)
    return {"status": "paused" if success else "error"}


@router.post("/robots/{device_id}/resume")
async def resume_robot(device_id: str):
    """
    恢复机器人运动
    """
    success = await motion_controller.resume_motion(device_id)
    return {"status": "resumed" if success else "error"}


# ==================== 轨迹规划端点 ====================

@router.post("/trajectory/generate")
async def generate_trajectory(request: TrajectoryRequest):
    """
    生成轨迹
    
    根据起始和结束位置生成轨迹
    """
    start_pose = request.start_pose or {"x": 0, "y": 0, "z": 0}
    end_pose = request.end_pose or {"x": 1, "y": 0, "z": 0}
    
    trajectory = await trajectory_planner.create_linear_trajectory(
        device_id="trajectory_planner",
        start_pose=start_pose,
        end_pose=end_pose,
        num_points=request.num_points
    )
    
    return {
        "trajectory_id": trajectory.id,
        "total_time": trajectory.total_time,
        "total_distance": trajectory.total_distance,
        "points_count": len(trajectory.points)
    }


@router.get("/trajectory/{trajectory_id}")
async def get_trajectory(trajectory_id: str):
    """
    获取轨迹详情
    """
    # 简化实现
    return {"trajectory_id": trajectory_id, "status": "generated"}


# ==================== 传感器端点 ====================

@router.get("/sensors/{device_id}/{sensor_type}")
async def get_sensor_reading(
    device_id: str,
    sensor_type: SensorType
):
    """
    获取传感器读数
    """
    reading = await sensor_processor.get_reading(device_id, sensor_type)
    if reading is None:
        # 生成模拟读数
        reading = await sensor_processor.generate_simulated_reading(device_id, sensor_type)
    return {
        "device_id": device_id,
        "sensor_type": sensor_type.value,
        "value": reading.value,
        "unit": reading.unit,
        "timestamp": reading.timestamp
    }


@router.get("/sensors/{device_id}/{sensor_type}/statistics")
async def get_sensor_statistics(
    device_id: str,
    sensor_type: SensorType
):
    """
    获取传感器统计数据
    """
    stats = await sensor_processor.get_statistics(device_id, sensor_type)
    return {
        "device_id": device_id,
        "sensor_type": sensor_type.value,
        "statistics": stats
    }


# ==================== 智能家居端点 ====================

@router.post("/smart-home/devices")
async def add_smart_home_device(request: SmartHomeDeviceRequest):
    """
    添加智能家居设备
    """
    import uuid
    device_id = f"sh_{uuid.uuid4().hex[:8]}"
    
    category = DeviceCategory(request.category)
    device = await smart_home_manager.add_device(
        device_id=device_id,
        category=category,
        name=request.name,
        room=request.room
    )
    
    return {
        "device_id": device_id,
        "name": request.name,
        "category": request.category,
        "status": "added"
    }


@router.get("/smart-home/devices")
async def list_smart_home_devices(
    category: Optional[str] = Query(None, description="设备类别过滤"),
    room: Optional[str] = Query(None, description="房间过滤")
):
    """
    列出智能家居设备
    """
    cat_enum = DeviceCategory(category) if category else None
    devices = await smart_home_manager.list_devices(category=cat_enum, room=room)
    return {"devices": devices, "count": len(devices)}


@router.post("/smart-home/devices/{device_id}/control")
async def control_smart_home_device(
    device_id: str,
    request: DeviceControlRequest
):
    """
    控制智能家居设备
    """
    result = await smart_home_manager.control_device(
        device_id,
        request.action,
        request.params
    )
    return result


@router.get("/smart-home/rooms/{room}/status")
async def get_room_status(room: str):
    """
    获取房间状态
    """
    status = await smart_home_manager.get_room_status(room)
    return status


@router.get("/smart-home/status")
async def get_smart_home_status():
    """
    获取智能家居系统状态
    """
    status = await smart_home_manager.get_all_status()
    return status


# ==================== 工业设备端点 ====================

@router.post("/industrial/devices")
async def add_industrial_device(request: IndustrialDeviceRequest):
    """
    添加工业设备
    """
    import uuid
    device_id = f"ind_{uuid.uuid4().hex[:8]}"
    
    device_class = DeviceClass(request.device_class)
    protocol = IndustrialProtocol(request.protocol)
    
    device = await industrial_manager.add_device(
        device_id=device_id,
        device_class=device_class,
        protocol=protocol,
        name=request.name
    )
    
    return {
        "device_id": device_id,
        "name": request.name,
        "device_class": request.device_class,
        "protocol": request.protocol,
        "status": "added"
    }


@router.get("/industrial/devices")
async def list_industrial_devices(
    device_class: Optional[str] = Query(None, description="设备类别过滤"),
    protocol: Optional[str] = Query(None, description="协议过滤")
):
    """
    列出工业设备
    """
    class_enum = DeviceClass(device_class) if device_class else None
    protocol_enum = IndustrialProtocol(protocol) if protocol else None
    
    devices = await industrial_manager.list_devices(
        device_class=class_enum,
        protocol=protocol_enum
    )
    return {"devices": devices, "count": len(devices)}


@router.get("/industrial/devices/{device_id}")
async def get_industrial_device(device_id: str):
    """
    获取工业设备详情
    """
    device = await industrial_manager.get_device(device_id)
    if not device:
        raise HTTPException(status_code=404, detail="Device not found")
    return await device.get_status()


@router.get("/industrial/devices/{device_id}/tags")
async def read_industrial_tags(
    device_id: str,
    tags: str = Query(None, description="标签列表，逗号分隔")
):
    """
    读取工业设备标签
    """
    tag_list = tags.split(",") if tags else None
    values = await industrial_manager.read_device(device_id, tag_list if tag_list else None)
    return {"device_id": device_id, "values": values}


@router.post("/industrial/devices/{device_id}/tags")
async def write_industrial_tags(
    device_id: str,
    values: Dict[str, Any]
):
    """
    写入工业设备标签
    """
    result = await industrial_manager.write_device(device_id, values)
    return result


@router.get("/industrial/alarms")
async def get_all_alarms():
    """
    获取所有报警
    """
    alarms = await industrial_manager.get_all_alarms()
    return {"alarms": alarms, "count": len(alarms)}


@router.get("/industrial/status")
async def get_industrial_status():
    """
    获取工业系统状态
    """
    status = await industrial_manager.get_statistics()
    return status


@router.get("/industrial/scan")
async def scan_network():
    """
    扫描网络设备
    """
    discovered = await industrial_manager.scan_network()
    return {"discovered": discovered, "count": len(discovered)}


# ==================== WebSocket实时流端点 ====================

@router.websocket("/stream/{device_id}")
async def websocket_stream(websocket: WebSocket, device_id: str):
    """
    设备实时数据流
    
    通过WebSocket接收设备实时数据
    """
    await websocket.accept()
    
    try:
        # 发送连接确认
        await websocket.send_json({
            "type": "connected",
            "device_id": device_id,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # 持续发送模拟数据
        while True:
            # 生成模拟传感器数据
            import random
            data = {
                "type": "telemetry",
                "device_id": device_id,
                "timestamp": datetime.utcnow().isoformat(),
                "data": {
                    "temperature": round(random.uniform(20, 30), 1),
                    "humidity": round(random.uniform(40, 60), 1),
                    "pressure": round(random.uniform(1000, 1020), 1)
                }
            }
            
            await websocket.send_json(data)
            await asyncio.sleep(1.0)
            
    except WebSocketDisconnect:
        pass


# ==================== 统计端点 ====================

@router.get("/stats")
async def get_stats():
    """
    获取系统统计信息
    """
    return {
        "devices": await device_manager.get_stats(),
        "motion": {
            "status": await motion_controller.get_all_states()
        },
        "sensors": {
            "registered": sensor_processor.get_sensor_list(),
            "buffer_stats": sensor_processor.get_buffer_stats()
        },
        "smart_home": await smart_home_manager.get_all_status(),
        "industrial": await industrial_manager.get_statistics()
    }


@router.get("/health")
async def health_check():
    """服务健康检查"""
    return {
        "status": "healthy",
        "service": "embodied-ai",
        "timestamp": datetime.utcnow().isoformat()
    }
