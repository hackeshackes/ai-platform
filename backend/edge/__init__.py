"""
边缘AI部署模块
支持ONNX/TensorRT优化、边缘设备管理、离线推理
"""

from .optimizer import ModelOptimizer, OptimizationConfig, create_optimizer, ONNXOptimizer
from .tensorrt import TensorRTEngine, TensorRTConfig, PrecisionMode, create_tensorrt_engine
from .device import DeviceManager, DeviceInfo, DeviceStatus, DeviceType, create_device_manager
from .offline import OfflineInferenceEngine, InferenceRequest, InferenceResponse, create_inference_engine

__all__ = [
    # Optimizer
    "ModelOptimizer",
    "OptimizationConfig", 
    "create_optimizer",
    "ONNXOptimizer",
    
    # TensorRT
    "TensorRTEngine",
    "TensorRTConfig",
    "PrecisionMode",
    "create_tensorrt_engine",
    
    # Device
    "DeviceManager",
    "DeviceInfo",
    "DeviceStatus",
    "DeviceType",
    "create_device_manager",
    
    # Offline Inference
    "OfflineInferenceEngine",
    "InferenceRequest",
    "InferenceResponse",
    "create_inference_engine"
]
