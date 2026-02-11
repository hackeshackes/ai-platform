# Edge AI Deployment Module

AI Platform v7 边缘AI部署模块，支持ONNX/TensorRT优化、边缘设备管理和离线推理。

## 文件结构

```
backend/edge/
├── __init__.py           # 模块初始化
├── optimizer.py          # 模型优化器 (ONNX导出、量化)
├── tensorrt.py           # TensorRT加速引擎
├── device.py             # 边缘设备管理
├── offline.py            # 离线推理引擎
└── requirements.txt      # 依赖列表
```

## 快速开始

### 1. 模型优化 (ONNX导出)

```python
from edge.optimizer import create_optimizer, OptimizationConfig

optimizer = create_optimizer()

# 导出PyTorch模型到ONNX
config = OptimizationConfig(
    input_shape=[1, 3, 224, 224],
    opset_version=11,
    optimize_for_inference=True
)
result = optimizer.export_to_onnx(model, config, "model.onnx")
```

### 2. TensorRT加速

```python
from edge.tensorrt import create_tensorrt_engine, TensorRTConfig, PrecisionMode

tensorrt = create_tensorrt_engine()

config = TensorRTConfig(
    max_batch_size=32,
    precision_mode=PrecisionMode.FP16
)
result = tensorrt.convert_onnx_to_engine("model.onnx", config, "engine.plan")
```

### 3. 设备管理

```python
from edge.device import create_device_manager, DeviceType, DeviceStatus

dm = create_device_manager()

# 注册设备
device = dm.register_device(
    device_name="Jetson Nano 1",
    device_type=DeviceType.JETSON_NANO,
    ip_address="192.168.1.100"
)

# 列出设备
devices = dm.list_devices(status=DeviceStatus.ONLINE)
```

### 4. 离线推理

```python
from edge.offline import create_inference_engine, InferenceMode

engine = create_inference_engine()

# 加载模型
engine.load_model("model_id", "model.onnx", "onnx", "cpu")

# 执行推理
response = engine.infer("model_id", {"input": data})
```

## API端点

| 端点 | 方法 | 描述 |
|------|------|------|
| `/api/v1/edge/optimize/onnx` | POST | ONNX导出 |
| `/api/v1/edge/optimize/tensorrt` | POST | TensorRT加速 |
| `/api/v1/edge/devices` | GET | 设备列表 |
| `/api/v1/edge/deploy` | POST | 部署到设备 |
| `/api/v1/edge/sync` | POST | 数据同步 |
| `/api/v1/edge/inference` | POST | 离线推理 |

## 支持的设备类型

- `jetson_nano` - NVIDIA Jetson Nano
- `jetson_xavier` - NVIDIA Jetson Xavier
- `jetson_orin` - NVIDIA Jetson Orin
- `raspberry_pi` - Raspberry Pi
- `edge_tpu` - Google Edge TPU
- `intel_ncs2` - Intel Neural Compute Stick 2
- `generic_x86` - Generic X86 Device

## 依赖安装

```bash
pip install torch torchvision tensorflow onnx onnxruntime tensorrt aiohttp
```
