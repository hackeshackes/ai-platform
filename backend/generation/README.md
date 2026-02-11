# AI Platform v7 - 多模态生成模块

## 概述

本模块提供图像、音频、视频生成能力，支持多种AI服务提供商。

## 功能特性

### 1. 图像生成 (Image Generation)
- **DALL-E 3** - OpenAI官方图像生成模型
- **Stable Diffusion** - 开源图像生成模型
- **图生图 (Image-to-Image)** - 基于参考图生成新图像
- **图像编辑** - 局部编辑和图像变体

### 2. 语音合成 (TTS)
- **OpenAI TTS** - 高质量语音合成
- **Azure TTS** - 微软认知服务语音
- **ElevenLabs** - 超真实语音合成
- **多语言支持** - 中英日韩等多种语言
- **语音参数调节** - 语速、音调、音量

### 3. 视频生成 (Video Generation)
- **Sora** - OpenAI视频生成模型
- **Runway** - 专业视频生成平台
- **Stable Video Diffusion** - 开源视频生成
- **图生视频** - 基于静态图像生成视频
- **视频增强** - 超分辨率、去噪等

### 4. 多模态生成
- **统一接口** - 一次调用生成多种模态
- **并行生成** - 同时生成图像和音频
- **智能组合** - 根据场景自动选择模型

## 快速开始

### 安装依赖

```bash
pip install -r backend/generation/requirements.txt
```

### 环境配置

在 `.env` 文件中配置API密钥：

```env
# OpenAI
OPENAI_API_KEY=your_openai_api_key

# Azure
AZURE_TTS_KEY=your_azure_tts_key
AZURE_TTS_REGION=eastus

# ElevenLabs
ELEVENLABS_API_KEY=your_elevenlabs_api_key

# Stable Diffusion
STABLE_DIFFUSION_API_URL=https://api.stability.ai
STABLE_DIFFUSION_TOKEN=your_stability_token
```

### 基本使用

```python
from backend.generation.unified import get_generation_manager

async def example():
    manager = get_generation_manager()
    
    # 图像生成
    image = await manager.generate(
        modality="image",
        prompt="一只可爱的猫咪",
        provider="dalle"
    )
    
    # 语音合成
    audio = await manager.generate(
        modality="audio",
        text="你好世界",
        provider="openai"
    )
    
    # 视频生成
    video = await manager.generate(
        modality="video",
        prompt="城市夜景延时摄影",
        provider="sora"
    )
```

## API 端点

### 基础信息

- **Base URL**: `/api/v1/generation`
- **Content-Type**: application/json

### API列表

| 方法 | 端点 | 描述 |
|------|------|------|
| POST | `/api/v1/generation/image` | 生成图像 |
| POST | `/api/v1/generation/audio` | 合成语音 |
| POST | `/api/v1/generation/video` | 生成视频 |
| POST | `/api/v1/generation/multimodal` | 多模态生成 |
| GET | `/api/v1/generation/models` | 获取可用模型 |
| GET | `/api/v1/generation/history` | 获取生成历史 |
| GET | `/api/v1/generation/models/{modality}` | 获取指定模态模型 |
| POST | `/api/v1/generation/cost/estimate` | 估算生成成本 |
| GET | `/api/v1/generation/health` | 健康检查 |

### 请求示例

#### 图像生成

```bash
curl -X POST http://localhost:8000/api/v1/generation/image \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "一只可爱的橘猫",
    "model": "dalle",
    "size": "1024x1024",
    "num_images": 1
  }'
```

#### 语音合成

```bash
curl -X POST http://localhost:8000/api/v1/generation/audio \
  -H "Content-Type: application/json" \
  -d '{
    "text": "欢迎使用AI Platform",
    "model": "openai",
    "voice": "nova",
    "speed": 1.0,
    "format": "mp3"
  }'
```

#### 多模态生成

```bash
curl -X POST http://localhost:8000/api/v1/generation/multimodal \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "一只小狗在草地上奔跑",
    "modalities": ["image", "audio"],
    "providers": {
      "image": "dalle",
      "audio": "openai"
    }
  }'
```

## 支持的参数

### 图像生成参数

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| prompt | string | 必填 | 图像描述 |
| model | string | "dalle" | 模型名称 |
| size | string | "1024x1024" | 图像尺寸 |
| style | string | - | 图像风格 |
| quality | string | "standard" | 图像质量 |
| num_images | int | 1 | 生成数量 |
| seed | int | - | 随机种子 |

### 语音合成参数

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| text | string | 必填 | 要转换的文字 |
| model | string | "openai" | TTS模型 |
| voice | string | - | 声音ID |
| speed | float | 1.0 | 语速 |
| pitch | float | 1.0 | 音调 |
| volume | float | 1.0 | 音量 |
| format | string | "mp3" | 音频格式 |

### 视频生成参数

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| prompt | string | 必填 | 视频描述 |
| model | string | "sora" | 视频模型 |
| duration | string | "5-10秒" | 视频时长 |
| resolution | string | "1080x1920" | 分辨率 |
| fps | int | 24 | 帧率 |
| style | string | "cinematic" | 视频风格 |

## 成本参考

| 模态 | 提供商 | 单价 |
|------|--------|------|
| 图像 | DALL-E 3 | $0.04/张 |
| 图像 | Stable Diffusion | $0.01/张 |
| 音频 | OpenAI TTS | $0.015/1000字符 |
| 音频 | Azure TTS | $0.01/1000字符 |
| 视频 | Sora | $0.50/秒 |
| 视频 | Runway | $0.30/秒 |

## 文件结构

```
backend/generation/
├── __init__.py           # 模块入口
├── base.py              # 基类和通用接口
├── image.py             # 图像生成
├── audio.py             # 语音合成
├── video.py             # 视频生成
├── unified.py           # 统一接口
├── requirements.txt     # 依赖列表
└── examples/
    └── quick_start.py   # 快速开始示例
```

## 扩展开发

### 添加新的图像生成器

```python
from backend.generation.image import BaseImageGenerator

class MyImageGenerator(BaseImageGenerator):
    async def generate(self, request):
        # 实现你的生成逻辑
        pass

# 注册到引擎
from backend.generation.image import get_image_engine
engine = get_image_engine()
engine.register_generator("my_generator", MyImageGenerator())
```

### 添加新的TTS引擎

```python
from backend.generation.audio import BaseTTSEngine

class MyTTSEngine(BaseTTSEngine):
    async def synthesize(self, request):
        # 实现你的TTS逻辑
        pass
```

## 注意事项

1. **API密钥安全**: 请勿在代码中硬编码API密钥，使用环境变量
2. **成本控制**: 建议设置使用限额和监控
3. **异步处理**: 所有API调用均为异步，支持高并发
4. **错误处理**: 建议实现重试和降级策略
5. **数据安全**: 注意用户上传内容的安全审查
