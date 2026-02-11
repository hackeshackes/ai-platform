"""
生成API端点
提供图像、音频、视频生成的REST API
"""

import os
import base64
import uuid
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends, Query
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum

from generation import (
    GenerationProvider,
    TaskStatus,
    GenerationType,
    get_unified_service,
    get_generation_manager
)


router = APIRouter(prefix="/generation", tags=["generation"])

# ============ 请求模型 ============

class ImageSizeEnum(str, Enum):
    """图像尺寸枚举"""
    SQUARE = "1024x1024"
    PORTRAIT = "1024x1792"
    LANDSCAPE = "1792x1024"
    SMALL = "512x512"
    MEDIUM = "768x768"


class ImageStyleEnum(str, Enum):
    """图像风格枚举"""
    REALISTIC = "realistic"
    ARTISTIC = "artistic"
    ANIME = "anime"
    ABSTRACT = "abstract"
    PHOTOGRAPHIC = "photographic"
    OIL_PAINTING = "oil_painting"
    WATERCOLOR = "watercolor"
    SKETCH = "sketch"


class ImageGenerateRequest(BaseModel):
    """图像生成请求"""
    prompt: str = Field(..., min_length=3, max_length=4000, description="图像描述提示词")
    model: Optional[str] = Field(default="dalle", description="模型名称")
    size: ImageSizeEnum = Field(default=ImageSizeEnum.SQUARE, description="图像尺寸")
    style: Optional[ImageStyleEnum] = Field(default=None, description="图像风格")
    quality: str = Field(default="standard", description="图像质量")
    negative_prompt: Optional[str] = Field(default=None, description="负面提示词")
    num_images: int = Field(default=1, ge=1, le=4, description="生成图像数量")
    seed: Optional[int] = Field(default=None, description="随机种子")


class ImageEditRequest(BaseModel):
    """图像编辑请求"""
    prompt: str = Field(..., description="编辑描述")
    mask: Optional[str] = Field(default=None, description="mask图片base64")


class AudioFormatEnum(str, Enum):
    """音频格式枚举"""
    MP3 = "mp3"
    WAV = "wav"
    OGG = "ogg"
    FLAC = "flac"
    AAC = "aac"


class AudioGenerateRequest(BaseModel):
    """音频生成请求"""
    text: str = Field(..., min_length=1, max_length=5000, description="要转换的文字")
    model: Optional[str] = Field(default="openai", description="TTS模型")
    voice: Optional[str] = Field(default=None, description="声音ID")
    voice_gender: str = Field(default="female", description="声音性别")
    accent: str = Field(default="zh-CN", description="口音")
    speed: float = Field(default=1.0, ge=0.5, le=2.0, description="语速")
    pitch: float = Field(default=1.0, ge=0.5, le=2.0, description="音调")
    volume: float = Field(default=1.0, ge=0.0, le=1.0, description="音量")
    format: AudioFormatEnum = Field(default=AudioFormatEnum.MP3, description="音频格式")


class VideoDurationEnum(str, Enum):
    """视频时长枚举"""
    SHORT = "3-5秒"
    MEDIUM = "5-10秒"
    LONG = "10-15秒"
    EXTENDED = "15-30秒"


class VideoResolutionEnum(str, Enum):
    """视频分辨率枚举"""
    SD = "576x1024"
    HD = "720x1280"
    FHD = "1080x1920"
    UHD = "2160x3840"


class VideoAspectRatioEnum(str, Enum):
    """视频宽高比枚举"""
    RATIO_9_16 = "9:16"
    RATIO_16_9 = "16:9"
    RATIO_1_1 = "1:1"


class VideoStyleEnum(str, Enum):
    """视频风格枚举"""
    REALISTIC = "realistic"
    CINEMATIC = "cinematic"
    ANIMATED = "animated"
    ABSTRACT = "abstract"
    DOCUMENTARY = "documentary"


class VideoGenerateRequest(BaseModel):
    """视频生成请求"""
    prompt: str = Field(..., min_length=3, max_length=2000, description="视频描述")
    model: Optional[str] = Field(default="sora", description="视频模型")
    negative_prompt: Optional[str] = Field(default=None, description="负面提示词")
    duration: VideoDurationEnum = Field(default=VideoDurationEnum.MEDIUM, description="视频时长")
    resolution: VideoResolutionEnum = Field(default=VideoResolutionEnum.HD, description="分辨率")
    aspect_ratio: VideoAspectRatioEnum = Field(default=VideoAspectRatioEnum.RATIO_9_16, description="宽高比")
    style: VideoStyleEnum = Field(default=VideoStyleEnum.CINEMATIC, description="视频风格")
    fps: int = Field(default=24, ge=24, le=60, description="帧率")
    seed: Optional[int] = Field(default=None, description="随机种子")


class MultimodalGenerateRequest(BaseModel):
    """多模态生成请求"""
    prompt: str = Field(..., description="统一提示词")
    modalities: List[str] = Field(default=["image", "audio"], description="需要生成的模态")
    providers: Dict[str, str] = Field(default_factory=dict, description="各模态提供商")


class MultimodalUnderstandRequest(BaseModel):
    """多模态理解请求"""
    text: Optional[str] = Field(default=None, description="输入文本")
    images: Optional[List[str]] = Field(default=None, description="图片base64列表")
    audio: Optional[str] = Field(default=None, description="音频base64")
    video: Optional[str] = Field(default=None, description="视频base64")
    question: str = Field(..., description="问题")


# ============ 响应模型 ============

class GenerationResponse(BaseModel):
    """生成响应"""
    success: bool
    task_id: str
    provider: str
    model: str
    error: Optional[str] = None
    elapsed_ms: float = 0
    data: Optional[Dict[str, Any]] = None


class ImageGenerationResponse(BaseModel):
    """图像生成响应"""
    success: bool
    task_id: str
    provider: str
    model: str
    images: List[str] = []  # base64 images
    image_urls: List[str] = []
    size: Optional[str] = None
    seed: Optional[int] = None
    error: Optional[str] = None
    elapsed_ms: float = 0


class AudioGenerationResponse(BaseModel):
    """音频生成响应"""
    success: bool
    task_id: str
    provider: str
    model: str
    audio_data: Optional[str] = None  # base64 audio
    audio_url: Optional[str] = None
    duration_ms: Optional[float] = None
    format: Optional[str] = None
    characters: Optional[int] = None
    error: Optional[str] = None
    elapsed_ms: float = 0


class VideoGenerationResponse(BaseModel):
    """视频生成响应"""
    success: bool
    task_id: str
    provider: str
    model: str
    video_url: Optional[str] = None
    thumbnail_url: Optional[str] = None
    duration_ms: Optional[float] = None
    resolution: Optional[str] = None
    fps: Optional[int] = None
    error: Optional[str] = None
    elapsed_ms: float = 0


class ModelInfo(BaseModel):
    """模型信息"""
    id: str
    name: str
    provider: str
    type: str
    capabilities: List[str] = []


class GenerationHistoryItem(BaseModel):
    """生成历史项"""
    task_id: str
    generation_type: str
    provider: str
    model: str
    prompt: str
    status: str
    created_at: datetime
    completed_at: Optional[datetime] = None
    duration_ms: float = 0


# ============ API端点 ============

@router.post("/image", response_model=ImageGenerationResponse)
async def generate_image(request: ImageGenerateRequest):
    """
    生成图像
    
    - **prompt**: 图像描述提示词
    - **model**: 模型名称 (dalle, stable_diffusion)
    - **size**: 图像尺寸
    - **num_images**: 生成图像数量 (1-4)
    """
    import time
    start_time = time.time()
    
    manager = get_generation_manager()
    
    try:
        # 构建参数字典
        params = {
            "size": request.size.value if hasattr(request.size, 'value') else request.size,
            "quality": request.quality,
            "negative_prompt": request.negative_prompt,
            "num_images": request.num_images,
            "seed": request.seed
        }
        
        if request.style:
            params["style"] = request.style.value if hasattr(request.style, 'value') else request.style
        
        # 生成图像
        response = await manager.generate(
            modality="image",
            prompt=request.prompt,
            provider=request.model or "dalle",
            **params
        )
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        return ImageGenerationResponse(
            success=response.success,
            task_id=response.task_id,
            provider=response.provider,
            model=response.model,
            images=getattr(response, 'images', []),
            image_urls=getattr(response, 'image_urls', []),
            size=getattr(response, 'size', None),
            seed=getattr(response, 'seed', None),
            error=response.error,
            elapsed_ms=elapsed_ms
        )
        
    except Exception as e:
        return ImageGenerationResponse(
            success=False,
            task_id=str(uuid.uuid4()),
            provider=request.model or "unknown",
            model="unknown",
            error=str(e)
        )


@router.post("/audio", response_model=AudioGenerationResponse)
async def generate_audio(request: AudioGenerateRequest):
    """
    合成语音
    
    - **text**: 要转换的文字
    - **model**: TTS模型 (openai, azure, elevenlabs)
    - **voice**: 声音ID
    - **speed**: 语速 (0.5-2.0)
    - **format**: 音频格式
    """
    import time
    start_time = time.time()
    
    manager = get_generation_manager()
    
    try:
        # 构建参数字典
        params = {
            "voice": request.voice,
            "speed": request.speed,
            "pitch": request.pitch,
            "volume": request.volume,
            "format": request.format.value if hasattr(request.format, 'value') else request.format
        }
        
        # 生成音频
        response = await manager.generate(
            modality="audio",
            prompt=request.text,
            provider=request.model or "openai",
            **params
        )
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        return AudioGenerationResponse(
            success=response.success,
            task_id=response.task_id,
            provider=response.provider,
            model=response.model,
            audio_data=getattr(response, 'audio_data', None),
            audio_url=getattr(response, 'audio_path', None),
            duration_ms=getattr(response, 'duration_ms', None),
            format=getattr(response, 'format', None),
            characters=getattr(response, 'characters', None),
            error=response.error,
            elapsed_ms=elapsed_ms
        )
        
    except Exception as e:
        return AudioGenerationResponse(
            success=False,
            task_id=str(uuid.uuid4()),
            provider=request.model or "unknown",
            model="unknown",
            error=str(e)
        )


@router.post("/video", response_model=VideoGenerationResponse)
async def generate_video(request: VideoGenerateRequest):
    """
    生成视频
    
    - **prompt**: 视频描述
    - **model**: 视频模型 (sora, runway, stable_video)
    - **duration**: 视频时长
    - **resolution**: 视频分辨率
    - **fps**: 帧率
    """
    import time
    start_time = time.time()
    
    manager = get_generation_manager()
    
    try:
        # 构建参数字典
        params = {
            "negative_prompt": request.negative_prompt,
            "duration": request.duration.value if hasattr(request.duration, 'value') else request.duration,
            "resolution": request.resolution.value if hasattr(request.resolution, 'value') else request.resolution,
            "aspect_ratio": request.aspect_ratio.value if hasattr(request.aspect_ratio, 'value') else request.aspect_ratio,
            "style": request.style.value if hasattr(request.style, 'value') else request.style,
            "fps": request.fps,
            "seed": request.seed
        }
        
        # 生成视频
        response = await manager.generate(
            modality="video",
            prompt=request.prompt,
            provider=request.model or "sora",
            **params
        )
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        return VideoGenerationResponse(
            success=response.success,
            task_id=response.task_id,
            provider=response.provider,
            model=response.model,
            video_url=getattr(response, 'video_url', None),
            thumbnail_url=getattr(response, 'thumbnail_url', None),
            duration_ms=getattr(response, 'duration_ms', None),
            resolution=getattr(response, 'resolution', None),
            fps=getattr(response, 'fps', None),
            error=response.error,
            elapsed_ms=elapsed_ms
        )
        
    except Exception as e:
        return VideoGenerationResponse(
            success=False,
            task_id=str(uuid.uuid4()),
            provider=request.model or "unknown",
            model="unknown",
            error=str(e)
        )


@router.post("/multimodal")
async def generate_multimodal(request: MultimodalGenerateRequest):
    """
    多模态生成
    
    - **prompt**: 统一提示词
    - **modalities**: 需要生成的模态列表
    - **providers**: 各模态提供商映射
    """
    import time
    start_time = time.time()
    
    manager = get_generation_manager()
    
    try:
        results = await manager.generate_multimodal(
            prompt=request.prompt,
            modalities=request.modalities,
            providers=request.providers
        )
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        # 格式化结果
        formatted_results = {}
        for modality, response in results.items():
            formatted_results[modality] = {
                "success": response.success,
                "task_id": response.task_id,
                "provider": response.provider,
                "model": response.model,
                "error": response.error,
                "data": getattr(response, 'images', None) or 
                       getattr(response, 'audio_data', None) or
                       getattr(response, 'video_url', None)
            }
        
        return {
            "success": all(r.success for r in results.values()),
            "results": formatted_results,
            "elapsed_ms": elapsed_ms
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "elapsed_ms": (time.time() - start_time) * 1000
        }


@router.get("/models", response_model=List[ModelInfo])
async def get_available_models():
    """
    获取可用模型列表
    """
    manager = get_generation_manager()
    
    try:
        models = await manager.get_available_models()
        
        all_models = []
        for modality, model_list in models.items():
            for model in model_list:
                all_models.append(ModelInfo(
                    id=model["id"],
                    name=model["name"],
                    provider=model["provider"],
                    type=modality,
                    capabilities=[]
                ))
        
        return all_models
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history", response_model=List[GenerationHistoryItem])
async def get_generation_history(
    modality: Optional[str] = Query(default=None, description="筛选模态类型"),
    status: Optional[str] = Query(default=None, description="筛选状态"),
    limit: int = Query(default=50, ge=1, le=100, description="返回数量限制")
):
    """
    获取生成历史
    """
    unified_service = get_unified_service()
    
    try:
        # 获取任务列表
        task_status = TaskStatus(status) if status else None
        gen_type = GenerationType(modality) if modality else None
        
        tasks = unified_service.list_tasks(
            status=task_status,
            generation_type=gen_type,
            limit=limit
        )
        
        # 转换为历史项
        history = []
        for task in tasks:
            history.append(GenerationHistoryItem(
                task_id=task.task_id,
                generation_type=task.generation_type.value,
                provider=task.provider,
                model=getattr(task.request, 'model', 'unknown') if task.request else 'unknown',
                prompt=getattr(task.request, 'prompt', '') if task.request else '',
                status=task.status.value,
                created_at=task.created_at,
                completed_at=task.completed_at,
                duration_ms=task.elapsed_ms
            ))
        
        return history
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/{modality}")
async def get_models_by_modality(modality: str):
    """
    获取指定模态的可用模型
    """
    manager = get_generation_manager()
    
    valid_modalities = ["image", "audio", "video"]
    modality = modality.lower()
    
    if modality not in valid_modalities:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的模态类型: {modality}"
        )
    
    try:
        models = await manager.get_available_models()
        return models.get(modality, [])
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cost/estimate")
async def estimate_cost(
    modality: str = Query(..., description="模态类型"),
    provider: str = Query(..., description="提供商"),
    params: Dict[str, Any] = None
):
    """
    估算生成成本
    """
    manager = get_generation_manager()
    
    try:
        result = await manager.estimate_cost(
            modality=modality,
            provider=provider,
            params=params or {}
        )
        
        if not result.get("estimated", False):
            raise HTTPException(
                status_code=400,
                detail=result.get("message", "无法估算成本")
            )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "service": "generation",
        "timestamp": datetime.utcnow().isoformat()
    }
