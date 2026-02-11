"""
多模态API端点
提供图像、音频、视频分析和多模态生成的REST API
"""

import base64
import os
from typing import Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends
from pydantic import BaseModel
from pathlib import Path

from multimodal.base import (
    BaseMultimodalEngine,
    BaseVisionModel,
    BaseAudioModel,
    BaseVideoModel,
    MultimodalProvider
)
from multimodal.vision import (
    create_vision_model,
    OpenAIVisionModel,
    AnthropicVisionModel,
    GoogleVisionModel,
    DeepSeekVLModel
)
from multimodal.audio import (
    create_audio_model,
    OpenAIAudioModel,
    GoogleAudioModel
)
from multimodal.video import (
    create_video_model,
    OpenAIVideoModel,
    GoogleVideoModel
)


router = APIRouter(prefix="/multimodal", tags=["multimodal"])

# 全局多模态引擎实例
_engine: Optional[BaseMultimodalEngine] = None


def get_engine() -> BaseMultimodalEngine:
    """获取多模态引擎实例"""
    global _engine
    if _engine is None:
        _engine = BaseMultimodalEngine()
        _engine.register_vision_model("openai", OpenAIVisionModel())
        _engine.register_vision_model("anthropic", AnthropicVisionModel())
        _engine.register_vision_model("google", GoogleVisionModel())
        _engine.register_vision_model("deepseek", DeepSeekVLModel())
        
        _engine.register_audio_model("openai", OpenAIAudioModel())
        _engine.register_audio_model("google", GoogleAudioModel())
        
        _engine.register_video_model("openai", OpenAIVideoModel())
        _engine.register_video_model("google", GoogleVideoModel())
    return _engine


# ============ 请求/响应模型 ============

class ImageAnalysisRequest(BaseModel):
    """图像分析请求"""
    prompt: str = "请描述这张图片的内容"
    provider: str = "openai"
    model: Optional[str] = None


class ImageAnalysisResponse(BaseModel):
    """图像分析响应"""
    success: bool
    content: Optional[str] = None
    provider: str
    model: str
    error: Optional[str] = None


class AudioTranscribeRequest(BaseModel):
    """音频转录请求"""
    language: Optional[str] = "zh-CN"
    provider: str = "openai"
    model: Optional[str] = None


class AudioTranscribeResponse(BaseModel):
    """音频转录响应"""
    success: bool
    content: Optional[str] = None
    provider: str
    model: str
    error: Optional[str] = None


class VideoAnalysisRequest(BaseModel):
    """视频分析请求"""
    prompt: Optional[str] = None
    num_frames: int = 10
    provider: str = "openai"
    model: Optional[str] = None


class VideoAnalysisResponse(BaseModel):
    """视频分析响应"""
    success: bool
    content: Optional[str] = None
    provider: str
    model: str
    error: Optional[str] = None


class GenerateRequest(BaseModel):
    """多模态生成请求"""
    prompt: str
    media_type: str  # image, audio, video
    provider: str = "openai"
    model: Optional[str] = None


class GenerateResponse(BaseModel):
    """多模态生成响应"""
    success: bool
    content: Optional[str] = None
    provider: str
    model: str
    error: Optional[str] = None


# ============ API端点 ============

@router.post("/image", response_model=ImageAnalysisResponse)
async def analyze_image(
    file: UploadFile = File(...),
    prompt: str = Form("请描述这张图片的内容"),
    provider: str = Form("openai"),
    model: Optional[str] = Form(None),
    engine: BaseMultimodalEngine = Depends(get_engine)
):
    """
    图像理解API
    
    上传图片并获取AI分析结果
    
    - **file**: 图片文件 (支持 jpg, png, webp 等格式)
    - **prompt**: 分析提示词
    - **provider**: 提供商 (openai, anthropic, google, deepseek)
    - **model**: 模型名称 (可选)
    """
    try:
        contents = await file.read()
        
        vision_model = create_vision_model(provider, model)
        result = await vision_model.analyze_image(contents, prompt)
        
        return ImageAnalysisResponse(
            success=result.success,
            content=result.content,
            provider=result.provider,
            model=result.model,
            error=result.error
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/audio", response_model=AudioTranscribeResponse)
async def transcribe_audio(
    file: UploadFile = File(...),
    language: str = Form("zh-CN"),
    provider: str = Form("openai"),
    model: Optional[str] = Form(None),
    engine: BaseMultimodalEngine = Depends(get_engine)
):
    """
    语音转文字API
    
    上传音频文件并获取转录结果
    
    - **file**: 音频文件 (支持 mp3, wav, m4a, ogg 等格式)
    - **language**: 语言代码 (如 zh-CN, en-US)
    - **provider**: 提供商 (openai, google)
    - **model**: 模型名称 (可选)
    """
    try:
        contents = await file.read()
        
        audio_model = create_audio_model(provider, model)
        result = await audio_model.transcribe(contents, language=language)
        
        return AudioTranscribeResponse(
            success=result.success,
            content=result.content,
            provider=result.provider,
            model=result.model,
            error=result.error
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/video", response_model=VideoAnalysisResponse)
async def analyze_video(
    file: UploadFile = File(...),
    prompt: str = Form(None),
    num_frames: int = Form(10),
    provider: str = Form("openai"),
    model: Optional[str] = Form(None),
    engine: BaseMultimodalEngine = Depends(get_engine)
):
    """
    视频分析API
    
    上传视频文件并获取AI分析结果
    
    - **file**: 视频文件 (支持 mp4, avi, mov 等格式)
    - **prompt**: 分析提示词 (可选)
    - **num_frames**: 提取的关键帧数量 (默认10)
    - **provider**: 提供商 (openai, google)
    - **model**: 模型名称 (可选)
    """
    try:
        contents = await file.read()
        
        # 保存临时文件
        temp_path = Path(f"/tmp/{file.filename}")
        temp_path.write_bytes(contents)
        
        video_model = create_video_model(provider, model)
        result = await video_model.analyze_video(
            temp_path,
            prompt=prompt,
            num_frames=num_frames
        )
        
        # 清理临时文件
        temp_path.unlink(missing_ok=True)
        
        return VideoAnalysisResponse(
            success=result.success,
            content=result.content,
            provider=result.provider,
            model=result.model,
            error=result.error
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate", response_model=GenerateResponse)
async def generate_content(request: GenerateRequest):
    """
    多模态内容生成API
    
    根据提示生图像、音频或视频内容
    
    - **prompt**: 生成提示词
    - **media_type**: 生成类型 (image, audio, video)
    - **provider**: 提供商 (openai, google)
    - **model**: 模型名称 (可选)
    """
    try:
        # 目前主要支持图像生成
        if request.media_type == "image":
            from multimodal.generators import ImageGenerator
            generator = ImageGenerator(provider=request.provider, model=request.model)
            result = await generator.generate(request.prompt)
        else:
            raise NotImplementedError(
                f"暂不支持 {request.media_type} 类型的生成"
            )
        
        return GenerateResponse(
            success=result.success,
            content=result.content,
            provider=result.provider,
            model=result.model,
            error=result.error
        )
    except NotImplementedError as e:
        raise HTTPException(status_code=501, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============ 辅助端点 ============

@router.get("/providers")
async def list_providers():
    """列出支持的提供商"""
    return {
        "vision": {
            "openai": {"models": ["gpt-4o", "gpt-4o-mini"]},
            "anthropic": {"models": ["claude-3-5-sonnet-20241022", "claude-3-opus-20240229"]},
            "google": {"models": ["gemini-1.5-flash", "gemini-1.5-pro"]},
            "deepseek": {"models": ["deepseek-vl"]}
        },
        "audio": {
            "openai": {"models": ["whisper-1"]},
            "google": {"models": ["latest"]}
        },
        "video": {
            "openai": {"models": ["gpt-4o"]},
            "google": {"models": ["gemini-1.5-flash"]}
        }
    }


@router.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy", "service": "multimodal"}
