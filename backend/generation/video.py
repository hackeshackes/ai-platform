"""
视频生成模块
支持视频生成、编辑和增强功能
"""

import os
import uuid
import asyncio
import base64
from typing import Optional, Dict, Any, List, Tuple
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from dataclasses import field

from .base import (
    BaseGenerationModel,
    GenerationProvider,
    GenerationRequest,
    GenerationResponse
)


class VideoDuration(str, Enum):
    """视频时长枚举"""
    SHORT = "3-5秒"
    MEDIUM = "5-10秒"
    LONG = "10-15秒"
    EXTENDED = "15-30秒"


class VideoResolution(str, Enum):
    """视频分辨率枚举"""
    SD = "576x1024"  # 9:16 竖屏
    HD = "720x1280"  # 720p
    FHD = "1080x1920"  # 1080p
    UHD = "2160x3840"  # 4K


class VideoAspectRatio(str, Enum):
    """视频宽高比枚举"""
    RATIO_9_16 = "9:16"  # 竖屏短视频
    RATIO_16_9 = "16:9"  # 横屏
    RATIO_1_1 = "1:1"    # 方形
    RATIO_4_3 = "4:3"    # 传统
    RATIO_3_4 = "3:4"    # 竖屏


class VideoStyle(str, Enum):
    """视频风格枚举"""
    REALISTIC = "realistic"
    CINEMATIC = "cinematic"
    ANIMATED = "animated"
    ABSTRACT = "abstract"
    DOCUMENTARY = "documentary"
    COMMERCIAL = "commercial"
    MUSIC_VIDEO = "music_video"


@dataclass
class VideoGenerationRequest(GenerationRequest):
    """视频生成请求"""
    prompt: str = ""
    negative_prompt: Optional[str] = None
    duration: VideoDuration = VideoDuration.MEDIUM
    resolution: VideoResolution = VideoResolution.HD
    aspect_ratio: VideoAspectRatio = VideoAspectRatio.RATIO_9_16
    style: VideoStyle = VideoStyle.CINEMATIC
    motion_bucket_id: Optional[int] = None
    fps: int = 24
    seed: Optional[int] = None
    guidance_scale: float = 1.0
    num_frames: Optional[int] = None


@dataclass
class VideoGenerationResponse(GenerationResponse):
    """视频生成响应"""
    video_data: Optional[bytes] = None  # 视频数据
    video_path: Optional[str] = None  # 视频文件路径
    video_url: Optional[str] = None  # 视频URL
    thumbnail_url: Optional[str] = None  # 缩略图URL
    duration_ms: Optional[float] = None  # 视频时长
    resolution: Optional[str] = None  # 分辨率
    fps: Optional[int] = None  # 帧率
    frame_count: Optional[int] = None  # 帧数
    video_size: Optional[int] = None  # 视频大小（字节）


@dataclass
class VideoEditRequest(GenerationRequest):
    """视频编辑请求"""
    video_path: str = ""
    operation: str = ""  # "trim", "crop", "overlay", "effect", "speed"
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VideoEditResponse(GenerationResponse):
    """视频编辑响应"""
    video_data: Optional[bytes] = None
    video_path: Optional[str] = None
    original_duration_ms: Optional[float] = None
    edited_duration_ms: Optional[float] = None


class BaseVideoGenerator(BaseGenerationModel):
    """视频生成器基类"""
    
    def __init__(self):
        super().__init__()
        self.supported_extensions = [".mp4", ".mov", ".avi", ".webm"]
    
    async def generate(
        self,
        request: VideoGenerationRequest
    ) -> VideoGenerationResponse:
        """生成视频"""
        raise NotImplementedError
    
    async def edit(
        self,
        request: VideoEditRequest
    ) -> VideoEditResponse:
        """编辑视频"""
        raise NotImplementedError
    
    async def enhance(
        self,
        video_data: bytes,
        enhancement_type: str
    ) -> VideoGenerationResponse:
        """增强视频质量"""
        raise NotImplementedError


class SoraVideoGenerator(BaseVideoGenerator):
    """OpenAI Sora视频生成器"""
    
    def __init__(
        self,
        api_key: str = None,
        model: str = "sora-1.0"
    ):
        super().__init__()
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.provider = GenerationProvider.OPENAI
    
    async def generate(
        self,
        request: VideoGenerationRequest
    ) -> VideoGenerationResponse:
        """使用Sora生成视频"""
        try:
            import openai
            client = openai.AsyncOpenAI(api_key=self.api_key)
            
            # 构建提示词
            enhanced_prompt = f"{request.prompt}, cinematic style, high quality"
            if request.negative_prompt:
                enhanced_prompt += f", no {request.negative_prompt}"
            
            # 解析时长
            duration_map = {
                VideoDuration.SHORT: 3,
                VideoDuration.MEDIUM: 5,
                VideoDuration.LONG: 10,
                VideoDuration.EXTENDED: 15
            }
            duration_seconds = duration_map.get(request.duration, 5)
            
            # 构建生成参数
            params = {
                "model": self.model,
                "prompt": enhanced_prompt,
                "duration": duration_seconds,
                "aspect_ratio": request.aspect_ratio.value
            }
            
            # 调用API (实际API可能不同，这里为示例)
            response = await client.video.generations.create(**params)
            
            # 处理响应
            return VideoGenerationResponse(
                success=True,
                task_id=str(uuid.uuid4()),
                provider=self.provider.value,
                model=self.model,
                video_url=response.data[0].url if hasattr(response.data[0], 'url') else None,
                duration_ms=duration_seconds * 1000,
                resolution=request.resolution.value,
                fps=request.fps
            )
            
        except Exception as e:
            return VideoGenerationResponse(
                success=False,
                error=str(e),
                provider=self.provider.value,
                model=self.model
            )
    
    async def edit(
        self,
        request: VideoEditRequest
    ) -> VideoEditResponse:
        """使用Sora编辑视频"""
        try:
            import openai
            client = openai.AsyncOpenAI(api_key=self.api_key)
            
            response = await client.video.edits.create(
                model=self.model,
                video=request.video_path,
                operation=request.operation,
                **request.params
            )
            
            return VideoEditResponse(
                success=True,
                task_id=str(uuid.uuid4()),
                provider=self.provider.value,
                model=self.model,
                video_data=response.content
            )
            
        except Exception as e:
            return VideoEditResponse(
                success=False,
                error=str(e),
                provider=self.provider.value,
                model=self.model
            )


class RunwayVideoGenerator(BaseVideoGenerator):
    """Runway视频生成器"""
    
    def __init__(
        self,
        api_key: str = None,
        model: str = "gen-2"
    ):
        super().__init__()
        self.api_key = api_key or os.getenv("RUNWAY_API_KEY")
        self.model = model
        self.provider = GenerationProvider.RUNWAY
        self.base_url = "https://api.runwayml.com/v1"
    
    async def generate(
        self,
        request: VideoGenerationRequest
    ) -> VideoGenerationResponse:
        """使用Runway生成视频"""
        try:
            import aiohttp
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # 构建生成参数
            payload = {
                "prompt": request.prompt,
                "negative_prompt": request.negative_prompt,
                "model_id": self.model,
                "aspect_ratio": request.aspect_ratio.value,
                "motion": request.motion_bucket_id or 5,
                "seed": request.seed
            }
            
            async with aiohttp.ClientSession() as session:
                # 创建生成任务
                async with session.post(
                    f"{self.base_url}/generations/video",
                    json=payload,
                    headers=headers
                ) as resp:
                    if resp.status != 200:
                        error = await resp.text()
                        return VideoGenerationResponse(
                            success=False,
                            error=f"创建任务失败: {error}",
                            provider=self.provider.value,
                            model=self.model
                        )
                    
                    task = await resp.json()
                    task_id = task["id"]
                
                # 轮询任务状态
                while True:
                    async with session.get(
                        f"{self.base_url}/generations/{task_id}",
                        headers=headers
                    ) as resp:
                        status = await resp.json()
                        
                        if status["status"] == "completed":
                            return VideoGenerationResponse(
                                success=True,
                                task_id=task_id,
                                provider=self.provider.value,
                                model=self.model,
                                video_url=status["output"]["url"],
                                duration_ms=status.get("duration", 0) * 1000
                            )
                        elif status["status"] == "failed":
                            return VideoGenerationResponse(
                                success=False,
                                error=status.get("error", "生成失败"),
                                provider=self.provider.value,
                                model=self.model
                            )
                        elif status["status"] in ["pending", "running"]:
                            await asyncio.sleep(2)
                            continue
                        else:
                            return VideoGenerationResponse(
                                success=False,
                                error=f"未知状态: {status['status']}",
                                provider=self.provider.value,
                                model=self.model
                            )
                            
        except Exception as e:
            return VideoGenerationResponse(
                success=False,
                error=str(e),
                provider=self.provider.value,
                model=self.model
            )
    
    async def enhance(
        self,
        video_data: bytes,
        enhancement_type: str = "upscale"
    ) -> VideoGenerationResponse:
        """增强视频质量"""
        try:
            import aiohttp
            
            headers = {
                "Authorization": f"Bearer {self.api_key}"
            }
            
            form_data = aiohttp.FormData()
            form_data.add_field(
                "video",
                video_data,
                filename="video.mp4",
                content_type="video/mp4"
            )
            form_data.add_field("type", enhancement_type)
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/enhancements/video",
                    data=form_data,
                    headers=headers
                ) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        
                        return VideoGenerationResponse(
                            success=True,
                            task_id=str(uuid.uuid4()),
                            provider=self.provider.value,
                            model=f"enhance-{enhancement_type}",
                            video_url=result["output"]["url"]
                        )
                    else:
                        return VideoGenerationResponse(
                            success=False,
                            error=f"增强失败",
                            provider=self.provider.value,
                            model=f"enhance-{enhancement_type}"
                        )
                        
        except Exception as e:
            return VideoGenerationResponse(
                success=False,
                error=str(e),
                provider=self.provider.value,
                model=f"enhance-{enhancement_type}"
            )


class StableVideoDiffusionGenerator(BaseVideoGenerator):
    """Stable Video Diffusion生成器"""
    
    def __init__(
        self,
        api_url: str = None,
        model: str = "stable-video-diffusion"
    ):
        super().__init__()
        self.api_url = api_url or os.getenv("STABLE_VIDEO_API_URL")
        self.model = model
        self.provider = GenerationProvider.STABILITY
    
    async def generate(
        self,
        request: VideoGenerationRequest
    ) -> VideoGenerationResponse:
        """使用Stable Video Diffusion生成视频"""
        try:
            import aiohttp
            import base64
            
            headers = {
                "Authorization": f"Bearer {os.getenv('STABLE_VIDEO_TOKEN', '')}"
            }
            
            payload = {
                "input": {
                    "prompt": request.prompt,
                    "negative_prompt": request.negative_prompt or "",
                    "video_length": request.num_frames or 25,
                    "fps": request.fps,
                    "motion_bucket_id": request.motion_bucket_id or 127,
                    "cfg_scale": request.guidance_scale,
                    "width": int(request.resolution.value.split("x")[0]),
                    "height": int(request.resolution.value.split("x")[1])
                }
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.api_url}/v1/generation/{self.model}/text-to-video",
                    json=payload,
                    headers=headers
                ) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        
                        # 处理返回的视频数据
                        video_data = None
                        if "video" in result:
                            video_data = base64.b64decode(result["video"])
                        
                        return VideoGenerationResponse(
                            success=True,
                            task_id=str(uuid.uuid4()),
                            provider=self.provider.value,
                            model=self.model,
                            video_data=video_data,
                            fps=request.fps,
                            frame_count=payload["input"]["video_length"]
                        )
                    else:
                        error = await resp.text()
                        return VideoGenerationResponse(
                            success=False,
                            error=f"生成失败: {error}",
                            provider=self.provider.value,
                            model=self.model
                        )
                        
        except Exception as e:
            return VideoGenerationResponse(
                success=False,
                error=str(e),
                provider=self.provider.value,
                model=self.model
            )
    
    async def img2video(
        self,
        image_data: bytes,
        prompt: str = "",
        motion_bucket_id: int = 127
    ) -> VideoGenerationResponse:
        """图生视频"""
        try:
            import aiohttp
            import base64
            
            headers = {
                "Authorization": f"Bearer {os.getenv('STABLE_VIDEO_TOKEN', '')}"
            }
            
            payload = {
                "input": {
                    "image": base64.b64encode(image_data).decode("utf-8"),
                    "prompt": prompt,
                    "motion_bucket_id": motion_bucket_id
                }
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.api_url}/v1/generation/{self.model}/image-to-video",
                    json=payload,
                    headers=headers
                ) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        
                        return VideoGenerationResponse(
                            success=True,
                            task_id=str(uuid.uuid4()),
                            provider=self.provider.value,
                            model=self.model
                        )
                    else:
                        return VideoGenerationResponse(
                            success=False,
                            error=f"图生视频失败",
                            provider=self.provider.value,
                            model=self.model
                        )
                        
        except Exception as e:
            return VideoGenerationResponse(
                success=False,
                error=str(e),
                provider=self.provider.value,
                model=self.model
            )


class VideoGenerationEngine:
    """视频生成引擎 - 管理所有视频生成器"""
    
    def __init__(self):
        self.generators: Dict[str, BaseVideoGenerator] = {}
        self._register_default_generators()
    
    def _register_default_generators(self):
        """注册默认生成器"""
        self.generators["sora"] = SoraVideoGenerator()
        self.generators["runway"] = RunwayVideoGenerator()
        self.generators["stable_video"] = StableVideoDiffusionGenerator()
    
    def register_generator(self, name: str, generator: BaseVideoGenerator):
        """注册自定义生成器"""
        self.generators[name] = generator
    
    def get_generator(self, name: str) -> Optional[BaseVideoGenerator]:
        """获取生成器"""
        return self.generators.get(name)
    
    async def generate(
        self,
        provider: str,
        request: VideoGenerationRequest
    ) -> VideoGenerationResponse:
        """生成视频"""
        generator = self.get_generator(provider)
        if not generator:
            return VideoGenerationResponse(
                success=False,
                error=f"未找到视频生成提供器: {provider}"
            )
        return await generator.generate(request)
    
    async def img2video(
        self,
        provider: str,
        image_data: bytes,
        prompt: str = ""
    ) -> VideoGenerationResponse:
        """图生视频"""
        generator = self.get_generator(provider)
        if not generator:
            return VideoGenerationResponse(
                success=False,
                error=f"未找到视频生成提供器: {provider}"
            )
        
        if hasattr(generator, 'img2video'):
            return await generator.img2video(image_data, prompt)
        
        return VideoGenerationResponse(
            success=False,
            error=f"该提供器不支持图生视频",
            provider=provider
        )
    
    async def enhance(
        self,
        provider: str,
        video_data: bytes,
        enhancement_type: str = "upscale"
    ) -> VideoGenerationResponse:
        """增强视频"""
        generator = self.get_generator(provider)
        if not generator:
            return VideoGenerationResponse(
                success=False,
                error=f"未找到视频生成提供器: {provider}"
            )
        
        if hasattr(generator, 'enhance'):
            return await generator.enhance(video_data, enhancement_type)
        
        return VideoGenerationResponse(
            success=False,
            error=f"该提供器不支持视频增强",
            provider=provider
        )


# 全局引擎实例
_video_engine: Optional[VideoGenerationEngine] = None


def get_video_engine() -> VideoGenerationEngine:
    """获取视频生成引擎"""
    global _video_engine
    if _video_engine is None:
        _video_engine = VideoGenerationEngine()
    return _video_engine


async def create_video_generator(
    provider: str,
    **kwargs
) -> BaseVideoGenerator:
    """工厂函数：创建视频生成器"""
    engine = get_video_engine()
    
    if provider == "sora" or provider == "openai":
        return SoraVideoGenerator(**kwargs)
    elif provider == "runway":
        return RunwayVideoGenerator(**kwargs)
    elif provider == "stable_video" or provider == "stability":
        return StableVideoDiffusionGenerator(**kwargs)
    else:
        generator = engine.get_generator(provider)
        if generator:
            return generator
        raise ValueError(f"不支持的视频生成提供器: {provider}")
