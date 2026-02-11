"""
图像生成模块
支持DALL-E、Stable Diffusion等图像生成模型
"""

import base64
import os
import uuid
from typing import Optional, Dict, Any, List
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
from PIL import Image
from io import BytesIO

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    aiohttp = None

from .base import (
    BaseGenerationModel,
    GenerationProvider,
    GenerationTask,
    TaskStatus,
    GenerationRequest,
    GenerationResponse
)


class ImageSize(str, Enum):
    """图像尺寸枚举"""
    SQUARE = "1024x1024"
    PORTRAIT = "1024x1792"
    LANDSCAPE = "1792x1024"
    SMALL = "512x512"
    MEDIUM = "768x768"


class ImageStyle(str, Enum):
    """图像风格枚举"""
    REALISTIC = "realistic"
    ARTISTIC = "artistic"
    ANIME = "anime"
    ABSTRACT = "abstract"
    PHOTOGRAPHIC = "photographic"
    OIL_PAINTING = "oil_painting"
    WATERCOLOR = "watercolor"
    SKETCH = "sketch"


@dataclass
class ImageGenerationRequest(GenerationRequest):
    """图像生成请求"""
    prompt: str = ""
    size: ImageSize = ImageSize.SQUARE
    style: Optional[ImageStyle] = None
    quality: str = "standard"  # standard, hd
    negative_prompt: Optional[str] = None
    num_images: int = 1
    seed: Optional[int] = None


@dataclass
class ImageGenerationResponse(GenerationResponse):
    """图像生成响应"""
    images: List[str] = None  # base64编码的图像列表
    image_urls: List[str] = None  # 图像URL列表
    seed: Optional[int] = None
    size: Optional[str] = None
    style: Optional[str] = None


class BaseImageGenerator(BaseGenerationModel):
    """图像生成器基类"""
    
    def __init__(self):
        super().__init__()
        self.supported_providers = ["dalle", "stable_diffusion", "midjourney"]
    
    async def generate(
        self, 
        request: ImageGenerationRequest
    ) -> ImageGenerationResponse:
        """生成图像"""
        raise NotImplementedError
    
    async def validate_prompt(self, prompt: str) -> Dict[str, Any]:
        """验证并优化提示词"""
        # 基础验证
        if len(prompt) < 3:
            return {"valid": False, "message": "提示词太短"}
        
        if len(prompt) > 4000:
            return {"valid": False, "message": "提示词太长"}
        
        # 优化建议
        suggestions = []
        if "请" in prompt:
            suggestions.append("建议使用英文提示词以获得更好的效果")
        if "生成" in prompt:
            suggestions.append("可以直接描述图像内容，不需要'生成'等指令")
        
        return {
            "valid": True,
            "message": "提示词验证通过",
            "suggestions": suggestions
        }


class DalleImageGenerator(BaseImageGenerator):
    """DALL-E图像生成器"""
    
    def __init__(self, api_key: str = None, model: str = "dall-e-3"):
        super().__init__()
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.provider = GenerationProvider.OPENAI
    
    async def generate(
        self, 
        request: ImageGenerationRequest
    ) -> ImageGenerationResponse:
        """使用DALL-E生成图像"""
        try:
            import openai
            client = openai.AsyncOpenAI(api_key=self.api_key)
            
            # 构建生成参数
            params = {
                "model": self.model,
                "prompt": request.prompt,
                "n": min(request.num_images, 4),
                "size": request.size.value if isinstance(request.size, ImageSize) else request.size,
                "quality": request.quality,
                "response_format": "b64_json"
            }
            
            # 调用API
            response = await client.images.generate(**params)
            
            # 处理响应
            images = []
            image_urls = []
            for data in response.data:
                if data.b64_json:
                    images.append(data.b64_json)
                if data.url:
                    image_urls.append(data.url)
            
            return ImageGenerationResponse(
                success=True,
                task_id=str(uuid.uuid4()),
                provider=self.provider.value,
                model=self.model,
                images=images,
                image_urls=image_urls,
                seed=request.seed,
                size=params["size"],
                elapsed_ms=0  # 可从响应中获取
            )
            
        except Exception as e:
            return ImageGenerationResponse(
                success=False,
                error=str(e),
                provider=self.provider.value,
                model=self.model
            )
    
    async def edit(
        self,
        image: bytes,
        mask: Optional[bytes],
        prompt: str,
        size: ImageSize = ImageSize.SQUARE
    ) -> ImageGenerationResponse:
        """编辑图像"""
        try:
            import openai
            client = openai.AsyncOpenAI(api_key=self.api_key)
            
            response = await client.images.edit(
                model=self.model,
                image=image,
                mask=mask,
                prompt=prompt,
                n=1,
                size=size.value if isinstance(size, ImageSize) else size
            )
            
            images = [data.b64_json for data in response.data if data.b64_json]
            
            return ImageGenerationResponse(
                success=True,
                task_id=str(uuid.uuid4()),
                provider=self.provider.value,
                model=self.model,
                images=images
            )
            
        except Exception as e:
            return ImageGenerationResponse(
                success=False,
                error=str(e),
                provider=self.provider.value,
                model=self.model
            )
    
    async def variations(
        self,
        image: bytes,
        size: ImageSize = ImageSize.SQUARE,
        num_images: int = 1
    ) -> ImageGenerationResponse:
        """生成图像变体"""
        try:
            import openai
            client = openai.AsyncOpenAI(api_key=self.api_key)
            
            response = await client.images.create_variation(
                image=image,
                n=min(num_images, 4),
                size=size.value if isinstance(size, ImageSize) else size,
                response_format="b64_json"
            )
            
            images = [data.b64_json for data in response.data if data.b64_json]
            
            return ImageGenerationResponse(
                success=True,
                task_id=str(uuid.uuid4()),
                provider=self.provider.value,
                model=self.model,
                images=images
            )
            
        except Exception as e:
            return ImageGenerationResponse(
                success=False,
                error=str(e),
                provider=self.provider.value,
                model=self.model
            )


class StableDiffusionGenerator(BaseImageGenerator):
    """Stable Diffusion图像生成器"""
    
    def __init__(
        self,
        api_url: str = None,
        model: str = "stable-diffusion-xl",
        auth_token: str = None
    ):
        super().__init__()
        self.api_url = api_url or os.getenv("STABLE_DIFFUSION_API_URL")
        self.model = model
        self.auth_token = auth_token or os.getenv("STABLE_DIFFUSION_TOKEN")
        self.provider = GenerationProvider.STABILITY
    
    async def generate(
        self,
        request: ImageGenerationRequest
    ) -> ImageGenerationResponse:
        """使用Stable Diffusion生成图像"""
        try:
            async with aiohttp.ClientSession() as session:
                headers = {"Authorization": f"Bearer {self.auth_token}"}
                
                # 构建请求体
                payload = {
                    "prompt": request.prompt,
                    "negative_prompt": request.negative_prompt or "",
                    "width": int(request.size.value.split("x")[0]) if isinstance(request.size, ImageSize) else 1024,
                    "height": int(request.size.value.split("x")[1]) if isinstance(request.size, ImageSize) else 1024,
                    "steps": 30,
                    "cfg_scale": 7,
                    "samples": request.num_images,
                    "seed": request.seed or -1
                }
                
                async with session.post(
                    f"{self.api_url}/v1/generation/{self.model}/text-to-image",
                    json=payload,
                    headers=headers
                ) as resp:
                    response = await resp.json()
                
                # 处理响应
                images = []
                if "artifacts" in response:
                    for artifact in response["artifacts"]:
                        if artifact.get("type") == "BASE64":
                            images.append(artifact["base64"])
                
                return ImageGenerationResponse(
                    success=True,
                    task_id=str(uuid.uuid4()),
                    provider=self.provider.value,
                    model=self.model,
                    images=images,
                    seed=payload["seed"],
                    size=f"{payload['width']}x{payload['height']}"
                )
                
        except Exception as e:
            return ImageGenerationResponse(
                success=False,
                error=str(e),
                provider=self.provider.value,
                model=self.model
            )
    
    async def img2img(
        self,
        source_image: bytes,
        prompt: str,
        strength: float = 0.5,
        size: ImageSize = ImageSize.SQUARE
    ) -> ImageGenerationResponse:
        """图生图"""
        try:
            async with aiohttp.ClientSession() as session:
                headers = {"Authorization": f"Bearer {self.auth_token}"}
                
                payload = {
                    "init_images": [base64.b64encode(source_image).decode("utf-8")],
                    "prompt": prompt,
                    "denoising_strength": 1 - strength,
                    "width": int(size.value.split("x")[0]) if isinstance(size, ImageSize) else 1024,
                    "height": int(size.value.split("x")[1]) if isinstance(size, ImageSize) else 1024
                }
                
                async with session.post(
                    f"{self.api_url}/v1/generation/{self.model}/image-to-image",
                    json=payload,
                    headers=headers
                ) as resp:
                    response = await resp.json()
                
                images = []
                if "artifacts" in response:
                    for artifact in response["artifacts"]:
                        if artifact.get("type") == "BASE64":
                            images.append(artifact["base64"])
                
                return ImageGenerationResponse(
                    success=True,
                    task_id=str(uuid.uuid4()),
                    provider=self.provider.value,
                    model=self.model,
                    images=images
                )
                
        except Exception as e:
            return ImageGenerationResponse(
                success=False,
                error=str(e),
                provider=self.provider.value,
                model=self.model
            )


class ImageGenerationEngine:
    """图像生成引擎 - 管理所有图像生成器"""
    
    def __init__(self):
        self.generators: Dict[str, BaseImageGenerator] = {}
        self._register_default_generators()
    
    def _register_default_generators(self):
        """注册默认生成器"""
        # DALL-E
        self.generators["dalle"] = DalleImageGenerator()
        
        # Stable Diffusion
        self.generators["stable_diffusion"] = StableDiffusionGenerator()
    
    def register_generator(self, name: str, generator: BaseImageGenerator):
        """注册自定义生成器"""
        self.generators[name] = generator
    
    def get_generator(self, name: str) -> BaseImageGenerator:
        """获取生成器"""
        return self.generators.get(name)
    
    async def generate(
        self,
        provider: str,
        request: ImageGenerationRequest
    ) -> ImageGenerationResponse:
        """生成图像"""
        generator = self.get_generator(provider)
        if not generator:
            return ImageGenerationResponse(
                success=False,
                error=f"未找到提供器: {provider}"
            )
        return await generator.generate(request)
    
    async def compare_providers(
        self,
        prompt: str,
        providers: List[str] = None
    ) -> Dict[str, ImageGenerationResponse]:
        """比较不同提供器的生成结果"""
        if providers is None:
            providers = list(self.generators.keys())
        
        request = ImageGenerationRequest(
            prompt=prompt,
            num_images=1
        )
        
        results = {}
        for provider in providers:
            results[provider] = await self.generate(provider, request)
        
        return results


# 全局引擎实例
_image_engine: Optional[ImageGenerationEngine] = None


def get_image_engine() -> ImageGenerationEngine:
    """获取图像生成引擎"""
    global _image_engine
    if _image_engine is None:
        _image_engine = ImageGenerationEngine()
    return _image_engine


async def create_image_generator(
    provider: str,
    **kwargs
) -> BaseImageGenerator:
    """工厂函数：创建图像生成器"""
    engine = get_image_engine()
    
    if provider == "dalle" or provider == "openai":
        return DalleImageGenerator(**kwargs)
    elif provider == "stable_diffusion" or provider == "stability":
        return StableDiffusionGenerator(**kwargs)
    else:
        # 尝试从已注册生成器获取
        generator = engine.get_generator(provider)
        if generator:
            return generator
        raise ValueError(f"不支持的图像生成提供器: {provider}")
