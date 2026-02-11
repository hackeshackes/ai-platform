"""
统一多模态生成接口
提供对图像、音频、视频生成的统一访问
"""

from .base import (
    GenerationProvider,
    TaskStatus,
    GenerationType,
    GenerationRequest,
    GenerationResponse,
    GenerationTask,
    GenerationHistory,
    BaseGenerationModel,
    BaseMultimodalEngine,
    UnifiedGenerationService,
    get_unified_service
)

from .image import (
    ImageGenerationRequest,
    ImageGenerationResponse,
    ImageSize,
    ImageStyle,
    BaseImageGenerator,
    DalleImageGenerator,
    StableDiffusionGenerator,
    ImageGenerationEngine,
    get_image_engine,
    create_image_generator
)

from .audio import (
    TTSRequest,
    TTSResponse,
    VoiceGender,
    VoiceAccent,
    AudioFormat,
    VoiceConfig,
    BaseTTSEngine,
    OpenAITTSEngine,
    AzureTTSEngine,
    ElevenLabsTTSEngine,
    TTSEngineManager,
    get_tts_engine,
    create_tts_engine
)

from .video import (
    VideoGenerationRequest,
    VideoGenerationResponse,
    VideoEditRequest,
    VideoEditResponse,
    VideoDuration,
    VideoResolution,
    VideoAspectRatio,
    VideoStyle,
    BaseVideoGenerator,
    SoraVideoGenerator,
    RunwayVideoGenerator,
    StableVideoDiffusionGenerator,
    VideoGenerationEngine,
    get_video_engine,
    create_video_generator
)

__all__ = [
    # 枚举类型
    "GenerationProvider",
    "TaskStatus", 
    "GenerationType",
    "VoiceGender",
    "VoiceAccent",
    "AudioFormat",
    "ImageSize",
    "ImageStyle",
    "VideoDuration",
    "VideoResolution",
    "VideoAspectRatio",
    "VideoStyle",
    
    # 数据类
    "GenerationRequest",
    "GenerationResponse",
    "GenerationTask",
    "GenerationHistory",
    "ImageGenerationRequest",
    "ImageGenerationResponse",
    "TTSRequest",
    "TTSResponse",
    "VideoGenerationRequest",
    "VideoGenerationResponse",
    "VideoEditRequest",
    "VideoEditResponse",
    "VoiceConfig",
    
    # 基类
    "BaseGenerationModel",
    "BaseMultimodalEngine",
    "BaseImageGenerator",
    "BaseTTSEngine",
    "BaseVideoGenerator",
    
    # 实现类
    "DalleImageGenerator",
    "StableDiffusionGenerator",
    "ImageGenerationEngine",
    "OpenAITTSEngine",
    "AzureTTSEngine",
    "ElevenLabsTTSEngine",
    "TTSEngineManager",
    "SoraVideoGenerator",
    "RunwayVideoGenerator",
    "StableVideoDiffusionGenerator",
    "VideoGenerationEngine",
    "UnifiedGenerationService",
    
    # 工厂函数
    "get_image_engine",
    "create_image_generator",
    "get_tts_engine",
    "create_tts_engine",
    "get_video_engine",
    "create_video_generator",
    "get_unified_service"
]


class GenerationManager:
    """
    多模态生成管理器
    提供简化的API来访问所有生成功能
    """
    
    def __init__(self):
        self._image_engine = None
        self._audio_engine = None
        self._video_engine = None
        self._unified_service = None
    
    @property
    def image_engine(self):
        """获取图像生成引擎"""
        if self._image_engine is None:
            from .image import get_image_engine
            self._image_engine = get_image_engine()
        return self._image_engine
    
    @property
    def audio_engine(self):
        """获取音频生成引擎"""
        if self._audio_engine is None:
            from .audio import get_tts_engine
            self._audio_engine = get_tts_engine()
        return self._audio_engine
    
    @property
    def video_engine(self):
        """获取视频生成引擎"""
        if self._video_engine is None:
            from .video import get_video_engine
            self._video_engine = get_video_engine()
        return self._video_engine
    
    @property
    def unified_service(self):
        """获取统一生成服务"""
        if self._unified_service is None:
            from .base import get_unified_service
            self._unified_service = get_unified_service()
        return self._unified_service
    
    async def generate(
        self,
        modality: str,
        prompt: str,
        provider: str = "openai",
        **kwargs
    ) -> GenerationResponse:
        """
        统一的生成接口
        
        Args:
            modality: 生成模态 (image, audio, video)
            prompt: 提示词
            provider: 提供商
            **kwargs: 其他参数
        
        Returns:
            GenerationResponse: 生成结果
        """
        modality = modality.lower()
        
        if modality == "image":
            return await self.unified_service.generate_image(prompt, provider, **kwargs)
        elif modality == "audio":
            return await self.unified_service.generate_audio(prompt, provider, **kwargs)
        elif modality == "video":
            return await self.unified_service.generate_video(prompt, provider, **kwargs)
        else:
            return GenerationResponse(
                success=False,
                error=f"不支持的模态: {modality}"
            )
    
    async def generate_multimodal(
        self,
        prompt: str,
        modalities: list = None,
        providers: dict = None
    ) -> dict:
        """
        多模态生成
        
        Args:
            prompt: 提示词
            modalities: 需要生成的模态列表
            providers: 各模态的提供商映射
        
        Returns:
            dict: 各模态的生成结果
        """
        if modalities is None:
            modalities = ["image", "audio"]
        if providers is None:
            providers = {}
        
        return await self.unified_service.multimodal_generate(
            prompt,
            modalities,
            providers
        )
    
    async def get_available_models(self) -> dict:
        """获取所有可用模型"""
        models = {
            "image": [],
            "audio": [],
            "video": []
        }
        
        # 图像模型
        for name in ["dalle", "stable_diffusion"]:
            generator = self.image_engine.get_generator(name)
            if generator:
                models["image"].append({
                    "id": name,
                    "name": generator.model if hasattr(generator, 'model') else name,
                    "provider": generator.provider.value if hasattr(generator, 'provider') else name
                })
        
        # 音频模型
        for name, engine in self.audio_engine.engines.items():
            models["audio"].append({
                "id": name,
                "name": engine.model if hasattr(engine, 'model') else name,
                "provider": engine.provider.value if hasattr(engine, 'provider') else name
            })
        
        # 视频模型
        for name in ["sora", "runway", "stable_video"]:
            generator = self.video_engine.get_generator(name)
            if generator:
                models["video"].append({
                    "id": name,
                    "name": generator.model if hasattr(generator, 'model') else name,
                    "provider": generator.provider.value if hasattr(generator, 'provider') else name
                })
        
        return models
    
    async def estimate_cost(
        self,
        modality: str,
        provider: str,
        params: dict
    ) -> dict:
        """
        估算生成成本
        
        Args:
            modality: 模态类型
            provider: 提供商
            params: 生成参数
        
        Returns:
            dict: 成本估算
        """
        # 简化的成本估算
        base_costs = {
            "image": {
                "dalle": 0.04,  # DALL-E 3 per image
                "stable_diffusion": 0.01
            },
            "audio": {
                "openai": 0.015,  # per 1000 characters
                "azure": 0.01,
                "elevenlabs": 0.03
            },
            "video": {
                "sora": 0.50,  # per second
                "runway": 0.30,
                "stable_video": 0.20
            }
        }
        
        modality = modality.lower()
        provider = provider.lower()
        
        if modality not in base_costs or provider not in base_costs.get(modality, {}):
            return {
                "estimated": False,
                "message": "未找到该配置的成本估算"
            }
        
        base_cost = base_costs[modality][provider]
        
        # 根据参数调整
        multiplier = 1.0
        if modality == "image":
            num_images = params.get("num_images", 1)
            multiplier = num_images
        
        estimated_cost = base_cost * multiplier
        
        return {
            "estimated": True,
            "modality": modality,
            "provider": provider,
            "base_cost": base_cost,
            "multiplier": multiplier,
            "total_cost": round(estimated_cost, 4),
            "currency": "USD"
        }


# 全局管理器实例
_generation_manager: GenerationManager = None


def get_generation_manager() -> GenerationManager:
    """获取生成管理器"""
    global _generation_manager
    if _generation_manager is None:
        _generation_manager = GenerationManager()
    return _generation_manager
