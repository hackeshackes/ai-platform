"""
多模态生成模块
提供图像、音频、视频生成能力
"""

__version__ = "1.0.0"

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

from .unified import (
    GenerationManager,
    get_generation_manager
)

__all__ = [
    # 版本
    "__version__",
    
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
    "get_unified_service",
    
    # 管理类
    "GenerationManager",
    "get_generation_manager"
]
