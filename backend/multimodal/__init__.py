"""
多模态能力模块
支持图像、音频、视频理解和生成
"""

from .base import (
    BaseMultimodalModel,
    BaseVisionModel,
    BaseAudioModel,
    BaseVideoModel,
    BaseMultimodalEngine,
    MultimodalProvider,
    MultimodalResult,
    MediaType
)

from .vision import (
    create_vision_model,
    OpenAIVisionModel,
    AnthropicVisionModel,
    GoogleVisionModel,
    DeepSeekVLModel
)

from .audio import (
    create_audio_model,
    OpenAIAudioModel,
    AnthropicAudioModel,
    GoogleAudioModel,
    DeepSeekAudioModel
)

from .video import (
    create_video_model,
    OpenAIVideoModel,
    GoogleVideoModel,
    AnthropicVideoModel,
    DeepSeekVideoModel
)

__all__ = [
    # Base
    "BaseMultimodalModel",
    "BaseVisionModel", 
    "BaseAudioModel",
    "BaseVideoModel",
    "BaseMultimodalEngine",
    "MultimodalProvider",
    "MultimodalResult",
    "MediaType",
    
    # Vision
    "create_vision_model",
    "OpenAIVisionModel",
    "AnthropicVisionModel",
    "GoogleVisionModel",
    "DeepSeekVLModel",
    
    # Audio
    "create_audio_model",
    "OpenAIAudioModel",
    "AnthropicAudioModel",
    "GoogleAudioModel",
    "DeepSeekAudioModel",
    
    # Video
    "create_video_model",
    "OpenAIVideoModel",
    "GoogleVideoModel",
    "AnthropicVideoModel",
    "DeepSeekVideoModel"
]
