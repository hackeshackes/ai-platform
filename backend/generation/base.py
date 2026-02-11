"""
多模态生成基类模块
定义通用的生成接口和数据结构
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List
import uuid


class GenerationProvider(str, Enum):
    """生成服务提供商枚举"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    STABILITY = "stability"
    MIDJOURNEY = "midjourney"
    RUNWAY = "runway"
    ELEVENLABS = "elevenlabs"
    AZURE = "azure"
    CUSTOM = "custom"


class TaskStatus(str, Enum):
    """任务状态枚举"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class GenerationType(str, Enum):
    """生成类型枚举"""
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    TEXT = "text"
    MULTIMODAL = "multimodal"


@dataclass
class GenerationRequest:
    """生成请求基类"""
    meta: Dict[str, Any] = field(default_factory=dict)
    prompt: str = ""
    negative_prompt: Optional[str] = None
    model: Optional[str] = None
    seed: Optional[int] = None
    num_outputs: int = 1
    width: Optional[int] = None
    height: Optional[int] = None
    guidance_scale: float = 7.5
    steps: int = 50
    scheduler: str = "EulerDiscreteScheduler"
    clip_skip: int = 1
    lora_scale: float = 0.8
    
    class Config:
        arbitrary_types_allowed = True


@dataclass
class GenerationResponse:
    """生成响应基类"""
    success: bool
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    provider: str = ""
    model: str = ""
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    elapsed_ms: float = 0
    meta: Dict[str, Any] = field(default_factory=dict)
    
    class Config:
        arbitrary_types_allowed = True


@dataclass
class GenerationTask:
    """生成任务"""
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    request: GenerationRequest = None
    status: TaskStatus = TaskStatus.PENDING
    provider: str = ""
    generation_type: GenerationType = GenerationType.TEXT
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: float = 0
    result: Optional[GenerationResponse] = None
    error_message: Optional[str] = None
    
    class Config:
        arbitrary_types_allowed = True


@dataclass
class GenerationHistory:
    """生成历史记录"""
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: Optional[str] = None
    generation_type: GenerationType = None
    provider: str = ""
    model: str = ""
    prompt: str = ""
    status: TaskStatus = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    duration_ms: float = 0
    parameters: Dict[str, Any] = field(default_factory=dict)
    result_meta: Dict[str, Any] = field(default_factory=dict)
    
    class Config:
        arbitrary_types_allowed = True


class BaseGenerationModel(ABC):
    """生成模型基类"""
    
    def __init__(self):
        self.provider = GenerationProvider.CUSTOM
        self.model_name = "base"
        self.supported_types: List[GenerationType] = []
    
    @property
    def capabilities(self) -> Dict[str, Any]:
        """返回模型能力"""
        return {
            "provider": self.provider.value if isinstance(self.provider, GenerationProvider) else self.provider,
            "model": self.model_name,
            "supported_types": [t.value for t in self.supported_types]
        }
    
    @abstractmethod
    async def generate(
        self,
        request: GenerationRequest
    ) -> GenerationResponse:
        """生成内容"""
        pass
    
    async def cancel(self, task_id: str) -> bool:
        """取消任务"""
        return False
    
    async def get_status(self) -> Dict[str, Any]:
        """获取模型状态"""
        return {
            "available": True,
            "provider": self.provider.value if isinstance(self.provider, GenerationProvider) else self.provider,
            "model": self.model_name
        }


class BaseMultimodalEngine:
    """多模态引擎基类"""
    
    def __init__(self):
        self.vision_models: Dict[str, BaseGenerationModel] = {}
        self.audio_models: Dict[str, BaseGenerationModel] = {}
        self.video_models: Dict[str, BaseGenerationModel] = {}
        self.text_models: Dict[str, BaseGenerationModel] = {}
    
    def register_vision_model(self, name: str, model: BaseGenerationModel):
        """注册视觉模型"""
        self.vision_models[name] = model
    
    def register_audio_model(self, name: str, model: BaseGenerationModel):
        """注册音频模型"""
        self.audio_models[name] = model
    
    def register_video_model(self, name: str, model: BaseGenerationModel):
        """注册视频模型"""
        self.video_models[name] = model
    
    def register_text_model(self, name: str, model: BaseGenerationModel):
        """注册文本模型"""
        self.text_models[name] = model
    
    def get_model(self, modality: str, name: str) -> Optional[BaseGenerationModel]:
        """获取模型"""
        models = getattr(self, f"{modality}_models", {})
        return models.get(name)
    
    async def list_models(self, modality: str = None) -> Dict[str, List[Dict[str, Any]]]:
        """列出所有模型"""
        result = {}
        
        if modality is None or modality == "vision":
            result["vision"] = [
                model.capabilities for model in self.vision_models.values()
            ]
        
        if modality is None or modality == "audio":
            result["audio"] = [
                model.capabilities for model in self.audio_models.values()
            ]
        
        if modality is None or modality == "video":
            result["video"] = [
                model.capabilities for model in self.video_models.values()
            ]
        
        if modality is None or modality == "text":
            result["text"] = [
                model.capabilities for model in self.text_models.values()
            ]
        
        return result


class UnifiedGenerationService:
    """统一生成服务"""
    
    def __init__(self):
        self.image_engine = None
        self.audio_engine = None
        self.video_engine = None
        self.task_history: Dict[str, GenerationTask] = {}
        self.max_history_size = 1000
    
    async def initialize(self):
        """初始化引擎"""
        from .image import get_image_engine
        from .audio import get_tts_engine
        from .video import get_video_engine
        
        self.image_engine = get_image_engine()
        self.audio_engine = get_tts_engine()
        self.video_engine = get_video_engine()
    
    def create_task(
        self,
        generation_type: GenerationType,
        request: GenerationRequest,
        provider: str
    ) -> GenerationTask:
        """创建生成任务"""
        task = GenerationTask(
            request=request,
            generation_type=generation_type,
            provider=provider,
            status=TaskStatus.PENDING
        )
        
        self.task_history[task.task_id] = task
        
        # 限制历史大小
        if len(self.task_history) > self.max_history_size:
            oldest_task_id = next(iter(self.task_history))
            del self.task_history[oldest_task_id]
        
        return task
    
    def get_task(self, task_id: str) -> Optional[GenerationTask]:
        """获取任务"""
        return self.task_history.get(task_id)
    
    def list_tasks(
        self,
        status: TaskStatus = None,
        generation_type: GenerationType = None,
        limit: int = 100
    ) -> List[GenerationTask]:
        """列出任务"""
        tasks = list(self.task_history.values())
        
        if status:
            tasks = [t for t in tasks if t.status == status]
        
        if generation_type:
            tasks = [t for t in tasks if t.generation_type == generation_type]
        
        return sorted(tasks, key=lambda t: t.created_at, reverse=True)[:limit]
    
    async def generate_image(
        self,
        prompt: str,
        provider: str = "dalle",
        **kwargs
    ) -> GenerationResponse:
        """生成图像"""
        from .image import ImageGenerationRequest
        
        request = ImageGenerationRequest(
            prompt=prompt,
            **kwargs
        )
        
        return await self.image_engine.generate(provider, request)
    
    async def generate_audio(
        self,
        text: str,
        provider: str = "openai",
        **kwargs
    ) -> GenerationResponse:
        """生成音频"""
        from .audio import TTSRequest
        
        request = TTSRequest(
            text=text,
            **kwargs
        )
        
        return await self.audio_engine.synthesize(provider, request)
    
    async def generate_video(
        self,
        prompt: str,
        provider: str = "sora",
        **kwargs
    ) -> GenerationResponse:
        """生成视频"""
        from .video import VideoGenerationRequest
        
        request = VideoGenerationRequest(
            prompt=prompt,
            **kwargs
        )
        
        return await self.video_engine.generate(provider, request)
    
    async def multimodal_generate(
        self,
        prompt: str,
        modalities: List[str] = None,
        provider_map: Dict[str, str] = None
    ) -> Dict[str, GenerationResponse]:
        """多模态生成"""
        if modalities is None:
            modalities = ["image", "audio"]
        
        if provider_map is None:
            provider_map = {}
        
        results = {}
        
        # 并行生成
        tasks = []
        for modality in modalities:
            provider = provider_map.get(modality, "openai")
            
            if modality == "image":
                tasks.append(self.generate_image(prompt, provider))
            elif modality == "audio":
                tasks.append(self.generate_audio(prompt, provider))
            elif modality == "video":
                tasks.append(self.generate_video(prompt, provider))
        
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        for modality, response in zip(modalities, responses):
            if isinstance(response, Exception):
                results[modality] = GenerationResponse(
                    success=False,
                    error=str(response)
                )
            else:
                results[modality] = response
        
        return results


# 全局服务实例
_unified_service: Optional[UnifiedGenerationService] = None


def get_unified_service() -> UnifiedGenerationService:
    """获取统一生成服务"""
    global _unified_service
    if _unified_service is None:
        _unified_service = UnifiedGenerationService()
    return _unified_service
