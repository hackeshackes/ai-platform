"""
Inference Module - High-performance LLM inference

推理模块，提供vLLM推理引擎、动态批处理和KV缓存优化。
"""

from .vllm_engine import (
    vLLMEngine,
    vLLMConfig,
    GenerationRequest,
    GenerationResponse,
    ChatCompletionRequest,
    ChatMessage,
    QuantizationType,
    create_vllm_engine,
    get_vllm_engine,
    shutdown_vllm_engine,
)

from .batching import (
    BatchScheduler,
    BatchingConfig,
    BatchingStrategy,
    BatchItem,
    BatchResult,
    KVCacheManager,
    PrefillScheduler,
    create_batch_scheduler,
    get_batch_scheduler,
)

__all__ = [
    # vLLM引擎
    "vLLMEngine",
    "vLLMConfig",
    "GenerationRequest",
    "GenerationResponse",
    "ChatCompletionRequest",
    "ChatMessage",
    "QuantizationType",
    "create_vllm_engine",
    "get_vllm_engine",
    "shutdown_vllm_engine",
    
    # 批处理
    "BatchScheduler",
    "BatchingConfig",
    "BatchingStrategy",
    "BatchItem",
    "BatchResult",
    "KVCacheManager",
    "PrefillScheduler",
    "create_batch_scheduler",
    "get_batch_scheduler",
]
