"""
Cross-Domain Reasoning Configuration
跨域推理系统配置
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum


class ConfigType(Enum):
    """配置类型"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"


@dataclass
class KnowledgeFusionConfig:
    """知识融合配置"""
    conflict_strategy: str = "evidence"
    min_confidence_threshold: float = 0.6
    enable_semantic_alignment: bool = True
    domain_priority: Dict[str, int] = field(default_factory=lambda: {
        "medicine": 10,
        "biology": 8,
        "chemistry": 8,
        "physics": 7,
        "computer_science": 6,
        "engineering": 5,
        "social_sciences": 4,
        "humanities": 3
    })


@dataclass
class TransferLearningConfig:
    """迁移学习配置"""
    default_method: str = "fine_tuning"
    default_alignment: str = "moment_matching"
    device: str = "cpu"
    batch_size: int = 32
    learning_rate: float = 0.001
    num_iterations: int = 100


@dataclass
class AnalogicalReasoningConfig:
    """类比推理配置"""
    min_similarity_threshold: float = 0.6
    max_mapping_candidates: int = 10
    enable_abstraction: bool = True
    abstraction_depth: int = 3


@dataclass
class UnifiedReasoningConfig:
    """统一推理配置"""
    default_reasoning_type: str = "deductive"
    confidence_threshold: float = 0.7
    enable_probabilistic: bool = True
    enable_causal: bool = True
    enable_common_sense: bool = True
    max_reasoning_steps: int = 10


@dataclass
class APIServerConfig:
    """API服务器配置"""
    host: str = "0.0.0.0"
    port: int = 8080
    debug: bool = False
    workers: int = 4
    cors_enabled: bool = True


@dataclass
class CrossDomainConfig:
    """主配置"""
    config_type: ConfigType = ConfigType.DEVELOPMENT
    
    # 子配置
    knowledge_fusion: KnowledgeFusionConfig = field(default_factory=KnowledgeFusionConfig)
    transfer_learning: TransferLearningConfig = field(default_factory=TransferLearningConfig)
    analogical_reasoning: AnalogicalReasoningConfig = field(default_factory=AnalogicalReasoningConfig)
    unified_reasoning: UnifiedReasoningConfig = field(default_factory=UnifiedReasoningConfig)
    api_server: APIServerConfig = field(default_factory=APIServerConfig)
    
    # 全局配置
    log_level: str = "INFO"
    enable_metrics: bool = True
    enable_profiling: bool = False
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CrossDomainConfig':
        """从字典创建配置"""
        config_type = ConfigType(data.get('config_type', 'development'))
        
        knowledge_fusion_data = data.get('knowledge_fusion', {})
        knowledge_fusion = KnowledgeFusionConfig(**knowledge_fusion_data)
        
        transfer_learning_data = data.get('transfer_learning', {})
        transfer_learning = TransferLearningConfig(**transfer_learning_data)
        
        analogical_reasoning_data = data.get('analogical_reasoning', {})
        analogical_reasoning = AnalogicalReasoningConfig(**analogical_reasoning_data)
        
        unified_reasoning_data = data.get('unified_reasoning', {})
        unified_reasoning = UnifiedReasoningConfig(**unified_reasoning_data)
        
        api_server_data = data.get('api_server', {})
        api_server = APIServerConfig(**api_server_data)
        
        return cls(
            config_type=config_type,
            knowledge_fusion=knowledge_fusion,
            transfer_learning=transfer_learning,
            analogical_reasoning=analogical_reasoning,
            unified_reasoning=unified_reasoning,
            api_server=api_server,
            log_level=data.get('log_level', 'INFO'),
            enable_metrics=data.get('enable_metrics', True),
            enable_profiling=data.get('enable_profiling', False)
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "config_type": self.config_type.value,
            "knowledge_fusion": {
                "conflict_strategy": self.knowledge_fusion.conflict_strategy,
                "min_confidence_threshold": self.knowledge_fusion.min_confidence_threshold,
                "enable_semantic_alignment": self.knowledge_fusion.enable_semantic_alignment,
                "domain_priority": self.knowledge_fusion.domain_priority
            },
            "transfer_learning": {
                "default_method": self.transfer_learning.default_method,
                "default_alignment": self.transfer_learning.default_alignment,
                "device": self.transfer_learning.device,
                "batch_size": self.transfer_learning.batch_size,
                "learning_rate": self.transfer_learning.learning_rate,
                "num_iterations": self.transfer_learning.num_iterations
            },
            "analogical_reasoning": {
                "min_similarity_threshold": self.analogical_reasoning.min_similarity_threshold,
                "max_mapping_candidates": self.analogical_reasoning.max_mapping_candidates,
                "enable_abstraction": self.analogical_reasoning.enable_abstraction,
                "abstraction_depth": self.analogical_reasoning.abstraction_depth
            },
            "unified_reasoning": {
                "default_reasoning_type": self.unified_reasoning.default_reasoning_type,
                "confidence_threshold": self.unified_reasoning.confidence_threshold,
                "enable_probabilistic": self.unified_reasoning.enable_probabilistic,
                "enable_causal": self.unified_reasoning.enable_causal,
                "enable_common_sense": self.unified_reasoning.enable_common_sense,
                "max_reasoning_steps": self.unified_reasoning.max_reasoning_steps
            },
            "api_server": {
                "host": self.api_server.host,
                "port": self.api_server.port,
                "debug": self.api_server.debug,
                "workers": self.api_server.workers,
                "cors_enabled": self.api_server.cors_enabled
            },
            "log_level": self.log_level,
            "enable_metrics": self.enable_metrics,
            "enable_profiling": self.enable_profiling
        }


# 默认配置
DEFAULT_CONFIG = CrossDomainConfig()


def load_config(config_path: Optional[str] = None) -> CrossDomainConfig:
    """加载配置"""
    import json
    
    if config_path is None:
        config_path = os.environ.get(
            'CROSS_DOMAIN_CONFIG',
            'config.json'
        )
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            data = json.load(f)
            return CrossDomainConfig.from_dict(data)
    
    return DEFAULT_CONFIG


def save_config(config: CrossDomainConfig, config_path: str = 'config.json') -> None:
    """保存配置"""
    import json
    
    with open(config_path, 'w') as f:
        json.dump(config.to_dict(), f, indent=2)
