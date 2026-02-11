"""
Capability Detector - 能力检测器
新行为识别、能力涌现检测、能力边界探索、自发技能发现
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
from collections import defaultdict


class EmergenceType(Enum):
    """涌现类型"""
    BEHAVIORAL = "behavioral"      # 行为涌现
    COGNITIVE = "cognitive"        # 认知涌现
    SKILL = "skill"                # 技能涌现
    CREATIVE = "creative"          # 创意涌现


class CapabilityLevel(Enum):
    """能力等级"""
    LATENT = "latent"              # 潜在能力
    EMERGING = "emerging"          # 涌现中
    STABLE = "stable"              # 稳定能力
    MASTERED = "mastered"          # 精通能力


@dataclass
class Capability:
    """能力数据类"""
    name: str
    emergence_type: EmergenceType
    level: CapabilityLevel
    confidence: float
    signature: str
    behaviors: List[str]
    boundaries: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BehaviorPattern:
    """行为模式"""
    pattern_id: str
    sequence: List[str]
    frequency: float
    novelty_score: float
    context: Dict[str, Any]


class CapabilityDetector:
    """
    能力检测器
    负责识别新行为、检测能力涌现、探索能力边界、发现自发技能
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.known_patterns: Dict[str, BehaviorPattern] = {}
        self.emergence_threshold = self.config.get('emergence_threshold', 0.7)
        self.novelty_threshold = self.config.get('novelty_threshold', 0.5)
        self.behavior_history: List[Dict] = []
        
    def _default_config(self) -> Dict:
        return {
            'emergence_threshold': 0.7,
            'novelty_threshold': 0.5,
            'max_pattern_memory': 10000,
            'boundary_precision': 0.01,
            'min_occurrence': 3,
            'similarity_window': 100
        }
    
    def detect(self, model: Any, interaction_data: Dict) -> Optional[Capability]:
        """
        检测能力涌现
        
        Args:
            model: AI模型实例
            interaction_data: 交互数据
            
        Returns:
            检测到的能力或None
        """
        behaviors = self._extract_behaviors(interaction_data)
        if not behaviors:
            return None
            
        # 检查是否为新行为模式
        is_novel, novelty_score = self._check_novelty(behaviors)
        
        if is_novel:
            # 检测能力边界
            boundaries = self._explore_boundaries(behaviors)
            
            # 判断涌现类型
            emergence_type = self._classify_emergence(behaviors, boundaries)
            
            # 计算置信度
            confidence = self._calculate_confidence(behaviors, novelty_score)
            
            # 创建能力签名
            signature = self._create_signature(behaviors, emergence_type)
            
            capability = Capability(
                name=f"capability_{signature[:8]}",
                emergence_type=emergence_type,
                level=CapabilityLevel.EMERGING,
                confidence=confidence,
                signature=signature,
                behaviors=behaviors,
                boundaries=boundaries,
                metadata={
                    'novelty_score': novelty_score,
                    'first_detected': len(self.behavior_history),
                    'model_type': type(model).__name__
                }
            )
            
            self._register_capability(capability)
            return capability
            
        return None
    
    def _extract_behaviors(self, interaction_data: Dict) -> List[str]:
        """从交互数据中提取行为序列"""
        behaviors = []
        
        # 提取动作序列
        if 'actions' in interaction_data:
            behaviors.extend([f"action:{a}" for a in interaction_data['actions']])
            
        # 提取决策模式
        if 'decisions' in interaction_data:
            behaviors.extend([f"decision:{d}" for d in interaction_data['decisions']])
            
        # 提取策略变化
        if 'strategies' in interaction_data:
            behaviors.extend([f"strategy:{s}" for s in interaction_data['strategies']])
            
        # 提取学习信号
        if 'learning' in interaction_data:
            behaviors.append(f"learning:{interaction_data['learning']}")
            
        self.behavior_history.append({
            'behaviors': behaviors,
            'timestamp': len(self.behavior_history)
        })
        
        return behaviors
    
    def _check_novelty(self, behaviors: List[str]) -> Tuple[bool, float]:
        """检查行为的新颖性"""
        if not self.known_patterns:
            return True, 1.0
            
        behavior_set = set(behaviors)
        max_similarity = 0.0
        
        for pattern in self.known_patterns.values():
            pattern_set = set(pattern.sequence)
            similarity = len(behavior_set & pattern_set) / len(behavior_set | pattern_set)
            max_similarity = max(max_similarity, similarity)
            
        novelty_score = 1.0 - max_similarity
        is_novel = novelty_score >= self.novelty_threshold
        
        return is_novel, novelty_score
    
    def _explore_boundaries(self, behaviors: List[str]) -> Dict[str, float]:
        """探索能力边界"""
        boundaries = {}
        
        # 分析行为频率分布
        behavior_freq = defaultdict(int)
        for behavior in behaviors:
            behavior_freq[behavior] += 1
            
        # 设置边界阈值
        total = sum(behavior_freq.values())
        for behavior, freq in behavior_freq.items():
            normalized_freq = freq / total if total > 0 else 0
            boundaries[behavior] = normalized_freq
            
        # 探索行为组合边界
        for i in range(len(behaviors)):
            for j in range(i + 1, len(behaviors)):
                combo = f"{behaviors[i]}+{behaviors[j]}"
                boundaries[combo] = 1.0 / (j - i + 1)
                
        return boundaries
    
    def _classify_emergence(self, behaviors: List[str], boundaries: Dict) -> EmergenceType:
        """分类涌现类型"""
        behavior_str = ' '.join(behaviors).lower()
        
        # 检测行为涌现特征
        behavioral_markers = ['action:', 'decision:', 'strategy:']
        if any(marker in behavior_str for marker in behavioral_markers):
            if any(m in behavior_str for m in ['question', 'response', 'dialog']):
                return EmergenceType.BEHAVIORAL
                
        # 检测认知涌现特征
        cognitive_markers = ['understand', 'concept', 'reasoning', 'learning']
        if any(marker in behavior_str for marker in cognitive_markers):
            return EmergenceType.COGNITIVE
            
        # 检测技能涌现特征
        skill_markers = ['tool', 'skill', 'ability', 'execute']
        if any(marker in behavior_str for marker in skill_markers):
            return EmergenceType.SKILL
            
        # 默认创意涌现
        return EmergenceType.CREATIVE
    
    def _calculate_confidence(self, behaviors: List[str], novelty_score: float) -> float:
        """计算能力置信度"""
        # 基于新颖性
        base_confidence = novelty_score
        
        # 基于行为复杂度
        complexity_bonus = min(len(behaviors) * 0.05, 0.3)
        
        # 基于历史一致性
        consistency = self._check_consistency(behaviors)
        
        return min(base_confidence + complexity_bonus + consistency * 0.2, 1.0)
    
    def _check_consistency(self, behaviors: List[str]) -> float:
        """检查行为一致性"""
        if len(self.behavior_history) < 2:
            return 0.5
            
        recent = self.behavior_history[-self.config['similarity_window']:]
        if not recent:
            return 0.5
            
        # 计算与历史行为的一致性
        recent_behaviors = set()
        for entry in recent:
            recent_behaviors.update(entry['behaviors'])
            
        current_set = set(behaviors)
        if not recent_behaviors:
            return 0.5
            
        overlap = len(current_set & recent_behaviors) / len(recent_behaviors)
        return overlap
    
    def _create_signature(self, behaviors: List[str], emergence_type: EmergenceType) -> str:
        """创建能力签名"""
        signature_data = {
            'behaviors': sorted(behaviors),
            'type': emergence_type.value,
            'length': len(behaviors)
        }
        signature_str = json.dumps(signature_data, sort_keys=True)
        return hashlib.sha256(signature_str.encode()).hexdigest()[:16]
    
    def _register_capability(self, capability: Capability):
        """注册检测到的能力"""
        pattern = BehaviorPattern(
            pattern_id=capability.signature,
            sequence=capability.behaviors,
            frequency=1.0,
            novelty_score=capability.metadata.get('novelty_score', 0.5),
            context=capability.metadata
        )
        self.known_patterns[capability.signature] = pattern
        
        # 限制模式数量
        if len(self.known_patterns) > self.config['max_pattern_memory']:
            oldest = list(self.known_patterns.keys())[0]
            del self.known_patterns[oldest]
    
    def batch_detect(self, model: Any, interactions: List[Dict]) -> List[Capability]:
        """批量检测能力"""
        capabilities = []
        for interaction in interactions:
            capability = self.detect(model, interaction)
            if capability:
                capabilities.append(capability)
        return capabilities
    
    def get_capability_history(self) -> List[Dict]:
        """获取能力检测历史"""
        return [
            {
                'signature': sig,
                'pattern': pat.sequence,
                'frequency': pat.frequency,
                'novelty': pat.novelty_score
            }
            for sig, pat in self.known_patterns.items()
        ]
    
    def update_threshold(self, emergence_threshold: float):
        """更新涌现阈值"""
        self.emergence_threshold = emergence_threshold
        self.config['emergence_threshold'] = emergence_threshold
