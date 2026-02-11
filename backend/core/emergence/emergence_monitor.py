"""
Emergence Monitor - 涌现监控
能力追踪、涌现事件日志、影响评估、安全性检查
"""

import time
import json
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime
from collections import defaultdict
import threading


class SafetyLevel(Enum):
    """安全等级"""
    SAFE = "safe"
    CAUTION = "caution"
    WARNING = "warning"
    CRITICAL = "critical"


class ImpactLevel(Enum):
    """影响等级"""
    MINOR = "minor"
    MODERATE = "moderate"
    SIGNIFICANT = "significant"
    MAJOR = "major"


@dataclass
class EmergenceEvent:
    """涌现事件"""
    event_id: str
    timestamp: float
    event_type: str
    capability_name: str
    emergence_type: str
    confidence: float
    impact: ImpactLevel
    safety: SafetyLevel
    description: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: str = "active"
    
    def to_dict(self) -> Dict:
        return {
            'event_id': self.event_id,
            'timestamp': self.timestamp,
            'datetime': datetime.fromtimestamp(self.timestamp).isoformat(),
            'event_type': self.event_type,
            'capability_name': self.capability_name,
            'emergence_type': self.emergence_type,
            'confidence': self.confidence,
            'impact': self.impact.value,
            'safety': self.safety.value,
            'description': self.description,
            'metadata': self.metadata,
            'status': self.status
        }


@dataclass
class CapabilityMetrics:
    """能力指标"""
    capability_name: str
    emergence_time: float
    quality_score: float
    stability_score: float
    usage_count: int
    success_rate: float
    last_updated: float
    
    
class EmergenceMonitor:
    """
    涌现监控系统
    负责能力追踪、涌现事件日志、影响评估和安全性检查
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.events: List[EmergenceEvent] = []
        self.capability_metrics: Dict[str, CapabilityMetrics] = {}
        self.event_callbacks: List[Callable] = []
        self.safety_checks: List[Dict] = []
        self.impact_history: List[Dict] = []
        
        self._lock = threading.Lock()
        self._event_counter = 0
        
    def _default_config(self) -> Dict:
        return {
            'max_events': 10000,
            'quality_threshold': 0.9,
            'safety_threshold': 0.8,
            'impact_window': 1000,
            'log_retention_days': 30,
            'auto_revert_enabled': True
        }
    
    def track(self, agent_behavior: Dict) -> Optional[EmergenceEvent]:
        """
        跟踪代理行为，检测涌现事件
        
        Args:
            agent_behavior: 代理行为数据
            
        Returns:
            检测到的涌现事件或None
        """
        with self._lock:
            # 分析行为
            emergence_detected, details = self._analyze_behavior(agent_behavior)
            
            if emergence_detected:
                event = self._create_event(details)
                self._log_event(event)
                self._update_metrics(details)
                
                # 安全检查
                safety_result = self._safety_check(event)
                event.safety = safety_result['level']
                
                # 评估影响
                impact = self._assess_impact(event)
                event.impact = impact['level']
                
                # 触发回调
                self._trigger_callbacks(event)
                
                # 必要时回滚
                if safety_result['level'] == SafetyLevel.CRITICAL and self.config['auto_revert_enabled']:
                    self._revert_capability(event.capability_name)
                
                return event
                
        return None
    
    def _analyze_behavior(self, behavior: Dict) -> Tuple[bool, Dict]:
        """分析行为"""
        # 检查是否有新行为模式
        behavior_features = self._extract_features(behavior)
        
        # 计算新颖性
        novelty = self._calculate_novelty(behavior_features)
        
        # 检测涌现特征
        if novelty > 0.6:
            return True, {
                'event_type': 'capability_emergence',
                'capability_name': f"capability_{int(time.time())}",
                'emergence_type': behavior.get('type', 'unknown'),
                'confidence': novelty,
                'description': f"检测到新行为模式: {behavior.get('description', '未知')}",
                'metadata': behavior_features
            }
        
        return False, {}
    
    def _extract_features(self, behavior: Dict) -> Dict:
        """提取行为特征"""
        features = {}
        
        # 提取动作特征
        if 'actions' in behavior:
            features['action_count'] = len(behavior['actions'])
            features['action_types'] = list(set(behavior['actions']))
            
        # 提取性能特征
        if 'performance' in behavior:
            features['performance'] = behavior['performance']
            
        # 提取上下文特征
        if 'context' in behavior:
            features['context'] = behavior['context']
            
        return features
    
    def _calculate_novelty(self, features: Dict) -> float:
        """计算新颖性"""
        # 基于特征数量和复杂性
        complexity = len(features) * 0.1
        
        # 基于唯一性
        unique_elements = 0
        for key, value in features.items():
            if isinstance(value, list):
                unique_elements += len(set(value))
            else:
                unique_elements += 1
        
        novelty = min(complexity + unique_elements * 0.05, 1.0)
        
        # 检查历史
        for event in self.events[-100:]:
            if event.metadata == features:
                novelty *= 0.5
                
        return min(novelty, 1.0)
    
    def _create_event(self, details: Dict) -> EmergenceEvent:
        """创建事件"""
        self._event_counter += 1
        
        event = EmergenceEvent(
            event_id=f"emg_{self._event_counter}_{int(time.time())}",
            timestamp=time.time(),
            event_type=details.get('event_type', 'unknown'),
            capability_name=details.get('capability_name', 'unknown'),
            emergence_type=details.get('emergence_type', 'unknown'),
            confidence=details.get('confidence', 0.0),
            impact=ImpactLevel.MODERATE,
            safety=SafetyLevel.SAFE,
            description=details.get('description', ''),
            metadata=details.get('metadata', {})
        )
        
        return event
    
    def _log_event(self, event: EmergenceEvent):
        """记录事件"""
        self.events.append(event)
        
        # 限制事件数量
        if len(self.events) > self.config['max_events']:
            self.events = self.events[-self.config['max_events']:]
    
    def _update_metrics(self, details: Dict):
        """更新能力指标"""
        name = details.get('capability_name')
        if name:
            if name not in self.capability_metrics:
                self.capability_metrics[name] = CapabilityMetrics(
                    capability_name=name,
                    emergence_time=time.time(),
                    quality_score=details.get('confidence', 0.0),
                    stability_score=0.5,
                    usage_count=0,
                    success_rate=0.0,
                    last_updated=time.time()
                )
            
            metrics = self.capability_metrics[name]
            metrics.usage_count += 1
            metrics.last_updated = time.time()
    
    def _safety_check(self, event: EmergenceEvent) -> Dict:
        """安全检查"""
        result = {
            'level': SafetyLevel.SAFE,
            'issues': [],
            'recommendations': []
        }
        
        # 检查置信度
        if event.confidence > 0.95:
            result['level'] = SafetyLevel.CAUTION
            result['issues'].append('极高的置信度可能表明过拟合')
            result['recommendations'].append('建议进行人工审核')
        
        # 检查影响范围
        if event.impact == ImpactLevel.MAJOR:
            result['level'] = SafetyLevel.WARNING
            result['issues'].append('重大影响可能影响系统稳定性')
            result['recommendations'].append('建议进行渐进式部署')
        
        # 检查行为一致性
        recent_events = [e for e in self.events[-10:] 
                        if e.capability_name == event.capability_name]
        if len(recent_events) > 5:
            result['level'] = SafetyLevel.CAUTION
            result['issues'].append('频繁涌现可能表明不稳定')
        
        # 记录检查结果
        self.safety_checks.append({
            'event_id': event.event_id,
            'timestamp': time.time(),
            'result': {k: v.value if isinstance(v, Enum) else v for k, v in result.items()}
        })
        
        return result
    
    def _assess_impact(self, event: EmergenceEvent) -> Dict:
        """评估影响"""
        impact_score = 0.0
        
        # 基于置信度
        impact_score += event.confidence * 0.3
        
        # 基于类型
        type_weights = {
            'behavioral': 0.2,
            'cognitive': 0.4,
            'skill': 0.3,
            'creative': 0.5
        }
        impact_score += type_weights.get(event.emergence_type, 0.1)
        
        # 基于历史频率
        recent_count = len([e for e in self.events[-100:] 
                           if e.capability_name == event.capability_name])
        impact_score += min(recent_count * 0.02, 0.2)
        
        # 分类影响等级
        if impact_score > 0.7:
            level = ImpactLevel.MAJOR
        elif impact_score > 0.5:
            level = ImpactLevel.SIGNIFICANT
        elif impact_score > 0.3:
            level = ImpactLevel.MODERATE
        else:
            level = ImpactLevel.MINOR
        
        impact_result = {
            'level': level,
            'score': impact_score,
            'factors': {
                'confidence': event.confidence,
                'type_weight': type_weights.get(event.emergence_type, 0.1),
                'frequency_factor': min(recent_count * 0.02, 0.2)
            }
        }
        
        self.impact_history.append({
            'event_id': event.event_id,
            'timestamp': time.time(),
            'impact': impact_result
        })
        
        return impact_result
    
    def _trigger_callbacks(self, event: EmergenceEvent):
        """触发回调"""
        for callback in self.event_callbacks:
            try:
                callback(event)
            except Exception:
                pass
    
    def _revert_capability(self, capability_name: str):
        """回滚能力"""
        # 标记为待回滚
        for event in self.events:
            if event.capability_name == capability_name:
                event.status = 'reverted'
        
        # 从指标中移除
        if capability_name in self.capability_metrics:
            del self.capability_metrics[capability_name]
    
    def register_callback(self, callback: Callable[[EmergenceEvent], None]):
        """注册事件回调"""
        self.event_callbacks.append(callback)
    
    def get_event_history(self, limit: int = 100) -> List[Dict]:
        """获取事件历史"""
        events = self.events[-limit:]
        return [e.to_dict() for e in events]
    
    def get_capability_status(self, capability_name: str) -> Optional[Dict]:
        """获取能力状态"""
        if capability_name not in self.capability_metrics:
            return None
            
        metrics = self.capability_metrics[capability_name]
        recent_events = [e.to_dict() for e in self.events 
                         if e.capability_name == capability_name]
        
        return {
            'name': metrics.capability_name,
            'emergence_time': metrics.emergence_time,
            'quality_score': metrics.quality_score,
            'stability_score': metrics.stability_score,
            'usage_count': metrics.usage_count,
            'success_rate': metrics.success_rate,
            'recent_events': recent_events[-10:]
        }
    
    def get_safety_summary(self) -> Dict:
        """获取安全摘要"""
        level_counts = defaultdict(int)
        for event in self.events:
            level_counts[event.safety] += 1
            
        return {
            'total_events': len(self.events),
            'safety_distribution': {k.value: v for k, v in level_counts.items()},
            'critical_events': len([e for e in self.events if e.safety == SafetyLevel.CRITICAL]),
            'auto_reverted': len([e for e in self.events if e.status == 'reverted'])
        }
    
    def get_impact_summary(self) -> Dict:
        """获取影响摘要"""
        impact_counts = defaultdict(int)
        for event in self.events:
            impact_counts[event.impact] += 1
            
        return {
            'total_events': len(self.events),
            'impact_distribution': {k.value: v for k, v in impact_counts.items()},
            'major_impacts': len([e for e in self.events if e.impact == ImpactLevel.MAJOR]),
            'recent_impacts': self.impact_history[-10:]
        }
    
    def generate_report(self) -> Dict:
        """生成监控报告"""
        return {
            'timestamp': time.time(),
            'summary': {
                'total_events': len(self.events),
                'active_capabilities': len(self.capability_metrics)
            },
            'safety': self.get_safety_summary(),
            'impact': self.get_impact_summary(),
            'recent_events': self.get_event_history(10),
            'recommendations': self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """生成建议"""
        recommendations = []
        
        # 基于安全级别
        critical = len([e for e in self.events if e.safety == SafetyLevel.CRITICAL])
        if critical > 0:
            recommendations.append('存在严重安全问题，需要立即处理')
        
        # 基于影响级别
        major = len([e for e in self.events if e.impact == ImpactLevel.MAJOR])
        if major > 5:
            recommendations.append('大量重大影响事件，建议审查涌现策略')
        
        # 基于趋势
        if len(self.events) > 100:
            recent_week = [e for e in self.events[-1000:] 
                          if e.timestamp > time.time() - 7*24*3600]
            if len(recent_week) > len(self.events) / 2:
                recommendations.append('涌现事件呈上升趋势，建议分析根本原因')
        
        return recommendations
    
    def clear_history(self, before_timestamp: float = None):
        """清除历史"""
        if before_timestamp is None:
            before_timestamp = time.time() - self.config['log_retention_days'] * 24 * 3600
            
        self.events = [e for e in self.events if e.timestamp > before_timestamp]
