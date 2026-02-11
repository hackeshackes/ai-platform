"""
用户画像模块 - user_profile.py

负责用户行为分析、偏好提取和特征向量构建
"""

import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
import hashlib
from datetime import datetime, timedelta


@dataclass
class UserBehavior:
    """用户行为数据"""
    user_id: str
    timestamp: datetime
    action_type: str  # view, click, use, like, share
    item_id: str
    item_type: str  # agent, pipeline, template, tutorial
    duration: float = 0.0  # 停留时间(秒)
    metadata: Dict = field(default_factory=dict)


@dataclass  
class UserPreferences:
    """用户偏好"""
    agent_preferences: Dict[str, float] = field(default_factory=dict)
    pipeline_preferences: Dict[str, float] = field(default_factory=dict)
    template_preferences: Dict[str, float] = field(default_factory=dict)
    tutorial_preferences: Dict[str, float] = field(default_factory=dict)
    industry_preferences: Dict[str, float] = field(default_factory=dict)
    skill_level: str = "beginner"  # beginner, intermediate, advanced
    preferred_complexity: str = "medium"  # low, medium, high


class UserProfile:
    """用户画像类"""
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.behaviors: List[UserBehavior] = []
        self.preferences = UserPreferences()
        self.feature_vector: Optional[np.ndarray] = None
        self.similarity_cache: Dict[str, float] = {}
        
        # 行为统计
        self.action_counts = defaultdict(int)
        self.item_interactions = defaultdict(set)
        self.daily_activity = defaultdict(int)
        
    def add_behavior(self, behavior: UserBehavior):
        """添加用户行为"""
        self.behaviors.append(behavior)
        self.action_counts[behavior.action_type] += 1
        self.item_interactions[behavior.item_id].add(behavior.action_type)
        
        # 记录日活跃度
        date_key = behavior.timestamp.strftime("%Y-%m-%d")
        self.daily_activity[date_key] += 1
        
    def analyze_behaviors(self, time_window: int = 30) -> Dict[str, Any]:
        """分析用户行为模式"""
        cutoff_date = datetime.now() - timedelta(days=time_window)
        recent_behaviors = [b for b in self.behaviors if b.timestamp >= cutoff_date]
        
        analysis = {
            "total_actions": len(recent_behaviors),
            "action_distribution": dict(self.action_counts),
            "active_days": len(self.daily_activity),
            "most_active_period": self._get_most_active_period(),
            "avg_session_duration": self._calculate_avg_duration(recent_behaviors),
            "interaction_diversity": self._calculate_diversity(),
            "preferred_item_types": self._get_preferred_item_types(),
            "engagement_score": self._calculate_engagement_score(recent_behaviors)
        }
        
        return analysis
    
    def extract_preferences(self) -> UserPreferences:
        """提取用户偏好"""
        if not self.behaviors:
            return self.preferences
            
        # 基于行为的权重计算
        action_weights = {
            'like': 1.0,
            'share': 0.9,
            'use': 0.8,
            'click': 0.5,
            'view': 0.2
        }
        
        # 按类型收集偏好
        type_preferences = defaultdict(lambda: defaultdict(float))
        
        for behavior in self.behaviors:
            weight = action_weights.get(behavior.action_type, 0.3)
            item_type = behavior.item_type
            
            if item_type == 'agent':
                self._update_agent_preferences(behavior, weight, type_preferences)
            elif item_type == 'pipeline':
                self._update_pipeline_preferences(behavior, weight, type_preferences)
            elif item_type == 'template':
                self._update_template_preferences(behavior, weight, type_preferences)
            elif item_type == 'tutorial':
                self._update_tutorial_preferences(behavior, weight, type_preferences)
                
        # 更新偏好对象
        self.preferences.agent_preferences = dict(type_preferences['agent'])
        self.preferences.pipeline_preferences = dict(type_preferences['pipeline'])
        self.preferences.template_preferences = dict(type_preferences['template'])
        self.preferences.tutorial_preferences = dict(type_preferences['tutorial'])
        
        # 推断技能水平
        self.preferences.skill_level = self._infer_skill_level()
        
        # 推断复杂度偏好
        self.preferences.preferred_complexity = self._infer_complexity_preference()
        
        return self.preferences
    
    def build_feature_vector(self, feature_dim: int = 128) -> np.ndarray:
        """构建用户特征向量"""
        features = np.zeros(feature_dim)
        
        # 行为特征 (0-31)
        self._build_behavior_features(features)
        
        # 偏好特征 (32-63)
        self._build_preference_features(features)
        
        # 交互特征 (64-95)
        self._build_interaction_features(features)
        
        # 时间特征 (96-127)
        self._build_temporal_features(features)
        
        self.feature_vector = features
        return features
    
    def get_similarity(self, other_profile: 'UserProfile') -> float:
        """计算与另一用户的相似度"""
        cache_key = other_profile.user_id
        
        if cache_key in self.similarity_cache:
            return self.similarity_cache[cache_key]
            
        if self.feature_vector is None:
            self.build_feature_vector()
        if other_profile.feature_vector is None:
            other_profile.build_feature_vector()
            
        # 余弦相似度
        similarity = self._cosine_similarity(self.feature_vector, other_profile.feature_vector)
        
        self.similarity_cache[cache_key] = similarity
        return similarity
    
    # 内部辅助方法
    def _get_most_active_period(self) -> str:
        """获取最活跃时段"""
        hours = [b.timestamp.hour for b in self.behaviors]
        if not hours:
            return "unknown"
            
        hour_buckets = defaultdict(int)
        for hour in hours:
            if 6 <= hour < 12:
                hour_buckets['morning'] += 1
            elif 12 <= hour < 18:
                hour_buckets['afternoon'] += 1
            elif 18 <= hour < 22:
                hour_buckets['evening'] += 1
            else:
                hour_buckets['night'] += 1
                
        return max(hour_buckets, key=hour_buckets.get)
    
    def _calculate_avg_duration(self, behaviors: List[UserBehavior]) -> float:
        """计算平均会话时长"""
        durations = [b.duration for b in behaviors if b.duration > 0]
        return np.mean(durations) if durations else 0.0
    
    def _calculate_diversity(self) -> float:
        """计算交互多样性"""
        unique_items = len(self.item_interactions)
        total_interactions = sum(self.action_counts.values())
        return unique_items / total_interactions if total_interactions > 0 else 0.0
    
    def _get_preferred_item_types(self) -> Dict[str, float]:
        """获取偏好的物品类型"""
        type_counts = defaultdict(int)
        for behavior in self.behaviors:
            type_counts[behavior.item_type] += 1
            
        total = sum(type_counts.values())
        return {k: v/total for k, v in type_counts.items()}
    
    def _calculate_engagement_score(self, behaviors: List[UserBehavior]) -> float:
        """计算用户参与度分数"""
        if not behaviors:
            return 0.0
            
        weights = {'like': 1.0, 'share': 0.9, 'use': 0.7, 'click': 0.4, 'view': 0.1}
        total_score = sum(weights.get(b.action_type, 0.3) for b in behaviors)
        
        # 归一化到0-1
        return min(1.0, total_score / (len(behaviors) * 3))
    
    def _update_agent_preferences(self, behavior: UserBehavior, weight: float, 
                                   preferences: defaultdict):
        """更新Agent偏好"""
        tags = behavior.metadata.get('tags', [])
        for tag in tags:
            preferences['agent'][tag] += weight
            
        scenario = behavior.metadata.get('scenario', 'general')
        preferences['agent'][f'scenario_{scenario}'] += weight * 1.2
    
    def _update_pipeline_preferences(self, behavior: UserBehavior, weight: float,
                                      preferences: defaultdict):
        """更新Pipeline偏好"""
        complexity = behavior.metadata.get('complexity', 'medium')
        preferences['pipeline'][complexity] += weight
        
        task_type = behavior.metadata.get('task_type', 'general')
        preferences['pipeline'][f'task_{task_type}'] += weight
    
    def _update_template_preferences(self, behavior: UserBehavior, weight: float,
                                     preferences: defaultdict):
        """更新模板偏好"""
        industry = behavior.metadata.get('industry', 'general')
        preferences['template'][industry] += weight
        
        frequency = behavior.metadata.get('frequency', 'occasional')
        preferences['template'][f'freq_{frequency}'] += weight
    
    def _update_tutorial_preferences(self, behavior: UserBehavior, weight: float,
                                     preferences: defaultdict):
        """更新教程偏好"""
        difficulty = behavior.metadata.get('difficulty', 'beginner')
        preferences['tutorial'][difficulty] += weight
        
        topic = behavior.metadata.get('topic', 'general')
        preferences['tutorial'][f'topic_{topic}'] += weight
    
    def _infer_skill_level(self) -> str:
        """推断技能水平"""
        total_interactions = sum(self.action_counts.values())
        
        if total_interactions < 10:
            return "beginner"
        elif total_interactions < 50:
            # 检查是否有高级操作
            if self.action_counts.get('share', 0) > 5:
                return "advanced"
            return "intermediate"
        else:
            if self.action_counts.get('share', 0) > 15:
                return "expert"
            return "advanced"
    
    def _infer_complexity_preference(self) -> str:
        """推断复杂度偏好"""
        complexity_usage = defaultdict(int)
        
        for behavior in self.behaviors:
            if behavior.item_type == 'pipeline':
                complexity = behavior.metadata.get('complexity', 'medium')
                complexity_usage[complexity] += 1
                
        if complexity_usage:
            return max(complexity_usage, key=complexity_usage.get)
        return "medium"
    
    def _build_behavior_features(self, features: np.ndarray):
        """构建行为特征"""
        # 行为频率特征
        features[0] = min(1.0, self.action_counts.get('view', 0) / 100)
        features[1] = min(1.0, self.action_counts.get('click', 0) / 50)
        features[2] = min(1.0, self.action_counts.get('use', 0) / 30)
        features[3] = min(1.0, self.action_counts.get('like', 0) / 20)
        features[4] = min(1.0, self.action_counts.get('share', 0) / 10)
        
        # 活跃度特征
        features[5] = min(1.0, len(self.daily_activity) / 30)
        features[6] = len(self.item_interactions) / 200 if self.behaviors else 0
        
        # 多样性
        features[7] = self._calculate_diversity()
    
    def _build_preference_features(self, features: np.ndarray):
        """构建偏好特征"""
        # 技能水平编码
        skill_map = {'beginner': 0.0, 'intermediate': 0.5, 'advanced': 0.8, 'expert': 1.0}
        features[32] = skill_map.get(self.preferences.skill_level, 0.5)
        
        # 复杂度偏好编码
        complex_map = {'low': 0.0, 'medium': 0.5, 'high': 1.0}
        features[33] = complex_map.get(self.preferences.preferred_complexity, 0.5)
        
        # 偏好强度
        total_preference = sum(self.preferences.agent_preferences.values())
        features[34] = min(1.0, total_preference / 100)
    
    def _build_interaction_features(self, features: np.ndarray):
        """构建交互特征"""
        # 交互数量
        total_interactions = len(self.behaviors)
        features[64] = min(1.0, total_interactions / 200)
        
        # 物品类型偏好
        type_prefs = self._get_preferred_item_types()
        features[65] = type_prefs.get('agent', 0)
        features[66] = type_prefs.get('pipeline', 0)
        features[67] = type_prefs.get('template', 0)
        features[68] = type_prefs.get('tutorial', 0)
    
    def _build_temporal_features(self, features: np.ndarray):
        """构建时间特征"""
        if not self.behaviors:
            return
            
        # 最近活跃度
        recent_behaviors = [b for b in self.behaviors 
                          if b.timestamp >= datetime.now() - timedelta(days=7)]
        features[96] = len(recent_behaviors) / max(1, len(self.behaviors))
        
        # 时段偏好
        period = self._get_most_active_period()
        period_map = {'morning': 0.0, 'afternoon': 0.33, 'evening': 0.66, 'night': 1.0}
        features[97] = period_map.get(period, 0.5)
        
        # 平均持续时间
        durations = [b.duration for b in self.behaviors if b.duration > 0]
        avg_duration = np.mean(durations) if durations else 0
        features[98] = min(1.0, avg_duration / 600)  # 归一化到10分钟
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """计算余弦相似度"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return dot_product / (norm1 * norm2)
    
    def to_dict(self) -> Dict:
        """导出用户画像"""
        return {
            'user_id': self.user_id,
            'preferences': {
                'agent_preferences': self.preferences.agent_preferences,
                'pipeline_preferences': self.preferences.pipeline_preferences,
                'template_preferences': self.preferences.template_preferences,
                'tutorial_preferences': self.preferences.tutorial_preferences,
                'skill_level': self.preferences.skill_level,
                'preferred_complexity': self.preferences.preferred_complexity
            },
            'feature_vector': self.feature_vector.tolist() if self.feature_vector is not None else None,
            'behavior_analysis': self.analyze_behaviors()
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'UserProfile':
        """从字典加载用户画像"""
        profile = cls(data['user_id'])
        profile.preferences = UserPreferences(**data.get('preferences', {}))
        if data.get('feature_vector'):
            profile.feature_vector = np.array(data['feature_vector'])
        return profile
