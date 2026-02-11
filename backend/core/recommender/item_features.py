"""
物品特征模块 - item_features.py

负责物品属性提取、标签处理和特征工程
"""

import numpy as np
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from collections import defaultdict
import hashlib


@dataclass
class ItemAttributes:
    """物品属性"""
    item_id: str
    item_type: str  # agent, pipeline, template, tutorial
    name: str
    description: str = ""
    tags: Set[str] = field(default_factory=set)
    categories: Set[str] = field(default_factory=set)
    metadata: Dict = field(default_factory=dict)
    
    # Agent特有
    scenarios: Set[str] = field(default_factory=set)
    capabilities: Set[str] = field(default_factory=set)
    
    # Pipeline特有
    complexity: str = "medium"
    task_types: Set[str] = field(default_factory=set)
    input_formats: Set[str] = field(default_factory=set)
    output_formats: Set[str] = field(default_factory=set)
    
    # Template特有
    industry: str = "general"
    use_cases: Set[str] = field(default_factory=set)
    
    # Tutorial特有
    difficulty: str = "beginner"
    prerequisites: Set[str] = field(default_factory=set)
    topics: Set[str] = field(default_factory=set)


class ItemFeatures:
    """物品特征类"""
    
    def __init__(self):
        self.items: Dict[str, ItemAttributes] = {}
        self.tag_index: Dict[str, Set[str]] = defaultdict(set)
        self.category_index: Dict[str, Set[str]] = defaultdict(set)
        self.feature_matrix: Optional[np.ndarray] = None
        self.item_order: List[str] = []
        
        # 特征维度配置
        self.feature_dim = 256
        
    def add_item(self, item: ItemAttributes):
        """添加物品"""
        self.items[item.item_id] = item
        self.item_order.append(item.item_id)
        
        # 更新索引
        for tag in item.tags:
            self.tag_index[tag].add(item.item_id)
        for cat in item.categories:
            self.category_index[cat].add(item.item_id)
    
    def extract_features(self, item_id: str) -> np.ndarray:
        """提取单个物品特征"""
        item = self.items.get(item_id)
        if item is None:
            return np.zeros(self.feature_dim)
            
        features = np.zeros(self.feature_dim)
        
        # 基本属性特征 (0-31)
        self._extract_basic_features(item, features)
        
        # 类型特征 (32-63)
        self._extract_type_features(item, features)
        
        # 标签特征 (64-95)
        self._extract_tag_features(item, features)
        
        # 元数据特征 (96-159)
        self._extract_metadata_features(item, features)
        
        # 文本特征 (160-255)
        self._extract_text_features(item, features)
        
        return features
    
    def build_feature_matrix(self) -> np.ndarray:
        """构建特征矩阵"""
        self.feature_matrix = np.zeros((len(self.items), self.feature_dim))
        
        for idx, item_id in enumerate(self.item_order):
            self.feature_matrix[idx] = self.extract_features(item_id)
            
        return self.feature_matrix
    
    def get_item_similarity(self, item_id1: str, item_id2: str) -> float:
        """计算两物品相似度"""
        features1 = self.extract_features(item_id1)
        features2 = self.extract_features(item_id2)
        
        return self._cosine_similarity(features1, features2)
    
    def find_similar_items(self, item_id: str, top_k: int = 10) -> List[tuple]:
        """查找相似物品"""
        if self.feature_matrix is None:
            self.build_feature_matrix()
            
        try:
            idx = self.item_order.index(item_id)
        except ValueError:
            return []
            
        target_features = self.feature_matrix[idx]
        
        # 计算余弦相似度
        similarities = []
        for i, features in enumerate(self.feature_matrix):
            if i != idx:
                sim = self._cosine_similarity(target_features, features)
                similarities.append((self.item_order[i], sim))
        
        # 排序返回top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def get_items_by_tag(self, tag: str) -> Set[str]:
        """获取指定标签的物品"""
        return self.tag_index.get(tag, set())
    
    def get_items_by_category(self, category: str) -> Set[str]:
        """获取指定类别的物品"""
        return self.category_index.get(category, set())
    
    def get_items_by_type(self, item_type: str) -> List[str]:
        """获取指定类型的物品"""
        return [item_id for item_id, item in self.items.items() 
                if item.item_type == item_type]
    
    def get_items_by_complexity(self, complexity: str) -> List[str]:
        """获取指定复杂度的Pipeline物品"""
        return [item_id for item_id, item in self.items.items() 
                if item.item_type == 'pipeline' and item.complexity == complexity]
    
    def get_items_by_difficulty(self, difficulty: str) -> List[str]:
        """获取指定难度的教程"""
        return [item_id for item_id, item in self.items.items() 
                if item.item_type == 'tutorial' and item.difficulty == difficulty]
    
    # 特征提取内部方法
    def _extract_basic_features(self, item: ItemAttributes, features: np.ndarray):
        """提取基本属性特征"""
        # 名称长度
        features[0] = min(1.0, len(item.name) / 100)
        
        # 描述长度
        features[1] = min(1.0, len(item.description) / 500)
        
        # 标签数量
        features[2] = min(1.0, len(item.tags) / 10)
        
        # 类别数量
        features[3] = min(1.0, len(item.categories) / 5)
    
    def _extract_type_features(self, item: ItemAttributes, features: np.ndarray):
        """提取类型特征"""
        type_map = {'agent': 0, 'pipeline': 1, 'template': 2, 'tutorial': 3}
        type_onehot = np.zeros(4)
        if item.item_type in type_map:
            type_onehot[type_map[item.item_type]] = 1
            
        features[32:36] = type_onehot
        
        # 复杂度编码 (Pipeline)
        complex_map = {'low': 0.0, 'medium': 0.5, 'high': 1.0}
        features[36] = complex_map.get(item.complexity, 0.5)
        
        # 难度编码 (Tutorial)
        diff_map = {'beginner': 0.0, 'intermediate': 0.5, 'advanced': 1.0}
        features[37] = diff_map.get(item.difficulty, 0.0)
        
        # 行业编码 (Template)
        industry_map = {
            'general': 0.0, 'technology': 0.2, 'finance': 0.4,
            'healthcare': 0.6, 'education': 0.8, 'entertainment': 1.0
        }
        features[38] = industry_map.get(item.industry, 0.0)
    
    def _extract_tag_features(self, item: ItemAttributes, features: np.ndarray):
        """提取标签特征"""
        # 使用标签的哈希值来生成特征
        for i, tag in enumerate(list(item.tags)[:32]):
            tag_hash = int(hashlib.md5(tag.encode()).hexdigest()[:8], 16)
            features[64 + i] = (tag_hash % 100) / 100.0
    
    def _extract_metadata_features(self, item: ItemAttributes, features: np.ndarray):
        """提取元数据特征"""
        idx = 96
        
        if item.item_type == 'agent':
            # Agent特有特征
            features[idx] = min(1.0, len(item.capabilities) / 10)
            features[idx + 1] = min(1.0, len(item.scenarios) / 10)
            idx += 2
            
        elif item.item_type == 'pipeline':
            # Pipeline特有特征
            complex_map = {'low': 0.0, 'medium': 0.5, 'high': 1.0}
            features[idx] = complex_map.get(item.complexity, 0.5)
            features[idx + 1] = min(1.0, len(item.task_types) / 5)
            features[idx + 2] = min(1.0, len(item.input_formats) / 5)
            features[idx + 3] = min(1.0, len(item.output_formats) / 5)
            idx += 4
            
        elif item.item_type == 'template':
            # Template特有特征
            features[idx] = min(1.0, len(item.use_cases) / 10)
            idx += 1
            
        elif item.item_type == 'tutorial':
            # Tutorial特有特征
            features[idx] = min(1.0, len(item.prerequisites) / 5)
            features[idx + 1] = min(1.0, len(item.topics) / 10)
            idx += 2
    
    def _extract_text_features(self, item: ItemAttributes, features: np.ndarray):
        """提取文本特征"""
        idx = 160
        
        # 描述TF-IDF风格特征
        text = (item.name + " " + item.description).lower()
        words = text.split()
        
        # 关键词特征
        keywords = ['intelligent', 'advanced', 'simple', 'fast', 'efficient', 
                   'professional', 'basic', 'comprehensive', 'automated', 'custom']
        
        for keyword in keywords:
            if idx >= self.feature_dim:
                break
            features[idx] = 1.0 if keyword in text else 0.0
            idx += 1
        
        # 情感/语气特征
        positive_words = ['easy', 'powerful', 'flexible', 'quick', 'simple']
        negative_words = ['complex', 'difficult', 'slow', 'limited']
        
        pos_count = sum(1 for word in words if word in positive_words)
        neg_count = sum(1 for word in words if word in negative_words)
        
        if idx < self.feature_dim:
            features[idx] = min(1.0, pos_count / 5)
        if idx + 1 < self.feature_dim:
            features[idx + 1] = min(1.0, neg_count / 5)
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """计算余弦相似度"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return dot_product / (norm1 * norm2)
    
    def get_feature_vector_size(self) -> int:
        """获取特征向量维度"""
        return self.feature_dim
    
    def export_features(self) -> Dict[str, List[float]]:
        """导出所有物品特征"""
        if self.feature_matrix is None:
            self.build_feature_matrix()
            
        return {
            item_id: self.feature_matrix[idx].tolist()
            for idx, item_id in enumerate(self.item_order)
        }
    
    def import_features(self, features: Dict[str, List[float]]):
        """导入特征"""
        for item_id, feature_list in features.items():
            if item_id in self.items:
                self.feature_matrix = np.array(feature_list)
                break
