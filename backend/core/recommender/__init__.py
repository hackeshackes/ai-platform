"""
推荐系统核心模块

智能推荐系统 - v12
提供Agent、Pipeline、模板和教程的个性化推荐
"""

__version__ = "1.0.0"
__author__ = "AI Platform Team"

from .user_profile import UserProfile
from .item_features import ItemFeatures
from .collaborative_filtering import CollaborativeFiltering
from .content_based import ContentBasedRecommender
from .hybrid_recommender import HybridRecommender
from .metrics import RecommenderMetrics

__all__ = [
    'UserProfile',
    'ItemFeatures', 
    'CollaborativeFiltering',
    'ContentBasedRecommender',
    'HybridRecommender',
    'RecommenderMetrics'
]
