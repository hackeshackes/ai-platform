"""
优化建议器 - Optimization Recommender v12

功能:
- 建议生成
- 优先级排序
- 影响评估
- 实施跟踪
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from collections import defaultdict

from .performance_analyzer import PerformanceReport, Bottleneck, MetricType
from .config import PerformanceConfig, OptimizationStrategy

logger = logging.getLogger(__name__)


class RecommendationPriority(Enum):
    """建议优先级"""
    CRITICAL = 1  # 关键
    HIGH = 2      # 高
    MEDIUM = 3    # 中
    LOW = 4       # 低


class ImpactLevel(Enum):
    """影响级别"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class OptimizationRecommendation:
    """优化建议"""
    id: str
    title: str
    description: str
    category: str
    priority: RecommendationPriority
    impact: ImpactLevel
    estimated_improvement: float  # 预期提升 (0-1)
    implementation_ease: str  # 实现难度: easy/medium/hard
    affected_components: List[str]
    prerequisites: List[str]
    steps: List[str]
    estimated_time: str  # 预计实施时间
    risks: List[str]
    rollback_plan: str
    created_at: float
    implemented: bool = False
    implemented_at: Optional[float] = None
    results: Optional[Dict[str, Any]] = None


@dataclass
class ImpactAssessment:
    """影响评估"""
    recommendation_id: str
    performance_impact: Dict[str, float]
    resource_impact: Dict[str, float]
    risk_level: str
    dependencies: List[str]
    compatibility: Dict[str, bool]
    rollback_feasibility: str


class OptimizationRecommender:
    """
    优化建议器

    基于性能分析结果生成优化建议，提供优先级排序、影响评估和实施跟踪。
    """

    def __init__(
        self,
        config: Optional[PerformanceConfig] = None,
        target_improvement: float = 0.3
    ):
        """
        初始化优化建议器

        Args:
            config: 性能配置
            target_improvement: 目标改进比例
        """
        self.config = config or PerformanceConfig()
        self.target_improvement = target_improvement
        self._recommendations: List[OptimizationRecommendation] = []
        self._implementation_history: Dict[str, Dict] = {}

        # 建议规则
        self._recommendation_rules = self._init_recommendation_rules()

    def _init_recommendation_rules(self) -> Dict[str, Dict]:
        """初始化建议规则"""
        return {
            "high_cpu": {
                "conditions": [("cpu", ">", 80)],
                "recommendations": [
                    {
                        "title": "优化CPU密集型操作",
                        "description": "CPU使用率过高，建议识别并优化CPU密集型代码",
                        "category": "performance",
                        "estimated_improvement": 0.25,
                        "implementation_ease": "medium",
                        "steps": [
                            "使用性能分析工具识别CPU热点",
                            "优化算法复杂度",
                            "考虑使用更高效的数据结构",
                            "考虑增加CPU资源"
                        ],
                        "estimated_time": "2-4小时",
                        "risks": ["可能引入新的性能问题"],
                        "rollback_plan": "回滚代码更改"
                    },
                    {
                        "title": "增加CPU资源",
                        "description": "通过水平或垂直扩展增加CPU资源",
                        "category": "infrastructure",
                        "estimated_improvement": 0.3,
                        "implementation_ease": "easy",
                        "steps": [
                            "评估当前资源使用情况",
                            "选择合适的扩展策略",
                            "执行资源扩展",
                            "监控效果"
                        ],
                        "estimated_time": "30分钟",
                        "risks": ["成本增加"],
                        "rollback_plan": "缩减资源"
                    }
                ]
            },
            "high_memory": {
                "conditions": [("memory", ">", 85)],
                "recommendations": [
                    {
                        "title": "内存使用优化",
                        "description": "内存使用率过高，建议优化内存使用",
                        "category": "performance",
                        "estimated_improvement": 0.2,
                        "implementation_ease": "medium",
                        "steps": [
                            "分析内存使用分布",
                            "识别内存泄漏",
                            "优化数据结构",
                            "启用对象池"
                        ],
                        "estimated_time": "1-2小时",
                        "risks": ["可能影响功能"],
                        "rollback_plan": "回滚更改"
                    }
                ]
            },
            "high_latency": {
                "conditions": [("latency", ">", 500)],
                "recommendations": [
                    {
                        "title": "API响应优化",
                        "description": "API响应时间过长，建议进行优化",
                        "category": "api",
                        "estimated_improvement": 0.4,
                        "implementation_ease": "medium",
                        "steps": [
                            "分析慢请求",
                            "添加缓存层",
                            "优化数据库查询",
                            "启用压缩"
                        ],
                        "estimated_time": "2-3小时",
                        "risks": ["可能影响数据一致性"],
                        "rollback_plan": "禁用缓存/压缩"
                    }
                ]
            },
            "low_throughput": {
                "conditions": [("throughput", "<", 100)],
                "recommendations": [
                    {
                        "title": "提高系统吞吐量",
                        "description": "系统吞吐量不足，建议进行扩展和优化",
                        "category": "infrastructure",
                        "estimated_improvement": 0.5,
                        "implementation_ease": "easy",
                        "steps": [
                            "增加实例数量",
                            "优化连接池配置",
                            "启用批处理"
                        ],
                        "estimated_time": "1小时",
                        "risks": ["成本增加"],
                        "rollback_plan": "缩减实例"
                    }
                ]
            },
            "slow_query": {
                "conditions": [("database", ">", 1000)],
                "recommendations": [
                    {
                        "title": "数据库查询优化",
                        "description": "数据库查询性能不佳，建议优化索引和查询",
                        "category": "database",
                        "estimated_improvement": 0.5,
                        "implementation_easy": "medium",
                        "steps": [
                            "分析慢查询日志",
                            "添加适当索引",
                            "优化查询语句",
                            "考虑读写分离"
                        ],
                        "estimated_time": "1-2小时",
                        "risks": ["索引维护开销"],
                        "rollback_plan": "删除索引"
                    }
                ]
            },
            "low_cache_hit": {
                "conditions": [("cache", "<", 0.5)],
                "recommendations": [
                    {
                        "title": "缓存优化",
                        "description": "缓存命中率过低，建议优化缓存策略",
                        "category": "cache",
                        "estimated_improvement": 0.3,
                        "implementation_ease": "easy",
                        "steps": [
                            "分析缓存使用模式",
                            "调整TTL设置",
                            "增加缓存容量",
                            "实施缓存预热"
                        ],
                        "estimated_time": "30分钟",
                        "risks": ["内存使用增加"],
                        "rollback_plan": "恢复原配置"
                    }
                ]
            }
        }

    def get_suggestions(
        self,
        current_metrics: Dict[str, Dict[str, float]],
        target_improvement: Optional[float] = None,
        constraints: Optional[Dict[str, Any]] = None
    ) -> List[OptimizationRecommendation]:
        """
        获取优化建议

        Args:
            current_metrics: 当前指标
            target_improvement: 目标改进
            constraints: 约束条件

        Returns:
            List[OptimizationRecommendation]: 优化建议列表
        """
        if target_improvement is None:
            target_improvement = self.target_improvement

        constraints = constraints or {}

        # 基于指标生成建议
        suggestions = []

        for metric_type, values in current_metrics.items():
            avg_value = values.get("avg", 0)
            suggestions.extend(self._generate_suggestions_for_metric(
                metric_type,
                avg_value,
                target_improvement,
                constraints
            ))

        # 优先级排序
        suggestions = self._prioritize_suggestions(suggestions)

        # 去重并保存
        self._recommendations = suggestions

        return suggestions

    def _generate_suggestions_for_metric(
        self,
        metric_type: str,
        value: float,
        target_improvement: float,
        constraints: Dict[str, Any]
    ) -> List[OptimizationRecommendation]:
        """为特定指标生成建议"""
        suggestions = []

        for rule_name, rule in self._recommendation_rules.items():
            for condition in rule["conditions"]:
                cond_metric, operator, threshold = condition

                if cond_metric == metric_type:
                    should_suggest = False
                    if operator == ">" and value > threshold:
                        should_suggest = True
                    elif operator == "<" and value < threshold:

                        should_suggest = True

                    if should_suggest:
                        for rec_template in rule["recommendations"]:
                            suggestion = self._create_recommendation(
                                rule_name,
                                rec_template,
                                metric_type,
                                value
                            )
                            suggestions.append(suggestion)

        return suggestions

    def _create_recommendation(
        self,
        rule_name: str,
        template: Dict,
        metric_type: str,
        current_value: float
    ) -> OptimizationRecommendation:
        """创建建议"""
        estimated_improvement = template.get("estimated_improvement", 0.2)

        # 根据当前值调整预期改进
        if metric_type in ["cpu", "memory", "latency"]:
            # 降低型指标
            potential_improvement = min(estimated_improvement, (current_value - 50) / 100)
        else:
            # 增加型指标
            potential_improvement = estimated_improvement

        return OptimizationRecommendation(
            id=f"rec_{rule_name}_{int(time.time())}",
            title=template["title"],
            description=template["description"],
            category=template["category"],
            priority=self._calculate_priority(potential_improvement),
            impact=self._calculate_impact(potential_improvement),
            estimated_improvement=potential_improvement,
            implementation_ease=template.get("implementation_ease", "medium"),
            affected_components=[metric_type],
            prerequisites=template.get("prerequisites", []),
            steps=template.get("steps", []),
            estimated_time=template.get("estimated_time", "1小时"),
            risks=template.get("risks", []),
            rollback_plan=template.get("rollback_plan", "回滚更改"),
            created_at=time.time()
        )

    def _calculate_priority(self, improvement: float) -> RecommendationPriority:
        """计算优先级"""
        if improvement >= 0.4:
            return RecommendationPriority.CRITICAL
        elif improvement >= 0.3:
            return RecommendationPriority.HIGH
        elif improvement >= 0.2:
            return RecommendationPriority.MEDIUM
        else:
            return RecommendationPriority.LOW

    def _calculate_impact(self, improvement: float) -> ImpactLevel:
        """计算影响级别"""
        if improvement >= 0.4:
            return ImpactLevel.HIGH
        elif improvement >= 0.2:
            return ImpactLevel.MEDIUM
        else:
            return ImpactLevel.LOW

    def _prioritize_suggestions(
        self,
        suggestions: List[OptimizationRecommendation]
    ) -> List[OptimizationRecommendation]:
        """对建议进行优先级排序"""
        # 按优先级分组
        priority_groups = defaultdict(list)

        for suggestion in suggestions:
            priority_groups[suggestion.priority].append(suggestion)

        # 按优先级排序
        sorted_suggestions = []
        for priority in sorted(priority_groups.keys()):
            # 同优先级内按影响级别排序
            suggestions_in_group = sorted(
                priority_groups[priority],
                key=lambda x: (
                    ImpactLevel[x.impact.upper()].value
                    if isinstance(x.impact, str) else x.impact.value
                ),
                reverse=True
            )
            sorted_suggestions.extend(suggestions_in_group)

        return sorted_suggestions

    def assess_impact(
        self,
        recommendation: OptimizationRecommendation
    ) -> ImpactAssessment:
        """评估建议的影响"""
        # 性能影响评估
        performance_impact = {
            "cpu": 0.1 if recommendation.category == "performance" else 0,
            "memory": 0.1 if recommendation.category == "performance" else 0,
            "latency": -0.2 if "latency" in recommendation.title.lower() else 0,
            "throughput": 0.3 if "throughput" in recommendation.title.lower() else 0
        }

        # 资源影响评估
        resource_impact = {
            "cpu": recommendation.estimated_improvement * 0.5,
            "memory": recommendation.estimated_improvement * 0.3,
            "disk": recommendation.estimated_improvement * 0.1,
            "network": recommendation.estimated_improvement * 0.2
        }

        # 风险评估
        risk_level = "high" if len(recommendation.risks) > 2 else "medium"

        return ImpactAssessment(
            recommendation_id=recommendation.id,
            performance_impact=performance_impact,
            resource_impact=resource_impact,
            risk_level=risk_level,
            dependencies=recommendation.prerequisites,
            compatibility={"python": True, "database": True},
            rollback_feasibility="high" if recommendation.rollback_plan else "low"
        )

    def track_implementation(
        self,
        recommendation_id: str,
        status: str,
        results: Optional[Dict[str, Any]] = None
    ) -> bool:
        """跟踪实施进度"""
        for recommendation in self._recommendations:
            if recommendation.id == recommendation_id:
                recommendation.implemented = status == "completed"
                if status == "completed":
                    recommendation.implemented_at = time.time()
                recommendation.results = results

                self._implementation_history[recommendation_id] = {
                    "status": status,
                    "timestamp": time.time(),
                    "results": results
                }
                return True

        return False

    def get_recommendations(
        self,
        category: Optional[str] = None,
        priority: Optional[RecommendationPriority] = None,
        implemented: Optional[bool] = None
    ) -> List[OptimizationRecommendation]:
        """获取建议列表"""
        results = self._recommendations

        if category:
            results = [r for r in results if r.category == category]

        if priority:
            results = [r for r in results if r.priority == priority]

        if implemented is not None:
            results = [r for r in results if r.implemented == implemented]

        return results

    def get_implementation_summary(self) -> Dict[str, Any]:
        """获取实施摘要"""
        total = len(self._recommendations)
        implemented = sum(1 for r in self._recommendations if r.implemented)
        pending = total - implemented

        total_improvement = sum(
            r.estimated_improvement
            for r in self._recommendations
            if r.implemented
        )

        return {
            "total_recommendations": total,
            "implemented": implemented,
            "pending": pending,
            "implementation_rate": implemented / total if total > 0 else 0,
            "achieved_improvement": total_improvement,
            "target_improvement": self.target_improvement,
            "progress": min(1.0, total_improvement / self.target_improvement)
        }

    def generate_report(self) -> Dict[str, Any]:
        """生成报告"""
        summary = self.get_implementation_summary()

        recommendations_by_priority = defaultdict(list)
        for rec in self._recommendations:
            recommendations_by_priority[rec.priority.name].append({
                "id": rec.id,
                "title": rec.title,
                "category": rec.category,
                "estimated_improvement": rec.estimated_improvement,
                "implemented": rec.implemented
            })

        return {
            "generated_at": time.time(),
            "summary": summary,
            "recommendations_by_priority": dict(recommendations_by_priority),
            "implementation_history": self._implementation_history
        }
