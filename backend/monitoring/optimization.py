"""
Optimization Engine - AI Platform v4

智能优化建议引擎 - 基于监控数据提供成本和性能优化建议
"""
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict
import json


class OptimizationCategory(Enum):
    """优化类别"""
    COST_REDUCTION = "cost_reduction"
    PERFORMANCE = "performance"
    TOKEN_USAGE = "token_usage"
    PROVIDER_ROUTING = "provider_routing"
    CACHING = "caching"
    BATCHING = "batching"


class OptimizationPriority(Enum):
    """优化优先级"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class OptimizationRecommendation:
    """优化建议"""
    recommendation_id: str
    category: OptimizationCategory
    priority: OptimizationPriority
    title: str
    description: str
    potential_savings: float
    implementation_difficulty: str
    estimated_impact: str
    actionable_steps: List[str]
    related_metrics: Dict[str, Any]
    created_at: datetime
    
    def to_dict(self) -> Dict:
        return {
            "recommendation_id": self.recommendation_id,
            "category": self.category.value,
            "priority": self.priority.value,
            "title": self.title,
            "description": self.description,
            "potential_savings": self.potential_savings,
            "implementation_difficulty": self.implementation_difficulty,
            "estimated_impact": self.estimated_impact,
            "actionable_steps": self.actionable_steps,
            "related_metrics": self.related_metrics,
            "created_at": self.created_at.isoformat()
        }


@dataclass
class UsagePattern:
    """使用模式分析"""
    peak_hours: List[int]
    low_usage_hours: List[int]
    average_daily_cost: float
    average_daily_tokens: int
    top_providers_by_usage: List[Dict]
    top_models_by_usage: List[Dict]
    seasonality_factor: float
    
    
class OptimizationEngine:
    """
    智能优化建议引擎
    
    功能:
    - 分析使用模式和成本结构
    - 识别优化机会
    - 生成可操作的优化建议
    - 估算节省潜力
    """
    
    def __init__(self):
        self._history_data: List[Dict] = []
        self._recommendations: List[OptimizationRecommendation] = []
        self._usage_patterns: Optional[UsagePattern] = None
        self._last_analysis: Optional[datetime] = None
        
    # ============ 数据收集 ============
    
    async def ingest_metrics(self, metrics: Dict[str, Any]):
        """收集指标数据用于分析"""
        self._history_data.append({
            "timestamp": datetime.now().isoformat(),
            **metrics
        })
        
        # 保留最近30天的数据
        cutoff = datetime.now() - timedelta(days=30)
        self._history_data = [
            m for m in self._history_data
            if datetime.fromisoformat(m["timestamp"]) >= cutoff
        ]
    
    async def analyze_usage_patterns(self) -> UsagePattern:
        """分析使用模式"""
        # 简化分析：基于历史数据
        costs_by_hour = defaultdict(list)
        tokens_by_hour = defaultdict(list)
        
        for data in self._history_data:
            # 假设有hour字段
            hour = datetime.fromisoformat(data.get("timestamp", datetime.now().isoformat())).hour
            if "cost" in data:
                costs_by_hour[hour].append(data["cost"])
            if "tokens" in data.get("usage", {}):
                tokens_by_hour[hour].append(data["usage"]["total_tokens"])
        
        # 计算峰值和非峰值时段
        avg_costs = {h: sum(costs) / max(len(costs), 1) for h, costs in costs_by_hour.items()}
        peak_hours = sorted(avg_costs, key=avg_costs.get, reverse=True)[:4]
        low_usage_hours = sorted(avg_costs, key=avg_costs.get)[:4]
        
        # 计算平均值
        total_cost = sum(d.get("cost", 0) for d in self._history_data)
        total_tokens = sum(
            d.get("usage", {}).get("total_tokens", 0) 
            for d in self._history_data
        )
        days = max(1, len(set(
            datetime.fromisoformat(d["timestamp"]).date() 
            for d in self._history_data
        )))
        
        self._usage_patterns = UsagePattern(
            peak_hours=peak_hours,
            low_usage_hours=low_usage_hours,
            average_daily_cost=total_cost / days,
            average_daily_tokens=total_tokens / days,
            top_providers_by_usage=[],  # 从数据中提取
            top_models_by_usage=[],      # 从数据中提取
            seasonality_factor=1.0
        )
        
        self._last_analysis = datetime.now()
        return self._usage_patterns
    
    # ============ 优化建议生成 ============
    
    async def generate_recommendations(self) -> List[OptimizationRecommendation]:
        """生成优化建议"""
        recommendations = []
        
        # 确保有使用模式数据
        if not self._usage_patterns:
            await self.analyze_usage_patterns()
        
        # 生成各类优化建议
        recommendations.extend(await self._analyze_cost_optimization())
        recommendations.extend(await self._analyze_performance())
        recommendations.extend(await self._analyze_token_usage())
        recommendations.extend(await self._analyze_provider_routing())
        recommendations.extend(await self._analyze_caching_opportunities())
        recommendations.extend(await self._analyze_batching())
        
        self._recommendations = recommendations
        return recommendations
    
    async def _analyze_cost_optimization(self) -> List[OptimizationRecommendation]:
        """分析成本优化机会"""
        recommendations = []
        
        # 检查是否有过高的单次请求成本
        avg_cost = self._usage_patterns.average_daily_cost
        if avg_cost > 50:
            recommendations.append(OptimizationRecommendation(
                recommendation_id=f"cost_high_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                category=OptimizationCategory.COST_REDUCTION,
                priority=OptimizationPriority.HIGH,
                title="High Daily Cost Detected",
                description=f"Your average daily cost of ${avg_cost:.2f} is significantly above typical usage. Consider reviewing your request patterns.",
                potential_savings=avg_cost * 0.2,
                implementation_difficulty="medium",
                estimated_impact="20% cost reduction",
                actionable_steps=[
                    "Review top expensive requests in the last week",
                    "Identify requests with high token counts",
                    "Consider implementing request size limits",
                    "Use cheaper models for non-critical tasks"
                ],
                related_metrics={
                    "average_daily_cost": avg_cost,
                    "daily_cost_threshold": 50
                },
                created_at=datetime.now()
            ))
        
        # 检查提供商成本分布
        provider_costs = defaultdict(float)
        for data in self._history_data:
            if "cost_by_provider" in data:
                for provider, cost in data["cost_by_provider"].items():
                    provider_costs[provider] += cost
        
        # 建议成本优化
        if len(provider_costs) > 1:
            sorted_providers = sorted(provider_costs.items(), key=lambda x: x[1], reverse=True)
            top_provider = sorted_providers[0]
            
            if top_provider[1] > sum(provider_costs.values()) * 0.7:
                recommendations.append(OptimizationRecommendation(
                    recommendation_id=f"provider_concentration_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                    category=OptimizationCategory.PROVIDER_ROUTING,
                    priority=OptimizationPriority.MEDIUM,
                    title="Provider Concentration Risk",
                    description=f"{top_provider[0]} accounts for {70}% of your costs. Consider diversifying.",
                    potential_savings=top_provider[1] * 0.15,
                    implementation_difficulty="easy",
                    estimated_impact="15% cost reduction",
                    actionable_steps=[
                        f"Evaluate alternative providers for {top_provider[0]} use cases",
                        "Set up A/B testing between providers",
                        "Configure automatic failover",
                        "Implement provider rotation strategy"
                    ],
                    related_metrics={
                        "top_provider": top_provider[0],
                        "top_provider_cost": top_provider[1],
                        "concentration": 0.7
                    },
                    created_at=datetime.now()
                ))
        
        return recommendations
    
    async def _analyze_performance(self) -> List[OptimizationRecommendation]:
        """分析性能优化"""
        recommendations = []
        
        # 检查延迟数据
        latencies = []
        for data in self._history_data:
            if "latency" in data:
                latencies.append(data["latency"])
        
        if latencies:
            avg_latency = sum(latencies) / len(latencies)
            p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]
            
            if p95_latency > 5000:  # 5秒
                recommendations.append(OptimizationRecommendation(
                    recommendation_id=f"latency_high_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                    category=OptimizationCategory.PERFORMANCE,
                    priority=OptimizationPriority.HIGH,
                    title="High P95 Latency Detected",
                    description=f"P95 latency of {p95_latency:.0f}ms exceeds recommended 5000ms threshold.",
                    potential_savings=0,  # 不直接省钱，但提升体验
                    implementation_difficulty="medium",
                    estimated_impact="50% latency reduction",
                    actionable_steps=[
                        "Review requests with longest response times",
                        "Consider faster provider for time-sensitive tasks",
                        "Implement request timeout limits",
                        "Use streaming for long responses",
                        "Cache frequent queries"
                    ],
                    related_metrics={
                        "avg_latency_ms": avg_latency,
                        "p95_latency_ms": p95_latency,
                        "threshold_ms": 5000
                    },
                    created_at=datetime.now()
                ))
        
        return recommendations
    
    async def _analyze_token_usage(self) -> List[OptimizationRecommendation]:
        """分析Token使用优化"""
        recommendations = []
        
        total_prompt = 0
        total_completion = 0
        
        for data in self._history_data:
            usage = data.get("usage", {})
            total_prompt += usage.get("prompt_tokens", 0)
            total_completion += usage.get("completion_tokens", 0)
        
        if total_prompt > 0:
            completion_ratio = total_completion / (total_prompt + total_completion)
            
            if completion_ratio > 0.5:
                recommendations.append(OptimizationRecommendation(
                    recommendation_id=f"completion_heavy_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                    category=OptimizationCategory.TOKEN_USAGE,
                    priority=OptimizationPriority.MEDIUM,
                    title="High Completion Token Usage",
                    description=f"Completion tokens make up {completion_ratio*100:.1f}% of total usage. This indicates verbose model responses.",
                    potential_savings=0.25,
                    implementation_difficulty="easy",
                    estimated_impact="25% token reduction",
                    actionable_steps=[
                        "Add response length constraints to prompts",
                        "Use max_tokens parameter",
                        "Implement response summarization",
                        "Fine-tune temperature for more concise outputs",
                        "Use response format specifications"
                    ],
                    related_metrics={
                        "completion_ratio": completion_ratio,
                        "prompt_tokens": total_prompt,
                        "completion_tokens": total_completion
                    },
                    created_at=datetime.now()
                ))
        
        return recommendations
    
    async def _analyze_provider_routing(self) -> List[OptimizationRecommendation]:
        """分析提供商路由优化"""
        recommendations = []
        
        # 基于时间模式建议路由
        peak_hours = self._usage_patterns.peak_hours
        low_usage_hours = self._usage_patterns.low_usage_hours
        
        recommendations.append(OptimizationRecommendation(
            recommendation_id=f"smart_routing_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            category=OptimizationCategory.PROVIDER_ROUTING,
            priority=OptimizationPriority.MEDIUM,
            title="Implement Smart Provider Routing",
            description="Route requests based on time patterns to optimize cost and performance.",
            potential_savings=0.15,
            implementation_difficulty="medium",
            estimated_impact="15% cost reduction",
            actionable_steps=[
                "Identify peak hours: " + ", ".join(map(str, peak_hours)),
                "Configure cheaper providers for off-peak",
                "Use premium providers only during peak",
                "Set up automatic routing rules",
                "Monitor and adjust routing weights"
            ],
            related_metrics={
                "peak_hours": peak_hours,
                "off_peak_hours": low_usage_hours
            },
            created_at=datetime.now()
        ))
        
        return recommendations
    
    async def _analyze_caching_opportunities(self) -> List[OptimizationRecommendation]:
        """分析缓存优化机会"""
        recommendations = []
        
        recommendations.append(OptimizationRecommendation(
            recommendation_id=f"caching_strategy_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            category=OptimizationCategory.CACHING,
            priority=OptimizationPriority.HIGH,
            title="Implement Response Caching",
            description="Cache frequent queries to reduce redundant API calls.",
            potential_savings=0.3,
            implementation_difficulty="medium",
            estimated_impact="30% cost reduction",
            actionable_steps=[
                "Identify repeated queries in your logs",
                "Implement semantic caching layer",
                "Set appropriate TTL for cached responses",
                "Monitor cache hit rate",
                "Invalidate cache on model updates"
            ],
            related_metrics={
                "estimated_cacheable_queries": 0.2,
                "recommended_ttl_seconds": 3600
            },
            created_at=datetime.now()
        ))
        
        return recommendations
    
    async def _analyze_batching(self) -> List[OptimizationRecommendation]:
        """分析批处理优化"""
        recommendations = []
        
        recommendations.append(OptimizationRecommendation(
            recommendation_id=f"batching_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            category=OptimizationCategory.BATCHING,
            priority=OptimizationPriority.LOW,
            title="Implement Request Batching",
            description="Batch multiple requests together to improve throughput.",
            potential_savings=0.1,
            implementation_difficulty="hard",
            estimated_impact="10% cost reduction, 2x throughput",
            actionable_steps=[
                "Identify batchable request patterns",
                "Implement batching middleware",
                "Configure optimal batch size",
                "Set batch timeout limits",
                "Monitor batch efficiency"
            ],
            related_metrics={
                "recommended_batch_size": 10,
                "recommended_timeout_ms": 500
            },
            created_at=datetime.now()
        ))
        
        return recommendations
    
    # ============ 建议管理 ============
    
    async def get_recommendations(
        self,
        category: Optional[OptimizationCategory] = None,
        priority: Optional[OptimizationPriority] = None,
        limit: int = 10
    ) -> List[OptimizationRecommendation]:
        """获取优化建议"""
        recommendations = self._recommendations
        
        if category:
            recommendations = [r for r in recommendations if r.category == category]
        if priority:
            recommendations = [r for r in recommendations if r.priority == priority]
        
        # 按优先级排序
        priority_order = {OptimizationPriority.HIGH: 0, OptimizationPriority.MEDIUM: 1, OptimizationPriority.LOW: 2}
        recommendations = sorted(recommendations, key=lambda r: priority_order.get(r.priority, 99))
        
        return recommendations[:limit]
    
    async def get_total_savings_potential(self) -> Dict[str, Any]:
        """获取总节省潜力"""
        total_cost_savings = sum(r.potential_savings for r in self._recommendations)
        
        return {
            "total_monthly_savings_usd": total_cost_savings,
            "recommendation_count": len(self._recommendations),
            "by_category": {},
            "by_priority": {}
        }
    
    async def dismiss_recommendation(self, recommendation_id: str) -> bool:
        """Dismiss a recommendation"""
        original_count = len(self._recommendations)
        self._recommendations = [
            r for r in self._recommendations 
            if r.recommendation_id != recommendation_id
        ]
        return len(self._recommendations) < original_count
    
    async def get_optimization_summary(self) -> Dict[str, Any]:
        """获取优化摘要"""
        return {
            "last_analysis": self._last_analysis.isoformat() if self._last_analysis else None,
            "total_recommendations": len(self._recommendations),
            "high_priority_count": len([r for r in self._recommendations if r.priority == OptimizationPriority.HIGH]),
            "total_potential_savings": sum(r.potential_savings for r in self._recommendations),
            "usage_patterns": {
                "peak_hours": self._usage_patterns.peak_hours if self._usage_patterns else [],
                "average_daily_cost": self._usage_patterns.average_daily_cost if self._usage_patterns else 0,
                "average_daily_tokens": self._usage_patterns.average_daily_tokens if self._usage_patterns else 0
            }
        }


# 创建全局优化引擎实例
optimization_engine = OptimizationEngine()


def get_optimization_engine() -> OptimizationEngine:
    """获取优化引擎实例"""
    return optimization_engine
