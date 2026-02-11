"""
智能调度系统 - 成本优化器

资源利用率分析/闲置资源回收/竞价实例/预留容量
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class InstanceType(Enum):
    ON_DEMAND = "ondemand"
    SPOT = "spot"
    RESERVED = "reserved"


@dataclass
class ResourceUsage:
    """资源使用情况"""
    resource_type: str
    used: float
    allocated: float
    unit: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class CostEntry:
    """成本条目"""
    resource_id: str
    resource_type: str
    instance_type: InstanceType
    cost: float
    duration_hours: float
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict = field(default_factory=dict)


@dataclass
class CostRecommendation:
    """成本优化推荐"""
    id: str
    type: str
    title: str
    description: str
    potential_savings: float
    confidence: float
    implementation: str
    priority: int  # 1-10, 10最高


class CostOptimizer:
    """成本优化器"""
    
    def __init__(
        self,
        spot_discount: float = 0.7,  # 竞价实例折扣
        reserved_discount: float = 0.3,  # 预留折扣
        idle_threshold: float = 0.2,  # 闲置阈值
        low_utilization_threshold: float = 0.3  # 低利用率阈值
    ):
        self.spot_discount = spot_discount
        self.reserved_discount = reserved_discount
        self.idle_threshold = idle_threshold
        self.low_utilization_threshold = low_utilization_threshold
        
        self.cost_history: List[CostEntry] = []
        self.usage_history: List[ResourceUsage] = []
        self.recommendations: List[CostRecommendation] = []
        
        # 定价基准 (每小时)
        self.pricing = {
            "small": {"ondemand": 0.05, "reserved": 0.035, "spot": 0.015},
            "medium": {"ondemand": 0.10, "reserved": 0.07, "spot": 0.03},
            "large": {"ondemand": 0.20, "reserved": 0.14, "spot": 0.06},
            "xlarge": {"ondemand": 0.40, "reserved": 0.28, "spot": 0.12}
        }
    
    def analyze(
        self,
        usage_patterns: List[Dict],
        reserved_vs_spot: bool = True
    ) -> Dict:
        """
        分析成本并生成优化建议
        
        Args:
            usage_patterns: 使用模式列表
            reserved_vs_spot: 是否分析预留vs竞价
            
        Returns:
            分析报告
        """
        analysis = {
            "current_cost": self._calculate_current_cost(usage_patterns),
            "optimized_cost": 0,
            "potential_savings": 0,
            "savings_percentage": 0,
            "recommendations": [],
            "instance_mix": {},
            "usage_patterns": {}
        }
        
        # 分析使用模式
        analysis["usage_patterns"] = self._analyze_usage_patterns(usage_patterns)
        
        # 生成优化建议
        self.recommendations = self._generate_recommendations(analysis["usage_patterns"])
        analysis["recommendations"] = [self._recommendation_to_dict(r) for r in self.recommendations]
        
        # 计算优化后的成本
        analysis["optimized_cost"] = self._calculate_optimized_cost(analysis)
        analysis["potential_savings"] = analysis["current_cost"] - analysis["optimized_cost"]
        
        if analysis["current_cost"] > 0:
            analysis["savings_percentage"] = (
                analysis["potential_savings"] / analysis["current_cost"] * 100
            )
        
        # 推荐实例组合
        if reserved_vs_spot:
            analysis["instance_mix"] = self._recommend_instance_mix(analysis["usage_patterns"])
        
        logger.info(f"成本分析完成: 当前${analysis['current_cost']:.2f}/月, 优化后${analysis['optimized_cost']:.2f}/月")
        
        return analysis
    
    def _calculate_current_cost(self, usage_patterns: List[Dict]) -> float:
        """计算当前成本"""
        total_cost = 0
        
        for pattern in usage_patterns:
            hours = pattern.get("hours_per_month", 720)
            instance_size = pattern.get("instance_size", "medium")
            instance_type = pattern.get("instance_type", "ondemand")
            
            price = self.pricing.get(instance_size, {}).get(instance_type, 0.10)
            count = pattern.get("instance_count", 1)
            
            total_cost += price * hours * count
        
        return total_cost
    
    def _analyze_usage_patterns(self, usage_patterns: List[Dict]) -> Dict:
        """分析使用模式"""
        patterns = {
            "stable_load": True,
            "avg_utilization": 0,
            "peak_hours": [],
            "off_peak_hours": [],
            "variable_utilization": False
        }
        
        if not usage_patterns:
            return patterns
        
        utilizations = []
        
        for pattern in usage_patterns:
            util = pattern.get("utilization", 0.5)
            utilizations.append(util)
            
            # 分析峰值时间
            if util > 0.8:
                hour = pattern.get("hour", 12)
                patterns["peak_hours"].append(hour)
            elif util < 0.3:
                patterns["off_peak_hours"].append(pattern.get("hour", 0))
        
        patterns["avg_utilization"] = sum(utilizations) / len(utilizations) if utilizations else 0
        
        # 检测负载波动
        if len(utilizations) > 1:
            variance = sum((u - patterns["avg_utilization"]) ** 2 for u in utilizations) / len(utilizations)
            patterns["variable_utilization"] = variance > 0.1
        
        # 检测稳定性
        patterns["stable_load"] = patterns["avg_utilization"] > 0.5 and not patterns["variable_utilization"]
        
        return patterns
    
    def _generate_recommendations(self, usage_patterns: Dict) -> List[CostRecommendation]:
        """生成优化建议"""
        recommendations = []
        
        avg_util = usage_patterns.get("avg_utilization", 0.5)
        is_stable = usage_patterns.get("stable_load", False)
        is_variable = usage_patterns.get("variable_utilization", False)
        
        # 1. 预留实例推荐
        if is_stable and avg_util > 0.6:
            recommendations.append(CostRecommendation(
                id="rec-001",
                type="reserved_instances",
                title="使用预留实例降低成本",
                description=f"检测到稳定负载(利用率{avg_util*100:.1f}%), 建议使用预留实例",
                potential_savings=self.reserved_discount * 100,
                confidence=0.9,
                implementation="将稳定工作负载迁移到预留实例, 承诺1年或3年使用期",
                priority=8
            ))
        
        # 2. 竞价实例推荐
        if is_variable and avg_util < 0.5:
            recommendations.append(CostRecommendation(
                id="rec-002",
                type="spot_instances",
                title="使用竞价实例处理弹性负载",
                description="检测到可预测的弹性负载, 建议使用竞价实例",
                potential_savings=self.spot_discount * 100,
                confidence=0.85,
                implementation="配置自动伸缩使用竞价实例, 设置中断处理机制",
                priority=7
            ))
        
        # 3. 闲置资源回收
        if avg_util < self.low_utilization_threshold:
            recommendations.append(CostRecommendation(
                id="rec-003",
                type="rightsizing",
                title="调整实例规格",
                description=f"平均利用率较低({avg_util*100:.1f}%), 建议缩小实例规格",
                potential_savings=(1 - avg_util) * 100,
                confidence=0.95,
                implementation="分析实际资源需求, 迁移到更小的实例类型",
                priority=9
            ))
        
        # 4. 自动伸缩优化
        recommendations.append(CostRecommendation(
            id="rec-004",
            type="auto_scaling",
            title="优化自动伸缩策略",
            description="配置积极的自动伸缩以匹配负载变化",
            potential_savings=20,
            confidence=0.75,
            implementation="缩短伸缩冷却时间, 降低扩容阈值",
            priority=5
        ))
        
        # 5. 关闭闲置资源
        if usage_patterns.get("off_peak_hours"):
            recommendations.append(CostRecommendation(
                id="rec-005",
                type="scheduled_scaling",
                title="配置定时伸缩策略",
                description=f"检测到低利用率时段: {usage_patterns['off_peak_hours']}",
                potential_savings=15,
                confidence=0.8,
                implementation="在低峰期自动缩减实例数量",
                priority=6
            ))
        
        # 按优先级排序
        recommendations.sort(key=lambda x: x.priority, reverse=True)
        
        return recommendations
    
    def _calculate_optimized_cost(self, analysis: Dict) -> float:
        """计算优化后的成本"""
        current_cost = analysis["current_cost"]
        instance_mix = analysis.get("instance_mix", {})
        
        # 应用实例组合优化
        ondemand_ratio = instance_mix.get("ondemand", 0.5)
        spot_ratio = instance_mix.get("spot", 0.3)
        reserved_ratio = instance_mix.get("reserved", 0.2)
        
        # 假设优化后平均节省
        weighted_savings = (
            ondemand_ratio * 0 +
            spot_ratio * self.spot_discount +
            reserved_ratio * self.reserved_discount
        )
        
        optimized_cost = current_cost * (1 - weighted_savings * 0.5)  # 保守估计节省50%
        
        return optimized_cost
    
    def _recommend_instance_mix(self, usage_patterns: Dict) -> Dict:
        """推荐实例组合"""
        avg_util = usage_patterns.get("avg_utilization", 0.5)
        is_stable = usage_patterns.get("stable_load", False)
        is_variable = usage_patterns.get("variable_utilization", False)
        
        mix = {
            "ondemand": 0.5,
            "spot": 0.3,
            "reserved": 0.2
        }
        
        if is_stable:
            # 稳定负载: 多用预留
            mix["reserved"] = 0.6
            mix["ondemand"] = 0.3
            mix["spot"] = 0.1
        elif is_variable:
            # 弹性负载: 多用竞价
            mix["spot"] = 0.5
            mix["ondemand"] = 0.3
            mix["reserved"] = 0.2
        elif avg_util < 0.3:
            # 低利用率: 竞价为主
            mix["spot"] = 0.7
            mix["ondemand"] = 0.2
            mix["reserved"] = 0.1
        
        return mix
    
    def _recommendation_to_dict(self, rec: CostRecommendation) -> Dict:
        """推荐转字典"""
        return {
            "id": rec.id,
            "type": rec.type,
            "title": rec.title,
            "description": rec.description,
            "potential_savings": f"{rec.potential_savings:.1f}%",
            "confidence": f"{rec.confidence * 100:.0f}%",
            "implementation": rec.implementation,
            "priority": rec.priority
        }
    
    def track_cost(self, entry: CostEntry):
        """跟踪成本"""
        self.cost_history.append(entry)
        
        # 保持历史记录在合理范围内
        if len(self.cost_history) > 10000:
            self.cost_history = self.cost_history[-5000:]
    
    def get_cost_report(self, period: str = "monthly") -> Dict:
        """获取成本报告"""
        now = datetime.now()
        
        if period == "daily":
            start_date = now - timedelta(days=1)
        elif period == "weekly":
            start_date = now - timedelta(weeks=1)
        else:
            start_date = now - timedelta(days=30)
        
        filtered_costs = [
            c for c in self.cost_history
            if c.timestamp >= start_date
        ]
        
        total_cost = sum(c.cost for c in filtered_costs)
        by_type = {}
        by_resource = {}
        
        for c in filtered_costs:
            # 按类型汇总
            if c.instance_type.value not in by_type:
                by_type[c.instance_type.value] = 0
            by_type[c.instance_type.value] += c.cost
            
            # 按资源汇总
            if c.resource_type not in by_resource:
                by_resource[c.resource_type] = 0
            by_resource[c.resource_type] += c.cost
        
        return {
            "period": period,
            "total_cost": total_cost,
            "cost_by_type": by_type,
            "cost_by_resource": by_resource,
            "trend": "stable",  # 可以基于历史数据计算
            "forecast_next_period": total_cost * 1.05  # 简单预测
        }
    
    def calculate_savings_vs_ondemand(
        self,
        instance_count: int,
        instance_size: str,
        instance_type: InstanceType,
        hours_per_month: float
    ) -> Dict:
        """计算与按需实例相比的节省"""
        ondemand_price = self.pricing.get(instance_size, {}).get("ondemand", 0.10)
        
        if instance_type == InstanceType.ON_DEMAND:
            return {
                "instance_type": "ondemand",
                "monthly_cost": ondemand_price * hours_per_month * instance_count,
                "savings": 0,
                "savings_percentage": 0
            }
        elif instance_type == InstanceType.SPOT:
            spot_price = self.pricing.get(instance_size, {}).get("spot", 0.03)
            monthly_cost = spot_price * hours_per_month * instance_count
            savings = (ondemand_price - spot_price) * hours_per_month * instance_count
            return {
                "instance_type": "spot",
                "monthly_cost": monthly_cost,
                "savings": savings,
                "savings_percentage": self.spot_discount * 100
            }
        elif instance_type == InstanceType.RESERVED:
            reserved_price = self.pricing.get(instance_size, {}).get("reserved", 0.07)
            monthly_cost = reserved_price * hours_per_month * instance_count
            savings = (ondemand_price - reserved_price) * hours_per_month * instance_count
            return {
                "instance_type": "reserved",
                "monthly_cost": monthly_cost,
                "savings": savings,
                "savings_percentage": self.reserved_discount * 100
            }
        
        return {}
    
    def optimize_bid_strategy(
        self,
        base_price: float,
        historical_prices: List[float],
        reliability_requirement: float = 0.95
    ) -> Dict:
        """优化竞价策略"""
        if not historical_prices:
            return {
                "recommended_bid": base_price * 1.1,
                "strategy": "conservative",
                "expected_reliability": 0.9
            }
        
        # 分析历史价格
        avg_price = sum(historical_prices) / len(historical_prices)
        max_price = max(historical_prices)
        min_price = min(historical_prices)
        
        # 基于可靠性要求计算竞价
        # 更高的竞价=更高的可靠性
        percentile = reliability_requirement * 100
        
        # 计算指定百分位的价格
        sorted_prices = sorted(historical_prices)
        percentile_index = int(len(sorted_prices) * reliability_requirement)
        percentile_price = sorted_prices[min(percentile_index, len(sorted_prices) - 1)]
        
        recommended_bid = max(base_price, percentile_price * 1.05)
        
        return {
            "recommended_bid": recommended_bid,
            "strategy": "aggressive" if reliability_requirement > 0.95 else "balanced",
            "expected_reliability": reliability_requirement,
            "price_analysis": {
                "average": avg_price,
                "min": min_price,
                "max": max_price,
                "percentile": percentile_price
            }
        }
