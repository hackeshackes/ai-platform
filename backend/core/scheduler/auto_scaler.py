"""
智能调度系统 - 自动伸缩器

水平扩展/垂直扩展/预测扩展/成本优化
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import logging
import json

logger = logging.getLogger(__name__)


class ScalingType(Enum):
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"
    PREDICTIVE = "predictive"
    SCHEDULED = "scheduled"


class ScalingAction(Enum):
    SCALE_OUT = "scale_out"
    SCALE_IN = "scale_in"
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    NO_ACTION = "no_action"


@dataclass
class ScalingPolicy:
    """伸缩策略"""
    name: str
    metric_name: str
    threshold_high: float
    threshold_low: float
    cooldown_seconds: int = 300
    scaling_type: ScalingType = ScalingType.HORIZONTAL
    step_size: int = 1
    min_instances: int = 1
    max_instances: int = 100
    
    
@dataclass
class Instance:
    """实例"""
    instance_id: str
    instance_type: str
    status: str
    created_at: datetime = field(default_factory=datetime.now)
    metrics: Dict = field(default_factory=dict)
    cost_per_hour: float = 0.0


@dataclass
class ScalingDecision:
    """伸缩决策"""
    action: ScalingAction
    scaling_type: ScalingType
    reason: str
    details: Dict
    timestamp: datetime = field(default_factory=datetime.now)
    confidence: float = 1.0


class AutoScaler:
    """自动伸缩器"""
    
    def __init__(
        self,
        default_cooldown: int = 300,
        enable_predictive: bool = True,
        cost_optimization: bool = True
    ):
        self.default_cooldown = default_cooldown
        self.enable_predictive = enable_predictive
        self.cost_optimization = cost_optimization
        
        self.policies: Dict[str, ScalingPolicy] = {}
        self.instances: Dict[str, Instance] = {}
        self.scaling_history: List[ScalingDecision] = []
        
        self.last_scaling_time = datetime.now()
        self.predictive_model = None
        
    def add_policy(self, policy: ScalingPolicy) -> bool:
        """添加伸缩策略"""
        self.policies[policy.name] = policy
        logger.info(f"伸缩策略已添加: {policy.name}")
        return True
    
    def decide(
        self,
        current_metrics: Dict[str, float],
        target_response_time: Optional[float] = None,
        predicted_load: Optional[Dict] = None
    ) -> ScalingDecision:
        """
        做出伸缩决策
        
        Args:
            current_metrics: 当前指标，如 {"cpu": 85, "qps": 1000, "memory": 70}
            target_response_time: 目标响应时间
            predicted_load: 预测负载
            
        Returns:
            ScalingDecision 伸缩决策
        """
        # 检查冷却期
        if datetime.now() - self.last_scaling_time < timedelta(seconds=self.default_cooldown):
            logger.debug("冷却期内，不执行伸缩")
            return ScalingDecision(
                action=ScalingAction.NO_ACTION,
                scaling_type=ScalingType.HORIZONTAL,
                reason="Cooldown period",
                details={"cooldown_remaining": str(timedelta(seconds=self.default_cooldown) - (datetime.now() - self.last_scaling_time))}
            )
        
        # 预测性伸缩
        if self.enable_predictive and predicted_load:
            predictive_decision = self._predictive_scaling(predicted_load)
            if predictive_decision:
                self._execute_scaling(predictive_decision)
                return predictive_decision
        
        # 基于指标的伸缩
        for policy_name, policy in self.policies.items():
            if policy.metric_name in current_metrics:
                value = current_metrics[policy.metric_name]
                decision = self._evaluate_policy(value, policy)
                
                if decision.action != ScalingAction.NO_ACTION:
                    self._execute_scaling(decision)
                    return decision
        
        # 响应时间驱动伸缩
        if target_response_time and "response_time" in current_metrics:
            current_rt = current_metrics["response_time"]
            if current_rt > target_response_time:
                return ScalingDecision(
                    action=ScalingAction.SCALE_OUT,
                    scaling_type=ScalingType.HORIZONTAL,
                    reason="Response time exceeds target",
                    details={
                        "current_response_time": current_rt,
                        "target_response_time": target_response_time,
                        "current_instances": len(self.instances)
                    }
                )
        
        return ScalingDecision(
            action=ScalingAction.NO_ACTION,
            scaling_type=ScalingType.HORIZONTAL,
            reason="No scaling needed",
            details={"current_metrics": current_metrics}
        )
    
    def _evaluate_policy(
        self,
        metric_value: float,
        policy: ScalingPolicy
    ) -> ScalingDecision:
        """评估策略"""
        if metric_value >= policy.threshold_high:
            # 需要扩容
            if policy.scaling_type == ScalingType.HORIZONTAL:
                return ScalingDecision(
                    action=ScalingAction.SCALE_OUT,
                    scaling_type=ScalingType.HORIZONTAL,
                    reason=f"Metric {policy.metric_name}={metric_value} exceeds threshold={policy.threshold_high}",
                    details={
                        "policy": policy.name,
                        "current_value": metric_value,
                        "threshold": policy.threshold_high,
                        "step_size": policy.step_size,
                        "current_instances": len(self.instances)
                    }
                )
            else:
                return ScalingDecision(
                    action=ScalingAction.SCALE_UP,
                    scaling_type=ScalingType.VERTICAL,
                    reason=f"Metric {policy.metric_name}={metric_value} exceeds threshold={policy.threshold_high}",
                    details={
                        "policy": policy.name,
                        "current_value": metric_value,
                        "threshold": policy.threshold_high
                    }
                )
        
        elif metric_value <= policy.threshold_low:
            # 需要缩容
            current_count = len(self.instances)
            if current_count > policy.min_instances:
                step_size = min(policy.step_size, current_count - policy.min_instances)
                
                if policy.scaling_type == ScalingType.HORIZONTAL:
                    return ScalingDecision(
                        action=ScalingAction.SCALE_IN,
                        scaling_type=ScalingType.HORIZONTAL,
                        reason=f"Metric {policy.metric_name}={metric_value} below threshold={policy.threshold_low}",
                        details={
                            "policy": policy.name,
                            "current_value": metric_value,
                            "threshold": policy.threshold_low,
                            "step_size": step_size
                        }
                    )
                else:
                    return ScalingDecision(
                        action=ScalingAction.SCALE_DOWN,
                        scaling_type=ScalingType.VERTICAL,
                        reason=f"Metric {policy.metric_name}={metric_value} below threshold={policy.threshold_low}",
                        details={
                            "policy": policy.name,
                            "current_value": metric_value,
                            "threshold": policy.threshold_low
                        }
                    )
        
        return ScalingDecision(
            action=ScalingAction.NO_ACTION,
            scaling_type=policy.scaling_type,
            reason=f"Metric {policy.metric_name}={metric_value} within range",
            details={"low": policy.threshold_low, "high": policy.threshold_high}
        )
    
    def _predictive_scaling(self, predicted_load: Dict) -> Optional[ScalingDecision]:
        """预测性伸缩"""
        predicted_qps = predicted_load.get("predicted_qps", 0)
        current_qps = predicted_load.get("current_qps", 0)
        
        if predicted_qps > current_qps * 1.2:  # 预测负载增长20%以上
            growth_rate = predicted_qps / current_qps if current_qps > 0 else 1.5
            predicted_instances = int(len(self.instances) * growth_rate)
            
            return ScalingDecision(
                action=ScalingAction.SCALE_OUT,
                scaling_type=ScalingType.PREDICTIVE,
                reason="Predictive scaling based on forecast",
                details={
                    "predicted_qps": predicted_qps,
                    "current_qps": current_qps,
                    "growth_rate": growth_rate,
                    "predicted_instances": predicted_instances
                },
                confidence=0.85
            )
        
        return None
    
    def _execute_scaling(self, decision: ScalingDecision) -> bool:
        """执行伸缩"""
        self.last_scaling_time = datetime.now()
        self.scaling_history.append(decision)
        
        logger.info(f"执行伸缩决策: {decision.action.value} - {decision.reason}")
        
        # 根据决策类型执行实际操作
        if decision.action == ScalingAction.SCALE_OUT:
            self._add_instances(decision.details.get("step_size", 1))
        elif decision.action == ScalingAction.SCALE_IN:
            self._remove_instances(decision.details.get("step_size", 1))
        elif decision.action == ScalingAction.SCALE_UP:
            self._upgrade_instances()
        elif decision.action == ScalingAction.SCALE_DOWN:
            self._downgrade_instances()
        
        return True
    
    def _add_instances(self, count: int) -> List[str]:
        """添加实例"""
        new_instances = []
        for i in range(count):
            instance_id = f"instance-{datetime.now().strftime('%Y%m%d%H%M%S')}-{i}"
            instance = Instance(
                instance_id=instance_id,
                instance_type="general",
                status="starting"
            )
            self.instances[instance_id] = instance
            new_instances.append(instance_id)
            logger.info(f"新实例已添加: {instance_id}")
        
        return new_instances
    
    def _remove_instances(self, count: int) -> List[str]:
        """移除实例"""
        # 选择要移除的实例(优先移除新创建的)
        removed = []
        sorted_instances = sorted(
            self.instances.items(),
            key=lambda x: x[1].created_at,
            reverse=True
        )
        
        for i in range(min(count, len(sorted_instances))):
            instance_id = sorted_instances[i][0]
            del self.instances[instance_id]
            removed.append(instance_id)
            logger.info(f"实例已移除: {instance_id}")
        
        return removed
    
    def _upgrade_instances(self):
        """升级实例规格"""
        for instance_id in self.instances:
            instance = self.instances[instance_id]
            logger.info(f"实例已升级: {instance_id} -> higher_spec")
    
    def _downgrade_instances(self):
        """降级实例规格"""
        for instance_id in self.instances:
            instance = self.instances[instance_id]
            logger.info(f"实例已降级: {instance_id} -> lower_spec")
    
    def register_instance(self, instance: Instance) -> bool:
        """注册实例"""
        self.instances[instance.instance_id] = instance
        logger.info(f"实例已注册: {instance.instance_id}")
        return True
    
    def get_scaling_recommendations(self) -> Dict:
        """获取伸缩推荐"""
        avg_cpu = 0
        avg_memory = 0
        
        if self.instances:
            cpu_values = [i.metrics.get("cpu", 0) for i in self.instances.values()]
            mem_values = [i.metrics.get("memory", 0) for i in self.instances.values()]
            avg_cpu = sum(cpu_values) / len(cpu_values)
            avg_memory = sum(mem_values) / len(mem_values)
        
        recommendations = {
            "current_instances": len(self.instances),
            "avg_cpu_utilization": avg_cpu,
            "avg_memory_utilization": avg_memory,
            "suggested_action": "maintain",
            "reasoning": []
        }
        
        if avg_cpu > 80:
            recommendations["suggested_action"] = "scale_out"
            recommendations["reasoning"].append("CPU utilization above 80%")
        elif avg_cpu < 30:
            recommendations["suggested_action"] = "scale_in"
            recommendations["reasoning"].append("CPU utilization below 30%")
        
        if avg_memory > 85:
            recommendations["suggested_action"] = "scale_out"
            recommendations["reasoning"].append("Memory utilization above 85%")
        elif avg_memory < 40:
            recommendations["reasoning"].append("Memory utilization below 40%")
        
        return recommendations
    
    def cost_optimized_scaling(
        self,
        current_metrics: Dict[str, float],
        spot_instances_available: bool = True,
        reserved_discount: float = 0.3
    ) -> Dict:
        """成本优化伸缩"""
        avg_utilization = 0
        if self.instances:
            util_values = [
                (i.metrics.get("cpu", 0) + i.metrics.get("memory", 0)) / 2
                for i in self.instances.values()
            ]
            avg_utilization = sum(util_values) / len(util_values)
        
        recommendations = {
            "instance_mix": {
                "ondemand": 0.7 if avg_utilization < 60 else 0.4,
                "spot": 0.3 if spot_instances_available and avg_utilization < 70 else 0.0,
                "reserved": 0.0
            },
            "cost_saving_opportunities": []
        }
        
        # 闲置资源回收
        if avg_utilization < 30:
            recommendations["cost_saving_opportunities"].append({
                "type": "rightsizing",
                "description": "实例规格过大，考虑缩小实例类型",
                "potential_savings": f"{(1 - avg_utilization/100) * 100:.1f}%"
            })
        
        # 竞价实例推荐
        if spot_instances_available and avg_utilization < 50:
            recommendations["instance_mix"]["spot"] = 0.5
            recommendations["cost_saving_opportunities"].append({
                "type": "spot_instances",
                "description": "使用竞价实例可节省最多70%成本",
                "potential_savings": "70%"
            })
        
        # 预留容量推荐
        if len(self.instances) > 5 and avg_utilization > 60:
            recommendations["instance_mix"]["reserved"] = reserved_discount
            recommendations["cost_saving_opportunities"].append({
                "type": "reserved_instances",
                "description": "稳定负载建议使用预留实例",
                "potential_savings": f"{reserved_discount * 100}%"
            })
        
        return recommendations
    
    def get_scaling_history(self, limit: int = 100) -> List[Dict]:
        """获取伸缩历史"""
        return [
            {
                "action": d.action.value,
                "type": d.scaling_type.value,
                "reason": d.reason,
                "details": d.details,
                "timestamp": d.timestamp.isoformat()
            }
            for d in self.scaling_history[-limit:]
        ]
