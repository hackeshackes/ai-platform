"""
Decision Engine - AI自动业务决策引擎核心
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum
from datetime import datetime
import json


class DecisionType(Enum):
    """决策类型"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    EXPAND = "expand"
    REDUCE = "reduce"
    LAUNCH = "launch"
    CANCEL = "cancel"


class DecisionPriority(Enum):
    """决策优先级"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class ConfidenceLevel(Enum):
    """置信度等级"""
    VERY_LOW = "very_low"      # < 0.3
    LOW = "low"                 # 0.3 - 0.5
    MEDIUM = "medium"           # 0.5 - 0.7
    HIGH = "high"               # 0.7 - 0.9
    VERY_HIGH = "very_high"     # > 0.9


@dataclass
class DecisionContext:
    """决策上下文"""
    business_data: Dict[str, Any]
    constraints: Dict[str, Any] = field(default_factory=dict)
    objectives: List[str] = field(default_factory=list)
    time_horizon: int = 30
    risk_tolerance: float = 0.5


@dataclass
class DecisionRequest:
    """决策请求"""
    context: DecisionContext
    options: List[str] = field(default_factory=list)
    enable_auto_execute: bool = False


@dataclass
class Decision:
    """决策结果"""
    type: DecisionType
    priority: DecisionPriority
    reasoning: str
    details: Dict[str, Any] = field(default_factory=dict)
    conditions: Dict[str, Any] = field(default_factory=dict)
    valid_from: datetime = field(default_factory=datetime.now)
    valid_until: Optional[datetime] = None


@dataclass
class AlternativeDecision:
    """备选决策"""
    id: str
    type: DecisionType
    description: str
    expected_outcome: str
    risk_factors: List[str]
    confidence: float


@dataclass
class DecisionResult:
    """完整决策结果"""
    decision: Decision
    confidence: float
    confidence_level: ConfidenceLevel
    risk_score: float
    expected_reward: float
    alternatives: List[AlternativeDecision]
    explanation: str = ""
    recommendations: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class RiskMetrics:
    """风险指标"""
    overall_score: float
    financial_risk: float
    operational_risk: float
    market_risk: float
    compliance_risk: float
    factors: List[Dict[str, Any]] = field(default_factory=list)


class DecisionEngine:
    """
    AI决策引擎
    
    完整决策流程：
    1. 收集数据
    2. 风险评估
    3. 收益预测
    4. 生成决策
    5. 置信度评估
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.risk_weight = self.config.get('risk_weight', 0.4)
        self.reward_weight = self.config.get('reward_weight', 0.6)
        self.min_confidence = self.config.get('min_confidence', 0.5)
        
    async def analyze(self, request: DecisionRequest) -> DecisionResult:
        """
        分析并生成决策
        
        Args:
            request: 决策请求
            
        Returns:
            DecisionResult: 决策结果
        """
        context = request.context
        
        # 1. 收集数据
        data = await self.collect_data(context)
        
        # 2. 风险评估
        risk = await self.assess_risk(data)
        
        # 3. 收益预测
        reward = await self.predict_reward(data)
        
        # 4. 生成决策
        decision = self.generate_decision(data, risk, reward)
        
        # 5. 置信度评估
        confidence = self.calculate_confidence(data, decision, risk)
        
        # 6. 生成备选方案
        alternatives = self.generate_alternatives(data, risk, reward)
        
        # 7. 生成决策解释
        explanation = self.generate_explanation(decision, risk, reward, confidence)
        
        # 8. 生成建议
        recommendations = self.generate_recommendations(decision, risk, data)
        
        return DecisionResult(
            decision=decision,
            confidence=confidence,
            confidence_level=self._get_confidence_level(confidence),
            risk_score=risk.overall_score,
            expected_reward=reward,
            alternatives=alternatives,
            explanation=explanation,
            recommendations=recommendations
        )
    
    async def collect_data(self, context: DecisionContext) -> Dict[str, Any]:
        """
        收集业务数据
        
        Args:
            context: 决策上下文
            
        Returns:
            收集的数据
        """
        # 合并业务数据和约束条件
        data = {
            'business': context.business_data,
            'constraints': context.constraints,
            'objectives': context.objectives,
            'time_horizon': context.time_horizon,
            'risk_tolerance': context.risk_tolerance,
            'collected_at': datetime.now().isoformat()
        }
        
        # 数据验证和清洗
        data['validated'] = self._validate_data(data)
        
        return data
    
    async def assess_risk(self, data: Dict[str, Any]) -> RiskMetrics:
        """
        评估业务风险
        
        Args:
            data: 业务数据
            
        Returns:
            RiskMetrics: 风险指标
        """
        business_data = data.get('business', {})
        
        # 计算各类风险
        financial_risk = self._calculate_financial_risk(business_data)
        operational_risk = self._calculate_operational_risk(business_data)
        market_risk = self._calculate_market_risk(business_data)
        compliance_risk = self._calculate_compliance_risk(business_data)
        
        # 计算综合风险分数 (加权平均)
        overall_score = (
            financial_risk * 0.3 +
            operational_risk * 0.25 +
            market_risk * 0.25 +
            compliance_risk * 0.2
        )
        
        # 识别风险因素
        factors = self._identify_risk_factors(business_data)
        
        return RiskMetrics(
            overall_score=overall_score,
            financial_risk=financial_risk,
            operational_risk=operational_risk,
            market_risk=market_risk,
            compliance_risk=compliance_risk,
            factors=factors
        )
    
    async def predict_reward(self, data: Dict[str, Any]) -> float:
        """
        预测预期收益
        
        Args:
            data: 业务数据
            
        Returns:
            float: 预期收益分数
        """
        business_data = data.get('business', {})
        time_horizon = data.get('time_horizon', 30)
        
        # 基于业务指标计算预期收益
        base_reward = 0.0
        
        # 收入增长潜力
        revenue_growth = business_data.get('revenue_growth', 0)
        base_reward += min(revenue_growth * 0.1, 0.3)
        
        # 市场机会
        market_opportunity = business_data.get('market_opportunity', 0)
        base_reward += min(market_opportunity * 0.2, 0.3)
        
        # 竞争优势
        competitive_advantage = business_data.get('competitive_advantage', 0)
        base_reward += min(competitive_advantage * 0.15, 0.2)
        
        # 时间衰减因子
        time_factor = min(time_horizon / 90, 1.0)
        base_reward *= time_factor
        
        # 约束影响
        constraints = data.get('constraints', {})
        constraint_penalty = len(constraints) * 0.05
        base_reward = max(0, base_reward - constraint_penalty)
        
        return round(base_reward, 3)
    
    def generate_decision(
        self, 
        data: Dict[str, Any], 
        risk: RiskMetrics,
        reward: float
    ) -> Decision:
        """
        生成决策
        
        Args:
            data: 业务数据
            risk: 风险指标
            reward: 预期收益
            
        Returns:
            Decision: 决策结果
        """
        # 计算决策分数
        decision_score = (reward * self.reward_weight) - (risk.overall_score * self.risk_weight)
        
        # 根据分数和约束确定决策类型
        decision_type = self._determine_decision_type(decision_score, risk, reward)
        priority = self._determine_priority(decision_score, risk)
        reasoning = self._build_reasoning(decision_type, decision_score, risk, reward)
        
        # 生成决策详情
        details = self._generate_decision_details(decision_type, data, risk, reward)
        
        # 生成决策条件
        conditions = self._generate_conditions(decision_type, risk)
        
        return Decision(
            type=decision_type,
            priority=priority,
            reasoning=reasoning,
            details=details,
            conditions=conditions
        )
    
    def calculate_confidence(
        self, 
        data: Dict[str, Any], 
        decision: Decision,
        risk: RiskMetrics
    ) -> float:
        """
        计算决策置信度
        
        Args:
            data: 业务数据
            decision: 决策结果
            risk: 风险指标
            
        Returns:
            float: 置信度分数 (0-1)
        """
        business_data = data.get('business', {})
        
        # 数据质量因子
        data_quality = self._assess_data_quality(business_data)
        
        # 历史准确性因子
        historical_accuracy = 0.85  # 默认85%
        
        # 风险一致性因子 (风险越高，置信度越低)
        risk_consistency = max(0, 1 - risk.overall_score * 0.5)
        
        # 市场稳定性因子
        market_stability = business_data.get('market_stability', 0.7)
        
        # 综合置信度
        confidence = (
            data_quality * 0.3 +
            historical_accuracy * 0.25 +
            risk_consistency * 0.25 +
            market_stability * 0.2
        )
        
        return round(min(max(confidence, 0), 1), 3)
    
    def generate_alternatives(
        self, 
        data: Dict[str, Any], 
        risk: RiskMetrics,
        reward: float
    ) -> List[AlternativeDecision]:
        """
        生成备选决策方案
        
        Args:
            data: 业务数据
            risk: 风险指标
            reward: 预期收益
            
        Returns:
            List[AlternativeDecision]: 备选方案列表
        """
        alternatives = []
        
        # 保守方案
        conservative = AlternativeDecision(
            id="alt_conservative_001",
            type=DecisionType.HOLD,
            description="保守方案 - 维持现状，等待更多信息",
            expected_outcome="风险最低，但可能错过增长机会",
            risk_factors=["机会成本", "市场变化"],
            confidence=0.8
        )
        alternatives.append(conservative)
        
        # 适度方案
        moderate = AlternativeDecision(
            id="alt_moderate_002",
            type=DecisionType.EXPAND if reward > 0.4 else DecisionType.HOLD,
            description="适度方案 - 有控制地扩展",
            expected_outcome="平衡风险与收益",
            risk_factors=["执行风险", "资源分配"],
            confidence=0.65
        )
        alternatives.append(moderate)
        
        # 积极方案
        aggressive = AlternativeDecision(
            id="alt_aggressive_003",
            type=DecisionType.LAUNCH if reward > 0.6 else DecisionType.EXPAND,
            description="积极方案 - 主动出击，抓住机会",
            expected_outcome="潜在收益最大化",
            risk_factors=["高财务风险", "执行失败风险"],
            confidence=0.5
        )
        alternatives.append(aggressive)
        
        return alternatives
    
    def generate_explanation(
        self,
        decision: Decision,
        risk: RiskMetrics,
        reward: float,
        confidence: float
    ) -> str:
        """生成决策解释"""
        explanation_parts = [
            f"基于当前业务数据分析，推荐决策为【{decision.type.value}】。",
            f"预期收益评估为【{reward:.2%}】，风险指数为【{risk.overall_score:.2%}】。",
            f"决策置信度为【{confidence:.2%}】。"
        ]
        
        if decision.type == DecisionType.BUY or decision.type == DecisionType.EXPAND:
            explanation_parts.append("该决策基于积极的收益预期和可控的风险水平。")
        elif decision.type == DecisionType.SELL or decision.type == DecisionType.REDUCE:
            explanation_parts.append("该决策考虑了市场风险和潜在的损失。")
        else:
            explanation_parts.append("建议观察市场变化，等待更明确的信号。")
        
        return " ".join(explanation_parts)
    
    def generate_recommendations(
        self,
        decision: Decision,
        risk: RiskMetrics,
        data: Dict[str, Any]
    ) -> List[str]:
        """生成决策建议"""
        recommendations = []
        
        if risk.financial_risk > 0.7:
            recommendations.append("建议加强财务监控，制定应急资金计划")
        if risk.operational_risk > 0.6:
            recommendations.append("建议优化运营流程，降低执行风险")
        if risk.market_risk > 0.6:
            recommendations.append("建议关注市场变化，保持策略灵活性")
        if risk.compliance_risk > 0.5:
            recommendations.append("建议进行合规性审查，确保符合法规要求")
        
        recommendations.append(f"建议在{decision.valid_from.strftime('%Y-%m-%d')}后重新评估")
        
        return recommendations
    
    # ============ 私有辅助方法 ============
    
    def _validate_data(self, data: Dict[str, Any]) -> bool:
        """验证数据完整性"""
        required_fields = ['business']
        return all(field in data for field in required_fields)
    
    def _calculate_financial_risk(self, data: Dict[str, Any]) -> float:
        """计算财务风险"""
        cash_flow = data.get('cash_flow_status', 'stable')
        debt_ratio = data.get('debt_ratio', 0.3)
        
        risk = 0.0
        if cash_flow == 'negative':
            risk += 0.4
        elif cash_flow == 'unstable':
            risk += 0.2
        
        risk += min(debt_ratio * 0.5, 0.4)
        
        return min(risk, 1.0)
    
    def _calculate_operational_risk(self, data: Dict[str, Any]) -> float:
        """计算运营风险"""
        resource_availability = data.get('resource_availability', 0.8)
        process_maturity = data.get('process_maturity', 0.7)
        
        risk = (1 - resource_availability) * 0.5 + (1 - process_maturity) * 0.5
        
        return min(risk, 1.0)
    
    def _calculate_market_risk(self, data: Dict[str, Any]) -> float:
        """计算市场风险"""
        market_volatility = data.get('market_volatility', 0.3)
        competition_intensity = data.get('competition_intensity', 0.5)
        
        risk = market_volatility * 0.6 + competition_intensity * 0.4
        
        return min(risk, 1.0)
    
    def _calculate_compliance_risk(self, data: Dict[str, Any]) -> float:
        """计算合规风险"""
        regulatory_status = data.get('regulatory_status', 'compliant')
        
        risk = 0.0
        if regulatory_status == 'non_compliant':
            risk = 0.9
        elif regulatory_status == 'under_review':
            risk = 0.5
        elif regulatory_status == 'partial_compliance':
            risk = 0.3
        
        return min(risk, 1.0)
    
    def _identify_risk_factors(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """识别风险因素"""
        factors = []
        
        if data.get('cash_flow_status') == 'negative':
            factors.append({
                'type': 'financial',
                'description': '现金流为负',
                'severity': 'high'
            })
        
        if data.get('market_volatility', 0) > 0.7:
            factors.append({
                'type': 'market',
                'description': '市场波动性高',
                'severity': 'medium'
            })
        
        if data.get('debt_ratio', 0) > 0.6:
            factors.append({
                'type': 'financial',
                'description': '负债率过高',
                'severity': 'high'
            })
        
        return factors
    
    def _determine_decision_type(
        self,
        decision_score: float,
        risk: RiskMetrics,
        reward: float
    ) -> DecisionType:
        """确定决策类型"""
        risk_tolerance = 0.5
        
        if reward > 0.6 and risk.overall_score < 0.4:
            return DecisionType.EXPAND
        elif reward > 0.4 and risk.overall_score < 0.5:
            return DecisionType.BUY
        elif reward > 0.3 and risk.overall_score < 0.6:
            return DecisionType.HOLD
        elif reward < 0.2 or risk.overall_score > 0.7:
            return DecisionType.SELL
        else:
            return DecisionType.HOLD
    
    def _determine_priority(
        self,
        decision_score: float,
        risk: RiskMetrics
    ) -> DecisionPriority:
        """确定决策优先级"""
        if risk.overall_score > 0.8:
            return DecisionPriority.CRITICAL
        elif decision_score > 0.5:
            return DecisionPriority.HIGH
        elif decision_score > 0:
            return DecisionPriority.MEDIUM
        else:
            return DecisionPriority.LOW
    
    def _build_reasoning(
        self,
        decision_type: DecisionType,
        decision_score: float,
        risk: RiskMetrics,
        reward: float
    ) -> str:
        """构建决策推理"""
        return f"基于收益评估({reward:.2%})和风险评估({risk.overall_score:.2%})，推荐执行【{decision_type.value}】操作"
    
    def _generate_decision_details(
        self,
        decision_type: DecisionType,
        data: Dict[str, Any],
        risk: RiskMetrics,
        reward: float
    ) -> Dict[str, Any]:
        """生成决策详情"""
        return {
            'recommended_action': decision_type.value,
            'expected_impact': reward,
            'risk_level': risk.overall_score,
            'execution_window': '30天',
            'key_metrics_to_monitor': ['收入增长率', '市场占有率', '客户满意度']
        }
    
    def _generate_conditions(
        self,
        decision_type: DecisionType,
        risk: RiskMetrics
    ) -> Dict[str, Any]:
        """生成决策执行条件"""
        return {
            'risk_threshold': risk.overall_score,
            'time_limit': '2024-12-31',
            'required_approvals': 1 if risk.overall_score < 0.5 else 2
        }
    
    def _assess_data_quality(self, data: Dict[str, Any]) -> float:
        """评估数据质量"""
        if not data:
            return 0.3
        
        required_fields = ['revenue_growth', 'market_opportunity', 'cash_flow_status']
        present_fields = sum(1 for field in required_fields if field in data)
        
        return min(0.5 + present_fields * 0.15, 1.0)
    
    def _get_confidence_level(self, confidence: float) -> ConfidenceLevel:
        """获取置信度等级"""
        if confidence < 0.3:
            return ConfidenceLevel.VERY_LOW
        elif confidence < 0.5:
            return ConfidenceLevel.LOW
        elif confidence < 0.7:
            return ConfidenceLevel.MEDIUM
        elif confidence < 0.9:
            return ConfidenceLevel.HIGH
        else:
            return ConfidenceLevel.VERY_HIGH
