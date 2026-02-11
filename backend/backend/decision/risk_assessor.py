"""
Risk Assessor - AI风险评估系统
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
from datetime import datetime
import json


class RiskLevel(Enum):
    """风险等级"""
    MINIMAL = "minimal"      # 0 - 0.2
    LOW = "low"               # 0.2 - 0.4
    MODERATE = "moderate"     # 0.4 - 0.6
    HIGH = "high"             # 0.6 - 0.8
    CRITICAL = "critical"     # 0.8 - 1.0


class RiskCategory(Enum):
    """风险类别"""
    FINANCIAL = "financial"
    OPERATIONAL = "operational"
    MARKET = "market"
    COMPLIANCE = "compliance"
    STRATEGIC = "strategic"
    REPUTATIONAL = "reputational"


class RiskTrend(Enum):
    """风险趋势"""
    DECREASING = "decreasing"
    STABLE = "stable"
    INCREASING = "increasing"


@dataclass
class RiskFactor:
    """风险因素"""
    id: str
    category: RiskCategory
    name: str
    description: str
    probability: float  # 发生概率 0-1
    impact: float        # 影响程度 0-1
    score: float         # 风险分数 = probability * impact
    severity: str        # low, medium, high, critical
    mitigations: List[str] = field(default_factory=list)
    indicators: List[str] = field(default_factory=list)
    detected_at: datetime = field(default_factory=datetime.now)


@dataclass
class RiskAssessment:
    """风险评估结果"""
    overall_score: float
    level: RiskLevel
    category_scores: Dict[RiskCategory, float]
    factors: List[RiskFactor]
    trend: RiskTrend
    recommendations: List[str] = field(default_factory=list)
    risk_tolerance_status: str = ""
    next_review_date: Optional[datetime] = None
    assessed_at: datetime = field(default_factory=datetime.now)


class RiskAssessor:
    """
    AI风险评估系统
    
    功能：
    1. 识别风险因素
    2. 计算风险分数
    3. 风险分类
    4. 生成风险建议
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.thresholds = {
            'low': 0.3,
            'moderate': 0.5,
            'high': 0.7,
            'critical': 0.85
        }
        self.risk_weights = self.config.get('risk_weights', {
            RiskCategory.FINANCIAL: 0.30,
            RiskCategory.OPERATIONAL: 0.25,
            RiskCategory.MARKET: 0.25,
            RiskCategory.COMPLIANCE: 0.15,
            RiskCategory.STRATEGIC: 0.03,
            RiskCategory.REPUTATIONAL: 0.02
        })
        
    async def assess(self, data: Dict[str, Any]) -> RiskAssessment:
        """
        评估业务风险
        
        Args:
            data: 业务数据
            
        Returns:
            RiskAssessment: 风险评估结果
        """
        # 1. 识别风险因素
        factors = self.identify_risk_factors(data)
        
        # 2. 计算风险分数
        overall_score, category_scores = self.calculate_risk_score(factors)
        
        # 3. 风险分类
        level = self.classify_risk(overall_score)
        
        # 4. 风险趋势分析
        trend = self.analyze_trend(data, category_scores)
        
        # 5. 风险容忍度评估
        risk_tolerance_status = self.check_risk_tolerance(overall_score, data)
        
        # 6. 生成风险建议
        recommendations = self.generate_recommendations(factors, level)
        
        # 7. 确定下次审查日期
        next_review_date = self.calculate_next_review_date(level)
        
        return RiskAssessment(
            overall_score=overall_score,
            level=level,
            category_scores=category_scores,
            factors=factors,
            trend=trend,
            recommendations=recommendations,
            risk_tolerance_status=risk_tolerance_status,
            next_review_date=next_review_date
        )
    
    def identify_risk_factors(self, data: Dict[str, Any]) -> List[RiskFactor]:
        """
        识别风险因素
        
        Args:
            data: 业务数据
            
        Returns:
            List[RiskFactor]: 风险因素列表
        """
        factors = []
        
        # 财务风险识别
        factors.extend(self._identify_financial_risks(data))
        
        # 运营风险识别
        factors.extend(self._identify_operational_risks(data))
        
        # 市场风险识别
        factors.extend(self._identify_market_risks(data))
        
        # 合规风险识别
        factors.extend(self._identify_compliance_risks(data))
        
        # 战略风险识别
        factors.extend(self._identify_strategic_risks(data))
        
        # 声誉风险识别
        factors.extend(self._identify_reputational_risks(data))
        
        return factors
    
    def calculate_risk_score(
        self, 
        factors: List[RiskFactor]
    ) -> tuple[float, Dict[RiskCategory, float]]:
        """
        计算风险分数
        
        Args:
            factors: 风险因素列表
            
        Returns:
            tuple: (综合风险分数, 各类别风险分数)
        """
        # 按类别分组
        category_factors = {}
        for factor in factors:
            if factor.category not in category_factors:
                category_factors[factor.category] = []
            category_factors[factor.category].append(factor)
        
        # 计算各类别风险分数
        category_scores = {}
        for category, cat_factors in category_factors.items():
            if cat_factors:
                # 使用最大风险分数而非平均，更保守
                max_score = max(f.score for f in cat_factors)
                # 加权平均
                avg_score = sum(f.score for f in cat_factors) / len(cat_factors)
                # 综合分数
                category_scores[category] = max_score * 0.6 + avg_score * 0.4
            else:
                category_scores[category] = 0.0
        
        # 计算综合风险分数
        overall_score = 0.0
        for category, weight in self.risk_weights.items():
            score = category_scores.get(category, 0.0)
            overall_score += score * weight
        
        return round(min(overall_score, 1.0), 3), category_scores
    
    def classify_risk(self, risk_score: float) -> RiskLevel:
        """
        风险分类
        
        Args:
            risk_score: 风险分数
            
        Returns:
            RiskLevel: 风险等级
        """
        if risk_score < self.thresholds['low']:
            return RiskLevel.MINIMAL
        elif risk_score < self.thresholds['moderate']:
            return RiskLevel.LOW
        elif risk_score < self.thresholds['high']:
            return RiskLevel.MODERATE
        elif risk_score < self.thresholds['critical']:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL
    
    def analyze_trend(
        self,
        data: Dict[str, Any],
        category_scores: Dict[RiskCategory, float]
    ) -> RiskTrend:
        """
        分析风险趋势
        
        Args:
            data: 业务数据
            category_scores: 各类别风险分数
            
        Returns:
            RiskTrend: 风险趋势
        """
        # 基于历史数据和当前指标判断趋势
        trend_indicators = data.get('risk_trend_indicators', {})
        
        increasing_factors = 0
        decreasing_factors = 0
        
        # 检查各维度的趋势
        for indicator, trend in trend_indicators.items():
            if trend == 'increasing':
                increasing_factors += 1
            elif trend == 'decreasing':
                decreasing_factors += 1
        
        # 财务趋势
        if category_scores.get(RiskCategory.FINANCIAL, 0) > 0.6:
            increasing_factors += 1
        
        # 市场趋势
        if data.get('market_volatility', 0) > 0.7:
            increasing_factors += 1
        
        # 运营趋势
        if data.get('operational_efficiency', 1) < 0.7:
            increasing_factors += 1
        
        if increasing_factors > decreasing_factors + 1:
            return RiskTrend.INCREASING
        elif decreasing_factors > increasing_factors + 1:
            return RiskTrend.DECREASING
        else:
            return RiskTrend.STABLE
    
    def check_risk_tolerance(
        self,
        overall_score: float,
        data: Dict[str, Any]
    ) -> str:
        """
        检查风险容忍度
        
        Args:
            overall_score: 综合风险分数
            data: 业务数据
            
        Returns:
            str: 风险容忍度状态
        """
        risk_tolerance = data.get('risk_tolerance', 0.5)
        
        if overall_score <= risk_tolerance * 0.7:
            return "良好 - 风险在可接受范围内"
        elif overall_score <= risk_tolerance:
            return "可接受 - 接近风险上限"
        elif overall_score <= risk_tolerance * 1.2:
            return "警告 - 超出风险容忍度"
        else:
            return "危险 - 严重超出风险容忍度"
    
    def generate_recommendations(
        self,
        factors: List[RiskFactor],
        level: RiskLevel
    ) -> List[str]:
        """
        生成风险建议
        
        Args:
            factors: 风险因素列表
            level: 风险等级
            
        Returns:
            List[str]: 建议列表
        """
        recommendations = []
        
        # 按风险分数排序
        sorted_factors = sorted(factors, key=lambda x: x.score, reverse=True)
        
        # 对高风险因素生成建议
        high_risk_factors = [f for f in sorted_factors if f.severity in ['high', 'critical']]
        
        for factor in high_risk_factors[:5]:  # 最多5个高风险因素
            recs = self._generate_mitigation_recommendations(factor)
            recommendations.extend(recs)
        
        # 通用建议
        if level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            recommendations.insert(0, "建议立即召开风险评估会议")
            recommendations.insert(1, "准备应急响应计划")
        
        if level == RiskLevel.CRITICAL:
            recommendations.insert(0, "建议暂停高风险业务活动")
            recommendations.insert(1, "立即向管理层报告")
        
        # 去重
        recommendations = list(dict.fromkeys(recommendations))
        
        return recommendations
    
    def calculate_next_review_date(self, level: RiskLevel) -> Optional[datetime]:
        """
        计算下次审查日期
        
        Args:
            level: 风险等级
            
        Returns:
            datetime: 下次审查日期
        """
        from datetime import timedelta
        
        review_days = {
            RiskLevel.MINIMAL: 90,
            RiskLevel.LOW: 60,
            RiskLevel.MODERATE: 30,
            RiskLevel.HIGH: 14,
            RiskLevel.CRITICAL: 7
        }
        
        days = review_days.get(level, 30)
        return datetime.now() + timedelta(days=days)
    
    # ============ 私有辅助方法 ============
    
    def _identify_financial_risks(self, data: Dict[str, Any]) -> List[RiskFactor]:
        """识别财务风险"""
        risks = []
        
        # 现金流风险
        cash_flow_status = data.get('cash_flow_status', 'stable')
        if cash_flow_status == 'negative':
            risks.append(RiskFactor(
                id="FIN001",
                category=RiskCategory.FINANCIAL,
                name="现金流紧张",
                description="企业现金流为负，可能影响日常运营",
                probability=0.8,
                impact=0.9,
                score=0.72,
                severity="high",
                mitigations=["寻求短期融资", "优化应收账款", "削减非必要支出"],
                indicators=["现金比率", "流动比率", "速动比率"]
            ))
        elif cash_flow_status == 'unstable':
            risks.append(RiskFactor(
                id="FIN002",
                category=RiskCategory.FINANCIAL,
                name="现金流不稳定",
                description="企业现金流波动较大",
                probability=0.6,
                impact=0.6,
                score=0.36,
                severity="medium",
                mitigations=["建立现金储备", "多元化收入来源"],
                indicators=["经营现金流变动率"]
            ))
        
        # 债务风险
        debt_ratio = data.get('debt_ratio', 0.3)
        if debt_ratio > 0.6:
            risks.append(RiskFactor(
                id="FIN003",
                category=RiskCategory.FINANCIAL,
                name="高负债率",
                description=f"企业负债率为{debt_ratio:.1%}，高于安全水平",
                probability=0.7,
                impact=0.8,
                score=0.56,
                severity="high",
                mitigations=["债务重组", "股权融资", "资产出售"],
                indicators=["资产负债率", "利息保障倍数"]
            ))
        
        # 盈利能力风险
        profit_margin = data.get('profit_margin', 0.1)
        if profit_margin < 0.05:
            risks.append(RiskFactor(
                id="FIN004",
                category=RiskCategory.FINANCIAL,
                name="盈利能力不足",
                description=f"净利润率为{profit_margin:.1%}，盈利能力弱",
                probability=0.6,
                impact=0.7,
                score=0.42,
                severity="medium",
                mitigations=["成本优化", "价格调整", "产品组合优化"],
                indicators=["毛利率", "净利率", "ROE"]
            ))
        
        return risks
    
    def _identify_operational_risks(self, data: Dict[str, Any]) -> List[RiskFactor]:
        """识别运营风险"""
        risks = []
        
        # 供应链风险
        supply_chain_status = data.get('supply_chain_status', 'stable')
        if supply_chain_status == 'disrupted':
            risks.append(RiskFactor(
                id="OPS001",
                category=RiskCategory.OPERATIONAL,
                name="供应链中断",
                description="主要供应商或物流渠道出现问题",
                probability=0.5,
                impact=0.9,
                score=0.45,
                severity="high",
                mitigations=["多元化供应商", "建立安全库存", "本地化采购"],
                indicators=["供应商交付准时率", "库存周转率"]
            ))
        
        # 技术风险
        tech_infrastructure = data.get('tech_infrastructure', 'stable')
        if tech_infrastructure == 'outdated':
            risks.append(RiskFactor(
                id="OPS002",
                category=RiskCategory.OPERATIONAL,
                name="技术基础设施过时",
                description="IT系统和技术设施需要更新升级",
                probability=0.7,
                impact=0.6,
                score=0.42,
                severity="medium",
                mitigations=["技术升级计划", "系统现代化改造"],
                indicators=["系统可用率", "IT维护成本占比"]
            ))
        
        # 人力资源风险
        employee_turnover = data.get('employee_turnover', 0.1)
        if employee_turnover > 0.2:
            risks.append(RiskFactor(
                id="OPS003",
                category=RiskCategory.OPERATIONAL,
                name="高员工流失率",
                description=f"员工流失率为{employee_turnover:.1%}，高于正常水平",
                probability=0.7,
                impact=0.5,
                score=0.35,
                severity="medium",
                mitigations=["薪酬激励优化", "职业发展规划", "企业文化改善"],
                indicators=["员工满意度", "关键岗位空缺率"]
            ))
        
        return risks
    
    def _identify_market_risks(self, data: Dict[str, Any]) -> List[RiskFactor]:
        """识别市场风险"""
        risks = []
        
        # 市场波动
        market_volatility = data.get('market_volatility', 0.3)
        if market_volatility > 0.7:
            risks.append(RiskFactor(
                id="MKT001",
                category=RiskCategory.MARKET,
                name="高市场波动性",
                description="市场环境变化剧烈，不确定性高",
                probability=0.8,
                impact=0.7,
                score=0.56,
                severity="high",
                mitigations=["对冲策略", "灵活定价", "业务多元化"],
                indicators=["市场指数波动率", "需求变化率"]
            ))
        
        # 竞争压力
        competition_intensity = data.get('competition_intensity', 0.5)
        if competition_intensity > 0.7:
            risks.append(RiskFactor(
                id="MKT002",
                category=RiskCategory.MARKET,
                name="激烈竞争",
                description="市场竞争激烈，可能导致市场份额下降",
                probability=0.8,
                impact=0.6,
                score=0.48,
                severity="high",
                mitigations=["差异化竞争", "品牌建设", "客户忠诚度计划"],
                indicators=["市场份额", "客户获取成本"]
            ))
        
        # 需求变化
        demand_stability = data.get('demand_stability', 0.8)
        if demand_stability < 0.5:
            risks.append(RiskFactor(
                id="MKT003",
                category=RiskCategory.MARKET,
                name="需求不稳定",
                description="市场需求波动大，难以预测",
                probability=0.7,
                impact=0.7,
                score=0.49,
                severity="high",
                mitigations=["需求预测优化", "产品线多元化", "弹性生产能力"],
                indicators=["需求预测准确率", "订单变化率"]
            ))
        
        return risks
    
    def _identify_compliance_risks(self, data: Dict[str, Any]) -> List[RiskFactor]:
        """识别合规风险"""
        risks = []
        
        # 监管状态
        regulatory_status = data.get('regulatory_status', 'compliant')
        if regulatory_status == 'non_compliant':
            risks.append(RiskFactor(
                id="CMP001",
                category=RiskCategory.COMPLIANCE,
                name="不合规状态",
                description="企业存在违反法规的情况",
                probability=0.9,
                impact=1.0,
                score=0.9,
                severity="critical",
                mitigations=["立即整改", "法律咨询", "合规审查"],
                indicators=["合规审计结果", "监管处罚记录"]
            ))
        elif regulatory_status == 'under_review':
            risks.append(RiskFactor(
                id="CMP002",
                category=RiskCategory.COMPLIANCE,
                name="监管审查中",
                description="企业正在接受监管机构审查",
                probability=0.6,
                impact=0.8,
                score=0.48,
                severity="high",
                mitigations=["配合审查", "主动披露", "整改准备"],
                indicators=["审查进度", "历史合规记录"]
            ))
        
        # 数据安全
        data_security = data.get('data_security', 'strong')
        if data_security == 'weak':
            risks.append(RiskFactor(
                id="CMP003",
                category=RiskCategory.COMPLIANCE,
                name="数据安全薄弱",
                description="数据安全措施不足，存在泄露风险",
                probability=0.5,
                impact=0.9,
                score=0.45,
                severity="high",
                mitigations=["安全升级", "员工培训", "保险覆盖"],
                indicators=["安全事件数量", "漏洞修复率"]
            ))
        
        return risks
    
    def _identify_strategic_risks(self, data: Dict[str, Any]) -> List[RiskFactor]:
        """识别战略风险"""
        risks = []
        
        # 战略执行
        strategy_alignment = data.get('strategy_alignment', 0.8)
        if strategy_alignment < 0.5:
            risks.append(RiskFactor(
                id="STR001",
                category=RiskCategory.STRATEGIC,
                name="战略执行偏差",
                description="业务方向与战略目标存在偏差",
                probability=0.5,
                impact=0.6,
                score=0.3,
                severity="medium",
                mitigations=["战略调整", "目标重设", "执行监控"],
                indicators=["战略KPI达成率"]
            ))
        
        # 技术变革
        tech_disruption = data.get('tech_disruption_risk', 0.3)
        if tech_disruption > 0.6:
            risks.append(RiskFactor(
                id="STR002",
                category=RiskCategory.STRATEGIC,
                name="技术颠覆风险",
                description="新兴技术可能颠覆现有业务模式",
                probability=0.4,
                impact=0.8,
                score=0.32,
                severity="medium",
                mitigations=["技术研发", "并购整合", "业务转型"],
                indicators=["技术投资比例", "创新产出"]
            ))
        
        return risks
    
    def _identify_reputational_risks(self, data: Dict[str, Any]) -> List[RiskFactor]:
        """识别声誉风险"""
        risks = []
        
        # 品牌声誉
        brand_reputation = data.get('brand_reputation', 0.8)
        if brand_reputation < 0.5:
            risks.append(RiskFactor(
                id="REP001",
                category=RiskCategory.REPUTATIONAL,
                name="品牌声誉受损",
                description="品牌形象和声誉面临风险",
                probability=0.4,
                impact=0.8,
                score=0.32,
                severity="medium",
                mitigations=["品牌管理", "危机公关", "客户关系修复"],
                indicators=["品牌指数", "社交媒体情绪"]
            ))
        
        # 客户关系
        customer_satisfaction = data.get('customer_satisfaction', 0.8)
        if customer_satisfaction < 0.6:
            risks.append(RiskFactor(
                id="REP002",
                category=RiskCategory.REPUTATIONAL,
                name="客户满意度低",
                description="客户满意度下降，可能影响口碑",
                probability=0.6,
                impact=0.5,
                score=0.3,
                severity="medium",
                mitigations=["服务改进", "客户反馈系统", "投诉处理优化"],
                indicators=["NPS评分", "客户投诉率"]
            ))
        
        return risks
    
    def _generate_mitigation_recommendations(self, factor: RiskFactor) -> List[str]:
        """为单个风险因素生成缓解建议"""
        recommendations = []
        
        for mitigation in factor.mitigations[:3]:  # 最多3个建议
            recommendations.append(f"[{factor.name}] {mitigation}")
        
        return recommendations
