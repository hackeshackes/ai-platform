"""
单元测试 - Decision Engine

测试覆盖率目标: > 80%
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from decision_engine import (
    DecisionEngine, DecisionRequest, DecisionContext, DecisionResult,
    DecisionType, DecisionPriority, ConfidenceLevel, RiskMetrics, Decision
)
from risk_assessor import (
    RiskAssessor, RiskAssessment, RiskLevel, RiskCategory, RiskTrend, RiskFactor
)
from predictor import (
    Predictor, PredictionResult, TrendDirection, SeasonalityType,
    ForecastPoint, TrendAnalysis, SeasonalityAnalysis, Anomaly
)


# ============ Fixtures ============

@pytest.fixture
def sample_business_data():
    """示例业务数据"""
    return {
        'revenue_growth': 0.08,
        'market_opportunity': 0.7,
        'competitive_advantage': 0.6,
        'cash_flow_status': 'stable',
        'debt_ratio': 0.35,
        'profit_margin': 0.12,
        'market_volatility': 0.3,
        'competition_intensity': 0.5,
        'time_series': [100 + i * 2 for i in range(60)]
    }


@pytest.fixture
def sample_decision_context(sample_business_data):
    """示例决策上下文"""
    return DecisionContext(
        business_data=sample_business_data,
        constraints={'budget': 100000, 'timeline': 90},
        objectives=['maximize_profit', 'minimize_risk'],
        time_horizon=30,
        risk_tolerance=0.5
    )


# ============ DecisionEngine Tests ============

class TestDecisionEngine:
    """决策引擎测试类"""
    
    def test_engine_initialization(self):
        """测试引擎初始化"""
        engine = DecisionEngine()
        assert engine.risk_weight == 0.4
        assert engine.reward_weight == 0.6
        assert engine.min_confidence == 0.5
        
        # 自定义配置
        engine2 = DecisionEngine({
            'risk_weight': 0.5,
            'reward_weight': 0.5,
            'min_confidence': 0.6
        })
        assert engine2.risk_weight == 0.5
        assert engine2.reward_weight == 0.5
        assert engine2.min_confidence == 0.6
    
    def test_calculate_financial_risk(self, sample_business_data):
        """测试财务风险计算"""
        engine = DecisionEngine()
        
        # 稳定现金流
        data = {'cash_flow_status': 'stable', 'debt_ratio': 0.3}
        risk = engine._calculate_financial_risk(data)
        assert 0 <= risk <= 0.4
        
        # 负现金流
        data_negative = {'cash_flow_status': 'negative', 'debt_ratio': 0.5}
        risk_negative = engine._calculate_financial_risk(data_negative)
        assert risk_negative > risk
        
        # 高负债
        data_high_debt = {'cash_flow_status': 'stable', 'debt_ratio': 0.8}
        risk_high_debt = engine._calculate_financial_risk(data_high_debt)
        assert risk_high_debt > 0.35
    
    def test_calculate_operational_risk(self):
        """测试运营风险计算"""
        engine = DecisionEngine()
        
        # 高资源可用性
        data_good = {'resource_availability': 0.9, 'process_maturity': 0.8}
        risk_good = engine._calculate_operational_risk(data_good)
        assert risk_good < 0.3
        
        # 低资源可用性
        data_bad = {'resource_availability': 0.4, 'process_maturity': 0.5}
        risk_bad = engine._calculate_operational_risk(data_bad)
        assert risk_bad > 0.4
    
    def test_calculate_market_risk(self):
        """测试市场风险计算"""
        engine = DecisionEngine()
        
        # 低波动、低竞争
        data_low = {'market_volatility': 0.2, 'competition_intensity': 0.3}
        risk_low = engine._calculate_market_risk(data_low)
        assert risk_low < 0.4
        
        # 高波动、高竞争
        data_high = {'market_volatility': 0.8, 'competition_intensity': 0.8}
        risk_high = engine._calculate_market_risk(data_high)
        assert risk_high > 0.5
    
    def test_calculate_compliance_risk(self):
        """测试合规风险计算"""
        engine = DecisionEngine()
        
        # 合规状态
        assert engine._calculate_compliance_risk({'regulatory_status': 'compliant'}) == 0.0
        assert engine._calculate_compliance_risk({'regulatory_status': 'partial_compliance'}) == 0.3
        assert engine._calculate_compliance_risk({'regulatory_status': 'under_review'}) == 0.5
        assert engine._calculate_compliance_risk({'regulatory_status': 'non_compliant'}) == 0.9
    
    def test_identify_risk_factors(self):
        """测试风险因素识别"""
        engine = DecisionEngine()
        
        data = {
            'cash_flow_status': 'negative',
            'market_volatility': 0.8,
            'debt_ratio': 0.7
        }
        
        factors = engine._identify_risk_factors(data)
        assert len(factors) >= 2
        
        # 检查识别的因素
        factor_types = [f.get('type') for f in factors]
        assert 'financial' in factor_types
    
    def test_determine_decision_type(self):
        """测试决策类型确定"""
        engine = DecisionEngine()
        
        # 高收益低风险 -> 扩展
        risk_low = RiskMetrics(0.3, 0.2, 0.3, 0.3, 0.2)
        decision_type = engine._determine_decision_type(0.6, risk_low, 0.7)
        assert decision_type in [DecisionType.EXPAND, DecisionType.BUY]
        
        # 低收益高风险 -> 卖出
        risk_high = RiskMetrics(0.8, 0.8, 0.8, 0.8, 0.8)
        decision_type_sell = engine._determine_decision_type(-0.2, risk_high, 0.1)
        assert decision_type_sell == DecisionType.SELL
    
    def test_determine_priority(self):
        """测试优先级确定"""
        engine = DecisionEngine()
        
        # 高分 -> 高优先级
        priority_high = engine._determine_priority(0.6, RiskMetrics(0.3, 0.3, 0.3, 0.3, 0.3))
        assert priority_high == DecisionPriority.HIGH
        
        # 低分 -> 低优先级
        priority_low = engine._determine_priority(-0.3, RiskMetrics(0.3, 0.3, 0.3, 0.3, 0.3))
        assert priority_low == DecisionPriority.LOW
        
        # 高风险 -> 紧急
        priority_critical = engine._determine_priority(0.2, RiskMetrics(0.9, 0.9, 0.9, 0.9, 0.9))
        assert priority_critical == DecisionPriority.CRITICAL
    
    def test_get_confidence_level(self):
        """测试置信度等级获取"""
        engine = DecisionEngine()
        
        assert engine._get_confidence_level(0.2) == ConfidenceLevel.VERY_LOW
        assert engine._get_confidence_level(0.4) == ConfidenceLevel.LOW
        assert engine._get_confidence_level(0.6) == ConfidenceLevel.MEDIUM
        assert engine._get_confidence_level(0.8) == ConfidenceLevel.HIGH
        assert engine._get_confidence_level(0.95) == ConfidenceLevel.VERY_HIGH
    
    @pytest.mark.asyncio
    async def test_collect_data(self, sample_decision_context):
        """测试数据收集"""
        engine = DecisionEngine()
        data = await engine.collect_data(sample_decision_context)
        
        assert 'business' in data
        assert 'constraints' in data
        assert 'objectives' in data
        assert data['validated'] == True
    
    @pytest.mark.asyncio
    async def test_analyze(self, sample_decision_context):
        """测试完整决策分析"""
        engine = DecisionEngine()
        request = DecisionRequest(context=sample_decision_context)
        
        result = await engine.analyze(request)
        
        assert isinstance(result, DecisionResult)
        assert result.decision is not None
        assert 0 <= result.confidence <= 1
        assert 0 <= result.risk_score <= 1
        assert 0 <= result.expected_reward <= 1
        assert len(result.alternatives) > 0
        assert result.explanation != ""
    
    @pytest.mark.asyncio
    async def test_predict_reward(self, sample_business_data):
        """测试收益预测"""
        engine = DecisionEngine()
        data = {'business': sample_business_data, 'time_horizon': 30}
        
        reward = await engine.predict_reward(data)
        assert 0 <= reward <= 1
    
    @pytest.mark.asyncio
    async def test_assess_risk(self, sample_business_data):
        """测试风险评估"""
        engine = DecisionEngine()
        data = {'business': sample_business_data}
        
        risk = await engine.assess_risk(data)
        
        assert isinstance(risk, RiskMetrics)
        assert 0 <= risk.overall_score <= 1
        assert 0 <= risk.financial_risk <= 1
        assert 0 <= risk.operational_risk <= 1
        assert 0 <= risk.market_risk <= 1
        assert 0 <= risk.compliance_risk <= 1
    
    def test_calculate_confidence(self, sample_business_data):
        """测试置信度计算"""
        engine = DecisionEngine()
        data = {'business': sample_business_data}
        decision = Decision(
            type=DecisionType.BUY,
            priority=DecisionPriority.HIGH,
            reasoning="测试推理"
        )
        risk = RiskMetrics(0.4, 0.3, 0.4, 0.4, 0.3)
        
        confidence = engine.calculate_confidence(data, decision, risk)
        assert 0 <= confidence <= 1
    
    def test_generate_alternatives(self, sample_business_data):
        """测试备选方案生成"""
        engine = DecisionEngine()
        data = {'business': sample_business_data}
        risk = RiskMetrics(0.4, 0.3, 0.4, 0.4, 0.3)
        
        alternatives = engine.generate_alternatives(data, risk, 0.5)
        
        assert len(alternatives) >= 3
        assert all(hasattr(alt, 'id') for alt in alternatives)
        assert all(hasattr(alt, 'type') for alt in alternatives)


# ============ RiskAssessor Tests ============

class TestRiskAssessor:
    """风险评估器测试类"""
    
    def test_assessor_initialization(self):
        """测试评估器初始化"""
        assessor = RiskAssessor()
        assert assessor.thresholds['low'] == 0.3
        assert assessor.thresholds['moderate'] == 0.5
        assert assessor.thresholds['high'] == 0.7
    
    def test_classify_risk(self):
        """测试风险分类"""
        assessor = RiskAssessor()
        
        # 阈值边界测试 (thresholds: low=0.3, moderate=0.5, high=0.7, critical=0.85)
        assert assessor.classify_risk(0.1) == RiskLevel.MINIMAL
        assert assessor.classify_risk(0.25) == RiskLevel.MINIMAL
        assert assessor.classify_risk(0.35) == RiskLevel.LOW
        assert assessor.classify_risk(0.49) == RiskLevel.LOW
        assert assessor.classify_risk(0.55) == RiskLevel.MODERATE
        assert assessor.classify_risk(0.69) == RiskLevel.MODERATE
        assert assessor.classify_risk(0.75) == RiskLevel.HIGH
        assert assessor.classify_risk(0.84) == RiskLevel.HIGH
        assert assessor.classify_risk(0.9) == RiskLevel.CRITICAL
    
    def test_calculate_risk_score(self):
        """测试风险分数计算"""
        assessor = RiskAssessor()
        
        factors = [
            RiskFactor(
                id="TEST001",
                category=RiskCategory.FINANCIAL,
                name="测试风险1",
                description="测试",
                probability=0.6,
                impact=0.7,
                score=0.42,
                severity="medium"
            ),
            RiskFactor(
                id="TEST002",
                category=RiskCategory.FINANCIAL,
                name="测试风险2",
                description="测试",
                probability=0.8,
                impact=0.5,
                score=0.4,
                severity="medium"
            )
        ]
        
        overall, category_scores = assessor.calculate_risk_score(factors)
        
        assert 0 <= overall <= 1
        assert RiskCategory.FINANCIAL in category_scores
    
    def test_identify_financial_risks(self):
        """测试财务风险识别"""
        assessor = RiskAssessor()
        
        # 负现金流
        data = {'cash_flow_status': 'negative'}
        risks = assessor._identify_financial_risks(data)
        assert len(risks) >= 1
        
        # 高负债
        data_high_debt = {'debt_ratio': 0.7}
        risks_high_debt = assessor._identify_financial_risks(data_high_debt)
        assert any(r.severity == 'high' for r in risks_high_debt)
    
    def test_identify_operational_risks(self):
        """测试运营风险识别"""
        assessor = RiskAssessor()
        
        # 供应链中断
        data = {'supply_chain_status': 'disrupted'}
        risks = assessor._identify_operational_risks(data)
        assert len(risks) >= 1
        
        # 高流失率
        data_turnover = {'employee_turnover': 0.25}
        risks_turnover = assessor._identify_operational_risks(data_turnover)
        assert any('流失' in r.name or 'turnover' in r.name.lower() for r in risks_turnover)
    
    def test_identify_market_risks(self):
        """测试市场风险识别"""
        assessor = RiskAssessor()
        
        # 高波动
        data_volatile = {'market_volatility': 0.8}
        risks = assessor._identify_market_risks(data_volatile)
        assert len(risks) >= 1
        
        # 激烈竞争
        data_competitive = {'competition_intensity': 0.8}
        risks_competitive = assessor._identify_market_risks(data_competitive)
        assert any(r.severity == 'high' for r in risks_competitive)
    
    def test_identify_compliance_risks(self):
        """测试合规风险识别"""
        assessor = RiskAssessor()
        
        # 不合规
        data_non_compliant = {'regulatory_status': 'non_compliant'}
        risks = assessor._identify_compliance_risks(data_non_compliant)
        assert len(risks) >= 1
        assert any(r.severity == 'critical' for r in risks)
    
    def test_analyze_trend(self):
        """测试趋势分析"""
        assessor = RiskAssessor()
        
        # 稳定趋势
        data_stable = {'risk_trend_indicators': {}}
        trend = assessor.analyze_trend(data_stable, {})
        assert trend == RiskTrend.STABLE
        
        # 上升趋势
        data_increasing = {'risk_trend_indicators': {'indicator1': 'increasing'}}
        trend_increasing = assessor.analyze_trend(data_increasing, {
            RiskCategory.FINANCIAL: 0.7
        })
        assert trend_increasing == RiskTrend.INCREASING
    
    def test_check_risk_tolerance(self):
        """测试风险容忍度检查"""
        assessor = RiskAssessor()
        
        # 良好状态
        status_good = assessor.check_risk_tolerance(0.3, {'risk_tolerance': 0.5})
        assert "良好" in status_good or "good" in status_good.lower()
        
        # 危险状态
        status_danger = assessor.check_risk_tolerance(0.8, {'risk_tolerance': 0.5})
        assert "危险" in status_danger or "danger" in status_danger.lower()
    
    def test_calculate_next_review_date(self):
        """测试下次审查日期计算"""
        assessor = RiskAssessor()
        
        from datetime import timedelta
        
        # 高风险 -> 7天内
        date_critical = assessor.calculate_next_review_date(RiskLevel.CRITICAL)
        assert date_critical <= datetime.now() + timedelta(days=7)
        
        # 低风险 -> 90天内
        date_minimal = assessor.calculate_next_review_date(RiskLevel.MINIMAL)
        assert date_minimal >= datetime.now() + timedelta(days=60)
    
    @pytest.mark.asyncio
    async def test_assess(self, sample_business_data):
        """测试完整风险评估"""
        assessor = RiskAssessor()
        result = await assessor.assess(sample_business_data)
        
        assert isinstance(result, RiskAssessment)
        assert 0 <= result.overall_score <= 1
        assert result.level in RiskLevel.__members__.values()
        assert len(result.recommendations) >= 0


# ============ Predictor Tests ============

class TestPredictor:
    """预测器测试类"""
    
    def test_predictor_initialization(self):
        """测试预测器初始化"""
        predictor = Predictor()
        assert predictor.default_horizon == 30
        assert predictor.confidence_level == 0.95
        
        predictor2 = Predictor({'default_horizon': 60, 'confidence_level': 0.9})
        assert predictor2.default_horizon == 60
    
    def test_determine_trend_direction(self):
        """测试趋势方向判断"""
        predictor = Predictor()
        
        # 稳定趋势
        stable_series = [100, 101, 100, 101, 100, 101, 100]
        direction = predictor._determine_trend_direction(stable_series)
        assert direction in [TrendDirection.STABLE, TrendDirection.VOLATILE]
        
        # 上升趋势
        up_series = [100, 105, 110, 115, 120, 125]
        direction_up = predictor._determine_trend_direction(up_series)
        assert direction_up == TrendDirection.UP
        
        # 下降趋势
        down_series = [100, 95, 90, 85, 80, 75]
        direction_down = predictor._determine_trend_direction(down_series)
        assert direction_down == TrendDirection.DOWN
    
    def test_calculate_slope(self):
        """测试斜率计算"""
        predictor = Predictor()
        
        # 水平线
        flat_series = [100, 100, 100, 100]
        slope = predictor._calculate_slope(flat_series)
        assert abs(slope) < 0.01
        
        # 上升线
        up_series = [100, 110, 120, 130]
        slope_up = predictor._calculate_slope(up_series)
        assert slope_up > 0
        
        # 下降线
        down_series = [130, 120, 110, 100]
        slope_down = predictor._calculate_slope(down_series)
        assert slope_down < 0
    
    def test_calculate_trend_strength(self):
        """测试趋势强度计算"""
        predictor = Predictor()
        
        # 弱趋势
        weak_series = [100 + i * 0.1 for i in range(50)]
        strength_weak = predictor._calculate_trend_strength(weak_series, 0.1)
        assert strength_weak < 0.5
        
        # 强趋势
        strong_series = [100 + i * 2 for i in range(50)]
        strength_strong = predictor._calculate_trend_strength(strong_series, 2)
        assert strength_strong > 0.5
    
    def test_check_weekly_seasonality(self):
        """测试周季节性检查"""
        predictor = Predictor()
        
        # 短序列
        short_series = [100, 101, 102]
        assert predictor._check_weekly_seasonality(short_series) == False
        
        # 模拟工作日/周末差异
        workday_series = [110] * 5 + [90] * 2
        long_series = workday_series * 15
        has_weekly = predictor._check_weekly_seasonality(long_series)
        assert has_weekly == True
    
    def test_calculate_seasonality_amplitude(self):
        """测试季节性振幅计算"""
        predictor = Predictor()
        
        # 无季节性
        no_season_series = [100] * 100
        amp = predictor._calculate_seasonality_amplitude(no_season_series, 7)
        assert amp < 0.1
    
    def test_calculate_confidence(self):
        """测试置信度计算"""
        predictor = Predictor()
        
        trend = TrendAnalysis(
            direction=TrendDirection.UP,
            slope=0.5,
            strength=0.8,
            start_date=datetime.now(),
            end_date=datetime.now(),
            data_points=50,
            r_squared=0.9
        )
        
        # 无异常
        confidence_no_anomaly = predictor.calculate_confidence(trend, [])
        assert confidence_no_anomaly > 0.7
        
        # 多个异常
        anomalies = [Mock(deviation=0.5)] * 10
        confidence_with_anomalies = predictor.calculate_confidence(trend, anomalies)
        assert confidence_with_anomalies < confidence_no_anomaly
    
    def test_get_confidence_interval(self):
        """测试置信区间获取"""
        predictor = Predictor()
        
        trend = TrendAnalysis(
            direction=TrendDirection.STABLE,
            slope=0,
            strength=0.5,
            start_date=datetime.now(),
            end_date=datetime.now(),
            data_points=50,
            r_squared=0.7
        )
        
        interval = predictor.get_confidence_interval(trend, 0.8)
        assert len(interval) == 2
        assert interval[0] < 0
        assert interval[1] > 0
    
    def test_standard_deviation(self):
        """测试标准差计算"""
        predictor = Predictor()
        
        # 常数序列
        std_const = predictor._standard_deviation([100, 100, 100])
        assert std_const == 0
        
        # 变化序列
        std_varying = predictor._standard_deviation([100, 110, 90, 105, 95])
        assert std_varying > 0
    
    def test_volatility(self):
        """测试波动率计算"""
        predictor = Predictor()
        
        # 稳定序列
        vol_stable = predictor._volatility([100, 101, 100, 101, 100])
        assert vol_stable < 0.1
        
        # 波动序列
        vol_volatile = predictor._volatility([100, 150, 80, 140, 90])
        assert vol_volatile > 0.2
    
    def test_generate_forecast(self):
        """测试预测生成"""
        predictor = Predictor()
        
        trend = TrendAnalysis(
            direction=TrendDirection.UP,
            slope=0.1,
            strength=0.5,
            start_date=datetime.now(),
            end_date=datetime.now(),
            data_points=50,
            r_squared=0.7
        )
        
        seasonality = SeasonalityAnalysis(
            has_seasonality=False,
            type=None,
            amplitude=0,
            phase=0,
            period=0,
            strength=0
        )
        
        forecast = predictor.generate_forecast(trend, seasonality, 10, 0.8)
        
        assert len(forecast) == 10
        assert all(isinstance(p, ForecastPoint) for p in forecast)
        assert all(p.value > 0 for p in forecast)
    
    def test_detect_anomalies(self):
        """测试异常检测"""
        predictor = Predictor()
        
        # 正常序列
        normal_series = list(range(100, 110))
        anomalies_normal = predictor.detect_anomalies({'time_series': normal_series})
        assert len(anomalies_normal) == 0
        
        # 包含异常的序列
        anomalous_series = [100] * 20 + [200] + [100] * 20
        anomalies = predictor.detect_anomalies({'time_series': anomalous_series})
        assert len(anomalies) >= 1
        assert any(a.severity in ['high', 'critical'] for a in anomalies)
    
    @pytest.mark.asyncio
    async def test_predict(self, sample_business_data):
        """测试完整预测"""
        predictor = Predictor()
        result = await predictor.predict(sample_business_data, 30)
        
        assert isinstance(result, PredictionResult)
        assert len(result.forecast) == 30
        assert result.trend is not None
        assert result.seasonality is not None
        assert result.model_accuracy > 0
        assert 0 <= result.model_accuracy <= 1


# ============ Integration Tests ============

class TestIntegration:
    """集成测试类"""
    
    @pytest.mark.asyncio
    async def test_full_decision_workflow(self, sample_decision_context):
        """测试完整决策工作流"""
        engine = DecisionEngine()
        request = DecisionRequest(context=sample_decision_context)
        
        # 执行决策
        result = await engine.analyze(request)
        
        # 验证结果完整性
        assert result.decision.type in DecisionType
        assert result.decision.priority in DecisionPriority
        assert result.confidence_level in ConfidenceLevel
        
        # 验证收益和风险
        assert 0 <= result.risk_score <= 1
        assert 0 <= result.expected_reward <= 1
        
        # 验证备选方案
        assert len(result.alternatives) > 0
        assert all(alt.type in DecisionType for alt in result.alternatives)
    
    @pytest.mark.asyncio
    async def test_risk_prediction_integration(self, sample_business_data):
        """测试风险预测集成"""
        risk_assessor = RiskAssessor()
        predictor = Predictor()
        
        # 风险评估
        risk_result = await risk_assessor.assess(sample_business_data)
        
        # 预测分析
        predict_result = await predictor.predict(sample_business_data, 30)
        
        # 验证一致性
        assert risk_result.overall_score >= 0
        assert len(predict_result.forecast) == 30
        
        # 预测趋势与风险等级应一致
        if risk_result.level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            assert predict_result.trend.direction in [TrendDirection.DOWN, TrendDirection.VOLATILE]


# ============ Edge Cases ============

class TestEdgeCases:
    """边界情况测试类"""
    
    def test_empty_data(self):
        """测试空数据处理"""
        engine = DecisionEngine()
        
        # 空业务数据
        empty_data = {}
        risk = asyncio.run(
            engine.assess_risk({'business': empty_data})
        )
        assert risk.overall_score >= 0
    
    def test_extreme_values(self):
        """测试极端值处理"""
        engine = DecisionEngine()
        
        # 极端风险数据
        extreme_data = {
            'cash_flow_status': 'negative',
            'debt_ratio': 1.0,
            'market_volatility': 1.0,
            'competition_intensity': 1.0
        }
        
        factors = engine._identify_risk_factors(extreme_data)
        assert len(factors) >= 1
    
    def test_missing_fields(self):
        """测试缺失字段处理"""
        engine = DecisionEngine()
        
        # 缺失部分字段
        partial_data = {'revenue_growth': 0.1}
        risk = asyncio.run(
            engine.assess_risk({'business': partial_data})
        )
        assert risk.overall_score >= 0
    
    def test_long_horizon_prediction(self):
        """测试长时域预测"""
        predictor = Predictor()
        
        data = {'time_series': list(range(100, 200))}
        
        # 90天预测
        result = asyncio.run(
            predictor.predict(data, 90)
        )
        
        assert len(result.forecast) == 90
        assert result.forecast[-1].timestamp > datetime.now() + timedelta(days=60)


# ============ Run Tests ============

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=decision_engine", "--cov=risk_assessor", "--cov=predictor"])
