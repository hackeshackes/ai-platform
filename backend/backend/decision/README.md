# V9 Phase 3 - 自主决策引擎 (Decision Engine)

AI自动业务决策引擎，支持完整的决策流程：数据→分析→决策→执行。

## 目录结构

```
backend/decision/
├── __init__.py          # 包初始化
├── decision_engine.py    # 核心决策引擎
├── risk_assessor.py     # 风险评估系统
├── predictor.py         # 预测分析引擎
├── api/
│   └── endpoints/
│       └── decision.py  # API端点
└── tests.py             # 单元测试
```

## 核心模块

### 1. DecisionEngine (decision_engine.py)

AI决策引擎，完整决策流程：

```python
from decision_engine import DecisionEngine, DecisionContext

engine = DecisionEngine()
context = DecisionContext(
    business_data={'revenue_growth': 0.08, 'market_opportunity': 0.7},
    constraints={'budget': 100000},
    objectives=['maximize_profit'],
    time_horizon=30,
    risk_tolerance=0.5
)

result = await engine.analyze(DecisionRequest(context=context))
```

**主要功能**：
- `analyze()` - 完整决策分析
- `assess_risk()` - 风险评估
- `predict_reward()` - 收益预测
- `generate_decision()` - 生成决策
- `calculate_confidence()` - 计算置信度
- `generate_alternatives()` - 生成备选方案

### 2. RiskAssessor (risk_assessor.py)

AI风险评估系统：

```python
from risk_assessor import RiskAssessor

assessor = RiskAssessor()
result = await assessor.assess(business_data)
```

**主要功能**：
- `identify_risk_factors()` - 识别风险因素
- `calculate_risk_score()` - 计算风险分数
- `classify_risk()` - 风险分类
- `analyze_trend()` - 趋势分析
- `generate_recommendations()` - 生成建议

**风险类别**：
- FINANCIAL - 财务风险
- OPERATIONAL - 运营风险
- MARKET - 市场风险
- COMPLIANCE - 合规风险
- STRATEGIC - 战略风险
- REPUTATIONAL - 声誉风险

### 3. Predictor (predictor.py)

预测分析引擎：

```python
from predictor import Predictor

predictor = Predictor()
result = await predictor.predict(business_data, horizon=30)
```

**主要功能**：
- `time_series_predict()` - 时间序列预测
- `analyze_seasonality()` - 季节性分析
- `detect_anomalies()` - 异常检测
- `calculate_confidence()` - 计算置信度
- `generate_forecast()` - 生成预测

## API端点

### 决策分析

```http
POST /decision/analyze
Content-Type: application/json

{
    "context": {
        "business_data": {
            "revenue_growth": 0.08,
            "market_opportunity": 0.7
        },
        "constraints": {"budget": 100000},
        "time_horizon": 30,
        "risk_tolerance": 0.5
    },
    "enable_auto_execute": false
}
```

### 风险评估

```http
POST /decision/risk/assess
Content-Type: application/json

{
    "revenue_growth": 0.08,
    "cash_flow_status": "stable",
    "debt_ratio": 0.35
}
```

### 预测分析

```http
POST /decision/predict
Content-Type: application/json

{
    "revenue_growth": 0.08,
    "time_series": [100, 102, 105, ...]
}
```

### 决策历史

```http
GET /decision/history?decision_type=buy&priority=high
```

## 枚举类型

### DecisionType
- BUY - 买入
- SELL - 卖出
- HOLD - 持有
- EXPAND - 扩展
- REDUCE - 缩减
- LAUNCH - 启动
- CANCEL - 取消

### DecisionPriority
- LOW - 低
- MEDIUM - 中
- HIGH - 高
- CRITICAL - 紧急

### RiskLevel
- MINIMAL - 极低
- LOW - 低
- MODERATE - 中等
- HIGH - 高
- CRITICAL - 严重

### ConfidenceLevel
- VERY_LOW - < 0.3
- LOW - 0.3 - 0.5
- MEDIUM - 0.5 - 0.7
- HIGH - 0.7 - 0.9
- VERY_HIGH - > 0.9

## 运行测试

```bash
cd /Users/yubao/.openclaw/workspace/backend/decision
python3 -m pytest tests.py -v --cov=.
```

**测试结果**：
- 测试数量: 45
- 通过率: 100%
- 覆盖率: > 80%

## 快速开始

```python
import asyncio
from decision_engine import DecisionEngine, DecisionRequest, DecisionContext
from risk_assessor import RiskAssessor
from predictor import Predictor

async def main():
    # 初始化引擎
    engine = DecisionEngine()
    risk_assessor = RiskAssessor()
    predictor = Predictor()
    
    # 准备业务数据
    business_data = {
        'revenue_growth': 0.08,
        'market_opportunity': 0.7,
        'competitive_advantage': 0.6,
        'cash_flow_status': 'stable',
        'debt_ratio': 0.35,
        'profit_margin': 0.12,
        'market_volatility': 0.3,
        'competition_intensity': 0.5,
        'time_series': list(range(100, 160))
    }
    
    # 1. 风险评估
    risk_result = await risk_assessor.assess(business_data)
    print(f"风险分数: {risk_result.overall_score}")
    print(f"风险等级: {risk_result.level.value}")
    
    # 2. 预测分析
    predict_result = await predictor.predict(business_data, 30)
    print(f"趋势方向: {predict_result.trend.direction.value}")
    print(f"预测准确度: {predict_result.model_accuracy}")
    
    # 3. 决策分析
    context = DecisionContext(
        business_data=business_data,
        time_horizon=30,
        risk_tolerance=0.5
    )
    decision_result = await engine.analyze(DecisionRequest(context=context))
    print(f"决策: {decision_result.decision.type.value}")
    print(f"置信度: {decision_result.confidence}")

asyncio.run(main())
```

## 版本

- Version: 1.0.0
- Author: OpenClaw V9
- Date: 2026-02-10
