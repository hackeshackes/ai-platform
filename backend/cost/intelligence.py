"""
Cost Intelligence模块 v2.4
"""
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from uuid import uuid4

class CostType(str, Enum):
    """成本类型"""
    API_CALL = "api_call"
    TOKEN = "token"
    COMPUTE = "compute"
    STORAGE = "storage"
    NETWORK = "network"

class Provider(str, Enum):
    """LLM提供商"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE = "azure"
    GOOGLE = "google"
    LOCAL = "local"
    OTHER = "other"

@dataclass
class CostEntry:
    """成本条目"""
    entry_id: str
    cost_type: CostType
    provider: Provider
    amount: float
    unit: str  # tokens, dollars, hours, GB
    metadata: Dict = field(default_factory=dict)
    project_id: Optional[str] = None
    user_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class TokenUsage:
    """Token使用"""
    usage_id: str
    provider: Provider
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost: float = 0.0
    latency_ms: float = 0.0
    metadata: Dict = field(default_factory=dict)
    project_id: Optional[str] = None
    user_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class Budget:
    """预算配置"""
    budget_id: str
    name: str
    total_limit: float
    used_amount: float = 0.0
    cost_type: CostType = CostType.API_CALL
    period: str = "monthly"  # daily, weekly, monthly, yearly
    alert_threshold: float = 0.8  # 80%告警
    enabled: bool = True
    project_id: Optional[str] = None
    user_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class CostForecast:
    """成本预测"""
    forecast_id: str
    current_spend: float
    predicted_daily: float
    predicted_weekly: float
    predicted_monthly: float
    trend: str  # increasing, stable, decreasing
    confidence: float
    based_on_days: int
    created_at: datetime = field(default_factory=datetime.utcnow)

class CostIntelligence:
    """成本智能引擎"""
    
    def __init__(self):
        self.cost_entries: List[CostEntry] = []
        self.token_usage: List[TokenUsage] = []
        self.budgets: Dict[str, Budget] = {}
        self.forecasts: List[CostForecast] = []
        
        # 定价配置
        self.pricing: Dict[Provider, Dict] = {
            Provider.OPENAI: {
                "gpt-4o": {"prompt": 0.00003, "completion": 0.00006},  # per token
                "gpt-4o-mini": {"prompt": 0.00001, "completion": 0.00004},
                "gpt-4-turbo": {"prompt": 0.00003, "completion": 0.00006},
                "gpt-3.5-turbo": {"prompt": 0.000005, "completion": 0.000015},
            },
            Provider.ANTHROPIC: {
                "claude-sonnet-4-20250514": {"prompt": 0.00003, "completion": 0.00015},
                "claude-opus-4-20250514": {"prompt": 0.00015, "completion": 0.00075},
                "claude-haiku-3-20250514": {"prompt": 0.0000025, "completion": 0.0000125},
            },
            Provider.AZURE: {
                "gpt-4": {"prompt": 0.00003, "completion": 0.00006},
            },
            Provider.GOOGLE: {
                "gemini-pro": {"prompt": 0.000005, "completion": 0.000015},
                "gemini-ultra": {"prompt": 0.00001, "completion": 0.00003},
            },
        }
        
        # 默认定价
        self.default_pricing = {"prompt": 0.00001, "completion": 0.00004}
    
    def calculate_cost(
        self,
        provider: Provider,
        model: str,
        prompt_tokens: int,
        completion_tokens: int
    ) -> float:
        """计算API调用成本"""
        model_pricing = self.pricing.get(provider, {}).get(model, self.default_pricing)
        
        prompt_cost = model_pricing.get("prompt", 0) * prompt_tokens
        completion_cost = model_pricing.get("completion", 0) * completion_tokens
        
        return round(prompt_cost + completion_cost, 6)
    
    def track_token_usage(
        self,
        provider: Provider,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: float = 0.0,
        metadata: Optional[Dict] = None,
        project_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> TokenUsage:
        """记录Token使用"""
        cost = self.calculate_cost(
            provider, model, prompt_tokens, completion_tokens
        )
        
        usage = TokenUsage(
            usage_id=str(uuid4()),
            provider=provider,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cost=cost,
            latency_ms=latency_ms,
            metadata=metadata or {},
            project_id=project_id,
            user_id=user_id
        )
        
        self.token_usage.append(usage)
        
        # 记录成本
        self.record_cost(
            cost_type=CostType.TOKEN,
            provider=provider,
            amount=cost,
            unit="dollars",
            metadata={
                "model": model,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            },
            project_id=project_id,
            user_id=user_id
        )
        
        return usage
    
    def record_cost(
        self,
        cost_type: CostType,
        provider: Provider,
        amount: float,
        unit: str,
        metadata: Optional[Dict] = None,
        project_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> CostEntry:
        """记录成本"""
        entry = CostEntry(
            entry_id=str(uuid4()),
            cost_type=cost_type,
            provider=provider,
            amount=amount,
            unit=unit,
            metadata=metadata or {},
            project_id=project_id,
            user_id=user_id
        )
        
        self.cost_entries.append(entry)
        
        # 更新预算
        self._check_budgets(amount, cost_type, project_id, user_id)
        
        return entry
    
    def _check_budgets(
        self,
        amount: float,
        cost_type: CostType,
        project_id: Optional[str],
        user_id: Optional[str]
    ):
        """检查预算"""
        for budget in self.budgets.values():
            if not budget.enabled:
                continue
            if budget.cost_type != cost_type:
                continue
            if budget.project_id and budget.project_id != project_id:
                continue
            if budget.user_id and budget.user_id != user_id:
                continue
            
            budget.used_amount += amount
            
            # 告警检查
            usage_ratio = budget.used_amount / budget.total_limit
            if usage_ratio >= budget.alert_threshold:
                # 实际应用中会发送告警
                print(f"Budget alert: {budget.name} used {usage_ratio:.1%}")
    
    def create_budget(
        self,
        name: str,
        total_limit: float,
        cost_type: CostType = CostType.API_CALL,
        period: str = "monthly",
        alert_threshold: float = 0.8,
        project_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> Budget:
        """创建预算"""
        budget = Budget(
            budget_id=str(uuid4()),
            name=name,
            total_limit=total_limit,
            cost_type=cost_type,
            period=period,
            alert_threshold=alert_threshold,
            project_id=project_id,
            user_id=user_id
        )
        
        self.budgets[budget.budget_id] = budget
        return budget
    
    def get_budget(self, budget_id: str) -> Optional[Budget]:
        """获取预算"""
        return self.budgets.get(budget_id)
    
    def list_budgets(
        self,
        project_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> List[Budget]:
        """列出预算"""
        budgets = list(self.budgets.values())
        
        if project_id:
            budgets = [b for b in budgets if b.project_id == project_id]
        if user_id:
            budgets = [b for b in budgets if b.user_id == user_id]
        
        return budgets
    
    def get_cost_summary(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        project_id: Optional[str] = None,
        user_id: Optional[str] = None,
        provider: Optional[Provider] = None
    ) -> Dict:
        """获取成本汇总"""
        start = start_date or datetime.utcnow() - timedelta(days=30)
        end = end_date or datetime.utcnow()
        
        entries = [
            e for e in self.cost_entries
            if start <= e.created_at <= end
        ]
        
        if project_id:
            entries = [e for e in entries if e.project_id == project_id]
        if user_id:
            entries = [e for e in entries if e.user_id == user_id]
        if provider:
            entries = [e for e in entries if e.provider == provider]
        
        by_type = {}
        by_provider = {}
        for entry in entries:
            t = entry.cost_type.value
            by_type[t] = by_type.get(t, 0) + entry.amount
            p = entry.provider.value
            by_provider[p] = by_provider.get(p, 0) + entry.amount
        
        return {
            "period": {
                "start": start.isoformat(),
                "end": end.isoformat()
            },
            "total_cost": sum(e.amount for e in entries),
            "by_type": by_type,
            "by_provider": by_provider,
            "entry_count": len(entries)
        }
    
    def get_token_summary(
        self,
        days: int = 30
    ) -> Dict:
        """获取Token使用汇总"""
        cutoff = datetime.utcnow() - timedelta(days=days)
        usage = [u for u in self.token_usage if u.created_at >= cutoff]
        
        by_model = {}
        by_provider = {}
        total_prompt = 0
        total_completion = 0
        total_cost = 0.0
        
        for u in usage:
            m = u.model
            by_model[m] = by_model.get(m, 0) + u.total_tokens
            p = u.provider.value
            by_provider[p] = by_provider.get(p, 0) + u.total_tokens
            total_prompt += u.prompt_tokens
            total_completion += u.completion_tokens
            total_cost += u.cost
        
        return {
            "period_days": days,
            "total_prompt_tokens": total_prompt,
            "total_completion_tokens": total_completion,
            "total_tokens": total_prompt + total_completion,
            "total_cost": round(total_cost, 4),
            "by_model": by_model,
            "by_provider": by_provider,
            "request_count": len(usage)
        }
    
    def forecast_cost(self, days: int = 30) -> CostForecast:
        """预测未来成本"""
        recent = self.token_usage[-100:]  # 最近100次调用
        
        if not recent:
            return CostForecast(
                forecast_id=str(uuid4()),
                current_spend=0,
                predicted_daily=0,
                predicted_weekly=0,
                predicted_monthly=0,
                trend="stable",
                confidence=0.5,
                based_on_days=0
            )
        
        # 简化预测 - 基于最近使用模式
        avg_cost = sum(u.cost for u in recent) / len(recent)
        avg_tokens = sum(u.total_tokens for u in recent) / len(recent)
        
        daily_usage = avg_cost * 10  # 假设每天10倍平均调用
        weekly_usage = daily_usage * 7
        monthly_usage = daily_usage * 30
        
        # 趋势分析
        if len(recent) >= 20:
            first_half = sum(u.cost for u in recent[:10]) / 10
            second_half = sum(u.cost for u in recent[10:]) / 10
            if second_half > first_half * 1.2:
                trend = "increasing"
            elif second_half < first_half * 0.8:
                trend = "decreasing"
            else:
                trend = "stable"
        else:
            trend = "stable"
        
        forecast = CostForecast(
            forecast_id=str(uuid4()),
            current_spend=sum(u.cost for u in recent),
            predicted_daily=round(daily_usage, 4),
            predicted_weekly=round(weekly_usage, 2),
            predicted_monthly=round(monthly_usage, 2),
            trend=trend,
            confidence=0.7,
            based_on_days=days
        )
        
        self.forecasts.append(forecast)
        return forecast
    
    def get_optimization_suggestions(self) -> List[Dict]:
        """获取优化建议"""
        suggestions = []
        
        # 基于Token使用分析
        token_summary = self.get_token_summary(7)
        
        if token_summary["total_cost"] > 100:
            suggestions.append({
                "type": "cost",
                "priority": "high",
                "title": "考虑使用更小的模型",
                "description": "最近一周成本较高，考虑使用GPT-4o-mini替代GPT-4o",
                "savings_estimate": "30-50%"
            })
        
        # 基于延迟分析
        slow_requests = [u for u in self.token_usage[-100:] if u.latency_ms > 10000]
        if len(slow_requests) > 10:
            suggestions.append({
                "type": "performance",
                "priority": "medium",
                "title": "优化慢速请求",
                "description": f"最近{len(slow_requests)}个请求延迟超过10秒",
                "savings_estimate": "降低延迟50%"
            })
        
        # 基于模型使用
        by_model = token_summary.get("by_model", {})
        if "gpt-4" in by_model and by_model["gpt-4"] < 10000:
            suggestions.append({
                "type": "cost",
                "priority": "low",
                "title": "评估GPT-4使用场景",
                "description": "GPT-4使用量较低，可能不需要续订",
                "savings_estimate": "可节省$50+/月"
            })
        
        return suggestions
    
    def export_report(
        self,
        start_date: datetime,
        end_date: datetime,
        format: str = "json"
    ) -> Dict:
        """导出报告"""
        cost_summary = self.get_cost_summary(start_date, end_date)
        token_summary = self.get_token_summary(30)
        forecast = self.forecast_cost(30)
        suggestions = self.get_optimization_suggestions()
        
        return {
            "report_period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "generated_at": datetime.utcnow().isoformat(),
            "cost_summary": cost_summary,
            "token_summary": token_summary,
            "forecast": {
                "predicted_monthly": forecast.predicted_monthly,
                "trend": forecast.trend
            },
            "optimization_suggestions": suggestions
        }

# CostIntelligence实例
cost_intelligence = CostIntelligence()
