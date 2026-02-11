"""
Predictor - 预测分析引擎
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
from datetime import datetime, timedelta
import math
import json


class TrendDirection(Enum):
    """趋势方向"""
    UP = "up"
    DOWN = "down"
    STABLE = "stable"
    VOLATILE = "volatile"


class SeasonalityType(Enum):
    """季节性类型"""
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"


@dataclass
class ForecastPoint:
    """预测点"""
    timestamp: datetime
    value: float
    lower_bound: float
    upper_bound: float


@dataclass
class TrendAnalysis:
    """趋势分析结果"""
    direction: TrendDirection
    slope: float  # 趋势斜率
    strength: float  # 趋势强度 0-1
    start_date: datetime
    end_date: datetime
    data_points: int
    r_squared: float  # 拟合优度


@dataclass
class SeasonalityAnalysis:
    """季节性分析结果"""
    has_seasonality: bool
    type: Optional[SeasonalityType]
    amplitude: float  # 季节性振幅
    phase: float  # 相位偏移
    period: int  # 周期长度
    strength: float  # 季节性强度 0-1
    peaks: List[Dict[str, Any]] = field(default_factory=list)  # 高峰时段
    troughs: List[Dict[str, Any]] = field(default_factory=list)  # 低谷时段


@dataclass
class Anomaly:
    """异常检测结果"""
    timestamp: datetime
    value: float
    expected_value: float
    deviation: float  # 偏差程度
    severity: str  # low, medium, high, critical
    description: str
    category: str  # spike, drop, persistent, trend_break


@dataclass
class PredictionResult:
    """预测结果"""
    forecast: List[ForecastPoint]
    confidence_interval: Tuple[float, float]
    trend: TrendAnalysis
    seasonality: SeasonalityAnalysis
    anomalies: List[Anomaly]
    model_accuracy: float
    forecast_horizon: int
    generated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class Predictor:
    """
    预测分析引擎
    
    功能：
    1. 时间序列预测
    2. 季节性分析
    3. 异常检测
    4. 置信区间计算
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.default_horizon = self.config.get('default_horizon', 30)
        self.confidence_level = self.config.get('confidence_level', 0.95)
        
    async def predict(
        self, 
        data: Dict[str, Any], 
        horizon: int = None
    ) -> PredictionResult:
        """
        生成预测
        
        Args:
            data: 业务数据
            horizon: 预测时间范围（天数）
            
        Returns:
            PredictionResult: 预测结果
        """
        horizon = horizon or self.default_horizon
        
        # 1. 时间序列预测
        trend = self.time_series_predict(data, horizon)
        
        # 2. 季节性分析
        seasonality = self.analyze_seasonality(data)
        
        # 3. 异常检测
        anomalies = self.detect_anomalies(data)
        
        # 4. 计算置信区间
        confidence = self.calculate_confidence(trend, anomalies)
        confidence_interval = self.get_confidence_interval(trend, confidence)
        
        # 5. 生成完整预测
        forecast = self.generate_forecast(trend, seasonality, horizon, confidence)
        
        # 6. 计算模型准确度
        model_accuracy = self.estimate_model_accuracy(data)
        
        return PredictionResult(
            forecast=forecast,
            confidence_interval=confidence_interval,
            trend=trend,
            seasonality=seasonality,
            anomalies=anomalies,
            model_accuracy=model_accuracy,
            forecast_horizon=horizon
        )
    
    def time_series_predict(
        self, 
        data: Dict[str, Any], 
        horizon: int
    ) -> TrendAnalysis:
        """
        时间序列预测
        
        Args:
            data: 业务数据
            horizon: 预测天数
            
        Returns:
            TrendAnalysis: 趋势分析结果
        """
        # 提取时间序列数据
        time_series = data.get('time_series', [])
        
        if not time_series:
            # 使用业务指标生成模拟数据
            time_series = self._generate_simulated_series(data)
        
        # 计算趋势
        direction = self._determine_trend_direction(time_series)
        slope = self._calculate_slope(time_series)
        strength = self._calculate_trend_strength(time_series, slope)
        
        # 统计信息
        start_date = datetime.now() - timedelta(days=len(time_series))
        end_date = datetime.now()
        
        # 拟合优度
        r_squared = self._calculate_r_squared(time_series, slope)
        
        return TrendAnalysis(
            direction=direction,
            slope=slope,
            strength=strength,
            start_date=start_date,
            end_date=end_date,
            data_points=len(time_series),
            r_squared=r_squared
        )
    
    def analyze_seasonality(self, data: Dict[str, Any]) -> SeasonalityAnalysis:
        """
        季节性分析
        
        Args:
            data: 业务数据
            
        Returns:
            SeasonalityAnalysis: 季节性分析结果
        """
        time_series = data.get('time_series', [])
        
        if len(time_series) < 14:
            # 数据不足，返回无季节性
            return SeasonalityAnalysis(
                has_seasonality=False,
                type=None,
                amplitude=0,
                phase=0,
                period=0,
                strength=0
            )
        
        # 检测季节性
        has_weekly = self._check_weekly_seasonality(time_series)
        has_monthly = self._check_monthly_seasonality(time_series)
        has_quarterly = self._check_quarterly_seasonality(time_series)
        
        # 确定主要季节性类型
        if has_quarterly:
            seasonality_type = SeasonalityType.QUARTERLY
            period = 90
        elif has_monthly:
            seasonality_type = SeasonalityType.MONTHLY
            period = 30
        elif has_weekly:
            seasonality_type = SeasonalityType.WEEKLY
            period = 7
        else:
            return SeasonalityAnalysis(
                has_seasonality=False,
                type=None,
                amplitude=0,
                phase=0,
                period=0,
                strength=0
            )
        
        # 计算振幅
        amplitude = self._calculate_seasonality_amplitude(time_series, period)
        
        # 计算相位
        phase = self._calculate_seasonality_phase(time_series, period)
        
        # 计算季节性强度
        strength = self._calculate_seasonality_strength(time_series, period)
        
        # 识别高峰和低谷
        peaks, troughs = self._identify_peaks_troughs(time_series, period)
        
        return SeasonalityAnalysis(
            has_seasonality=True,
            type=seasonality_type,
            amplitude=amplitude,
            phase=phase,
            period=period,
            strength=strength,
            peaks=peaks,
            troughs=troughs
        )
    
    def detect_anomalies(self, data: Dict[str, Any]) -> List[Anomaly]:
        """
        异常检测
        
        Args:
            data: 业务数据
            
        Returns:
            List[Anomaly]: 检测到的异常列表
        """
        time_series = data.get('time_series', [])
        detected_anomalies = []
        
        if len(time_series) < 10:
            return detected_anomalies
        
        # 计算统计量
        values = time_series if isinstance(time_series[0], (int, float)) else [t.get('value', 0) for t in time_series]
        
        mean_val = sum(values) / len(values)
        std_val = self._standard_deviation(values)
        
        if std_val == 0:
            return detected_anomalies
        
        # Z-score 异常检测
        threshold = 2.5  # Z-score阈值
        
        for i, value in enumerate(values):
            if isinstance(time_series[i], dict):
                timestamp = time_series[i].get('timestamp', datetime.now() - timedelta(days=len(values)-i))
                actual_value = time_series[i].get('value', value)
            else:
                timestamp = datetime.now() - timedelta(days=len(values)-i)
                actual_value = value
            
            z_score = (actual_value - mean_val) / std_val
            
            if abs(z_score) > threshold:
                deviation = abs(z_score) / threshold
                severity = self._get_anomaly_severity(deviation)
                description = self._get_anomaly_description(z_score, actual_value, mean_val)
                category = self._get_anomaly_category(z_score, values, i)
                
                detected_anomalies.append(Anomaly(
                    timestamp=timestamp,
                    value=actual_value,
                    expected_value=mean_val,
                    deviation=deviation,
                    severity=severity,
                    description=description,
                    category=category
                ))
        
        return detected_anomalies
    
    def calculate_confidence(
        self, 
        trend: TrendAnalysis, 
        anomalies: List[Anomaly]
    ) -> float:
        """
        计算置信度
        
        Args:
            trend: 趋势分析
            anomalies: 检测到的异常
            
        Returns:
            float: 置信度分数 0-1
        """
        # 基于趋势强度
        trend_confidence = trend.strength
        
        # 基于拟合优度
        model_confidence = trend.r_squared
        
        # 基于异常数量（异常越多，置信度越低）
        anomaly_penalty = min(len(anomalies) * 0.05, 0.2)
        
        # 综合置信度
        confidence = (
            trend_confidence * 0.4 +
            model_confidence * 0.4 +
            (1 - anomaly_penalty) * 0.2
        )
        
        return round(min(max(confidence, 0), 1), 3)
    
    def get_confidence_interval(
        self, 
        trend: TrendAnalysis, 
        confidence: float
    ) -> Tuple[float, float]:
        """
        获取置信区间
        
        Args:
            trend: 趋势分析
            confidence: 置信度
            
        Returns:
            Tuple[float, float]: (下限, 上限)
        """
        # 基础不确定性
        base_uncertainty = 0.1
        
        # 基于趋势方向调整
        if trend.direction == TrendDirection.VOLATILE:
            base_uncertainty *= 2
        elif trend.direction == TrendDirection.STABLE:
            base_uncertainty *= 0.8
        
        # 基于趋势强度调整
        strength_factor = 1 - trend.strength * 0.5
        
        # 综合不确定性
        uncertainty = base_uncertainty * strength_factor
        
        # 置信区间宽度
        interval_width = uncertainty * 2 * (1 - confidence + 0.5)
        
        return (-interval_width, interval_width)
    
    def generate_forecast(
        self,
        trend: TrendAnalysis,
        seasonality: SeasonalityAnalysis,
        horizon: int,
        confidence: float
    ) -> List[ForecastPoint]:
        """
        生成预测
        
        Args:
            trend: 趋势分析
            seasonality: 季节性分析
            horizon: 预测天数
            confidence: 置信度
            
        Returns:
            List[ForecastPoint]: 预测点列表
        """
        forecast = []
        
        # 基础值
        base_value = 100  # 归一化基础值
        
        # 趋势影响因子
        trend_factor = 1 + trend.slope * 0.01
        
        for day in range(1, horizon + 1):
            timestamp = datetime.now() + timedelta(days=day)
            
            # 趋势预测值
            trend_value = base_value * (trend_factor ** day)
            
            # 季节性调整
            if seasonality.has_seasonality:
                day_of_period = day % seasonality.period
                seasonal_factor = 1 + seasonality.amplitude * math.sin(
                    2 * math.pi * day_of_period / seasonality.period + seasonality.phase
                )
                trend_value *= seasonal_factor
            
            # 置信区间
            interval = self.get_confidence_interval(trend, confidence)
            lower = trend_value * (1 + interval[0])
            upper = trend_value * (1 + interval[1])
            
            forecast.append(ForecastPoint(
                timestamp=timestamp,
                value=round(trend_value, 2),
                lower_bound=round(lower, 2),
                upper_bound=round(upper, 2)
            ))
        
        return forecast
    
    def estimate_model_accuracy(self, data: Dict[str, Any]) -> float:
        """
        估计模型准确度
        
        Args:
            data: 业务数据
            
        Returns:
            float: 准确度估计 0-1
        """
        time_series = data.get('time_series', [])
        
        if len(time_series) < 10:
            return 0.6  # 数据不足，默认60%
        
        # 基于数据量调整
        data_factor = min(len(time_series) / 100, 1.0) * 0.3
        
        # 基于数据质量
        data_quality = self._assess_data_quality(data) * 0.4
        
        # 基于历史准确度
        historical_accuracy = data.get('model_accuracy', 0.8) * 0.3
        
        return round(data_factor + data_quality + historical_accuracy, 3)
    
    # ============ 私有辅助方法 ============
    
    def _generate_simulated_series(self, data: Dict[str, Any]) -> List[float]:
        """生成模拟时间序列数据"""
        n_points = 60  # 默认60个数据点
        
        # 基础趋势
        base_trend = data.get('revenue_growth', 0.05) / 30  # 日增长率
        base_value = 100
        
        series = []
        for i in range(n_points):
            value = base_value * ((1 + base_trend) ** i)
            # 添加噪声
            noise = (value * 0.02) * (hash(i) % 10 - 5) / 5
            value += noise
            series.append(max(0, value))
        
        return series
    
    def _determine_trend_direction(self, series: List[float]) -> TrendDirection:
        """判断趋势方向"""
        if len(series) < 2:
            return TrendDirection.STABLE
        
        first_half = sum(series[:len(series)//2]) / (len(series)//2)
        second_half = sum(series[len(series)//2:]) / (len(series) - len(series)//2)
        
        change_rate = (second_half - first_half) / first_half
        
        if abs(change_rate) < 0.02:
            # 检查波动性
            volatility = self._volatility(series)
            if volatility > 0.1:
                return TrendDirection.VOLATILE
            return TrendDirection.STABLE
        elif change_rate > 0:
            return TrendDirection.UP
        else:
            return TrendDirection.DOWN
    
    def _calculate_slope(self, series: List[float]) -> float:
        """计算趋势斜率"""
        if len(series) < 2:
            return 0
        
        # 简单线性回归斜率
        n = len(series)
        x_sum = sum(range(n))
        y_sum = sum(series)
        xy_sum = sum(x * y for x, y in enumerate(series))
        x2_sum = sum(x * x for x in range(n))
        
        slope = (n * xy_sum - x_sum * y_sum) / (n * x2_sum - x_sum * x_sum)
        
        return round(slope, 4)
    
    def _calculate_trend_strength(self, series: List[float], slope: float) -> float:
        """计算趋势强度"""
        if len(series) < 2:
            return 0
        
        mean_val = sum(series) / len(series)
        if mean_val == 0:
            return 0
        
        # 相对斜率
        relative_slope = abs(slope * len(series)) / mean_val
        
        # 转换为0-1范围
        strength = min(relative_slope, 1.0)
        
        return round(strength, 3)
    
    def _calculate_r_squared(self, series: List[float], slope: float) -> float:
        """计算拟合优度"""
        if len(series) < 2:
            return 0
        
        n = len(series)
        mean_val = sum(series) / n
        
        # 总平方和
        ss_total = sum((y - mean_val) ** 2 for y in series)
        
        # 残差平方和
        ss_residual = sum((y - (slope * x + mean_val)) ** 2 for x, y in enumerate(series))
        
        if ss_total == 0:
            return 1.0
        
        r_squared = 1 - ss_residual / ss_total
        
        return round(max(0, min(1, r_squared)), 3)
    
    def _check_weekly_seasonality(self, series: List[float]) -> bool:
        """检查周季节性"""
        if len(series) < 14:
            return False
        
        # 比较工作日与周末数据
        weekdays = []
        weekends = []
        for i, value in enumerate(series):
            day_of_week = i % 7
            if day_of_week < 5:  # 工作日
                weekdays.append(value)
            else:
                weekends.append(value)
        
        if not weekdays or not weekends:
            return False
        
        weekday_avg = sum(weekdays) / len(weekdays)
        weekend_avg = sum(weekends) / len(weekends)
        
        if weekday_avg == 0:
            return False
        
        ratio = abs(weekday_avg - weekend_avg) / weekday_avg
        
        return ratio > 0.1
    
    def _check_monthly_seasonality(self, series: List[float]) -> bool:
        """检查月季节性"""
        if len(series) < 60:
            return False
        
        # 按月分组
        monthly_avgs = []
        for month in range(12):
            month_values = [series[i] for i in range(month, len(series), 12)]
            if month_values:
                monthly_avgs.append(sum(month_values) / len(month_values))
        
        if len(monthly_avgs) < 6:
            return False
        
        overall_avg = sum(monthly_avgs) / len(monthly_avgs)
        variance = sum((m - overall_avg) ** 2 for m in monthly_avgs) / len(monthly_avgs)
        
        return variance / overall_avg > 0.05
    
    def _check_quarterly_seasonality(self, series: List[float]) -> bool:
        """检查季度季节性"""
        if len(series) < 90:
            return False
        
        # 按季度分组
        quarterly_avgs = []
        for quarter in range(4):
            quarter_values = series[quarter * 90 : (quarter + 1) * 90]
            if quarter_values:
                quarterly_avgs.append(sum(quarter_values) / len(quarter_values))
        
        if len(quarterly_avgs) < 2:
            return False
        
        overall_avg = sum(quarterly_avgs) / len(quarterly_avgs)
        variance = sum((q - overall_avg) ** 2 for q in quarterly_avgs) / len(quarterly_avgs)
        
        return variance / overall_avg > 0.08
    
    def _calculate_seasonality_amplitude(
        self, 
        series: List[float], 
        period: int
    ) -> float:
        """计算季节性振幅"""
        if len(series) < period * 2:
            return 0
        
        period_averages = []
        for i in range(period):
            values = [series[j] for j in range(i, len(series), period)]
            if values:
                period_averages.append(sum(values) / len(values))
        
        if not period_averages:
            return 0
        
        overall_avg = sum(period_averages) / len(period_averages)
        if overall_avg == 0:
            return 0
        
        amplitude = (max(period_averages) - min(period_averages)) / 2 / overall_avg
        
        return round(amplitude, 3)
    
    def _calculate_seasonality_phase(
        self, 
        series: List[float], 
        period: int
    ) -> float:
        """计算季节性相位"""
        # 简化实现，返回0
        return 0.0
    
    def _calculate_seasonality_strength(
        self, 
        series: List[float], 
        period: int
    ) -> float:
        """计算季节性强度"""
        amplitude = self._calculate_seasonality_amplitude(series, period)
        
        # 振幅的平方作为强度指标
        return round(min(amplitude * 2, 1.0), 3)
    
    def _identify_peaks_troughs(
        self, 
        series: List[float], 
        period: int
    ) -> Tuple[List[Dict], List[Dict]]:
        """识别高峰和低谷"""
        peaks = []
        troughs = []
        
        if len(series) < period:
            return peaks, troughs
        
        period_averages = []
        for i in range(period):
            values = [series[j] for j in range(i, len(series), period)]
            if values:
                period_averages.append((i, sum(values) / len(values)))
        
        if not period_averages:
            return peaks, troughs
        
        avg_value = sum(p[1] for p in period_averages) / len(period_averages)
        
        for position, value in period_averages:
            if value > avg_value * 1.1:
                peaks.append({
                    'position': position,
                    'value': value,
                    'description': f'第{position}个时段为高峰'
                })
            elif value < avg_value * 0.9:
                troughs.append({
                    'position': position,
                    'value': value,
                    'description': f'第{position}个时段为低谷'
                })
        
        return peaks, troughs
    
    def _standard_deviation(self, values: List[float]) -> float:
        """计算标准差"""
        if len(values) < 2:
            return 0
        
        mean_val = sum(values) / len(values)
        variance = sum((v - mean_val) ** 2 for v in values) / (len(values) - 1)
        
        return math.sqrt(variance)
    
    def _volatility(self, series: List[float]) -> float:
        """计算波动率"""
        if len(series) < 2:
            return 0
        
        returns = []
        for i in range(1, len(series)):
            if series[i-1] != 0:
                returns.append((series[i] - series[i-1]) / series[i-1])
        
        if not returns:
            return 0
        
        return self._standard_deviation(returns)
    
    def _get_anomaly_severity(self, deviation: float) -> str:
        """获取异常严重程度"""
        if deviation < 0.5:
            return "low"
        elif deviation < 1.0:
            return "medium"
        elif deviation < 2.0:
            return "high"
        else:
            return "critical"
    
    def _get_anomaly_description(
        self, 
        z_score: float, 
        value: float, 
        expected: float
    ) -> str:
        """获取异常描述"""
        if z_score > 0:
            return f"数值异常偏高: {value:.2f} (期望值: {expected:.2f})"
        else:
            return f"数值异常偏低: {value:.2f} (期望值: {expected:.2f})"
    
    def _get_anomaly_category(
        self, 
        z_score: float, 
        values: List[float], 
        index: int
    ) -> str:
        """获取异常类别"""
        if index > 0 and index < len(values) - 1:
            prev_diff = values[index] - values[index-1]
            next_diff = values[index+1] - values[index]
            
            if prev_diff * next_diff < 0:
                return "trend_break"
        
        if z_score > 0:
            return "spike"
        else:
            return "drop"
    
    def _assess_data_quality(self, data: Dict[str, Any]) -> float:
        """评估数据质量"""
        time_series = data.get('time_series', [])
        
        if not time_series:
            return 0.5
        
        # 检查数据完整性
        completeness = min(len(time_series) / 100, 1.0)
        
        # 检查数据一致性
        consistency = 1.0
        if len(time_series) > 1:
            values = [t.get('value', t) if isinstance(t, dict) else t for t in time_series]
            consistency = 1 - min(self._volatility(values), 1.0)
        
        return (completeness + consistency) / 2
