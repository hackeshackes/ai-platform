"""
Advanced Visualization 模块 v2.4
对标: W&B Reports, Neptune AI
"""
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from uuid import uuid4

class ChartType(str, Enum):
    """图表类型"""
    LINE = "line"
    BAR = "bar"
    SCATTER = "scatter"
    HISTOGRAM = "histogram"
    HEATMAP = "heatmap"
    BOX = "box"
    PARALLEL = "parallel"

class ReportType(str, Enum):
    """报表类型"""
    EXPERIMENT_COMPARISON = "experiment_comparison"
    MODEL_PERFORMANCE = "model_performance"
    COST_ANALYSIS = "cost_analysis"
    CUSTOM = "custom"

@dataclass
class ChartConfig:
    """图表配置"""
    chart_id: str
    name: str
    chart_type: ChartType
    data_source: str  # experiments, models, cost
    x_axis: str
    y_axis: List[str]
    group_by: Optional[str] = None
    filters: Dict = field(default_factory=dict)
    title: Optional[str] = None
    description: Optional[str] = None
    created_by: str
    created_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class Report:
    """报表"""
    report_id: str
    name: str
    description: str
    report_type: ReportType
    charts: List[Dict] = field(default_factory=list)
    sections: List[Dict] = field(default_factory=dict)
    template_id: Optional[str] = None
    created_by: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class ExperimentComparison:
    """实验对比配置"""
    comparison_id: str
    name: str
    experiment_ids: List[str]
    metrics: List[str]
    group_by: Optional[str] = None
    created_by: str
    created_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class Dashboard:
    """仪表板"""
    dashboard_id: str
    name: str
    description: str
    widgets: List[Dict] = field(default_factory=list)
    layout: Dict = field(default_factory=dict)
    refresh_interval: int = 300  # seconds
    created_by: str
    created_at: datetime = field(default_factory=datetime.utcnow)

class VisualizationEngine:
    """可视化引擎 v2.4"""
    
    def __init__(self):
        self.charts: Dict[str, ChartConfig] = {}
        self.reports: Dict[str, Report] = {}
        self.comparisons: Dict[str, ExperimentComparison] = {}
        self.dashboards: Dict[str, Dashboard] = {}
        
        # 初始化示例数据
        self._init_sample_data()
    
    def _init_sample_data(self):
        """初始化示例数据"""
        # 创建示例仪表板
        dashboard = Dashboard(
            dashboard_id="default",
            name="默认仪表板",
            description="系统默认仪表板",
            widgets=[
                {"type": "metric", "metric": "total_experiments"},
                {"type": "chart", "chart_type": "line", "metric": "accuracy"},
                {"type": "table", "data": "recent_experiments"}
            ],
            layout={"columns": 3},
            created_by="system"
        )
        self.dashboards[dashboard.dashboard_id] = dashboard
    
    # ==================== 图表管理 ====================
    
    def create_chart(
        self,
        name: str,
        chart_type: ChartType,
        data_source: str,
        x_axis: str,
        y_axis: List[str],
        created_by: str,
        group_by: Optional[str] = None,
        filters: Optional[Dict] = None,
        title: Optional[str] = None,
        description: Optional[str] = None
    ) -> ChartConfig:
        """创建图表"""
        chart = ChartConfig(
            chart_id=str(uuid4()),
            name=name,
            chart_type=chart_type,
            data_source=data_source,
            x_axis=x_axis,
            y_axis=y_axis,
            group_by=group_by,
            filters=filters or {},
            title=title or name,
            description=description,
            created_by=created_by
        )
        
        self.charts[chart.chart_id] = chart
        return chart
    
    def get_chart(self, chart_id: str) -> Optional[ChartConfig]:
        """获取图表"""
        return self.charts.get(chart_id)
    
    def list_charts(
        self,
        chart_type: Optional[ChartType] = None,
        data_source: Optional[str] = None
    ) -> List[ChartConfig]:
        """列出图表"""
        charts = list(self.charts.values())
        if chart_type:
            charts = [c for c in charts if c.chart_type == chart_type]
        if data_source:
            charts = [c for c in charts if c.data_source == data_source]
        return charts
    
    def delete_chart(self, chart_id: str) -> bool:
        """删除图表"""
        if chart_id in self.charts:
            del self.charts[chart_id]
            return True
        return False
    
    # ==================== 报表管理 ====================
    
    def create_report(
        self,
        name: str,
        description: str,
        report_type: ReportType,
        created_by: str,
        charts: Optional[List[Dict]] = None,
        sections: Optional[List[Dict]] = None,
        template_id: Optional[str] = None
    ) -> Report:
        """创建报表"""
        report = Report(
            report_id=str(uuid4()),
            name=name,
            description=description,
            report_type=report_type,
            charts=charts or [],
            sections=sections or [],
            template_id=template_id,
            created_by=created_by
        )
        
        self.reports[report.report_id] = report
        return report
    
    def get_report(self, report_id: str) -> Optional[Report]:
        """获取报表"""
        return self.reports.get(report_id)
    
    def list_reports(
        self,
        report_type: Optional[ReportType] = None
    ) -> List[Report]:
        """列出报表"""
        reports = list(self.reports.values())
        if report_type:
            reports = [r for r in reports if r.report_type == report_type]
        return reports
    
    def generate_report(self, report_id: str) -> Dict:
        """生成报表"""
        report = self.reports.get(report_id)
        if not report:
            raise ValueError(f"Report {report_id} not found")
        
        # 模拟报表生成
        return {
            "report_id": report_id,
            "name": report.name,
            "generated_at": datetime.utcnow().isoformat(),
            "sections": [
                {
                    "title": "Summary",
                    "content": "Generated summary content",
                    "metrics": {"total": 100, "success": 95}
                },
                {
                    "title": "Charts",
                    "charts": [
                        {"id": c["chart_id"], "data": self._generate_chart_data(c)}
                        for c in report.charts
                    ]
                }
            ]
        }
    
    def _generate_chart_data(self, chart_config: Dict) -> List[Dict]:
        """生成图表数据"""
        return [
            {"x": "2026-02-0" + str(i+1), "y": 0.8 + (i % 3) * 0.05}
            for i in range(7)
        ]
    
    def delete_report(self, report_id: str) -> bool:
        """删除报表"""
        if report_id in self.reports:
            del self.reports[report_id]
            return True
        return False
    
    # ==================== 实验对比 ====================
    
    def create_comparison(
        self,
        name: str,
        experiment_ids: List[str],
        metrics: List[str],
        created_by: str,
        group_by: Optional[str] = None
    ) -> ExperimentComparison:
        """创建实验对比"""
        comparison = ExperimentComparison(
            comparison_id=str(uuid4()),
            name=name,
            experiment_ids=experiment_ids,
            metrics=metrics,
            group_by=group_by,
            created_by=created_by
        )
        
        self.comparisons[comparison.comparison_id] = comparison
        return comparison
    
    def get_comparison(self, comparison_id: str) -> Optional[ExperimentComparison]:
        """获取对比"""
        return self.comparisons.get(comparison_id)
    
    def get_comparison_results(self, comparison_id: str) -> Dict:
        """获取对比结果"""
        comparison = self.comparisons.get(comparison_id)
        if not comparison:
            raise ValueError(f"Comparison {comparison_id} not found")
        
        # 模拟对比结果
        return {
            "comparison_id": comparison_id,
            "name": comparison.name,
            "experiments": [
                {
                    "id": exp_id,
                    "metrics": {
                        m: 0.8 + (hash(exp_id + m) % 20) / 100
                        for m in comparison.metrics
                    }
                }
                for exp_id in comparison.experiment_ids
            ],
            "summary": {
                "best_experiment": comparison.experiment_ids[0],
                "improvement": "5.2%"
            }
        }
    
    # ==================== 仪表板 ====================
    
    def create_dashboard(
        self,
        name: str,
        description: str,
        created_by: str,
        widgets: Optional[List[Dict]] = None,
        layout: Optional[Dict] = None,
        refresh_interval: int = 300
    ) -> Dashboard:
        """创建仪表板"""
        dashboard = Dashboard(
            dashboard_id=str(uuid4()),
            name=name,
            description=description,
            widgets=widgets or [],
            layout=layout or {"columns": 3},
            refresh_interval=refresh_interval,
            created_by=created_by
        )
        
        self.dashboards[dashboard.dashboard_id] = dashboard
        return dashboard
    
    def get_dashboard(self, dashboard_id: str) -> Optional[Dashboard]:
        """获取仪表板"""
        return self.dashboards.get(dashboard_id)
    
    def list_dashboards(self) -> List[Dashboard]:
        """列出仪表板"""
        return list(self.dashboards.values())
    
    def update_dashboard(self, dashboard_id: str, widgets: List[Dict]) -> bool:
        """更新仪表板"""
        dashboard = self.dashboards.get(dashboard_id)
        if not dashboard:
            return False
        
        dashboard.widgets = widgets
        dashboard.updated_at = datetime.utcnow()
        return True
    
    # ==================== 数据查询 ====================
    
    def query_metrics(
        self,
        experiment_ids: List[str],
        metrics: List[str],
        time_range: Optional[Dict] = None
    ) -> List[Dict]:
        """查询指标数据"""
        # 模拟查询结果
        return [
            {
                "experiment_id": exp_id,
                "metric": metric,
                "values": [
                    {"timestamp": f"2026-02-0{i+1}T10:00:00Z", "value": 0.8 + (i % 5) * 0.02}
                    for i in range(7)
                ]
            }
            for exp_id in experiment_ids
            for metric in metrics
        ]
    
    def get_metric_summary(self, metric: str) -> Dict:
        """获取指标汇总"""
        return {
            "metric": metric,
            "min": 0.5,
            "max": 0.99,
            "mean": 0.85,
            "std": 0.08
        }
    
    # ==================== 统计信息 ====================
    
    def get_summary(self) -> Dict:
        """获取统计"""
        return {
            "total_charts": len(self.charts),
            "total_reports": len(self.reports),
            "total_comparisons": len(self.comparisons),
            "total_dashboards": len(self.dashboards)
        }

# VisualizationEngine实例
visualization_engine = VisualizationEngine()
