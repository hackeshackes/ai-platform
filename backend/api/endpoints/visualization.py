"""
Visualization API端点 v2.4
"""
from fastapi import APIRouter, HTTPException
from typing import List, Optional, Dict, Any

# 直接导入模块
import importlib.util
import os

backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
module_path = os.path.join(backend_dir, 'visualization/engine.py')

spec = importlib.util.spec_from_file_location("viz_module", module_path)
module = importlib.util.module_from_spec(spec)

try:
    spec.loader.exec_module(module)
    visualization_engine = module.visualization_engine
    ChartType = module.ChartType
    ReportType = module.ReportType
except Exception as e:
    print(f"Failed to import visualization module: {e}")
    visualization_engine = None
    ChartType = None
    ReportType = None

router = APIRouter()

from pydantic import BaseModel

class CreateChartModel(BaseModel):
    name: str
    chart_type: str
    data_source: str
    x_axis: str
    y_axis: List[str]
    group_by: Optional[str] = None
    title: Optional[str] = None
    description: Optional[str] = None

class CreateReportModel(BaseModel):
    name: str
    description: str
    report_type: str
    charts: Optional[List[Dict]] = None
    sections: Optional[List[Dict]] = None

class CreateComparisonModel(BaseModel):
    name: str
    experiment_ids: List[str]
    metrics: List[str]
    group_by: Optional[str] = None

class CreateDashboardModel(BaseModel):
    name: str
    description: str
    widgets: Optional[List[Dict]] = None
    layout: Optional[Dict] = None
    refresh_interval: int = 300

# ==================== 图表管理 ====================

@router.get("/charts")
async def list_charts(
    chart_type: Optional[str] = None,
    data_source: Optional[str] = None
):
    """列出图表"""
    ctype = ChartType(chart_type) if chart_type else None
    charts = visualization_engine.list_charts(chart_type=ctype, data_source=data_source)
    
    return {
        "total": len(charts),
        "charts": [
            {
                "chart_id": c.chart_id,
                "name": c.name,
                "type": c.chart_type.value,
                "data_source": c.data_source,
                "created_by": c.created_by
            }
            for c in charts
        ]
    }

@router.post("/charts")
async def create_chart(request: CreateChartModel):
    """创建图表"""
    try:
        ctype = ChartType(request.chart_type)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid chart type: {request.chart_type}")
    
    chart = visualization_engine.create_chart(
        name=request.name,
        chart_type=ctype,
        data_source=request.data_source,
        x_axis=request.x_axis,
        y_axis=request.y_axis,
        created_by="user",
        group_by=request.group_by,
        title=request.title,
        description=request.description
    )
    
    return {
        "chart_id": chart.chart_id,
        "name": chart.name,
        "message": "Chart created"
    }

@router.get("/charts/{chart_id}")
async def get_chart(chart_id: str):
    """获取图表"""
    chart = visualization_engine.get_chart(chart_id)
    if not chart:
        raise HTTPException(status_code=404, detail="Chart not found")
    
    return {
        "chart_id": chart.chart_id,
        "name": chart.name,
        "type": chart.chart_type.value,
        "data_source": chart.data_source,
        "x_axis": chart.x_axis,
        "y_axis": chart.y_axis,
        "config": chart.filters
    }

@router.delete("/charts/{chart_id}")
async def delete_chart(chart_id: str):
    """删除图表"""
    result = visualization_engine.delete_chart(chart_id)
    if not result:
        raise HTTPException(status_code=404, detail="Chart not found")
    return {"message": "Chart deleted"}

# ==================== 报表管理 ====================

@router.get("/reports")
async def list_reports(report_type: Optional[str] = None):
    """列出报表"""
    rtype = ReportType(report_type) if report_type else None
    reports = visualization_engine.list_reports(report_type=rtype)
    
    return {
        "total": len(reports),
        "reports": [
            {
                "report_id": r.report_id,
                "name": r.name,
                "type": r.report_type.value,
                "created_by": r.created_by
            }
            for r in reports
        ]
    }

@router.post("/reports")
async def create_report(request: CreateReportModel):
    """创建报表"""
    try:
        rtype = ReportType(request.report_type)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid report type: {request.report_type}")
    
    report = visualization_engine.create_report(
        name=request.name,
        description=request.description,
        report_type=rtype,
        created_by="user",
        charts=request.charts,
        sections=request.sections
    )
    
    return {
        "report_id": report.report_id,
        "name": report.name,
        "message": "Report created"
    }

@router.get("/reports/{report_id}")
async def get_report(report_id: str):
    """获取报表"""
    report = visualization_engine.get_report(report_id)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    
    return {
        "report_id": report.report_id,
        "name": report.name,
        "description": report.description,
        "type": report.report_type.value,
        "charts_count": len(report.charts),
        "created_by": report.created_by
    }

@router.post("/reports/{report_id}/generate")
async def generate_report(report_id: str):
    """生成报表"""
    try:
        result = visualization_engine.generate_report(report_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.delete("/reports/{report_id}")
async def delete_report(report_id: str):
    """删除报表"""
    result = visualization_engine.delete_report(report_id)
    if not result:
        raise HTTPException(status_code=404, detail="Report not found")
    return {"message": "Report deleted"}

# ==================== 实验对比 ====================

@router.post("/comparisons")
async def create_comparison(request: CreateComparisonModel):
    """创建实验对比"""
    comparison = visualization_engine.create_comparison(
        name=request.name,
        experiment_ids=request.experiment_ids,
        metrics=request.metrics,
        created_by="user",
        group_by=request.group_by
    )
    
    return {
        "comparison_id": comparison.comparison_id,
        "name": comparison.name,
        "message": "Comparison created"
    }

@router.get("/comparisons/{comparison_id}")
async def get_comparison(comparison_id: str):
    """获取对比"""
    comparison = visualization_engine.get_comparison(comparison_id)
    if not comparison:
        raise HTTPException(status_code=404, detail="Comparison not found")
    
    return {
        "comparison_id": comparison.comparison_id,
        "name": comparison.name,
        "experiments_count": len(comparison.experiment_ids),
        "metrics": comparison.metrics
    }

@router.get("/comparisons/{comparison_id}/results")
async def get_comparison_results(comparison_id: str):
    """获取对比结果"""
    try:
        results = visualization_engine.get_comparison_results(comparison_id)
        return results
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

# ==================== 仪表板 ====================

@router.get("/dashboards")
async def list_dashboards():
    """列出仪表板"""
    dashboards = visualization_engine.list_dashboards()
    
    return {
        "total": len(dashboards),
        "dashboards": [
            {
                "dashboard_id": d.dashboard_id,
                "name": d.name,
                "description": d.description,
                "widgets_count": len(d.widgets)
            }
            for d in dashboards
        ]
    }

@router.post("/dashboards")
async def create_dashboard(request: CreateDashboardModel):
    """创建仪表板"""
    dashboard = visualization_engine.create_dashboard(
        name=request.name,
        description=request.description,
        created_by="user",
        widgets=request.widgets,
        layout=request.layout,
        refresh_interval=request.refresh_interval
    )
    
    return {
        "dashboard_id": dashboard.dashboard_id,
        "name": dashboard.name,
        "message": "Dashboard created"
    }

@router.get("/dashboards/{dashboard_id}")
async def get_dashboard(dashboard_id: str):
    """获取仪表板"""
    dashboard = visualization_engine.get_dashboard(dashboard_id)
    if not dashboard:
        raise HTTPException(status_code=404, detail="Dashboard not found")
    
    return {
        "dashboard_id": dashboard.dashboard_id,
        "name": dashboard.name,
        "widgets": dashboard.widgets,
        "layout": dashboard.layout
    }

@router.put("/dashboards/{dashboard_id}")
async def update_dashboard(dashboard_id: str, widgets: List[Dict]):
    """更新仪表板"""
    result = visualization_engine.update_dashboard(dashboard_id, widgets)
    if not result:
        raise HTTPException(status_code=404, detail="Dashboard not found")
    return {"message": "Dashboard updated"}

# ==================== 数据查询 ====================

@router.post("/query/metrics")
async def query_metrics(
    experiment_ids: List[str],
    metrics: List[str],
    time_range: Optional[Dict] = None
):
    """查询指标数据"""
    results = visualization_engine.query_metrics(
        experiment_ids=experiment_ids,
        metrics=metrics,
        time_range=time_range
    )
    return {"data": results}

@router.get("/metrics/{metric}/summary")
async def get_metric_summary(metric: str):
    """获取指标汇总"""
    summary = visualization_engine.get_metric_summary(metric)
    return summary

@router.get("/summary")
async def get_summary():
    """获取统计"""
    return visualization_engine.get_summary()

@router.get("/health")
async def visualization_health():
    """健康检查"""
    return {
        "status": "healthy",
        "charts": len(visualization_engine.charts),
        "reports": len(visualization_engine.reports)
    }
