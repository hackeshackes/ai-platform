"""
Chart Generator - AI Platform v5

图表生成模块 - 生成各类训练可视化图表数据
"""
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import copy


@dataclass
class LossChartConfig:
    """Loss曲线配置"""
    title: str = "Training Loss"
    x_label: str = "Steps"
    y_label: str = "Loss"
    show_train_val: bool = True
    colors: List[str] = field(default_factory=lambda: ["#3b82f6", "#ef4444"])
    smooth_factor: float = 0.1  # 平滑因子 (0-1)


@dataclass
class GPUChartConfig:
    """GPU监控配置"""
    title: str = "GPU Usage"
    x_label: str = "Time"
    y_label: str = "Usage (%)"
    show_memory: bool = True
    colors: Dict[str, str] = field(default_factory=lambda: {
        "utilization": "#10b981",
        "memory": "#f59e0b",
        "temperature": "#ef4444"
    })


@dataclass
class MetricsChartConfig:
    """评估指标配置"""
    title: str = "Evaluation Metrics"
    x_label: str = "Epoch"
    y_label: str = "Score"
    metric_types: List[str] = field(default_factory=lambda: ["accuracy", "f1", "precision", "recall"])
    colors: List[str] = field(default_factory=lambda: [
        "#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6"
    ])


@dataclass
class LearningRateChartConfig:
    """学习率曲线配置"""
    title: str = "Learning Rate"
    x_label: str = "Steps"
    y_label: str = "Learning Rate"
    colors: List[str] = field(default_factory=lambda: ["#8b5cf6"])


class ChartGenerator:
    """
    图表数据生成器
    
    生成符合Chart.js/Recharts格式的图表数据
    """
    
    def __init__(self):
        self.smooth_factor = 0.1
    
    def _smooth_data(self, data: List[float], factor: float = 0.1) -> List[float]:
        """数据平滑处理"""
        if not data or len(data) < 2:
            return data
        
        smoothed = [data[0]]
        for i in range(1, len(data)):
            smoothed.append(smoothed[-1] * factor + data[i] * (1 - factor))
        return smoothed
    
    def _format_timestamp(self, dt: datetime) -> str:
        """格式化时间戳"""
        return dt.strftime("%H:%M:%S")
    
    # ============ Loss Chart ============
    
    def generate_loss_chart(
        self,
        train_loss: List[float],
        val_loss: Optional[List[float]] = None,
        steps: Optional[List[int]] = None,
        config: Optional[LossChartConfig] = None
    ) -> Dict[str, Any]:
        """
        生成Loss曲线图表数据
        
        Args:
            train_loss: 训练损失列表
            val_loss: 验证损失列表 (可选)
            steps: 步数列表 (可选，默认自动生成)
            config: 图表配置
        
        Returns:
            Chart.js格式的图表数据
        """
        if config is None:
            config = LossChartConfig()
        
        if steps is None:
            steps = list(range(len(train_loss)))
        
        # 数据平滑
        train_loss_smoothed = self._smooth_data(train_loss, config.smooth_factor)
        val_loss_smoothed = self._smooth_data(val_loss, config.smooth_factor) if val_loss else None
        
        datasets = [
            {
                "label": "Training Loss",
                "data": train_loss_smoothed,
                "borderColor": config.colors[0],
                "backgroundColor": f"{config.colors[0]}20",
                "fill": True,
                "tension": 0.4,
                "pointRadius": 0,
                "borderWidth": 2,
            }
        ]
        
        if val_loss_smoothed and config.show_train_val:
            datasets.append({
                "label": "Validation Loss",
                "data": val_loss_smoothed,
                "borderColor": config.colors[1],
                "backgroundColor": f"{config.colors[1]}20",
                "fill": True,
                "tension": 0.4,
                "pointRadius": 0,
                "borderWidth": 2,
            })
        
        return {
            "type": "line",
            "data": {
                "labels": steps,
                "datasets": datasets,
            },
            "options": {
                "responsive": True,
                "maintainAspectRatio": False,
                "plugins": {
                    "legend": {
                        "display": True,
                        "position": "top",
                    },
                    "title": {
                        "display": True,
                        "text": config.title,
                        "font": {"size": 16},
                    },
                },
                "scales": {
                    "x": {
                        "title": {
                            "display": True,
                            "text": config.x_label,
                        },
                        "grid": {
                            "display": False,
                        },
                    },
                    "y": {
                        "title": {
                            "display": True,
                            "text": config.y_label,
                        },
                        "beginAtZero": True,
                    },
                },
                "interaction": {
                    "mode": "index",
                    "intersect": False,
                },
            },
        }
    
    # ============ GPU Chart ============
    
    def generate_gpu_chart(
        self,
        gpu_utilization: List[float],
        gpu_memory: Optional[List[float]] = None,
        gpu_temperature: Optional[List[float]] = None,
        timestamps: Optional[List[str]] = None,
        config: Optional[GPUChartConfig] = None
    ) -> Dict[str, Any]:
        """
        生成GPU监控图表数据
        
        Args:
            gpu_utilization: GPU利用率列表
            gpu_memory: GPU内存使用列表
            gpu_temperature: GPU温度列表
            timestamps: 时间戳列表
            config: 图表配置
        
        Returns:
            Chart.js格式的图表数据
        """
        if config is None:
            config = GPUChartConfig()
        
        if timestamps is None:
            timestamps = [self._format_timestamp(datetime.now()) for _ in range(len(gpu_utilization))]
        
        datasets = []
        
        # GPU利用率
        datasets.append({
            "label": "GPU Utilization (%)",
            "data": gpu_utilization,
            "borderColor": config.colors["utilization"],
            "backgroundColor": f"{config.colors['utilization']}20",
            "fill": True,
            "tension": 0.4,
            "pointRadius": 0,
            "borderWidth": 2,
            "yAxisID": "y",
        })
        
        # GPU内存
        if gpu_memory and config.show_memory:
            datasets.append({
                "label": "GPU Memory (MB)",
                "data": gpu_memory,
                "borderColor": config.colors["memory"],
                "backgroundColor": f"{config.colors['memory']}20",
                "fill": True,
                "tension": 0.4,
                "pointRadius": 0,
                "borderWidth": 2,
                "yAxisID": "y1",
            })
        
        # GPU温度
        if gpu_temperature:
            datasets.append({
                "label": "Temperature (°C)",
                "data": gpu_temperature,
                "borderColor": config.colors["temperature"],
                "backgroundColor": "transparent",
                "fill": False,
                "tension": 0.4,
                "pointRadius": 2,
                "borderWidth": 2,
                "yAxisID": "y2",
            })
        
        return {
            "type": "line",
            "data": {
                "labels": timestamps,
                "datasets": datasets,
            },
            "options": {
                "responsive": True,
                "maintainAspectRatio": False,
                "plugins": {
                    "legend": {
                        "display": True,
                        "position": "top",
                    },
                    "title": {
                        "display": True,
                        "text": config.title,
                        "font": {"size": 16},
                    },
                },
                "scales": {
                    "x": {
                        "title": {
                            "display": True,
                            "text": config.x_label,
                        },
                        "grid": {
                            "display": False,
                        },
                    },
                    "y": {
                        "type": "linear",
                        "display": True,
                        "position": "left",
                        "title": {
                            "display": True,
                            "text": "Usage (%)",
                        },
                        "min": 0,
                        "max": 100,
                    },
                    "y1": {
                        "type": "linear",
                        "display": True,
                        "position": "right",
                        "title": {
                            "display": True,
                            "text": "Memory (MB)",
                        },
                        "grid": {
                            "drawOnChartArea": False,
                        },
                    },
                    "y2": {
                        "type": "linear",
                        "display": False,
                        "min": 0,
                        "max": 100,
                    },
                },
            },
        }
    
    # ============ Metrics Chart ============
    
    def generate_metrics_chart(
        self,
        metrics: Dict[str, List[float]],
        epochs: Optional[List[int]] = None,
        config: Optional[MetricsChartConfig] = None
    ) -> Dict[str, Any]:
        """
        生成评估指标图表数据
        
        Args:
            metrics: 指标字典，key为指标名，value为分数列表
            epochs: 轮次列表
            config: 图表配置
        
        Returns:
            Chart.js格式的图表数据
        """
        if config is None:
            config = MetricsChartConfig()
        
        if epochs is None:
            epochs = list(range(len(list(metrics.values())[0]))) if metrics else []
        
        datasets = []
        color_idx = 0
        for metric_name, values in metrics.items():
            if metric_name in config.metric_types:
                color = config.colors[color_idx % len(config.colors)]
                datasets.append({
                    "label": metric_name.capitalize(),
                    "data": values,
                    "borderColor": color,
                    "backgroundColor": f"{color}20",
                    "fill": True,
                    "tension": 0.4,
                    "pointRadius": 3,
                    "borderWidth": 2,
                })
                color_idx += 1
        
        return {
            "type": "line",
            "data": {
                "labels": epochs,
                "datasets": datasets,
            },
            "options": {
                "responsive": True,
                "maintainAspectRatio": False,
                "plugins": {
                    "legend": {
                        "display": True,
                        "position": "top",
                    },
                    "title": {
                        "display": True,
                        "text": config.title,
                        "font": {"size": 16},
                    },
                },
                "scales": {
                    "x": {
                        "title": {
                            "display": True,
                            "text": config.x_label,
                        },
                        "grid": {
                            "display": False,
                        },
                    },
                    "y": {
                        "title": {
                            "display": True,
                            "text": config.y_label,
                        },
                        "min": 0,
                        "max": 1,
                    },
                },
            },
        }
    
    # ============ Learning Rate Chart ============
    
    def generate_lr_chart(
        self,
        learning_rates: List[float],
        steps: Optional[List[int]] = None,
        config: Optional[LearningRateChartConfig] = None
    ) -> Dict[str, Any]:
        """
        生成学习率曲线图表数据
        
        Args:
            learning_rates: 学习率列表
            steps: 步数列表
            config: 图表配置
        
        Returns:
            Chart.js格式的图表数据
        """
        if config is None:
            config = LearningRateChartConfig()
        
        if steps is None:
            steps = list(range(len(learning_rates)))
        
        return {
            "type": "line",
            "data": {
                "labels": steps,
                "datasets": [
                    {
                        "label": "Learning Rate",
                        "data": learning_rates,
                        "borderColor": config.colors[0],
                        "backgroundColor": f"{config.colors[0]}20",
                        "fill": True,
                        "tension": 0.4,
                        "pointRadius": 0,
                        "borderWidth": 2,
                    }
                ],
            },
            "options": {
                "responsive": True,
                "maintainAspectRatio": False,
                "plugins": {
                    "legend": {
                        "display": True,
                        "position": "top",
                    },
                    "title": {
                        "display": True,
                        "text": config.title,
                        "font": {"size": 16},
                    },
                },
                "scales": {
                    "x": {
                        "title": {
                            "display": True,
                            "text": config.x_label,
                        },
                        "grid": {
                            "display": False,
                        },
                    },
                    "y": {
                        "title": {
                            "display": True,
                            "text": config.y_label,
                        },
                        "type": "logarithmic" if max(learning_rates) > 0 else "linear",
                    },
                },
            },
        }
    
    # ============ Dashboard ============
    
    def generate_dashboard(
        self,
        job_data: Dict[str, Any],
        include_gpu: bool = True,
        include_metrics: bool = True
    ) -> Dict[str, Any]:
        """
        生成完整的训练仪表盘数据
        
        Args:
            job_data: 训练作业数据
            include_gpu: 是否包含GPU图表
            include_metrics: 是否包含评估指标图表
        
        Returns:
            仪表盘完整数据
        """
        dashboard = {
            "overview": {
                "job_id": job_data.get("job_id"),
                "status": job_data.get("status", "running"),
                "current_epoch": job_data.get("current_epoch", 0),
                "total_epochs": job_data.get("total_epochs", 1),
                "current_step": job_data.get("current_step", 0),
                "total_steps": job_data.get("total_steps", 0),
                "elapsed_time": job_data.get("elapsed_time", 0),
                "estimated_remaining": job_data.get("estimated_remaining", 0),
                "last_updated": datetime.now().isoformat(),
            },
            "charts": {},
        }
        
        # Loss曲线
        if "loss" in job_data:
            train_loss = job_data["loss"].get("train", [])
            val_loss = job_data["loss"].get("val")
            dashboard["charts"]["loss"] = self.generate_loss_chart(
                train_loss, val_loss,
                config=LossChartConfig(title=f"Loss - {job_data.get('job_id', 'Training')}")
            )
        
        # GPU监控
        if include_gpu and "gpu" in job_data:
            gpu_data = job_data["gpu"]
            dashboard["charts"]["gpu"] = self.generate_gpu_chart(
                gpu_data.get("utilization", []),
                gpu_data.get("memory"),
                gpu_data.get("temperature"),
                config=GPUChartConfig(title=f"GPU Monitor - {job_data.get('job_id', 'Training')}")
            )
        
        # 学习率
        if "learning_rate" in job_data:
            dashboard["charts"]["learning_rate"] = self.generate_lr_chart(
                job_data["learning_rate"],
                config=LearningRateChartConfig(
                    title=f"Learning Rate - {job_data.get('job_id', 'Training')}"
                )
            )
        
        # 评估指标
        if include_metrics and "metrics" in job_data:
            dashboard["charts"]["metrics"] = self.generate_metrics_chart(
                job_data["metrics"],
                config=MetricsChartConfig(
                    title=f"Metrics - {job_data.get('job_id', 'Training')}"
                )
            )
        
        return dashboard
    
    def export_chart_json(self, chart_data: Dict[str, Any], indent: int = 2) -> str:
        """导出图表数据为JSON字符串"""
        return json.dumps(chart_data, indent=indent, ensure_ascii=False)


# 全局单例
_chart_generator_instance: Optional[ChartGenerator] = None


def get_chart_generator() -> ChartGenerator:
    """获取图表生成器单例"""
    global _chart_generator_instance
    if _chart_generator_instance is None:
        _chart_generator_instance = ChartGenerator()
    return _chart_generator_instance
