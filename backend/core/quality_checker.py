"""
数据质量检查服务 v1.1
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class QualityReport:
    """数据质量报告"""
    dataset_id: int
    version_id: Optional[int]
    checked_at: datetime
    
    # 基础统计
    total_rows: int
    total_columns: int
    
    # 空值检测
    null_counts: Dict[str, int]
    null_percentages: Dict[str, float]
    columns_with_nulls: List[str]
    null_quality_score: float  # 0-100
    
    # 重复检测
    duplicate_rows: int
    duplicate_percentage: float
    duplicate_quality_score: float  # 0-100
    
    # 格式检测
    format_issues: List[Dict[str, Any]]
    format_quality_score: float  # 0-100
    
    # 总体评分
    overall_score: float
    issues: List[str]
    recommendations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "version_id": self.version_id,
            "checked_at": self.checked_at.isoformat(),
            "total_rows": self.total_rows,
            "total_columns": self.total_columns,
            "null_counts": self.null_counts,
            "null_percentages": self.null_percentages,
            "columns_with_nulls": self.columns_with_nulls,
            "null_quality_score": self.null_quality_score,
            "duplicate_rows": self.duplicate_rows,
            "duplicate_percentage": self.duplicate_percentage,
            "duplicate_quality_score": self.duplicate_quality_score,
            "format_issues": self.format_issues,
            "format_quality_score": self.format_quality_score,
            "overall_score": self.overall_score,
            "issues": self.issues,
            "recommendations": self.recommendations,
        }


class DataQualityChecker:
    """数据质量检查器"""
    
    NULL_THRESHOLD = 0.1  # 10% 空值阈值
    DUPLICATE_THRESHOLD = 0.05  # 5% 重复阈值
    
    def __init__(self, null_threshold: float = None, duplicate_threshold: float = None):
        self.null_threshold = null_threshold or self.NULL_THRESHOLD
        self.duplicate_threshold = duplicate_threshold or self.DUPLICATE_THRESHOLD
    
    def check(self, df: pd.DataFrame, dataset_id: int, version_id: int = None) -> QualityReport:
        """执行完整的数据质量检查"""
        
        issues = []
        recommendations = []
        
        # 1. 空值检测
        null_counts = df.isnull().sum().to_dict()
        null_percentages = (df.isnull().sum() / len(df) * 100).to_dict()
        columns_with_nulls = [col for col, pct in null_percentages.items() if pct > 0]
        
        if len(columns_with_nulls) > 0:
            issues.append(f"发现 {len(columns_with_nulls)} 个列存在空值")
            if len(columns_with_nulls) > len(df.columns) * 0.5:
                recommendations.append("超过50%的列存在空值，建议检查数据收集流程")
        
        null_quality_score = self._calculate_null_score(null_percentages)
        
        # 2. 重复检测
        duplicate_rows = df.duplicated().sum()
        duplicate_percentage = duplicate_rows / len(df) if len(df) > 0 else 0
        
        if duplicate_rows > 0:
            issues.append(f"发现 {duplicate_rows} 行重复数据 ({duplicate_percentage*100:.2f}%)")
            if duplicate_percentage > self.duplicate_threshold:
                recommendations.append("重复数据比例过高，建议去重")
        
        duplicate_quality_score = max(0, 100 - duplicate_percentage * 100)
        
        # 3. 格式检测
        format_issues = self._check_format(df)
        format_quality_score = self._calculate_format_score(format_issues)
        
        if len(format_issues) > 0:
            issues.append(f"发现 {len(format_issues)} 个格式问题")
            recommendations.extend([issue["suggestion"] for issue in format_issues[:3]])
        
        # 4. 总体评分
        overall_score = (null_quality_score + duplicate_quality_score + format_quality_score) / 3
        
        return QualityReport(
            dataset_id=dataset_id,
            version_id=version_id,
            checked_at=datetime.utcnow(),
            total_rows=len(df),
            total_columns=len(df.columns),
            null_counts=null_counts,
            null_percentages=null_percentages,
            columns_with_nulls=columns_with_nulls,
            null_quality_score=null_quality_score,
            duplicate_rows=duplicate_rows,
            duplicate_percentage=duplicate_percentage,
            duplicate_quality_score=duplicate_quality_score,
            format_issues=format_issues,
            format_quality_score=format_quality_score,
            overall_score=overall_score,
            issues=issues,
            recommendations=recommendations,
        )
    
    def _calculate_null_score(self, null_percentages: Dict[str, float]) -> float:
        """计算空值质量分数"""
        if not null_percentages:
            return 100.0
        
        scores = []
        for col, pct in null_percentages.items():
            if pct == 0:
                scores.append(100)
            elif pct < self.null_threshold * 100:
                scores.append(80)
            elif pct < self.null_threshold * 200:
                scores.append(60)
            else:
                scores.append(0)
        
        return np.mean(scores) if scores else 100.0
    
    def _check_format(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """检查数据格式问题"""
        issues = []
        
        # 检查每列
        for col in df.columns:
            # 检查数据类型一致性
            if df[col].dtype == 'object':
                # 检查是否应该是数值
                try:
                    numeric_count = pd.to_numeric(df[col], errors='coerce').notna().sum()
                    if numeric_count / len(df) > 0.8:
                        issues.append({
                            "column": col,
                            "issue": "大量数值但存储为文本",
                            "suggestion": f"列 '{col}' 建议转换为数值类型"
                        })
                except:
                    pass
        
        # 检查索引
        if not isinstance(df.index, pd.RangeIndex):
            issues.append({
                "column": "index",
                "issue": "非标准索引",
                "suggestion": "建议使用RangeIndex"
            })
        
        return issues
    
    def _calculate_format_score(self, format_issues: List[Dict[str, Any]]) -> float:
        """计算格式质量分数"""
        if not format_issues:
            return 100.0
        
        # 严重程度权重
        severe_count = sum(1 for issue in format_issues if "数值" in issue.get("issue", ""))
        minor_count = len(format_issues) - severe_count
        
        score = 100 - (severe_count * 20) - (minor_count * 5)
        return max(0, score)
    
    def quick_check(self, df: pd.DataFrame) -> Dict[str, Any]:
        """快速检查，返回关键指标"""
        return {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "null_percentage": float(df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100),
            "duplicate_percentage": float(df.duplicated().sum() / len(df) * 100) if len(df) > 0 else 0,
            "memory_usage_mb": float(df.memory_usage(deep=True).sum() / 1024 / 1024),
        }
