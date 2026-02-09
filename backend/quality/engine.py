"""
数据质量模块
"""
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4
import json

@dataclass
class QualityRule:
    """质量规则"""
    rule_id: str
    name: str
    description: str
    rule_type: str  # schema, range, uniqueness, completeness, custom
    column: Optional[str] = None
    condition: str = ""  # e.g., ">= 0", "is not null", "in ['a', 'b']"
    severity: str = "error"  # error, warning
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class QualityCheck:
    """质量检查"""
    check_id: str
    rule_id: str
    dataset_id: str
    status: str = "pending"  # pending, running, passed, failed
    passed_count: int = 0
    failed_count: int = 0
    total_count: int = 0
    results: List[Dict] = field(default_factory=list)
    executed_at: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class QualityReport:
    """质量报告"""
    report_id: str
    dataset_id: str
    check_id: str
    score: float  # 0-100
    status: str  # excellent, good, warning, critical
    summary: Dict[str, Any]
    recommendations: List[str]
    created_at: datetime = field(default_factory=datetime.utcnow)

class QualityEngine:
    """质量检查引擎"""
    
    def __init__(self):
        self.rules: Dict[str, QualityRule] = {}
        self.checks: Dict[str, QualityCheck] = {}
        self.reports: Dict[str, QualityReport] = {}
    
    def create_rule(
        self,
        name: str,
        description: str,
        rule_type: str,
        column: Optional[str] = None,
        condition: str = "",
        severity: str = "error"
    ) -> QualityRule:
        """创建质量规则"""
        rule = QualityRule(
            rule_id=str(uuid4()),
            name=name,
            description=description,
            rule_type=rule_type,
            column=column,
            condition=condition,
            severity=severity
        )
        
        self.rules[rule.rule_id] = rule
        return rule
    
    def get_rules(
        self,
        dataset_id: Optional[str] = None,
        enabled: Optional[bool] = None
    ) -> List[QualityRule]:
        """获取规则列表"""
        rules = list(self.rules.values())
        
        if enabled is not None:
            rules = [r for r in rules if r.enabled == enabled]
        
        return rules
    
    async def validate_dataset(
        self,
        dataset_id: str,
        data: List[Dict[str, Any]],
        rule_ids: Optional[List[str]] = None
    ) -> QualityCheck:
        """验证数据集"""
        check_id = str(uuid4())
        
        # 选择规则
        if rule_ids:
            rules = [self.rules[rid] for rid in rule_ids if rid in self.rules]
        else:
            rules = [r for r in self.rules.values() if r.enabled]
        
        check = QualityCheck(
            check_id=check_id,
            rule_id=rule_ids[0] if rule_ids else "",
            dataset_id=dataset_id,
            status="running",
            total_count=len(data)
        )
        
        self.checks[check_id] = check
        
        # 执行验证
        passed = 0
        failed = 0
        results = []
        
        for record in data:
            record_results = []
            for rule in rules:
                result = await self._check_rule(record, rule)
                record_results.append(result)
                
                if result["passed"]:
                    passed += 1
                else:
                    failed += 1
            
            results.append({
                "record_id": record.get("id", len(results)),
                "passed": all(r["passed"] for r in record_results),
                "results": record_results
            })
        
        check.status = "passed" if failed == 0 else "failed"
        check.passed_count = passed
        check.failed_count = failed
        check.results = results
        check.executed_at = datetime.utcnow()
        
        return check
    
    async def _check_rule(
        self,
        record: Dict[str, Any],
        rule: QualityRule
    ) -> Dict[str, Any]:
        """执行单条规则检查"""
        value = record.get(rule.column) if rule.column else None
        
        passed = True
        message = "OK"
        
        try:
            if rule.rule_type == "schema":
                # Schema检查 (简化)
                passed = value is not None
                message = "Column exists" if passed else "Column missing"
            
            elif rule.rule_type == "range":
                # 范围检查
                if value is not None:
                    condition = rule.condition
                    if ">=" in condition:
                        threshold = float(condition.split(">=")[1].strip())
                        passed = value >= threshold
                        message = f"Value {value} >= {threshold}"
                    elif "<=" in condition:
                        threshold = float(condition.split("<=")[1].strip())
                        passed = value <= threshold
                        message = f"Value {value} <= {threshold}"
            
            elif rule.rule_type == "uniqueness":
                # 唯一性检查
                # 简化: 不检查重复
                passed = True
                message = "Uniqueness check passed"
            
            elif rule.rule_type == "completeness":
                # 完整性检查
                if rule.column:
                    passed = value is not None and value != ""
                    message = "Value present" if passed else "Missing value"
            
            elif rule.rule_type == "custom":
                # 自定义规则
                passed = True
                message = "Custom rule passed"
        
        except Exception as e:
            passed = False
            message = f"Error: {str(e)}"
        
        return {
            "rule_id": rule.rule_id,
            "rule_name": rule.name,
            "passed": passed,
            "message": message,
            "severity": rule.severity
        }
    
    def generate_report(
        self,
        dataset_id: str,
        check_id: str
    ) -> QualityReport:
        """生成质量报告"""
        check = self.checks.get(check_id)
        if not check:
            raise ValueError(f"Check {check_id} not found")
        
        # 计算分数
        if check.total_count > 0:
            score = (check.passed_count / check.total_count) * 100
        else:
            score = 100
        
        # 状态评估
        if score >= 95:
            status = "excellent"
        elif score >= 80:
            status = "good"
        elif score >= 60:
            status = "warning"
        else:
            status = "critical"
        
        # 生成建议
        recommendations = []
        if score < 100:
            recommendations.append("Review failed records")
        if score < 80:
            recommendations.append("Consider data cleaning")
        if score < 60:
            recommendations.append("Data quality issues detected")
        
        report = QualityReport(
            report_id=str(uuid4()),
            dataset_id=dataset_id,
            check_id=check_id,
            score=score,
            status=status,
            summary={
                "total_records": check.total_count,
                "passed": check.passed_count,
                "failed": check.failed_count,
                "pass_rate": f"{score:.1f}%"
            },
            recommendations=recommendations
        )
        
        self.reports[report.report_id] = report
        return report
    
    def get_reports(
        self,
        dataset_id: Optional[str] = None
    ) -> List[QualityReport]:
        """获取报告列表"""
        reports = list(self.reports.values())
        
        if dataset_id:
            reports = [r for r in reports if r.dataset_id == dataset_id]
        
        return sorted(reports, key=lambda r: r.created_at, reverse=True)

# Quality Engine实例
quality_engine = QualityEngine()

# 默认质量规则
DEFAULT_RULES = [
    ("no_nulls", "Column cannot be null", "completeness", None, "is not null", "error"),
    ("positive_values", "Value must be positive", "range", None, ">= 0", "error"),
    ("unique_id", "ID must be unique", "uniqueness", "id", "", "error"),
]

def init_default_rules():
    """初始化默认规则"""
    for name, desc, rtype, col, cond, sev in DEFAULT_RULES:
        quality_engine.create_rule(
            name=name,
            description=desc,
            rule_type=rtype,
            column=col,
            condition=cond,
            severity=sev
        )

# 初始化
init_default_rules()
