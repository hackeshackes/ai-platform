"""
合规检查模块
实现多种数据保护法规的合规检查
"""

import re
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class RegulationType(str, Enum):
    """法规类型"""
    GDPR = "GDPR"  # 欧盟通用数据保护条例
    PIPL = "PIPL"  # 中国个人信息保护法
    CCPA = "CCPA"  # 加州消费者隐私法案
    HIPAA = "HIPAA"  # 健康保险便携性和责任法案
    PCI_DSS = "PCI_DSS"  # 支付卡行业数据安全标准
    SOX = "SOX"  # 萨班斯-奥克斯利法案
    ISO27001 = "ISO27001"  # 信息安全管理体系
    CUSTOM = "CUSTOM"  # 自定义规则


class ComplianceLevel(str, Enum):
    """合规级别"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ComplianceIssue:
    """合规问题"""
    code: str
    title: str
    description: str
    level: ComplianceLevel
    regulation: RegulationType
    affected_fields: List[str]
    recommendation: str
    timestamp: datetime


class DataCategory(str, Enum):
    """数据类别"""
    PERSONAL = "personal"
    SENSITIVE = "sensitive"
    FINANCIAL = "financial"
    HEALTH = "health"
    BIOMETRIC = "biometric"
    GENETIC = "genetic"
    LOCATION = "location"
    CHILDREN = "children"
    PUBLIC = "public"


class ComplianceChecker:
    """
    合规检查器

    支持:
    - GDPR 第15-22条 (数据主体权利)
    - PIPL 个人信息保护法
    - CCPA 加州消费者隐私法
    - PCI-DSS 支付卡标准
    """

    def __init__(self):
        self.rules: Dict[str, Dict] = {}
        self._init_default_rules()

    def _init_default_rules(self) -> None:
        """初始化默认规则"""

        # ============= GDPR 规则 =============

        self.rules["GDPR_DATA_MINIMIZATION"] = {
            "regulation": RegulationType.GDPR,
            "level": ComplianceLevel.ERROR,
            "check": self._check_data_minimization
        }

        self.rules["GDPR_CONSENT_RECORD"] = {
            "regulation": RegulationType.GDPR,
            "level": ComplianceLevel.ERROR,
            "check": self._check_consent_record
        }

        self.rules["GDPR_RIGHT_TO_ERASURE"] = {
            "regulation": RegulationType.GDPR,
            "level": ComplianceLevel.WARNING,
            "check": self._check_erasure_capability
        }

        self.rules["GDPR_DATA_RETENTION"] = {
            "regulation": RegulationType.GDPR,
            "level": ComplianceLevel.WARNING,
            "check": self._check_data_retention
        }

        # ============= PIPL 规则 =============

        self.rules["PIPL_BASIC_INFO"] = {
            "regulation": RegulationType.PIPL,
            "level": ComplianceLevel.INFO,
            "check": self._check_pipl_basic_info
        }

        self.rules["PIPL_SENSITIVE_HANDLING"] = {
            "regulation": RegulationType.PIPL,
            "level": ComplianceLevel.ERROR,
            "check": self._check_sensitive_handling
        }

        # ============= PCI-DSS 规则 =============

        self.rules["PCI_CARD_NUMBER_MASKING"] = {
            "regulation": RegulationType.PCI_DSS,
            "level": ComplianceLevel.ERROR,
            "check": self._check_card_number_masking
        }

        self.rules["PCI_ENCRYPTION"] = {
            "regulation": RegulationType.PCI_DSS,
            "level": ComplianceLevel.CRITICAL,
            "check": self._check_pci_encryption
        }

        # ============= 通用规则 =============

        self.rules["SENSITIVE_DATA_EXPOSURE"] = {
            "regulation": RegulationType.CUSTOM,
            "level": ComplianceLevel.ERROR,
            "check": self._check_sensitive_exposure
        }

        self.rules["PII_ENCRYPTION"] = {
            "regulation": RegulationType.CUSTOM,
            "level": ComplianceLevel.CRITICAL,
            "check": self._check_pii_encryption
        }

    # ============= 检查方法 =============

    def _check_data_minimization(
        self,
        data: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Tuple[bool, Optional[ComplianceIssue]]:
        """检查数据最小化"""
        # 定义必要字段
        necessary_fields = context.get("necessary_fields", [])
        collected_fields = set(data.keys())

        extra_fields = collected_fields - set(necessary_fields)

        if extra_fields:
            return False, ComplianceIssue(
                code="GDPR_DATA_MINIMIZATION",
                title="收集了非必要的数据字段",
                description=f"收集了额外的数据字段: {', '.join(extra_fields)}",
                level=ComplianceLevel.WARNING,
                regulation=RegulationType.GDPR,
                affected_fields=list(extra_fields),
                recommendation="仅收集业务必需的数据字段",
                timestamp=datetime.utcnow()
            )

        return True, None

    def _check_consent_record(
        self,
        data: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Tuple[bool, Optional[ComplianceIssue]]:
        """检查同意记录"""
        required_consent_fields = ["consent_given", "consent_date", "consent_version"]

        missing = [f for f in required_consent_fields if f not in data]

        if missing:
            return False, ComplianceIssue(
                code="GDPR_CONSENT_RECORD",
                title="缺少必要的同意记录字段",
                description=f"缺少同意记录字段: {', '.join(missing)}",
                level=ComplianceLevel.ERROR,
                regulation=RegulationType.GDPR,
                affected_fields=missing,
                recommendation="记录用户同意的时间、版本和范围",
                timestamp=datetime.utcnow()
            )

        return True, None

    def _check_erasure_capability(
        self,
        data: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Tuple[bool, Optional[ComplianceIssue]]:
        """检查数据擦除能力"""
        # 检查是否有唯一标识符用于删除请求
        identifier_field = context.get("identifier_field", "id")

        if identifier_field not in data:
            return False, ComplianceIssue(
                code="GDPR_RIGHT_TO_ERASURE",
                title="缺少数据标识符",
                description="数据缺少唯一标识符，无法响应删除请求",
                level=ComplianceLevel.WARNING,
                regulation=RegulationType.GDPR,
                affected_fields=[identifier_field],
                recommendation="确保每条记录有唯一标识符",
                timestamp=datetime.utcnow()
            )

        return True, None

    def _check_data_retention(
        self,
        data: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Tuple[bool, Optional[ComplianceIssue]]:
        """检查数据保留期限"""
        retention_days = context.get("retention_days", 365)
        record_date = data.get("created_at") or data.get("date")

        if record_date:
            if isinstance(record_date, str):
                record_date = datetime.fromisoformat(record_date)

            age = (datetime.utcnow() - record_date).days

            if age > retention_days:
                return False, ComplianceIssue(
                    code="GDPR_DATA_RETENTION",
                    title="数据超过保留期限",
                    description=f"数据已保留 {age} 天，超过规定的 {retention_days} 天",
                    level=ComplianceLevel.WARNING,
                    regulation=RegulationType.GDPR,
                    affected_fields=["created_at", "date"],
                    recommendation="删除或归档超过保留期限的数据",
                    timestamp=datetime.utcnow()
                )

        return True, None

    def _check_pipl_basic_info(
        self,
        data: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Tuple[bool, Optional[ComplianceIssue]]:
        """检查PIPL基本信息保护"""
        required_fields = ["data_subject_id", "purpose", "legal_basis"]

        missing = [f for f in required_fields if f not in data]

        if missing:
            return False, ComplianceIssue(
                code="PIPL_BASIC_INFO",
                title="缺少PIPL要求的基本信息",
                description=f"缺少必需字段: {', '.join(missing)}",
                level=ComplianceLevel.INFO,
                regulation=RegulationType.PIPL,
                affected_fields=missing,
                recommendation="添加数据主体ID、处理目的和法律依据",
                timestamp=datetime.utcnow()
            )

        return True, None

    def _check_sensitive_handling(
        self,
        data: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Tuple[bool, Optional[ComplianceIssue]]:
        """检查敏感数据处理"""
        sensitive_fields = ["biometric", "genetic", "health", "financial"]

        found_sensitive = [f for f in sensitive_fields if f in data]

        if found_sensitive:
            # 检查是否有额外的安全措施
            if context.get("sensitive_protection") != "enhanced":
                return False, ComplianceIssue(
                    code="PIPL_SENSITIVE_HANDLING",
                    title="敏感数据需要增强保护",
                    description=f"检测到敏感数据字段: {', '.join(found_sensitive)}",
                    level=ComplianceLevel.ERROR,
                    regulation=RegulationType.PIPL,
                    affected_fields=found_sensitive,
                    recommendation="对敏感数据实施加密存储和访问控制",
                    timestamp=datetime.utcnow()
                )

        return True, None

    def _check_card_number_masking(
        self,
        data: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Tuple[bool, Optional[ComplianceIssue]]:
        """检查卡号脱敏"""
        card_fields = ["card_number", "credit_card", "debit_card"]

        for field in card_fields:
            if field in data:
                value = str(data[field])

                # 检查是否已脱敏
                if len(value) >= 4:
                    visible_digits = len(re.findall(r"\d", value))
                    if visible_digits > 4:
                        return False, ComplianceIssue(
                            code="PCI_CARD_NUMBER_MASKING",
                            title="信用卡号未完全脱敏",
                            description=f"字段 '{field}' 包含可见的卡号",
                            level=ComplianceLevel.ERROR,
                            regulation=RegulationType.PCI_DSS,
                            affected_fields=[field],
                            recommendation="卡号只显示前6后4位，中间用*代替",
                            timestamp=datetime.utcnow()
                        )

        return True, None

    def _check_pci_encryption(
        self,
        data: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Tuple[bool, Optional[ComplianceIssue]]:
        """检查PCI加密"""
        card_fields = ["card_number", "credit_card", "cvv", "pin"]

        for field in card_fields:
            if field in data:
                if not context.get("encrypted", False):
                    return False, ComplianceIssue(
                        code="PCI_ENCRYPTION",
                        title="支付数据未加密",
                        description=f"敏感支付字段 '{field}' 未加密",
                        level=ComplianceLevel.CRITICAL,
                        regulation=RegulationType.PCI_DSS,
                        affected_fields=[field],
                        recommendation="使用AES-256或更强加密算法保护支付数据",
                        timestamp=datetime.utcnow()
                    )

        return True, None

    def _check_sensitive_exposure(
        self,
        data: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Tuple[bool, Optional[ComplianceIssue]]:
        """检查敏感数据暴露"""
        sensitive_patterns = {
            "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
            "phone": r"1[3-9]\d{9}",
            "id_card": r"\d{17}[\dXx]",
            "ssn": r"\d{3}-\d{2}-\d{4}"
        }

        issues = []

        for field, value in data.items():
            if isinstance(value, str):
                for pattern_name, pattern in sensitive_patterns.items():
                    if re.search(pattern, value) and not context.get("masked", False):
                        issues.append(field)
                        break

        if issues:
            return False, ComplianceIssue(
                code="SENSITIVE_DATA_EXPOSURE",
                title="检测到敏感数据暴露风险",
                description=f"字段可能包含未脱敏的敏感数据: {', '.join(issues)}",
                level=ComplianceLevel.ERROR,
                regulation=RegulationType.CUSTOM,
                affected_fields=issues,
                recommendation="对敏感数据进行脱敏处理",
                timestamp=datetime.utcnow()
            )

        return True, None

    def _check_pii_encryption(
        self,
        data: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Tuple[bool, Optional[ComplianceIssue]]:
        """检查PII加密"""
        pii_fields = ["password", "secret", "token", "api_key", "private_key"]

        for field in pii_fields:
            if field in data:
                value = str(data[field])

                # 检查是否明文存储
                if not value.startswith("***") and not context.get("encrypted", False):
                    return False, ComplianceIssue(
                        code="PII_ENCRYPTION",
                        title="PII数据未加密",
                        description=f"字段 '{field}' 可能包含明文敏感信息",
                        level=ComplianceLevel.CRITICAL,
                        regulation=RegulationType.CUSTOM,
                        affected_fields=[field],
                        recommendation="使用加密存储敏感信息",
                        timestamp=datetime.utcnow()
                    )

        return True, None

    # ============= 公共接口 =============

    def check(
        self,
        data: Dict[str, Any],
        regulations: Optional[List[RegulationType]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> List[ComplianceIssue]:
        """
        执行合规检查

        Args:
            data: 待检查的数据
            regulations: 适用的法规列表
            context: 额外的上下文信息

        Returns:
            发现的问题列表
        """
        context = context or {}
        issues = []

        for rule_name, rule in self.rules.items():
            # 如果指定了法规，过滤
            if regulations:
                if rule["regulation"] not in regulations:
                    continue

            passed, issue = rule["check"](data, context)
            if not passed and issue:
                issues.append(issue)

        return issues

    def check_field(
        self,
        field_name: str,
        field_value: Any,
        regulations: List[RegulationType]
    ) -> List[ComplianceIssue]:
        """检查单个字段"""
        data = {field_name: field_value}
        return self.check(data, regulations)

    def validate_data_classification(
        self,
        data: Dict[str, Any]
    ) -> Dict[DataCategory, List[str]]:
        """
        验证数据分类

        Args:
            data: 待分类的数据

        Returns:
            数据类别映射
        """
        classification: Dict[DataCategory, List[str]] = {
            DataCategory.PERSONAL: [],
            DataCategory.SENSITIVE: [],
            DataCategory.FINANCIAL: [],
            DataCategory.HEALTH: [],
            DataCategory.BIOMETRIC: [],
            DataCategory.GENETIC: [],
            DataCategory.LOCATION: [],
            DataCategory.CHILDREN: [],
            DataCategory.PUBLIC: []
        }

        patterns = {
            DataCategory.PERSONAL: ["name", "email", "phone", "address"],
            DataCategory.SENSITIVE: ["password", "secret", "token"],
            DataCategory.FINANCIAL: ["card_number", "cvv", "bank_account"],
            DataCategory.HEALTH: ["diagnosis", "medical", "health", "disease"],
            DataCategory.BIOMETRIC: ["fingerprint", "face", "voice", "biometric"],
            DataCategory.GENETIC: ["genetic", "dna", "chromosome"],
            DataCategory.LOCATION: ["gps", "latitude", "longitude", "location"],
            DataCategory.CHILDREN: ["child", "minor", "age"]
        }

        for field_name in data.keys():
            field_lower = field_name.lower()

            for category, keywords in patterns.items():
                if any(kw in field_lower for kw in keywords):
                    classification[category].append(field_name)
                    break

        # 清理空类别
        return {k: v for k, v in classification.items() if v}

    def generate_compliance_report(
        self,
        data: Dict[str, Any],
        regulations: List[RegulationType],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        生成合规报告

        Args:
            data: 待检查的数据
            regulations: 适用的法规
            context: 上下文信息

        Returns:
            合规报告
        """
        issues = self.check(data, regulations, context)
        classification = self.validate_data_classification(data)

        # 按级别分组
        by_level: Dict[ComplianceLevel, List[Dict]] = {
            level: [] for level in ComplianceLevel
        }

        for issue in issues:
            issue_dict = {
                "code": issue.code,
                "title": issue.title,
                "description": issue.description,
                "affected_fields": issue.affected_fields,
                "recommendation": issue.recommendation
            }
            by_level[issue.level].append(issue_dict)

        # 统计
        summary = {
            "total_issues": len(issues),
            "by_level": {
                level.value: len(items)
                for level, items in by_level.items()
            },
            "by_regulation": {}
        }

        for issue in issues:
            reg = issue.regulation.value
            summary["by_regulation"][reg] = summary["by_regulation"].get(reg, 0) + 1

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "regulations_checked": [r.value for r in regulations],
            "classification": {
                cat.value: fields for cat, fields in classification.items()
            },
            "summary": summary,
            "issues": by_level,
            "compliance_score": self._calculate_compliance_score(issues)
        }

    def _calculate_compliance_score(self, issues: List[ComplianceIssue]) -> float:
        """计算合规分数 (0-100)"""
        if not issues:
            return 100.0

        weights = {
            ComplianceLevel.INFO: 1,
            ComplianceLevel.WARNING: 3,
            ComplianceLevel.ERROR: 5,
            ComplianceLevel.CRITICAL: 10
        }

        total_weight = sum(weights.get(i.level, 1) for i in issues)
        max_score = 100.0

        deduction = min(total_weight, max_score)
        return round(max_score - deduction, 2)

    def add_custom_rule(
        self,
        code: str,
        title: str,
        description: str,
        level: ComplianceLevel,
        regulation: RegulationType,
        affected_fields: List[str],
        recommendation: str,
        check_func: callable
    ) -> None:
        """添加自定义规则"""
        self.rules[code] = {
            "regulation": regulation,
            "level": level,
            "check": check_func
        }
