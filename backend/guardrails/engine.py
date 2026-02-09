"""
LLM Guardrails模块 v2.4
"""
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import re

class GuardrailType(str, Enum):
    """护栏类型"""
    INPUT = "input"
    OUTPUT = "output"
    BEHAVIOR = "behavior"

class Severity(str, Enum):
    """严重程度"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class Action(str, Enum):
    """触发动作"""
    ALLOW = "allow"
    BLOCK = "block"
    WARN = "warn"
    REDACT = "redact"
    ESCALATE = "escalate"

@dataclass
class GuardrailRule:
    """护栏规则"""
    rule_id: str
    name: str
    description: str
    guardrail_type: GuardrailType
    severity: Severity
    pattern: Optional[str] = None  # Regex pattern
    keywords: List[str] = field(default_factory=list)
    action: Action = Action.BLOCK
    enabled: bool = True
    config: Dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class GuardrailResult:
    """护栏检查结果"""
    result_id: str
    rule_id: str
    rule_name: str
    guardrail_type: GuardrailType
    passed: bool
    action: Action
    message: str
    details: Dict = field(default_factory=dict)
    checked_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class GuardrailConfig:
    """护栏配置"""
    config_id: str
    name: str
    description: str
    input_rules: List[str] = field(default_factory=list)  # rule_ids
    output_rules: List[str] = field(default_factory=list)
    behavior_rules: List[str] = field(default_factory=list)
    default_action: Action = Action.BLOCK
    enabled: bool = True
    created_by: str
    created_at: datetime = field(default_factory=datetime.utcnow)

class GuardrailsEngine:
    """Guardrails引擎"""
    
    def __init__(self):
        self.rules: Dict[str, GuardrailRule] = {}
        self.configs: Dict[str, GuardrailConfig] = {}
        self.violations: List[GuardrailResult] = []
        
        # 内置护栏规则
        self._init_builtin_rules()
        
        # 自定义检查器
        self.custom_checkers: Dict[str, Callable] = {}
    
    def _init_builtin_rules(self):
        """初始化内置护栏规则"""
        
        # 1. 敏感词过滤 (输入)
        self.create_rule(
            name="敏感词过滤",
            description="检测并过滤敏感词汇",
            guardrail_type=GuardrailType.INPUT,
            severity=Severity.HIGH,
            keywords=[
                "password", "secret", "credential", "api_key", "token",
                "ssn", "credit card", "social security"
            ],
            action=Action.BLOCK,
            config={"replacement": "***"}
        )
        
        # 2. PII检测 (输入)
        self.create_rule(
            name="PII检测",
            description="检测个人身份信息",
            guardrail_type=GuardrailType.INPUT,
            severity=Severity.CRITICAL,
            pattern=r"\b\d{3}[-.]?\d{2}[-.]?\d{4}\b",  # SSN pattern
            keywords=["email", "phone", "address", "birth date"],
            action=Action.REDACT,
            config={"pii_type": "detected"}
        )
        
        # 3. 毒性检测 (输入/输出)
        self.create_rule(
            name="毒性检测",
            description="检测有毒/冒犯性内容",
            guardrail_type=GuardrailType.OUTPUT,
            severity=Severity.HIGH,
            keywords=["hate", "harassment", "violence", "self-harm"],
            action=Action.BLOCK,
            config={"threshold": 0.8}
        )
        
        # 4. 格式验证 (输出)
        self.create_rule(
            name="JSON格式验证",
            description="验证输出是否为有效JSON",
            guardrail_type=GuardrailType.OUTPUT,
            severity=Severity.MEDIUM,
            pattern=r"^\{.*\}$|^\[.*\]$",
            action=Action.WARN,
            config={"required_format": "json"}
        )
        
        # 5. 长度限制 (输入/输出)
        self.create_rule(
            name="响应长度限制",
            description="限制响应长度",
            guardrail_type=GuardrailType.OUTPUT,
            severity=Severity.LOW,
            action=Action.BLOCK,
            config={"max_length": 4000}
        )
        
        # 6. 主题限制 (行为)
        self.create_rule(
            name="政治话题限制",
            description="限制讨论敏感政治话题",
            guardrail_type=GuardrailType.BEHAVIOR,
            severity=Severity.MEDIUM,
            keywords=["politics", "election", "government policy"],
            action=Action.WARN,
            config={"allowed_discussion_level": "general"}
        )
        
        # 7. 越狱检测 (行为)
        self.create_rule(
            name="越狱检测",
            description="检测提示词注入/越狱尝试",
            guardrail_type=GuardrailType.INPUT,
            severity=Severity.CRITICAL,
            keywords=[
                "ignore previous instructions",
                "system prompt",
                "developer mode",
                "jailbreak",
                " DAN (Do Anything Now)"
            ],
            action=Action.BLOCK,
            config={"strict_mode": True}
        )
        
        # 8. 代码安全 (输出)
        self.create_rule(
            name="代码安全检查",
            description="检测潜在不安全的代码",
            guardrail_type=GuardrailType.OUTPUT,
            severity=Severity.HIGH,
            keywords=[
                "eval(",
                "exec(",
                "os.system",
                "subprocess",
                "shell=True"
            ],
            action=Action.WARN,
            config={"security_level": "strict"}
        )
    
    def create_rule(
        self,
        name: str,
        description: str,
        guardrail_type: GuardrailType,
        severity: Severity,
        pattern: Optional[str] = None,
        keywords: Optional[List[str]] = None,
        action: Action = Action.BLOCK,
        config: Optional[Dict] = None
    ) -> GuardrailRule:
        """创建护栏规则"""
        import uuid
        rule = GuardrailRule(
            rule_id=str(uuid.uuid4()),
            name=name,
            description=description,
            guardrail_type=guardrail_type,
            severity=severity,
            pattern=pattern,
            keywords=keywords or [],
            action=action,
            config=config or {}
        )
        
        self.rules[rule.rule_id] = rule
        return rule
    
    def get_rule(self, rule_id: str) -> Optional[GuardrailRule]:
        """获取规则"""
        return self.rules.get(rule_id)
    
    def list_rules(
        self,
        guardrail_type: Optional[GuardrailType] = None,
        enabled: Optional[bool] = None
    ) -> List[GuardrailRule]:
        """列出规则"""
        rules = list(self.rules.values())
        
        if guardrail_type:
            rules = [r for r in rules if r.guardrail_type == guardrail_type]
        if enabled is not None:
            rules = [r for r in rules if r.enabled == enabled]
        
        return rules
    
    def update_rule(
        self,
        rule_id: str,
        enabled: Optional[bool] = None,
        action: Optional[Action] = None,
        config: Optional[Dict] = None
    ) -> bool:
        """更新规则"""
        rule = self.rules.get(rule_id)
        if not rule:
            return False
        
        if enabled is not None:
            rule.enabled = enabled
        if action:
            rule.action = action
        if config:
            rule.config.update(config)
        
        return True
    
    def delete_rule(self, rule_id: str) -> bool:
        """删除规则"""
        if rule_id in self.rules:
            del self.rules[rule_id]
            return True
        return False
    
    def check(
        self,
        text: str,
        guardrail_type: GuardrailType,
        context: Optional[Dict] = None
    ) -> List[GuardrailResult]:
        """检查文本"""
        results = []
        
        for rule in self.rules.values():
            if rule.guardrail_type != guardrail_type or not rule.enabled:
                continue
            
            # 关键词检查
            if rule.keywords:
                for keyword in rule.keywords:
                    if keyword.lower() in text.lower():
                        results.append(self._create_violation(rule, text, f"Keyword detected: {keyword}"))
                        continue
            
            # 正则检查
            if rule.pattern:
                if re.search(rule.pattern, text, re.IGNORECASE):
                    results.append(self._create_violation(rule, text, f"Pattern matched: {rule.pattern}"))
        
        return results
    
    def _create_violation(
        self,
        rule: GuardrailRule,
        text: str,
        message: str
    ) -> GuardrailResult:
        """创建违规记录"""
        import uuid
        
        # 根据action决定是否通过
        passed = rule.action in [Action.ALLOW, Action.WARN]
        
        result = GuardrailResult(
            result_id=str(uuid.uuid4()),
            rule_id=rule.rule_id,
            rule_name=rule.name,
            guardrail_type=rule.guardrail_type,
            passed=passed,
            action=rule.action,
            message=message,
            details={
                "severity": rule.severity.value,
                "rule_config": rule.config
            }
        )
        
        if not passed:
            self.violations.append(result)
        
        return result
    
    def check_input(self, text: str, context: Optional[Dict] = None) -> List[GuardrailResult]:
        """检查输入"""
        return self.check(text, GuardrailType.INPUT, context)
    
    def check_output(self, text: str, context: Optional[Dict] = None) -> List[GuardrailResult]:
        """检查输出"""
        return self.check(text, GuardrailType.OUTPUT, context)
    
    def check_behavior(self, text: str, context: Optional[Dict] = None) -> List[GuardrailResult]:
        """检查行为"""
        return self.check(text, GuardrailType.BEHAVIOR, context)
    
    def sanitize(
        self,
        text: str,
        guardrail_type: GuardrailType
    ) -> str:
        """净化文本"""
        violations = self.check(text, guardrail_type)
        
        for violation in violations:
            if violation.action == Action.REDACT:
                for rule in self.rules.values():
                    if rule.rule_id == violation.rule_id:
                        for keyword in rule.keywords:
                            text = text.replace(keyword, "***")
                        if rule.pattern:
                            text = re.sub(rule.pattern, "***", text)
        
        return text
    
    def create_config(
        self,
        name: str,
        description: str,
        input_rules: Optional[List[str]] = None,
        output_rules: Optional[List[str]] = None,
        behavior_rules: Optional[List[str]] = None,
        default_action: Action = Action.BLOCK,
        created_by: str = "user"
    ) -> GuardrailConfig:
        """创建护栏配置"""
        import uuid
        config = GuardrailConfig(
            config_id=str(uuid.uuid4()),
            name=name,
            description=description,
            input_rules=input_rules or [],
            output_rules=output_rules or [],
            behavior_rules=behavior_rules or [],
            default_action=default_action,
            created_by=created_by
        )
        
        self.configs[config.config_id] = config
        return config
    
    def get_config(self, config_id: str) -> Optional[GuardrailConfig]:
        """获取配置"""
        return self.configs.get(config_id)
    
    def list_configs(self) -> List[GuardrailConfig]:
        """列出配置"""
        return list(self.configs.values())
    
    def apply_config(
        self,
        config_id: str,
        text: str,
        guardrail_type: GuardrailType
    ) -> List[GuardrailResult]:
        """应用配置"""
        config = self.configs.get(config_id)
        if not config:
            raise ValueError(f"Config {config_id} not found")
        
        rule_ids = []
        if guardrail_type == GuardrailType.INPUT:
            rule_ids = config.input_rules
        elif guardrail_type == GuardrailType.OUTPUT:
            rule_ids = config.output_rules
        else:
            rule_ids = config.behavior_rules
        
        results = []
        for rule_id in rule_ids:
            rule = self.rules.get(rule_id)
            if rule and rule.enabled:
                # 简化检查
                if rule.keywords:
                    for keyword in rule.keywords:
                        if keyword.lower() in text.lower():
                            results.append(self._create_violation(rule, text, f"Keyword: {keyword}"))
                            break
        
        return results
    
    def get_violations(
        self,
        limit: int = 100
    ) -> List[GuardrailResult]:
        """获取违规记录"""
        return self.violations[-limit:]
    
    def get_stats(self) -> Dict:
        """获取统计"""
        by_type = {}
        by_severity = {}
        for v in self.violations:
            t = v.guardrail_type.value
            by_type[t] = by_type.get(t, 0) + 1
            s = v.details.get("severity", "unknown")
            by_severity[s] = by_severity.get(s, 0) + 1
        
        return {
            "total_violations": len(self.violations),
            "by_type": by_type,
            "by_severity": by_severity,
            "total_rules": len(self.rules),
            "enabled_rules": len([r for r in self.rules.values() if r.enabled]),
            "total_configs": len(self.configs)
        }

# GuardrailsEngine实例
guardrails_engine = GuardrailsEngine()
