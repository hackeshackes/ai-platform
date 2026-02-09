"""
质量门禁 - Phase 2
"""
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

class GateStatus(Enum):
    PENDING = "pending"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"

class GateType(Enum):
    COVERAGE = "coverage"
    LINT = "lint"
    SECURITY = "security"
    PERFORMANCE = "performance"
    CUSTOM = "custom"

@dataclass
class QualityGate:
    gate_id: str
    name: str
    type: GateType
    threshold: float
    operator: str
    action_on_fail: str
    enabled: bool = True

@dataclass
class GateResult:
    gate_id: str
    name: str
    status: GateStatus
    value: Optional[float] = None
    threshold: Optional[float] = None
    message: Optional[str] = None
    checked_at: Optional[datetime] = None

class QualityGates:
    """质量门禁检查器"""
    
    def __init__(self):
        self.gates: Dict[str, QualityGate] = {}
        self._register_default_gates()
    
    def _register_default_gates(self):
        default_gates = [
            QualityGate("coverage", "测试覆盖率", GateType.COVERAGE, 80.0, ">=", "fail"),
            QualityGate("lint", "代码规范", GateType.LINT, 0, "==", "fail"),
            QualityGate("security-high", "安全扫描(高危)", GateType.SECURITY, 0, "==", "fail"),
            QualityGate("security-critical", "安全扫描(严重)", GateType.SECURITY, 0, "==", "fail"),
            QualityGate("performance", "性能基线", GateType.PERFORMANCE, 1000, "<=", "warn")
        ]
        for gate in default_gates:
            self.gates[gate.gate_id] = gate
    
    def evaluate_condition(self, value: float, threshold: float, operator: str) -> bool:
        if operator == ">": return value > threshold
        elif operator == ">=": return value >= threshold
        elif operator == "<": return value < threshold
        elif operator == "<=": return value <= threshold
        elif operator == "==": return value == threshold
        return False
    
    async def check(self, gate_id: str, value: Optional[float] = None) -> GateResult:
        gate = self.gates.get(gate_id)
        if not gate:
            raise ValueError(f"Gate {gate_id} not found")
        
        if not gate.enabled:
            return GateResult(gate_id, gate.name, GateStatus.SKIPPED)
        
        # 模拟检查
        if value is None:
            value = 85.0 if gate.gate_id == "coverage" else 0
        
        passed = self.evaluate_condition(value, gate.threshold, gate.operator)
        
        return GateResult(
            gate_id=gate_id,
            name=gate.name,
            status=GateStatus.PASSED if passed else GateStatus.FAILED,
            value=value,
            threshold=gate.threshold,
            message=f"Value {value} {'>=' if passed else '<'} {gate.threshold}"
        )
    
    async def check_all(self) -> List[GateResult]:
        results = []
        for gate_id in self.gates:
            result = await self.check(gate_id)
            results.append(result)
        return results
    
    def get_summary(self, results: List[GateResult]) -> Dict:
        passed = sum(1 for r in results if r.status == GateStatus.PASSED)
        failed = sum(1 for r in results if r.status == GateStatus.FAILED)
        return {
            "total": len(results),
            "passed": passed,
            "failed": failed,
            "all_passed": failed == 0
        }

quality_gates = QualityGates()
