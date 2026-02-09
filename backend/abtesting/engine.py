"""
A/B Testing 模块 v2.4
"""
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from uuid import uuid4
import json

class ExperimentStatus(str, Enum):
    """实验状态"""
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    ARCHIVED = "archived"

class TrafficAllocation(str, Enum):
    """流量分配策略"""
    EVEN = "even"
    WEIGHTED = "weighted"
    PROGRESSIVE = "progressive"

@dataclass
class Experiment:
    """A/B实验"""
    experiment_id: str
    name: str
    description: str
    status: ExperimentStatus
    variants: List[Dict] = field(default_factory=list)
    traffic_allocation: TrafficAllocation = TrafficAllocation.EVEN
    target_metrics: List[str] = field(default_factory=list)
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    created_by: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class ExperimentResult:
    """实验结果"""
    result_id: str
    experiment_id: str
    variant_id: str
    sample_size: int = 0
    conversions: int = 0
    conversion_rate: float = 0.0
    mean_value: float = 0.0
    confidence_interval: tuple = (0.0, 0.0)
    statistical_significance: float = 0.0
    p_value: float = 1.0
    updated_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class VariantAssignment:
    """变体分配"""
    assignment_id: str
    experiment_id: str
    user_id: str
    variant_id: str
    assigned_at: datetime = field(default_factory=datetime.utcnow)

class ABTestingEngine:
    """A/B测试引擎 v2.4"""
    
    def __init__(self):
        self.experiments: Dict[str, Experiment] = {}
        self.results: Dict[str, List[ExperimentResult]] = {}
        self.assignments: List[VariantAssignment] = []
        self.user_assignments: Dict[str, Dict[str, str]] = {}
    
    def create_experiment(
        self,
        name: str,
        description: str,
        variants: List[Dict],
        target_metrics: Optional[List[str]] = None,
        created_by: str = "user"
    ) -> Experiment:
        """创建实验"""
        experiment = Experiment(
            experiment_id=str(uuid4()),
            name=name,
            description=description,
            status=ExperimentStatus.DRAFT,
            variants=variants,
            target_metrics=target_metrics or ["conversion_rate"],
            created_by=created_by
        )
        
        self.experiments[experiment.experiment_id] = experiment
        self.results[experiment.experiment_id] = []
        
        return experiment
    
    def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """获取实验"""
        return self.experiments.get(experiment_id)
    
    def list_experiments(
        self,
        status: Optional[ExperimentStatus] = None
    ) -> List[Experiment]:
        """列出实验"""
        experiments = list(self.experiments.values())
        if status:
            experiments = [e for e in experiments if e.status == status]
        return experiments
    
    def start_experiment(self, experiment_id: str) -> bool:
        """开始实验"""
        experiment = self.experiments.get(experiment_id)
        if not experiment:
            return False
        
        experiment.status = ExperimentStatus.RUNNING
        experiment.start_date = datetime.utcnow()
        return True
    
    def complete_experiment(self, experiment_id: str) -> bool:
        """完成实验"""
        experiment = self.experiments.get(experiment_id)
        if not experiment:
            return False
        
        experiment.status = ExperimentStatus.COMPLETED
        experiment.end_date = datetime.utcnow()
        return True
    
    def assign_variant(
        self,
        experiment_id: str,
        user_id: str
    ) -> Optional[str]:
        """为用户分配变体"""
        experiment = self.experiments.get(experiment_id)
        if not experiment or experiment.status != ExperimentStatus.RUNNING:
            return None
        
        if user_id in self.user_assignments:
            if experiment_id in self.user_assignments[user_id]:
                return self.user_assignments[user_id][experiment_id]
        
        import hashlib
        hash_value = int(hashlib.md5(f"{experiment_id}:{user_id}".encode()).hexdigest(), 16)
        variant_index = hash_value % len(experiment.variants)
        variant_id = experiment.variants[variant_index]["variant_id"]
        
        if user_id not in self.user_assignments:
            self.user_assignments[user_id] = {}
        self.user_assignments[user_id][experiment_id] = variant_id
        
        return variant_id
    
    def track_conversion(
        self,
        experiment_id: str,
        user_id: str,
        value: float = 1.0
    ) -> bool:
        """跟踪转化"""
        variant_id = self.get_variant_assignment(experiment_id, user_id)
        if not variant_id:
            return False
        
        results = self.results.get(experiment_id, [])
        for result in results:
            if result.variant_id == variant_id:
                result.sample_size += 1
                result.conversions += 1
                result.conversion_rate = result.conversions / result.sample_size
                return True
        
        result = ExperimentResult(
            result_id=str(uuid4()),
            experiment_id=experiment_id,
            variant_id=variant_id,
            sample_size=1,
            conversions=1,
            conversion_rate=1.0
        )
        results.append(result)
        self.results[experiment_id] = results
        
        return True
    
    def get_variant_assignment(
        self,
        experiment_id: str,
        user_id: str
    ) -> Optional[str]:
        """获取用户变体分配"""
        if user_id in self.user_assignments:
            return self.user_assignments[user_id].get(experiment_id)
        return None
    
    def get_results(self, experiment_id: str) -> List[ExperimentResult]:
        """获取实验结果"""
        return self.results.get(experiment_id, [])
    
    def get_leaderboard(self, experiment_id: str) -> List[Dict]:
        """获取排行榜"""
        results = self.results.get(experiment_id, [])
        experiment = self.experiments.get(experiment_id)
        
        leaderboard = []
        for result in results:
            variant_name = "Unknown"
            if experiment:
                for v in experiment.variants:
                    if v["variant_id"] == result.variant_id:
                        variant_name = v.get("name", result.variant_id)
                        break
            
            leaderboard.append({
                "variant_id": result.variant_id,
                "variant_name": variant_name,
                "sample_size": result.sample_size,
                "conversions": result.conversions,
                "conversion_rate": result.conversion_rate
            })
        
        leaderboard.sort(key=lambda x: x["conversion_rate"], reverse=True)
        return leaderboard
    
    def get_summary(self) -> Dict:
        """获取统计"""
        return {
            "total_experiments": len(self.experiments),
            "running_experiments": len([e for e in self.experiments.values() if e.status == ExperimentStatus.RUNNING])
        }

# ABTestingEngine实例
ab_testing_engine = ABTestingEngine()
