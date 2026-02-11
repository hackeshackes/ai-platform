"""
超参数优化 - Phase 3
"""
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from uuid import uuid4
import asyncio
import random
import math

class HPOMethod(Enum):
    """HPO方法"""
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    BAYESIAN = "bayesian"
    HYPERBAND = "hyperband"

@dataclass
class HPOParam:
    """超参数"""
    name: str
    type: str  # categorical, continuous, integer
    values: List[Any]  # 离散值 或 (min, max)
    log_scale: bool = False

@dataclass
class HPOTrial:
    """优化试验"""
    trial_id: str
    params: Dict[str, Any]
    objective: float
    status: str  # pending, running, completed, pruned
    started_at: datetime
    completed_at: Optional[datetime] = None

@dataclass
class HPOResult:
    """优化结果"""
    optimization_id: str
    best_trial: HPOTrial
    best_params: Dict[str, Any]
    best_objective: float
    trials: List[HPOTrial]
    total_trials: int
    completed_at: datetime

class HyperParameterOptimizer:
    """超参数优化器"""
    
    def __init__(self):
        self.optimizations: Dict[str, Dict] = {}
    
    async def optimize(
        self,
        objective_fn: Callable,
        params: List[HPOParam],
        method: HPOMethod = HPOMethod.BAYESIAN,
        max_trials: int = 100,
        timeout_seconds: int = 3600,
        direction: str = "maximize"  # maximize or minimize
    ) -> HPOResult:
        """
        执行超参数优化
        
        Args:
            objective_fn: 目标函数，接受params返回分数
            params: 超参数定义
            method: 优化方法
            max_trials: 最大试验次数
            timeout_seconds: 超时时间
            direction: 优化方向
        """
        optimization_id = str(uuid4())
        
        result = {
            "optimization_id": optimization_id,
            "method": method,
            "max_trials": max_trials,
            "trials": [],
            "best_trial": None,
            "best_params": {},
            "best_objective": float("-inf") if direction == "maximize" else float("inf"),
            "direction": direction,
            "started_at": datetime.utcnow()
        }
        
        # 选择优化方法
        if method == HPOMethod.GRID_SEARCH:
            trials = await self._grid_search(params, max_trials)
        elif method == HPOMethod.RANDOM_SEARCH:
            trials = await self._random_search(params, max_trials)
        elif method == HPOMethod.BAYESIAN:
            trials = await self._bayesian_search(params, max_trials)
        elif method == HPOMethod.HYPERBAND:
            trials = await self._hyperband_search(params, max_trials)
        else:
            trials = await self._random_search(params, max_trials)
        
        # 执行试验
        for trial_params in trials:
            if (datetime.utcnow() - result["started_at"]).total_seconds() > timeout_seconds:
                break
            
            trial = HPOTrial(
                trial_id=str(uuid4()),
                params=trial_params,
                objective=0,
                status="running",
                started_at=datetime.utcnow()
            )
            
            try:
                # 运行目标函数
                objective = await objective_fn(trial_params)
                trial.objective = objective
                trial.status = "completed"
                trial.completed_at = datetime.utcnow()
                
                # 更新最佳结果
                if direction == "maximize":
                    if objective > result["best_objective"]:
                        result["best_objective"] = objective
                        result["best_trial"] = trial
                        result["best_params"] = trial_params
                else:
                    if objective < result["best_objective"]:
                        result["best_objective"] = objective
                        result["best_trial"] = trial
                        result["best_params"] = trial_params
                        
            except Exception as e:
                trial.status = "failed"
                trial.objective = float("-inf") if direction == "maximize" else float("inf")
            
            result["trials"].append(trial)
        
        # 清理未完成的试验
        for trial in result["trials"]:
            if trial.status == "running":
                trial.status = "pruned"
        
        return HPOResult(
            optimization_id=optimization_id,
            best_trial=result["best_trial"],
            best_params=result["best_params"],
            best_objective=result["best_objective"],
            trials=result["trials"],
            total_trials=len(result["trials"]),
            completed_at=datetime.utcnow()
        )
    
    async def _grid_search(
        self,
        params: List[HPOParam],
        max_trials: int
    ) -> List[Dict[str, Any]]:
        """网格搜索"""
        # 生成所有组合
        all_combinations = self._generate_combinations(params)
        
        # 限制数量
        random.shuffle(all_combinations)
        return all_combinations[:max_trials]
    
    async def _random_search(
        self,
        params: List[HPOParam],
        max_trials: int
    ) -> List[Dict[str, Any]]:
        """随机搜索"""
        trials = []
        for _ in range(max_trials):
            trial_params = {}
            for param in params:
                trial_params[param.name] = self._sample_param(param)
            trials.append(trial_params)
        return trials
    
    async def _bayesian_search(
        self,
        params: List[HPOParam],
        max_trials: int
    ) -> List[Dict[str, Any]]:
        """贝叶斯优化 (简化版)"""
        # 简化: 使用随机搜索 +  exploitation
        trials = []
        
        # 首先随机探索
        explore_trials = min(max_trials // 3, 20)
        for _ in range(explore_trials):
            trial_params = {}
            for param in params:
                trial_params[param.name] = self._sample_param(param)
            trials.append(trial_params)
        
        # 然后在最佳区域精细搜索
        best_trial = trials[0] if trials else {}
        for _ in range(max_trials - explore_trials):
            trial_params = {}
            for param in params:
                # 简化: 继续随机搜索
                trial_params[param.name] = self._sample_param(param)
            trials.append(trial_params)
        
        return trials[:max_trials]
    
    async def _hyperband_search(
        self,
        params: List[HPOParam],
        max_trials: int
    ) -> List[Dict[str, Any]]:
        """Hyperband (简化版)"""
        # 简化: 使用随机搜索
        return await self._random_search(params, max_trials)
    
    def _generate_combinations(
        self,
        params: List[HPOParam]
    ) -> List[Dict[str, Any]]:
        """生成所有参数组合"""
        from itertools import product
        
        value_lists = []
        for param in params:
            if param.type == "categorical":
                value_lists.append(param.values)
            elif param.type == "continuous":
                # 离散化为5个点
                min_val, max_val = param.values
                step = (max_val - min_val) / 4
                value_lists.append([min_val + i * step for i in range(5)])
            elif param.type == "integer":
                min_val, max_val = param.values
                value_lists.append(list(range(min_val, max_val + 1)))
        
        combinations = []
        for combo in product(*value_lists):
            trial = {}
            for i, param in enumerate(params):
                trial[param.name] = combo[i]
            combinations.append(trial)
        
        return combinations
    
    def _sample_param(self, param: HPOParam) -> Any:
        """采样参数值"""
        if param.type == "categorical":
            return random.choice(param.values)
        elif param.type == "continuous":
            min_val, max_val = param.values
            if param.log_scale:
                return math.exp(random.uniform(math.log(min_val), math.log(max_val)))
            else:
                return random.uniform(min_val, max_val)
        elif param.type == "integer":
            min_val, max_val = param.values
            return random.randint(min_val, max_val)
        return param.values[0]
    
    def get_optimization_status(self, optimization_id: str) -> Dict:
        """获取优化状态"""
        opt = self.optimizations.get(optimization_id)
        if not opt:
            return {}
        
        return {
            "optimization_id": optimization_id,
            "method": opt["method"],
            "total_trials": len(opt["trials"]),
            "completed_trials": sum(1 for t in opt["trials"] if t.status == "completed"),
            "best_objective": opt["best_objective"],
            "elapsed_seconds": (datetime.utcnow() - opt["started_at"]).total_seconds()
        }

# 优化器实例
hpo_optimizer = HyperParameterOptimizer()
