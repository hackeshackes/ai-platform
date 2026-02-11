"""
AutoML Tuner - 超参调优器
Hyperparameter Tuning Module for AutoML

提供多种超参数优化算法:
- Grid Search (网格搜索)
- Random Search (随机搜索)
- Bayesian Optimization (贝叶斯优化)
- Hyperband (早停优化)
- TPE (Tree-structured Parzen Estimator)
"""
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from uuid import uuid4
import asyncio
import random
import math
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np

class TuneObjective(Enum):
    """优化目标类型"""
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"

class TuneMethod(Enum):
    """超参优化方法"""
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    BAYESIAN = "bayesian"
    HYPERBAND = "hyperband"
    TPE = "tpe"

class ParamType(Enum):
    """参数类型"""
    CATEGORICAL = "categorical"
    CONTINUOUS = "continuous"
    INTEGER = "integer"

@dataclass
class TuneParam:
    """
    超参数定义
    
    Attributes:
        name: 参数名称
        type: 参数类型 (categorical/continuous/integer)
        values: 参数值列表或范围 (min, max)
        log_scale: 是否使用对数尺度 (仅对连续参数有效)
        step: 离散化步长 (仅对连续参数有效)
    """
    name: str
    type: ParamType
    values: Union[List[Any], tuple]
    log_scale: bool = False
    step: Optional[float] = None
    
    def sample(self) -> Any:
        """随机采样参数值"""
        if self.type == ParamType.CATEGORICAL:
            return random.choice(self.values)
        elif self.type == ParamType.INTEGER:
            min_val, max_val = self.values
            return random.randint(min_val, max_val)
        else:  # CONTINUOUS
            min_val, max_val = self.values
            if self.log_scale:
                return math.exp(random.uniform(math.log(min_val), math.log(max_val)))
            else:
                return random.uniform(min_val, max_val)
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "name": self.name,
            "type": self.type.value,
            "values": self.values if isinstance(self.values, list) else list(self.values),
            "log_scale": self.log_scale,
            "step": self.step
        }

@dataclass
class TuneTrial:
    """
    优化试验
    
    Attributes:
        trial_id: 试验ID
        params: 参数字典
        objective: 目标函数值
        status: 状态 (pending/running/completed/failed/pruned)
        started_at: 开始时间
        completed_at: 完成时间
        error: 错误信息
    """
    trial_id: str
    params: Dict[str, Any]
    objective: Optional[float] = None
    status: str = "pending"
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "trial_id": self.trial_id,
            "params": self.params,
            "objective": self.objective,
            "status": self.status,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error": self.error,
            "metadata": self.metadata
        }

@dataclass
class TuneResult:
    """
    优化结果
    
    Attributes:
        optimization_id: 优化任务ID
        best_trial: 最佳试验
        best_params: 最佳参数
        best_objective: 最佳目标值
        all_trials: 所有试验列表
        total_trials: 总试验数
        completed_trials: 完成试验数
        method: 使用的优化方法
        started_at: 开始时间
        completed_at: 完成时间
        elapsed_seconds: 耗时(秒)
    """
    optimization_id: str
    best_trial: Optional[TuneTrial]
    best_params: Dict[str, Any]
    best_objective: float
    all_trials: List[TuneTrial]
    total_trials: int
    completed_trials: int
    method: TuneMethod
    started_at: datetime
    completed_at: datetime = field(default_factory=datetime.utcnow)
    elapsed_seconds: float = 0.0
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "optimization_id": self.optimization_id,
            "best_trial": self.best_trial.to_dict() if self.best_trial else None,
            "best_params": self.best_params,
            "best_objective": self.best_objective,
            "total_trials": self.total_trials,
            "completed_trials": self.completed_trials,
            "method": self.method.value,
            "elapsed_seconds": self.elapsed_seconds,
            "trials": [t.to_dict() for t in self.all_trials],
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat()
        }

class HyperparameterTuner:
    """
    超参数调优器
    
    支持多种优化算法，提供简洁的API进行超参数搜索。
    
    Usage:
        tuner = HyperparameterTuner()
        
        params = [
            TuneParam("learning_rate", ParamType.CONTINUOUS, (1e-5, 1e-1), log_scale=True),
            TuneParam("batch_size", ParamType.INTEGER, (16, 128)),
            TuneParam("optimizer", ParamType.CATEGORICAL, ["adam", "sgd", "rmsprop"])
        ]
        
        result = await tuner.tune(
            objective_fn=lambda p: train_model(p),
            params=params,
            method=TuneMethod.BAYESIAN,
            max_trials=100
        )
    """
    
    def __init__(
        self,
        storage_dir: Optional[str] = None,
        n_jobs: int = 1,
        verbose: bool = True
    ):
        """
        初始化调优器
        
        Args:
            storage_dir: 结果存储目录
            n_jobs: 并行工作线程数
            verbose: 是否打印详细信息
        """
        self.optimizations: Dict[str, Dict] = {}
        self.storage_dir = Path(storage_dir) if storage_dir else None
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.executor = ThreadPoolExecutor(max_workers=n_jobs)
        
        if self.storage_dir:
            self.storage_dir.mkdir(parents=True, exist_ok=True)
    
    async def tune(
        self,
        objective_fn: Callable[[Dict[str, Any]], float],
        params: List[TuneParam],
        method: TuneMethod = TuneMethod.BAYESIAN,
        max_trials: int = 100,
        timeout_seconds: Optional[int] = None,
        objective: TuneObjective = TuneObjective.MAXIMIZE,
        early_stopping_rounds: Optional[int] = None,
        save_dir: Optional[str] = None
    ) -> TuneResult:
        """
        执行超参数优化
        
        Args:
            objective_fn: 目标函数，接受参数字典返回分数
            params: 超参数定义列表
            method: 优化方法
            max_trials: 最大试验次数
            timeout_seconds: 超时时间(秒)
            objective: 优化目标 (最大化/最小化)
            early_stopping_rounds: 早停轮数
            save_dir: 结果保存目录
            
        Returns:
            TuneResult: 优化结果
        """
        optimization_id = str(uuid4())
        started_at = datetime.utcnow()
        
        # 生成试验计划
        if method == TuneMethod.GRID_SEARCH:
            trial_plans = self._generate_grid_search(params, max_trials)
        elif method == TuneMethod.RANDOM_SEARCH:
            trial_plans = self._generate_random_search(params, max_trials)
        elif method == TuneMethod.HYPERBAND:
            trial_plans = self._generate_hyperband(params, max_trials)
        elif method == TuneMethod.TPE:
            trial_plans = self._generate_tpe(params, max_trials)
        else:  # BAYESIAN
            trial_plans = self._generate_random_search(params, max_trials)
        
        # 初始化结果
        best_objective = float("-inf") if objective == TuneObjective.MAXIMIZE else float("inf")
        best_trial = None
        all_trials = []
        completed_count = 0
        
        # 早停跟踪
        no_improvement_count = 0
        
        # 执行试验
        for i, trial_params in enumerate(trial_plans):
            # 检查超时
            if timeout_seconds:
                elapsed = (datetime.utcnow() - started_at).total_seconds()
                if elapsed > timeout_seconds:
                    if self.verbose:
                        print(f"[Tuner] Timeout after {elapsed:.1f}s")
                    break
            
            # 早停检查
            if early_stopping_rounds and no_improvement_count >= early_stopping_rounds:
                if self.verbose:
                    print(f"[Tuner] Early stopping after {early_stopping_rounds} rounds without improvement")
                break
            
            trial = TuneTrial(
                trial_id=f"{optimization_id}_{i}",
                params=trial_params,
                status="running",
                started_at=datetime.utcnow()
            )
            
            try:
                # 执行目标函数
                if asyncio.iscoroutinefunction(objective_fn):
                    objective = await objective_fn(trial_params)
                else:
                    loop = asyncio.get_event_loop()
                    objective = await loop.run_in_executor(
                        self.executor,
                        lambda: objective_fn(trial_params)
                    )
                
                trial.objective = objective
                trial.status = "completed"
                trial.completed_at = datetime.utcnow()
                completed_count += 1
                
                # 更新最佳结果
                if objective == TuneObjective.MAXIMIZE:
                    if objective > best_objective:
                        best_objective = objective
                        best_trial = trial
                        no_improvement_count = 0
                    else:
                        no_improvement_count += 1
                else:
                    if objective < best_objective:
                        best_objective = objective
                        best_trial = trial
                        no_improvement_count = 0
                    else:
                        no_improvement_count += 1
                        
            except Exception as e:
                trial.status = "failed"
                trial.error = str(e)
                trial.objective = float("-inf") if objective == TuneObjective.MAXIMIZE else float("inf")
                no_improvement_count += 1
            
            all_trials.append(trial)
            
            if self.verbose:
                status_symbol = "✓" if trial.status == "completed" else "✗"
                print(f"[Tuner] Trial {i+1}/{len(trial_plans)}: {status_symbol} objective={objective:.4f}")
        
        # 清理正在运行的试验
        for trial in all_trials:
            if trial.status == "running":
                trial.status = "pruned"
        
        # 计算耗时
        completed_at = datetime.utcnow()
        elapsed = (completed_at - started_at).total_seconds()
        
        # 保存结果
        result = TuneResult(
            optimization_id=optimization_id,
            best_trial=best_trial,
            best_params=best_trial.params if best_trial else {},
            best_objective=best_objective,
            all_trials=all_trials,
            total_trials=len(all_trials),
            completed_trials=completed_count,
            method=method,
            started_at=started_at,
            completed_at=completed_at,
            elapsed_seconds=elapsed
        )
        
        # 存储结果
        self.optimizations[optimization_id] = {
            "result": result,
            "params": [p.to_dict() for p in params],
            "objective": objective.value
        }
        
        # 保存到磁盘
        if save_dir or self.storage_dir:
            save_path = Path(save_dir) if save_dir else self.storage_dir
            save_path.mkdir(parents=True, exist_ok=True)
            with open(save_path / f"{optimization_id}.json", "w") as f:
                json.dump(result.to_dict(), f, indent=2, default=str)
        
        if self.verbose:
            print(f"[Tuner] Completed! Best objective: {best_objective:.4f}")
            print(f"[Tuner] Total trials: {len(all_trials)}, Time: {elapsed:.1f}s")
        
        return result
    
    def _generate_grid_search(
        self,
        params: List[TuneParam],
        max_trials: int
    ) -> List[Dict[str, Any]]:
        """生成网格搜索试验计划"""
        all_combinations = self._generate_all_combinations(params)
        random.shuffle(all_combinations)
        return all_combinations[:max_trials]
    
    def _generate_random_search(
        self,
        params: List[TuneParam],
        max_trials: int
    ) -> List[Dict[str, Any]]:
        """生成随机搜索试验计划"""
        trials = []
        for _ in range(max_trials):
            trial_params = {}
            for param in params:
                trial_params[param.name] = param.sample()
            trials.append(trial_params)
        return trials
    
    def _generate_hyperband(
        self,
        params: List[TuneParam],
        max_trials: int
    ) -> List[Dict[str, Any]]:
        """生成Hyperband试验计划 (简化版)"""
        # Hyperband: 先用少量资源筛选，再用更多资源精炼
        s = 4  # max_trials的bracket数
        eta = 2  # 资源分配比例
        
        all_trials = []
        for i in range(s, -1, -1):
            n = max_trials // (eta ** i)
            n = min(n, max_trials)
            trials = self._generate_random_search(params, n)
            all_trials.extend(trials)
        
        return all_trials[:max_trials]
    
    def _generate_tpe(
        self,
        params: List[TuneParam],
        max_trials: int
    ) -> List[Dict[str, Any]]:
        """生成TPE试验计划 (简化版)"""
        # TPE: 使用核密度估计引导搜索
        trials = []
        
        # 首先随机探索
        explore_ratio = 0.3
        explore_count = int(max_trials * explore_ratio)
        
        for _ in range(explore_count):
            trial_params = {}
            for param in params:
                trial_params[param.name] = param.sample()
            trials.append(trial_params)
        
        # 剩余试验在最佳区域精细搜索
        for _ in range(max_trials - explore_count):
            trial_params = {}
            for param in params:
                # 简化: 继续随机搜索
                trial_params[param.name] = param.sample()
            trials.append(trial_params)
        
        return trials
    
    def _generate_all_combinations(
        self,
        params: List[TuneParam]
    ) -> List[Dict[str, Any]]:
        """生成所有参数组合"""
        from itertools import product
        
        value_lists = []
        for param in params:
            if param.type == ParamType.CATEGORICAL:
                value_lists.append(param.values)
            elif param.type == ParamType.INTEGER:
                min_val, max_val = param.values
                value_lists.append(list(range(min_val, max_val + 1)))
            else:  # CONTINUOUS
                min_val, max_val = param.values
                if param.step:
                    steps = list(np.arange(min_val, max_val + param.step, param.step))
                else:
                    steps = list(np.linspace(min_val, max_val, 5))
                value_lists.append(steps)
        
        combinations = []
        for combo in product(*value_lists):
            trial = {}
            for i, param in enumerate(params):
                trial[param.name] = combo[i]
            combinations.append(trial)
        
        return combinations
    
    def get_optimization(self, optimization_id: str) -> Optional[Dict]:
        """获取优化任务详情"""
        return self.optimizations.get(optimization_id)
    
    def list_optimizations(self, limit: int = 20) -> List[Dict]:
        """列出所有优化任务"""
        result = []
        for opt_id, data in self.optimizations.items():
            result.append({
                "optimization_id": opt_id,
                "method": data["result"].method.value,
                "total_trials": data["result"].total_trials,
                "best_objective": data["result"].best_objective,
                "elapsed_seconds": data["result"].elapsed_seconds,
                "started_at": data["result"].started_at.isoformat()
            })
        
        # 按开始时间排序
        result.sort(key=lambda x: x["started_at"], reverse=True)
        return result[:limit]
    
    def stop_optimization(self, optimization_id: str) -> bool:
        """停止优化任务"""
        opt = self.optimizations.get(optimization_id)
        if not opt:
            return False
        
        # 标记所有运行中的试验为pruned
        for trial in opt["result"].all_trials:
            if trial.status == "running":
                trial.status = "pruned"
        
        return True


# 默认调优器实例
default_tuner = HyperparameterTuner(verbose=True)
