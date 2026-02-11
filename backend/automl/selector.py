"""
AutoML Selector - 模型自动选择器
Automatic Model Selection Module for AutoML

提供自动模型选择功能:
- 基于数据的模型推荐
- 模型性能比较
- 集成学习支持
- 快速模型原型
"""
from typing import Dict, List, Optional, Any, Type, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from uuid import uuid4
import asyncio
import json
import warnings
from pathlib import Path

# 忽略sklearn弃用警告
warnings.filterwarnings('ignore', category=FutureWarning)

class TaskType(Enum):
    """任务类型"""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    RECOMMENDATION = "recommendation"

class ModelCategory(Enum):
    """模型类别"""
    LINEAR = "linear"
    TREE_BASED = "tree_based"
    ENSEMBLE = "ensemble"
    NEURAL_NETWORK = "neural_network"
    BOOSTING = "boosting"

@dataclass
class ModelCandidate:
    """
    模型候选
    
    Attributes:
        model_id: 模型ID
        name: 模型名称
        category: 模型类别
        task_type: 适合的任务类型
        class_path: 模型类路径
        hyperparameters: 默认超参
        expected_accuracy: 预期准确率范围
        training_time: 预期训练时间(秒)
        model_size: 模型大小(MB)
        description: 模型描述
    """
    model_id: str
    name: str
    category: ModelCategory
    task_type: List[TaskType]
    class_path: str
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    expected_accuracy: tuple = (0.0, 1.0)
    training_time: float = 60.0
    model_size: float = 1.0
    description: str = ""
    dependencies: List[str] = field(default_factory=list)

@dataclass
class ModelScore:
    """
    模型评分
    
    Attributes:
        model_id: 模型ID
        model_name: 模型名称
        accuracy: 准确率
        precision: 精确率
        recall: 召回率
        f1_score: F1分数
        training_time: 训练时间(秒)
        inference_time: 推理时间(ms)
        model_size: 模型大小(MB)
        cross_val_scores: 交叉验证分数
        ranking: 综合排名
    """
    model_id: str
    model_name: str
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    training_time: float = 0.0
    inference_time: float = 0.0
    model_size: float = 0.0
    cross_val_scores: List[float] = field(default_factory=list)
    ranking: int = 0
    
    def to_dict(self) -> Dict:
        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "training_time": self.training_time,
            "inference_time": self.inference_time,
            "model_size": self.model_size,
            "cross_val_scores": self.cross_val_scores,
            "ranking": self.ranking
        }

@dataclass
class SelectionResult:
    """
    模型选择结果
    
    Attributes:
        selection_id: 选择任务ID
        task_type: 任务类型
        data_info: 数据信息
        tested_models: 测试的模型列表
        best_model: 最佳模型
        all_scores: 所有模型评分
        recommendations: 推荐列表
        total_time: 总耗时
        created_at: 创建时间
    """
    selection_id: str
    task_type: TaskType
    data_info: Dict[str, Any]
    tested_models: List[str]
    best_model: Optional[ModelScore]
    all_scores: List[ModelScore]
    recommendations: List[Dict]
    total_time: float
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict:
        return {
            "selection_id": self.selection_id,
            "task_type": self.task_type.value,
            "data_info": self.data_info,
            "tested_models": self.tested_models,
            "best_model": self.best_model.to_dict() if self.best_model else None,
            "all_scores": [s.to_dict() for s in self.all_scores],
            "recommendations": self.recommendations,
            "total_time": self.total_time,
            "created_at": self.created_at.isoformat()
        }


class ModelSelector:
    """
    自动模型选择器
    
    基于数据特征自动推荐和评估最佳模型。
    
    Usage:
        selector = ModelSelector()
        
        result = await selector.select(
            X_train=X_train,
            y_train=y_train,
            task_type=TaskType.CLASSIFICATION,
            time_budget=300  # 5分钟
        )
        
        print(f"Best model: {result.best_model.model_name}")
    """
    
    def __init__(self, verbose: bool = True):
        """初始化选择器"""
        self.verbose = verbose
        self.model_registry = self._build_model_registry()
        self.selections: Dict[str, SelectionResult] = {}
    
    def _build_model_registry(self) -> Dict[str, ModelCandidate]:
        """构建模型注册表"""
        return {
            # 线性模型
            "logistic_regression": ModelCandidate(
                model_id="logistic_regression",
                name="Logistic Regression",
                category=ModelCategory.LINEAR,
                task_type=[TaskType.CLASSIFICATION],
                class_path="sklearn.linear_model.LogisticRegression",
                hyperparameters={
                    "C": 1.0,
                    "penalty": "l2",
                    "solver": "lbfgs",
                    "max_iter": 1000
                },
                expected_accuracy=(0.7, 0.95),
                training_time=10.0,
                model_size=0.1,
                description="经典线性分类模型，适合线性可分数据"
            ),
            "linear_regression": ModelCandidate(
                model_id="linear_regression",
                name="Linear Regression",
                category=ModelCategory.LINEAR,
                task_type=[TaskType.REGRESSION],
                class_path="sklearn.linear_model.LinearRegression",
                hyperparameters={},
                expected_accuracy=(0.5, 0.9),
                training_time=5.0,
                model_size=0.05,
                description="简单线性回归模型"
            ),
            "ridge": ModelCandidate(
                model_id="ridge",
                name="Ridge Regression",
                category=ModelCategory.LINEAR,
                task_type=[TaskType.REGRESSION],
                class_path="sklearn.linear_model.Ridge",
                hyperparameters={"alpha": 1.0},
                expected_accuracy=(0.5, 0.9),
                training_time=5.0,
                model_size=0.05,
                description="L2正则化回归"
            ),
            
            # 树模型
            "decision_tree": ModelCandidate(
                model_id="decision_tree",
                name="Decision Tree",
                category=ModelCategory.TREE_BASED,
                task_type=[TaskType.CLASSIFICATION, TaskType.REGRESSION],
                class_path="sklearn.tree.DecisionTreeClassifier",
                hyperparameters={
                    "max_depth": 10,
                    "min_samples_split": 2,
                    "min_samples_leaf": 1
                },
                expected_accuracy=(0.6, 0.9),
                training_time=10.0,
                model_size=0.2,
                description="决策树，易于解释"
            ),
            "random_forest": ModelCandidate(
                model_id="random_forest",
                name="Random Forest",
                category=ModelCategory.ENSEMBLE,
                task_type=[TaskType.CLASSIFICATION, TaskType.REGRESSION],
                class_path="sklearn.ensemble.RandomForestClassifier",
                hyperparameters={
                    "n_estimators": 100,
                    "max_depth": 10,
                    "min_samples_split": 2
                },
                expected_accuracy=(0.75, 0.98),
                training_time=60.0,
                model_size=5.0,
                description="随机森林，集成学习方法"
            ),
            
            # 梯度提升
            "gradient_boosting": ModelCandidate(
                model_id="gradient_boosting",
                name="Gradient Boosting",
                category=ModelCategory.BOOSTING,
                task_type=[TaskType.CLASSIFICATION, TaskType.REGRESSION],
                class_path="sklearn.ensemble.GradientBoostingClassifier",
                hyperparameters={
                    "n_estimators": 100,
                    "learning_rate": 0.1,
                    "max_depth": 5
                },
                expected_accuracy=(0.8, 0.98),
                training_time=90.0,
                model_size=3.0,
                description="梯度提升树，高性能"
            ),
            "xgboost": ModelCandidate(
                model_id="xgboost",
                name="XGBoost",
                category=ModelCategory.BOOSTING,
                task_type=[TaskType.CLASSIFICATION, TaskType.REGRESSION],
                class_path="xgboost.XGBClassifier",
                hyperparameters={
                    "n_estimators": 100,
                    "learning_rate": 0.1,
                    "max_depth": 6
                },
                expected_accuracy=(0.8, 0.99),
                training_time=60.0,
                model_size=2.0,
                description="XGBoost，高效梯度提升"
            ),
            "lightgbm": ModelCandidate(
                model_id="lightgbm",
                name="LightGBM",
                category=ModelCategory.BOOSTING,
                task_type=[TaskType.CLASSIFICATION, TaskType.REGRESSION],
                class_path="lightgbm.LGBMClassifier",
                hyperparameters={
                    "n_estimators": 100,
                    "learning_rate": 0.1,
                    "max_depth": -1
                },
                expected_accuracy=(0.8, 0.99),
                training_time=40.0,
                model_size=1.0,
                description="LightGBM，快速高效"
            ),
            
            # SVM
            "svm": ModelCandidate(
                model_id="svm",
                name="SVM",
                category=ModelCategory.LINEAR,
                task_type=[TaskType.CLASSIFICATION],
                class_path="sklearn.svm.SVC",
                hyperparameters={
                    "C": 1.0,
                    "kernel": "rbf",
                    "gamma": "scale"
                },
                expected_accuracy=(0.7, 0.95),
                training_time=30.0,
                model_size=0.5,
                description="支持向量机"
            ),
            
            # KNN
            "knn": ModelCandidate(
                model_id="knn",
                name="KNN",
                category=ModelCategory.LINEAR,
                task_type=[TaskType.CLASSIFICATION, TaskType.REGRESSION],
                class_path="sklearn.neighbors.KNeighborsClassifier",
                hyperparameters={"n_neighbors": 5},
                expected_accuracy=(0.6, 0.9),
                training_time=5.0,
                model_size=0.1,
                description="K近邻算法"
            ),
            
            # 神经网络
            "mlp": ModelCandidate(
                model_id="mlp",
                name="MLP Neural Network",
                category=ModelCategory.NEURAL_NETWORK,
                task_type=[TaskType.CLASSIFICATION, TaskType.REGRESSION],
                class_path="sklearn.neural_network.MLPClassifier",
                hyperparameters={
                    "hidden_layer_sizes": (100, 50),
                    "max_iter": 500,
                    "alpha": 0.001
                },
                expected_accuracy=(0.7, 0.95),
                training_time=60.0,
                model_size=0.5,
                description="多层感知机"
            ),
        }
    
    def get_recommended_models(
        self,
        task_type: TaskType,
        data_info: Optional[Dict[str, Any]] = None,
        top_k: int = 5
    ) -> List[ModelCandidate]:
        """
        根据任务类型获取推荐模型
        
        Args:
            task_type: 任务类型
            data_info: 数据信息 (可选，用于更精细的推荐)
            top_k: 返回前k个模型
            
        Returns:
            推荐模型列表
        """
        # 筛选适合任务类型的模型
        candidates = [
            m for m in self.model_registry.values()
            if task_type in m.task_type
        ]
        
        # 根据数据特征调整推荐
        if data_info:
            n_samples = data_info.get("n_samples", 1000)
            n_features = data_info.get("n_features", 10)
            is_large_data = n_samples > 10000
            is_high_dim = n_features > 100
            
            # 大数据优先使用轻量级模型
            if is_large_data:
                candidates.sort(key=lambda m: m.training_time)
            # 高维数据优先使用正则化模型
            if is_high_dim:
                candidates = [m for m in candidates if "linear" in m.category.value]
        
        return candidates[:top_k]
    
    async def select(
        self,
        X_train: Any,
        y_train: Any,
        task_type: TaskType,
        time_budget: int = 300,
        cv_folds: int = 5,
        test_size: float = 0.2,
        metric: Optional[str] = None,
        models: Optional[List[str]] = None
    ) -> SelectionResult:
        """
        执行自动模型选择
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            task_type: 任务类型
            time_budget: 时间预算(秒)
            cv_folds: 交叉验证折数
            test_size: 测试集比例
            metric: 评估指标
            models: 指定测试的模型列表
            
        Returns:
            SelectionResult: 选择结果
        """
        selection_id = str(uuid4())
        started_at = datetime.utcnow()
        
        if self.verbose:
            print(f"[Selector] Starting model selection for {task_type.value}")
        
        # 数据信息
        try:
            import pandas as pd
            if hasattr(X_train, "shape"):
                n_samples = X_train.shape[0]
                n_features = X_train.shape[1] if len(X_train.shape) > 1 else 1
            elif isinstance(X_train, (list, pd.DataFrame)):
                n_samples = len(X_train)
                n_features = len(X_train[0]) if X_train and isinstance(X_train[0], (list, tuple)) else 1
            else:
                n_samples = 1000
                n_features = 10
        except:
            n_samples = 1000
            n_features = 10
        
        data_info = {
            "n_samples": n_samples,
            "n_features": n_features,
            "cv_folds": cv_folds,
            "test_size": test_size
        }
        
        # 确定评估指标
        if metric is None:
            metric = "accuracy" if task_type == TaskType.CLASSIFICATION else "mse"
        
        # 获取候选模型
        if models:
            candidates = [
                self.model_registry[m] for m in models
                if m in self.model_registry
            ]
        else:
            # 根据时间预算选择模型数量
            max_models = max(3, min(8, time_budget // 30))
            candidates = self.get_recommended_models(task_type, data_info, max_models)
        
        # 评估模型
        all_scores = []
        tested_models = []
        
        for candidate in candidates:
            # 检查时间预算
            elapsed = (datetime.utcnow() - started_at).total_seconds()
            if elapsed > time_budget * 0.9:  # 保留10%时间做总结
                if self.verbose:
                    print(f"[Selector] Time budget reached, stopping...")
                break
            
            try:
                score = await self._evaluate_model(
                    candidate, X_train, y_train, task_type,
                    cv_folds, test_size, metric
                )
                all_scores.append(score)
                tested_models.append(candidate.model_id)
                
                if self.verbose:
                    print(f"[Selector] {candidate.name}: accuracy={score.accuracy:.4f}")
                    
            except Exception as e:
                if self.verbose:
                    print(f"[Selector] {candidate.name}: Failed - {str(e)}")
        
        # 计算排名
        all_scores.sort(key=lambda x: x.accuracy, reverse=True)
        for i, score in enumerate(all_scores):
            score.ranking = i + 1
        
        # 最佳模型
        best_model = all_scores[0] if all_scores else None
        
        # 生成推荐
        recommendations = self._generate_recommendations(
            all_scores, task_type, data_info
        )
        
        total_time = (datetime.utcnow() - started_at).total_seconds()
        
        result = SelectionResult(
            selection_id=selection_id,
            task_type=task_type,
            data_info=data_info,
            tested_models=tested_models,
            best_model=best_model,
            all_scores=all_scores,
            recommendations=recommendations,
            total_time=total_time
        )
        
        self.selections[selection_id] = result
        
        if self.verbose:
            print(f"[Selector] Done! Best model: {best_model.model_name if best_model else 'None'}")
            print(f"[Selector] Total time: {total_time:.1f}s")
        
        return result
    
    async def _evaluate_model(
        self,
        candidate: ModelCandidate,
        X_train: Any,
        y_train: Any,
        task_type: TaskType,
        cv_folds: int,
        test_size: float,
        metric: str
    ) -> ModelScore:
        """评估单个模型"""
        import time
        from sklearn.model_selection import train_test_split, cross_val_score
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score,
            f1_score, mean_squared_error
        )
        
        # 分割数据
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_train, y_train, test_size=test_size, random_state=42
        )
        
        # 导入并实例化模型
        class_path = candidate.class_path
        module_path, class_name = class_path.rsplit(".", 1)
        
        module = __import__(module_path, fromlist=[class_name])
        ModelClass = getattr(module, class_name)
        
        # 根据任务类型选择合适的类
        if task_type == TaskType.REGRESSION:
            if "Classifier" in class_name:
                class_name = class_name.replace("Classifier", "Regressor")
                ModelClass = getattr(module, class_name)
        
        model = ModelClass(**candidate.hyperparameters)
        
        # 训练并计时
        start_time = time.time()
        model.fit(X_tr, y_tr)
        training_time = time.time() - start_time
        
        # 推理并计时
        start_time = time.time()
        y_pred = model.predict(X_te)
        inference_time = (time.time() - start_time) / len(X_te) * 1000  # ms per sample
        
        # 计算指标
        if task_type == TaskType.CLASSIFICATION:
            accuracy = accuracy_score(y_te, y_pred)
            precision = precision_score(y_te, y_pred, average="weighted", zero_division=0)
            recall = recall_score(y_te, y_pred, average="weighted", zero_division=0)
            f1 = f1_score(y_te, y_pred, average="weighted", zero_division=0)
        else:
            accuracy = 1 - mean_squared_error(y_te, y_pred)  # 简化的准确率
            precision = accuracy
            recall = accuracy
            f1 = accuracy
        
        return ModelScore(
            model_id=candidate.model_id,
            model_name=candidate.name,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            training_time=training_time,
            inference_time=inference_time,
            model_size=candidate.model_size
        )
    
    def _generate_recommendations(
        self,
        scores: List[ModelScore],
        task_type: TaskType,
        data_info: Dict
    ) -> List[Dict]:
        """生成模型推荐"""
        recommendations = []
        
        if not scores:
            return [{"message": "No models were successfully evaluated"}]
        
        # 最佳模型推荐
        best = scores[0]
        recommendations.append({
            "type": "best_model",
            "model_id": best.model_id,
            "model_name": best.model_name,
            "reason": f"Highest accuracy: {best.accuracy:.4f}"
        })
        
        # 速度最快推荐
        if len(scores) > 1:
            fastest = min(scores, key=lambda x: x.training_time)
            if fastest.model_id != best.model_id:
                recommendations.append({
                    "type": "fastest",
                    "model_id": fastest.model_id,
                    "model_name": fastest.model_name,
                    "reason": f"Fastest training: {fastest.training_time:.1f}s"
                })
        
        # 最佳性价比推荐
        if len(scores) > 2:
            # 准确率/训练时间的比值
            efficiency_scores = []
            for s in scores:
                if s.training_time > 0:
                    score = s.accuracy / (s.training_time / 60)  # accuracy per minute
                    efficiency_scores.append((s, score))
            efficiency_scores.sort(key=lambda x: x[1], reverse=True)
            
            if efficiency_scores and efficiency_scores[0][0].model_id != best.model_id:
                efficient = efficiency_scores[0][0]
                recommendations.append({
                    "type": "efficient",
                    "model_id": efficient.model_id,
                    "model_name": efficient.model_name,
                    "reason": f"Best accuracy/time ratio"
                })
        
        return recommendations
    
    def get_selection(self, selection_id: str) -> Optional[SelectionResult]:
        """获取选择任务结果"""
        return self.selections.get(selection_id)
    
    def list_selections(self, limit: int = 20) -> List[Dict]:
        """列出所有选择任务"""
        result = []
        for sel_id, sel in self.selections.items():
            result.append({
                "selection_id": sel_id,
                "task_type": sel.task_type.value,
                "best_model": sel.best_model.model_name if sel.best_model else None,
                "best_accuracy": sel.best_model.accuracy if sel.best_model else None,
                "tested_models": len(sel.tested_models),
                "total_time": sel.total_time,
                "created_at": sel.created_at.isoformat()
            })
        
        result.sort(key=lambda x: x["created_at"], reverse=True)
        return result[:limit]


# 默认选择器实例
default_selector = ModelSelector(verbose=True)
