"""
持续学习系统 API 接口

提供RESTful API风格的接口
"""

import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class APIResponse:
    """API响应"""
    success: bool
    message: str
    data: Optional[Dict] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict())
    
    @classmethod
    def success(cls, message: str, data: Optional[Dict] = None) -> 'APIResponse':
        return cls(success=True, message=message, data=data)
    
    @classmethod
    def error(cls, message: str, data: Optional[Dict] = None) -> 'APIResponse':
        return cls(success=False, message=message, data=data)


class ContinualLearningAPI:
    """
    持续学习系统 API
    
    提供RESTful风格的接口
    """
    
    def __init__(
        self,
        learner: 'IncrementalLearner',
        curriculum: 'CurriculumLearning',
        memory_replay: 'MemoryReplay',
        knowledge_consolidation: 'KnowledgeConsolidation'
    ):
        self.learner = learner
        self.curriculum = curriculum
        self.memory_replay = memory_replay
        self.knowledge_consolidation = knowledge_consolidation
        
    # ==================== 任务管理 API ====================
    
    def add_task(
        self,
        task_id: str,
        name: str,
        data: Dict,
        difficulty: float = 1.0,
        priority: int = 0
    ) -> APIResponse:
        """添加新任务"""
        try:
            # 转换数据格式
            from torch.utils.data import TensorDataset
            
            if 'features' in data and 'labels' in data:
                features = data['features']
                labels = data['labels']
                
                import torch
                dataset = TensorDataset(
                    torch.tensor(features, dtype=torch.float32),
                    torch.tensor(labels, dtype=torch.long)
                )
            else:
                return APIResponse.error("Invalid data format")
            
            # 添加任务
            self.learner.add_task(dataset, task_id, name, difficulty)
            self.curriculum.add_task(task_id, dataset, difficulty, name, priority)
            
            return APIResponse.success(
                f"Task {task_id} added successfully",
                {"task_id": task_id, "difficulty": difficulty}
            )
            
        except Exception as e:
            return APIResponse.error(f"Failed to add task: {str(e)}")
    
    def get_tasks(self) -> APIResponse:
        """获取所有任务"""
        try:
            tasks = [
                {
                    "task_id": task_id,
                    "name": info.name,
                    "difficulty": info.difficulty,
                    "completed": info.completed,
                    "accuracy": info.accuracy
                }
                for task_id, info in self.learner.tasks.items()
            ]
            
            return APIResponse.success("Tasks retrieved", {"tasks": tasks})
            
        except Exception as e:
            return APIResponse.error(f"Failed to get tasks: {str(e)}")
    
    def get_task(self, task_id: str) -> APIResponse:
        """获取单个任务"""
        try:
            if task_id not in self.learner.tasks:
                return APIResponse.error(f"Task {task_id} not found")
                
            task = self.learner.tasks[task_id]
            
            return APIResponse.success(
                "Task retrieved",
                {
                    "task_id": task.task_id,
                    "name": task.name,
                    "difficulty": task.difficulty,
                    "num_samples": task.num_samples,
                    "completed": task.completed,
                    "accuracy": task.accuracy
                }
            )
            
        except Exception as e:
            return APIResponse.error(f"Failed to get task: {str(e)}")
    
    def delete_task(self, task_id: str) -> APIResponse:
        """删除任务"""
        try:
            if task_id in self.learner.tasks:
                del self.learner.tasks[task_id]
                
            if task_id in self.curriculum.tasks:
                del self.curriculum.tasks[task_id]
                
            return APIResponse.success(f"Task {task_id} deleted")
            
        except Exception as e:
            return APIResponse.error(f"Failed to delete task: {str(e)}")
    
    # ==================== 训练 API ====================
    
    def train_task(
        self,
        task_id: str,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        use_replay: Optional[bool] = None,
        use_consolidation: Optional[bool] = None
    ) -> APIResponse:
        """训练单个任务"""
        try:
            if task_id not in self.learner.tasks:
                return APIResponse.error(f"Task {task_id} not found")
            
            # 使用配置或默认值
            epochs = epochs or self.learner.config.training.epochs
            batch_size = batch_size or self.learner.config.training.batch_size
            use_replay = use_replay if use_replay is not None else self.learner.config.training.use_replay
            use_consolidation = use_consolidation if use_consolidation is not None else self.learner.config.training.use_consolidation
            
            # 执行训练
            result = self.learner.train(
                task_id=task_id,
                epochs=epochs,
                batch_size=batch_size,
                use_replay=use_replay,
                use_consolidation=use_consolidation
            )
            
            # 标记课程完成
            self.curriculum.complete_task(task_id, {"accuracy": result.accuracy})
            
            return APIResponse.success(
                f"Task {task_id} trained successfully",
                {
                    "task_id": result.task_id,
                    "loss": result.loss,
                    "accuracy": result.accuracy,
                    "time_seconds": result.time_seconds,
                    "samples_processed": result.samples_processed
                }
            )
            
        except Exception as e:
            return APIResponse.error(f"Failed to train task: {str(e)}")
    
    def train_all(self, task_ids: Optional[List[str]] = None) -> APIResponse:
        """训练所有任务"""
        try:
            if task_ids is None:
                task_ids = self.learner.task_order
                
            results = self.learner.train_task_sequence(
                task_ids=task_ids,
                epochs=self.learner.config.training.epochs,
                batch_size=self.learner.config.training.batch_size,
                use_replay=self.learner.config.training.use_replay,
                use_consolidation=self.learner.config.training.use_consolidation
            )
            
            # 更新课程
            for result in results:
                self.curriculum.complete_task(result.task_id, {"accuracy": result.accuracy})
            
            return APIResponse.success(
                "Training completed",
                {
                    "num_tasks": len(results),
                    "average_accuracy": sum(r.accuracy for r in results) / len(results),
                    "total_time": sum(r.time_seconds for r in results),
                    "results": [
                        {
                            "task_id": r.task_id,
                            "accuracy": r.accuracy,
                            "loss": r.loss
                        }
                        for r in results
                    ]
                }
            )
            
        except Exception as e:
            return APIResponse.error(f"Failed to train all tasks: {str(e)}")
    
    def train_curriculum(self, max_tasks: Optional[int] = None) -> APIResponse:
        """按课程训练"""
        try:
            ordered_tasks = self.curriculum.get_ordered_tasks()
            
            if max_tasks:
                ordered_tasks = ordered_tasks[:max_tasks]
            
            results = []
            
            for task in ordered_tasks:
                if task.completed:
                    continue
                    
                result = self.learner.train(
                    task_id=task.task_id,
                    epochs=self.learner.config.training.epochs,
                    batch_size=self.learner.config.training.batch_size,
                    use_replay=self.learner.config.training.use_replay,
                    use_consolidation=self.learner.config.training.use_consolidation
                )
                
                results.append(result)
                self.curriculum.complete_task(task.task_id, {"accuracy": result.accuracy})
            
            return APIResponse.success(
                "Curriculum training completed",
                {
                    "num_tasks": len(results),
                    "completed_tasks": [r.task_id for r in results],
                    "average_accuracy": sum(r.accuracy for r in results) / len(results) if results else 0
                }
            )
            
        except Exception as e:
            return APIResponse.error(f"Failed to train curriculum: {str(e)}")
    
    # ==================== 评估 API ====================
    
    def evaluate(self, task_id: Optional[str] = None) -> APIResponse:
        """评估模型"""
        try:
            if task_id:
                # 评估单个任务
                if task_id not in self.learner.tasks:
                    return APIResponse.error(f"Task {task_id} not found")
                    
                accuracy = self.learner._evaluate(self.learner.tasks[task_id].dataset)
                
                return APIResponse.success(
                    f"Evaluation on task {task_id}",
                    {"task_id": task_id, "accuracy": accuracy}
                )
            else:
                # 评估所有任务
                accuracies = {}
                
                for tid, task in self.learner.tasks.items():
                    accuracy = self.learner._evaluate(task.dataset)
                    accuracies[tid] = accuracy
                
                avg_accuracy = sum(accuracies.values()) / len(accuracies) if accuracies else 0
                
                return APIResponse.success(
                    "Evaluation on all tasks",
                    {"accuracies": accuracies, "average_accuracy": avg_accuracy}
                )
                
        except Exception as e:
            return APIResponse.error(f"Failed to evaluate: {str(e)}")
    
    def get_forgetting(self) -> APIResponse:
        """获取遗忘率"""
        try:
            forgetting = self.learner.forgetting_ratio
            
            return APIResponse.success(
                "Forgetting metric",
                {"forgetting_ratio": forgetting}
            )
            
        except Exception as e:
            return APIResponse.error(f"Failed to compute forgetting: {str(e)}")
    
    # ==================== 记忆回放 API ====================
    
    def get_replay_statistics(self) -> APIResponse:
        """获取记忆回放统计"""
        try:
            stats = self.memory_replay.get_statistics()
            
            return APIResponse.success("Replay statistics", stats)
            
        except Exception as e:
            return APIResponse.error(f"Failed to get replay statistics: {str(e)}")
    
    def clear_replay_buffer(self) -> APIResponse:
        """清空回放缓冲区"""
        try:
            self.memory_replay.clear()
            
            return APIResponse.success("Replay buffer cleared")
            
        except Exception as e:
            return APIResponse.error(f"Failed to clear replay buffer: {str(e)}")
    
    # ==================== 课程管理 API ====================
    
    def get_curriculum(self) -> APIResponse:
        """获取当前课程"""
        try:
            curriculum = self.curriculum.get_curriculum()
            
            return APIResponse.success(
                "Curriculum retrieved",
                {"curriculum": curriculum, "strategy": self.curriculum.strategy}
            )
            
        except Exception as e:
            return APIResponse.error(f"Failed to get curriculum: {str(e)}")
    
    def select_samples(self, task_id: str, num_samples: int = 32) -> APIResponse:
        """选择样本"""
        try:
            samples = self.curriculum.select_samples(task_id, num_samples)
            
            return APIResponse.success(
                "Samples selected",
                {"task_id": task_id, "samples": samples}
            )
            
        except Exception as e:
            return APIResponse.error(f"Failed to select samples: {str(e)}")
    
    def update_curriculum(
        self,
        task_difficulties: Dict[str, float]
    ) -> APIResponse:
        """更新课程难度"""
        try:
            for task_id, difficulty in task_difficulties.items():
                if task_id in self.curriculum.tasks:
                    self.curriculum.tasks[task_id].difficulty = difficulty
            
            return APIResponse.success("Curriculum updated")
            
        except Exception as e:
            return APIResponse.error(f"Failed to update curriculum: {str(e)}")
    
    # ==================== 模型管理 API ====================
    
    def get_model(self) -> APIResponse:
        """获取模型信息"""
        try:
            model = self.learner.get_model()
            
            return APIResponse.success(
                "Model retrieved",
                {"model_type": type(model).__name__, "num_parameters": sum(p.numel() for p in model.parameters())}
            )
            
        except Exception as e:
            return APIResponse.error(f"Failed to get model: {str(e)}")
    
    def save_checkpoint(self, path: str) -> APIResponse:
        """保存检查点"""
        try:
            self.learner.save_checkpoint(path)
            
            return APIResponse.success(f"Checkpoint saved to {path}")
            
        except Exception as e:
            return APIResponse.error(f"Failed to save checkpoint: {str(e)}")
    
    def load_checkpoint(self, path: str) -> APIResponse:
        """加载检查点"""
        try:
            self.learner.load_checkpoint(path)
            
            return APIResponse.success(f"Checkpoint loaded from {path}")
            
        except Exception as e:
            return APIResponse.error(f"Failed to load checkpoint: {str(e)}")
    
    # ==================== 统计 API ====================
    
    def get_statistics(self) -> APIResponse:
        """获取系统统计"""
        try:
            learner_stats = self.learner.get_statistics()
            curriculum_stats = self.curriculum.get_statistics()
            replay_stats = self.memory_replay.get_statistics()
            
            stats = {
                "learner": learner_stats,
                "curriculum": curriculum_stats,
                "replay": replay_stats,
                "timestamp": datetime.now().isoformat()
            }
            
            return APIResponse.success("Statistics retrieved", stats)
            
        except Exception as e:
            return APIResponse.error(f"Failed to get statistics: {str(e)}")
    
    def get_training_history(self) -> APIResponse:
        """获取训练历史"""
        try:
            history = [
                {
                    "task_id": r.task_id,
                    "success": r.success,
                    "loss": r.loss,
                    "accuracy": r.accuracy,
                    "time_seconds": r.time_seconds
                }
                for r in self.learner.training_history
            ]
            
            return APIResponse.success("Training history retrieved", {"history": history})
            
        except Exception as e:
            return APIResponse.error(f"Failed to get training history: {str(e)}")
    
    # ==================== 知识巩固 API ====================
    
    def consolidate(
        self,
        method: Optional[str] = None,
        **kwargs
    ) -> APIResponse:
        """执行知识巩固"""
        try:
            if method:
                self.knowledge_consolidation.switch_method(method, **kwargs)
            
            stats = self.knowledge_consolidation.get_statistics()
            
            return APIResponse.success(
                "Knowledge consolidation completed",
                stats
            )
            
        except Exception as e:
            return APIResponse.error(f"Failed to consolidate: {str(e)}")
    
    def get_consolidation_statistics(self) -> APIResponse:
        """获取巩固统计"""
        try:
            stats = self.knowledge_consolidation.get_statistics()
            
            return APIResponse.success("Consolidation statistics", stats)
            
        except Exception as e:
            return APIResponse.error(f"Failed to get consolidation statistics: {str(e)}")
    
    # ==================== 配置管理 API ====================
    
    def get_config(self) -> APIResponse:
        """获取当前配置"""
        try:
            config_dict = self.learner.config.to_dict()
            
            return APIResponse.success("Configuration retrieved", config_dict)
            
        except Exception as e:
            return APIResponse.error(f"Failed to get configuration: {str(e)}")
    
    def update_config(self, config_updates: Dict[str, Any]) -> APIResponse:
        """更新配置"""
        try:
            from .config import ContinualLearningConfig
            
            # 更新配置
            for key, value in config_updates.items():
                if hasattr(self.learner.config, key):
                    setattr(self.learner.config, key, value)
            
            return APIResponse.success("Configuration updated")
            
        except Exception as e:
            return APIResponse.error(f"Failed to update configuration: {str(e)}")


def create_api(
    learner: 'IncrementalLearner',
    curriculum: 'CurriculumLearning',
    memory_replay: 'MemoryReplay',
    knowledge_consolidation: 'KnowledgeConsolidation'
) -> ContinualLearningAPI:
    """创建API实例"""
    return ContinualLearningAPI(
        learner=learner,
        curriculum=curriculum,
        memory_replay=memory_replay,
        knowledge_consolidation=knowledge_consolidation
    )
