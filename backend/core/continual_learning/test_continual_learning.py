"""
持续学习系统测试用例

测试核心功能：
- 增量学习
- 记忆回放
- 知识巩固
- 课程学习
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from typing import List, Dict, Tuple
import sys
import os

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ==================== 测试工具 ====================

def set_seed(seed: int = 42):
    """设置随机种子"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def generate_mock_data(
    num_samples: int = 100,
    input_dim: int,
    num_classes: int = 5,
    seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor]:
    """生成模拟数据"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    features = np.random.randn(num_samples, input_dim).astype(np.float32) * 0.1
    labels = np.random.randint(0, num_classes, num_samples)
    
    return (
        torch.tensor(features, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.long)
    )


class SimpleNet(nn.Module):
    """用于测试的简单网络"""
    
    def __init__(self, input_dim: int = 50, hidden_dim: int = 32, output_dim: int = 5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# ==================== 测试类 ====================

class TestIncrementalLearner:
    """测试增量学习器"""
    
    def setup_method(self):
        """设置测试环境"""
        set_seed(42)
        self.model = SimpleNet()
        
    def test_add_task(self):
        """测试添加任务"""
        from continual_learning import IncrementalLearner
        
        learner = IncrementalLearner(model=self.model)
        
        features, labels = generate_mock_data(num_samples=100)
        dataset = TensorDataset(features, labels)
        
        # 添加任务
        task = learner.add_task(dataset, task_id="task_1", name="Test Task")
        
        assert task is not None
        assert task.task_id == "task_1"
        assert task.num_samples == 100
        assert len(learner.tasks) == 1
        assert len(learner.task_order) == 1
        
    def test_train_single_task(self):
        """测试单个任务训练"""
        from continual_learning import IncrementalLearner
        
        learner = IncrementalLearner(model=self.model)
        
        features, labels = generate_mock_data(num_samples=200)
        dataset = TensorDataset(features, labels)
        
        learner.add_task(dataset, task_id="task_1")
        
        # 训练
        result = learner.train(task_id="task_1", epochs=3, batch_size=32)
        
        assert result.success
        assert result.loss > 0
        assert result.samples_processed > 0
        assert result.task_id == "task_1"
        
    def test_train_multiple_tasks(self):
        """测试多任务顺序训练"""
        from continual_learning import IncrementalLearner
        
        learner = IncrementalLearner(model=self.model)
        
        # 添加多个任务
        for i in range(3):
            features, labels = generate_mock_data(num_samples=200, seed=42+i)
            dataset = TensorDataset(features, labels)
            learner.add_task(dataset, task_id=f"task_{i}")
        
        # 训练所有任务
        results = learner.train_task_sequence(epochs=2)
        
        assert len(results) == 3
        assert all(r.success for r in results)
        
    def test_online_update(self):
        """测试在线更新"""
        from continual_learning import IncrementalLearner
        
        learner = IncrementalLearner(model=self.model)
        
        features, labels = generate_mock_data(num_samples=50)
        
        # 在线更新
        learner.online_update(features, labels)
        
        assert learner.current_task_id is None  # 没有设置任务
        
    def test_sample_efficient_learning(self):
        """测试样本高效学习"""
        from continual_learning import IncrementalLearner
        
        learner = IncrementalLearner(model=self.model)
        
        features, labels = generate_mock_data(num_samples=500)
        dataset = TensorDataset(features, labels)
        
        learner.add_task(dataset, task_id="task_1")
        
        # 使用10%样本训练
        result = learner.sample_efficient_learning("task_1", sample_ratio=0.1, epochs=2)
        
        assert result.success
        
    def test_get_statistics(self):
        """测试获取统计信息"""
        from continual_learning import IncrementalLearner
        
        learner = IncrementalLearner(model=self.model)
        
        features, labels = generate_mock_data(num_samples=200)
        dataset = TensorDataset(features, labels)
        
        learner.add_task(dataset, task_id="task_1")
        learner.train(task_id="task_1", epochs=2)
        
        stats = learner.get_statistics()
        
        assert "total_tasks" in stats
        assert "completed_tasks" in stats
        assert "average_accuracy" in stats
        assert stats["total_tasks"] == 1
        
    def test_checkpoint(self, tmp_path):
        """测试检查点保存和加载"""
        from continual_learning import IncrementalLearner
        
        learner = IncrementalLearner(model=self.model)
        
        features, labels = generate_mock_data(num_samples=200)
        dataset = TensorDataset(features, labels)
        
        learner.add_task(dataset, task_id="task_1")
        learner.train(task_id="task_1", epochs=2)
        
        # 保存
        checkpoint_path = str(tmp_path / "checkpoint.pt")
        learner.save_checkpoint(checkpoint_path)
        
        # 加载到新学习器
        new_learner = IncrementalLearner(model=SimpleNet())
        new_learner.load_checkpoint(checkpoint_path)
        
        assert len(new_learner.tasks) == 1
        assert len(new_learner.training_history) == 1


class TestMemoryReplay:
    """测试记忆回放"""
    
    def test_experience_replay(self):
        """测试经验回放"""
        from continual_learning import ExperienceReplay
        
        replay = ExperienceReplay(capacity=100)
        
        features, labels = generate_mock_data(num_samples=50)
        dataset = TensorDataset(features, labels)
        
        # 存储
        count = replay.store(dataset)
        assert count == 50
        assert len(replay) == 50
        
        # 采样
        batch_features, batch_labels = replay.sample(batch_size=10)
        assert len(batch_features) == 10
        
    def test_importance_sampling_replay(self):
        """测试优先级回放"""
        from continual_learning import ImportanceSamplingReplay
        
        replay = ImportanceSamplingReplay(capacity=100, alpha=0.6, beta=0.4)
        
        features, labels = generate_mock_data(num_samples=50)
        dataset = TensorDataset(features, labels)
        
        replay.store(dataset)
        assert len(replay) == 50
        
        # 采样
        batch_features, batch_labels = replay.sample(batch_size=10)
        assert len(batch_features) == 10
        
    def test_compressed_replay(self):
        """测试压缩回放"""
        from continual_learning import CompressedReplay
        
        replay = CompressedReplay(capacity=100, compression_ratio=0.1)
        
        features, labels = generate_mock_data(num_samples=50, input_dim=100)
        dataset = TensorDataset(features, labels)
        
        replay.store(dataset)
        
        # 采样
        batch_features, batch_labels = replay.sample(batch_size=10)
        assert batch_features.shape[1] < 100  # 压缩后维度降低
        
    def test_memory_replay_switch_strategy(self):
        """测试策略切换"""
        from continual_learning import MemoryReplay
        
        replay = MemoryReplay(strategy="experience", capacity=100)
        
        features, labels = generate_mock_data(num_samples=50)
        dataset = TensorDataset(features, labels)
        
        replay.store(dataset)
        assert replay.strategy == "experience"
        
        # 切换策略
        replay.switch_strategy("importance")
        assert replay.strategy == "importance"
        
    def test_replay_statistics(self):
        """测试回放统计"""
        from continual_learning import MemoryReplay
        
        replay = MemoryReplay(strategy="experience", capacity=100)
        
        features, labels = generate_mock_data(num_samples=50)
        dataset = TensorDataset(features, labels)
        
        replay.store(dataset)
        
        stats = replay.get_statistics()
        
        assert "strategy" in stats
        assert "buffer_size" in stats
        assert "capacity" in stats


class TestKnowledgeConsolidation:
    """测试知识巩固"""
    
    def test_ewc(self):
        """测试EWC"""
        from continual_learning import EWC, IncrementalLearner
        
        set_seed(42)
        model = SimpleNet()
        
        ewc = EWC(model, importance_weight=1000)
        
        # 注册任务
        ewc.register_task("task_1")
        
        # 计算巩固损失
        loss = ewc.compute_consolidation_loss(model, "task_1")
        assert isinstance(loss, torch.Tensor)
        
        # 训练后保存参数
        ewc.after_training("task_1")
        
    def test_regularization(self):
        """测试正则化"""
        from continual_learning import Regularization
        
        set_seed(42)
        model = SimpleNet()
        
        reg = Regularization(model, weight_decay=0.01)
        
        reg.register_task("task_1")
        
        loss = reg.compute_consolidation_loss(model, "task_1")
        assert isinstance(loss, torch.Tensor)
        
    def test_knowledge_distillation(self):
        """测试知识蒸馏"""
        from continual_learning import KnowledgeDistillation
        
        set_seed(42)
        model = SimpleNet()
        
        kd = KnowledgeDistillation(model, temperature=2.0, alpha=0.5)
        
        kd.register_task("task_1")
        
        # 创建新模型
        new_model = SimpleNet()
        
        loss = kd.compute_consolidation_loss(new_model, "task_1")
        assert isinstance(loss, torch.Tensor)
        
    def test_consolidation_switch_method(self):
        """测试方法切换"""
        from continual_learning import KnowledgeConsolidation
        
        set_seed(42)
        model = SimpleNet()
        
        consolidation = KnowledgeConsolidation(model, method="ewc")
        
        # 切换到SI
        consolidation.switch_method("si", importance_weight=1000)
        
        stats = consolidation.get_statistics()
        assert "num_tasks" in stats
        
    def test_fisher_update(self):
        """测试Fisher信息更新"""
        from continual_learning import EWC, IncrementalLearner
        
        set_seed(42)
        model = SimpleNet()
        
        ewc = EWC(model, importance_weight=1000)
        
        features, labels = generate_mock_data(num_samples=100)
        dataset = TensorDataset(features, labels)
        
        # 添加任务
        learner = IncrementalLearner(model=model)
        learner.add_task(dataset, task_id="task_1")
        
        # 更新Fisher信息
        ewc.update_fisher(dataset=dataset)
        
        norm = ewc.get_fisher_norm("task_1")
        assert norm >= 0


class TestCurriculumLearning:
    """测试课程学习"""
    
    def test_difficulty_scheduler(self):
        """测试难度调度器"""
        from continual_learning import CurriculumLearning, TaskInfo
        from torch.utils.data import TensorDataset
        
        curriculum = CurriculumLearning(strategy="difficulty")
        
        # 添加任务
        for i in range(3):
            features, labels = generate_mock_data(num_samples=100, seed=42+i)
            dataset = TensorDataset(features, labels)
            
            curriculum.add_task(
                task_id=f"task_{i}",
                dataset=dataset,
                difficulty=0.5 + i * 0.2,
                name=f"Task {i}"
            )
        
        # 获取排序任务
        ordered = curriculum.get_ordered_tasks()
        
        assert len(ordered) == 3
        # 难度应该递增
        assert ordered[0].difficulty <= ordered[1].difficulty <= ordered[2].difficulty
        
    def test_progressive_scheduler(self):
        """测试渐进式调度器"""
        from continual_learning import CurriculumLearning, TaskInfo
        from torch.utils.data import TensorDataset
        
        curriculum = CurriculumLearning(strategy="progressive")
        
        # 添加任务
        for i in range(3):
            features, labels = generate_mock_data(num_samples=100, seed=42+i)
            dataset = TensorDataset(features, labels)
            
            curriculum.add_task(
                task_id=f"task_{i}",
                dataset=dataset,
                difficulty=1.0
            )
        
        # 获取当前难度
        current_difficulty = curriculum.curriculum_strategy.get_current_difficulty()
        assert current_difficulty >= 0.3
        
    def test_active_learning(self):
        """测试主动学习"""
        from continual_learning import CurriculumLearning
        from torch.utils.data import TensorDataset
        
        set_seed(42)
        model = SimpleNet()
        
        curriculum = CurriculumLearning(model=model, strategy="active")
        
        features, labels = generate_mock_data(num_samples=100)
        dataset = TensorDataset(features, labels)
        
        curriculum.add_task(task_id="task_1", dataset=dataset, difficulty=1.0)
        
        # 选择样本
        samples = curriculum.select_samples("task_1", num_samples=10)
        
        assert len(samples) == 10
        
    def test_adaptive_curriculum(self):
        """测试自适应课程"""
        from continual_learning import CurriculumLearning, TaskInfo
        from torch.utils.data import TensorDataset
        
        set_seed(42)
        model = SimpleNet()
        
        curriculum = CurriculumLearning(model=model, strategy="adaptive")
        
        # 添加任务
        for i in range(3):
            features, labels = generate_mock_data(num_samples=100, seed=42+i)
            dataset = TensorDataset(features, labels)
            
            curriculum.add_task(
                task_id=f"task_{i}",
                dataset=dataset,
                difficulty=1.0
            )
        
        # 标记完成
        curriculum.complete_task("task_1", {"accuracy": 0.9})
        curriculum.complete_task("task_2", {"accuracy": 0.6})
        
        # 获取更新后的顺序
        ordered = curriculum.get_ordered_tasks()
        
        assert len(ordered) == 3
        
    def test_estimate_difficulty(self):
        """测试难度估计"""
        from continual_learning import estimate_difficulty
        from torch.utils.data import TensorDataset
        
        set_seed(42)
        model = SimpleNet()
        
        features, labels = generate_mock_data(num_samples=50)
        dataset = TensorDataset(features, labels)
        
        difficulty = estimate_difficulty(model, dataset, sample_size=30)
        
        assert 0.1 <= difficulty <= 2.0


class TestConfig:
    """测试配置"""
    
    def test_default_config(self):
        """测试默认配置"""
        from continual_learning.config import ContinualLearningConfig
        
        config = ContinualLearningConfig()
        
        assert config.random_seed == 42
        assert config.experiment_name == "continual_learning"
        assert config.training.epochs == 10
        
    def test_config_to_dict(self):
        """测试配置序列化"""
        from continual_learning.config import ContinualLearningConfig
        
        config = ContinualLearningConfig(
            experiment_name="test",
            random_seed=123
        )
        
        config_dict = config.to_dict()
        
        assert config_dict["experiment_name"] == "test"
        assert config_dict["random_seed"] == 123
        
    def test_config_from_dict(self):
        """测试配置反序列化"""
        from continual_learning.config import ContinualLearningConfig
        
        config_dict = {
            "experiment_name": "from_dict",
            "random_seed": 456,
            "training": {
                "epochs": 20,
                "batch_size": 64
            }
        }
        
        config = ContinualLearningConfig.from_dict(config_dict)
        
        assert config.experiment_name == "from_dict"
        assert config.random_seed == 456
        assert config.training.epochs == 20
        assert config.training.batch_size == 64
        
    def test_predefined_configs(self):
        """测试预定义配置"""
        from continual_learning.config import get_config, CONFIGS
        
        # 获取预定义配置
        ewc_config = get_config("ewc_benchmark")
        
        assert ewc_config is not None
        assert ewc_config.consolidation.importance_weight > 0
        
        # 检查所有预定义配置
        assert len(CONFIGS) > 0


class TestAPI:
    """测试API"""
    
    def test_create_api(self):
        """测试创建API"""
        from continual_learning import (
            IncrementalLearner,
            MemoryReplay,
            KnowledgeConsolidation,
            CurriculumLearning
        )
        from continual_learning.api import create_api
        
        set_seed(42)
        model = SimpleNet()
        
        learner = IncrementalLearner(model=model)
        replay = MemoryReplay(strategy="experience")
        consolidation = KnowledgeConsolidation(model, method="ewc")
        curriculum = CurriculumLearning(model=model)
        
        api = create_api(learner, curriculum, replay, consolidation)
        
        assert api is not None
        assert api.learner is learner
        
    def test_api_add_task(self):
        """测试API添加任务"""
        from continual_learning import (
            IncrementalLearner,
            MemoryReplay,
            KnowledgeConsolidation,
            CurriculumLearning
        )
        from continual_learning.api import create_api
        
        set_seed(42)
        model = SimpleNet()
        
        learner = IncrementalLearner(model=model)
        replay = MemoryReplay(strategy="experience")
        consolidation = KnowledgeConsolidation(model, method="ewc")
        curriculum = CurriculumLearning(model=model)
        
        api = create_api(learner, curriculum, replay, consolidation)
        
        data = {
            'features': np.random.randn(100, 50).astype(np.float32),
            'labels': np.random.randint(0, 5, 100)
        }
        
        response = api.add_task(
            task_id="task_1",
            name="Test",
            data=data,
            difficulty=1.0
        )
        
        assert response.success
        assert len(learner.tasks) == 1
        
    def test_api_train(self):
        """测试API训练"""
        from continual_learning import (
            IncrementalLearner,
            MemoryReplay,
            KnowledgeConsolidation,
            CurriculumLearning
        )
        from continual_learning.api import create_api
        
        set_seed(42)
        model = SimpleNet()
        
        learner = IncrementalLearner(model=model)
        replay = MemoryReplay(strategy="experience")
        consolidation = KnowledgeConsolidation(model, method="ewc")
        curriculum = CurriculumLearning(model=model)
        
        api = create_api(learner, curriculum, replay, consolidation)
        
        # 添加任务
        features, labels = generate_mock_data(num_samples=200)
        dataset = TensorDataset(features, labels)
        
        learner.add_task(dataset, task_id="task_1")
        curriculum.add_task(task_id="task_1", dataset=dataset, difficulty=1.0)
        
        # 训练
        response = api.train_task(task_id="task_1", epochs=2)
        
        assert response.success
        assert response.data["accuracy"] >= 0
        
    def test_api_evaluate(self):
        """测试API评估"""
        from continual_learning import (
            IncrementalLearner,
            MemoryReplay,
            KnowledgeConsolidation,
            CurriculumLearning
        )
        from continual_learning.api import create_api
        
        set_seed(42)
        model = SimpleNet()
        
        learner = IncrementalLearner(model=model)
        replay = MemoryReplay(strategy="experience")
        consolidation = KnowledgeConsolidation(model, method="ewc")
        curriculum = CurriculumLearning(model=model)
        
        api = create_api(learner, curriculum, replay, consolidation)
        
        # 评估空模型
        response = api.evaluate()
        
        assert response.success


# ==================== 集成测试 ====================

class TestIntegration:
    """集成测试"""
    
    def test_full_pipeline(self):
        """完整流程测试"""
        from continual_learning import (
            IncrementalLearner,
            MemoryReplay,
            KnowledgeConsolidation,
            CurriculumLearning
        )
        from continual_learning.api import create_api
        from continual_learning.config import ContinualLearningConfig
        
        set_seed(42)
        
        # 配置
        config = ContinualLearningConfig(
            training=TrainingConfig(epochs=3, batch_size=32)
        )
        
        # 模型
        model = SimpleNet(input_dim=50, hidden_dim=32, output_dim=5)
        
        # 组件
        replay = MemoryReplay(strategy="experience", capacity=1000)
        consolidation = KnowledgeConsolidation(model, method="ewc")
        curriculum = CurriculumLearning(model=model, strategy="difficulty")
        
        # 学习器
        learner = IncrementalLearner(
            model=model,
            memory_replay=replay,
            knowledge_consolidation=consolidation,
            config=config
        )
        
        # 添加任务
        for i in range(3):
            features, labels = generate_mock_data(num_samples=200, seed=42+i)
            dataset = TensorDataset(features, labels)
            
            learner.add_task(dataset, task_id=f"task_{i}", name=f"Task {i}")
            curriculum.add_task(
                task_id=f"task_{i}",
                dataset=dataset,
                difficulty=0.5 + i * 0.2
            )
        
        # 按课程训练
        for task in curriculum.get_ordered_tasks():
            result = learner.train(
                task_id=task.task_id,
                epochs=2,
                use_replay=True,
                use_consolidation=True
            )
            curriculum.complete_task(task.task_id, {"accuracy": result.accuracy})
        
        # 评估
        stats = learner.get_statistics()
        
        assert stats["completed_tasks"] == 3
        assert stats["average_accuracy"] > 0
        
    def test_replay_and_ewc_combined(self):
        """回放和EWC组合测试"""
        from continual_learning import (
            IncrementalLearner,
            MemoryReplay,
            EWC
        )
        from continual_learning.config import ContinualLearningConfig
        
        set_seed(42)
        
        model = SimpleNet()
        replay = MemoryReplay(strategy="importance")
        ewc = EWC(model, importance_weight=2000)
        
        learner = IncrementalLearner(
            model=model,
            memory_replay=replay,
            knowledge_consolidation=ewc,
            config=ContinualLearningConfig()
        )
        
        # 训练多个任务
        for i in range(3):
            features, labels = generate_mock_data(num_samples=200, seed=42+i)
            dataset = TensorDataset(features, labels)
            
            learner.add_task(dataset, task_id=f"task_{i}")
            
            result = learner.train(
                task_id=f"task_{i}",
                epochs=2,
                use_replay=True,
                use_consolidation=True
            )
            
            # 更新Fisher
            ewc.update_fisher(dataset=dataset)
        
        # 遗忘率应该较低
        assert learner.forgetting_ratio < 0.5


# ==================== 运行测试 ====================

if __name__ == "__main__":
    # 运行所有测试
    pytest.main([__file__, "-v"])
