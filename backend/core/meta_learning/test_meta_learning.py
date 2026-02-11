"""
元学习框架测试用例
"""

import torch
import torch.nn as nn
import pytest
import numpy as np
from .learner import MetaLearner, MAML, PrototypicalNetworks, Reptile
from .few_shot_learner import FewShotLearner, NWayKShotDataset, ContrastiveLoss
from .task_generator import TaskGenerator, Task, DifficultyGrader, TaskSimilarity
from .adaptation_engine import AdaptationEngine, FastAdapter, GradientAdapter
from .config import MetaLearningConfig


class TestMAML:
    """MAML算法测试"""
    
    def test_maml_initialization(self):
        """测试MAML初始化"""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5)
        )
        maml = MAML(model, inner_lr=0.01, outer_lr=0.001)
        assert maml.inner_lr == 0.01
        assert maml.outer_lr == 0.001
    
    def test_maml_inner_update(self):
        """测试MAML内层更新"""
        model = nn.Linear(10, 5)
        maml = MAML(model, inner_lr=0.1)
        
        support_x = torch.randn(5, 10)
        support_y = torch.tensor([0, 1, 2, 3, 4])
        
        adapted_model = maml.inner_update(support_x, support_y, torch.device("cpu"))
        
        assert adapted_model is not None
        assert isinstance(adapted_model, nn.Module)
    
    def test_maml_forward(self):
        """测试MAML前向传播"""
        model = nn.Linear(10, 5)
        maml = MAML(model)
        
        x = torch.randn(3, 10)
        output = maml(x)
        
        assert output.shape == (3, 5)


class TestPrototypicalNetworks:
    """原型网络测试"""
    
    def test_prototypical_initialization(self):
        """测试原型网络初始化"""
        encoder = nn.Linear(64, 32)
        proto = PrototypicalNetworks(encoder, distance_metric="euclidean")
        assert proto.distance_metric == "euclidean"
    
    def test_prototypical_prototypes(self):
        """测试原型计算"""
        encoder = nn.Linear(64, 32)
        proto = PrototypicalNetworks(encoder)
        
        embeddings = torch.randn(10, 32)
        labels = torch.tensor([0, 0, 1, 1, 2, 2])
        n_way = 3
        
        prototypes = proto.get_prototypes(embeddings, labels, n_way)
        
        assert prototypes.shape == (n_way, 32)
    
    def test_prototypical_classify(self):
        """测试原型分类"""
        encoder = nn.Linear(64, 32)
        proto = PrototypicalNetworks(encoder)
        
        embeddings = torch.randn(10, 32)
        prototypes = torch.randn(5, 32)
        
        logits = proto.classify(embeddings, prototypes)
        
        assert logits.shape == (10, 5)


class TestReptile:
    """Reptile算法测试"""
    
    def test_reptile_initialization(self):
        """测试Reptile初始化"""
        model = nn.Linear(10, 5)
        reptile = Reptile(model, inner_lr=0.1, outer_lr=0.001)
        assert reptile.inner_lr == 0.1
    
    def test_reptile_inner_update(self):
        """测试Reptile内层更新"""
        model = nn.Linear(10, 5)
        reptile = Reptile(model, inner_lr=0.1)
        
        support_x = torch.randn(5, 10)
        support_y = torch.tensor([0, 1, 2, 3, 4])
        
        result = reptile.inner_update(support_x, support_y, torch.device("cpu"), steps=3)
        
        assert "initial" in result
        assert "final" in result
        assert "model" in result


class TestMetaLearner:
    """元学习器主类测试"""
    
    def test_meta_learner_initialization(self):
        """测试元学习器初始化"""
        meta = MetaLearner(algorithm="maml", n_way=5, k_shot=1)
        assert meta.algorithm == "maml"
        assert meta.n_way == 5
    
    def test_meta_learner_algorithms(self):
        """测试不同算法初始化"""
        for algo in ["maml", "prototypical", "reptile"]:
            meta = MetaLearner(algorithm=algo)
            assert meta.algorithm == algo
    
    def test_meta_learner_adapt(self):
        """测试适应功能"""
        meta = MetaLearner(algorithm="maml", n_way=5, k_shot=1)
        
        support_x = torch.randn(5, 1, 28, 28)
        support_y = torch.tensor([0, 1, 2, 3, 4])
        
        adapted = meta.adapt(support_x, support_y, steps=3)
        
        assert adapted is not None
    
    def test_meta_learner_evaluate(self):
        """测试评估功能"""
        meta = MetaLearner(algorithm="maml", n_way=5, k_shot=1)
        
        query_x = torch.randn(15, 1, 28, 28)
        query_y = torch.randint(0, 5, (15,))
        
        accuracy = meta.evaluate(meta.model, query_x, query_y)
        
        assert 0.0 <= accuracy <= 1.0


class TestFewShotLearner:
    """少样本学习器测试"""
    
    def test_few_shot_initialization(self):
        """测试少样本学习器初始化"""
        learner = FewShotLearner(n_way=5, k_shot=1, embedding_dim=64)
        assert learner.n_way == 5
        assert learner.k_shot == 1
    
    def test_few_shot_prototypes(self):
        """测试原型计算"""
        learner = FewShotLearner(n_way=5, k_shot=1)
        
        support_x = torch.randn(5, 1, 28, 28)
        support_y = torch.tensor([0, 1, 2, 3, 4])
        
        prototypes, accuracy = learner.adapt(support_x, support_y)
        
        assert prototypes.shape[0] == 5
    
    def test_contrastive_loss(self):
        """测试对比损失"""
        loss_fn = ContrastiveLoss(temperature=0.07)
        
        query_emb = torch.randn(8, 64)
        support_emb = torch.randn(5, 64)
        query_labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 4])
        support_labels = torch.tensor([0, 1, 2, 3, 4])
        
        loss = loss_fn(query_emb, support_emb, query_labels, support_labels)
        
        assert loss.item() >= 0


class TestTaskGenerator:
    """任务生成器测试"""
    
    def test_task_generator_initialization(self):
        """测试任务生成器初始化"""
        dataset = {
            0: torch.randn(20, 1, 28, 28),
            1: torch.randn(20, 1, 28, 28),
            2: torch.randn(20, 1, 28, 28),
        }
        labels = {
            0: torch.zeros(20, dtype=torch.long),
            1: torch.ones(20, dtype=torch.long),
            2: torch.full((20,), 2, dtype=torch.long),
        }
        
        generator = TaskGenerator(dataset, labels)
        
        assert len(generator.classes) == 3
    
    def test_sample_task(self):
        """测试任务采样"""
        dataset = {
            i: torch.randn(20, 1, 28, 28) for i in range(10)
        }
        labels = {
            i: torch.full((20,), i, dtype=torch.long) for i in range(10)
        }
        
        generator = TaskGenerator(dataset, labels)
        task = generator.sample_task(n_way=5, k_shot=1)
        
        assert task.task_id.startswith("task_")
        assert len(task.support_x) == 5
        assert len(task.query_x) == 15
    
    def test_difficulty_grader(self):
        """测试难度分级"""
        grader = DifficultyGrader()
        
        data = torch.randn(10, 1, 28, 28)
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
        
        difficulty = grader.compute_difficulty(data, labels)
        
        assert 0.0 <= difficulty <= 1.0
    
    def test_task_similarity(self):
        """测试任务相似度"""
        similarity = TaskSimilarity(method="wasserstein")
        
        dataset = {
            0: torch.randn(20, 1, 28, 28),
            1: torch.randn(20, 1, 28, 28),
        }
        labels = {
            0: torch.zeros(20, dtype=torch.long),
            1: torch.ones(20, dtype=torch.long),
        }
        
        generator = TaskGenerator(dataset, labels)
        
        task1 = generator.sample_task(n_way=2, k_shot=1)
        task2 = generator.sample_task(n_way=2, k_shot=1)
        
        sim = similarity.compute_similarity(task1, task2)
        
        assert 0.0 <= sim <= 1.0


class TestAdaptationEngine:
    """适应引擎测试"""
    
    def test_adaptation_engine_initialization(self):
        """测试适应引擎初始化"""
        engine = AdaptationEngine(strategy="auto")
        assert engine.strategy == "auto"
    
    def test_gradient_adapter(self):
        """测试梯度适配器"""
        adapter = GradientAdapter(lr=0.1)
        model = nn.Linear(10, 5)
        
        support_x = torch.randn(5, 10)
        support_y = torch.tensor([0, 1, 2, 3, 4])
        
        adapted = adapter.adapt(model, support_x, support_y, torch.device("cpu"))
        
        assert adapted is not None
    
    def test_fast_adapter(self):
        """测试快速适配器"""
        adapter = FastAdapter(strategy="maml", steps=5)
        model = nn.Linear(10, 5)
        
        support_x = torch.randn(5, 10)
        support_y = torch.tensor([0, 1, 2, 3, 4])
        query_x = torch.randn(10, 10)
        query_y = torch.randint(0, 5, (10,))
        
        result = adapter.adapt(model, support_x, support_y, query_x, query_y)
        
        assert "model" in result
        assert "accuracy" in result
    
    def test_adaptation_batch(self):
        """测试批量适应"""
        engine = AdaptationEngine(strategy="maml")
        model = nn.Linear(10, 5)
        
        task_batch = [
            {
                "support_x": torch.randn(5, 10),
                "support_y": torch.tensor([0, 1, 2, 3, 4]),
                "query_x": torch.randn(10, 10),
                "query_y": torch.randint(0, 5, (10,))
            }
            for _ in range(3)
        ]
        
        results = engine.batch_adapt(model, task_batch)
        
        assert len(results) == 3


class TestConfig:
    """配置测试"""
    
    def test_config_initialization(self):
        """测试配置初始化"""
        config = MetaLearningConfig(
            n_way=5,
            k_shot=1,
            algorithm="maml",
            meta_lr=0.001
        )
        
        assert config.n_way == 5
        assert config.algorithm == "maml"
    
    def test_config_to_dict(self):
        """测试配置转字典"""
        config = MetaLearningConfig(n_way=5, k_shot=1)
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict["n_way"] == 5


class TestIntegration:
    """集成测试"""
    
    def test_full_workflow(self):
        """测试完整工作流程"""
        # 1. 创建配置
        config = MetaLearningConfig(
            n_way=5,
            k_shot=1,
            algorithm="maml",
            meta_lr=0.001
        )
        
        # 2. 初始化元学习器
        meta = MetaLearner(
            algorithm=config.algorithm,
            n_way=config.n_way,
            k_shot=config.k_shot,
            outer_lr=config.meta_lr
        )
        
        # 3. 创建任务数据
        support_x = torch.randn(5, 1, 28, 28)
        support_y = torch.tensor([0, 1, 2, 3, 4])
        query_x = torch.randn(15, 1, 28, 28)
        query_y = torch.randint(0, 5, (15,))
        
        # 4. 适应任务
        adapted = meta.adapt(support_x, support_y)
        
        # 5. 评估
        accuracy = meta.evaluate(adapted, query_x, query_y)
        
        assert 0.0 <= accuracy <= 1.0
    
    def test_all_algorithms(self):
        """测试所有算法"""
        for algo in ["maml", "prototypical", "reptile"]:
            meta = MetaLearner(algorithm=algo, n_way=5, k_shot=1)
            
            support_x = torch.randn(5, 1, 28, 28)
            support_y = torch.tensor([0, 1, 2, 3, 4])
            
            adapted = meta.adapt(support_x, support_y)
            
            assert adapted is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
