"""
元学习框架使用示例
"""

import torch
import torch.nn as nn
from .learner import MetaLearner
from .few_shot_learner import FewShotLearner
from .task_generator import TaskGenerator, Task
from .adaptation_engine import AdaptationEngine
from .config import MetaLearningConfig


def example_maml():
    """MAML算法示例"""
    print("=" * 50)
    print("MAML Algorithm Example")
    print("=" * 50)
    
    # 初始化元学习器
    meta = MetaLearner(
        algorithm="maml",
        n_way=5,
        k_shot=1,
        inner_lr=0.01,
        outer_lr=0.001
    )
    
    print(f"Device: {meta.device}")
    print(f"Algorithm: {meta.algorithm}")
    print("MetaLearner initialized successfully!")
    
    return meta


def example_prototypical():
    """原型网络示例"""
    print("=" * 50)
    print("Prototypical Networks Example")
    print("=" * 50)
    
    # 创建编码器
    encoder = nn.Sequential(
        nn.Conv2d(1, 64, 3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(64, 64, 3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(64 * 7 * 7, 128),
        nn.ReLU(),
        nn.Linear(128, 64)
    )
    
    # 初始化原型网络
    proto_net = MetaLearner(
        algorithm="prototypical",
        model=encoder,
        n_way=5,
        k_shot=1
    )
    
    print("Prototypical Networks initialized successfully!")
    return proto_net


def example_reptile():
    """Reptile算法示例"""
    print("=" * 50)
    print("Reptile Algorithm Example")
    print("=" * 50)
    
    meta = MetaLearner(
        algorithm="reptile",
        n_way=5,
        k_shot=1,
        inner_lr=0.1,
        outer_lr=0.001
    )
    
    print("Reptile initialized successfully!")
    return meta


def example_few_shot_learning():
    """少样本学习示例"""
    print("=" * 50)
    print("Few-Shot Learning Example")
    print("=" * 50)
    
    # 初始化少样本学习器
    few_shot = FewShotLearner(
        n_way=5,
        k_shot=1,
        embedding_dim=64,
        use_contrastive=True,
        use_metric=True
    )
    
    print(f"Embedding dimension: {few_shot.embedding_dim}")
    print(f"N-way: {few_shot.n_way}, K-shot: {few_shot.k_shot}")
    
    # 模拟支持集和查询集
    support_x = torch.randn(5, 1, 28, 28)  # 5-way 1-shot
    support_y = torch.tensor([0, 1, 2, 3, 4])
    query_x = torch.randn(15, 1, 28, 28)
    query_y = torch.tensor([0] * 3 + [1] * 3 + [2] * 3 + [3] * 3 + [4] * 3)
    
    # 适应任务
    prototypes, accuracy = few_shot.adapt(support_x, support_y, query_x, query_y)
    
    print(f"Adaptation accuracy: {accuracy:.4f}")
    print("Few-shot learning example completed!")
    
    return few_shot


def example_task_generator():
    """任务生成器示例"""
    print("=" * 50)
    print("Task Generator Example")
    print("=" * 50)
    
    # 创建模拟数据集
    dataset = {
        0: torch.randn(20, 1, 28, 28),
        1: torch.randn(20, 1, 28, 28),
        2: torch.randn(20, 1, 28, 28),
        3: torch.randn(20, 1, 28, 28),
        4: torch.randn(20, 1, 28, 28),
        5: torch.randn(20, 1, 28, 28),
        6: torch.randn(20, 1, 28, 28),
        7: torch.randn(20, 1, 28, 28),
    }
    
    labels = {
        0: torch.zeros(20, dtype=torch.long),
        1: torch.ones(20, dtype=torch.long),
        2: torch.full((20,), 2, dtype=torch.long),
        3: torch.full((20,), 3, dtype=torch.long),
        4: torch.full((20,), 4, dtype=torch.long),
        5: torch.full((20,), 5, dtype=torch.long),
        6: torch.full((20,), 6, dtype=torch.long),
        7: torch.full((20,), 7, dtype=torch.long),
    }
    
    # 创建任务生成器
    generator = TaskGenerator(dataset, labels)
    
    # 采样任务
    task = generator.sample_task(n_way=5, k_shot=1)
    print(f"Task ID: {task.task_id}")
    print(f"Difficulty: {task.difficulty:.4f}")
    print(f"Support set size: {len(task.support_x)}")
    print(f"Query set size: {len(task.query_x)}")
    
    # 采样批量任务
    tasks = generator.sample_task_batch(
        batch_size=4,
        n_way=5,
        k_shot=1,
        difficulty_range=(0.3, 0.7)
    )
    
    print(f"Generated {len(tasks)} tasks")
    print("Task generator example completed!")
    
    return generator


def example_adaptation_engine():
    """适应引擎示例"""
    print("=" * 50)
    print("Adaptation Engine Example")
    print("=" * 50)
    
    # 创建基础模型
    base_model = nn.Sequential(
        nn.Conv2d(1, 64, 3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(64, 64, 3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(64 * 7 * 7, 10)
    )
    
    # 创建适应引擎
    engine = AdaptationEngine(strategy="auto")
    
    # 模拟任务数据
    support_x = torch.randn(5, 1, 28, 28)
    support_y = torch.tensor([0, 1, 2, 3, 4])
    query_x = torch.randn(15, 1, 28, 28)
    query_y = torch.tensor([0] * 3 + [1] * 3 + [2] * 3 + [3] * 3 + [4] * 3)
    
    task_data = {
        "support_x": support_x,
        "support_y": support_y,
        "query_x": query_x,
        "query_y": query_y,
        "metadata": {"task_id": "example_task"}
    }
    
    # 适应任务
    result = engine.adapt(base_model, task_data)
    
    print(f"Selected strategy: {result['strategy']}")
    print(f"Adaptation accuracy: {result['accuracy']:.4f}")
    print("Adaptation engine example completed!")
    
    return engine


def example_config():
    """配置示例"""
    print("=" * 50)
    print("Configuration Example")
    print("=" * 50)
    
    # 创建默认配置
    config = MetaLearningConfig(
        n_way=5,
        k_shot=1,
        algorithm="maml",
        meta_lr=0.001,
        inner_lr=0.01,
        outer_lr=0.001,
        meta_epochs=1000,
        adaptation_steps=5
    )
    
    print("Configuration created:")
    print(f"  N-way: {config.n_way}")
    print(f"  K-shot: {config.k_shot}")
    print(f"  Algorithm: {config.algorithm}")
    print(f"  Meta LR: {config.meta_lr}")
    print(f"  Meta epochs: {config.meta_epochs}")
    
    # 保存配置
    config.save("/tmp/meta_learning_config.json")
    print("Configuration saved to /tmp/meta_learning_config.json")
    
    # 加载配置
    loaded_config = MetaLearningConfig.load("/tmp/meta_learning_config.json")
    print(f"Loaded configuration: {loaded_config.algorithm}")
    
    return config


def run_all_examples():
    """运行所有示例"""
    print("\n" + "=" * 60)
    print("META LEARNING FRAMEWORK - ALL EXAMPLES")
    print("=" * 60 + "\n")
    
    example_maml()
    print()
    
    example_prototypical()
    print()
    
    example_reptile()
    print()
    
    example_few_shot_learning()
    print()
    
    example_task_generator()
    print()
    
    example_adaptation_engine()
    print()
    
    example_config()
    
    print("\n" + "=" * 60)
    print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_examples()
