"""
持续学习系统示例

提供使用示例和使用场景
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from typing import List, Dict, Tuple


# ==================== 示例模型 ====================

class SimpleMLP(nn.Module):
    """简单多层感知机"""
    
    def __init__(self, input_dim: int = 784, hidden_dims: List[int] = [256, 128], output_dim: int = 10):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            prev_dim = hidden_dim
            
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.network(x)


class ConvNet(nn.Module):
    """简单卷积网络"""
    
    def __init__(self, input_channels: int = 1, output_dim: int = 10):
        super().__init__()
        
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, output_dim)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# ==================== 数据生成器 ====================

def generate_task_data(
    task_id: int,
    num_samples: int = 500,
    input_dim: int = 100,
    num_classes: int = 2,
    noise: float = 0.1,
    seed: int = None
) -> Dict:
    """
    生成模拟任务数据
    
    Args:
        task_id: 任务ID
        num_samples: 样本数
        input_dim: 输入维度
        num_classes: 类别数
        noise: 噪声水平
        seed: 随机种子
        
    Returns:
        包含features和labels的字典
    """
    if seed is not None:
        np.random.seed(seed + task_id)
        torch.manual_seed(seed + task_id)
    
    # 生成类别相关的中心点
    centers = np.random.randn(num_classes, input_dim) * 2
    
    # 生成样本
    features = []
    labels = []
    
    for i in range(num_samples):
        # 随机选择类别
        label = np.random.randint(0, num_classes)
        
        # 在中心点附近生成样本
        feature = centers[label] + np.random.randn(input_dim) * noise
        
        features.append(feature)
        labels.append(label)
    
    return {
        'features': np.array(features, dtype=np.float32),
        'labels': np.array(labels, dtype=np.int64)
    }


def generate_split_mnist(
    task_id: int,
    num_samples: int = 1000,
    permute_features: bool = False,
    seed: int = 42
) -> Dict:
    """
    生成Split MNIST风格的任务数据
    
    每个任务学习区分两个数字
    """
    np.random.seed(seed + task_id)
    torch.manual_seed(seed + task_id)
    
    # 定义数字对
    digit_pairs = [
        (0, 1), (2, 3), (4, 5), (6, 7), (8, 9),
        (0, 2), (1, 3), (4, 6), (5, 7), (8, 9)
    ]
    
    if task_id >= len(digit_pairs):
        task_id = task_id % len(digit_pairs)
    
    digit1, digit2 = digit_pairs[task_id]
    
    # 模拟MNIST数据（实际应用中加载真实MNIST）
    features = np.random.randn(num_samples, 784) * 0.1
    labels = np.random.randint(0, 2, num_samples)
    
    # 添加类别相关的信号
    for i in range(num_samples):
        if labels[i] == 0:
            features[i, :100] += 0.5
        else:
            features[i, 100:200] += 0.5
    
    # 特征变换
    if permute_features:
        perm = np.random.permutation(784)
        features = features[:, perm]
    
    return {
        'features': features.astype(np.float32),
        'labels': labels.astype(np.int64)
    }


# ==================== 使用示例 ====================

def example_basic_usage():
    """基本使用示例"""
    print("=" * 60)
    print("Basic Usage Example")
    print("=" * 60)
    
    from continual_learning import (
        IncrementalLearner, 
        MemoryReplay, 
        KnowledgeConsolidation,
        CurriculumLearning
    )
    from continual_learning.config import ContinualLearningConfig
    
    # 创建配置
    config = ContinualLearningConfig(
        experiment_name="basic_example",
        random_seed=42
    )
    
    # 创建模型
    model = SimpleMLP(input_dim=100, hidden_dims=[64, 32], output_dim=10)
    
    # 创建系统组件
    memory_replay = MemoryReplay(strategy="experience", capacity=1000)
    knowledge_consolidation = KnowledgeConsolidation(model, method="ewc", config=config.consolidation)
    
    # 创建增量学习器
    learner = IncrementalLearner(
        model=model,
        memory_replay=memory_replay,
        knowledge_consolidation=knowledge_consolidation,
        config=config
    )
    
    # 创建课程学习
    curriculum = CurriculumLearning(model=model, strategy="difficulty")
    
    # 生成并添加任务
    for task_id in range(3):
        data = generate_task_data(task_id, num_samples=200, input_dim=100, num_classes=10)
        
        learner.add_task(data, task_id=f"task_{task_id}", name=f"Task {task_id}")
        curriculum.add_task(
            task_id=f"task_{task_id}",
            dataset=learner.tasks[f"task_{task_id}"].dataset,
            difficulty=0.5 + task_id * 0.2
        )
    
    print(f"Added {len(learner.tasks)} tasks")
    
    # 按课程训练
    ordered_tasks = curriculum.get_ordered_tasks()
    
    for task in ordered_tasks:
        print(f"\nTraining {task.name}...")
        result = learner.train(
            task_id=task.task_id,
            epochs=5,
            batch_size=32
        )
        print(f"  Loss: {result.loss:.4f}, Accuracy: {result.accuracy:.4f}")
    
    # 评估
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    
    avg_accuracy = 0
    for task_id, task in learner.tasks.items():
        accuracy = learner._evaluate(task.dataset)
        avg_accuracy += accuracy
        print(f"{task.name}: {accuracy:.4f}")
    
    print(f"\nAverage Accuracy: {avg_accuracy / len(learner.tasks):.4f}")
    print(f"Forgetting Ratio: {learner.forgetting_ratio:.4f}")


def example_with_ewc():
    """EWC示例"""
    print("\n" + "=" * 60)
    print("EWC Example")
    print("=" * 60)
    
    from continual_learning import IncrementalLearner, MemoryReplay, KnowledgeConsolidation, EWC
    from continual_learning.config import ContinualLearningConfig
    
    config = ContinualLearningConfig()
    model = SimpleMLP(input_dim=100, hidden_dims=[64, 32], output_dim=10)
    
    # 创建EWC
    ewc = EWC(model, importance_weight=5000)
    
    # 创建学习器
    learner = IncrementalLearner(
        model=model,
        knowledge_consolidation=ewc,
        config=config
    )
    
    # 添加任务
    for task_id in range(5):
        data = generate_task_data(task_id, num_samples=300, input_dim=100, num_classes=10)
        learner.add_task(data, task_id=f"task_{task_id}")
    
    # 训练每个任务
    for task_id in learner.task_order:
        print(f"\nTraining {task_id} with EWC...")
        
        result = learner.train(task_id, epochs=5)
        
        # 更新Fisher信息
        ewc.update_fisher(dataset=learner.tasks[task_id].dataset)
        
        print(f"  Accuracy: {result.accuracy:.4f}")
    
    print(f"\nFinal Forgetting Ratio: {learner.forgetting_ratio:.4f}")


def example_with_replay():
    """经验回放示例"""
    print("\n" + "=" * 60)
    print("Experience Replay Example")
    print("=" * 60)
    
    from continual_learning import IncrementalLearner, MemoryReplay
    from continual_learning.config import ContinualLearningConfig
    
    config = ContinualLearningConfig()
    model = SimpleMLP(input_dim=100, hidden_dims=[64, 32], output_dim=10)
    
    # 创建经验回放
    replay = MemoryReplay(strategy="experience", capacity=5000)
    
    # 创建学习器
    learner = IncrementalLearner(
        model=model,
        memory_replay=replay,
        config=config
    )
    
    # 添加并训练任务
    for task_id in range(5):
        data = generate_task_data(task_id, num_samples=300, input_dim=100, num_classes=10)
        learner.add_task(data, task_id=f"task_{task_id}")
        
        print(f"\nTraining {task_id} with Experience Replay...")
        
        result = learner.train(
            task_id=f"task_{task_id}",
            epochs=5,
            use_replay=True,
            use_consolidation=False
        )
        
        print(f"  Accuracy: {result.accuracy:.4f}")
        print(f"  Replay Buffer Size: {len(replay)}")


def example_with_curriculum():
    """课程学习示例"""
    print("\n" + "=" * 60)
    print("Curriculum Learning Example")
    print("=" * 60)
    
    from continual_learning import (
        IncrementalLearner, 
        CurriculumLearning,
        auto_generate_curriculum
    )
    from continual_learning.config import ContinualLearningConfig
    
    config = ContinualLearningConfig()
    model = SimpleMLP(input_dim=100, hidden_dims=[64, 32], output_dim=10)
    
    learner = IncrementalLearner(model=model, config=config)
    curriculum = CurriculumLearning(model=model, strategy="progressive")
    
    # 创建难度不同的任务
    task_configs = [
        (0, 200, 0.3),  # 简单任务
        (1, 300, 0.5),
        (2, 400, 0.7),  # 困难任务
        (3, 250, 0.4),
        (4, 350, 0.6)
    ]
    
    for task_id, num_samples, difficulty in task_configs:
        data = generate_task_data(task_id, num_samples, input_dim=100, num_classes=10)
        
        learner.add_task(data, task_id=f"task_{task_id}", name=f"Task {task_id}")
        curriculum.add_task(
            task_id=f"task_{task_id}",
            dataset=learner.tasks[f"task_{task_id}"].dataset,
            difficulty=difficulty
        )
    
    # 获取排序后的任务
    ordered = curriculum.get_ordered_tasks()
    
    print("Task Order (by difficulty):")
    for i, task in enumerate(ordered):
        print(f"  {i+1}. {task.name} (difficulty: {task.difficulty:.2f})")
    
    # 按课程训练
    for task in ordered:
        result = learner.train(task.task_id, epochs=5)
        curriculum.complete_task(task.task_id, {"accuracy": result.accuracy})
        print(f"\n{task.name}: Accuracy = {result.accuracy:.4f}")


def example_api_usage():
    """API使用示例"""
    print("\n" + "=" * 60)
    print("API Usage Example")
    print("=" * 60)
    
    from continual_learning import (
        IncrementalLearner,
        MemoryReplay,
        KnowledgeConsolidation,
        CurriculumLearning
    )
    from continual_learning.api import create_api
    from continual_learning.config import ContinualLearningConfig
    
    config = ContinualLearningConfig()
    model = SimpleMLP(input_dim=100, hidden_dims=[64, 32], output_dim=10)
    
    learner = IncrementalLearner(model=model, config=config)
    memory_replay = MemoryReplay(strategy="experience")
    knowledge_consolidation = KnowledgeConsolidation(model, method="ewc")
    curriculum = CurriculumLearning(model=model, strategy="difficulty")
    
    # 创建API
    api = create_api(learner, curriculum, memory_replay, knowledge_consolidation)
    
    # 添加任务
    for i in range(3):
        data = generate_task_data(i, num_samples=200, input_dim=100)
        response = api.add_task(
            task_id=f"task_{i}",
            name=f"Task {i}",
            data=data,
            difficulty=0.5
        )
        print(f"Add Task: {response.message}")
    
    # 获取任务列表
    response = api.get_tasks()
    print(f"Total Tasks: {len(response.data['tasks'])}")
    
    # 训练任务
    for task in response.data['tasks']:
        train_response = api.train_task(task['task_id'], epochs=3)
        print(f"Train {task['task_id']}: {train_response.message}")
    
    # 评估
    eval_response = api.evaluate()
    print(f"\nAverage Accuracy: {eval_response.data['average_accuracy']:.4f}")
    
    # 获取统计
    stats_response = api.get_statistics()
    print(f"Forgetting Ratio: {stats_response.data['learner']['forgetting_ratio']:.4f}")


def example_multiple_strategies():
    """多种策略对比示例"""
    print("\n" + "=" * 60)
    print("Multiple Strategies Comparison")
    print("=" * 60)
    
    from continual_learning import IncrementalLearner, MemoryReplay, KnowledgeConsolidation
    from continual_learning.config import ContinualLearningConfig
    
    strategies = ["ewc", "si", "regularization"]
    results = {}
    
    for strategy in strategies:
        print(f"\n--- Testing {strategy.upper()} ---")
        
        config = ContinualLearningConfig()
        model = SimpleMLP(input_dim=100, hidden_dims=[64, 32], output_dim=10)
        
        replay = MemoryReplay(strategy="experience")
        consolidation = KnowledgeConsolidation(model, method=strategy)
        
        learner = IncrementalLearner(
            model=model,
            memory_replay=replay,
            knowledge_consolidation=consolidation,
            config=config
        )
        
        # 添加任务
        for task_id in range(5):
            data = generate_task_data(task_id, num_samples=300, input_dim=100, num_classes=10)
            learner.add_task(data, task_id=f"task_{task_id}")
        
        # 训练
        accuracies = []
        for task_id in learner.task_order:
            result = learner.train(task_id, epochs=3)
            accuracies.append(result.accuracy)
        
        avg_acc = sum(accuracies) / len(accuracies)
        forgetting = learner.forgetting_ratio
        
        results[strategy] = {"avg_accuracy": avg_acc, "forgetting": forgetting}
        
        print(f"  Average Accuracy: {avg_acc:.4f}")
        print(f"  Forgetting: {forgetting:.4f}")
    
    # 对比结果
    print("\n" + "=" * 60)
    print("Comparison Results")
    print("=" * 60)
    
    for strategy, metrics in results.items():
        print(f"{strategy.upper()}: Acc={metrics['avg_accuracy']:.4f}, Forgetting={metrics['forgetting']:.4f}")


def run_all_examples():
    """运行所有示例"""
    print("\n" + "#" * 60)
    print("# Continual Learning System Examples")
    print("#" * 60 + "\n")
    
    try:
        example_basic_usage()
        example_with_ewc()
        example_with_replay()
        example_with_curriculum()
        example_api_usage()
        example_multiple_strategies()
        
        print("\n" + "#" * 60)
        print("# All Examples Completed Successfully!")
        print("#" * 60)
        
    except ImportError as e:
        print(f"Import Error: {e}")
        print("Please make sure all modules are properly installed.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_examples()
