"""
DeepSpeed集成 - Phase 2
"""
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from uuid import uuid4
import json

class ZeROStage(Enum):
    """ZeRO优化阶段"""
    STAGE_1 = 1  # 优化器状态分片
    STAGE_2 = 2  # 梯度分片
    STAGE_3 = 3  # 参数分片

@dataclass
class DeepSpeedConfig:
    """DeepSpeed配置"""
    train_batch_size: int = 32
    gradient_accumulation_steps: int = 1
    optimizer:
        type: str = "Adam"
        params:
            lr: float = 0.001
    scheduler:
        type: str = "WarmupLR"
        params:
            warmup_min_lr: float = 0.0
            warmup_num_steps: int = 100
    zero_optimization:
        stage: int = 2
        offload_optimizer: bool = False
        offload_param: bool = False
    fp16:
        enabled: bool = True
        loss_scale: float = 0.0
        initial_scale_power: int = 16
    gradient_clipping: float = 1.0
    steps_per_print: int = 10
    wall_clock_breakdown: bool = False

class DeepSpeedLauncher:
    """DeepSpeed启动器"""
    
    def __init__(self):
        self.config_template = {
            "train_batch_size": 32,
            "gradient_accumulation_steps": 1,
            "optimizer": {
                "type": "Adam",
                "params": {"lr": 0.001}
            },
            "scheduler": {
                "type": "WarmupLR",
                "params": {"warmup_num_steps": 100}
            },
            "zero_optimization": {
                "stage": 2,
                "offload_optimizer": False,
                "offload_param": False
            },
            "fp16": {
                "enabled": True,
                "loss_scale": 0.0,
                "initial_scale_power": 16
            },
            "gradient_clipping": 1.0
        }
    
    def create_config(
        self,
        model_size: str = "7b",
        world_size: int = 1,
        gpu_memory: int = 80
    ) -> Dict:
        """
        根据模型大小创建DeepSpeed配置
        
        模型大小参考:
        - 7B: 最小14GB GPU Memory
        - 13B: 最小26GB GPU Memory
        - 30B: 最小60GB GPU Memory
        - 65B: 最小130GB GPU Memory
        """
        configs = {
            "7b": {"stage": 2, "offload": False},
            "13b": {"stage": 2, "offload": False},
            "30b": {"stage": 3, "offload": True},
            "65b": {"stage": 3, "offload": True}
        }
        
        config = self.config_template.copy()
        model_config = configs.get(model_size, configs["7b"])
        
        # ZeRO配置
        zero_stage = model_config["stage"]
        config["zero_optimization"]["stage"] = zero_stage
        
        # 根据GPU内存调整offload
        if gpu_memory < 40:
            config["zero_optimization"]["offload_optimizer"] = True
            config["zero_optimization"]["offload_param"] = True
        elif gpu_memory < 80:
            if zero_stage >= 2:
                config["zero_optimization"]["offload_optimizer"] = True
        
        # 根据world_size调整
        if world_size > 1:
            config["gradient_accumulation_steps"] = max(1, 32 // world_size)
        
        return config
    
    def generate_launch_command(
        self,
        script: str,
        world_size: int,
        node_rank: int = 0,
        master_addr: str = "localhost",
        master_port: int = 29500,
        config_file: Optional[str] = None
    ) -> str:
        """
        生成启动命令
        
        用法:
        deepspeed --num_gpus=$ngpus \
            --num_nodes=$nnodes \
            --node_rank=$node_rank \
            --master_addr=$master_addr \
            --master_port=$master_port \
            --deepspeed_config=$config_file \
            $script
        """
        cmd = [
            "deepspeed",
            f"--num_gpus={world_size}",
            f"--num_nodes={1}",
            f"--node_rank={node_rank}",
            f"--master_addr={master_addr}",
            f"--master_port={master_port}"
        ]
        
        if config_file:
            cmd.append(f"--deepspeed_config={config_file}")
        
        cmd.append(script)
        
        return " ".join(cmd)
    
    def save_config(self, config: Dict, path: str):
        """保存配置文件"""
        with open(path, "w") as f:
            json.dump(config, f, indent=2)
    
    def get_memory_requirement(
        self,
        model_size: str,
        zero_stage: int,
        world_size: int = 1
    ) -> Dict:
        """计算内存需求"""
        base_memory = {
            "7b": 14,
            "13b": 26,
            "30b": 60,
            "65b": 130
        }
        
        memory_per_gpu = base_memory.get(model_size, 14)
        
        # ZeRO减少的内存
        if zero_stage == 1:
            memory_per_gpu = memory_per_gpu * 0.6
        elif zero_stage == 2:
            memory_per_gpu = memory_per_gpu * 0.3
        elif zero_stage == 3:
            memory_per_gpu = memory_per_gpu * 0.15
        
        # 多GPU线性减少
        total_memory = memory_per_gpu * world_size
        
        return {
            "model_size_gb": base_memory.get(model_size, 14),
            "per_gpu_gb": round(memory_per_gpu, 2),
            "total_gb": round(total_memory, 2),
            "recommended_gpus": max(1, world_size)
        }

# DeepSpeed启动器实例
deepspeed_launcher = DeepSpeedLauncher()
