# AI Platform v2.0 Phase 2 - Pipeline模板
"""
预定义的Pipeline模板
"""

# 预训练模板
PRETRAIN_TEMPLATE = {
    "name": "pretrain",
    "description": "预训练流程",
    "steps": [
        {
            "id": "data_collection",
            "name": "数据收集",
            "type": "data",
            "config": {"action": "collect"},
            "depends_on": []
        },
        {
            "id": "data_validation",
            "name": "数据验证",
            "type": "data",
            "config": {"action": "validate"},
            "depends_on": ["data_collection"]
        },
        {
            "id": "data_preprocessing",
            "name": "数据预处理",
            "type": "data",
            "config": {"action": "preprocess"},
            "depends_on": ["data_validation"]
        },
        {
            "id": "tokenization",
            "name": "分词处理",
            "type": "data",
            "config": {"action": "tokenize"},
            "depends_on": ["data_preprocessing"]
        },
        {
            "id": "pretrain_model",
            "name": "模型预训练",
            "type": "training",
            "config": {
                "training_type": "pretrain",
                "model_size": "7b"
            },
            "depends_on": ["tokenization"]
        },
        {
            "id": "evaluate_model",
            "name": "模型评估",
            "type": "evaluation",
            "config": {"tasks": ["lm_eval"]},
            "depends_on": ["pretrain_model"]
        }
    ]
}

# 微调模板
FINETUNE_TEMPLATE = {
    "name": "finetune",
    "description": "微调流程",
    "steps": [
        {
            "id": "data_preparation",
            "name": "数据准备",
            "type": "data",
            "config": {"action": "prepare_finetune"},
            "depends_on": []
        },
        {
            "id": "load_base_model",
            "name": "加载基础模型",
            "type": "training",
            "config": {"action": "load_base"},
            "depends_on": ["data_preparation"]
        },
        {
            "id": "finetune_model",
            "name": "微调模型",
            "type": "training",
            "config": {
                "training_type": "finetune",
                "method": "lora"
            },
            "depends_on": ["load_base_model"]
        },
        {
            "id": "evaluate_finetuned",
            "name": "评估微调模型",
            "type": "evaluation",
            "config": {"tasks": ["squad", "glue"]},
            "depends_on": ["finetune_model"]
        }
    ]
}

# 评估模板
EVALUATION_TEMPLATE = {
    "name": "evaluation",
    "description": "评估流程",
    "steps": [
        {
            "id": "load_model",
            "name": "加载模型",
            "type": "training",
            "config": {"action": "load"},
            "depends_on": []
        },
        {
            "id": "run_evaluation",
            "name": "执行评估",
            "type": "evaluation",
            "config": {"tasks": ["all"]},
            "depends_on": ["load_model"]
        },
        {
            "id": "generate_report",
            "name": "生成报告",
            "type": "shell",
            "command": "python scripts/generate_report.py",
            "depends_on": ["run_evaluation"]
        }
    ]
}

# 推理部署模板
INFERENCE_TEMPLATE = {
    "name": "inference",
    "description": "推理部署流程",
    "steps": [
        {
            "id": "load_model",
            "name": "加载模型",
            "type": "training",
            "config": {"action": "load"},
            "depends_on": []
        },
        {
            "id": "test_inference",
            "name": "测试推理",
            "type": "inference",
            "config": {"test_samples": 10},
            "depends_on": ["load_model"]
        },
        {
            "id": "deploy_model",
            "name": "部署模型",
            "type": "inference",
            "config": {"replicas": 2},
            "depends_on": ["test_inference"]
        },
        {
            "id": "health_check",
            "name": "健康检查",
            "type": "condition",
            "condition": lambda c: True,
            "depends_on": ["deploy_model"]
        }
    ]
}

# 自定义模板
CUSTOM_TEMPLATE = {
    "name": "custom",
    "description": "自定义流程",
    "steps": []
}

TEMPLATES = {
    "pretrain": PRETRAIN_TEMPLATE,
    "finetune": FINETUNE_TEMPLATE,
    "evaluation": EVALUATION_TEMPLATE,
    "inference": INFERENCE_TEMPLATE,
    "custom": CUSTOM_TEMPLATE
}

def get_template(name: str) -> Dict:
    """获取模板"""
    return TEMPLATES.get(name, CUSTOM_TEMPLATE.copy())

def list_templates() -> List[str]:
    """列出所有模板"""
    return list(TEMPLATES.keys())
