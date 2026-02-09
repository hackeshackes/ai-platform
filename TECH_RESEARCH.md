# 开源项目技术调研报告

## 概述

本报告对5个主流开源项目进行了深入技术调研，为大模型全生命周期管理平台提供技术参考。

---

## 一、MLflow 技术分析

### 1.1 项目概述
- **定位**: ML/LLM全生命周期管理平台
- **Stars**: 18,000+
- **语言**: Python为主,支持Java/R/JS
- **维护者**: Databricks

### 1.2 核心组件

```
MLflow Components
├── MLflow Tracking    # 实验跟踪
│   ├── Runs管理
│   ├── Metrics/Parameters
│   └── Artifacts
├── MLflow Models     # 模型打包
│   ├── Model Format (MLmodel)
│   └── Flavors (sklearn/pytorch/tensorflow)
├── MLflow Registry   # 模型注册
│   ├── Stage (None/Staging/Production/Archived)
│   └── Model Version
├── MLflow Projects   # 项目打包
│   └── Conda/Docker环境
└── MLflow Deployment # 模型部署
    └── Tracking Server / Local / Kubernetes
```

### 1.3 技术架构

```
┌─────────────────────────────────────────────────┐
│              MLflow UI (React)                  │
└────────────────────┬────────────────────────────┘
                     │ REST API
                     ▼
┌─────────────────────────────────────────────────┐
│              MLflow REST API Server             │
│  ┌─────────────┐  ┌─────────────┐             │
│  │ Tracking    │  │ Registry    │             │
│  │ Store       │  │ Store       │             │
│  └─────────────┘  └─────────────┘             │
└────────────────────┬────────────────────────────┘
                     │
        ┌────────────┼────────────┐
        ▼            ▼            ▼
┌────────────┐ ┌────────────┐ ┌────────────┐
│ SQLite     │ │ MySQL      │ │ PostgreSQL │
│ (开发)     │ │ (测试)     │ │ (生产)     │
└────────────┘ └────────────┘ └────────────┘
```

### 1.4 API设计

```python
# 跟踪实验
import mlflow

# 设置跟踪URI
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("my-experiment")

# 开始运行
with mlflow.start_run():
    mlflow.log_param("lr", 0.001)
    mlflow.log_metric("accuracy", 0.95)
    mlflow.log_artifact("model.pkl")
    
# 模型注册
mlflow.register_model(
    "runs:/<run_id>/model",
    "my-model"
)
```

### 1.5 优点借鉴
- ✅ 简洁的API设计
- ✅ 灵活的跟踪机制
- ✅ 丰富的集成支持
- ✅ 易于部署（单文件启动）

### 1.6 缺点/避免
- ❌ 无分布式训练支持
- ❌ UI设计偏技术人员
- ❌ 前端技术较老（React旧版）

---

## 二、Kubeflow 技术分析

### 2.1 项目概述
- **定位**: Kubernetes原生ML平台
- **Stars**: 14,000+
- **架构**: K8s Operator模式
- **维护者**: Google

### 2.2 核心组件

```
Kubeflow Ecosystem
├── Kubeflow Central Dashboard    # 统一门户
├── Kubeflow Pipelines            # 工作流编排
├── Kubeflow Katib                # 超参数搜索
├── Kubeflow Training Operator    # 分布式训练
│   ├── PyTorchJob
│   ├── TFJob (TensorFlow)
│   └── MXNetJob
├── KServe                        # 模型服务
├── Kubeflow Notebooks            # Jupyter环境
├── Kubeflow Model Registry       # 模型注册
└── Kubeflow Spark Operator       # 数据处理
```

### 2.3 架构设计

```
┌──────────────────────────────────────────────────────┐
│              Kubeflow Central Dashboard              │
│         (React + Material UI)                        │
└────────────────────┬─────────────────────────────────┘
                     │
         ┌───────────┴───────────┐
         ▼                       ▼
┌─────────────────┐   ┌─────────────────────────┐
│ Istio Ingress   │   │ OIDC Auth Service       │
│ (流量网关)       │   │ (认证授权)              │
└────────┬────────┘   └─────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────┐
│              Kubernetes Cluster                      │
│  ┌──────────────────────────────────────────────┐   │
│  │ Pipeline Pods                                │   │
│  │ - Argo Workflow (执行引擎)                   │   │
│  │ - ML Pipeline DSL (定义语言)                 │   │
│  └──────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────┐   │
│  │ Training Jobs                                │   │
│  │ - PyTorchJob CRD                             │   │
│  │ - TFJob CRD                                  │   │
│  │ - GPU Scheduling                             │   │
│  └──────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────┐   │
│  │ KServe InferenceServices                     │   │
│  │ - ModelMesh                                  │   │
│  │ - Canary Rollout                            │   │
│  └──────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────┘
```

### 2.4 训练任务定义 (CRD)

```yaml
# PyTorchJob Example
apiVersion: kubeflow.org/v1
kind: PyTorchJob
metadata:
  name: pytorch-distributed-example
spec:
  elasticPolicy:
    rdzvBackend: c10d
    minReplicas: 2
    maxReplicas: 4
  pytorchReplicaSpecs:
    Master:
      replicas: 1
      template:
        spec:
          containers:
          - name: pytorch
            image: pytorch/pytorch:latest
            resources:
              limits:
                nvidia.com/gpu: 1
    Worker:
      replicas: 2
      template:
        spec:
          containers:
          - name: pytorch
            image: pytorch/pytorch:latest
            resources:
              limits:
                nvidia.com/gpu: 1
```

### 2.5 Pipeline DSL

```python
# Kubeflow Pipeline Example
from kfp import dsl

@dsl.pipeline(name="my-pipeline")
def my_pipeline(
    learning_rate: float = 0.01,
    epochs: int = 10
):
    # 预处理组件
    preprocess = dsl.ContainerOp(
        name="preprocess",
        image="preprocess:latest",
        command=["python", "preprocess.py"]
    )
    
    # 训练组件
    train = dsl.ContainerOp(
        name="train",
        image="train:latest",
        command=["python", "train.py"],
        arguments=[
            "--lr", learning_rate,
            "--epochs", epochs
        ]
    )
    train.after(preprocess)
    
    # 评估组件
    evaluate = dsl.ContainerOp(
        name="evaluate",
        image="evaluate:latest",
        command=["python", "evaluate.py"]
    )
    evaluate.after(train)
```

### 2.6 优点借鉴
- ✅ K8s原生设计，可扩展性强
- ✅ CRD定义清晰，扩展性好
- ✅ 组件模块化，可独立使用
- ✅ GPU调度完善

### 2.7 缺点/避免
- ❌ 部署复杂，需要K8s专业知识
- ❌ 对Mac/Windows支持差
- ❌ 学习曲线陡峭
- ❌ 资源消耗大

---

## 三、Label Studio 技术分析

### 3.1 项目概述
- **定位**: 多类型数据标注平台
- **Stars**: 14,000+
- **类型**: 前后端分离Web应用
- **维护者**: HumanSignal

### 3.2 核心功能

```
Label Studio Features
├── 数据标注类型
│   ├── 图像 (分类/检测/分割)
│   ├── 音频 (转写/分类)
│   ├── 文本 (NER/情感/分类)
│   ├── 视频 (跟踪/分类)
│   └── 时序数据
├── 项目管理
│   ├── 项目创建/配置
│   ├── 标注模板设计
│   └── 任务分配
├── 标注管理
│   ├── 标注队列
│   ├── 质量控制
│   └── 标注审核
└── 导出功能
    ├── 多种格式 (JSON/COCO/VOC)
    └── 集成ML预标注
```

### 3.3 技术架构

```
┌─────────────────────────────────────────────────┐
│              React Frontend                      │
│  ┌─────────────┐  ┌─────────────┐             │
│  │ Annotation  │  │ Project     │             │
│  │ Canvas      │  │ Management  │             │
│  └─────────────┘  └─────────────┘             │
└────────────────────┬────────────────────────────┘
                     │ REST API + WebSocket
                     ▼
┌─────────────────────────────────────────────────┐
│              Django REST Framework              │
│  ┌─────────────┐  ┌─────────────┐             │
│  │ Projects    │  │ Tasks       │             │
│  │ Views       │  │ Views       │             │
│  └─────────────┘  └─────────────┘             │
│  ┌─────────────┐  ┌─────────────┐             │
│  │ Annotations │  │ ML Backend  │             │
│  │ Views       │  │ Integration │             │
│  └─────────────┘  └─────────────┘             │
└────────────────────┬────────────────────────────┘
                     │
         ┌───────────┴───────────┐
         ▼                       ▼
┌─────────────────┐   ┌─────────────────────────┐
│ PostgreSQL      │   │ S3/MinIO               │
│ (元数据)         │   │ (文件存储)              │
└─────────────────┘   └─────────────────────────┘
```

### 3.4 API设计

```python
# Label Studio API Usage
from label_studio_sdk import LabelStudio

# 初始化
ls = LabelStudio(
    base_url="http://localhost:8080",
    api_key="your-api-key"
)

# 创建项目
project = ls.projects.create(
    title="My Annotation Project",
    label_config="""
    <View>
      <Text name="text" value="$text"/>
      <Choices name="sentiment" toName="text">
        <Choice value="Positive"/>
        <Choice value="Negative"/>
      </Choices>
    </View>
    """
)

# 导入数据
ls.projects.import_tasks(
    project.id,
    data=[
        {"text": "这是一个正面评价"},
        {"text": "这是一个负面评价"}
    ]
)

# 导出标注
annotations = ls.projects.export(
    project.id,
    export_format="JSON"
)
```

### 3.5 标注配置 (XML)

```xml
<!-- Label Studio 配置示例 -->
<View>
  <!-- 文本显示 -->
  <Text name="text" value="$text"/>
  
  <!-- 分类标注 -->
  <Choices name="sentiment" toName="text" choice="single">
    <Choice value="Positive"/>
    <Choice value="Negative"/>
    <Choice value="Neutral"/>
  </Choices>
  
  <!-- NER标注 -->
  <Labels name="ner" toName="text">
    <Label value="PER" background="red"/>
    <Label value="ORG" background="blue"/>
    <Label value="LOC" background="green"/>
  </Labels>
  
  <!-- 提交按钮 -->
  <Button value="Submit">Submit</Button>
</View>
```

### 3.6 优点借鉴
- ✅ 标注配置灵活（XML模板）
- ✅ API设计清晰
- ✅ 支持多种数据格式
- ✅ 支持ML预标注

### 3.7 缺点/避免
- ❌ 前端技术较老
- ❌ Django框架较重
- ❌ 无分布式训练能力
- ❌ 标注效率可优化

---

## 四、Ollama 技术分析

### 4.1 项目概述
- **定位**: 本地LLM运行工具
- **Stars**: 85,000+
- **特点**: 极简体验、跨平台
- **维护者**: Ollama团队

### 4.2 核心功能

```
Ollama Features
├── 模型管理
│   ├── 模型库 (Llama/Gemma/Mistral等)
│   ├── 模型下载/更新
│   └── 自定义模型 (Modelfile)
├── 本地推理
│   ├── REST API
│   ├── 命令行交互
│   └── 多模型并发
├── 跨平台支持
│   ├── macOS (Apple Silicon)
│   ├── Linux
│   ├── Windows (WSL)
│   └── Docker
└── 高级功能
    ├── GPU加速
    ├── 内存优化
    └── 多模态支持
```

### 4.3 技术架构

```
┌─────────────────────────────────────────────────┐
│              CLI / REST API                     │
│  ┌─────────────┐  ┌─────────────┐             │
│  │ Command     │  │ HTTP Server │             │
│  │ Parser      │  │ (Gorilla)   │             │
│  └─────────────┘  └─────────────┘             │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│              Llama.cpp Backend                  │
│  ┌─────────────┐  ┌─────────────┐             │
│  │ Model Loader│  │ KV Cache    │             │
│  │ (GGUF格式)   │  │ Management  │             │
│  └─────────────┘  └─────────────┘             │
│  ┌─────────────┐  ┌─────────────┐             │
│  │ Tokenizer   │  │ Sampler     │             │
│  └─────────────┘  └─────────────┘             │
└────────────────────┬────────────────────────────┘
                     │
         ┌───────────┴───────────┐
         ▼                       ▼
┌─────────────────┐   ┌─────────────────────────┐
│ CUDA/ROCm       │   │ CPU (ARM64/x64)        │
│ (GPU加速)        │   │ (纯CPU模式)             │
└─────────────────┘   └─────────────────────────┘
```

### 4.4 API设计

```bash
# Ollama REST API

# 1. 列出模型
curl http://localhost:11434/api/tags

# 2. 生成响应
curl http://localhost:11434/api/generate \
  -d '{
    "model": "llama3.2",
    "prompt": "你好，请介绍一下你自己",
    "stream": false
  }'

# 3. 聊天模式
curl http://localhost:11434/api/chat \
  -d '{
    "model": "llama3.2",
    "messages": [
      {"role": "user", "content": "你好"}
    ],
    "stream": false
  }'

# 4. 模型管理
curl http://localhost:11434/api/pull -d '{"model": "llama3.2:3b"}'
curl http://localhost:11434/api/delete -d '{"name": "llama3.2:3b"}'
```

```python
# Python SDK
import ollama

# 生成
response = ollama.generate(
    model='llama3.2',
    prompt='Explain quantum computing'
)

# 聊天
response = ollama.chat(
    model='llama3.2',
    messages=[
        {'role': 'user', 'content': 'Hi!'}
    ]
)

# 流式输出
for chunk in ollama.generate(
    model='llama3.2',
    prompt='Write a story',
    stream=True
):
    print(chunk['response'], end='', flush=True)
```

### 4.5 Modelfile

```dockerfile
# 自定义模型配置
FROM llama3.2

# 设置系统提示词
SYSTEM """你是AI助手，请用简洁的方式回答问题。"""

# 设置参数
PARAMETER temperature 0.7
PARAMETER top_k 50
PARAMETER top_p 0.9

# 设置模板
TEMPLATE """{{ .System }}
{{ .Prompt }}"""
```

### 4.6 优点借鉴
- ✅ 极简的API设计
- ✅ 优秀的跨平台体验
- ✅ 模型格式统一（GGUF）
- ✅ 内存管理高效
- ✅ Docker支持完善

### 4.7 缺点/避免
- ❌ 无训练功能
- ❌ 无分布式能力
- ❌ 模型库有限（需自己导入）
- ❌ 商业化转型中

---

## 五、vLLM 技术分析

### 5.1 项目概述
- **定位**: 高性能LLM推理引擎
- **Stars**: 12,000+
- **特点**: PagedAttention、高吞吐量
- **维护者**: UC Berkeley

### 5.2 核心功能

```
vLLM Features
├── 高性能推理
│   ├── PagedAttention (减少内存浪费)
│   ├── Continuous Batching
│   └── Tensor Parallelism
├── 模型支持
│   ├── HuggingFace格式
│   ├── AWQ/GPTQ量化
│   └── 多模态模型
├── 部署方式
│   ├── OpenAI兼容API
│   ├── gRPC服务
│   └── HuggingFace Text Generation
└── 高级特性
    ├── 多 LoRA 适配器
    ├── 注意力缓存优化
    └── V1 Engine (新架构)
```

### 5.3 架构设计

```
┌─────────────────────────────────────────────────┐
│              vLLM Server                        │
│  ┌─────────────┐  ┌─────────────┐             │
│  │ Async LLM   │  │ Scheduler   │             │
│  │ Engine      │  │ (OomProfiler)              │
│  └─────────────┘  └─────────────┘             │
└────────────────────┬────────────────────────────┘
                     │
         ┌───────────┴───────────┐
         ▼                       ▼
┌─────────────────┐   ┌─────────────────────────┐
│ PagedAttention  │   │ Model Executor          │
│ Kernel          │   │ ( CUDA kernels)         │
└─────────────────┘   └─────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│              Transformer Model                  │
│         (PyTorch / TensorRT)                    │
└─────────────────────────────────────────────────┘
```

### 5.4 部署方式

```bash
# 1. 使用Docker部署
docker run --runtime nvidia \
  -p 8000:8000 \
  -v ./data:/data \
  vllm/vllm-openai:latest \
  --model meta-llama/Llama-2-7b-hf \
  --dtype half \
  --tensor-parallel-size 2

# 2. 使用Python
from vllm import LLM, SamplingParams

# 加载模型
llm = LLM(model="meta-llama/Llama-2-7b-hf")

# 采样参数
sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=512
)

# 批量推理
prompts = [
    "介绍一下人工智能",
    "什么是机器学习"
]
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(f"Prompt: {output.prompt}")
    print(f"Output: {output.outputs[0].text}")
```

### 5.5 OpenAI兼容API

```bash
# vLLM提供OpenAI兼容API
# 无需修改代码即可使用OpenAI SDK

export OPENAI_API_BASE="http://localhost:8000/v1"
export OPENAI_API_KEY="none"

# 使用OpenAI SDK
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="none"
)

response = client.chat.completions.create(
    model="meta-llama/Llama-2-7b-hf",
    messages=[{"role": "user", "content": "Hello!"}],
    temperature=0.7
)
```

### 5.6 优点借鉴
- ✅ 极高的推理吞吐量
- ✅ OpenAI API兼容
- ✅ 内存管理优秀（PagedAttention）
- ✅ 分布式推理支持
- ✅ 量化支持完善

### 5.7 缺点/避免
- ❌ 部署复杂度中等
- ❌ 对硬件有要求（GPU）
- ❌ 无训练功能
- ❌ 资源占用较大

---

## 六、技术对比总结

### 6.1 功能对比矩阵

| 功能 | MLflow | Kubeflow | Label Studio | Ollama | vLLM |
|------|--------|----------|--------------|--------|------|
| **实验跟踪** | ✅ | ✅ | ❌ | ❌ | ❌ |
| **训练编排** | ❌ | ✅ | ❌ | ❌ | ❌ |
| **数据标注** | ❌ | ❌ | ✅ | ❌ | ❌ |
| **模型服务** | ✅ | ✅ | ❌ | ✅ | ✅ |
| **分布式训练** | ❌ | ✅ | ❌ | ❌ | ❌ |
| **本地推理** | ❌ | ❌ | ❌ | ✅ | ❌ |
| **高并发推理** | ❌ | ❌ | ❌ | ❌ | ✅ |
| **跨平台** | ✅ | ❌ | ✅ | ✅ | ❌ |
| **易用性** | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |

### 6.2 技术栈对比

| 项目 | 前端 | 后端 | 数据库 | 部署 |
|------|------|------|--------|------|
| **MLflow** | React | Python(Flask) | SQLite/MySQL/PostgreSQL | Docker |
| **Kubeflow** | React | Go (K8s) | etcd | K8s |
| **Label Studio** | React | Django | PostgreSQL | Docker |
| **Ollama** | CLI | Go | 无 | Docker |
| **vLLM** | 无 | Python | 无 | Docker |

### 6.3 集成优先级

| 优先级 | 项目 | 集成方式 | 理由 |
|--------|------|---------|------|
| **P0** | MLflow | REST API | 成熟、稳定、Python友好 |
| **P0** | Ollama | HTTP API | 简单、跨平台、LLM友好 |
| **P0** | vLLM | OpenAI API | 高性能、API兼容 |
| **P1** | Label Studio | Webhook + API | 标注刚需、API完善 |
| **P1** | Kubeflow | K8s Operator | 生产环境、复杂训练 |

---

## 七、对我们架构的启示

### 7.1 推荐的架构模式

```
┌─────────────────────────────────────────────────────────────┐
│                    AI Platform                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                  Web UI (React)                      │   │
│  │  - Ant Design组件库                                  │   │
│  │  - 可视化工作流设计器                                 │   │
│  │  - 实时任务监控                                       │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────┬───────────────────────────────────┘
                          │ REST API + WebSocket
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                 API Server (FastAPI)                        │
│  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐  │
│  │ 任务调度   │ │ 资源管理   │ │ 用户认证   │ │ 消息队列   │  │
│  │ Celery    │ │ GPU管理    │ │ JWT/OIDC  │ │ Redis     │  │
│  └───────────┘ └───────────┘ └───────────┘ └───────────┘  │
└─────────────────────────┬───────────────────────────────────┘
                          │
    ┌─────────────────────┼─────────────────────┐
    ▼                     ▼                     ▼
┌─────────────┐   ┌─────────────────┐   ┌─────────────┐
│   MLflow    │   │     Ollama      │   │Label Studio │
│  (实验跟踪)  │   │  (本地推理)     │   │  (数据标注)  │
└─────────────┘   └─────────────────┘   └─────────────┘
                          │
                          ▼
                  ┌─────────────────┐
                  │     vLLM        │
                  │  (高并发推理)    │
                  └─────────────────┘
```

### 7.2 关键设计决策

| 决策点 | 推荐方案 | 理由 |
|--------|---------|------|
| **前端框架** | React + Ant Design | 企业级、组件丰富 |
| **后端框架** | FastAPI | Python生态、异步高性能 |
| **数据库** | PostgreSQL + Redis | 成熟、稳定、缓存 |
| **任务队列** | Celery + Redis | 简单、可靠 |
| **容器化** | Docker Compose | 跨平台、开发友好 |
| **认证** | JWT | 标准、安全 |
| **训练编排** | Celery (单机) / Ray (分布式) | 简化部署 |

### 7.3 API设计参考

借鉴MLflow的简洁风格：

```python
# 统一的API风格
from fastapi import APIRouter, Depends

router = APIRouter(prefix="/api/v1")

# 实验跟踪
@router.post("/experiments")
async def create_experiment(exp: Experiment):
    ...

@router.get("/experiments/{id}")
async def get_experiment(id: str):
    ...

# 任务管理
@router.post("/tasks")
async def create_task(task: Task):
    ...

@router.get("/tasks/{id}/logs")
async def get_task_logs(id: str):
    ...

# 模型服务
@router.post("/models/{id}/deploy")
async def deploy_model(id: str):
    ...
```

### 7.4 数据库设计参考

借鉴MLflow的简洁表结构：

```sql
-- 核心表设计

-- 项目表
CREATE TABLE projects (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- 实验表
CREATE TABLE experiments (
    id SERIAL PRIMARY KEY,
    project_id INTEGER REFERENCES projects(id),
    name VARCHAR(255) NOT NULL,
    status VARCHAR(50) DEFAULT 'running',
    created_at TIMESTAMP DEFAULT NOW()
);

-- 运行表
CREATE TABLE runs (
    id SERIAL PRIMARY KEY,
    experiment_id INTEGER REFERENCES experiments(id),
    metrics JSONB,
    params JSONB,
    artifacts JSONB,
    status VARCHAR(50) DEFAULT 'running'
);

-- 模型表
CREATE TABLE models (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    version INTEGER,
    stage VARCHAR(50) DEFAULT 'staging',
    path VARCHAR(500),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- 任务表
CREATE TABLE tasks (
    id SERIAL PRIMARY KEY,
    type VARCHAR(50) NOT NULL,  -- 'training', 'inference', 'distillation'
    status VARCHAR(50) DEFAULT 'pending',
    config JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);
```

### 7.5 UI设计参考

借鉴各项目的优秀实践：

| 组件 | 参考项目 | 借鉴点 |
|------|---------|--------|
| 仪表盘 | MLflow | 简洁的指标展示 |
| 任务列表 | Kubeflow | 状态筛选、实时更新 |
| 标注界面 | Label Studio | 拖拽式标注 |
| 模型卡片 | MLflow | 清晰的元信息展示 |
| 工作流设计 | Kubeflow Pipelines | 可视化编排 |

---

## 八、结论与建议

### 8.1 技术选型总结

| 层级 | 推荐方案 | 替代方案 |
|------|---------|---------|
| **前端** | React + Ant Design | Vue + Element |
| **后端** | FastAPI + Celery | Flask + Celery |
| **数据库** | PostgreSQL + Redis | MySQL + Redis |
| **任务队列** | Celery | Dramatiq |
| **容器化** | Docker Compose | Podman |
| **实验跟踪** | MLflow (集成) | 自研 |
| **推理服务** | vLLM (集成) | TGI |
| **本地推理** | Ollama (集成) | 自研 |
| **数据标注** | Label Studio (集成) | 自研 |

### 8.2 集成策略

**策略：分层集成，渐进式接入**

```
Phase 1 (MVP)
├── Ollama (简单集成)
├── MLflow (REST API)
└── vLLM (容器化)

Phase 2 (核心功能)
├── Label Studio (标注)
├── 自研训练管理
└── 自研工作流

Phase 3 (企业级)
├── Kubeflow (可选)
├── 多集群管理
└── 高级监控
```

### 8.3 避免的坑

| 坑点 | 解决方案 |
|------|---------|
| **版本兼容** | 使用Docker隔离，固定版本 |
| **资源抢占** | 实现资源配额管理 |
| **单点故障** | 使用Celery + Redis的高可用部署 |
| **跨平台** | 主要支持Linux，Mac/Windows降级支持 |
| **学习曲线** | 提供模板和向导，降低使用门槛 |

---

## 附录：参考资源

### 官方文档
- MLflow: https://mlflow.org/docs/latest/
- Kubeflow: https://www.kubeflow.org/docs/
- Label Studio: https://labelstud.io/guide/
- Ollama: https://github.com/ollama/ollama
- vLLM: https://docs.vllm.ai/

### GitHub仓库
- MLflow: https://github.com/mlflow/mlflow
- Kubeflow: https://github.com/kubeflow/kubeflow
- Label Studio: https://github.com/HumanSignal/label-studio
- Ollama: https://github.com/ollama/ollama
- vLLM: https://github.com/vllm-project/vllm

---

**报告完成时间**: 2026-02-07  
**版本**: v1.0  
**状态**: 已完成技术调研
