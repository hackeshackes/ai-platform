"""API Router"""
from fastapi import APIRouter

# 导入所有端点
from api.endpoints import auth, users, projects, experiments, tasks, datasets, models, health, gpu, metrics, training, inference, settings, versions, quality, pipeline

# v2.3: 动态导入模块
import importlib
import sys
import os

def load_module(name, filepath):
    """加载模块"""
    try:
        spec = importlib.util.spec_from_file_location(name, filepath)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        print(f"Failed to load {name}: {e}")
        return None

backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
endpoints_dir = os.path.join(backend_dir, 'api', 'endpoints')

gateway_module = load_module('gateway_endpoint', os.path.join(endpoints_dir, 'gateway.py'))
assistant_module = load_module('assistant_endpoint', os.path.join(endpoints_dir, 'assistant.py'))
judges_module = load_module('judges_endpoint', os.path.join(endpoints_dir, 'judges.py'))
ray_module = load_module('ray_endpoint', os.path.join(endpoints_dir, 'ray.py'))
optimization_module = load_module('optimization_endpoint', os.path.join(endpoints_dir, 'optimization.py'))

router = APIRouter()

# v2.3: 注册路由
if gateway_module:
    router.include_router(gateway_module.router, prefix="/gateway", tags=["AI Gateway"])
    AI_GATEWAY_ENABLED = True
else:
    AI_GATEWAY_ENABLED = False

if assistant_module:
    router.include_router(assistant_module.router, prefix="/assistant", tags=["AI Assistant"])
    AI_ASSISTANT_ENABLED = True
else:
    AI_ASSISTANT_ENABLED = False

if judges_module:
    router.include_router(judges_module.router, prefix="/judges", tags=["Judge Builder"])
    JUDGE_BUILDER_ENABLED = True
else:
    JUDGE_BUILDER_ENABLED = False

if ray_module:
    router.include_router(ray_module.router, prefix="/ray", tags=["Ray Data"])
    RAY_DATA_ENABLED = True
else:
    RAY_DATA_ENABLED = False

if optimization_module:
    router.include_router(optimization_module.router, prefix="/optimization", tags=["Performance"])
    OPTIMIZATION_ENABLED = True
else:
    OPTIMIZATION_ENABLED = False

# 认证
router.include_router(auth.router, prefix="/auth", tags=["Authentication"])

# 用户
router.include_router(users.router, prefix="/users", tags=["Users"])

# 项目
router.include_router(projects.router, prefix="/projects", tags=["Projects"])

# 实验
router.include_router(experiments.router, prefix="/experiments", tags=["Experiments"])

# 任务
router.include_router(tasks.router, prefix="/tasks", tags=["Tasks"])

# 数据集
router.include_router(datasets.router, prefix="/datasets", tags=["Datasets"])

# 模型
router.include_router(models.router, prefix="/models", tags=["Models"])

# GPU监控
router.include_router(gpu.router, prefix="/gpu", tags=["GPU"])

# 训练指标
router.include_router(metrics.router, prefix="/metrics", tags=["Metrics"])

# 训练任务
router.include_router(training.router, prefix="/training", tags=["Training"])

# 推理服务
router.include_router(inference.router, prefix="/inference", tags=["Inference"])

# 系统设置
router.include_router(settings.router, prefix="/settings", tags=["Settings"])

# v1.1: 数据集版本
router.include_router(versions.router, prefix="/datasets", tags=["Versions"])

# v1.1: 数据质量检查
router.include_router(quality.router, prefix="/datasets", tags=["Quality"])

# v2.0 Phase 2: Pipeline编排
router.include_router(pipeline.router, prefix="/pipelines", tags=["Pipelines"])

# v2.0 Phase 2: CI/CD
try:
    from api.endpoints.cicd import deploy, distributed
    router.include_router(deploy.router, prefix="/cicd", tags=["CI/CD"])
    router.include_router(distributed.router, prefix="/cluster", tags=["Cluster"])
    CICD_ENABLED = True
except ImportError:
    CICD_ENABLED = False

# v2.0 Phase 2: 监控告警
try:
    from api.endpoints import monitoring
    router.include_router(monitoring.router, prefix="/monitoring", tags=["Monitoring"])
    MONITORING_ENABLED = True
except ImportError:
    MONITORING_ENABLED = False

# v2.0 Phase 3: AutoML
try:
    from api.endpoints.automl import hpo
    router.include_router(hpo.router, prefix="/automl", tags=["AutoML"])
    AUTOML_ENABLED = True
except ImportError:
    AUTOML_ENABLED = False

# v2.0 Phase 3: RAG
try:
    from api.endpoints.rag import pipeline
    router.include_router(pipeline.router, prefix="/rag", tags=["RAG"])
    RAG_ENABLED = True
except ImportError:
    RAG_ENABLED = False

# v2.0 Phase 3: Agent编排
try:
    from api.endpoints import agents
    router.include_router(agents.router, prefix="/agents", tags=["Agents"])
    AGENTS_ENABLED = True
except ImportError:
    AGENTS_ENABLED = False

# v2.1: Feature Store
try:
    from api.endpoints.feature_store import api as feature_store_api
    router.include_router(feature_store_api.router, prefix="/feature-store", tags=["Feature Store"])
    FEATURE_STORE_ENABLED = True
except ImportError:
    FEATURE_STORE_ENABLED = False

# v2.1: Model Registry
try:
    from api.endpoints import registry
    router.include_router(registry.router, prefix="/model-registry", tags=["Model Registry"])
    MODEL_REGISTRY_ENABLED = True
except ImportError:
    MODEL_REGISTRY_ENABLED = False

# v2.1: Model Lineage
try:
    from api.endpoints import lineage
    router.include_router(lineage.router, prefix="/lineage", tags=["Lineage"])
    LINEAGE_ENABLED = True
except ImportError:
    LINEAGE_ENABLED = False

# v2.1: 数据质量
try:
    from api.endpoints import quality
    router.include_router(quality.router, prefix="/quality", tags=["Data Quality"])
    QUALITY_ENABLED = True
except ImportError:
    QUALITY_ENABLED = False

# v2.1: Notebooks
try:
    from api.endpoints import notebooks
    router.include_router(notebooks.router, prefix="/notebooks", tags=["Notebooks"])
    NOTEBOOKS_ENABLED = True
except ImportError:
    NOTEBOOKS_ENABLED = False

# v2.2: LLM Tracing
try:
    from api.endpoints import tracing
    router.include_router(tracing.router, prefix="/tracing", tags=["LLM Tracing"])
    LLM_TRACING_ENABLED = True
except ImportError:
    LLM_TRACING_ENABLED = False

# v2.2: 模型评估
try:
    from api.endpoints import evaluation
    router.include_router(evaluation.router, prefix="/evaluation", tags=["Model Evaluation"])
    EVALUATION_ENABLED = True
except ImportError:
    EVALUATION_ENABLED = False

# v2.2: 多租户
try:
    from api.endpoints import multi_tenant
    router.include_router(multi_tenant.router, prefix="/tenants", tags=["Multi-Tenant"])
    MULTI_TENANT_ENABLED = True
except ImportError:
    MULTI_TENANT_ENABLED = False

# v2.2: 分布式训练
try:
    from api.endpoints import distributed
    router.include_router(distributed.router, prefix="/distributed", tags=["Distributed Training"])
    DISTRIBUTED_ENABLED = True
except ImportError:
    DISTRIBUTED_ENABLED = False

# v2.2: 插件系统
try:
    from api.endpoints import plugins
    router.include_router(plugins.router, prefix="/plugins", tags=["Plugins"])
    PLUGINS_ENABLED = True
except ImportError:
    PLUGINS_ENABLED = False

# v2.3: AI Gateway
try:
    from api.endpoints import gateway
    router.include_router(gateway.router, prefix="/gateway", tags=["AI Gateway"])
    AI_GATEWAY_ENABLED = True
except ImportError:
    AI_GATEWAY_ENABLED = False

# v2.3: AI Assistant
try:
    from api.endpoints import assistant
    router.include_router(assistant.router, prefix="/assistant", tags=["AI Assistant"])
    AI_ASSISTANT_ENABLED = True
except ImportError:
    AI_ASSISTANT_ENABLED = False

# v2.3: Judge Builder
try:
    from api.endpoints import judges
    router.include_router(judges.router, prefix="/judges", tags=["Judge Builder"])
    JUDGE_BUILDER_ENABLED = True
except ImportError:
    JUDGE_BUILDER_ENABLED = False

# v2.3: Ray Data
try:
    from api.endpoints import ray
    router.include_router(ray.router, prefix="/ray", tags=["Ray Data"])
    RAY_DATA_ENABLED = True
except ImportError:
    RAY_DATA_ENABLED = False

# v2.3: 性能优化
try:
    from api.endpoints import optimization
    router.include_router(optimization.router, prefix="/optimization", tags=["Performance"])
    OPTIMIZATION_ENABLED = True
except ImportError:
    OPTIMIZATION_ENABLED = False

# v2.4: Prompt Management
try:
    from api.endpoints import prompt
    router.include_router(prompt.router, prefix="/prompts", tags=["Prompt Management"])
    PROMPT_MANAGEMENT_ENABLED = True
except ImportError:
    PROMPT_MANAGEMENT_ENABLED = False

# v2.4: LLM Guardrails
try:
    from api.endpoints import guardrails
    router.include_router(guardrails.router, prefix="/guardrails", tags=["LLM Guardrails"])
    GUARDRAILS_ENABLED = True
except ImportError:
    GUARDRAILS_ENABLED = False

# v2.4: Cost Intelligence
try:
    from api.endpoints import cost
    router.include_router(cost.router, prefix="/cost", tags=["Cost Intelligence"])
    COST_INTELLIGENCE_ENABLED = True
except ImportError:
    COST_INTELLIGENCE_ENABLED = False

# v2.4: Model Serving
try:
    from api.endpoints import serving
    router.include_router(serving.router, prefix="/serving", tags=["Model Serving"])
    MODEL_SERVING_ENABLED = True
except ImportError:
    MODEL_SERVING_ENABLED = False

# v2.4: A/B Testing
try:
    from api.endpoints import abtesting
    router.include_router(abtesting.router, prefix="/abtesting", tags=["A/B Testing"])
    AB_TESTING_ENABLED = True
except ImportError:
    AB_TESTING_ENABLED = False

# v2.4: Edge Inference
try:
    from api.endpoints import edge
    router.include_router(edge.router, prefix="/edge", tags=["Edge Inference"])
    EDGE_INFERENCE_ENABLED = True
except ImportError:
    EDGE_INFERENCE_ENABLED = False

# v2.4: Visualization
try:
    from api.endpoints import visualization
    router.include_router(visualization.router, prefix="/visualization", tags=["Visualization"])
    VISUALIZATION_ENABLED = True
except ImportError:
    VISUALIZATION_ENABLED = False

# v2.4: Collaboration
try:
    from api.endpoints import collaboration
    router.include_router(collaboration.router, prefix="/collaboration", tags=["Collaboration"])
    COLLABORATION_ENABLED = True
except ImportError:
    COLLABORATION_ENABLED = False

# v2.4: CLI
try:
    from api.endpoints import cli
    router.include_router(cli.router, prefix="/cli", tags=["CLI"])
    CLI_ENABLED = True
except ImportError:
    CLI_ENABLED = False

# v2.4: Cloud Integration
try:
    from api.endpoints import cloud
    router.include_router(cloud.router, prefix="/cloud", tags=["Cloud"])
    CLOUD_INTEGRATION_ENABLED = True
except ImportError:
    CLOUD_INTEGRATION_ENABLED = False

# v2.4: Plugin Marketplace
try:
    from api.endpoints import plugins
    router.include_router(plugins.router, prefix="/plugins", tags=["Plugins"])
    PLUGIN_MARKETPLACE_ENABLED = True
except ImportError:
    PLUGIN_MARKETPLACE_ENABLED = False

# v3: Distillation Engine
try:
    from api.endpoints.distillation import router as distillation_router
    router.include_router(distillation_router, prefix="/distillation", tags=["Distillation"])
    DISTILLATION_ENABLED = True
except ImportError as e:
    print(f"Distillation module not available: {e}")
    DISTILLATION_ENABLED = False

# v3: LLM Providers Registry
try:
    from core.providers.registry import router as providers_router
    router.include_router(providers_router, prefix="/providers", tags=["Providers"])
    PROVIDERS_ENABLED = True
except ImportError as e:
    print(f"Providers module not available: {e}")
    PROVIDERS_ENABLED = False

# v3: Enhanced RAG
try:
    from api.endpoints.rag_enhanced import router as rag_enhanced_router
    router.include_router(rag_enhanced_router, prefix="/rag", tags=["RAG Enhanced"])
    RAG_ENHANCED_ENABLED = True
except ImportError as e:
    print(f"RAG Enhanced module not available: {e}")
    RAG_ENHANCED_ENABLED = False

# v3: Agents Framework
try:
    from agents.api.fastapi_endpoints import create_agent_router
    router.include_router(create_agent_router(), prefix="/agents", tags=["Agents"])
    AGENTS_ENABLED = True
except ImportError as e:
    print(f"Agents module not available: {e}")
    AGENTS_ENABLED = False

# v3: Dataset Generation
try:
    from api.endpoints.dataset_gen import router as dataset_gen_router
    router.include_router(dataset_gen_router, prefix="/datasets", tags=["Dataset Generation"])
    DATASET_GEN_ENABLED = True
except ImportError as e:
    print(f"Dataset Generation module not available: {e}")
    DATASET_GEN_ENABLED = False

# ML集成 - 使用条件导入，Docker环境自动启用
try:
    from api.endpoints import mlflow, ollama
    router.include_router(mlflow.router, prefix="/mlflow", tags=["MLflow"])
    router.include_router(ollama.router, prefix="/ollama", tags=["Ollama"])
    ML_INTEGRATION_ENABLED = True
except ImportError:
    ML_INTEGRATION_ENABLED = False

# 健康检查
router.include_router(health.router, prefix="/health", tags=["Health"])

# 系统信息端点
@router.get("/info")
async def system_info():
    """获取系统集成状态"""
    return {
        "ml_integration": ML_INTEGRATION_ENABLED,
        "services": {
            "backend": "running",
            "database": "postgresql",
            "cache": "redis",
            "pipeline": "v2.0",
            "automl": AUTOML_ENABLED,
            "rag": RAG_ENABLED,
            "agents": AGENTS_ENABLED,
            "feature_store": FEATURE_STORE_ENABLED,
            "model_registry": MODEL_REGISTRY_ENABLED,
            "lineage": LINEAGE_ENABLED,
            "quality": QUALITY_ENABLED,
            "notebooks": NOTEBOOKS_ENABLED,
            "llm_tracing": LLM_TRACING_ENABLED,
            "evaluation": EVALUATION_ENABLED,
            "multi_tenant": MULTI_TENANT_ENABLED,
            "distributed": DISTRIBUTED_ENABLED,
            "plugins": PLUGINS_ENABLED,
            "ai_gateway": AI_GATEWAY_ENABLED,
            "ai_assistant": AI_ASSISTANT_ENABLED,
            "judge_builder": JUDGE_BUILDER_ENABLED,
            "ray_data": RAY_DATA_ENABLED,
            "optimization": OPTIMIZATION_ENABLED,
            "prompt_management": PROMPT_MANAGEMENT_ENABLED,
            "guardrails": GUARDRAILS_ENABLED,
            "cost_intelligence": COST_INTELLIGENCE_ENABLED,
            "model_serving": MODEL_SERVING_ENABLED,
            "ab_testing": AB_TESTING_ENABLED,
            "edge_inference": EDGE_INFERENCE_ENABLED,
            "visualization": VISUALIZATION_ENABLED,
            "collaboration": COLLABORATION_ENABLED,
            "cli": CLI_ENABLED,
            "cloud_integration": CLOUD_INTEGRATION_ENABLED,
            "plugin_marketplace": PLUGIN_MARKETPLACE_ENABLED,
            "distillation": DISTILLATION_ENABLED,
            "providers": PROVIDERS_ENABLED
        },
        "version": "2.4.0-beta"
    }
