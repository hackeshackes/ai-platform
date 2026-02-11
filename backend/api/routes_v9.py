# V9: 自适应学习、联邦学习、自主决策

# Agent自适应学习
try:
    from backend.agents.adaptive.api import router as adaptive_router
    router.include_router(adaptive_router, prefix="/adaptive", tags=["Adaptive Learning"])
    ADAPTIVE_LEARNING_ENABLED = True
except ImportError as e:
    print(f"Adaptive learning module not available: {e}")
    ADAPTIVE_LEARNING_ENABLED = False

# 联邦学习平台
try:
    from backend.federated.api.endpoints import router as federated_router
    router.include_router(federated_router, prefix="/federated", tags=["Federated Learning"])
    FEDERATED_LEARNING_ENABLED = True
except ImportError as e:
    print(f"Federated learning module not available: {e}")
    FEDERATED_LEARNING_ENABLED = False

# 自主决策引擎
try:
    from backend.decision.api.endpoints import router as decision_router
    router.include_router(decision_router, prefix="/decision", tags=["Decision Engine"])
    DECISION_ENGINE_ENABLED = True
except ImportError as e:
    print(f"Decision engine module not available: {e}")
    DECISION_ENGINE_ENABLED = False
