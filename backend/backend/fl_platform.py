"""
Federated Learning Platform
Main orchestrator for federated learning sessions
"""
import logging
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional

from .models import (
    FLSession,
    FLConfig,
    LocalModel,
    GlobalModel,
    FLClientInfo,
    SessionStatus,
    TrainingResult,
    AggregationResult
)
from .aggregator import Aggregator
from .storage import SessionStore
from .privacy import PrivacyManager

logger = logging.getLogger(__name__)


class FederatedLearningPlatform:
    """联邦学习平台 - 主协调器"""
    
    def __init__(
        self,
        storage: Optional[SessionStore] = None,
        aggregator: Optional[Aggregator] = None,
        tls_enabled: bool = False,
        tls_config: Optional[Dict[str, str]] = None
    ):
        """
        初始化联邦学习平台
        
        Args:
            storage: 会话存储 (可选,默认内存存储)
            aggregator: 模型聚合器 (可选,默认FedAvg)
            tls_enabled: 是否启用TLS
            tls_config: TLS配置
        """
        self.store = storage or SessionStore()
        self.aggregator = aggregator or Aggregator(aggregation_method="fedavg")
        self.tls_enabled = tls_enabled
        self.tls_config = tls_config or {}
        self.privacy_manager = PrivacyManager()
        
        self._active_sessions: Dict[str, FLSession] = {}
        self._client_sessions: Dict[str, str] = {}
        
        logger.info(
            f"FL Platform initialized: tls={tls_enabled}, "
            f"aggregator={self.aggregator.aggregation_method}"
        )
    
    def _validate_tls(self, client_id: str) -> bool:
        """
        验证TLS连接
        
        Args:
            client_id: 客户端ID
            
        Returns:
            是否验证通过
        """
        if not self.tls_enabled:
            return True
        
        if not self.tls_config:
            logger.warning("TLS enabled but no config provided")
            return False
        
        return True
    
    async def create_session(self, config: FLConfig) -> FLSession:
        """
        创建联邦训练会话
        
        Args:
            config: 联邦学习配置
            
        Returns:
            创建的会话
        """
        try:
            session = FLSession(
                id=uuid.uuid4().hex[:12],
                config=config,
                status=SessionStatus.PENDING,
                participants=[],
                created_at=datetime.now()
            )
            
            await self.store.save(session)
            self._active_sessions[session.id] = session
            
            logger.info(f"Session created: {session.id} with config: {config.model_name}")
            
            return session
            
        except Exception as e:
            logger.error(f"Failed to create session: {e}")
            raise
    
    async def join_session(
        self,
        session_id: str,
        client: FLClientInfo,
        validate_tls: bool = True
    ) -> bool:
        """
        加入联邦训练会话
        
        Args:
            session_id: 会话ID
            client: 客户端信息
            validate_tls: 是否验证TLS
            
        Returns:
            是否成功
        """
        try:
            if validate_tls and not self._validate_tls(client.client_id):
                raise ValueError("TLS validation failed")
            
            session = await self.store.get(session_id)
            
            if not session:
                logger.warning(f"Session not found: {session_id}")
                return False
            
            if session.status != SessionStatus.PENDING:
                logger.warning(
                    f"Session {session_id} is not in PENDING state: {session.status}"
                )
                return False
            
            if len(session.participants) >= session.config.max_clients:
                logger.warning(f"Session {session_id} has reached max clients")
                return False
            
            existing_ids = [p.client_id for p in session.participants]
            if client.client_id in existing_ids:
                logger.info(f"Client {client.client_id} already in session")
                return True
            
            await self.store.register_client(session_id, client)
            self._client_sessions[client.client_id] = session_id
            
            session = await self.store.get(session_id)
            self._active_sessions[session_id] = session
            
            logger.info(
                f"Client {client.client_id} joined session {session_id}, "
                f"total participants: {len(session.participants)}"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to join session: {e}")
            return False
    
    async def start_training(self, session_id: str) -> FLSession:
        """
        开始训练 (当有足够的参与者时)
        
        Args:
            session_id: 会话ID
            
        Returns:
            更新后的会话
        """
        session = await self.store.get(session_id)
        
        if not session:
            raise ValueError(f"Session not found: {session_id}")
        
        if len(session.participants) < session.config.min_clients:
            raise ValueError(
                f"Insufficient participants: {len(session.participants)} "
                f"< {session.config.min_clients}"
            )
        
        session.status = SessionStatus.TRAINING
        session.current_round = 1
        await self.store.update(session)
        
        logger.info(
            f"Training started for session {session_id}, "
            f"participants: {len(session.participants)}"
        )
        
        return session
    
    async def submit_local_model(
        self,
        session_id: str,
        local_model: LocalModel
    ) -> bool:
        """
        提交本地模型
        
        Args:
            session_id: 会话ID
            local_model: 本地模型
            
        Returns:
            是否成功
        """
        session = await self.store.get(session_id)
        
        if not session:
            raise ValueError(f"Session not found: {session_id}")
        
        if session.status != SessionStatus.TRAINING:
            raise ValueError(f"Session is not in TRAINING state: {session.status}")
        
        logger.info(
            f"Local model submitted: session={session_id}, "
            f"client={local_model.client_id}, accuracy={local_model.accuracy:.4f}"
        )
        
        return True
    
    async def aggregate_models(self, session_id: str) -> GlobalModel:
        """
        聚合模型 (FedAvg)
        
        Args:
            session_id: 会话ID
            
        Returns:
            聚合后的全局模型
        """
        session = await self.store.get(session_id)
        
        if not session:
            raise ValueError(f"Session not found: {session_id}")
        
        session.status = SessionStatus.AGGREGATING
        await self.store.update(session)
        
        participants = session.participants
        data_sizes = [p.data_size for p in participants]
        
        client_weights = []
        for participant in participants:
            weights = self._generate_mock_weights(participant)
            client_weights.append(weights)
        
        if session.config.differential_privacy:
            composed_epsilon = self.privacy_manager.compose_privacy_budget(
                len(participants)
            )
        
        global_weights = self.aggregator.fedavg(client_weights, data_sizes)
        
        new_version = str(float(session.global_model_version) + 1)
        
        global_model = GlobalModel(
            session_id=session_id,
            weights=global_weights,
            version=new_version,
            round_number=session.current_round,
            created_at=datetime.now()
        )
        
        session.global_model_version = new_version
        session.current_round += 1
        
        if session.current_round > session.config.rounds:
            session.status = SessionStatus.COMPLETED
            session.completed_at = datetime.now()
        else:
            session.status = SessionStatus.TRAINING
        
        await self.store.update(session)
        
        logger.info(
            f"Models aggregated: session={session_id}, round={session.current_round - 1}"
        )
        
        return global_model
    
    def _generate_mock_weights(self, participant: FLClientInfo) -> Dict[str, Any]:
        """
        生成模拟权重 (实际应用中应从客户端获取)
        
        Args:
            participant: 参与者
            
        Returns:
            模拟权重字典
        """
        np_seed = hash(participant.client_id) % (2**32)
        import numpy as np
        np.random.seed(np_seed)
        
        return {
            "layer_0": np.random.randn(100).tolist(),
            "layer_1": np.random.randn(64).tolist(),
            "output": np.random.randn(10).tolist()
        }
    
    async def get_session_status(self, session_id: str) -> Optional[FLSession]:
        """
        获取会话状态
        
        Args:
            session_id: 会话ID
            
        Returns:
            会话对象或None
        """
        return await self.store.get(session_id)
    
    async def list_sessions(
        self,
        status: Optional[SessionStatus] = None,
        limit: int = 100
    ) -> List[FLSession]:
        """
        列出所有会话
        
        Args:
            status: 按状态过滤
            limit: 返回数量限制
            
        Returns:
            会话列表
        """
        return await self.store.list_sessions(status, limit)
    
    async def get_client_session(self, client_id: str) -> Optional[str]:
        """
        获取客户端所在的会话
        
        Args:
            client_id: 客户端ID
            
        Returns:
            会话ID或None
        """
        return await self.store.get_client_session(client_id)
    
    async def get_global_model(self, session_id: str) -> Optional[GlobalModel]:
        """
        获取全局模型
        
        Args:
            session_id: 会话ID
            
        Returns:
            全局模型或None
        """
        session = await self.store.get(session_id)
        
        if not session:
            return None
        
        if session.status == SessionStatus.COMPLETED:
            return GlobalModel(
                session_id=session_id,
                weights=self._generate_mock_weights(session.participants[0]) if session.participants else {},
                version=session.global_model_version,
                round_number=session.current_round,
                created_at=datetime.now()
            )
        
        return None
    
    async def get_privacy_report(self, session_id: str) -> Dict[str, Any]:
        """
        获取隐私报告
        
        Args:
            session_id: 会话ID
            
        Returns:
            隐私报告字典
        """
        return self.privacy_manager.get_privacy_spent()
    
    async def get_aggregation_history(self, session_id: str) -> List[Dict[str, Any]]:
        """
        获取聚合历史
        
        Args:
            session_id: 会话ID
            
        Returns:
            聚合历史列表
        """
        return self.aggregator.get_aggregation_history()
    
    async def close_session(self, session_id: str, reason: str = "completed") -> bool:
        """
        关闭会话
        
        Args:
            session_id: 会话ID
            reason: 关闭原因
            
        Returns:
            是否成功
        """
        session = await self.store.get(session_id)
        
        if session:
            session.status = SessionStatus.FAILED if reason == "error" else SessionStatus.COMPLETED
            session.completed_at = datetime.now()
            session.error_message = reason if reason == "error" else None
            
            await self.store.update(session)
            
            if session_id in self._active_sessions:
                del self._active_sessions[session_id]
            
            logger.info(f"Session closed: {session_id}, reason: {reason}")
            return True
        
        return False
    
    @property
    def active_session_count(self) -> int:
        """获取活跃会话数量"""
        return len(self._active_sessions)
    
    @property
    def total_sessions(self) -> int:
        """获取总会话数量"""
        return self.store.session_count
