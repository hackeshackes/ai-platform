"""
Session storage for Federated Learning Platform
In-memory storage implementation (can be replaced with database)
"""
import logging
from datetime import datetime
from typing import Dict, Optional, List
from .models import FLSession, FLConfig, FLClientInfo, SessionStatus

logger = logging.getLogger(__name__)


class SessionStore:
    """会话存储 - 管理联邦学习会话的持久化"""
    
    def __init__(self):
        self._sessions: Dict[str, FLSession] = {}
        self._client_sessions: Dict[str, str] = {}  # client_id -> session_id
    
    async def save(self, session: FLSession) -> bool:
        """
        保存会话
        
        Args:
            session: 要保存的会话
            
        Returns:
            是否成功
        """
        try:
            session.updated_at = datetime.now()
            self._sessions[session.id] = session
            logger.info(f"Session saved: {session.id}")
            return True
        except Exception as e:
            logger.error(f"Failed to save session: {e}")
            return False
    
    async def get(self, session_id: str) -> Optional[FLSession]:
        """
        获取会话
        
        Args:
            session_id: 会话ID
            
        Returns:
            会话对象或None
        """
        session = self._sessions.get(session_id)
        if session:
            logger.debug(f"Session retrieved: {session_id}")
        else:
            logger.debug(f"Session not found: {session_id}")
        return session
    
    async def update(self, session: FLSession) -> bool:
        """
        更新会话
        
        Args:
            session: 要更新的会话
            
        Returns:
            是否成功
        """
        try:
            session.updated_at = datetime.now()
            if session.id in self._sessions:
                self._sessions[session.id] = session
                logger.info(f"Session updated: {session.id}")
                return True
            else:
                logger.warning(f"Session not found for update: {session.id}")
                return False
        except Exception as e:
            logger.error(f"Failed to update session: {e}")
            return False
    
    async def delete(self, session_id: str) -> bool:
        """
        删除会话
        
        Args:
            session_id: 会话ID
            
        Returns:
            是否成功
        """
        try:
            if session_id in self._sessions:
                del self._sessions[session_id]
                logger.info(f"Session deleted: {session_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to delete session: {e}")
            return False
    
    async def list_sessions(
        self,
        status: Optional[SessionStatus] = None,
        limit: int = 100
    ) -> List[FLSession]:
        """
        列出会话
        
        Args:
            status: 按状态过滤
            limit: 返回数量限制
            
        Returns:
            会话列表
        """
        sessions = list(self._sessions.values())
        
        if status:
            sessions = [s for s in sessions if s.status == status]
        
        sessions.sort(key=lambda x: x.created_at, reverse=True)
        
        return sessions[:limit]
    
    async def register_client(
        self,
        session_id: str,
        client: FLClientInfo
    ) -> bool:
        """
        注册客户端到会话
        
        Args:
            session_id: 会话ID
            client: 客户端信息
            
        Returns:
            是否成功
        """
        session = await self.get(session_id)
        if session:
            if client.client_id not in [p.client_id for p in session.participants]:
                session.participants.append(client)
                await self.update(session)
                
                self._client_sessions[client.client_id] = session_id
                logger.info(f"Client {client.client_id} registered to session {session_id}")
                return True
        return False
    
    async def get_client_session(self, client_id: str) -> Optional[str]:
        """
        获取客户端所在的会话ID
        
        Args:
            client_id: 客户端ID
            
        Returns:
            会话ID或None
        """
        return self._client_sessions.get(client_id)
    
    async def get_session_participants(self, session_id: str) -> List[FLClientInfo]:
        """
        获取会话参与者
        
        Args:
            session_id: 会话ID
            
        Returns:
            参与者列表
        """
        session = await self.get(session_id)
        if session:
            return session.participants
        return []
    
    def clear(self):
        """清空所有会话"""
        self._sessions.clear()
        self._client_sessions.clear()
        logger.info("All sessions cleared")
    
    @property
    def session_count(self) -> int:
        """获取会话数量"""
        return len(self._sessions)


class InMemoryStore(SessionStore):
    """内存存储的别名"""
    pass
