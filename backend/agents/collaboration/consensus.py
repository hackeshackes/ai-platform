"""
共识机制模块
实现Agent协作中的共识决策算法
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

from .models import ConsensusProposal, AgentInfo

logger = logging.getLogger(__name__)


class ConsensusAlgorithm(str, Enum):
    """共识算法"""
    VOTING = "voting"               # 投票共识
    QUORUM = "quorum"               # 法定人数共识
    LEADER_ELECTION = "leader"      # 领导者选举
    ROUND_ROBIN = "round_robin"     # 轮询共识
    WEIGHTED = "weighted"           # 加权共识


@dataclass
class Vote:
    """投票"""
    voter_id: str
    vote: bool
    weight: float = 1.0
    reason: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ConsensusRound:
    """共识轮次"""
    round_number: int
    proposals: List[ConsensusProposal]
    votes: Dict[str, List[Vote]]
    quorum_reached: bool = False
    consensus_reached: bool = False
    winning_proposal: Optional[ConsensusProposal] = None


class ConsensusManager:
    """共识管理器"""
    
    def __init__(self):
        self.active_proposals: Dict[str, ConsensusProposal] = {}
        self.consensus_history: List[ConsensusRound] = []
        self.vote_records: Dict[str, List[Vote]] = {}
        self._lock = asyncio.Lock()
        self.threshold = 0.5  # 默认阈值
    
    async def create_proposal(
        self,
        session_id: str,
        proposer_id: str,
        proposal_type: str,
        content: Dict[str, Any],
        participants: List[str],
        threshold: Optional[float] = None,
        timeout_seconds: int = 60
    ) -> ConsensusProposal:
        """创建提案"""
        proposal = ConsensusProposal(
            session_id=session_id,
            proposer_id=proposer_id,
            proposal_type=proposal_type,
            content=content,
            threshold=threshold or self.threshold,
            expires_at=datetime.utcnow() + timedelta(seconds=timeout_seconds)
        )
        
        async with self._lock:
            self.active_proposals[proposal.proposal_id] = proposal
            self.vote_records[proposal.proposal_id] = []
        
        logger.info(f"Proposal created: {proposal.proposal_id}")
        return proposal
    
    async def submit_vote(
        self,
        proposal_id: str,
        voter_id: str,
        vote: bool,
        weight: float = 1.0,
        reason: str = ""
    ) -> bool:
        """提交投票"""
        if proposal_id not in self.active_proposals:
            logger.warning(f"Proposal {proposal_id} not found")
            return False
        
        proposal = self.active_proposals[proposal_id]
        
        # 检查是否过期
        if datetime.utcnow() > proposal.expires_at:
            proposal.status = "expired"
            return False
        
        # 创建投票记录
        vote_record = Vote(
            voter_id=voter_id,
            vote=vote,
            weight=weight,
            reason=reason
        )
        
        async with self._lock:
            self.vote_records[proposal_id].append(vote_record)
            proposal.votes[voter_id] = vote
        
        logger.debug(f"Vote submitted: {voter_id} -> {vote}")
        return True
    
    async def count_votes(self, proposal_id: str) -> Tuple[int, int, float]:
        """统计投票"""
        if proposal_id not in self.active_proposals:
            return 0, 0, 0.0
        
        proposal = self.active_proposals[proposal_id]
        votes = self.vote_records.get(proposal_id, [])
        
        yes_weight = sum(v.weight for v in votes if v.vote)
        no_weight = sum(v.weight for v in votes if not v.vote)
        total_weight = yes_weight + no_weight
        
        yes_ratio = yes_weight / total_weight if total_weight > 0 else 0.0
        
        return int(yes_weight), int(no_weight), yes_ratio
    
    async def check_consensus(
        self,
        proposal_id: str,
        algorithm: ConsensusAlgorithm = ConsensusAlgorithm.VOTING
    ) -> bool:
        """检查是否达成共识"""
        if proposal_id not in self.active_proposals:
            return False
        
        proposal = self.active_proposals[proposal_id]
        yes_weight, no_weight, yes_ratio = await self.count_votes(proposal_id)
        
        if algorithm == ConsensusAlgorithm.VOTING:
            return yes_ratio >= proposal.threshold
        elif algorithm == ConsensusAlgorithm.QUORUM:
            total_votes = yes_weight + no_weight
            required_votes = len(proposal.votes) * 0.6  # 60%参与率
            return yes_ratio >= proposal.threshold and total_votes >= required_votes
        elif algorithm == ConsensusAlgorithm.WEIGHTED:
            return yes_ratio >= proposal.threshold
        
        return False
    
    async def resolve_consensus(
        self,
        proposal_id: str,
        algorithm: ConsensusAlgorithm = ConsensusAlgorithm.VOTING
    ) -> Optional[ConsensusProposal]:
        """解决共识"""
        async with self._lock:
            if proposal_id not in self.active_proposals:
                return None
            
            proposal = self.active_proposals[proposal_id]
            
            # 检查是否达成共识
            if await self.check_consensus(proposal_id, algorithm):
                proposal.status = "accepted"
                consensus_reached = True
            else:
                proposal.status = "rejected"
                consensus_reached = False
            
            # 记录到历史
            round_info = ConsensusRound(
                round_number=len(self.consensus_history) + 1,
                proposals=[proposal],
                votes=self.vote_records.get(proposal_id, []),
                consensus_reached=consensus_reached
            )
            self.consensus_history.append(round_info)
            
            # 清理
            del self.active_proposals[proposal_id]
        
        return proposal
    
    async def run_consensus_round(
        self,
        session_id: str,
        proposals: List[ConsensusProposal],
        participants: List[AgentInfo],
        algorithm: ConsensusAlgorithm = ConsensusAlgorithm.VOTING,
        max_rounds: int = 3
    ) -> Optional[ConsensusProposal]:
        """运行多轮共识"""
        current_round = 0
        
        while current_round < max_rounds:
            current_round += 1
            logger.info(f"Consensus round {current_round}")
            
            # 检查每个提案
            for proposal in proposals:
                if proposal.status == "pending":
                    # 等待投票
                    await asyncio.sleep(1)
                    
                    # 检查共识
                    if await self.check_consensus(proposal.proposal_id, algorithm):
                        return await self.resolve_consensus(proposal.proposal_id, algorithm)
            
            # 如果没有达成共识，尝试下一轮
            if current_round < max_rounds:
                # 可以在这里实现更复杂的轮次逻辑
                await asyncio.sleep(2)
        
        # 超时，返回最早的提案
        if proposals:
            return await self.resolve_consensus(proposals[0].proposal_id, algorithm)
        
        return None
    
    async def elect_leader(
        self,
        session_id: str,
        candidates: List[AgentInfo],
        criteria: Dict[str, Any] = None
    ) -> Optional[AgentInfo]:
        """选举领导者"""
        if not candidates:
            return None
        
        criteria = criteria or {"weight": 1.0}
        
        # 基于权重选举
        best_candidate = candidates[0]
        best_score = 0
        
        for candidate in candidates:
            score = 0
            
            # 能力评分
            if "capabilities" in criteria:
                capability_score = len(candidate.capabilities) * criteria.get("capability_weight", 0.1)
                score += capability_score
            
            # 角色权重
            role_weights = {
                "coordinator": 1.0,
                "supervisor": 0.9,
                "reviewer": 0.7,
                "worker": 0.5
            }
            score += role_weights.get(candidate.role.value, 0.5)
            
            if score > best_score:
                best_score = score
                best_candidate = candidate
        
        logger.info(f"Leader elected: {best_candidate.agent_id}")
        return best_candidate
    
    def get_proposal_status(self, proposal_id: str) -> Dict[str, Any]:
        """获取提案状态"""
        if proposal_id not in self.active_proposals:
            return {"status": "not_found"}
        
        proposal = self.active_proposals[proposal_id]
        votes = self.vote_records.get(proposal_id, [])
        
        yes_count = sum(1 for v in votes if v.vote)
        no_count = len(votes) - yes_count
        
        return {
            "proposal": proposal,
            "vote_count": {
                "yes": yes_count,
                "no": no_count,
                "total": len(votes)
            },
            "status": proposal.status
        }


class VotingConsensus:
    """投票共识实现"""
    
    def __init__(self):
        self.votes: Dict[str, Dict[str, bool]] = {}
    
    async def propose(
        self,
        session_id: str,
        proposal_id: str,
        options: List[str]
    ) -> Dict[str, Any]:
        """发起投票"""
        self.votes[proposal_id] = {opt: 0 for opt in options}
        return {"proposal_id": proposal_id, "options": options}
    
    async def cast_vote(
        self,
        proposal_id: str,
        voter_id: str,
        choice: str
    ) -> bool:
        """投票"""
        if proposal_id not in self.votes:
            return False
        
        if choice not in self.votes[proposal_id]:
            return False
        
        self.votes[proposal_id][choice] += 1
        return True
    
    async def get_results(self, proposal_id: str) -> Dict[str, Any]:
        """获取投票结果"""
        if proposal_id not in self.votes:
            return {"error": "Proposal not found"}
        
        votes = self.votes[proposal_id]
        total = sum(votes.values())
        
        return {
            "proposal_id": proposal_id,
            "votes": votes,
            "total_votes": total,
            "winner": max(votes, key=votes.get) if total > 0 else None,
            "distribution": {k: v/total if total > 0 else 0 for k, v in votes.items()}
        }


def create_consensus_manager() -> ConsensusManager:
    """创建共识管理器"""
    return ConsensusManager()


def create_voting_system() -> VotingConsensus:
    """创建投票系统"""
    return VotingConsensus()
