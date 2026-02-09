"""
Collaboration 模块 v2.4
对标: W&B Teams, GitHub Projects
"""
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from uuid import uuid4

class TeamRole(str, Enum):
    """团队角色"""
    OWNER = "owner"
    ADMIN = "admin"
    MEMBER = "member"
    VIEWER = "viewer"

class ReviewStatus(str, Enum):
    """评审状态"""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    CHANGES_REQUESTED = "changes_requested"

class NotificationType(str, Enum):
    """通知类型"""
    MENTION = "mention"
    REVIEW_REQUESTED = "review_requested"
    REVIEW_COMPLETED = "review_completed"
    COMMENT = "comment"
    ASSIGNMENT = "assignment"
    DEPLOYMENT = "deployment"

@dataclass
class Team:
    """团队"""
    team_id: str
    name: str
    description: str
    members: List[Dict] = field(default_factory=list)
    projects: List[str] = field(default_factory=list)
    created_by: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class TeamMember:
    """团队成员"""
    member_id: str
    team_id: str
    user_id: str
    role: TeamRole
    joined_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class Review:
    """评审"""
    review_id: str
    title: str
    description: str
    review_type: str  # code, model, experiment
    target_id: str  # commit, model_version, experiment_id
    status: ReviewStatus
    comments: List[Dict] = field(default_factory=list)
    requested_by: str
    reviewers: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class ReviewComment:
    """评审评论"""
    comment_id: str
    review_id: str
    user_id: str
    content: str
    line_number: Optional[int] = None
    parent_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class Notification:
    """通知"""
    notification_id: str
    user_id: str
    notification_type: NotificationType
    title: str
    message: str
    link: Optional[str] = None
    read: bool = False
    created_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class AuditLog:
    """审计日志"""
    log_id: str
    user_id: str
    action: str
    resource_type: str
    resource_id: str
    details: Dict = field(default_factory=dict)
    ip_address: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)

class CollaborationEngine:
    """协作引擎 v2.4"""
    
    def __init__(self):
        self.teams: Dict[str, Team] = {}
        self.members: Dict[str, TeamMember] = {}
        self.reviews: Dict[str, Review] = {}
        self.comments: Dict[str, ReviewComment] = {}
        self.notifications: Dict[str, Notification] = {}
        self.audit_logs: List[AuditLog] = []
        
        # 初始化示例数据
        self._init_sample_data()
    
    def _init_sample_data(self):
        """初始化示例数据"""
        # 创建示例团队
        team = Team(
            team_id="default",
            name="AI Platform Team",
            description="主开发团队",
            members=[
                {"user_id": "user1", "role": "owner"},
                {"user_id": "user2", "role": "member"}
            ],
            created_by="user1"
        )
        self.teams[team.team_id] = team
    
    # ==================== 团队管理 ====================
    
    def create_team(
        self,
        name: str,
        description: str,
        created_by: str
    ) -> Team:
        """创建团队"""
        team = Team(
            team_id=str(uuid4()),
            name=name,
            description=description,
            created_by=created_by
        )
        
        self.teams[team.team_id] = team
        
        # 添加创建者为owner
        member = TeamMember(
            member_id=str(uuid4()),
            team_id=team.team_id,
            user_id=created_by,
            role=TeamRole.OWNER
        )
        self.members[member.member_id] = member
        
        return team
    
    def get_team(self, team_id: str) -> Optional[Team]:
        """获取团队"""
        return self.teams.get(team_id)
    
    def list_teams(self, user_id: Optional[str] = None) -> List[Team]:
        """列出团队"""
        if user_id:
            # 过滤用户所在的团队
            user_teams = []
            for team in self.teams.values():
                for member in team.members:
                    if member.get("user_id") == user_id:
                        user_teams.append(team)
                        break
            return user_teams
        return list(self.teams.values())
    
    def update_team(
        self,
        team_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None
    ) -> bool:
        """更新团队"""
        team = self.teams.get(team_id)
        if not team:
            return False
        
        if name:
            team.name = name
        if description:
            team.description = description
        team.updated_at = datetime.utcnow()
        return True
    
    def delete_team(self, team_id: str) -> bool:
        """删除团队"""
        if team_id in self.teams:
            del self.teams[team_id]
            return True
        return False
    
    # ==================== 成员管理 ====================
    
    def add_member(
        self,
        team_id: str,
        user_id: str,
        role: TeamRole = TeamRole.MEMBER
    ) -> TeamMember:
        """添加成员"""
        team = self.teams.get(team_id)
        if not team:
            raise ValueError(f"Team {team_id} not found")
        
        # 检查是否已是成员
        for member in team.members:
            if member.get("user_id") == user_id:
                raise ValueError(f"User {user_id} is already a member")
        
        member = TeamMember(
            member_id=str(uuid4()),
            team_id=team_id,
            user_id=user_id,
            role=role
        )
        
        self.members[member.member_id] = member
        team.members.append({"user_id": user_id, "role": role.value})
        
        return member
    
    def remove_member(self, team_id: str, user_id: str) -> bool:
        """移除成员"""
        team = self.teams.get(team_id)
        if not team:
            return False
        
        for i, member in enumerate(team.members):
            if member.get("user_id") == user_id:
                team.members.pop(i)
                return True
        return False
    
    def update_member_role(
        self,
        team_id: str,
        user_id: str,
        role: TeamRole
    ) -> bool:
        """更新成员角色"""
        team = self.teams.get(team_id)
        if not team:
            return False
        
        for member in team.members:
            if member.get("user_id") == user_id:
                member["role"] = role.value
                return True
        return False
    
    # ==================== 评审系统 ====================
    
    def create_review(
        self,
        title: str,
        description: str,
        review_type: str,
        target_id: str,
        requested_by: str,
        reviewers: List[str]
    ) -> Review:
        """创建评审"""
        review = Review(
            review_id=str(uuid4()),
            title=title,
            description=description,
            review_type=review_type,
            target_id=target_id,
            status=ReviewStatus.PENDING,
            requested_by=requested_by,
            reviewers=reviewers
        )
        
        self.reviews[review.review_id] = review
        
        # 发送通知
        for reviewer in reviewers:
            self.create_notification(
                user_id=reviewer,
                notification_type=NotificationType.REVIEW_REQUESTED,
                title="Review Requested",
                message=f"You have been requested to review: {title}",
                link=f"/reviews/{review.review_id}"
            )
        
        return review
    
    def get_review(self, review_id: str) -> Optional[Review]:
        """获取评审"""
        return self.reviews.get(review_id)
    
    def list_reviews(
        self,
        status: Optional[ReviewStatus] = None,
        reviewer: Optional[str] = None
    ) -> List[Review]:
        """列出评审"""
        reviews = list(self.reviews.values())
        if status:
            reviews = [r for r in reviews if r.status == status]
        if reviewer:
            reviews = [r for r in reviews if reviewer in r.reviewers]
        return reviews
    
    def add_comment(
        self,
        review_id: str,
        user_id: str,
        content: str,
        line_number: Optional[int] = None,
        parent_id: Optional[str] = None
    ) -> ReviewComment:
        """添加评论"""
        review = self.reviews.get(review_id)
        if not review:
            raise ValueError(f"Review {review_id} not found")
        
        comment = ReviewComment(
            comment_id=str(uuid4()),
            review_id=review_id,
            user_id=user_id,
            content=content,
            line_number=line_number,
            parent_id=parent_id
        )
        
        self.comments[comment.comment_id] = comment
        review.comments.append(comment)
        
        # 发送通知
        self.create_notification(
            user_id=review.requested_by,
            notification_type=NotificationType.COMMENT,
            title="New Comment",
            message=f"New comment on your review: {title}",
            link=f"/reviews/{review_id}"
        )
        
        return comment
    
    def update_review_status(
        self,
        review_id: str,
        status: ReviewStatus
    ) -> bool:
        """更新评审状态"""
        review = self.reviews.get(review_id)
        if not review:
            return False
        
        review.status = status
        review.updated_at = datetime.utcnow()
        
        # 发送通知
        self.create_notification(
            user_id=review.requested_by,
            notification_type=NotificationType.REVIEW_COMPLETED,
            title="Review Completed",
            message=f"Your review has been {status.value}: {review.title}",
            link=f"/reviews/{review_id}"
        )
        
        return True
    
    # ==================== 通知系统 ====================
    
    def create_notification(
        self,
        user_id: str,
        notification_type: NotificationType,
        title: str,
        message: str,
        link: Optional[str] = None
    ) -> Notification:
        """创建通知"""
        notification = Notification(
            notification_id=str(uuid4()),
            user_id=user_id,
            notification_type=notification_type,
            title=title,
            message=message,
            link=link
        )
        
        self.notifications[notification.notification_id] = notification
        return notification
    
    def get_notifications(
        self,
        user_id: str,
        unread_only: bool = False
    ) -> List[Notification]:
        """获取通知"""
        notifications = [
            n for n in self.notifications.values()
            if n.user_id == user_id
        ]
        if unread_only:
            notifications = [n for n in notifications if not n.read]
        return sorted(notifications, key=lambda n: n.created_at, reverse=True)
    
    def mark_as_read(self, notification_id: str) -> bool:
        """标记为已读"""
        notification = self.notifications.get(notification_id)
        if not notification:
            return False
        
        notification.read = True
        return True
    
    def mark_all_as_read(self, user_id: str) -> int:
        """标记全部已读"""
        count = 0
        for notification in self.notifications.values():
            if notification.user_id == user_id and not notification.read:
                notification.read = True
                count += 1
        return count
    
    # ==================== 审计日志 ====================
    
    def log_action(
        self,
        user_id: str,
        action: str,
        resource_type: str,
        resource_id: str,
        details: Optional[Dict] = None,
        ip_address: Optional[str] = None
    ):
        """记录操作"""
        log = AuditLog(
            log_id=str(uuid4()),
            user_id=user_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            details=details or {},
            ip_address=ip_address
        )
        self.audit_logs.append(log)
    
    def get_audit_logs(
        self,
        user_id: Optional[str] = None,
        resource_type: Optional[str] = None,
        limit: int = 100
    ) -> List[AuditLog]:
        """获取审计日志"""
        logs = self.audit_logs
        if user_id:
            logs = [l for l in logs if l.user_id == user_id]
        if resource_type:
            logs = [l for l in logs if l.resource_type == resource_type]
        return logs[-limit:]
    
    # ==================== 统计信息 ====================
    
    def get_summary(self) -> Dict:
        """获取统计"""
        return {
            "total_teams": len(self.teams),
            "total_members": len(self.members),
            "total_reviews": len(self.reviews),
            "pending_reviews": len([r for r in self.reviews.values() if r.status == ReviewStatus.PENDING]),
            "total_notifications": len(self.notifications),
            "unread_notifications": len([n for n in self.notifications.values() if not n.read])
        }

# CollaborationEngine实例
collaboration_engine = CollaborationEngine()
