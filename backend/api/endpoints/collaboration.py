"""
Collaboration API端点 v2.4
"""
from fastapi import APIRouter, HTTPException
from typing import List, Optional, Dict, Any

# 直接导入模块
import importlib.util
import os

backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
module_path = os.path.join(backend_dir, 'collaboration/engine.py')

spec = importlib.util.spec_from_file_location("collab_module", module_path)
module = importlib.util.module_from_spec(spec)

try:
    spec.loader.exec_module(module)
    collaboration_engine = module.collaboration_engine
    TeamRole = module.TeamRole
    ReviewStatus = module.ReviewStatus
    NotificationType = module.NotificationType
except Exception as e:
    print(f"Failed to import collaboration module: {e}")
    collaboration_engine = None
    TeamRole = None
    ReviewStatus = None
    NotificationType = None

router = APIRouter()

from pydantic import BaseModel

class CreateTeamModel(BaseModel):
    name: str
    description: str

class AddMemberModel(BaseModel):
    user_id: str
    role: str = "member"

class CreateReviewModel(BaseModel):
    title: str
    description: str
    review_type: str
    target_id: str
    reviewers: List[str]

class AddCommentModel(BaseModel):
    user_id: str
    content: str
    line_number: Optional[int] = None
    parent_id: Optional[str] = None

class CreateNotificationModel(BaseModel):
    user_id: str
    notification_type: str
    title: str
    message: str
    link: Optional[str] = None

# ==================== 团队管理 ====================

@router.get("/teams")
async def list_teams(user_id: Optional[str] = None):
    """列出团队"""
    teams = collaboration_engine.list_teams(user_id=user_id)
    
    return {
        "total": len(teams),
        "teams": [
            {
                "team_id": t.team_id,
                "name": t.name,
                "description": t.description,
                "members_count": len(t.members),
                "created_by": t.created_by
            }
            for t in teams
        ]
    }

@router.post("/teams")
async def create_team(request: CreateTeamModel):
    """创建团队"""
    team = collaboration_engine.create_team(
        name=request.name,
        description=request.description,
        created_by="user"
    )
    
    return {
        "team_id": team.team_id,
        "name": team.name,
        "message": "Team created"
    }

@router.get("/teams/{team_id}")
async def get_team(team_id: str):
    """获取团队"""
    team = collaboration_engine.get_team(team_id)
    if not team:
        raise HTTPException(status_code=404, detail="Team not found")
    
    return {
        "team_id": team.team_id,
        "name": team.name,
        "description": team.description,
        "members": team.members,
        "projects": team.projects
    }

@router.put("/teams/{team_id}")
async def update_team(team_id: str, name: Optional[str] = None, description: Optional[str] = None):
    """更新团队"""
    result = collaboration_engine.update_team(team_id, name=name, description=description)
    if not result:
        raise HTTPException(status_code=404, detail="Team not found")
    return {"message": "Team updated"}

@router.delete("/teams/{team_id}")
async def delete_team(team_id: str):
    """删除团队"""
    result = collaboration_engine.delete_team(team_id)
    if not result:
        raise HTTPException(status_code=404, detail="Team not found")
    return {"message": "Team deleted"}

# ==================== 成员管理 ====================

@router.post("/teams/{team_id}/members")
async def add_member(team_id: str, request: AddMemberModel):
    """添加成员"""
    try:
        role = TeamRole(request.role)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid role: {request.role}")
    
    try:
        member = collaboration_engine.add_member(
            team_id=team_id,
            user_id=request.user_id,
            role=role
        )
        return {
            "member_id": member.member_id,
            "user_id": member.user_id,
            "role": member.role.value,
            "message": "Member added"
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.delete("/teams/{team_id}/members/{user_id}")
async def remove_member(team_id: str, user_id: str):
    """移除成员"""
    result = collaboration_engine.remove_member(team_id, user_id)
    if not result:
        raise HTTPException(status_code=404, detail="Member not found")
    return {"message": "Member removed"}

@router.put("/teams/{team_id}/members/{user_id}")
async def update_member_role(team_id: str, user_id: str, role: str):
    """更新成员角色"""
    try:
        team_role = TeamRole(role)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid role: {role}")
    
    result = collaboration_engine.update_member_role(team_id, user_id, team_role)
    if not result:
        raise HTTPException(status_code=404, detail="Member not found")
    return {"message": "Member role updated"}

# ==================== 评审系统 ====================

@router.get("/reviews")
async def list_reviews(status: Optional[str] = None, reviewer: Optional[str] = None):
    """列出评审"""
    rstatus = ReviewStatus(status) if status else None
    reviews = collaboration_engine.list_reviews(status=rstatus, reviewer=reviewer)
    
    return {
        "total": len(reviews),
        "reviews": [
            {
                "review_id": r.review_id,
                "title": r.title,
                "type": r.review_type,
                "status": r.status.value,
                "requested_by": r.requested_by,
                "reviewers_count": len(r.reviewers)
            }
            for r in reviews
        ]
    }

@router.post("/reviews")
async def create_review(request: CreateReviewModel):
    """创建评审"""
    review = collaboration_engine.create_review(
        title=request.title,
        description=request.description,
        review_type=request.review_type,
        target_id=request.target_id,
        requested_by="user",
        reviewers=request.reviewers
    )
    
    return {
        "review_id": review.review_id,
        "title": review.title,
        "message": "Review created"
    }

@router.get("/reviews/{review_id}")
async def get_review(review_id: str):
    """获取评审"""
    review = collaboration_engine.get_review(review_id)
    if not review:
        raise HTTPException(status_code=404, detail="Review not found")
    
    return {
        "review_id": review.review_id,
        "title": review.title,
        "description": review.description,
        "type": review.review_type,
        "status": review.status.value,
        "comments_count": len(review.comments),
        "requested_by": review.requested_by,
        "reviewers": review.reviewers
    }

@router.post("/reviews/{review_id}/comments")
async def add_comment(review_id: str, request: AddCommentModel):
    """添加评论"""
    try:
        comment = collaboration_engine.add_comment(
            review_id=review_id,
            user_id=request.user_id,
            content=request.content,
            line_number=request.line_number,
            parent_id=request.parent_id
        )
        return {
            "comment_id": comment.comment_id,
            "message": "Comment added"
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.post("/reviews/{review_id}/status")
async def update_review_status(review_id: str, status: str):
    """更新评审状态"""
    try:
        rstatus = ReviewStatus(status)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid status: {status}")
    
    result = collaboration_engine.update_review_status(review_id, rstatus)
    if not result:
        raise HTTPException(status_code=404, detail="Review not found")
    return {"message": f"Review status updated to {status}"}

# ==================== 通知系统 ====================

@router.get("/notifications")
async def get_notifications(user_id: str, unread_only: bool = False):
    """获取通知"""
    notifications = collaboration_engine.get_notifications(user_id, unread_only)
    
    return {
        "total": len(notifications),
        "unread": len([n for n in notifications if not n.read]),
        "notifications": [
            {
                "notification_id": n.notification_id,
                "type": n.notification_type.value,
                "title": n.title,
                "message": n.message,
                "link": n.link,
                "read": n.read,
                "created_at": n.created_at.isoformat()
            }
            for n in notifications
        ]
    }

@router.post("/notifications/{notification_id}/read")
async def mark_as_read(notification_id: str):
    """标记为已读"""
    result = collaboration_engine.mark_as_read(notification_id)
    if not result:
        raise HTTPException(status_code=404, detail="Notification not found")
    return {"message": "Marked as read"}

@router.post("/notifications/read_all")
async def mark_all_as_read(user_id: str):
    """标记全部已读"""
    count = collaboration_engine.mark_all_as_read(user_id)
    return {"message": f"{count} notifications marked as read"}

# ==================== 审计日志 ====================

@router.get("/audit")
async def get_audit_logs(
    user_id: Optional[str] = None,
    resource_type: Optional[str] = None,
    limit: int = 100
):
    """获取审计日志"""
    logs = collaboration_engine.get_audit_logs(
        user_id=user_id,
        resource_type=resource_type,
        limit=limit
    )
    
    return {
        "total": len(logs),
        "logs": [
            {
                "log_id": l.log_id,
                "user_id": l.user_id,
                "action": l.action,
                "resource_type": l.resource_type,
                "resource_id": l.resource_id,
                "created_at": l.created_at.isoformat()
            }
            for l in logs
        ]
    }

@router.post("/audit")
async def log_action(
    user_id: str,
    action: str,
    resource_type: str,
    resource_id: str,
    details: Optional[Dict] = None
):
    """记录操作"""
    collaboration_engine.log_action(
        user_id=user_id,
        action=action,
        resource_type=resource_type,
        resource_id=resource_id,
        details=details
    )
    return {"message": "Action logged"}

# ==================== 统计信息 ====================

@router.get("/summary")
async def get_summary():
    """获取统计"""
    return collaboration_engine.get_summary()

@router.get("/health")
async def collaboration_health():
    """健康检查"""
    return {
        "status": "healthy",
        "teams": len(collaboration_engine.teams),
        "reviews": len(collaboration_engine.reviews)
    }
