"""
CLI API端点 v2.4
"""
from fastapi import APIRouter, HTTPException
from typing import List, Optional, Dict, Any

# 直接导入模块
import importlib.util
import os

backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
module_path = os.path.join(backend_dir, 'cli/engine.py')

spec = importlib.util.spec_from_file_location("cli_module", module_path)
module = importlib.util.module_from_spec(spec)

try:
    spec.loader.exec_module(module)
    cli_engine = module.cli_engine
    CommandCategory = module.CommandCategory
except Exception as e:
    print(f"Failed to import CLI module: {e}")
    cli_engine = None
    CommandCategory = None

router = APIRouter()

from pydantic import BaseModel

class RegisterCommandModel(BaseModel):
    name: str
    category: str
    description: str
    command_template: str
    options: Optional[List[Dict]] = None
    examples: Optional[List[str]] = None

class ExecuteCommandModel(BaseModel):
    options: Dict[str, Any]
    dry_run: bool = False

class CreateTemplateModel(BaseModel):
    name: str
    description: str
    category: str
    content: str
    variables: Optional[List[Dict]] = None

class GenerateScriptModel(BaseModel):
    variables: Dict[str, Any]

# ==================== 命令管理 ====================

@router.get("/commands")
async def list_commands(category: Optional[str] = None):
    """列出命令"""
    ctype = CommandCategory(category) if category else None
    commands = cli_engine.list_commands(category=ctype)
    
    return {
        "total": len(commands),
        "commands": [
            {
                "command_id": c.command_id,
                "name": c.name,
                "category": c.category.value,
                "description": c.description,
                "options_count": len(c.options)
            }
            for c in commands
        ]
    }

@router.get("/commands/{command_id}")
async def get_command(command_id: str):
    """获取命令详情"""
    command = cli_engine.get_command(command_id)
    if not command:
        raise HTTPException(status_code=404, detail="Command not found")
    
    return {
        "command_id": command.command_id,
        "name": command.name,
        "category": command.category.value,
        "description": command.description,
        "template": command.command_template,
        "options": [
            {
                "name": o.name,
                "type": o.type,
                "required": o.required,
                "description": o.description
            }
            for o in command.options
        ],
        "examples": command.examples
    }

@router.post("/commands/register")
async def register_command(request: RegisterCommandModel):
    """注册命令"""
    try:
        ctype = CommandCategory(request.category)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid category: {request.category}")
    
    command = cli_engine.register_command(
        name=request.name,
        category=ctype,
        description=request.description,
        command_template=request.command_template,
        options=request.options,
        examples=request.examples
    )
    
    return {
        "command_id": command.command_id,
        "name": command.name,
        "message": "Command registered"
    }

@router.post("/commands/{command_id}/execute")
async def execute_command(command_id: str, request: ExecuteCommandModel):
    """执行命令"""
    try:
        result = cli_engine.execute_command(
            command_id=command_id,
            options=request.options,
            dry_run=request.dry_run
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.get("/commands/{command_id}/generate")
async def generate_command(command_id: str, options: Dict[str, Any]):
    """生成命令"""
    try:
        cmd = cli_engine.generate_command(command_id, options)
        return {"command": cmd}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.get("/completions")
async def get_completions(partial: str, category: Optional[str] = None):
    """获取补全建议"""
    ctype = CommandCategory(category) if category else None
    completions = cli_engine.get_completions(partial, category=ctype)
    return {"completions": completions}

# ==================== 脚本模板 ====================

@router.get("/templates")
async def list_templates(category: Optional[str] = None):
    """列出模板"""
    templates = cli_engine.list_templates(category=category)
    
    return {
        "total": len(templates),
        "templates": [
            {
                "template_id": t.template_id,
                "name": t.name,
                "category": t.category,
                "description": t.description
            }
            for t in templates
        ]
    }

@router.post("/templates")
async def create_template(request: CreateTemplateModel):
    """创建模板"""
    template = cli_engine.create_template(
        name=request.name,
        description=request.description,
        category=request.category,
        content=request.content,
        variables=request.variables
    )
    
    return {
        "template_id": template.template_id,
        "name": template.name,
        "message": "Template created"
    }

@router.get("/templates/{template_id}")
async def get_template(template_id: str):
    """获取模板"""
    template = cli_engine.get_template(template_id)
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")
    
    return {
        "template_id": template.template_id,
        "name": template.name,
        "description": template.description,
        "category": template.category,
        "content": template.content,
        "variables": template.variables
    }

@router.post("/templates/{template_id}/generate")
async def generate_script(template_id: str, request: GenerateScriptModel):
    """生成脚本"""
    try:
        script = cli_engine.generate_script(template_id, request.variables)
        return {"script": script}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

# ==================== 历史记录 ====================

@router.get("/history")
async def get_history(limit: int = 50):
    """获取命令历史"""
    history = cli_engine.get_history(limit=limit)
    return {"total": len(history), "history": history}

@router.delete("/history")
async def clear_history():
    """清空历史"""
    cli_engine.clear_history()
    return {"message": "History cleared"}

# ==================== 统计信息 ====================

@router.get("/summary")
async def get_summary():
    """获取统计"""
    return cli_engine.get_summary()

@router.get("/health")
async def cli_health():
    """健康检查"""
    return {
        "status": "healthy",
        "commands": len(cli_engine.commands)
    }
