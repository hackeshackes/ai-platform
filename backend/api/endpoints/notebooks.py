"""
Notebooks API端点 v2.1
"""
from fastapi import APIRouter, HTTPException, Depends
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from datetime import datetime

from backend.notebooks.manager import notebook_manager
from api.endpoints.auth import get_current_user

router = APIRouter()

class CreateNotebookModel(BaseModel):
    name: str
    description: str = ""

class UpdateNotebookModel(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    cells: Optional[List[Dict[str, Any]]] = None

class AddCellModel(BaseModel):
    cell_type: str  # code, markdown
    source: str

class ExecuteCellModel(BaseModel):
    cell_index: int
    code: str

@router.post("")
async def create_notebook(
    request: CreateNotebookModel,
    current_user = Depends(get_current_user)
):
    """
    创建Notebook
    
    v2.1: Notebooks
    """
    try:
        notebook = await notebook_manager.create_notebook(
            name=request.name,
            description=request.description,
            owner_id=str(current_user.id)
        )
        
        return {
            "notebook_id": notebook.notebook_id,
            "name": notebook.name,
            "description": notebook.description,
            "owner_id": notebook.owner_id,
            "created_at": notebook.created_at.isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("")
async def list_notebooks(
    skip: int = 0,
    limit: int = 100,
    current_user = Depends(get_current_user)
):
    """
    列出Notebooks
    
    v2.1: Notebooks
    """
    notebooks = await notebook_manager.list_notebooks(
        owner_id=str(current_user.id),
        skip=skip,
        limit=limit
    )
    
    return {
        "total": len(notebooks),
        "notebooks": [
            {
                "notebook_id": n.notebook_id,
                "name": n.name,
                "description": n.description,
                "cells_count": len(n.cells),
                "created_at": n.created_at.isoformat(),
                "updated_at": n.updated_at.isoformat()
            }
            for n in notebooks
        ]
    }

@router.get("/{notebook_id}")
async def get_notebook(
    notebook_id: str,
    current_user = Depends(get_current_user)
):
    """
    获取Notebook详情
    
    v2.1: Notebooks
    """
    notebook = await notebook_manager.get_notebook(notebook_id)
    if not notebook:
        raise HTTPException(status_code=404, detail="Notebook not found")
    
    return {
        "notebook_id": notebook.notebook_id,
        "name": notebook.name,
        "description": notebook.description,
        "owner_id": notebook.owner_id,
        "cells": notebook.cells,
        "metadata": notebook.metadata,
        "created_at": notebook.created_at.isoformat(),
        "updated_at": notebook.updated_at.isoformat()
    }

@router.put("/{notebook_id}")
async def update_notebook(
    notebook_id: str,
    request: UpdateNotebookModel,
    current_user = Depends(get_current_user)
):
    """
    更新Notebook
    
    v2.1: Notebooks
    """
    try:
        notebook = await notebook_manager.update_notebook(
            notebook_id=notebook_id,
            name=request.name,
            description=request.description,
            cells=request.cells
        )
        
        return {
            "notebook_id": notebook.notebook_id,
            "name": notebook.name,
            "updated_at": notebook.updated_at.isoformat()
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.delete("/{notebook_id}")
async def delete_notebook(
    notebook_id: str,
    current_user = Depends(get_current_user)
):
    """
    删除Notebook
    
    v2.1: Notebooks
    """
    result = await notebook_manager.delete_notebook(notebook_id)
    if not result:
        raise HTTPException(status_code=404, detail="Notebook not found")
    
    return {"message": "Notebook deleted"}

@router.post("/{notebook_id}/cells")
async def add_cell(
    notebook_id: str,
    request: AddCellModel,
    current_user = Depends(get_current_user)
):
    """
    添加Cell
    
    v2.1: Notebooks
    """
    try:
        notebook = await notebook_manager.add_cell(
            notebook_id=notebook_id,
            cell_type=request.cell_type,
            source=request.source
        )
        
        return {
            "notebook_id": notebook.notebook_id,
            "cells_count": len(notebook.cells),
            "last_cell": notebook.cells[-1]
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.post("/{notebook_id}/cells/{cell_index}/execute")
async def execute_cell(
    notebook_id: str,
    request: ExecuteCellModel,
    current_user = Depends(get_current_user)
):
    """
    执行Cell
    
    v2.1: Notebooks
    """
    try:
        result = await notebook_manager.execute_cell(
            notebook_id=notebook_id,
            cell_index=request.cell_index,
            code=request.code
        )
        
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/{notebook_id}/run")
async def run_all_cells(
    notebook_id: str,
    current_user = Depends(get_current_user)
):
    """
    运行所有Cells
    
    v2.1: Notebooks
    """
    try:
        results = await notebook_manager.run_all_cells(notebook_id)
        
        return {
            "notebook_id": notebook_id,
            "total_cells": len(results),
            "results": results
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.post("/{notebook_id}/share")
async def share_notebook(
    notebook_id: str,
    share_with: List[str],
    current_user = Depends(get_current_user)
):
    """
    分享Notebook
    
    v2.1: Notebooks
    """
    try:
        result = await notebook_manager.share_notebook(
            notebook_id=notebook_id,
            share_with=share_with
        )
        
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.get("/{notebook_id}/export")
async def export_notebook(
    notebook_id: str,
    format: str = "json"
):
    """
    导出Notebook
    
    v2.1: Notebooks
    """
    notebook = await notebook_manager.get_notebook(notebook_id)
    if not notebook:
        raise HTTPException(status_code=404, detail="Notebook not found")
    
    if format == "json":
        return {
            "notebook": {
                "id": notebook.notebook_id,
                "name": notebook.name,
                "cells": notebook.cells
            }
        }
    else:
        raise HTTPException(status_code=400, detail="Unsupported format")
