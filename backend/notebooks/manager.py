"""
Notebooks模块
"""
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4
import json

@dataclass
class Notebook:
    """Notebook"""
    notebook_id: str
    name: str
    description: str
    owner_id: str
    cells: List[Dict] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

class NotebookManager:
    """Notebook管理器"""
    
    def __init__(self):
        self.notebooks: Dict[str, Notebook] = {}
        self.executions: Dict[str, Dict] = {}
    
    async def create_notebook(
        self,
        name: str,
        description: str,
        owner_id: str
    ) -> Notebook:
        """创建Notebook"""
        notebook = Notebook(
            notebook_id=str(uuid4()),
            name=name,
            description=description,
            owner_id=owner_id
        )
        
        self.notebooks[notebook.notebook_id] = notebook
        return notebook
    
    async def get_notebook(self, notebook_id: str) -> Optional[Notebook]:
        """获取Notebook"""
        return self.notebooks.get(notebook_id)
    
    async def list_notebooks(
        self,
        owner_id: Optional[str] = None,
        skip: int = 0,
        limit: int = 100
    ) -> List[Notebook]:
        """列出Notebooks"""
        notebooks = list(self.notebooks.values())
        
        if owner_id:
            notebooks = [n for n in notebooks if n.owner_id == owner_id]
        
        return notebooks[skip:skip+limit]
    
    async def update_notebook(
        self,
        notebook_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        cells: Optional[List[Dict]] = None
    ) -> Notebook:
        """更新Notebook"""
        notebook = self.notebooks.get(notebook_id)
        if not notebook:
            raise ValueError(f"Notebook {notebook_id} not found")
        
        if name:
            notebook.name = name
        if description:
            notebook.description = description
        if cells:
            notebook.cells = cells
        
        notebook.updated_at = datetime.utcnow()
        return notebook
    
    async def delete_notebook(self, notebook_id: str) -> bool:
        """删除Notebook"""
        if notebook_id in self.notebooks:
            del self.notebooks[notebook_id]
            return True
        return False
    
    async def execute_cell(
        self,
        notebook_id: str,
        cell_index: int,
        code: str
    ) -> Dict[str, Any]:
        """执行Cell"""
        notebook = self.notebooks.get(notebook_id)
        if not notebook:
            raise ValueError(f"Notebook {notebook_id} not found")
        
        if cell_index >= len(notebook.cells):
            raise ValueError(f"Cell {cell_index} not found")
        
        # 模拟执行
        execution_id = str(uuid4())
        
        # 简化: 模拟输出
        result = {
            "execution_id": execution_id,
            "cell_index": cell_index,
            "status": "completed",
            "output": {
                "output_type": "stream",
                "text": f"Executed: {code[:50]}..."
            },
            "executed_at": datetime.utcnow().isoformat()
        }
        
        # 记录执行
        self.executions[execution_id] = result
        
        # 更新cell
        notebook.cells[cell_index]["execution_id"] = execution_id
        notebook.cells[cell_index]["last_output"] = result["output"]
        
        return result
    
    async def run_all_cells(
        self,
        notebook_id: str
    ) -> List[Dict[str, Any]]:
        """运行所有Cells"""
        notebook = self.notebooks.get(notebook_id)
        if not notebook:
            raise ValueError(f"Notebook {notebook_id} not found")
        
        results = []
        for i, cell in enumerate(notebook.cells):
            if cell.get("cell_type") == "code":
                code = cell.get("source", "")
                try:
                    result = await self.execute_cell(notebook_id, i, code)
                    results.append(result)
                except Exception as e:
                    results.append({
                        "cell_index": i,
                        "status": "error",
                        "error": str(e)
                    })
        
        return results
    
    async def add_cell(
        self,
        notebook_id: str,
        cell_type: str,  # code, markdown
        source: str
    ) -> Notebook:
        """添加Cell"""
        notebook = self.notebooks.get(notebook_id)
        if not notebook:
            raise ValueError(f"Notebook {notebook_id} not found")
        
        notebook.cells.append({
            "cell_type": cell_type,
            "source": source,
            "outputs": [],
            "metadata": {}
        })
        
        notebook.updated_at = datetime.utcnow()
        return notebook
    
    async def share_notebook(
        self,
        notebook_id: str,
        share_with: List[str]  # user_ids
    ) -> Dict[str, Any]:
        """分享Notebook"""
        notebook = self.notebooks.get(notebook_id)
        if not notebook:
            raise ValueError(f"Notebook {notebook_id} not found")
        
        notebook.metadata["shared_with"] = share_with
        
        return {
            "notebook_id": notebook_id,
            "shared_with": share_with,
            "share_link": f"/notebooks/shared/{notebook_id}"
        }

# Notebook Manager实例
notebook_manager = NotebookManager()
