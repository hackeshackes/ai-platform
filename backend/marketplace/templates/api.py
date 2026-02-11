"""
模板市场API接口
提供模板浏览、搜索、下载等功能
"""
import json
import os
from typing import Dict, Any, List, Optional
from pathlib import Path


class TemplateAPI:
    """模板市场API"""
    
    def __init__(self, templates_dir: str = None):
        self.templates_dir = templates_dir or os.path.dirname(__file__)
        self.index_file = os.path.join(self.templates_dir, "index.json")
        self._index = None
        
    @property
    def index(self) -> Dict:
        """加载模板索引"""
        if self._index is None:
            with open(self.index_file, 'r', encoding='utf-8') as f:
                self._index = json.load(f)
        return self._index
    
    def list_categories(self) -> List[Dict]:
        """列出所有分类"""
        return [
            {
                "id": cat["id"],
                "name": cat["name"],
                "description": cat["description"],
                "template_count": len(cat["templates"]),
                "icon": cat.get("icon", "box")
            }
            for cat in self.index["categories"]
        ]
    
    def get_category(self, category_id: str) -> Optional[Dict]:
        """获取分类详情"""
        for cat in self.index["categories"]:
            if cat["id"] == category_id:
                return cat
        return None
    
    def list_templates(
        self,
        category: str = None,
        tags: List[str] = None,
        sort_by: str = "downloads",
        order: str = "desc",
        page: int = 1,
        page_size: int = 20
    ) -> Dict[str, Any]:
        """列出模板"""
        templates = []
        
        for cat in self.index["categories"]:
            if category and cat["id"] != category:
                continue
            templates.extend(cat["templates"])
        
        # 过滤标签
        if tags:
            templates = [
                t for t in templates
                if any(tag in t.get("tags", []) for tag in tags)
            ]
        
        # 排序
        reverse = order == "desc"
        if sort_by in ["downloads", "rating", "name"]:
            templates.sort(key=lambda x: x.get(sort_by, 0), reverse=reverse)
        
        # 分页
        total = len(templates)
        start = (page - 1) * page_size
        end = start + page_size
        paginated = templates[start:end]
        
        return {
            "templates": paginated,
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": (total + page_size - 1) // page_size
        }
    
    def get_template(self, template_id: str) -> Optional[Dict]:
        """获取模板详情"""
        for cat in self.index["categories"]:
            for template in cat["templates"]:
                if template["id"] == template_id:
                    return template
        return None
    
    def search_templates(self, query: str, limit: int = 10) -> List[Dict]:
        """搜索模板"""
        results = []
        query_lower = query.lower()
        
        for cat in self.index["categories"]:
            for template in cat["templates"]:
                score = 0
                if query_lower in template["name"].lower():
                    score += 10
                if query_lower in template.get("description", "").lower():
                    score += 5
                if any(query_lower in tag.lower() for tag in template.get("tags", [])):
                    score += 3
                
                if score > 0:
                    template_copy = template.copy()
                    template_copy["search_score"] = score
                    results.append(template_copy)
        
        results.sort(key=lambda x: x["search_score"], reverse=True)
        return results[:limit]
    
    def get_template_files(self, template_id: str) -> List[str]:
        """获取模板文件列表"""
        template = self.get_template(template_id)
        if not template:
            return []
        
        base_dir = self.templates_dir
        files = []
        
        for key in ["pipeline", "config"]:
            if key in template:
                path = os.path.join(base_dir, template[key])
                if os.path.exists(path):
                    files.append(template[key])
        
        return files
    
    def get_statistics(self) -> Dict:
        """获取市场统计"""
        stats = self.index.get("statistics", {})
        stats["categories_count"] = len(self.index["categories"])
        stats["templates_count"] = self.index["total_templates"]
        return stats


# 便捷函数
def create_api(templates_dir: str = None) -> TemplateAPI:
    """创建API实例"""
    return TemplateAPI(templates_dir)


# Flask路由示例
def register_routes(app):
    """注册Flask路由"""
    api = create_api()
    
    @app.route("/api/categories")
    def categories():
        return {"categories": api.list_categories()}
    
    @app.route("/api/categories/<category_id>")
    def category_detail(category_id):
        cat = api.get_category(category_id)
        if cat:
            return {"category": cat}
        return {"error": "Category not found"}, 404
    
    @app.route("/api/templates")
    def templates():
        return api.list_templates(
            category=request.args.get("category"),
            tags=request.args.getlist("tags"),
            sort_by=request.args.get("sort_by", "downloads"),
            page=int(request.args.get("page", 1)),
            page_size=int(request.args.get("page_size", 20))
        )
    
    @app.route("/api/templates/<template_id>")
    def template_detail(template_id):
        template = api.get_template(template_id)
        if template:
            return {"template": template}
        return {"error": "Template not found"}, 404
    
    @app.route("/api/templates/<template_id>/files")
    def template_files(template_id):
        files = api.get_template_files(template_id)
        return {"files": files}
    
    @app.route("/api/search")
    def search():
        query = request.args.get("q", "")
        return {"results": api.search_templates(query)}
    
    @app.route("/api/statistics")
    def statistics():
        return api.get_statistics()
