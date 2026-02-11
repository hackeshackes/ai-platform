"""
Markdown写入器
将生成的文档写入Markdown文件
"""

import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from .api_extractor import ModuleInfo
from .doc_generator import DocGenerator


class MarkdownWriter:
    """Markdown写入器"""
    
    def __init__(self, output_dir: str = "./docs", language: str = 'python'):
        self.output_dir = Path(output_dir)
        self.language = language
        self.doc_generator = DocGenerator(language)
        self._ensure_output_dir()
    
    def _ensure_output_dir(self):
        """确保输出目录存在"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def write_module_doc(self, module: ModuleInfo, filename: Optional[str] = None) -> str:
        """写入模块文档"""
        if filename is None:
            filename = f"{module.name}.md"
        
        filepath = self.output_dir / filename
        
        content = self.doc_generator.generate_module_doc(module)
        
        filepath.write_text(content, encoding='utf-8')
        
        return str(filepath)
    
    def write_full_doc(self, modules: List[ModuleInfo], filename: str = "README.md") -> str:
        """写入完整文档"""
        filepath = self.output_dir / filename
        
        content = self.doc_generator.generate_full_doc(modules)
        
        filepath.write_text(content, encoding='utf-8')
        
        return str(filepath)
    
    def write_api_reference(self, apis: List[Dict[str, Any]], filename: str = "api_reference.md") -> str:
        """写入API参考"""
        filepath = self.output_dir / filename
        
        # 转换为APIInfo对象
        from .api_extractor import APIInfo
        api_objects = []
        for api_dict in apis:
            api_objects.append(APIInfo(
                name=api_dict.get('name', ''),
                type=api_dict.get('type', 'function'),
                signature=api_dict.get('signature', ''),
                description=api_dict.get('description', ''),
                parameters=api_dict.get('parameters', []),
                returns=api_dict.get('returns', {}),
                examples=api_dict.get('examples', []),
                notes=api_dict.get('notes', []),
                raises=api_dict.get('raises', [])
            ))
        
        content = self.doc_generator.generate_api_reference(api_objects)
        
        filepath.write_text(content, encoding='utf-8')
        
        return str(filepath)
    
    def write_best_practices(self, apis: List[Dict[str, Any]], filename: str = "best_practices.md") -> str:
        """写入最佳实践"""
        filepath = self.output_dir / filename
        
        content = self.doc_generator.generate_best_practices([])
        
        filepath.write_text(content, encoding='utf-8')
        
        return str(filepath)
    
    def write_index(self, modules: List[ModuleInfo], title: str = "API Documentation") -> str:
        """写入索引文件"""
        filepath = self.output_dir / "SUMMARY.md"
        
        lines = []
        lines.append(f"# {title}\n")
        lines.append(f"Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        lines.append("---\n")
        lines.append("## Modules\n")
        lines.append("| Module | Description |")
        lines.append("|--------|-------------|")
        
        for module in modules:
            desc = module.description.split('\n')[0] if module.description else ""
            lines.append(f"| [{module.name}]({module.name}.md) | {desc} |")
        
        lines.append("")
        
        filepath.write_text('\n'.join(lines), encoding='utf-8')
        
        return str(filepath)
    
    def write_sidebar(self, modules: List[ModuleInfo], filename: str = "_sidebar.md") -> str:
        """写入侧边栏（用于docsify等）"""
        filepath = self.output_dir / filename
        
        lines = []
        lines.append("- [Home](README.md)")
        lines.append("- **Modules**")
        
        for module in modules:
            lines.append(f"  - [{module.name}]({module.name}.md)")
        
        filepath.write_text('\n'.join(lines), encoding='utf-8')
        
        return str(filepath)
    
    def generate_tree_structure(self, depth: int = 3) -> str:
        """生成目录结构"""
        lines = []
        
        for filepath in sorted(self.output_dir.rglob("*.md")):
            if filepath.name.startswith('_'):
                continue
            
            # 计算缩进
            rel_path = filepath.relative_to(self.output_dir)
            level = len(rel_path.parts) - 1
            indent = "  " * level
            
            # 显示文件名（不含扩展名）
            name = filepath.stem
            link = str(rel_path)
            
            lines.append(f"{indent}- [{name}]({link})")
        
        return '\n'.join(lines)
