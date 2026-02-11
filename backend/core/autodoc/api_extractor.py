"""
API提取器
从解析的代码中提取API信息
"""

from typing import Dict, List, Any, Optional
import re
from pathlib import Path
from dataclasses import dataclass, field
from .code_parser import CodeParser, ParsedFunction, ParsedClass


@dataclass
class APIInfo:
    """API信息"""
    name: str
    type: str  # "function" or "method"
    signature: str
    description: str
    parameters: List[Dict[str, Any]] = field(default_factory=list)
    returns: Dict[str, Any] = field(default_factory=dict)
    examples: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    raises: List[Dict[str, str]] = field(default_factory=list)
    location: str = ""


@dataclass
class ModuleInfo:
    """模块信息"""
    name: str
    description: str
    apis: List[APIInfo] = field(default_factory=list)
    classes: List[Dict[str, Any]] = field(default_factory=list)
    functions: List[Dict[str, Any]] = field(default_factory=list)


class APIExtractor:
    """API提取器"""
    
    def __init__(self, language: str = 'python'):
        self.parser = CodeParser(language)
        self.modules: Dict[str, ModuleInfo] = {}
    
    def extract_from_file(self, file_path: str) -> ModuleInfo:
        """从文件提取API"""
        parsed = self.parser.parse_file(file_path)
        return self._build_api_info(parsed, file_path)
    
    def extract_from_code(self, code: str, module_name: str = "module") -> ModuleInfo:
        """从代码字符串提取API"""
        parsed = self.parser.parse_code(code)
        return self._build_api_info(parsed, module_name)
    
    def extract_directory(self, dir_path: str, recursive: bool = True) -> List[ModuleInfo]:
        """从目录提取所有API"""
        import os
        from pathlib import Path
        
        modules = []
        path = Path(dir_path)
        
        for file_path in path.rglob("*.py") if self.parser.language == 'python' else []:
            try:
                module = self.extract_from_file(str(file_path))
                modules.append(module)
            except Exception as e:
                print(f"Warning: Failed to parse {file_path}: {e}")
        
        return modules
    
    def _build_api_info(self, parsed: Dict[str, Any], source: str) -> ModuleInfo:
        """构建API信息"""
        module = ModuleInfo(
            name=Path(source).stem if source else "module",
            description="",
            apis=[]
        )
        
        # 处理函数
        for func in parsed.get('functions', []):
            if isinstance(func, ParsedFunction):
                api = self._function_to_api(func)
            else:
                api = self._dict_to_api(func)
            module.apis.append(api)
            module.functions.append(api.__dict__ if hasattr(api, '__dict__') else api)
        
        # 处理类
        for cls in parsed.get('classes', []):
            if isinstance(cls, ParsedClass):
                class_info = self._class_to_dict(cls)
                module.classes.append(class_info)
                
                # 处理类的方法
                for method in cls.methods:
                    api = self._method_to_api(method, cls.name)
                    module.apis.append(api)
        
        return module
    
    def _function_to_api(self, func: ParsedFunction) -> APIInfo:
        """将ParsedFunction转换为APIInfo"""
        return APIInfo(
            name=func.name,
            type="function",
            signature=func.signature,
            description=func.docstring.split('\n')[0] if func.docstring else "",
            parameters=[{
                'name': p['name'],
                'type': p.get('type', ''),
                'required': p.get('required', True),
                'description': self._extract_param_description(func.docstring, p['name'])
            } for p in func.params],
            returns={
                'type': func.return_type,
                'description': func.return_doc or self._extract_return_description(func.docstring)
            },
            examples=[],
            notes=[]
        )
    
    def _dict_to_api(self, func_dict: Dict[str, Any]) -> APIInfo:
        """将字典转换为APIInfo"""
        return APIInfo(
            name=func_dict.get('name', ''),
            type="function",
            signature=func_dict.get('signature', ''),
            description=func_dict.get('docstring', '').split('\n')[0] if func_dict.get('docstring') else "",
            parameters=func_dict.get('params', []),
            returns={
                'type': func_dict.get('return_type', ''),
                'description': ''
            },
            examples=[],
            notes=[]
        )
    
    def _method_to_api(self, method: ParsedFunction, class_name: str) -> APIInfo:
        """将方法转换为APIInfo"""
        api = self._function_to_api(method)
        api.type = "method"
        api.name = f"{class_name}.{method.name}"
        return api
    
    def _class_to_dict(self, cls: ParsedClass) -> Dict[str, Any]:
        """将类转换为字典"""
        return {
            'name': cls.name,
            'description': cls.docstring.split('\n')[0] if cls.docstring else "",
            'full_description': cls.docstring,
            'bases': cls.bases,
            'methods': [m.name for m in cls.methods]
        }
    
    def _extract_param_description(self, docstring: str, param_name: str) -> str:
        """从文档字符串提取参数描述"""
        if not docstring:
            return ""
        
        lines = docstring.split('\n')
        for i, line in enumerate(lines):
            line_lower = line.lower().strip()
            if line_lower.startswith(f'param {param_name}') or \
               line_lower.startswith(f'{param_name}') or \
               (':' in line and param_name in line):
                # 提取描述
                parts = line.split(':', 1)
                if len(parts) > 1:
                    return parts[-1].strip()
                # 检查下一行
                if i + 1 < len(lines):
                    return lines[i + 1].strip()
        
        return ""
    
    def _extract_return_description(self, docstring: str) -> str:
        """从文档字符串提取返回描述"""
        if not docstring:
            return ""
        
        # 查找 "Returns:" 或 ":return:" 部分
        patterns = [
            r'Returns?:\s*(.*)',
            r':return[s]?:\s*(.*)',
            r'Return:\s*(.*)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, docstring, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()
        
        return ""
