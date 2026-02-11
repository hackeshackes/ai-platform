"""
代码解析器
使用AST解析Python/TypeScript/Go代码
"""

import ast
import re
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


@dataclass
class ParsedFunction:
    """解析后的函数信息"""
    name: str
    signature: str
    docstring: str
    params: List[Dict[str, Any]] = field(default_factory=list)
    return_type: str = ""
    return_doc: str = ""
    decorators: List[str] = field(default_factory=list)
    is_async: bool = False


@dataclass  
class ParsedClass:
    """解析后的类信息"""
    name: str
    docstring: str
    methods: List[ParsedFunction] = field(default_factory=list)
    bases: List[str] = field(default_factory=list)


class CodeParser:
    """代码解析器"""
    
    SUPPORTED_LANGUAGES = ['python', 'typescript', 'javascript', 'go']
    
    def __init__(self, language: str = 'python'):
        if language.lower() not in self.SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language: {language}")
        self.language = language.lower()
    
    def parse_file(self, file_path: str) -> Dict[str, Any]:
        """解析代码文件"""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        content = path.read_text(encoding='utf-8')
        
        if self.language == 'python':
            return self._parse_python(content)
        elif self.language in ['typescript', 'javascript']:
            return self._parse_typescript(content)
        elif self.language == 'go':
            return self._parse_go(content)
        
        return {}
    
    def parse_code(self, code: str) -> Dict[str, Any]:
        """解析代码字符串"""
        if self.language == 'python':
            return self._parse_python(code)
        elif self.language in ['typescript', 'javascript']:
            return self._parse_typescript(code)
        elif self.language == 'go':
            return self._parse_go(code)
        
        return {}
    
    def _parse_python(self, code: str) -> Dict[str, Any]:
        """解析Python代码"""
        tree = ast.parse(code)
        classes = []
        functions = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                parsed_class = self._parse_python_class(node)
                classes.append(parsed_class)
            elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                parsed_func = self._parse_python_function(node)
                functions.append(parsed_func)
        
        return {
            'classes': classes,
            'functions': functions,
            'language': 'python'
        }
    
    def _parse_python_class(self, node: ast.ClassDef) -> ParsedClass:
        """解析Python类"""
        methods = []
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                methods.append(self._parse_python_function(item))
        
        bases = [ast.unparse(base) if hasattr(ast, 'unparse') else str(getattr(base, 'id', '')) 
                 for base in node.bases]
        
        docstring = ast.get_docstring(node) or ""
        
        return ParsedClass(
            name=node.name,
            docstring=docstring,
            methods=methods,
            bases=bases
        )
    
    def _parse_python_function(self, node: ast.FunctionDef) -> ParsedFunction:
        """解析Python函数"""
        params = []
        
        # 解析参数
        for arg in node.args.args:
            param = {
                'name': arg.arg,
                'type': self._get_type_hint(arg.annotation),
                'default': None,
                'required': True
            }
            params.append(param)
        
        # 处理默认参数
        defaults = node.args.defaults
        args_without_default = len(node.args.args) - len(defaults)
        for i, default in enumerate(defaults):
            param_idx = args_without_default + i
            if param_idx < len(params):
                params[param_idx]['default'] = ast.unparse(default) if hasattr(ast, 'unparse') else str(default)
                params[param_idx]['required'] = False
        
        # 处理装饰器
        decorators = [ast.unparse(d) if hasattr(ast, 'unparse') else str(d) for d in node.decorator_list]
        
        # 返回类型
        return_type = self._get_type_hint(node.returns)
        
        docstring = ast.get_docstring(node) or ""
        
        # 解析参数描述
        param_descs = self._parse_param_docstring(docstring)
        
        return ParsedFunction(
            name=node.name,
            signature=self._build_python_signature(node),
            docstring=docstring,
            params=params,
            return_type=return_type,
            decorators=decorators,
            is_async=isinstance(node, ast.AsyncFunctionDef)
        )
    
    def _get_type_hint(self, annotation) -> str:
        """获取类型提示"""
        if annotation is None:
            return ""
        try:
            if hasattr(ast, 'unparse'):
                return ast.unparse(annotation)
            else:
                return str(annotation)
        except:
            return str(annotation)
    
    def _build_python_signature(self, node: ast.FunctionDef) -> str:
        """构建Python函数签名"""
        args = []
        for arg in node.args.args:
            arg_str = arg.arg
            if arg.annotation:
                arg_str += f": {self._get_type_hint(arg.annotation)}"
            args.append(arg_str)
        
        defaults = node.args.defaults
        args_without_default = len(node.args.args) - len(defaults)
        
        for i, default in enumerate(defaults):
            idx = args_without_default + i
            if idx < len(args):
                args[idx] += f" = {ast.unparse(default)}"
        
        async_prefix = "async " if isinstance(node, ast.AsyncFunctionDef) else ""
        return f"{async_prefix}def {node.name}({', '.join(args)})"
    
    def _parse_param_docstring(self, docstring: str) -> Dict[str, str]:
        """从文档字符串解析参数描述"""
        param_descs = {}
        
        # 简单的正则匹配
        patterns = [
            r':param\s+(\w+):\s*(.+)',
            r':param\s+(\w+)\s+(.+):',
            r'Args:\s*\n\s*(\w+):\s*(.+)',
            r'Parameters:\s*\n\s*(\w+):\s*(.+)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, docstring, re.MULTILINE)
            for match in matches:
                param_descs[match[0]] = match[1].strip()
        
        return param_descs
    
    def _parse_typescript(self, code: str) -> Dict[str, Any]:
        """解析TypeScript/JavaScript代码（简化实现）"""
        classes = []
        functions = []
        
        # 使用正则表达式提取函数
        func_pattern = r'(?:async\s+)?function\s+(\w+)\s*\(([^)]*)\)\s*(?::\s*([^=\n]+))?\s*\{'
        matches = re.findall(func_pattern, code)
        
        for match in matches:
            name, params_str, return_type = match
            params = self._parse_ts_params(params_str)
            
            functions.append({
                'name': name,
                'signature': f"function {name}({params_str})",
                'params': params,
                'return_type': return_type.strip() if return_type else "",
                'docstring': "",
                'is_async': 'async' in code[:code.find(name)]
            })
        
        # 提取类
        class_pattern = r'class\s+(\w+)\s*(?:extends\s+(\w+))?\s*\{'
        class_matches = re.findall(class_pattern, code)
        
        for match in class_matches:
            classes.append({
                'name': match[0],
                'docstring': "",
                'methods': [],
                'bases': [match[1]] if match[1] else []
            })
        
        return {
            'classes': classes,
            'functions': functions,
            'language': 'typescript'
        }
    
    def _parse_ts_params(self, params_str: str) -> List[Dict[str, Any]]:
        """解析TypeScript参数"""
        params = []
        if not params_str.strip():
            return params
        
        for param in params_str.split(','):
            param = param.strip()
            parts = param.split(':')
            if len(parts) >= 2:
                params.append({
                    'name': parts[0].strip(),
                    'type': parts[1].strip(),
                    'required': '?' not in parts[0]
                })
            elif param:
                params.append({'name': param, 'type': '', 'required': True})
        
        return params
    
    def _parse_go(self, code: str) -> Dict[str, Any]:
        """解析Go代码（简化实现）"""
        classes = []
        functions = []
        
        # 提取函数
        func_pattern = r'func\s*(?:\([^)]*\))?\s*(\w+)\s*\(([^)]*)\)\s*(?:\(([^)]*)\))?\s*\{'
        matches = re.findall(func_pattern, code)
        
        for match in matches:
            name, params_str, return_str = match
            params = self._parse_go_params(params_str)
            
            functions.append({
                'name': name,
                'signature': f"func {name}({params_str})",
                'params': params,
                'return_type': return_str.strip() if return_str else "",
                'docstring': ""
            })
        
        # 提取结构体
        struct_pattern = r'type\s+(\w+)\s*struct\s*\{([^}]+)\}'
        struct_matches = re.findall(struct_pattern, code, re.DOTALL)
        
        for match in struct_matches:
            classes.append({
                'name': match[0],
                'docstring': "",
                'methods': [],
                'bases': []
            })
        
        return {
            'classes': classes,
            'functions': functions,
            'language': 'go'
        }
    
    def _parse_go_params(self, params_str: str) -> List[Dict[str, Any]]:
        """解析Go参数"""
        params = []
        if not params_str.strip():
            return params
        
        for param in params_str.split(','):
            param = param.strip()
            parts = param.split()
            if len(parts) >= 2:
                params.append({
                    'name': parts[0],
                    'type': parts[-1],
                    'required': True
                })
        
        return params
