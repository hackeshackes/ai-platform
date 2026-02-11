"""
文档生成器
生成API文档、使用示例和最佳实践
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
from .api_extractor import APIInfo, ModuleInfo
from .example_generator import Example, ExampleGenerator


class DocGenerator:
    """文档生成器"""
    
    def __init__(self, language: str = 'python', project_name: str = "API Documentation"):
        self.language = language
        self.project_name = project_name
        self.example_generator = ExampleGenerator(language)
    
    def generate_module_doc(self, module: ModuleInfo) -> str:
        """生成模块文档"""
        lines = []
        
        # 模块标题
        lines.append(f"# {module.name}\n")
        
        # 模块描述
        if module.description:
            lines.append(f"{module.description}\n")
        
        # 目录
        if module.classes:
            lines.append("## 类\n")
            for cls in module.classes:
                lines.append(f"- [{cls['name']}](#{cls['name'].lower()})")
            lines.append("")
        
        if module.functions:
            lines.append("## 函数\n")
            for api in module.apis:
                if api.type == "function":
                    lines.append(f"- [{api.name}](#{api.name})")
            lines.append("")
        
        lines.append("---\n")
        
        # 类详细文档
        for cls in module.classes:
            lines.extend(self._generate_class_doc(cls))
        
        # 函数详细文档
        for api in module.apis:
            if api.type == "function":
                lines.extend(self._generate_function_doc(api))
        
        return '\n'.join(lines)
    
    def _generate_class_doc(self, class_info: Dict[str, Any]) -> List[str]:
        """生成类文档"""
        lines = []
        
        lines.append(f"## {class_info['name']}\n")
        
        if class_info.get('description'):
            lines.append(f"{class_info['description']}\n")
        
        # 基类
        if class_info.get('bases'):
            lines.append(f"**继承:** {', '.join(class_info['bases'])}\n")
        
        lines.append("### 方法\n")
        
        for method_name in class_info.get('methods', []):
            lines.append(f"- [{method_name}](#{class_info['name'].lower()}{method_name})")
        
        lines.append("")
        
        return lines
    
    def _generate_function_doc(self, api: APIInfo) -> List[str]:
        """生成函数文档"""
        lines = []
        
        # 标题
        lines.append(f"### {api.name}\n")
        
        # 描述
        if api.description:
            lines.append(f"{api.description}\n")
        
        # 签名
        lines.append("**签名:**")
        lines.append(f"```{self.language}")
        lines.append(api.signature)
        lines.append("```\n")
        
        # 参数表格
        if api.parameters:
            lines.append("**参数:**")
            lines.append("| 参数 | 类型 | 必填 | 说明 |")
            lines.append("|------|------|------|------|")
            
            for param in api.parameters:
                name = param.get('name', '')
                param_type = param.get('type', '')
                required = "是" if param.get('required', True) else "否"
                desc = param.get('description', '')
                
                lines.append(f"| {name} | {param_type} | {required} | {desc} |")
            
            lines.append("")
        
        # 返回值
        if api.returns:
            lines.append("**返回值:**")
            if api.returns.get('type'):
                lines.append(f"- **类型:** `{api.returns['type']}`")
            if api.returns.get('description'):
                lines.append(f"- **说明:** {api.returns['description']}")
            lines.append("")
        
        # 示例
        examples = self.example_generator.generate_examples(api)
        if examples:
            lines.append("**示例:**")
            for example in examples:
                lines.append(f"\n#### {example.title}")
                lines.append(f"{example.description}")
                lines.append(example.code)
            lines.append("")
        
        # 异常
        if api.raises:
            lines.append("**异常:**")
            for exc in api.raises:
                lines.append(f"- `{exc.get('type', 'Exception')}`: {exc.get('description', '')}")
            lines.append("")
        
        # 注意事项
        if api.notes:
            lines.append("**注意:**")
            for note in api.notes:
                lines.append(f"- {note}")
            lines.append("")
        
        lines.append("---\n")
        
        return lines
    
    def generate_full_doc(self, modules: List[ModuleInfo]) -> str:
        """生成完整文档"""
        lines = []
        
        # 标题
        lines.append(f"# {self.project_name}\n")
        
        lines.append(f"*自动生成于 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
        
        lines.append("---\n")
        
        # 快速开始
        lines.append("## 快速开始\n")
        lines.append("```python")
        lines.append(f"import {self.project_name.lower().replace(' ', '_')}")
        lines.append("```\n")
        
        lines.append("---\n")
        
        # 模块列表
        lines.append("## 模块\n")
        for module in modules:
            lines.append(f"- [{module.name}](#{module.name})")
        lines.append("")
        
        lines.append("---\n")
        
        # 详细文档
        for module in modules:
            lines.append(self.generate_module_doc(module))
        
        return '\n'.join(lines)
    
    def generate_api_reference(self, apis: List[APIInfo]) -> str:
        """生成API参考文档"""
        lines = []
        
        lines.append(f"# API 参考\n\n")
        
        for api in apis:
            lines.extend(self._generate_function_doc(api))
        
        return '\n'.join(lines)
    
    def generate_best_practices(self, apis: List[APIInfo]) -> str:
        """生成最佳实践文档"""
        lines = []
        
        lines.append("# 最佳实践\n\n")
        
        lines.append("## 通用指南\n")
        lines.append("1. 始终进行参数验证")
        lines.append("2. 使用适当的错误处理")
        lines.append("3. 注意资源清理\n")
        
        lines.append("## 性能优化\n")
        lines.append("- 使用批量操作代替循环调用")
        lines.append("- 合理设置超时")
        lines.append("- 使用缓存减少重复调用\n")
        
        lines.append("## 安全建议\n")
        lines.append("- 验证所有输入参数")
        lines.append("- 不要在日志中打印敏感信息")
        lines.append("- 使用环境变量管理配置\n")
        
        return '\n'.join(lines)
