"""
示例生成器
生成代码示例、使用示例和场景示例
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from .api_extractor import APIInfo


@dataclass
class Example:
    """示例"""
    title: str
    description: str
    code: str
    language: str = "python"
    tags: List[str] = None


class ExampleGenerator:
    """示例生成器"""
    
    def __init__(self, language: str = 'python'):
        self.language = language
        self.example_templates = self._load_templates()
    
    def _load_templates(self) -> Dict[str, Dict[str, str]]:
        """加载示例模板"""
        return {
            'basic_usage': {
                'python': '''
```python
# 基础使用示例
{code}
```
''',
                'typescript': '''
```typescript
// 基础使用示例
{code}
```
'''
            },
            'async_usage': {
                'python': '''
```python
# 异步使用示例
import asyncio

async def main():
    {code}

asyncio.run(main())
```
'''
            }
        }
    
    def generate_examples(self, api: APIInfo) -> List[Example]:
        """为API生成示例"""
        examples = []
        
        # 基本示例
        basic_example = self._generate_basic_example(api)
        if basic_example:
            examples.append(basic_example)
        
        # 高级示例
        advanced_example = self._generate_advanced_example(api)
        if advanced_example:
            examples.append(advanced_example)
        
        # 场景示例
        scenario_examples = self._generate_scenario_examples(api)
        examples.extend(scenario_examples)
        
        return examples
    
    def _generate_basic_example(self, api: APIInfo) -> Optional[Example]:
        """生成基本示例"""
        code = self._build_basic_code(api)
        if not code:
            return None
        
        return Example(
            title="基础使用",
            description=f"展示{api.name}的基本用法",
            code=code,
            language=self.language
        )
    
    def _generate_advanced_example(self, api: APIInfo) -> Optional[Example]:
        """生成高级示例"""
        if not api.parameters:
            return None
        
        code = self._build_advanced_code(api)
        if not code:
            return None
        
        return Example(
            title="高级用法",
            description=f"展示{api.name}的高级用法和配置",
            code=code,
            language=self.language
        )
    
    def _generate_scenario_examples(self, api: APIInfo) -> List[Example]:
        """生成场景示例"""
        examples = []
        
        # 错误处理示例
        error_example = self._generate_error_handling_example(api)
        if error_example:
            examples.append(error_example)
        
        # 最佳实践示例
        best_practice = self._generate_best_practice_example(api)
        if best_practice:
            examples.append(best_practice)
        
        return examples
    
    def _generate_error_handling_example(self, api: APIInfo) -> Optional[Example]:
        """生成错误处理示例"""
        code = f'''try:
    result = {api.name}({self._format_params(api.parameters, with_values=True)})
except Exception as e:
    print(f"Error: {{e}}")
    # 错误处理逻辑
'''
        
        return Example(
            title="错误处理",
            description="展示如何正确处理异常",
            code=code.strip(),
            language=self.language
        )
    
    def _generate_best_practice_example(self, api: APIInfo) -> Optional[Example]:
        """生成最佳实践示例"""
        code = f'''# 最佳实践示例
# 1. 参数验证
# 2. 错误处理
# 3. 资源清理

result = {api.name}({self._format_params(api.parameters, with_values=True)})

# 使用结果
if result:
    print("Success!")
'''
        
        return Example(
            title="最佳实践",
            description="展示推荐的使用模式",
            code=code.strip(),
            language=self.language
        )
    
    def _build_basic_code(self, api: APIInfo) -> str:
        """构建基本代码"""
        if not api.parameters:
            return f"# 调用{api.name}\nresult = {api.name}()"
        
        # 只使用必需参数
        required_params = [p for p in api.parameters if p.get('required', True)]
        if not required_params:
            required_params = api.parameters[:1]
        
        params_str = self._format_params(required_params, with_values=True)
        
        return f"# 调用{api.name}\nresult = {api.name}({params_str})"
    
    def _build_advanced_code(self, api: APIInfo) -> str:
        """构建高级代码"""
        if not api.parameters:
            return f"# 高级调用{api.name}\nresult = {api.name}()"
        
        # 使用所有参数
        params_str = self._format_params(api.parameters, with_values=True)
        
        return f"# 高级调用{api.name}\nresult = {api.name}({params_str})"
    
    def _format_params(self, params: List[Dict[str, Any]], with_values: bool = False) -> str:
        """格式化参数列表"""
        parts = []
        for param in params:
            name = param.get('name', '')
            if with_values:
                value = self._get_sample_value(param)
                parts.append(f"{name}={value}")
            else:
                parts.append(name)
        return ', '.join(parts)
    
    def _get_sample_value(self, param: Dict[str, Any]) -> str:
        """获取参数的示例值"""
        param_type = param.get('type', '').lower()
        required = param.get('required', True)
        
        # 常见类型映射 (按优先级排序，复杂的在前)
        type_mappings = [
            ('dict[str, str]', '{}'),
            ('list[str]', '[]'),
            ('dict[str, any]', '{}'),
            ('optional[str]', 'None'),
            ('str', '"example"'),
            ('string', '"example"'),
            ('int', '1'),
            ('integer', '1'),
            ('float', '1.0'),
            ('bool', 'True'),
            ('boolean', 'True'),
            ('list', '[]'),
            ('dict', '{}'),
            ('none', 'None'),
            ('path', '"/path/to/file"'),
            ('uri', '"https://example.com"'),
            ('datetime', 'datetime.now()'),
            ('date', 'date.today()')
        ]
        
        for key, value in type_mappings:
            if key in param_type:
                return value
        
        if not required:
            return 'None'
        
        return 'value'
    
    def generate_interactive_example(self, api: APIInfo) -> str:
        """生成交互式示例（用于Jupyter等）"""
        code = f'''# 交互式示例
# 尝试修改参数值并运行

{api.name}(
{self._format_params_for_interactive(api.parameters)}
)
'''
        
        return code
    
    def _format_params_for_interactive(self, params: List[Dict[str, Any]]) -> str:
        """格式化交互式参数"""
        lines = []
        for param in params:
            name = param.get('name', '')
            param_type = param.get('type', '')
            default = param.get('default')
            
            if default is not None:
                display = f"# {name}={default}  # {param_type}"
            else:
                display = f"# {name}=<{param_type or 'value'}>"
            
            lines.append(f"    {display}")
        
        return '\n'.join(lines)
    
    def generate_module_examples(self, module_name: str, apis: List[APIInfo]) -> List[Example]:
        """生成模块级示例"""
        examples = []
        
        # 快速开始示例
        quick_start = self._generate_quick_start(module_name, apis)
        examples.append(quick_start)
        
        # 完整示例
        full_example = self._generate_full_example(module_name, apis)
        examples.append(full_example)
        
        return examples
    
    def _generate_quick_start(self, module_name: str, apis: List[APIInfo]) -> Example:
        """生成快速开始示例"""
        if not apis:
            code = f"# 导入模块\nimport {module_name}"
        else:
            main_api = apis[0]
            code = f'''# 导入模块
import {module_name}

# 快速开始
result = {main_api.name}({self._format_params(main_api.parameters, with_values=True)})
print(result)
'''
        
        return Example(
            title="快速开始",
            description="5分钟内上手",
            code=code.strip(),
            language=self.language
        )
    
    def _generate_full_example(self, module_name: str, apis: List[APIInfo]) -> Example:
        """生成完整示例"""
        if not apis:
            code = f"# 完整示例\nimport {module_name}"
        else:
            lines = [f"# 完整示例\nimport {module_name}\n", "# 使用多个API"]
            for api in apis[:3]:
                lines.append(f"# {api.name}")
                lines.append(f"result_{api.name} = {api.name}({self._format_params(api.parameters[:2], with_values=True)})")
            code = '\n'.join(lines)
        
        return Example(
            title="完整示例",
            description="展示模块的所有主要功能",
            code=code.strip(),
            language=self.language
        )
