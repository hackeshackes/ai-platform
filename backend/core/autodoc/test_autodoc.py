#!/usr/bin/env python3
"""
测试用例
验证autodoc的功能
"""

import unittest
import tempfile
import os
from pathlib import Path

from .code_parser import CodeParser, ParsedFunction
from .api_extractor import APIExtractor, APIInfo
from .example_generator import ExampleGenerator
from .doc_generator import DocGenerator
from .markdown_writer import MarkdownWriter


class TestCodeParser(unittest.TestCase):
    """测试代码解析器"""
    
    def setUp(self):
        self.parser = CodeParser(language='python')
    
    def test_parse_simple_function(self):
        """测试解析简单函数"""
        code = '''
def hello(name: str) -> str:
    """Say hello to someone."""
    return f"Hello, {name}!"
'''
        result = self.parser._parse_python(code)
        
        self.assertEqual(len(result['functions']), 1)
        func = result['functions'][0]
        self.assertEqual(func.name, 'hello')
        self.assertEqual(func.return_type, 'str')
        self.assertTrue(func.docstring)
    
    def test_parse_async_function(self):
        """测试解析异步函数"""
        code = '''
async def fetch_data(url: str) -> dict:
    """Fetch data from URL."""
    pass
'''
        result = self.parser._parse_python(code)
        
        self.assertEqual(len(result['functions']), 1)
        func = result['functions'][0]
        self.assertTrue(func.is_async)
        self.assertIn('async', func.signature)
    
    def test_parse_class(self):
        """测试解析类"""
        code = '''
class Agent:
    """An AI agent."""
    
    def __init__(self, name: str):
        self.name = name
    
    def think(self, message: str) -> str:
        """Process a message."""
        return message
'''
        result = self.parser._parse_python(code)
        
        self.assertEqual(len(result['classes']), 1)
        cls = result['classes'][0]
        self.assertEqual(cls.name, 'Agent')
        self.assertTrue(len(cls.methods) >= 1)
    
    def test_parse_with_defaults(self):
        """测试解析带默认参数的函数"""
        code = '''
def create(name: str, age: int = 18, city: str = None):
    pass
'''
        result = self.parser._parse_python(code)
        
        self.assertEqual(len(result['functions']), 1)
        func = result['functions'][0]
        
        # 检查参数
        self.assertEqual(len(func.params), 3)
        self.assertTrue(func.params[0]['required'])
        self.assertFalse(func.params[1]['required'])
        self.assertFalse(func.params[2]['required'])
    
    def test_parse_decorators(self):
        """测试解析装饰器"""
        code = '''
@staticmethod
@decorator(arg="value")
def method():
    pass
'''
        result = self.parser._parse_python(code)
        
        self.assertEqual(len(result['functions']), 1)
        func = result['functions'][0]
        self.assertEqual(len(func.decorators), 2)
    
    def test_parse_file(self):
        """测试解析文件"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write('def test(): pass')
            f.flush()
            
            result = self.parser.parse_file(f.name)
            
            self.assertEqual(len(result['functions']), 1)
            self.assertEqual(result['functions'][0].name, 'test')
            
            os.unlink(f.name)
    
    def test_language_support(self):
        """测试语言支持"""
        for lang in ['python', 'typescript', 'javascript', 'go']:
            parser = CodeParser(language=lang)
            self.assertIn(lang, parser.SUPPORTED_LANGUAGES)
    
    def test_unsupported_language(self):
        """测试不支持的语言"""
        with self.assertRaises(ValueError):
            CodeParser(language='unsupported')


class TestAPIExtractor(unittest.TestCase):
    """测试API提取器"""
    
    def setUp(self):
        self.extractor = APIExtractor(language='python')
    
    def test_extract_from_code(self):
        """测试从代码提取API"""
        code = '''
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b
'''
        module = self.extractor.extract_from_code(code, 'test_module')
        
        self.assertEqual(module.name, 'test_module')
        self.assertEqual(len(module.apis), 1)
        
        api = module.apis[0]
        self.assertEqual(api.name, 'add')
        self.assertEqual(api.type, 'function')
        self.assertEqual(len(api.parameters), 2)
    
    def test_extract_method(self):
        """测试提取方法"""
        code = '''
class Calculator:
    def multiply(self, a: float, b: float) -> float:
        """Multiply two numbers."""
        return a * b
'''
        module = self.extractor.extract_from_code(code, 'test')
        
        # 应该提取一个类和一个方法
        self.assertEqual(len(module.classes), 1)
        self.assertTrue(len(module.apis) >= 1)
    
    def test_extract_complex_params(self):
        """测试提取复杂参数"""
        code = '''
def complex_func(
    name: str,
    items: List[str] = None,
    config: Dict[str, Any] = None
) -> Optional[Result]:
    pass
'''
        module = self.extractor.extract_from_code(code)
        
        api = module.apis[0]
        self.assertEqual(len(api.parameters), 3)
        
        # 检查返回类型
        self.assertIn('Optional', api.returns.get('type', ''))


class TestExampleGenerator(unittest.TestCase):
    """测试示例生成器"""
    
    def setUp(self):
        self.generator = ExampleGenerator(language='python')
    
    def test_generate_examples(self):
        """测试生成示例"""
        api = APIInfo(
            name='test_func',
            type='function',
            signature='def test_func(a: int, b: str) -> bool',
            description='Test function',
            parameters=[
                {'name': 'a', 'type': 'int', 'required': True, 'description': ''},
                {'name': 'b', 'type': 'str', 'required': False, 'description': ''}
            ],
            returns={'type': 'bool', 'description': ''}
        )
        
        examples = self.generator.generate_examples(api)
        
        self.assertTrue(len(examples) >= 1)
        # 至少应该有基本示例
        self.assertTrue(any(e.title == '基础使用' for e in examples))
    
    def test_get_sample_value(self):
        """测试获取示例值"""
        self.assertEqual(self.generator._get_sample_value({'type': 'str'}), '"example"')
        self.assertEqual(self.generator._get_sample_value({'type': 'int'}), '1')
        self.assertEqual(self.generator._get_sample_value({'type': 'bool'}), 'True')
        self.assertEqual(self.generator._get_sample_value({'type': 'list'}), '[]')
        self.assertEqual(self.generator._get_sample_value({'type': 'list[str]'}), '[]')
        self.assertEqual(self.generator._get_sample_value({'type': 'Dict[str, str]'}), '{}')
    
    def test_generate_quick_start(self):
        """测试生成快速开始示例"""
        module_name = 'test_module'
        apis = [
            APIInfo(name='func1', type='function', signature='', description='', parameters=[], returns={}),
            APIInfo(name='func2', type='function', signature='', description='', parameters=[], returns={})
        ]
        
        examples = self.generator.generate_module_examples(module_name, apis)
        
        self.assertTrue(len(examples) >= 1)
        self.assertTrue(any(e.title == '快速开始' for e in examples))


class TestDocGenerator(unittest.TestCase):
    """测试文档生成器"""
    
    def setUp(self):
        self.generator = DocGenerator(language='python')
    
    def test_generate_function_doc(self):
        """测试生成函数文档"""
        api = APIInfo(
            name='my_function',
            type='function',
            signature='def my_function(name: str, age: int = 0) -> str',
            description='A test function',
            parameters=[
                {'name': 'name', 'type': 'str', 'required': True, 'description': 'Person name'},
                {'name': 'age', 'type': 'int', 'required': False, 'description': 'Person age'}
            ],
            returns={'type': 'str', 'description': 'Greeting message'}
        )
        
        lines = self.generator._generate_function_doc(api)
        
        # 检查基本内容
        doc = '\n'.join(lines)
        self.assertIn('### my_function', doc)
        self.assertIn('**签名:**', doc)
        self.assertIn('```python', doc)
        self.assertIn('**参数:**', doc)
        self.assertIn('| name |', doc)
        self.assertIn('**返回值:**', doc)
    
    def test_generate_module_doc(self):
        """测试生成模块文档"""
        from .api_extractor import ModuleInfo
        
        module = ModuleInfo(
            name='test_module',
            description='A test module',
            apis=[],
            classes=[{'name': 'TestClass', 'description': 'A test class', 'bases': [], 'methods': []}],
            functions=[]
        )
        
        doc = self.generator.generate_module_doc(module)
        
        self.assertIn('# test_module', doc)
        self.assertIn('## 类', doc)


class TestMarkdownWriter(unittest.TestCase):
    """测试Markdown写入器"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.writer = MarkdownWriter(output_dir=self.temp_dir, language='python')
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_write_module_doc(self):
        """测试写入模块文档"""
        from .api_extractor import ModuleInfo
        
        module = ModuleInfo(
            name='test_module',
            description='Test module',
            apis=[],
            classes=[],
            functions=[]
        )
        
        filepath = self.writer.write_module_doc(module)
        
        self.assertTrue(os.path.exists(filepath))
        self.assertTrue(filepath.endswith('.md'))
        
        content = Path(filepath).read_text()
        self.assertIn('test_module', content)
    
    def test_write_index(self):
        """测试写入索引"""
        modules = []
        
        filepath = self.writer.write_index(modules)
        
        self.assertTrue(os.path.exists(filepath))
        self.assertTrue(filepath.endswith('SUMMARY.md'))
    
    def test_generate_tree_structure(self):
        """测试生成目录结构"""
        # 创建一些文件
        (Path(self.temp_dir) / 'README.md').write_text('# README')
        (Path(self.temp_dir) / 'api.md').write_text('# API')
        
        tree = self.writer.generate_tree_structure()
        
        self.assertIn('README', tree)
        self.assertIn('api', tree)


class TestIntegration(unittest.TestCase):
    """集成测试"""
    
    def test_full_pipeline(self):
        """测试完整流程"""
        code = '''
class Calculator:
    """A simple calculator."""
    
    def add(self, a: int, b: int) -> int:
        """Add two numbers."""
        return a + b
    
    def multiply(self, a: int, b: int) -> int:
        """Multiply two numbers."""
        return a * b

def create_calculator() -> Calculator:
    """Create a new calculator."""
    return Calculator()
'''
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # 解析
            parser = CodeParser(language='python')
            parsed = parser.parse_code(code)
            
            # 提取
            extractor = APIExtractor(language='python')
            module = extractor.extract_from_code(code, 'calculator')
            
            # 生成
            generator = DocGenerator(language='python')
            doc = generator.generate_module_doc(module)
            
            # 写入
            writer = MarkdownWriter(output_dir=temp_dir)
            writer.write_module_doc(module)
            
            # 验证
            self.assertTrue(len(module.apis) >= 2)
            self.assertTrue('add' in [api.name for api in module.apis])
            self.assertTrue('multiply' in [api.name for api in module.apis])


if __name__ == '__main__':
    unittest.main()
