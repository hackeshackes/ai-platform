"""
示例文件
展示autodoc的使用方法
"""

# 快速开始示例
QUICK_START = '''
## 快速开始

```python
from autodoc import CodeParser, APIExtractor, DocGenerator

# 1. 解析代码
parser = CodeParser(language='python')
parsed = parser.parse_file('your_module.py')

# 2. 提取API
extractor = APIExtractor(language='python')
module = extractor.extract_from_file('your_module.py')

# 3. 生成文档
generator = DocGenerator(language='python')
doc = generator.generate_module_doc(module)
print(doc)
'''

# 完整使用示例
FULL_EXAMPLE = '''
## 完整使用示例

```python
from pathlib import Path
from autodoc import CodeParser, APIExtractor, DocGenerator, MarkdownWriter

# 初始化组件
parser = CodeParser(language='python')
extractor = APIExtractor(language='python')
writer = MarkdownWriter(output_dir='./docs', language='python')

# 处理单个文件
module = extractor.extract_from_file('src/agent.py')
writer.write_module_doc(module)

# 处理整个目录
modules = extractor.extract_directory('src/', recursive=True)
writer.write_full_doc(modules)

# 生成最佳实践
writer.write_best_practices([])
```

## 命令行使用

```bash
# 处理当前目录
autodoc .

# 处理指定目录
autodoc ./src --output ./docs --language python

# 递归处理
autodoc ./src --recursive --output ./docs

# 只生成API参考
autodoc ./src --api-only
```

## 支持的语言

- Python (`.py`)
- TypeScript (`.ts`, `.tsx`)
- JavaScript (`.js`, `.jsx`)
- Go (`.go`)

## 输出示例

生成的Markdown文档包含：

- 模块概述
- 类文档
- 函数签名
- 参数表格
- 返回值说明
- 代码示例
- 最佳实践
'''

# 配置文件示例
CONFIG_EXAMPLE = '''
# autodoc.config.py

AUTODOC_CONFIG = {
    'source': './src',
    'output': './docs',
    'language': 'python',
    'recursive': True,
    'project_name': 'My Project API',
    'include_examples': True,
    'best_practices': True,
    'exclude_patterns': [
        '**/test_*.py',
        '**/__pycache__/**',
        '**/migrations/**'
    ]
}
'''
