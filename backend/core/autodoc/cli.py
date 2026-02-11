#!/usr/bin/env python3
"""
自动文档生成器CLI工具
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

from .code_parser import CodeParser
from .api_extractor import APIExtractor
from .doc_generator import DocGenerator
from .markdown_writer import MarkdownWriter


def create_parser():
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description="自动文档生成器 - 从代码生成API文档",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s ./src --output ./docs --language python
  %(prog)s ./src --recursive --format markdown
  %(prog)s single_file.py --api-only
        """
    )
    
    parser.add_argument(
        'source',
        nargs='?',
        default='.',
        help='源代码文件或目录 (默认: 当前目录)'
    )
    
    parser.add_argument(
        '--output', '-o',
        default='./docs',
        help='输出目录 (默认: ./docs)'
    )
    
    parser.add_argument(
        '--language', '-l',
        default='python',
        choices=['python', 'typescript', 'javascript', 'go'],
        help='源代码语言 (默认: python)'
    )
    
    parser.add_argument(
        '--recursive', '-r',
        action='store_true',
        help='递归处理目录'
    )
    
    parser.add_argument(
        '--format', '-f',
        default='markdown',
        choices=['markdown'],
        help='输出格式 (默认: markdown)'
    )
    
    parser.add_argument(
        '--api-only',
        action='store_true',
        help='只生成API参考'
    )
    
    parser.add_argument(
        '--examples',
        action='store_true',
        default=True,
        help='包含代码示例 (默认: True)'
    )
    
    parser.add_argument(
        '--best-practices',
        action='store_true',
        help='生成最佳实践文档'
    )
    
    parser.add_argument(
        '--project-name',
        default='API Documentation',
        help='项目名称 (默认: API Documentation)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='详细输出'
    )
    
    return parser


def process_file(file_path: str, language: str, output_dir: str) -> List[str]:
    """处理单个文件"""
    output_files = []
    
    try:
        extractor = APIExtractor(language)
        module = extractor.extract_from_file(file_path)
        
        writer = MarkdownWriter(output_dir, language)
        
        if module.apis:
            output_file = writer.write_module_doc(module)
            output_files.append(output_file)
            if verbose:
                print(f"Generated: {output_file}")
        else:
            if verbose:
                print(f"No APIs found in: {file_path}")
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}", file=sys.stderr)
    
    return output_files


def process_directory(dir_path: str, language: str, output_dir: str, recursive: bool) -> List[str]:
    """处理目录"""
    output_files = []
    path = Path(dir_path)
    
    # 根据语言选择文件扩展名
    ext_map = {
        'python': ['.py'],
        'typescript': ['.ts', '.tsx'],
        'javascript': ['.js', '.jsx'],
        'go': ['.go']
    }
    
    extensions = ext_map.get(language, ['.py'])
    
    # 收集所有文件
    if recursive:
        file_paths = path.rglob("*")
    else:
        file_paths = path.glob("*")
    
    source_files = [
        f for f in file_paths
        if f.is_file() and f.suffix.lower() in extensions
    ]
    
    if verbose:
        print(f"Found {len(source_files)} source files")
    
    # 处理每个文件
    for file_path in sorted(source_files):
        files = process_file(str(file_path), language, output_dir)
        output_files.extend(files)
    
    # 生成索引文件
    if output_files:
        writer = MarkdownWriter(output_dir, language)
        writer.write_index([])
        writer.write_sidebar([])
    
    return output_files


def main():
    """主函数"""
    global verbose
    verbose = False
    
    parser = create_parser()
    args = parser.parse_args()
    
    source = args.source
    output_dir = args.output
    language = args.language
    recursive = args.recursive
    project_name = args.project_name
    verbose = args.verbose
    
    # 检查源是否存在
    source_path = Path(source)
    if not source_path.exists():
        print(f"Error: Source not found: {source}", file=sys.stderr)
        sys.exit(1)
    
    if verbose:
        print(f"Source: {source}")
        print(f"Output: {output_dir}")
        print(f"Language: {language}")
        print(f"Recursive: {recursive}")
    
    # 处理
    output_files = []
    
    if source_path.is_file():
        output_files = process_file(source, language, output_dir)
    elif source_path.is_dir():
        output_files = process_directory(source, language, output_dir, recursive)
    
    # 生成额外文档
    if not args.api_only and output_files:
        writer = MarkdownWriter(output_dir, language)
        
        if args.best_practices:
            writer.write_best_practices([])
            if verbose:
                print(f"Generated: {output_dir}/best_practices.md")
    
    if verbose:
        print(f"\nTotal files generated: {len(output_files)}")
    
    print("Done!")
    return 0


def run():
    """运行CLI"""
    sys.exit(main())


if __name__ == '__main__':
    run()
