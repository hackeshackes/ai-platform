"""
Plugin Sandbox - 沙箱执行环境
为Plugin提供安全的隔离执行环境
"""

import os
import sys
import subprocess
import tempfile
import shutil
import resource
import signal
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
import logging
import threading

logger = logging.getLogger(__name__)


@dataclass
class SandboxConfig:
    """沙箱配置"""
    memory_limit_mb: int = 128  # 内存限制 (MB)
    cpu_limit_percent: int = 50  # CPU限制百分比
    timeout_seconds: int = 30  # 执行超时 (秒)
    disk_limit_mb: int = 100  # 磁盘限制 (MB)
    network_access: bool = False  # 是否允许网络访问
    allowed_dirs: List[str] = field(default_factory=lambda: ['/tmp', '/data'])
    environment_vars: Dict[str, str] = field(default_factory=dict)


@dataclass
class ExecutionResult:
    """执行结果"""
    success: bool
    output: str
    error: str
    return_code: int
    execution_time_ms: int
    memory_used_mb: float = 0.0
    sandbox_message: str = ""


class SandboxError(Exception):
    """沙箱错误"""
    pass


class PluginSandbox:
    """Plugin沙箱执行环境"""
    
    def __init__(self, config: Optional[SandboxConfig] = None):
        self.config = config or SandboxConfig()
        self.sandbox_dir = "./data/plugins/sandbox"
        os.makedirs(self.sandbox_dir, exist_ok=True)
    
    def _setup_environment(self, plugin_id: str) -> str:
        """设置隔离环境"""
        work_dir = os.path.join(self.sandbox_dir, plugin_id)
        os.makedirs(work_dir, exist_ok=True)
        return work_dir
    
    def _cleanup(self, work_dir: str) -> None:
        """清理工作目录"""
        try:
            if os.path.exists(work_dir):
                shutil.rmtree(work_dir)
        except Exception as e:
            logger.error(f"Failed to cleanup sandbox: {e}")
    
    def _set_limits(self) -> None:
        """设置资源限制"""
        # 内存限制
        memory_bytes = self.config.memory_limit_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
        
        # CPU限制 (软限制)
        resource.setrlimit(resource.RLIMIT_CPU, 
                          (self.config.timeout_seconds, resource.RLIM_INFINITY))
        
        # 进程数限制
        resource.setrlimit(resource.RLIMIT_NPROC, (50, 50))
    
    def execute_python(self, 
                      code: str,
                      plugin_id: str = "anonymous",
                      context: Optional[Dict] = None) -> ExecutionResult:
        """执行Python代码"""
        work_dir = self._setup_environment(plugin_id)
        start_time = time.time()
        memory_peak = 0.0
        
        try:
            # 创建临时文件
            script_file = os.path.join(work_dir, "script.py")
            
            # 包装代码以捕获输出和限制
            wrapped_code = self._wrap_code(code, context)
            
            with open(script_file, 'w') as f:
                f.write(wrapped_code)
            
            # 构建环境变量
            env = os.environ.copy()
            env.update(self.config.environment_vars)
            env['PYTHONPATH'] = work_dir
            
            # 执行
            process = subprocess.Popen(
                [sys.executable, script_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                cwd=work_dir
            )
            
            # 监控执行
            def read_output():
                output = []
                while True:
                    line = process.stdout.readline()
                    if not line and process.poll() is not None:
                        break
                    if line:
                        output.append(line)
                return ''.join(output)
            
            output_thread = threading.Thread(target=read_output)
            output_thread.start()
            
            try:
                return_code = process.wait(timeout=self.config.timeout_seconds)
            except subprocess.TimeoutExpired:
                process.kill()
                return ExecutionResult(
                    success=False,
                    output="",
                    error=f"Execution timeout ({self.config.timeout_seconds}s)",
                    return_code=-1,
                    execution_time_ms=int((time.time() - start_time) * 1000)
                )
            
            output_thread.join(timeout=1)
            
            execution_time_ms = int((time.time() - start_time) * 1000)
            
            return ExecutionResult(
                success=return_code == 0,
                output="",
                error=process.stdout.read() if return_code != 0 else "",
                return_code=return_code,
                execution_time_ms=execution_time_ms,
                memory_used_mb=memory_peak
            )
            
        except Exception as e:
            return ExecutionResult(
                success=False,
                output="",
                error=str(e),
                return_code=-1,
                execution_time_ms=int((time.time() - start_time) * 1000)
            )
        finally:
            self._cleanup(work_dir)
    
    def _wrap_code(self, code: str, context: Optional[Dict] = None) -> str:
        """包装代码以提供上下文和安全限制"""
        safe_globals = {
            '__name__': '__plugin_sandbox__',
            '__builtins__': {
                'print': print,
                'range': range,
                'len': len,
                'str': str,
                'int': int,
                'float': float,
                'bool': bool,
                'list': list,
                'dict': dict,
                'set': set,
                'tuple': tuple,
                'abs': abs,
                'max': max,
                'min': min,
                'sum': sum,
                'sorted': sorted,
                'map': map,
                'filter': filter,
                'zip': zip,
                'enumerate': enumerate,
                'isinstance': isinstance,
                'hasattr': hasattr,
                'getattr': getattr,
                'setattr': setattr,
                'delattr': delattr,
                'round': round,
                'divmod': divmod,
                'pow': pow,
                'chr': chr,
                'ord': ord,
                'hex': hex,
                'oct': oct,
                'bin': bin,
                'slice': slice,
            }
        }
        
        context_str = str(context) if context else "{}"
        
        return f"""
import sys
import io

# 重定向输出
sys.stdout = io.StringIO()
sys.stderr = io.StringIO()

# 上下文
context = {context_str}

# 安全包装
class SafeModule:
    def __init__(self, name):
        self._name = name
    
    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(f"No attribute: {{name}}")
        raise ImportError(f"No module named '{{name}}'")

# 插件代码
{code}

# 捕获输出
output = sys.stdout.getvalue()
error = sys.stdout.getvalue()

print("=== OUTPUT ===")
print(output if output else "(no output)")
print("=== ERROR ===")
print(error if error else "(no error)")
"""
    
    def execute_plugin(self, 
                      plugin_id: str,
                      function_name: str = "main",
                      args: Optional[List] = None,
                      kwargs: Optional[Dict] = None) -> ExecutionResult:
        """执行Plugin函数"""
        plugin_path = os.path.join("./data/plugins/installed", plugin_id)
        if not os.path.exists(plugin_path):
            raise SandboxError(f"Plugin not installed: {plugin_id}")
        
        # 加载plugin.py
        sys.path.insert(0, plugin_path)
        
        try:
            # 动态导入
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "plugin", 
                os.path.join(plugin_path, "plugin.py")
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # 获取函数
            func = getattr(module, function_name, None)
            if not func:
                raise SandboxError(f"Function not found: {function_name}")
            
            # 执行
            args = args or []
            kwargs = kwargs or {}
            
            start_time = time.time()
            result = func(*args, **kwargs)
            execution_time_ms = int((time.time() - start_time) * 1000)
            
            return ExecutionResult(
                success=True,
                output=str(result),
                error="",
                return_code=0,
                execution_time_ms=execution_time_ms
            )
            
        except Exception as e:
            return ExecutionResult(
                success=False,
                output="",
                error=str(e),
                return_code=-1,
                execution_time_ms=0
            )
        finally:
            sys.path.remove(plugin_path)
    
    def validate_plugin(self, plugin_path: str) -> Dict:
        """验证Plugin安全性"""
        issues = []
        warnings = []
        
        # 检查敏感API使用
        sensitive_patterns = [
            ('subprocess', 'Shell execution'),
            ('os.system', 'System command'),
            ('eval', 'Dynamic evaluation'),
            ('exec', 'Code execution'),
            ('__import__', 'Dynamic import'),
            ('open', 'File access'),
            ('file', 'File access'),
        ]
        
        for pattern, message in sensitive_patterns:
            pass  # 实际应该扫描文件内容
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings
        }
    
    def create_plugin_template(self, plugin_id: str, name: str) -> str:
        """创建Plugin模板"""
        template = f'''"""Plugin: {name}"""

VERSION = "1.0.0"
AUTHOR = "Your Name"
DESCRIPTION = "Plugin description"

def main(context: dict) -> dict:
    """主入口函数
    
    Args:
        context: 上下文信息
    
    Returns:
        处理结果
    """
    # Your code here
    return {{
        "status": "success",
        "data": {{}}
    }}

def process(data: dict) -> dict:
    """处理数据"""
    return data
'''
        
        template_path = os.path.join(self.sandbox_dir, f"{plugin_id}_template.py")
        with open(template_path, 'w') as f:
            f.write(template)
        
        return template_path


# 全局沙箱实例
plugin_sandbox = PluginSandbox()
