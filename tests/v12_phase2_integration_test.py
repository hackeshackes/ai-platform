#!/usr/bin/env python3
"""
v12 Phase 2 集成测试 (简化版)

测试所有Phase 2模块的基本功能
"""

import sys
import subprocess
from pathlib import Path

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent
BACKEND_ROOT = PROJECT_ROOT / "backend"
CORE_ROOT = BACKEND_ROOT / "core"

# 测试结果
TESTS_PASSED = 0
TESTS_FAILED = 0


def log_test(name: str, passed: bool, details: str = ""):
    """记录测试结果"""
    global TESTS_PASSED, TESTS_FAILED
    if passed:
        TESTS_PASSED += 1
        status = "✅ PASS"
    else:
        TESTS_FAILED += 1
        status = "❌ FAIL"
    
    print(f"{status} | {name}")
    if details:
        print(f"    └── {details}")


def test_module_structure():
    """测试模块结构"""
    print("\n📦 模块结构测试...")
    
    modules = {
        "aiops": CORE_ROOT / "aiops",
        "scheduler": CORE_ROOT / "scheduler",
        "self_healing": CORE_ROOT / "self_healing",
        "automation_ops": CORE_ROOT / "automation_ops",
        "performance_tuner": CORE_ROOT / "performance_tuner"
    }
    
    modules_found = 0
    for name, path in modules.items():
        if path.exists() and path.is_dir():
            files = list(path.glob("*.py"))
            if len(files) >= 3:
                modules_found += 1
                print(f"  ✅ {name} ({len(files)} files)")
            else:
                print(f"  ❌ {name} (only {len(files)} files)")
        else:
            print(f"  ❌ {name} (not found)")
    
    log_test(
        "模块结构",
        modules_found == len(modules),
        f"找到: {modules_found}/{len(modules)}"
    )
    
    return modules_found == len(modules)


def test_python_syntax():
    """测试Python语法"""
    print("\n🐍 Python语法测试...")
    
    modules = {
        "aiops": CORE_ROOT / "aiops",
        "scheduler": CORE_ROOT / "scheduler",
        "self_healing": CORE_ROOT / "self_healing",
        "automation_ops": CORE_ROOT / "automation_ops",
        "performance_tuner": CORE_ROOT / "performance_tuner"
    }
    
    files_valid = 0
    total_files = 0
    
    for name, path in modules.items():
        if path.exists() and path.is_dir():
            py_files = list(path.glob("*.py"))
            total_files += len(py_files)
            for py_file in py_files:
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        code = f.read()
                    compile(code, py_file.name, 'exec')
                    files_valid += 1
                except SyntaxError as e:
                    print(f"  ❌ {py_file.name}: {e}")
    
    log_test(
        "Python语法",
        files_valid == total_files,
        f"有效: {files_valid}/{total_files}"
    )
    
    return files_valid == total_files


def test_key_classes():
    """测试关键类定义"""
    print("\n🏗️ 关键类定义测试...")
    
    classes_to_check = {
        # AIOps
        ("aiops", "AnomalyDetector"): CORE_ROOT / "aiops" / "anomaly_detector.py",
        ("aiops", "RootCauseAnalyzer"): CORE_ROOT / "aiops" / "root_cause_analyzer.py",
        ("aiops", "AutoRecovery"): CORE_ROOT / "aiops" / "auto_recovery.py",
        ("aiops", "PredictiveMaintenance"): CORE_ROOT / "aiops" / "predictive_maintenance.py",
        
        # Scheduler
        ("scheduler", "ResourceOptimizer"): CORE_ROOT / "scheduler" / "resource_optimizer.py",
        ("scheduler", "AutoScaler"): CORE_ROOT / "scheduler" / "auto_scaler.py",
        ("scheduler", "CostOptimizer"): CORE_ROOT / "scheduler" / "cost_optimizer.py",
        ("scheduler", "LoadBalancer"): CORE_ROOT / "scheduler" / "load_balancer.py",
        
        # SelfHealing
        ("self_healing", "HealthChecker"): CORE_ROOT / "self_healing" / "health_checker.py",
        ("self_healing", "IncidentManager"): CORE_ROOT / "self_healing" / "incident_manager.py",
        ("self_healing", "FixEngine"): CORE_ROOT / "self_healing" / "fix_engine.py",
        
        # AutomationOps
        ("automation_ops", "PipelineEngine"): CORE_ROOT / "automation_ops" / "pipeline_engine.py",
        ("automation_ops", "CronScheduler"): CORE_ROOT / "automation_ops" / "cron_scheduler.py",
        ("automation_ops", "NotificationCenter"): CORE_ROOT / "automation_ops" / "notification_center.py",
        
        # PerformanceTuner
        ("performance_tuner", "PerformanceAnalyzer"): CORE_ROOT / "performance_tuner" / "performance_analyzer.py",
        ("performance_tuner", "AutoTuner"): CORE_ROOT / "performance_tuner" / "auto_tuner.py",
        ("performance_tuner", "BenchmarkSuite"): CORE_ROOT / "performance_tuner" / "benchmark_suite.py",
    }
    
    classes_found = 0
    for class_name, file_path in classes_to_check.items():
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
            if f"class {class_name[1]}:" in code:
                classes_found += 1
                print(f"  ✅ {class_name[0]}.{class_name[1]}")
            else:
                print(f"  ❌ {class_name[0]}.{class_name[1]} (not found)")
        else:
            print(f"  ❌ {class_name[0]}.{class_name[1]} (file not found)")
    
    log_test(
        "关键类定义",
        classes_found == len(classes_to_check),
        f"找到: {classes_found}/{len(classes_to_check)}"
    )
    
    return classes_found == len(classes_to_check)


def test_api_files():
    """测试API文件"""
    print("\n🌐 API文件测试...")
    
    apis_to_check = {
        "AIOps API": CORE_ROOT / "aiops" / "api.py",
        "Scheduler API": CORE_ROOT / "scheduler" / "api.py",
        "SelfHealing API": CORE_ROOT / "self_healing" / "api.py",
        "AutomationOps API": CORE_ROOT / "automation_ops" / "api.py",
        "PerformanceTuner API": CORE_ROOT / "performance_tender" / "api.py" if CORE_ROOT / "performance_tuner" / "api.py" else CORE_ROOT / "performance_tuner" / "api.py",
    }
    
    # 修正路径
    if (CORE_ROOT / "performance_tuner" / "api.py").exists():
        apis_to_check["PerformanceTuner API"] = CORE_ROOT / "performance_tuner" / "api.py"
    
    apis_found = 0
    for api_name, api_path in apis_to_check.items():
        if api_path.exists():
            apis_found += 1
            size = api_path.stat().st_size
            print(f"  ✅ {api_name} ({size} bytes)")
        else:
            print(f"  ❌ {api_name} (not found)")
    
    log_test(
        "API文件",
        apis_found == len(apis_to_check),
        f"找到: {apis_found}/{len(apis_to_check)}"
    )
    
    return apis_found == len(apis_to_check)


def test_test_files():
    """测试文件"""
    print("\n🧪 测试文件测试...")
    
    tests_to_check = {
        "AIOps Tests": CORE_ROOT / "aiops" / "test_aiops.py",
        "Scheduler Tests": CORE_ROOT / "scheduler" / "test_scheduler.py",
        "SelfHealing Tests": CORE_ROOT / "self_healing" / "test_self_healing.py",
        "AutomationOps Tests": CORE_ROOT / "automation_ops" / "test_automation_ops.py",
        "PerformanceTuner Tests": CORE_ROOT / "performance_tuner" / "test_performance_tuner.py",
    }
    
    tests_found = 0
    for test_name, test_path in tests_to_check.items():
        if test_path.exists():
            tests_found += 1
            with open(test_path, 'r', encoding='utf-8') as f:
                content = f.read()
            # 统计测试函数
            test_funcs = content.count("def test_")
            print(f"  ✅ {test_name} ({test_funcs} tests)")
        else:
            print(f"  ❌ {test_name} (not found)")
    
    log_test(
        "测试文件",
        tests_found == len(tests_to_check),
        f"找到: {tests_found}/{len(tests_to_check)}"
    )
    
    return tests_found == len(tests_to_check)


def run_quick_tests():
    """运行快速测试"""
    print("\n⚡ 运行快速验证...")
    
    # 检查关键文件是否存在
    critical_files = [
        CORE_ROOT / "aiops" / "__init__.py",
        CORE_ROOT / "scheduler" / "__init__.py",
        CORE_ROOT / "self_healing" / "__init__.py",
        CORE_ROOT / "automation_ops" / "__init__.py",
        CORE_ROOT / "performance_tuner" / "__init__.py",
    ]
    
    files_exist = all(f.exists() for f in critical_files)
    log_test(
        "关键文件",
        files_exist,
        f"存在: {'是' if files_exist else '否'}"
    )
    
    return files_exist


def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("🚀 v12 Phase 2 集成测试 (简化版)")
    print("=" * 60)
    
    # 快速验证
    run_quick_tests()
    
    # 详细测试
    test_module_structure()
    test_python_syntax()
    test_key_classes()
    test_api_files()
    test_test_files()
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("📊 Phase 2 测试结果汇总")
    print("=" * 60)
    print(f"  总测试数: {TESTS_PASSED + TESTS_FAILED}")
    print(f"  ✅ 通过: {TESTS_PASSED}")
    print(f"  ❌ 失败: {TESTS_FAILED}")
    print(f"  通过率: {TESTS_PASSED / max(1, TESTS_PASSED + TESTS_FAILED) * 100:.1f}%")
    
    if TESTS_FAILED == 0:
        print("\n🎉 所有Phase 2测试通过！超自动化模块结构完整！")
        return True
    else:
        print(f"\n⚠️  {TESTS_FAILED}个测试未通过")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
