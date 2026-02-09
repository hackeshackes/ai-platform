#!/usr/bin/env python3
"""
v2.1 端到端测试运行器
"""
import subprocess
import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

def run_tests():
    """运行所有v2.1测试"""
    print("=" * 60)
    print("AI Platform v2.1 端到端测试")
    print("=" * 60)
    
    test_dir = os.path.dirname(__file__)
    
    tests = [
        "test_feature_store.py",
        "test_model_registry.py",
        "test_lineage.py",
        "test_quality.py",
        "test_notebooks.py"
    ]
    
    results = {}
    
    for test in tests:
        test_file = os.path.join(test_dir, test)
        print(f"\n{'='*60}")
        print(f"Running: {test}")
        print("="*60)
        
        result = subprocess.run(
            ["python", "-m", "pytest", test_file, "-v", "--tb=short"],
            cwd=os.path.dirname(os.path.dirname(__file__))
        )
        
        results[test] = {
            "passed": result.returncode == 0,
            "returncode": result.returncode
        }
    
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for test, result in results.items():
        status = "✅ PASS" if result["passed"] else "❌ FAIL"
        print(f"{test}: {status}")
        
        if result["passed"]:
            passed += 1
        else:
            failed += 1
    
    print(f"\n总计: {passed} 通过, {failed} 失败")
    
    return failed == 0

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
