#!/usr/bin/env python3
"""
v12 Phase 1 Integration Tests

测试所有Phase 1模块的协同工作能力
"""

import asyncio
import sys
import os
from pathlib import Path

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent
BACKEND_ROOT = PROJECT_ROOT / "backend"
CORE_ROOT = BACKEND_ROOT / "core"
sys.path.insert(0, str(BACKEND_ROOT))
sys.path.insert(0, str(CORE_ROOT))
sys.path.insert(0, str(CORE_ROOT / "nl_generator"))
sys.path.insert(0, str(CORE_ROOT / "recommender"))
sys.path.insert(0, str(CORE_ROOT / "autodoc"))
sys.path.insert(0, str(BACKEND_ROOT / "marketplace" / "templates"))

# 测试结果
TESTS_PASSED = 0
TESTS_FAILED = 0
TEST_RESULTS = []


def log_test(name: str, passed: bool, details: str = ""):
    """记录测试结果"""
    global TESTS_PASSED, TESTS_FAILED
    if passed:
        TESTS_PASSED += 1
        status = "✅ PASS"
    else:
        TESTS_FAILED += 1
        status = "❌ FAIL"
    
    result = {
        "name": name,
        "passed": passed,
        "details": details
    }
    TEST_RESULTS.append(result)
    print(f"{status} | {name}")
    if details:
        print(f"    └── {details}")


def test_nl_generator():
    """测试自然语言生成器"""
    print("\n🔍 测试自然语言生成器...")
    
    try:
        from nl_generator import NLUnderstand, PipelineGenerator, AgentGenerator
        
        # 测试1: 意图识别
        nl = NLUnderstand()
        result = nl.understand("创建一个客服机器人")
        log_test(
            "NL意图识别",
            result.intent.value.startswith("create") and result.confidence > 0.5,
            f"意图: {result.intent.value}, 置信度: {result.confidence:.2f}"
        )
        
        # 测试2: Pipeline生成 (传入理解结果)
        pg = PipelineGenerator()
        pipeline = pg.generate(result)
        pipeline_dict = pipeline.to_dict() if hasattr(pipeline, 'to_dict') else pipeline
        log_test(
            "Pipeline生成",
            pipeline is not None and hasattr(pipeline, 'id'),
            f"Pipeline ID: {pipeline.id if hasattr(pipeline, 'id') else 'N/A'}"
        )
        
        # 测试3: Agent生成 (传入理解结果)
        ag = AgentGenerator()
        agent = ag.generate(result)
        agent_dict = agent.to_dict() if hasattr(agent, 'to_dict') else agent
        log_test(
            "Agent生成",
            agent is not None and hasattr(agent, 'name'),
            f"Agent名称: {agent.name if hasattr(agent, 'name') else 'N/A'}"
        )
        
        return True
    except Exception as e:
        log_test("NL Generator模块加载", False, str(e))
        return False


def test_recommender():
    """测试智能推荐系统"""
    print("\n🔍 测试智能推荐系统...")
    
    try:
        from recommender import HybridRecommender, UserProfile
        
        # 测试1: 用户画像
        profile = UserProfile(user_id="test_user_001")
        log_test(
            "用户画像创建",
            profile is not None and profile.user_id == "test_user_001",
            f"用户: {profile.user_id}"
        )
        
        # 测试2: 混合推荐
        recommender = HybridRecommender()
        has_recommend = hasattr(recommender, 'recommend')
        log_test(
            "混合推荐方法存在",
            has_recommend,
            "recommend方法可用"
        )
        
        return True
    except Exception as e:
        log_test("Recommender模块加载", False, str(e))
        return False


def test_template_marketplace():
    """测试AI模板市场"""
    print("\n🔍 测试AI模板市场...")
    
    try:
        import json
        from pathlib import Path
        
        template_path = BACKEND_ROOT / "marketplace" / "templates" / "index.json"
        
        # 测试1: 模板索引存在
        log_test(
            "模板索引存在",
            template_path.exists(),
            str(template_path)
        )
        
        # 测试2: 加载模板 (注意: JSON结构是 {categories: [...], templates: [...]})
        if template_path.exists():
            with open(template_path) as f:
                data = json.load(f)
            
            # 支持两种格式: 直接模板数组 或 带categories的格式
            if isinstance(data, list):
                templates = data
            else:
                templates = data.get("templates", []) or data.get("categories", [{}])[0].get("templates", [])
            
            log_test(
                "模板加载",
                len(templates) >= 3,
                f"模板数: {len(templates)}"
            )
            
            # 测试3: 模板结构
            if templates:
                template = templates[0]
                required_fields = ["id", "name", "description", "category"]
                has_all = all(field in template for field in required_fields)
                log_test(
                    "模板结构正确",
                    has_all,
                    f"模板: {template.get('name', 'N/A')}"
                )
                # 检查可选字段
                has_downloads = "downloads" in template
                has_rating = "rating" in template
                log_test(
                    "模板统计完整",
                    has_downloads or has_rating,
                    f"下载:{template.get('downloads', 'N/A')} 评分:{template.get('rating', 'N/A')}"
                )
        
        return True
    except Exception as e:
        log_test("Template Marketplace模块", False, str(e))
        return False


def test_autodoc():
    """测试自动文档生成器"""
    print("\n🔍 测试自动文档生成器...")
    
    try:
        from autodoc.code_parser import CodeParser
        from autodoc.api_extractor import APIExtractor
        
        # 测试1: 代码解析
        parser = CodeParser(language='python')
        test_code = '''
def example_function(param1: str, param2: int) -> bool:
    """Example function"""
    return True
'''
        parsed = parser.parse_code(test_code)
        log_test(
            "代码解析",
            parsed is not None and "functions" in parsed,
            f"解析函数数: {len(parsed.get('functions', []))}"
        )
        
        # 测试2: API提取 (需要module_name参数)
        extractor = APIExtractor(language='python')
        module = extractor.extract_from_code(test_code, module_name="test_module")
        log_test(
            "API提取",
            module is not None and hasattr(module, 'functions'),
            f"API数: {len(module.functions) if hasattr(module, 'functions') else 0}"
        )
        
        return True
    except Exception as e:
        log_test("AutoDoc模块", False, str(e))
        return False


def test_integration():
    """端到端集成测试"""
    print("\n🔍 测试端到端集成...")
    
    try:
        # 测试1: NL → Pipeline
        from nl_generator import NLUnderstand, PipelineGenerator
        
        nl = NLUnderstand()
        pg = PipelineGenerator()
        
        # 自然语言 → 理解结果 → Pipeline
        result = nl.understand("销售数据分析")
        pipeline = pg.generate(result)
        pipeline_id = pipeline.id if hasattr(pipeline, 'id') else "N/A"
        log_test(
            "NL→Pipeline 生成",
            pipeline is not None,
            f"Pipeline ID: {pipeline_id}"
        )
        
        # 测试2: 模板完整性验证
        import json
        template_path = BACKEND_ROOT / "marketplace" / "templates" / "index.json"
        if template_path.exists():
            with open(template_path) as f:
                data = json.load(f)
            
            # 支持两种格式: 直接模板数组 或 带categories的格式
            if isinstance(data, list):
                templates = data
            else:
                templates = data.get("templates", []) or data.get("categories", [{}])[0].get("templates", []) if data.get("categories") else []
            
            if templates:
                template = templates[0]
                has_name = "name" in template
                has_desc = "description" in template
                has_category = "category" in template
                all_fields = has_name and has_desc and has_category
                log_test(
                    "模板数据完整",
                    all_fields,
                    f"字段完整: {all_fields}"
                )
        
        return True
    except Exception as e:
        log_test("端到端集成", False, str(e))
        return False


def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("🚀 v12 Phase 1 集成测试")
    print("=" * 60)
    
    # 测试模块导入
    print("\n📦 模块导入测试...")
    
    modules_tested = 0
    try:
        import nl_generator
        modules_tested += 1
        print("  ✅ nl_generator")
    except Exception as e:
        print(f"  ❌ nl_generator: {e}")
    
    try:
        import recommender
        modules_tested += 1
        print("  ✅ recommender")
    except Exception as e:
        print(f"  ❌ recommender: {e}")
    
    try:
        import autodoc
        modules_tested += 1
        print("  ✅ autodoc")
    except Exception as e:
        print(f"  ❌ autodoc: {e}")
    
    log_test("模块导入", modules_tested >= 3, f"成功导入: {modules_tested}/3")
    
    # 运行各类测试
    test_nl_generator()
    test_recommender()
    test_template_marketplace()
    test_autodoc()
    test_integration()
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("📊 测试结果汇总")
    print("=" * 60)
    print(f"  总测试数: {TESTS_PASSED + TESTS_FAILED}")
    print(f"  ✅ 通过: {TESTS_PASSED}")
    print(f"  ❌ 失败: {TESTS_FAILED}")
    print(f"  通过率: {TESTS_PASSED / max(1, TESTS_PASSED + TESTS_FAILED) * 100:.1f}%")
    
    if TESTS_FAILED == 0:
        print("\n🎉 所有测试通过！v12 Phase 1 集成成功！")
    else:
        print(f"\n⚠️  {TESTS_FAILED}个测试失败，需要修复")
    
    return TESTS_FAILED == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
