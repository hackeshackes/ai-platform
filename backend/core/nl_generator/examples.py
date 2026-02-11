"""
使用示例 - Examples

展示自然语言生成器的使用方式
"""

from typing import Dict, List, Any
from .nl_understand import NLUnderstand, UnderstandingResult
from .pipeline_generator import PipelineGenerator, Pipeline
from .agent_generator import AgentGenerator, Agent
from .code_generator import CodeGenerator
from .validator import Validator


def example_create_pipeline():
    """示例：创建Pipeline"""
    print("=" * 60)
    print("示例1: 创建Pipeline")
    print("=" * 60)
    
    # 1. 初始化组件
    nl = NLUnderstand()
    pipeline_gen = PipelineGenerator()
    code_gen = CodeGenerator()
    validator = Validator()
    
    # 2. 用户输入
    user_input = "创建一个处理用户输入的pipeline，包含输入节点、处理节点和输出节点"
    
    print(f"\n用户输入: {user_input}")
    
    # 3. 理解意图
    understanding = nl.understand(user_input)
    print(f"\n理解结果:")
    print(f"  - 意图: {understanding.intent.value}")
    print(f"  - 置信度: {understanding.confidence:.2%}")
    print(f"  - 实体: {[e.value for e in understanding.entities]}")
    
    # 4. 生成Pipeline
    pipeline = pipeline_gen.generate(understanding)
    print(f"\n生成的Pipeline:")
    print(f"  - ID: {pipeline.id}")
    print(f"  - 名称: {pipeline.name}")
    print(f"  - 节点数: {len(pipeline.nodes)}")
    print(f"  - 连接数: {len(pipeline.connections)}")
    
    # 5. 验证
    results = validator.run_all_checks(understanding, pipeline=pipeline)
    summary = validator.get_summary(results)
    print(f"\n验证结果:")
    print(f"  - 错误: {summary['total_errors']}")
    print(f"  - 警告: {summary['total_warnings']}")
    
    # 6. 生成代码
    files = code_gen.generate_pipeline_code(pipeline)
    print(f"\n生成的文件:")
    for f in files:
        print(f"  - {f.filename}")
    
    return understanding, pipeline, files


def example_create_agent():
    """示例：创建Agent"""
    print("\n" + "=" * 60)
    print("示例2: 创建Agent")
    print("=" * 60)
    
    # 1. 初始化组件
    nl = NLUnderstand()
    agent_gen = AgentGenerator()
    code_gen = CodeGenerator()
    validator = Validator()
    
    # 2. 用户输入
    user_input = "创建一个友好的助手Agent，具备对话能力和知识问答功能，使用长期记忆"
    
    print(f"\n用户输入: {user_input}")
    
    # 3. 理解意图
    understanding = nl.understand(user_input)
    print(f"\n理解结果:")
    print(f"  - 意图: {understanding.intent.value}")
    print(f"  - 置信度: {understanding.confidence:.2%}")
    print(f"  - 实体: {[e.value for e in understanding.entities]}")
    
    # 4. 生成Agent
    agent = agent_gen.generate(understanding)
    print(f"\n生成的Agent:")
    print(f"  - ID: {agent.id}")
    print(f"  - 名称: {agent.name}")
    print(f"  - 技能数: {len(agent.skills)}")
    print(f"  - 记忆类型: {agent.memory.type.value}")
    print(f"  - 人格特质: {[t.value for t in agent.personality.traits]}")
    
    # 5. 验证
    results = validator.run_all_checks(understanding, agent=agent)
    summary = validator.get_summary(results)
    print(f"\n验证结果:")
    print(f"  - 错误: {summary['total_errors']}")
    print(f"  - 警告: {summary['total_warnings']}")
    
    # 6. 生成代码
    files = code_gen.generate_agent_code(agent)
    print(f"\n生成的文件:")
    for f in files:
        print(f"  - {f.filename}")
    
    return understanding, agent, files


def example_complex_pipeline():
    """示例：复杂Pipeline"""
    print("\n" + "=" * 60)
    print("示例3: 复杂Pipeline")
    print("=" * 60)
    
    nl = NLUnderstand()
    pipeline_gen = PipelineGenerator()
    code_gen = CodeGenerator()
    validator = Validator()
    
    # 用户输入（指定具体节点）
    user_input = "创建一个数据处理pipeline，输入是API数据，经过滤、转换后输出JSON格式，包含LLM节点进行智能处理"
    
    print(f"\n用户输入: {user_input}")
    
    understanding = nl.understand(user_input)
    print(f"\n理解结果:")
    print(f"  - 意图: {understanding.intent.value}")
    print(f"  - 置信度: {understanding.confidence:.2%}")
    
    pipeline = pipeline_gen.generate(understanding)
    print(f"\nPipeline详情:")
    for node in pipeline.nodes:
        print(f"  - 节点: {node.name} ({node.type.value})")
        print(f"    配置: {node.config}")
    
    # 验证
    results = validator.run_all_checks(understanding, pipeline=pipeline)
    summary = validator.get_summary(results)
    print(f"\n验证摘要: {summary}")
    
    return understanding, pipeline


def example_update_pipeline():
    """示例：更新Pipeline"""
    print("\n" + "=" * 60)
    print("示例4: 更新Pipeline")
    print("=" * 60)
    
    nl = NLUnderstand()
    
    user_input = "修改我的数据处理pipeline，增加一个并行处理节点"
    
    print(f"\n用户输入: {user_input}")
    
    understanding = nl.understand(user_input)
    print(f"\n理解结果:")
    print(f"  - 意图: {understanding.intent.value}")
    print(f"  - 置信度: {understanding.confidence:.2%}")
    print(f"  - 实体: {[e.value for e in understanding.entities]}")
    
    return understanding


def example_multiple_intents():
    """示例：多种意图"""
    print("\n" + "=" * 60)
    print("示例5: 多种意图测试")
    print("=" * 60)
    
    nl = NLUnderstand()
    
    test_cases = [
        ("创建一个文本摘要的pipeline", IntentType.CREATE_PIPELINE),
        ("删除名为test的agent", IntentType.DELETE_AGENT),
        ("运行这个任务", IntentType.EXECUTE_TASK),
        ("查看所有pipeline", IntentType.QUERY_PIPELINE),
    ]
    
    for query, expected_intent in test_cases:
        understanding = nl.understand(query)
        status = "✓" if understanding.intent == expected_intent else "✗"
        print(f"\n{status} 输入: {query}")
        print(f"   识别: {understanding.intent.value} (期望: {expected_intent.value})")
        print(f"   置信度: {understanding.confidence:.2%}")


def example_validation():
    """示例：验证功能"""
    print("\n" + "=" * 60)
    print("示例6: 验证功能")
    print("=" * 60)
    
    validator = Validator()
    
    # 测试无效代码
    invalid_code = """
def broken_function(
    # 缺少闭合括号
    return "hello"
"""
    
    results = validator.validate_code(invalid_code)
    print("\n无效代码验证结果:")
    for result in results:
        print(f"  [{result.level.value}] {result.message}")
    
    # 测试有效代码
    valid_code = """
def hello():
    '''Say hello'''
    return "Hello, World!"
"""
    
    results = validator.validate_code(valid_code)
    print("\n有效代码验证结果:")
    print(f"  通过: {len(results) == 0}")


def example_full_workflow():
    """示例：完整工作流"""
    print("\n" + "=" * 60)
    print("完整工作流示例")
    print("=" * 60)
    
    # 1. 初始化
    nl = NLUnderstand()
    pipeline_gen = PipelineGenerator()
    agent_gen = AgentGenerator()
    code_gen = CodeGenerator()
    validator = Validator()
    
    # 2. 输入
    query = "创建一个智能客服Agent，支持对话、FAQ问答，使用工作记忆"
    
    # 3. 理解
    understanding = nl.understand(query)
    
    # 4. 生成Agent
    agent = agent_gen.generate(understanding)
    
    # 5. 验证
    results = validator.run_all_checks(understanding, agent=agent)
    summary = validator.get_summary(results)
    
    # 6. 输出
    print(f"\n最终结果:")
    print(f"  - Agent名称: {agent.name}")
    print(f"  - 技能数量: {len(agent.skills)}")
    print(f"  - 记忆类型: {agent.memory.type.value}")
    print(f"  - 验证错误: {summary['total_errors']}")
    print(f"  - 验证警告: {summary['total_warnings']}")
    
    # 7. 生成配置
    config = agent.to_dict()
    print(f"\nAgent配置:")
    print(f"  {json.dumps(config, ensure_ascii=False, indent=2)[:500]}...")
    
    return agent, config


def run_all_examples():
    """运行所有示例"""
    print("自然语言生成器 - 使用示例")
    print("=" * 60)
    
    # 示例1: 创建Pipeline
    example_create_pipeline()
    
    # 示例2: 创建Agent
    example_create_agent()
    
    # 示例3: 复杂Pipeline
    example_complex_pipeline()
    
    # 示例4: 更新Pipeline
    example_update_pipeline()
    
    # 示例5: 多种意图
    example_multiple_intents()
    
    # 示例6: 验证功能
    example_validation()
    
    # 完整工作流
    example_full_workflow()
    
    print("\n" + "=" * 60)
    print("所有示例完成!")
    print("=" * 60)


if __name__ == "__main__":
    import json
    
    run_all_examples()
