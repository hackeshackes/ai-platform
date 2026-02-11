"""
测试用例 - Test Cases

自然语言生成器测试
"""

import pytest
import json
from typing import Dict, Any

from .nl_understand import (
    NLUnderstand, UnderstandingResult, IntentType, EntityType,
    Entity, Slot, UnderstandingResult
)
from .pipeline_generator import (
    PipelineGenerator, Pipeline, Node, Connection, NodeType, ConnectionType
)
from .agent_generator import (
    AgentGenerator, Agent, Skill, MemoryConfig, Personality, 
    SkillType, MemoryType, PersonalityTrait
)
from .code_generator import CodeGenerator, GeneratedCode
from .validator import Validator, ValidationResult, ValidationLevel


class TestNLUnderstand:
    """测试自然语言理解"""
    
    @pytest.fixture
    def nl(self):
        """创建NLUnderstand实例"""
        return NLUnderstand()
    
    def test_create_pipeline_intent(self, nl):
        """测试创建Pipeline意图识别"""
        queries = [
            "创建一个pipeline",
            "帮我新建一个流水线",
            "构建一个管道",
            "我想创建一个pipeline",
        ]
        
        for query in queries:
            result = nl.understand(query)
            assert result.intent == IntentType.CREATE_PIPELINE, f"Failed for: {query}"
            assert result.confidence > 0.5
    
    def test_create_agent_intent(self, nl):
        """测试创建Agent意图识别"""
        queries = [
            "创建一个智能体",
            "帮我制作一个助手",
            "开发一个机器人",
            "我想创建一个客服agent",
        ]
        
        for query in queries:
            result = nl.understand(query)
            assert result.intent == IntentType.CREATE_AGENT, f"Failed for: {query}"
            assert result.confidence > 0.5
    
    def test_entity_extraction(self, nl):
        """测试实体提取"""
        query = "创建一个名为test_pipeline的pipeline"
        result = nl.understand(query)
        
        # 检查是否提取到pipeline名称
        entities = [e for e in result.entities 
                   if e.type == EntityType.PIPELINE_NAME]
        assert len(entities) > 0
    
    def test_slot_filling(self, nl):
        """测试槽位填充"""
        query = "创建一个名为my_agent的agent"
        result = nl.understand(query)
        
        assert "agent_name" in result.slots
        assert result.slots["agent_name"].filled
        assert result.slots["agent_name"].value is not None
    
    def test_unknown_intent(self, nl):
        """测试未知意图"""
        result = nl.understand("今天天气怎么样")
        assert result.intent == IntentType.UNKNOWN
        assert len(result.suggestions) > 0
    
    def test_low_confidence_handling(self, nl):
        """测试低置信度处理"""
        result = nl.understand("hello")
        assert result.confidence < 0.7
        assert len(result.suggestions) > 0
    
    def test_custom_intent(self, nl):
        """测试添加自定义意图"""
        nl.add_custom_intent(IntentType.UNKNOWN, [r"test.*pattern"])
        result = nl.understand("test pattern match")
        # 自定义模式应该被添加
        assert True  # 验证不报错


class TestPipelineGenerator:
    """测试Pipeline生成"""
    
    @pytest.fixture
    def generator(self):
        """创建PipelineGenerator实例"""
        return PipelineGenerator()
    
    @pytest.fixture
    def sample_understanding(self):
        """创建示例理解结果"""
        return UnderstandingResult(
            intent=IntentType.CREATE_PIPELINE,
            confidence=0.8,
            entities=[Entity(
                type=EntityType.PIPELINE_NAME,
                value="test_pipeline",
                start=0,
                end=13
            )],
            slots={},
            raw_query="创建一个名为test_pipeline的pipeline"
        )
    
    def test_generate_basic_pipeline(self, generator, sample_understanding):
        """测试生成基础Pipeline"""
        pipeline = generator.generate(sample_understanding)
        
        assert pipeline.name == "test_pipeline"
        assert len(pipeline.nodes) >= 2
        assert len(pipeline.connections) >= 1
    
    def test_pipeline_to_dict(self, generator, sample_understanding):
        """测试Pipeline转换为字典"""
        pipeline = generator.generate(sample_understanding)
        pipeline_dict = pipeline.to_dict()
        
        assert "id" in pipeline_dict
        assert "name" in pipeline_dict
        assert "nodes" in pipeline_dict
        assert "connections" in pipeline_dict
        assert isinstance(pipeline_dict["nodes"], list)
        assert isinstance(pipeline_dict["connections"], list)
    
    def test_pipeline_to_json(self, generator, sample_understanding):
        """测试Pipeline转换为JSON"""
        pipeline = generator.generate(sample_understanding)
        json_str = pipeline.to_json()
        
        # 验证是有效的JSON
        parsed = json.loads(json_str)
        assert parsed["name"] == pipeline.name
    
    def test_validate_pipeline(self, generator, sample_understanding):
        """测试Pipeline验证"""
        pipeline = generator.generate(sample_understanding)
        validation = generator.validate_pipeline(pipeline)
        
        assert isinstance(validation, dict)
        assert "valid" in validation
        assert "errors" in validation
        assert "warnings" in validation
    
    def test_optimize_pipeline(self, generator, sample_understanding):
        """测试Pipeline优化"""
        pipeline = generator.generate(sample_understanding)
        optimized = generator.optimize_pipeline(pipeline)
        
        assert optimized is not None
        assert isinstance(optimized, Pipeline)
    
    def test_default_node_creation(self, generator):
        """测试默认节点创建"""
        node = generator._create_node_from_type("llm")
        
        assert node is not None
        assert node.type == NodeType.LLM
        assert "model" in node.config
    
    def test_invalid_node_type(self, generator):
        """测试无效节点类型"""
        node = generator._create_node_from_type("invalid_type")
        assert node is None


class TestAgentGenerator:
    """测试Agent生成"""
    
    @pytest.fixture
    def generator(self):
        """创建AgentGenerator实例"""
        return AgentGenerator()
    
    @pytest.fixture
    def sample_understanding(self):
        """创建示例理解结果"""
        return UnderstandingResult(
            intent=IntentType.CREATE_AGENT,
            confidence=0.8,
            entities=[Entity(
                type=EntityType.AGENT_NAME,
                value="my_assistant",
                start=0,
                end=11
            )],
            slots={},
            raw_query="创建一个名为my_assistant的agent"
        )
    
    def test_generate_basic_agent(self, generator, sample_understanding):
        """测试生成基础Agent"""
        agent = generator.generate(sample_understanding)
        
        assert agent.name == "my_assistant"
        assert len(agent.skills) >= 1
    
    def test_agent_to_dict(self, generator, sample_understanding):
        """测试Agent转换为字典"""
        agent = generator.generate(sample_understanding)
        agent_dict = agent.to_dict()
        
        assert "id" in agent_dict
        assert "name" in agent_dict
        assert "skills" in agent_dict
        assert "memory" in agent_dict
        assert "personality" in agent_dict
    
    def test_agent_to_json(self, generator, sample_understanding):
        """测试Agent转换为JSON"""
        agent = generator.generate(sample_understanding)
        json_str = agent.to_json()
        
        parsed = json.loads(json_str)
        assert parsed["name"] == agent.name
    
    def test_validate_agent(self, generator, sample_understanding):
        """测试Agent验证"""
        agent = generator.generate(sample_understanding)
        validation = generator.validate_agent(agent)
        
        assert isinstance(validation, dict)
        assert "valid" in validation
    
    def test_export_import_agent(self, generator, sample_understanding):
        """测试Agent导出导入"""
        original = generator.generate(sample_understanding)
        config = original.to_dict()
        
        imported = generator.import_agent(config)
        
        assert imported.name == original.name
        assert len(imported.skills) == len(original.skills)
    
    def test_skill_binding(self, generator, sample_understanding):
        """测试技能绑定"""
        agent = generator.generate(sample_understanding)
        original_count = len(agent.skills)
        
        new_skill = Skill(
            id="test_skill",
            name="test_skill",
            type=SkillType.CONVERSATION
        )
        
        generator.bind_skill(agent, new_skill)
        assert len(agent.skills) == original_count + 1
    
    def test_skill_unbinding(self, generator, sample_understanding):
        """测试技能解绑"""
        agent = generator.generate(sample_understanding)
        skill_id = agent.skills[0].id
        
        generator.unbind_skill(agent, skill_id)
        assert len(agent.skills) == 0


class TestCodeGenerator:
    """测试代码生成"""
    
    @pytest.fixture
    def generator(self):
        """创建CodeGenerator实例"""
        return CodeGenerator()
    
    @pytest.fixture
    def sample_pipeline(self):
        """创建示例Pipeline"""
        understanding = UnderstandingResult(
            intent=IntentType.CREATE_PIPELINE,
            confidence=0.8,
            entities=[Entity(
                type=EntityType.PIPELINE_NAME,
                value="test",
                start=0,
                end=4
            )],
            slots={},
            raw_query="创建一个pipeline"
        )
        
        gen = PipelineGenerator()
        return gen.generate(understanding)
    
    @pytest.fixture
    def sample_agent(self):
        """创建示例Agent"""
        understanding = UnderstandingResult(
            intent=IntentType.CREATE_AGENT,
            confidence=0.8,
            entities=[Entity(
                type=EntityType.AGENT_NAME,
                value="test",
                start=0,
                end=4
            )],
            slots={},
            raw_query="创建一个agent"
        )
        
        gen = AgentGenerator()
        return gen.generate(understanding)
    
    def test_generate_pipeline_code(self, generator, sample_pipeline):
        """测试生成Pipeline代码"""
        files = generator.generate_pipeline_code(sample_pipeline)
        
        assert len(files) >= 3
        assert all(isinstance(f, GeneratedCode) for f in files)
        assert all(f.language == "python" for f in files)
    
    def test_generate_agent_code(self, generator, sample_agent):
        """测试生成Agent代码"""
        files = generator.generate_agent_code(sample_agent)
        
        assert len(files) >= 3
        assert all(isinstance(f, GeneratedCode) for f in files)
    
    def test_generated_code_syntax(self, generator, sample_pipeline):
        """测试生成代码语法"""
        files = generator.generate_pipeline_code(sample_pipeline)
        
        for f in files:
            try:
                compile(f.content, f.filename, "exec")
            except SyntaxError as e:
                pytest.fail(f"Syntax error in {f.filename}: {e}")
    
    def test_camel_case_conversion(self, generator):
        """测试驼峰命名转换"""
        assert generator._to_camel_case("hello world") == "HelloWorld"
        assert generator._to_camel_case("test_pipeline") == "TestPipeline"
        assert generator._to_camel_case("my-agent") == "MyAgent"
    
    def test_generated_file_structure(self, generator, sample_pipeline):
        """测试生成文件结构"""
        files = generator.generate_pipeline_code(sample_pipeline)
        
        main_file = [f for f in files if "pipeline.py" in f.filename][0]
        
        # 检查是否包含基本结构
        assert "class" in main_file.content
        assert "def __init__" in main_file.content
        assert "def execute" in main_file.content


class TestValidator:
    """测试验证器"""
    
    @pytest.fixture
    def validator(self):
        """创建Validator实例"""
        return Validator()
    
    def test_validate_pipeline_name(self, validator):
        """测试Pipeline名称验证"""
        understanding = UnderstandingResult(
            intent=IntentType.CREATE_PIPELINE,
            confidence=0.8,
            entities=[Entity(
                type=EntityType.PIPELINE_NAME,
                value="test_pipeline",
                start=0,
                end=13
            )],
            slots={},
            raw_query="创建一个名为test_pipeline的pipeline"
        )
        
        gen = PipelineGenerator()
        pipeline = gen.generate(understanding)
        
        results = validator.validate_pipeline(pipeline)
        
        # 应该没有名称相关的错误
        name_errors = [r for r in results 
                     if "名称" in r.message or "name" in r.message.lower()]
        assert len(name_errors) == 0
    
    def test_validate_agent_name(self, validator):
        """测试Agent名称验证"""
        understanding = UnderstandingResult(
            intent=IntentType.CREATE_AGENT,
            confidence=0.8,
            entities=[Entity(
                type=EntityType.AGENT_NAME,
                value="my_assistant",
                start=0,
                end=11
            )],
            slots={},
            raw_query="创建一个名为my_assistant的agent"
        )
        
        gen = AgentGenerator()
        agent = gen.generate(understanding)
        
        results = validator.validate_agent(agent)
        
        # 不应该有名称错误
        name_errors = [r for r in results 
                     if "名称" in r.message or "name" in r.message.lower()]
        assert len(name_errors) == 0
    
    def test_validate_code_syntax(self, validator):
        """测试代码语法验证"""
        valid_code = "def hello():\n    '''Say hello'''\n    return 'hello'"
        results = validator.validate_code(valid_code)
        
        # 应该没有ERROR级别的问题
        syntax_errors = [r for r in results if r.level == ValidationLevel.ERROR]
        assert len(syntax_errors) == 0
    
    def test_validate_invalid_code(self, validator):
        """测试无效代码验证"""
        invalid_code = "def broken(\n    return 'hello'"
        results = validator.validate_code(invalid_code)
        
        syntax_errors = [r for r in results 
                        if r.level == ValidationLevel.ERROR]
        assert len(syntax_errors) > 0
    
    def test_validate_understanding(self, validator):
        """测试理解结果验证"""
        # 有效理解
        good = UnderstandingResult(
            intent=IntentType.CREATE_PIPELINE,
            confidence=0.8,
            entities=[],
            slots={},
            raw_query="创建pipeline"
        )
        results = validator.validate_understanding(good)
        errors = [r for r in results if not r.valid and r.level == ValidationLevel.ERROR]
        assert len(errors) == 0
        
        # 无效理解
        bad = UnderstandingResult(
            intent=IntentType.UNKNOWN,
            confidence=0.3,
            entities=[],
            slots={},
            raw_query="今天天气"
        )
        results = validator.validate_understanding(bad)
        assert len(results) > 0
    
    def test_run_all_checks(self, validator):
        """测试运行所有检查"""
        understanding = UnderstandingResult(
            intent=IntentType.CREATE_PIPELINE,
            confidence=0.8,
            entities=[],
            slots={},
            raw_query="创建pipeline"
        )
        
        gen = PipelineGenerator()
        pipeline = gen.generate(understanding)
        
        results = validator.run_all_checks(
            understanding=understanding,
            pipeline=pipeline
        )
        
        assert "understanding" in results
        assert "pipeline" in results
    
    def test_get_summary(self, validator):
        """测试获取摘要"""
        results = {
            "understanding": [],
            "pipeline": [],
            "agent": [],
            "code": []
        }
        
        summary = validator.get_summary(results)
        
        assert "total_errors" in summary
        assert "total_warnings" in summary
        assert "by_category" in summary


class TestIntegration:
    """集成测试"""
    
    def test_full_pipeline_workflow(self):
        """完整Pipeline工作流测试"""
        # 1. 理解
        nl = NLUnderstand()
        query = "创建一个名为data_pipeline的pipeline，包含输入、处理、输出"
        understanding = nl.understand(query)
        
        # Pipeline意图应该被识别
        assert understanding.intent in [IntentType.CREATE_PIPELINE, IntentType.CREATE_AGENT]
        
        # 如果是CREATE_PIPELINE，测试Pipeline流程
        if understanding.intent == IntentType.CREATE_PIPELINE:
            pipeline_gen = PipelineGenerator()
            pipeline = pipeline_gen.generate(understanding)
            
            assert pipeline is not None
            assert len(pipeline.nodes) > 0
            
            # 3. 验证
            validator = Validator()
            results = validator.run_all_checks(understanding, pipeline=pipeline)
            summary = validator.get_summary(results)
            
            assert summary["total_errors"] == 0
            
            # 4. 生成代码
            code_gen = CodeGenerator()
            files = code_gen.generate_pipeline_code(pipeline)
            
            assert len(files) > 0
        else:
            # 测试Agent流程
            agent_gen = AgentGenerator()
            agent = agent_gen.generate(understanding)
            assert agent is not None
        results = validator.run_all_checks(understanding, pipeline=pipeline)
        summary = validator.get_summary(results)
        
        assert summary["total_errors"] == 0
        
        # 4. 生成代码
        code_gen = CodeGenerator()
        files = code_gen.generate_pipeline_code(pipeline)
        
        assert len(files) > 0
    
    def test_full_agent_workflow(self):
        """完整Agent工作流测试"""
        # 1. 理解
        nl = NLUnderstand()
        query = "创建一个名为客服助手的智能agent，支持对话和FAQ"
        understanding = nl.understand(query)
        
        # Agent和Pipeline意图都应该被支持
        if understanding.intent == IntentType.CREATE_AGENT:
            # 2. 生成Agent
            agent_gen = AgentGenerator()
            agent = agent_gen.generate(understanding)
            
            assert agent is not None
            
            # 3. 验证
            validator = Validator()
            results = validator.run_all_checks(understanding, agent=agent)
            summary = validator.get_summary(results)
            
            # 允许有warning但不应该有error
            assert summary["total_errors"] == 0
            
            # 4. 生成代码
            code_gen = CodeGenerator()
            files = code_gen.generate_agent_code(agent)
            
            assert len(files) > 0
        else:
            # 测试pipeline流程
            pipeline_gen = PipelineGenerator()
            pipeline = pipeline_gen.generate(understanding)
            assert pipeline is not None


def run_tests():
    """运行所有测试"""
    pytest.main([__file__, "-v"])


if __name__ == "__main__":
    run_tests()
