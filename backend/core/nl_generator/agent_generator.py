"""
Agent生成器模块 - Agent Generator

负责从理解结果生成Agent配置
"""

import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from .nl_understand import NLUnderstand, UnderstandingResult, IntentType, EntityType


class SkillType(Enum):
    """技能类型"""
    CONVERSATION = "conversation"
    KNOWLEDGE_QNA = "knowledge_qna"
    TASK_EXECUTION = "task_execution"
    CODE_GENERATION = "code_generation"
    DATA_ANALYSIS = "data_analysis"
    TRANSLATION = "translation"
    TEXT_SUMMARY = "text_summary"
    IMAGE_GENERATION = "image_generation"
    WEB_SEARCH = "web_search"
    CUSTOM = "custom"


class MemoryType(Enum):
    """记忆类型"""
    NONE = "none"
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"


class PersonalityTrait(Enum):
    """人格特质"""
    FRIENDLY = "friendly"
    PROFESSIONAL = "professional"
    HUMOROUS = "humorous"
    PATIENT = "patient"
    ENTHUSIASTIC = "enthusiastic"
    ANALYTICAL = "analytical"
    CREATIVE = "creative"


@dataclass
class Skill:
    """Agent技能"""
    id: str
    name: str
    type: SkillType
    config: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    priority: int = 0
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type.value,
            "config": self.config,
            "enabled": self.enabled,
            "priority": self.priority
        }


@dataclass
class MemoryConfig:
    """记忆配置"""
    type: MemoryType
    max_items: int = 100
    ttl_seconds: Optional[int] = None
    embedding_model: str = "default"
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "type": self.type.value,
            "max_items": self.max_items,
            "ttl_seconds": self.ttl_seconds,
            "embedding_model": self.embedding_model
        }


@dataclass
class Personality:
    """人格配置"""
    traits: List[PersonalityTrait] = field(default_factory=list)
    tone: str = "neutral"
    style: str = "concise"
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "traits": [t.value for t in self.traits],
            "tone": self.tone,
            "style": self.style
        }


@dataclass
class Agent:
    """Agent定义"""
    id: str
    name: str
    description: str = ""
    version: str = "1.0.0"
    skills: List[Skill] = field(default_factory=list)
    memory: MemoryConfig = field(default_factory=lambda: MemoryConfig(type=MemoryType.NONE))
    personality: Personality = field(default_factory=lambda: Personality())
    config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "skills": [s.to_dict() for s in self.skills],
            "memory": self.memory.to_dict(),
            "personality": self.personality.to_dict(),
            "config": self.config,
            "metadata": self.metadata
        }
    
    def to_json(self) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


class AgentGenerator:
    """Agent生成器"""
    
    def __init__(self):
        self.skill_counter = 0
    
    def generate(self, understanding: UnderstandingResult) -> Agent:
        """
        从理解结果生成Agent
        
        Args:
            understanding: 自然语言理解结果
            
        Returns:
            Agent: 生成的Agent
        """
        # 1. 确定Agent基本信息
        agent = self._create_base_agent(understanding)
        
        # 2. 生成技能
        skills = self._generate_skills(understanding)
        agent.skills = skills
        
        # 3. 配置记忆
        agent.memory = self._configure_memory(understanding)
        
        # 4. 配置人格
        agent.personality = self._configure_personality(understanding)
        
        # 5. 设置配置参数
        agent.config = self._extract_config(understanding)
        
        return agent
    
    def _create_base_agent(self, understanding: UnderstandingResult) -> Agent:
        """创建基础Agent"""
        # 提取名称
        name = "unnamed_agent"
        for entity in understanding.entities:
            if entity.type == EntityType.AGENT_NAME:
                name = entity.value
                break
        
        return Agent(
            id=self._generate_id("agent"),
            name=name,
            description=f"Generated from: {understanding.raw_query}",
            metadata={
                "source": "nl_generator",
                "confidence": understanding.confidence,
                "generated_at": self._get_timestamp()
            }
        )
    
    def _generate_skills(self, understanding: UnderstandingResult) -> List[Skill]:
        """生成技能"""
        skills = []
        
        # 分析意图来确定需要的技能
        intent_skills = self._get_skills_for_intent(understanding.intent)
        
        # 添加实体中指定的技能
        skill_entities = self._extract_skill_entities(understanding)
        
        all_skills = intent_skills + skill_entities
        skill_names = set()
        
        for skill_info in all_skills:
            if isinstance(skill_info, tuple):
                skill_type, skill_name = skill_info
            else:
                skill_type = self._string_to_skill_type(skill_info)
                skill_name = skill_info
            
            if skill_name in skill_names:
                continue
            
            skill_names.add(skill_name)
            skill = self._create_skill(skill_type, skill_name)
            skills.append(skill)
        
        # 至少添加一个对话技能
        if not skills:
            default_skill = self._create_skill(SkillType.CONVERSATION, "conversation")
            skills.append(default_skill)
        
        return skills
    
    def _get_skills_for_intent(self, intent: IntentType) -> List[Skill]:
        """根据意图获取默认技能"""
        intent_skill_map = {
            IntentType.CREATE_AGENT: [
                (SkillType.CONVERSATION, "conversation"),
            ],
            IntentType.UPDATE_AGENT: [
                (SkillType.CONVERSATION, "conversation"),
            ],
            IntentType.EXECUTE_TASK: [
                (SkillType.TASK_EXECUTION, "task_execution"),
            ],
        }
        
        return intent_skill_map.get(intent, [])
    
    def _extract_skill_entities(self, understanding: UnderstandingResult) -> List[str]:
        """从实体中提取技能"""
        skills = []
        
        # 从实体值中解析技能
        for entity in understanding.entities:
            value_lower = entity.value.lower()
            skill_keywords = [
                "conversation", "对话", "qa", "问答", "知识",
                "task", "任务", "代码", "code", "分析", "analysis",
                "翻译", "translation", "总结", "summary", "搜索", "search"
            ]
            
            for keyword in skill_keywords:
                if keyword in value_lower:
                    skills.append(keyword)
                    break
        
        return skills
    
    def _string_to_skill_type(self, skill_name: str) -> SkillType:
        """将技能名称转换为类型"""
        name_lower = skill_name.lower()
        
        if "conversation" in name_lower or "对话" in name_lower:
            return SkillType.CONVERSATION
        elif "qa" in name_lower or "问答" in name_lower or "知识" in name_lower:
            return SkillType.KNOWLEDGE_QNA
        elif "task" in name_lower or "任务" in name_lower:
            return SkillType.TASK_EXECUTION
        elif "code" in name_lower or "代码" in name_lower:
            return SkillType.CODE_GENERATION
        elif "analysis" in name_lower or "分析" in name_lower:
            return SkillType.DATA_ANALYSIS
        elif "translation" in name_lower or "翻译" in name_lower:
            return SkillType.TRANSLATION
        elif "summary" in name_lower or "总结" in name_lower:
            return SkillType.TEXT_SUMMARY
        elif "search" in name_lower or "搜索" in name_lower:
            return SkillType.WEB_SEARCH
        else:
            return SkillType.CUSTOM
    
    def _create_skill(self, skill_type: SkillType, name: str) -> Skill:
        """创建技能"""
        return Skill(
            id=self._generate_id("skill"),
            name=name,
            type=skill_type,
            config=self._get_default_skill_config(skill_type)
        )
    
    def _get_default_skill_config(self, skill_type: SkillType) -> Dict[str, Any]:
        """获取技能默认配置"""
        config_maps = {
            SkillType.CONVERSATION: {
                "greeting": "你好！有什么可以帮助你的吗？",
                "max_history": 10
            },
            SkillType.KNOWLEDGE_QNA: {
                "knowledge_base": "",
                "top_k": 5
            },
            SkillType.TASK_EXECUTION: {
                "allowed_operations": ["read", "write"],
                "require_confirmation": True
            },
            SkillType.CODE_GENERATION: {
                "language": "python",
                "safety_check": True
            },
            SkillType.DATA_ANALYSIS: {
                "visualization": True,
                "max_rows": 10000
            },
            SkillType.TRANSLATION: {
                "source_languages": ["auto"],
                "target_language": "zh"
            },
            SkillType.TEXT_SUMMARY: {
                "max_length": 200,
                "style": "concise"
            },
            SkillType.WEB_SEARCH: {
                "max_results": 5,
                "safe_search": True
            },
            SkillType.CUSTOM: {}
        }
        
        return config_maps.get(skill_type, {})
    
    def _configure_memory(self, understanding: UnderstandingResult) -> MemoryConfig:
        """配置记忆"""
        # 默认使用短期记忆
        memory = MemoryConfig(type=MemoryType.SHORT_TERM)
        
        # 检查实体中是否指定了记忆类型
        for entity in understanding.entities:
            if "长期" in entity.value or "long_term" in entity.value.lower():
                memory.type = MemoryType.LONG_TERM
            elif "工作" in entity.value or "working" in entity.value.lower():
                memory.type = MemoryType.WORKING
            elif "情景" in entity.value or "episodic" in entity.value.lower():
                memory.type = MemoryType.EPISODIC
            elif "语义" in entity.value or "semantic" in entity.value.lower():
                memory.type = MemoryType.SEMANTIC
            elif "无" in entity.value or "none" in entity.value.lower():
                memory.type = MemoryType.NONE
        
        return memory
    
    def _configure_personality(self, understanding: UnderstandingResult) -> Personality:
        """配置人格"""
        personality = Personality()
        
        # 分析用户描述中的性格关键词
        query = understanding.raw_query.lower()
        
        if "友好" in query or "friendly" in query:
            personality.traits.append(PersonalityTrait.FRIENDLY)
        if "专业" in query or "professional" in query:
            personality.traits.append(PersonalityTrait.PROFESSIONAL)
        if "幽默" in query or "humorous" in query:
            personality.traits.append(PersonalityTrait.HUMOROUS)
        if "耐心" in query or "patient" in query:
            personality.traits.append(PersonalityTrait.PATIENT)
        if "热情" in query or "enthusiastic" in query:
            personality.traits.append(PersonalityTrait.ENTHUSIASTIC)
        if "分析" in query or "analytical" in query:
            personality.traits.append(PersonalityTrait.ANALYTICAL)
        if "创意" in query or "creative" in query:
            personality.traits.append(PersonalityTrait.CREATIVE)
        
        # 设置语气
        if "正式" in query or "formal" in query:
            personality.tone = "formal"
        elif "随意" in query or "casual" in query:
            personality.tone = "casual"
        elif "友好" in query:
            personality.tone = "friendly"
        
        # 设置风格
        if "详细" in query or "detailed" in query:
            personality.style = "detailed"
        elif "简洁" in query or "concise" in query:
            personality.style = "concise"
        
        return personality
    
    def _extract_config(self, understanding: UnderstandingResult) -> Dict[str, Any]:
        """提取配置参数"""
        config = {}
        
        for entity in understanding.entities:
            if entity.type == EntityType.PARAMETER:
                parts = entity.value.split("=")
                if len(parts) == 2:
                    config[parts[0].strip()] = parts[1].strip()
        
        return config
    
    def _generate_id(self, prefix: str) -> str:
        """生成唯一ID"""
        import uuid
        return f"{prefix}_{uuid.uuid4().hex[:8]}"
    
    def _get_timestamp(self) -> str:
        """获取时间戳"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def bind_skill(self, agent: Agent, skill: Skill, position: int = None):
        """
        绑定技能到Agent
        
        Args:
            agent: Agent实例
            skill: 技能
            position: 位置（可选）
        """
        if position is not None:
            agent.skills.insert(position, skill)
        else:
            agent.skills.append(skill)
    
    def unbind_skill(self, agent: Agent, skill_id: str):
        """解绑技能"""
        agent.skills = [s for s in agent.skills if s.id != skill_id]
    
    def validate_agent(self, agent: Agent) -> Dict[str, Any]:
        """
        验证Agent配置
        
        Args:
            agent: Agent实例
            
        Returns:
            Dict: 验证结果
        """
        errors = []
        warnings = []
        
        # 检查名称
        if not agent.name or len(agent.name) < 2:
            errors.append("Agent名称不能为空且至少2个字符")
        
        # 检查技能
        if not agent.skills:
            warnings.append("Agent没有绑定任何技能")
        
        # 检查技能ID唯一性
        skill_ids = [s.id for s in agent.skills]
        if len(skill_ids) != len(set(skill_ids)):
            errors.append("存在重复的技能ID")
        
        # 检查记忆配置
        if agent.memory.type == MemoryType.NONE:
            warnings.append("Agent没有配置记忆，可能影响上下文理解")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }
    
    def export_agent(self, agent: Agent, format: str = "json") -> str:
        """
        导出Agent配置
        
        Args:
            agent: Agent实例
            format: 导出格式 (json/yaml)
            
        Returns:
            str: 导出的配置字符串
        """
        if format == "json":
            return agent.to_json()
        else:
            # 简单YAML转换
            import yaml
            return yaml.dump(agent.to_dict(), allow_unicode=True)
    
    def import_agent(self, config: Dict) -> Agent:
        """
        导入Agent配置
        
        Args:
            config: 配置字典
            
        Returns:
            Agent: Agent实例
        """
        # 解析技能
        skills = []
        for skill_dict in config.get("skills", []):
            skill = Skill(
                id=skill_dict["id"],
                name=skill_dict["name"],
                type=SkillType(skill_dict["type"]),
                config=skill_dict.get("config", {}),
                enabled=skill_dict.get("enabled", True),
                priority=skill_dict.get("priority", 0)
            )
            skills.append(skill)
        
        # 解析记忆
        memory = MemoryConfig(
            type=MemoryType(config["memory"]["type"]),
            max_items=config["memory"].get("max_items", 100),
            ttl_seconds=config["memory"].get("ttl_seconds"),
            embedding_model=config["memory"].get("embedding_model", "default")
        )
        
        # 解析人格
        personality = Personality(
            traits=[PersonalityTrait(t) for t in config.get("personality", {}).get("traits", [])],
            tone=config.get("personality", {}).get("tone", "neutral"),
            style=config.get("personality", {}).get("style", "concise")
        )
        
        return Agent(
            id=config["id"],
            name=config["name"],
            description=config.get("description", ""),
            version=config.get("version", "1.0.0"),
            skills=skills,
            memory=memory,
            personality=personality,
            config=config.get("config", {}),
            metadata=config.get("metadata", {})
        )
