"""
Prompt Management模块 v2.4
"""
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4
from enum import Enum

class PromptType(str, Enum):
    """Prompt类型"""
    COMPLETION = "completion"
    CHAT = "chat"
    RAG = "rag"
    AGENT = "agent"
    CLASSIFICATION = "classification"
    EXTRACTION = "extraction"
    SUMMARIZATION = "summarization"
    TRANSLATION = "translation"
    CUSTOM = "custom"

class PromptStatus(str, Enum):
    """Prompt状态"""
    DRAFT = "draft"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"

@dataclass
class PromptParameter:
    """Prompt参数"""
    name: str
    type: str  # string, integer, float, boolean, choice
    required: bool = False
    default: Optional[Any] = None
    description: str = ""
    options: List[str] = field(default_factory=list)  # for choice type
    min_value: Optional[float] = None
    max_value: Optional[float] = None

@dataclass
class PromptTemplate:
    """Prompt模板"""
    template_id: str
    name: str
    description: str
    prompt_type: PromptType
    template: str  # Jinja2 template
    parameters: List[PromptParameter] = field(default_factory=list)
    examples: List[Dict] = field(default_factory=list)
    created_by: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class PromptVersion:
    """Prompt版本"""
    version_id: str
    prompt_id: str
    version: int
    template: str
    parameters: List[PromptParameter] = field(default_factory=list)
    changelog: str = ""
    created_by: str
    created_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class Prompt:
    """Prompt实例"""
    prompt_id: str
    name: str
    description: str
    template_id: str
    prompt_type: PromptType
    status: PromptStatus
    tags: List[str] = field(default_factory=list)
    versions: List[PromptVersion] = field(default_factory=list)
    current_version: int = 1
    metrics: Dict = field(default_factory=dict)  # usage, score, etc.
    created_by: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class PromptTestResult:
    """Prompt测试结果"""
    result_id: str
    prompt_id: str
    version: int
    test_input: Dict
    test_output: str
    expected_output: Optional[str] = None
    score: Optional[float] = None
    latency_ms: float = 0.0
    token_count: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)

class PromptManager:
    """Prompt管理器"""
    
    def __init__(self):
        self.templates: Dict[str, PromptTemplate] = {}
        self.prompts: Dict[str, Prompt] = {}
        self.test_results: Dict[str, List[PromptTestResult]] = {}
        
        # 初始化内置模板
        self._init_builtin_templates()
    
    def _init_builtin_templates(self):
        """初始化内置Prompt模板"""
        
        # 通用聊天模板
        self.create_template(
            name="通用聊天",
            description="通用的对话式Prompt模板",
            prompt_type=PromptType.CHAT,
            template="""You are a helpful AI assistant.
Your name is {{ name | default: "AI Assistant" }}.
Your personality is {{ personality | default: "friendly and professional" }}.

Current conversation:
{% for message in conversation %}
{{ message.role }}: {{ message.content }}
{% endfor %}

{% if context %}
Relevant context:
{{ context }}
{% endif %}

Please respond to the user's last message in {{ tone | default: "a helpful and professional" }} tone.""",
            parameters=[
                PromptParameter(name="name", type="string", description="AI助手名称"),
                PromptParameter(name="personality", type="string", default="friendly and professional", description="助手性格"),
                PromptParameter(name="tone", type="string", default="helpful and professional", description="回复语气"),
                PromptParameter(name="conversation", type="list", description="对话历史"),
                PromptParameter(name="context", type="string", description="上下文信息"),
            ],
            examples=[
                {"name": "Assistant", "personality": "friendly", "conversation": []},
            ],
            created_by="system"
        )
        
        # 摘要生成模板
        self.create_template(
            name="文本摘要",
            description="将长文本压缩为简洁摘要",
            prompt_type=PromptType.SUMMARIZATION,
            template="""Please summarize the following text in {{ style | default: "concise" }} style.

Target length: {{ max_length | default: 100 }} words

Text to summarize:
{{ text }}

{% if key_points %}
Make sure to include these key points:
{% for point in key_points %}
- {{ point }}
{% endfor %}
{% endif %}

Summary:""",
            parameters=[
                PromptParameter(name="style", type="choice", options=["concise", "detailed", "bullet"], default="concise", description="摘要风格"),
                PromptParameter(name="max_length", type="integer", default=100, min_value=50, max_value=500, description="目标长度(词)"),
                PromptParameter(name="text", type="string", required=True, description="待摘要文本"),
                PromptParameter(name="key_points", type="list", description="必须包含的关键点"),
            ],
            examples=[],
            created_by="system"
        )
        
        # 分类模板
        self.create_template(
            name="文本分类",
            description="将文本分类到预定义类别",
            prompt_type=PromptType.CLASSIFICATION,
            template="""Classify the following text into one of these categories:
{% for category in categories %}
- {{ category }}
{% endfor %}

Text to classify:
{{ text }}

{% if include_reasoning %}
Please provide your reasoning before giving the answer.
{% endif %}

Category:""",
            parameters=[
                PromptParameter(name="categories", type="list", required=True, description="分类类别列表"),
                PromptParameter(name="text", type="string", required=True, description="待分类文本"),
                PromptParameter(name="include_reasoning", type="boolean", default=False, description="是否包含推理过程"),
            ],
            examples=[],
            created_by="system"
        )
        
        # RAG问答模板
        self.create_template(
            name="RAG问答",
            description="基于检索增强的问答系统",
            prompt_type=PromptType.RAG,
            template="""You are a helpful AI assistant. Use the following context to answer the user's question.
If the answer cannot be found in the context, say "I don't have enough information to answer this question."

Context:
{% for chunk in context_chunks %}
---
{{ chunk }}
---
{% endfor %}

Question: {{ question }}

{% if system_prompt %}
System instruction: {{ system_prompt }}
{% endif %}

Answer based on the context. If you need to say you don't know, please do so.""",
            parameters=[
                PromptParameter(name="context_chunks", type="list", required=True, description="检索到的上下文块"),
                PromptParameter(name="question", type="string", required=True, description="用户问题"),
                PromptParameter(name="system_prompt", type="string", description="系统指令"),
            ],
            examples=[],
            created_by="system"
        )
        
        # 数据提取模板
        self.create_template(
            name="数据提取",
            description="从文本中提取结构化数据",
            prompt_type=PromptType.EXTRACTION,
            template="""Extract the following fields from the given text.
If a field is not found, set it to null.

Fields to extract:
{% for field in fields %}
- {{ field.name }}: {{ field.description }} ({{ field.type }})
{% endfor %}

Text:
{{ text }}

{% if format == "json" %}
Provide the output in JSON format:
```json
{
  "extracted_field": "value"
}
```
{% elif format == "yaml" %}
Provide the output in YAML format:
```yaml
extracted_field: value
```
{% else %}
Provide the output in the specified format.
{% endif %}""",
            parameters=[
                PromptParameter(name="fields", type="list", required=True, description="要提取的字段列表"),
                PromptParameter(name="text", type="string", required=True, description="源文本"),
                PromptParameter(name="format", type="choice", options=["json", "yaml", "xml", "csv"], default="json", description="输出格式"),
            ],
            examples=[],
            created_by="system"
        )
        
        # Agent模板
        self.create_template(
            name="Agent指令",
            description="通用的Agent系统Prompt",
            prompt_type=PromptType.AGENT,
            template="""# Role
You are {{ role | default: "a helpful AI assistant" }}.

# Goal
{{ goal }}

# Capabilities
{% for capability in capabilities %}
- {{ capability }}
{% endfor %}

# Constraints
{% for constraint in constraints %}
- {{ constraint }}
{% endfor %}

# Tools Available
{% for tool in tools %}
- {{ tool.name }}: {{ tool.description }}
  Parameters: {{ tool.parameters | to_json }}
{% endfor %}

# Conversation
{% for message in conversation %}
{{ message.role }}: {{ message.content }}
{% endfor %}

# Your Response
Think step by step and then provide your response. When using tools, use the tool_calls format:
```json
{
  "tool": "tool_name",
  "parameters": {
    "param1": "value1"
  }
}
```""",
            parameters=[
                PromptParameter(name="role", type="string", default="a helpful AI assistant", description="Agent角色"),
                PromptParameter(name="goal", type="string", required=True, description="Agent目标"),
                PromptParameter(name="capabilities", type="list", description="Agent能力"),
                PromptParameter(name="constraints", type="list", description="约束条件"),
                PromptParameter(name="tools", type="list", description="可用工具"),
                PromptParameter(name="conversation", type="list", description="对话历史"),
            ],
            examples=[],
            created_by="system"
        )
    
    # 模板管理
    def create_template(
        self,
        name: str,
        description: str,
        prompt_type: PromptType,
        template: str,
        parameters: Optional[List[PromptParameter]] = None,
        examples: Optional[List[Dict]] = None,
        created_by: str = "user"
    ) -> PromptTemplate:
        """创建Prompt模板"""
        template_obj = PromptTemplate(
            template_id=str(uuid4()),
            name=name,
            description=description,
            prompt_type=prompt_type,
            template=template,
            parameters=parameters or [],
            examples=examples or [],
            created_by=created_by
        )
        
        self.templates[template_obj.template_id] = template_obj
        return template_obj
    
    def get_template(self, template_id: str) -> Optional[PromptTemplate]:
        """获取模板"""
        return self.templates.get(template_id)
    
    def list_templates(
        self,
        prompt_type: Optional[PromptType] = None,
        created_by: Optional[str] = None
    ) -> List[PromptTemplate]:
        """列出模板"""
        templates = list(self.templates.values())
        
        if prompt_type:
            templates = [t for t in templates if t.prompt_type == prompt_type]
        if created_by:
            templates = [t for t in templates if t.created_by == created_by]
        
        return templates
    
    def update_template(
        self,
        template_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        template: Optional[str] = None,
        parameters: Optional[List[PromptParameter]] = None
    ) -> bool:
        """更新模板"""
        template_obj = self.templates.get(template_id)
        if not template_obj:
            return False
        
        if name:
            template_obj.name = name
        if description:
            template_obj.description = description
        if template:
            template_obj.template = template
        if parameters:
            template_obj.parameters = parameters
        
        template_obj.updated_at = datetime.utcnow()
        return True
    
    def delete_template(self, template_id: str) -> bool:
        """删除模板"""
        if template_id in self.templates:
            del self.templates[template_id]
            return True
        return False
    
    # Prompt管理
    def create_prompt(
        self,
        name: str,
        description: str,
        template_id: str,
        prompt_type: PromptType,
        tags: Optional[List[str]] = None,
        created_by: str = "user"
    ) -> Prompt:
        """创建Prompt"""
        template = self.templates.get(template_id)
        if not template:
            raise ValueError(f"Template {template_id} not found")
        
        prompt = Prompt(
            prompt_id=str(uuid4()),
            name=name,
            description=description,
            template_id=template_id,
            prompt_type=prompt_type,
            status=PromptStatus.DRAFT,
            tags=tags or [],
            created_by=created_by
        )
        
        # 创建初始版本
        version = PromptVersion(
            version_id=str(uuid4()),
            prompt_id=prompt.prompt_id,
            version=1,
            template=template.template,
            parameters=template.parameters,
            changelog="Initial version",
            created_by=created_by
        )
        prompt.versions = [version]
        
        self.prompts[prompt.prompt_id] = prompt
        self.test_results[prompt.prompt_id] = []
        
        return prompt
    
    def get_prompt(self, prompt_id: str) -> Optional[Prompt]:
        """获取Prompt"""
        return self.prompts.get(prompt_id)
    
    def list_prompts(
        self,
        status: Optional[PromptStatus] = None,
        prompt_type: Optional[PromptType] = None,
        tags: Optional[List[str]] = None
    ) -> List[Prompt]:
        """列出Prompts"""
        prompts = list(self.prompts.values())
        
        if status:
            prompts = [p for p in prompts if p.status == status]
        if prompt_type:
            prompts = [p for p in prompts if p.prompt_type == prompt_type]
        if tags:
            prompts = [p for p in prompts if any(t in p.tags for t in tags)]
        
        return prompts
    
    def update_prompt(
        self,
        prompt_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        status: Optional[PromptStatus] = None,
        tags: Optional[List[str]] = None
    ) -> bool:
        """更新Prompt"""
        prompt = self.prompts.get(prompt_id)
        if not prompt:
            return False
        
        if name:
            prompt.name = name
        if description:
            prompt.description = description
        if status:
            prompt.status = status
        if tags:
            prompt.tags = tags
        
        prompt.updated_at = datetime.utcnow()
        return True
    
    def create_version(
        self,
        prompt_id: str,
        template: str,
        parameters: Optional[List[PromptParameter]] = None,
        changelog: str = "",
        created_by: str = "user"
    ) -> Optional[PromptVersion]:
        """创建新版本"""
        prompt = self.prompts.get(prompt_id)
        if not prompt:
            return None
        
        version = PromptVersion(
            version_id=str(uuid4()),
            prompt_id=prompt_id,
            version=prompt.current_version + 1,
            template=template,
            parameters=parameters or [],
            changelog=changelog,
            created_by=created_by
        )
        
        prompt.versions.append(version)
        prompt.current_version = version.version
        prompt.updated_at = datetime.utcnow()
        
        return version
    
    def get_current_template(self, prompt_id: str) -> Optional[str]:
        """获取当前版本模板"""
        prompt = self.prompts.get(prompt_id)
        if not prompt or not prompt.versions:
            return None
        
        current = [v for v in prompt.versions if v.version == prompt.current_version]
        return current[0].template if current else None
    
    def delete_prompt(self, prompt_id: str) -> bool:
        """删除Prompt"""
        if prompt_id in self.prompts:
            del self.prompts[prompt_id]
            if prompt_id in self.test_results:
                del self.test_results[prompt_id]
            return True
        return False
    
    # 测试管理
    def add_test_result(
        self,
        prompt_id: str,
        test_input: Dict,
        test_output: str,
        expected_output: Optional[str] = None,
        latency_ms: float = 0.0,
        token_count: int = 0
    ) -> PromptTestResult:
        """添加测试结果"""
        result = PromptTestResult(
            result_id=str(uuid4()),
            prompt_id=prompt_id,
            version=self.prompts[prompt_id].current_version,
            test_input=test_input,
            test_output=test_output,
            expected_output=expected_output,
            latency_ms=latency_ms,
            token_count=token_count
        )
        
        self.test_results[prompt_id].append(result)
        
        # 更新metrics
        prompt = self.prompts[prompt_id]
        prompt.metrics["test_count"] = len(self.test_results[prompt_id])
        
        return result
    
    def evaluate_test_result(
        self,
        result_id: str,
        prompt_id: str,
        score: float
    ) -> bool:
        """评估测试结果"""
        results = self.test_results.get(prompt_id, [])
        for result in results:
            if result.result_id == result_id:
                result.score = score
                
                # 更新metrics
                prompt = self.prompts.get(prompt_id)
                if prompt:
                    scores = [r.score for r in self.test_results[prompt_id] if r.score is not None]
                    if scores:
                        prompt.metrics["avg_score"] = sum(scores) / len(scores)
                        prompt.metrics["min_score"] = min(scores)
                        prompt.metrics["max_score"] = max(scores)
                return True
        return False
    
    def get_test_results(self, prompt_id: str) -> List[PromptTestResult]:
        """获取测试结果"""
        return self.test_results.get(prompt_id, [])
    
    def get_template_summary(self) -> Dict:
        """获取模板统计"""
        by_type = {}
        for template in self.templates.values():
            t = template.prompt_type.value
            by_type[t] = by_type.get(t, 0) + 1
        
        return {
            "total": len(self.templates),
            "by_type": by_type,
            "builtin": len([t for t in self.templates.values() if t.created_by == "system"])
        }
    
    def get_prompt_summary(self) -> Dict:
        """获取Prompt统计"""
        by_status = {}
        by_type = {}
        for prompt in self.prompts.values():
            s = prompt.status.value
            by_status[s] = by_status.get(s, 0) + 1
            t = prompt.prompt_type.value
            by_type[t] = by_type.get(t, 0) + 1
        
        return {
            "total": len(self.prompts),
            "by_status": by_status,
            "by_type": by_type
        }

# PromptManager实例
prompt_manager = PromptManager()
