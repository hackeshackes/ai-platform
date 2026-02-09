"""
Judge Builder模块 v2.3
"""
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4
import json

@dataclass
class Judge:
    """评估器"""
    judge_id: str
    name: str
    description: str
    prompt: str
    instructions: str
    judge_type: str  # llm, rule-based
    criteria: List[Dict] = field(default_factory=list)
    created_by: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class JudgeRun:
    """评估运行"""
    run_id: str
    judge_id: str
    traces: List[str] = field(default_factory=list)
    results: List[Dict] = field(default_factory=list)
    status: str = "pending"  # pending, running, completed, failed
    summary: Dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None

@dataclass
class JudgeResult:
    """评估结果"""
    result_id: str
    run_id: str
    trace_id: str
    score: float
    passing: bool
    feedback: str
    criteria_scores: List[Dict] = field(default_factory=list)

class JudgeBuilder:
    """Judge Builder - 评估器构建器"""
    
    def __init__(self):
        self.judges: Dict[str, Judge] = {}
        self.runs: Dict[str, JudgeRun] = {}
        self.results: Dict[str, JudgeResult] = {}
        
        # 内置评估器
        self._init_builtin_judges()
    
    def _init_builtin_judges(self):
        """初始化内置评估器"""
        # 正确性评估
        self.create_judge(
            name="Correctness",
            description="检查回答是否正确",
            prompt="You are an expert evaluator. Evaluate if the predicted answer correctly addresses the question.\n\nQuestion: {question}\nExpected: {expected}\nPredicted: {prediction}\n\nRate the correctness on a scale of 1-5.",
            instructions="Evaluate factual correctness",
            judge_type="llm",
            criteria=[
                {"name": "accuracy", "weight": 1.0}
            ],
            created_by="system"
        )
        
        # 相关性评估
        self.create_judge(
            name="Relevance",
            description="检查回答是否相关",
            prompt="You are an expert evaluator. Evaluate if the predicted answer is relevant to the question.\n\nQuestion: {question}\nAnswer: {answer}\n\nRate relevance on a scale of 1-5.",
            instructions="Evaluate topic relevance",
            judge_type="llm",
            criteria=[
                {"name": "relevance", "weight": 1.0}
            ],
            created_by="system"
        )
        
        # 完整性评估
        self.create_judge(
            name="Completeness",
            description="检查回答是否完整",
            prompt="You are an expert evaluator. Evaluate if the answer is complete and covers all aspects.\n\nQuestion: {question}\nAnswer: {answer}\n\nRate completeness on a scale of 1-5.",
            instructions="Evaluate coverage and completeness",
            judge_type="llm",
            criteria=[
                {"name": "completeness", "weight": 1.0}
            ],
            created_by="system"
        )
        
        # 指南合规
        self.create_judge(
            name="Guidelines",
            description="检查是否遵循指南",
            prompt="You are an expert evaluator. Check if the answer follows the guidelines.\n\nGuidelines: {guidelines}\nAnswer: {answer}\n\nRate compliance on a scale of 1-5.",
            instructions="Evaluate guideline adherence",
            judge_type="llm",
            criteria=[
                {"name": "compliance", "weight": 1.0}
            ],
            created_by="system"
        )
    
    # 评估器管理
    def create_judge(
        self,
        name: str,
        description: str,
        prompt: str,
        instructions: str,
        judge_type: str,
        criteria: Optional[List[Dict]] = None,
        created_by: str = "user"
    ) -> Judge:
        """创建评估器"""
        judge = Judge(
            judge_id=str(uuid4()),
            name=name,
            description=description,
            prompt=prompt,
            instructions=instructions,
            judge_type=judge_type,
            criteria=criteria or [],
            created_by=created_by
        )
        
        self.judges[judge.judge_id] = judge
        return judge
    
    def get_judge(self, judge_id: str) -> Optional[Judge]:
        """获取评估器"""
        return self.judges.get(judge_id)
    
    def list_judges(
        self,
        judge_type: Optional[str] = None,
        created_by: Optional[str] = None
    ) -> List[Judge]:
        """列出评估器"""
        judges = list(self.judges.values())
        
        if judge_type:
            judges = [j for j in judges if j.judge_type == judge_type]
        if created_by:
            judges = [j for j in judges if j.created_by == created_by]
        
        return judges
    
    def update_judge(
        self,
        judge_id: str,
        name: Optional[str] = None,
        prompt: Optional[str] = None,
        instructions: Optional[str] = None,
        criteria: Optional[List[Dict]] = None
    ) -> bool:
        """更新评估器"""
        judge = self.judges.get(judge_id)
        if not judge:
            return False
        
        if name:
            judge.name = name
        if prompt:
            judge.prompt = prompt
        if instructions:
            judge.instructions = instructions
        if criteria:
            judge.criteria = criteria
        
        judge.updated_at = datetime.utcnow()
        return True
    
    def delete_judge(self, judge_id: str) -> bool:
        """删除评估器"""
        if judge_id in self.judges:
            del self.judges[judge_id]
            return True
        return False
    
    # 评估运行
    async def create_run(
        self,
        judge_id: str,
        traces: List[str],
        created_by: str = "user"
    ) -> JudgeRun:
        """创建评估运行"""
        judge = self.judges.get(judge_id)
        if not judge:
            raise ValueError(f"Judge {judge_id} not found")
        
        run = JudgeRun(
            run_id=str(uuid4()),
            judge_id=judge_id,
            traces=traces,
            created_by=created_by
        )
        
        self.runs[run.run_id] = run
        
        # 模拟评估
        await self._run_evaluation(run)
        
        return run
    
    async def _run_evaluation(self, run: JudgeRun):
        """执行评估"""
        run.status = "running"
        
        judge = self.judges.get(run.judge_id)
        if not judge:
            run.status = "failed"
            return
        
        # 模拟评估每个trace
        results = []
        for trace_id in run.traces:
            # 模拟评分
            score = 3.0 + (hash(trace_id) % 20) / 10  # 3.0-5.0
            passing = score >= 3.5
            
            result = JudgeResult(
                result_id=str(uuid4()),
                run_id=run.run_id,
                trace_id=trace_id,
                score=round(score, 2),
                passing=passing,
                feedback=f"Evaluated by {judge.name}",
                criteria_scores=[{
                    "criteria": judge.criteria[0]["name"] if judge.criteria else "default",
                    "score": round(score, 2),
                    "weight": 1.0
                }]
            )
            
            self.results[result.result_id] = result
            results.append({
                "result_id": result.result_id,
                "trace_id": result.trace_id,
                "score": result.score,
                "passing": result.passing
            })
        
        run.results = results
        run.status = "completed"
        run.completed_at = datetime.utcnow()
        
        # 生成摘要
        scores = [r["score"] for r in results]
        run.summary = {
            "total": len(results),
            "passed": sum(1 for r in results if r["passing"]),
            "failed": sum(1 for r in results if not r["passing"]),
            "avg_score": round(sum(scores) / len(scores), 2) if scores else 0,
            "max_score": max(scores) if scores else 0,
            "min_score": min(scores) if scores else 0
        }
    
    def get_run(self, run_id: str) -> Optional[JudgeRun]:
        """获取评估运行"""
        return self.runs.get(run_id)
    
    def list_runs(
        self,
        judge_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100
    ) -> List[JudgeRun]:
        """列出评估运行"""
        runs = list(self.runs.values())
        runs.sort(key=lambda r: r.created_at, reverse=True)
        
        if judge_id:
            runs = [r for r in runs if r.judge_id == judge_id]
        if status:
            runs = [r for r in runs if r.status == status]
        
        return runs[:limit]
    
    def get_run_results(self, run_id: str) -> List[JudgeResult]:
        """获取评估结果"""
        run = self.runs.get(run_id)
        if not run:
            return []
        
        return [
            self.results[rid] 
            for rid in [r["result_id"] for r in run.results]
            if rid in self.results
        ]
    
    # 测试运行
    async def test_judge(
        self,
        judge_id: str,
        trace_data: Dict
    ) -> JudgeResult:
        """测试评估器"""
        judge = self.judges.get(judge_id)
        if not judge:
            raise ValueError(f"Judge {judge_id} not found")
        
        # 模拟评估
        score = 4.0
        result = JudgeResult(
            result_id=str(uuid4()),
            run_id="test",
            trace_id="test",
            score=score,
            passing=score >= 3.5,
            feedback=f"Test result from {judge.name}",
            criteria_scores=[{
                "criteria": judge.criteria[0]["name"] if judge.criteria else "default",
                "score": score,
                "weight": 1.0
            }]
        )
        
        return result
    
    # 模板
    def get_templates(self) -> List[Dict]:
        """获取评估器模板"""
        return [
            {
                "name": "Correctness Template",
                "description": "用于检查回答正确性的模板",
                "prompt": "You are an expert evaluator...",
                "criteria": [{"name": "accuracy", "weight": 1.0}]
            },
            {
                "name": "Relevance Template",
                "description": "用于检查相关性的模板",
                "prompt": "You are an expert evaluator...",
                "criteria": [{"name": "relevance", "weight": 1.0}]
            },
            {
                "name": "Custom Template",
                "description": "自定义评估模板",
                "prompt": "",
                "criteria": []
            }
        ]

# JudgeBuilder实例
judge_builder = JudgeBuilder()
