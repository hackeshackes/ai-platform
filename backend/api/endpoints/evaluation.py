"""
模型评估API端点 v2.2
"""
from fastapi import APIRouter, HTTPException, Depends
from typing import List, Optional, Dict, Any, Callable
from pydantic import BaseModel

from backend.llm.evaluation import evaluator
from backend.core.auth import get_current_user

router = APIRouter()

class RunEvaluationModel(BaseModel):
    name: str
    model_versions: List[str]
    dataset_id: str
    scorers: List[str] = ["accuracy"]

class PredictFnModel(BaseModel):
    data: List[Dict[str, Any]]

@router.post("/runs")
async def run_evaluation(
    request: RunEvaluationModel,
    predict_fn_data: PredictFnModel = None
):
    """
    执行模型评估
    
    v2.2: 模型评估
    """
    async def mock_predict_fn(input_data: str) -> str:
        # 模拟预测函数
        return f"prediction_for_{input_data}"
    
    run = await evaluator.run_evaluation(
        name=request.name,
        model_versions=request.model_versions,
        dataset_id=request.dataset_id,
        predict_fn=mock_predict_fn,
        scorers=request.scorers
    )
    
    return {
        "run_id": run.run_id,
        "name": run.name,
        "status": run.status,
        "results": run.results,
        "created_at": run.created_at.isoformat()
    }

@router.get("/runs")
async def list_evaluation_runs(
    limit: int = 100,
    offset: int = 0
):
    """
    列出评估运行
    
    v2.2: 模型评估
    """
    runs = evaluator.list_runs(limit=limit, offset=offset)
    
    return {
        "total": len(runs),
        "runs": runs
    }

@router.get("/runs/{run_id}")
async def get_evaluation_run(run_id: str):
    """
    获取评估运行详情
    
    v2.2: 模型评估
    """
    run = evaluator.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    
    return {
        "run_id": run.run_id,
        "name": run.name,
        "model_versions": run.model_versions,
        "dataset_id": run.dataset_id,
        "scorers": run.scorers,
        "status": run.status,
        "results": run.results,
        "created_at": run.created_at.isoformat()
    }

@router.post("/metrics/accuracy")
async def calculate_accuracy(
    predictions: List[str],
    labels: List[str]
):
    """
    计算准确率
    
    v2.2: 模型评估
    """
    accuracy = await evaluator.accuracy(predictions, labels)
    return {"metric": "accuracy", "value": accuracy}

@router.post("/metrics/precision")
async def calculate_precision(
    predictions: List[str],
    labels: List[str]
):
    """
    计算精确率
    
    v2.2: 模型评估
    """
    precision = await evaluator.precision(predictions, labels)
    return {"metric": "precision", "value": precision}

@router.post("/metrics/recall")
async def calculate_recall(
    predictions: List[str],
    labels: List[str]
):
    """
    计算召回率
    
    v2.2: 模型评估
    """
    recall = await evaluator.recall(predictions, labels)
    return {"metric": "recall", "value": recall}

@router.post("/metrics/f1")
async def calculate_f1(
    predictions: List[str],
    labels: List[str]
):
    """
    计算F1分数
    
    v2.2: 模型评估
    """
    f1 = await evaluator.f1(predictions, labels)
    return {"metric": "f1", "value": f1}

@router.post("/metrics/bleu")
async def calculate_bleu(
    references: List[List[str]],
    candidates: List[str]
):
    """
    计算BLEU分数
    
    v2.2: 模型评估
    """
    bleu = await evaluator.bleu(references, candidates)
    return {"metric": "bleu", "value": bleu}

@router.post("/metrics/rouge")
async def calculate_rouge(
    references: List[str],
    candidates: List[str]
):
    """
    计算ROUGE分数
    
    v2.2: 模型评估
    """
    rouge = await evaluator.rouge_l(references, candidates)
    return {"metric": "rouge_l", "value": rouge}

@router.post("/llm/correctness")
async def evaluate_correctness(
    prediction: str,
    expectation: str
):
    """
    LLM正确性评估
    
    v2.2: 模型评估
    """
    score = await evaluator.correctness(prediction, expectation)
    return {"metric": "correctness", "value": score}

@router.post("/llm/relevance")
async def evaluate_relevance(
    response: str,
    question: str
):
    """
    LLM相关性评估
    
    v2.2: 模型评估
    """
    score = await evaluator.relevance(response, question)
    return {"metric": "relevance", "value": score}

@router.get("/summary")
async def get_evaluation_summary():
    """
    获取评估概览
    
    v2.2: 模型评估
    """
    runs = list(evaluator.runs.values())
    completed = sum(1 for r in runs if r.status == "completed")
    failed = sum(1 for r in runs if r.status == "failed")
    running = sum(1 for r in runs if r.status == "running")
    
    return {
        "total_runs": len(runs),
        "completed": completed,
        "failed": failed,
        "running": running
    }
