"""
LLM Tracing 端到端测试
"""
import pytest
from backend.llm.tracing import llm_tracer

class TestLLMTracing:
    """LLM Tracing测试"""
    
    def test_start_trace(self):
        """测试开始Trace"""
        trace_id = llm_tracer.start_trace("test-trace")
        assert trace_id is not None
        assert len(trace_id) == 36  # UUID格式
        print("✅ start_trace 正常")
    
    def test_end_trace(self):
        """测试结束Trace"""
        trace_id = llm_tracer.start_trace("test-trace-2")
        # 先创建一个span
        span_id = llm_tracer.start_span(name="test-span", span_type="llm")
        llm_tracer.end_span(span_id)
        
        llm_tracer.end_trace(trace_id)
        trace = llm_tracer.get_trace(trace_id)
        assert trace is not None
        assert trace.metadata.get("total_duration_ms") is not None
        print("✅ end_trace 正常")
    
    def test_span_lifecycle(self):
        """测试Span生命周期"""
        trace_id = llm_tracer.start_trace("test-span")
        
        # 开始span
        span_id = llm_tracer.start_span(
            name="test-span-op",
            span_type="llm",
            inputs={"prompt": "test"}
        )
        assert span_id is not None
        
        # 结束span
        llm_tracer.end_span(span_id, outputs={"response": "ok"})
        
        trace = llm_tracer.get_trace(trace_id)
        assert len(trace.spans) == 1
        assert trace.spans[0].status == "ok"
        print("✅ span_lifecycle 正常")
    
    def test_span_tree(self):
        """测试Span树"""
        trace_id = llm_tracer.start_trace("test-tree")
        
        # 创建父子span
        parent_id = llm_tracer.start_span(name="parent", span_type="chain")
        child_id = llm_tracer.start_span(
            name="child",
            span_type="llm",
            parent_span_id=parent_id
        )
        
        llm_tracer.end_span(child_id)
        llm_tracer.end_span(parent_id)
        
        tree = llm_tracer.get_span_tree(trace_id)
        assert tree is not None
        assert tree["name"] == "test-tree"
        assert len(tree["tree"]) == 1
        print("✅ span_tree 正常")
    
    def test_list_traces(self):
        """测试列出Traces"""
        # 创建几个traces
        for i in range(3):
            llm_tracer.start_trace(f"trace-{i}")
        
        traces = llm_tracer.list_traces(limit=10)
        assert len(traces) >= 3
        print("✅ list_traces 正常")
    
    def test_export_trace(self):
        """测试导出Trace"""
        trace_id = llm_tracer.start_trace("test-export")
        llm_tracer.start_span(name="op", span_type="llm")
        
        exported = llm_tracer.export_trace(trace_id)
        assert exported is not None
        assert "trace_id" in exported
        assert "spans" in exported
        print("✅ export_trace 正常")

class TestLLMEvaluation:
    """LLM Evaluation测试"""
    
    def test_accuracy(self):
        """测试准确率"""
        from backend.llm.evaluation import evaluator
        import asyncio
        
        predictions = ["1", "0", "1", "1", "0"]
        labels = ["1", "0", "1", "0", "0"]
        
        accuracy = asyncio.run(evaluator.accuracy(predictions, labels))
        assert accuracy == 0.8  # 4/5
        print("✅ accuracy 正常")
    
    def test_precision(self):
        """测试精确率"""
        from backend.llm.evaluation import evaluator
        import asyncio
        
        predictions = ["1", "1", "1", "0", "0"]
        labels = ["1", "0", "1", "1", "0"]
        
        precision = asyncio.run(evaluator.precision(predictions, labels))
        assert precision == 2/3  # 2/3
        print("✅ precision 正常")
    
    def test_f1(self):
        """测试F1"""
        from backend.llm.evaluation import evaluator
        import asyncio
        
        predictions = ["1", "1", "0", "0"]
        labels = ["1", "0", "0", "0"]
        
        f1 = asyncio.run(evaluator.f1(predictions, labels))
        assert f1 > 0
        print("✅ f1 正常")
    
    def test_run_evaluation(self):
        """测试评估运行"""
        from backend.llm.evaluation import evaluator
        import asyncio
        
        async def mock_predict(input_data: str) -> str:
            return "1"
        
        run = asyncio.run(evaluator.run_evaluation(
            name="test-evaluation",
            model_versions=["v1"],
            dataset_id="test-ds",
            predict_fn=mock_predict,
            scorers=["accuracy"]
        ))
        
        assert run is not None
        assert run.status == "completed"
        assert "accuracy" in run.results
        print("✅ run_evaluation 正常")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
