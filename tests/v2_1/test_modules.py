"""
v2.1 模块独立测试 - 不依赖主应用
"""
import pytest
import sys
import os

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# 单独测试各模块
class TestFeatureStoreModule:
    """Feature Store模块测试"""
    
    def test_import_store(self):
        """测试导入FeatureStore"""
        from backend.feature_store.store import FeatureStore, FeatureGroup
        assert FeatureStore is not None
        print("✅ FeatureStore 导入成功")
    
    def test_import_ingestion(self):
        """测试导入Ingestion"""
        from backend.feature_store.ingestion import ingestion_service
        assert ingestion_service is not None
        print("✅ FeatureIngestionService 导入成功")
    
    def test_import_serving(self):
        """测试导入Serving"""
        from backend.feature_store.serving import serving_service
        assert serving_service is not None
        print("✅ FeatureServingService 导入成功")


class TestModelRegistryModule:
    """Model Registry模块测试"""
    
    def test_import_registry(self):
        """测试导入ModelRegistry"""
        from backend.models.registry import ModelRegistry
        assert ModelRegistry is not None
        print("✅ ModelRegistry 导入成功")
    
    def test_registry_classes(self):
        """测试导入数据类"""
        from backend.models.registry import RegisteredModel, ModelVersion
        assert RegisteredModel is not None
        assert ModelVersion is not None
        print("✅ Registry数据类 导入成功")


class TestLineageModule:
    """Model Lineage模块测试"""
    
    def test_import_lineage(self):
        """测试导入LineageGraph"""
        from backend.lineage.graph import LineageGraph
        assert LineageGraph is not None
        print("✅ LineageGraph 导入成功")
    
    def test_lineage_classes(self):
        """测试导入血缘类"""
        from backend.lineage.graph import LineageNode, LineageEdge
        assert LineageNode is not None
        assert LineageEdge is not None
        print("✅ Lineage数据类 导入成功")


class TestQualityModule:
    """数据质量模块测试"""
    
    def test_import_quality(self):
        """测试导入QualityEngine"""
        from backend.quality.engine import QualityEngine
        assert QualityEngine is not None
        print("✅ QualityEngine 导入成功")
    
    def test_quality_classes(self):
        """测试导入质量类"""
        from backend.quality.engine import QualityRule, QualityCheck
        assert QualityRule is not None
        assert QualityCheck is not None
        print("✅ Quality数据类 导入成功")


class TestNotebooksModule:
    """Notebooks模块测试"""
    
    def test_import_notebooks(self):
        """测试导入NotebookManager"""
        from backend.notebooks.manager import NotebookManager
        assert NotebookManager is not None
        print("✅ NotebookManager 导入成功")
    
    def test_notebooks_classes(self):
        """测试导入Notebook类"""
        from backend.notebooks.manager import Notebook
        assert Notebook is not None
        print("✅ Notebook数据类 导入成功")


class TestAPIEndpoints:
    """API端点测试"""
    
    def test_import_feature_store_api(self):
        """测试导入Feature Store API"""
        from backend.api.endpoints.feature_store import api as fs_api
        assert hasattr(fs_api, 'router')
        print("✅ Feature Store API 导入成功")
    
    def test_import_registry_api(self):
        """测试导入Registry API"""
        from backend.api.endpoints import registry
        assert hasattr(registry, 'router')
        print("✅ Model Registry API 导入成功")
    
    def test_import_lineage_api(self):
        """测试导入Lineage API"""
        from backend.api.endpoints import lineage
        assert hasattr(lineage, 'router')
        print("✅ Model Lineage API 导入成功")
    
    def test_import_quality_api(self):
        """测试导入Quality API"""
        from backend.api.endpoints import quality
        assert hasattr(quality, 'router')
        print("✅ Data Quality API 导入成功")
    
    def test_import_notebooks_api(self):
        """测试导入Notebooks API"""
        from backend.api.endpoints import notebooks
        assert hasattr(notebooks, 'router')
        print("✅ Notebooks API 导入成功")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
