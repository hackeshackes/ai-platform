"""
API端点单元测试 v2.0
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch

# 导入应用
from backend.main import app
from backend.core.cache import CacheManager

client = TestClient(app)

class TestHealthEndpoint:
    """健康检查测试"""
    
    def test_health_check(self):
        """测试健康检查"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
    
    def test_health_with_cache(self):
        """测试健康检查(含缓存)"""
        with patch('backend.main.cache') as mock_cache:
            mock_cache.ping.return_value = True
            response = client.get("/health")
            assert response.status_code == 200


class TestAuthEndpoint:
    """认证测试"""
    
    @patch('backend.api.endpoints.auth.verify_password')
    @patch('backend.api.endpoints.auth.create_access_token')
    def test_login_success(self, mock_token, mock_verify):
        """测试成功登录"""
        mock_verify.return_value = True
        mock_token.return_value = "test_token"
        
        response = client.post("/api/v1/auth/token", data={
            "username": "admin",
            "password": "admin123"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
    
    def test_login_failure(self):
        """测试失败登录"""
        response = client.post("/api/v1/auth/token", data={
            "username": "wrong",
            "password": "wrong"
        })
        assert response.status_code == 401


class TestProjectsEndpoint:
    """项目端点测试"""
    
    @patch('backend.api.endpoints.projects.cache_service')
    def test_list_projects(self, mock_cache):
        """测试获取项目列表"""
        mock_cache.get_user_projects_cached.return_value = None
        
        response = client.get("/api/v1/projects")
        
        assert response.status_code == 200
        data = response.json()
        assert "total" in data
        assert "projects" in data
    
    @patch('backend.api.endpoints.auth.get_current_user')
    @patch('backend.api.endpoints.projects.cache_service')
    def test_create_project(self, mock_cache, mock_user):
        """测试创建项目"""
        mock_user.id = 1
        mock_cache.invalidate_user_projects.return_value = None
        
        response = client.post(
            "/api/v1/projects",
            json={"name": "Test Project", "description": "Test"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Test Project"
    
    def test_create_project_validation(self):
        """测试创建项目验证"""
        response = client.post("/api/v1/projects", json={})
        assert response.status_code == 422  # Validation error


class TestTasksEndpoint:
    """任务端点测试"""
    
    def test_list_tasks(self):
        """测试获取任务列表"""
        response = client.get("/api/v1/tasks")
        
        assert response.status_code == 200
        data = response.json()
        assert "total" in data
        assert "tasks" in data
    
    def test_get_task_status(self):
        """测试获取任务状态"""
        response = client.get("/api/v1/tasks/1")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
    
    @patch('backend.api.endpoints.tasks.cache')
    def test_get_gpu_metrics(self, mock_cache):
        """测试获取GPU指标"""
        mock_cache.get.return_value = {"gpu_count": 1, "utilization": 45}
        
        response = client.get("/api/v1/gpu")
        
        assert response.status_code == 200


class TestDatasetsEndpoint:
    """数据集端点测试"""
    
    def test_list_datasets(self):
        """测试获取数据集列表"""
        response = client.get("/api/v1/datasets")
        
        assert response.status_code == 200
        data = response.json()
        assert "total" in data
    
    @patch('backend.api.endpoints.datasets.quality_checker')
    def test_check_quality(self, mock_checker):
        """测试质量检查"""
        mock_checker.return_value = {
            "overall_score": 95,
            "issues": [],
            "recommendations": []
        }
        
        response = client.post(
            "/api/v1/datasets/quality/check",
            json={"dataset_id": 1}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "overall_score" in data


class TestTrainingEndpoint:
    """训练端点测试"""
    
    def test_list_training_models(self):
        """测试获取训练模型列表"""
        response = client.get("/api/v1/training/models")
        
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
    
    def test_list_training_templates(self):
        """测试获取训练模板"""
        response = client.get("/api/v1/training/templates")
        
        assert response.status_code == 200
        data = response.json()
        assert "templates" in data
    
    @patch('backend.api.endpoints.training.cache')
    @patch('backend.api.endpoints.training.run_training_task.delay')
    def test_submit_training(self, mock_task, mock_cache):
        """测试提交训练"""
        mock_cache.set.return_value = None
        
        response = client.post(
            "/api/v1/training/submit",
            json={
                "model_id": "llama2-7b",
                "dataset_id": 1,
                "template_id": "lora"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "task_id" in data or "status" in data


class TestInferenceEndpoint:
    """推理端点测试"""
    
    def test_list_inference_models(self):
        """测试获取推理模型列表"""
        response = client.get("/api/v1/inference/models")
        
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
    
    @patch('backend.api.endpoints.inference.run_inference')
    def test_run_inference(self, mock_inference):
        """测试执行推理"""
        mock_inference.return_value = {
            "output": "这是一个测试回答",
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 50
            }
        }
        
        response = client.post(
            "/api/v1/inference/generate",
            json={
                "model_id": "llama2-7b-chat",
                "prompt": "你好",
                "max_tokens": 100
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "output" in data


class TestSettingsEndpoint:
    """设置端点测试"""
    
    def test_get_system_settings(self):
        """测试获取系统设置"""
        response = client.get("/api/v1/settings/system")
        
        assert response.status_code == 200
        data = response.json()
        assert "site_name" in data
    
    def test_get_storage_settings(self):
        """测试获取存储设置"""
        response = client.client.get("/api/v1/settings/storage")
        
        assert response.status_code == 200
        data = response.json()
        assert "max_dataset_size_gb" in data
