/**
 * AI Platform API Client
 */

import axios, { AxiosInstance, AxiosError } from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

const apiClient: AxiosInstance = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: { 'Content-Type': 'application/json' },
});

apiClient.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('access_token');
    if (token) config.headers.Authorization = `Bearer ${token}`;
    return config;
  },
  (error) => Promise.reject(error)
);

apiClient.interceptors.response.use(
  (response) => response.data,
  (error: AxiosError) => {
    const { response } = error;
    if (response?.status === 401) {
      localStorage.removeItem('access_token');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

// 认证API
export const authAPI = {
  login: (data: { username: string; password: string }) =>
    apiClient.post('/api/v1/auth/token', new URLSearchParams({
      username: data.username,
      password: data.password
    }), {
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    }),
  register: (data: any) => apiClient.post('/api/v1/auth/register', data),
  logout: () => apiClient.post('/api/v1/auth/logout'),
  me: () => apiClient.get('/api/v1/auth/me'),
};

// 项目API
export const projectAPI = {
  list: () => apiClient.get('/api/v1/projects'),
  create: (data: any) => apiClient.post('/api/v1/projects', data),
  get: (id: string) => apiClient.get(`/api/v1/projects/${id}`),
  update: (id: string, data: any) => apiClient.put(`/api/v1/projects/${id}`, data),
  delete: (id: string) => apiClient.delete(`/api/v1/projects/${id}`),
};

// 任务API
export const taskAPI = {
  list: (params?: any) => apiClient.get('/api/v1/tasks', { params }),
  create: (data: any) => apiClient.post('/api/v1/tasks', data),
  get: (id: string) => apiClient.get(`/api/v1/tasks/${id}`),
  update: (id: string, data: any) => apiClient.put(`/api/v1/tasks/${id}`, data),
  delete: (id: string) => apiClient.delete(`/api/v1/tasks/${id}`),
  logs: (id: string) => apiClient.get(`/api/v1/tasks/${id}/logs`),
};

// 指标API
export const metricsAPI = {
  loss: (taskId: string) => apiClient.get('/api/v1/metrics/loss', { params: { experiment_id: taskId } }),
  gpu: () => apiClient.get('/api/v1/gpu'),
};

// 数据集API
export const datasetAPI = {
  list: (params?: any) => apiClient.get('/api/v1/datasets', { params }),
  create: (data: any) => apiClient.post('/api/v1/datasets', data),
  get: (id: string) => apiClient.get(`/api/v1/datasets/${id}`),
  delete: (id: string) => apiClient.delete(`/api/v1/datasets/${id}`),
};

// 版本API v1.1
export const versionAPI = {
  list: (params?: any) => apiClient.get('/api/v1/datasets/versions', { params }),
  create: (data: any) => apiClient.post('/api/v1/datasets/versions', data),
  get: (id: string) => apiClient.get(`/api/v1/datasets/versions/${id}`),
  delete: (id: string) => apiClient.delete(`/api/v1/datasets/versions/${id}`),
};

// 质量API v1.1
export const qualityAPI = {
  check: (data: any) => apiClient.post('/api/v1/datasets/quality/check', data),
  checkFile: (formData: FormData) => apiClient.post('/api/v1/datasets/quality/check/file', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  }),
  quick: (datasetId: number) => apiClient.get('/api/v1/datasets/quality/quick', { params: { dataset_id: datasetId } }),
};

// 用户API v1.1
export const usersAPI = {
  list: (params?: any) => apiClient.get('/api/v1/users', { params }),
  create: (data: any) => apiClient.post('/api/v1/users', data),
  get: (id: string) => apiClient.get(`/api/v1/users/${id}`),
  update: (id: string, data: any) => apiClient.put(`/api/v1/users/${id}`, data),
  delete: (id: string) => apiClient.delete(`/api/v1/users/${id}`),
};

// 训练API
export const trainingAPI = {
  models: () => apiClient.get('/api/v1/training/models'),
  datasets: () => apiClient.get('/api/v1/training/datasets'),
  templates: () => apiClient.get('/api/v1/training/templates'),
  submit: (data: any) => apiClient.post('/api/v1/training/submit', data),
  jobs: (status?: string) => apiClient.get('/api/v1/training/jobs', { params: { status } }),
  jobStatus: (jobId: string) => apiClient.get(`/api/v1/training/jobs/${jobId}`),
};

// 导出统一API对象
export const api = {
  auth: authAPI,
  projects: projectAPI,
  tasks: taskAPI,
  metrics: metricsAPI,
  training: trainingAPI,
  datasets: datasetAPI,
  versions: versionAPI,
  quality: qualityAPI,
  users: usersAPI,
};

export default apiClient;
