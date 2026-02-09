# AI Platform K8s部署配置 v2.2

## 架构

```
┌─────────────────────────────────────────────────────────────┐
│                    Ingress Controller                        │
│                    (nginx-ingress)                          │
└─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
      ┌───────────┐   ┌───────────┐   ┌───────────┐
      │  Backend  │   │   Front   │   │ MLflow    │
      │   Pod     │   │   Pod     │   │   Pod     │
      └───────────┘   └───────────┘   └───────────┘
              │               │               │
              └───────────────┼───────────────┘
                              ▼
              ┌───────────────────────────────┐
              │     PostgreSQL StatefulSet    │
              └───────────────────────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │       Redis Deployment        │
              └───────────────────────────────┘
```

## 快速部署

```bash
# 1. 创建namespace
kubectl create namespace ai-platform

# 2. 应用配置
kubectl apply -f k8s/ -n ai-platform

# 3. 查看状态
kubectl get pods -n ai-platform
kubectl get svc -n ai-platform
```

## 文件结构

```
k8s/
├── namespace.yaml           # Namespace
├── backend/
│   ├── deployment.yaml      # 后端部署
│   ├── service.yaml         # 后端服务
│   └── hpa.yaml             # 自动扩缩容
├── frontend/
│   ├── deployment.yaml      # 前端部署
│   └── service.yaml         # 前端服务
├── postgres/
│   ├── statefulset.yaml     # PostgreSQL
│   └── pvc.yaml             # 持久化存储
├── redis/
│   └── deployment.yaml      # Redis
├── mlflow/
│   └── deployment.yaml      # MLflow
├── ingress/
│   └── ingress.yaml         # 入口配置
└── configmap.yaml           # 配置
```

## 配置说明

### Backend Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ai-platform-backend
  namespace: ai-platform
spec:
  replicas: 2
  selector:
    matchLabels:
      app: ai-platform-backend
  template:
    spec:
      containers:
      - name: backend
        image: ai-platform-backend:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            configMapKeyRef:
              name: ai-platform-config
              key: database_url
        - name: REDIS_URL
          valueFrom:
            configMapKeyRef:
              name: ai-platform-config
              key: redis_url
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
```

### Ingress

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ai-platform-ingress
  namespace: ai-platform
spec:
  ingressClassName: nginx
  rules:
  - host: ai-platform.local
    http:
      paths:
      - path: /api
        pathType: Prefix
        backend:
          service:
            name: ai-platform-backend
            port:
              number: 80
      - path: /
        pathType: Prefix
        backend:
          service:
            name: ai-platform-frontend
            port:
              number: 80
```

## GPU支持

```yaml
# backend-gpu/deployment.yaml
spec:
  template:
    spec:
      containers:
      - name: backend
        resources:
          limits:
            nvidia.com/gpu: 1
```

## 监控

```bash
# 查看日志
kubectl logs -f deployment/ai-platform-backend -n ai-platform

# 查看资源
kubectl top pods -n ai-platform

# 重启
kubectl rollout restart deployment/ai-platform-backend -n ai-platform
```

## 更新流程

```bash
# 1. 构建新镜像
docker build -t ai-platform-backend:v2.2 ./backend

# 2. 推送镜像
docker tag ai-platform-backend:v2.2 registry.example.com/ai-platform-backend:v2.2
docker push registry.example.com/ai-platform-backend:v2.2

# 3. 更新部署
kubectl set image deployment/ai-platform-backend backend=registry.example.com/ai-platform-backend:v2.2 -n ai-platform

# 4. 验证
kubectl rollout status deployment/ai-platform-backend -n ai-platform
```
