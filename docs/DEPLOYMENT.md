# AI Platform 部署文档

## 环境要求

- Node.js >= 18
- Python >= 3.10
- Git

## 开发环境

### 后端

```bash
cd backend
pip install -r requirements.txt
python3 main.py
# 运行在 http://localhost:8000
```

### 前端

```bash
cd frontend
npm install
npm run dev
# 运行在 http://localhost:3000
```

## Docker部署

```bash
docker-compose up -d
```

## 生产环境

### 前端构建

```bash
cd frontend
npm run build
```

### Nginx配置

```nginx
server {
    listen 80;
    server_name your-domain;
    
    location / {
        root /var/www/ai-platform/dist;
        try_files $uri $uri/ /index.html;
    }
    
    location /api/ {
        proxy_pass http://localhost:8000;
    }
}
```

## 故障排除

- **端口被占用**: 杀掉进程或更换端口
- **CORS错误**: 检查后端CORS配置
- **Token过期**: 重新登录
