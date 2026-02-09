# AI Platform 开发文档

## 项目结构

```
ai-platform/
├── frontend/         # React前端
│   ├── src/pages/    # 页面组件
│   ├── src/api/      # API客户端
│   └── src/App.tsx   # 主应用
│
├── backend/          # FastAPI后端
│   ├── api/endpoints/ # API端点
│   └── main.py       # 入口
│
└── docs/            # 文档
```

## 开发命令

### 后端

```bash
cd backend
python3 main.py
```

### 前端

```bash
cd frontend
npm install
npm run dev
```

## 代码规范

- Python: PEP 8
- TypeScript: ESLint + Prettier

## API开发

1. 创建endpoint文件
2. 在routes.py注册
3. 添加数据模型

## 前端开发

1. 创建页面组件
2. 在App.tsx添加路由
3. 使用API客户端

## 测试

```bash
# 后端
pytest

# 前端
npm run test
```
