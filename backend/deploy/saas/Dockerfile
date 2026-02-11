FROM python:3.11-slim

LABEL maintainer="AI Platform Team"
LABEL description="AI Platform SaaS Deployment System"

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    vim \
    && rm -rf /var/lib/apt/lists/*

# 安装Python依赖
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY *.py .
COPY config.yaml .

# 创建日志目录
RUN mkdir -p /var/log/ai-platform \
    && mkdir -p /var/run \
    && chmod -R 777 /var/log/ai-platform

# 设置环境变量
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    LOG_LEVEL=INFO

# 暴露端口
EXPOSE 8080

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# 启动命令
CMD ["python", "api.py"]
