#!/bin/bash

# AI Platform 启动脚本

set -e

echo "========================================"
echo "  AI Platform - 启动脚本"
echo "========================================"
echo ""

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 检查Python
check_python() {
    if command -v python3 &> /dev/null; then
        echo -e "${GREEN}✓${NC} Python3 已安装"
    else
        echo -e "${RED}✗${NC} Python3 未安装"
        exit 1
    fi
}

# 检查Node.js
check_node() {
    if command -v node &> /dev/null; then
        echo -e "${GREEN}✓${NC} Node.js 已安装"
    else
        echo -e "${YELLOW}!${NC} Node.js 未安装，前端将无法启动"
    fi
}

# 检查Docker
check_docker() {
    if command -v docker &> /dev/null; then
        echo -e "${GREEN}✓${NC} Docker 已安装"
        return 0
    else
        echo -e "${YELLOW}!${NC} Docker 未安装"
        return 1
    fi
}

# 安装依赖
install_deps() {
    echo ""
    echo "========================================"
    echo "  安装依赖"
    echo "========================================"
    
    # 后端依赖
    echo "安装后端依赖..."
    cd backend
    python3 -m pip install --break-system-packages -q -r requirements.txt 2>/dev/null || true
    cd ..
    
    # 前端依赖
    if [ -d "frontend" ]; then
        echo "安装前端依赖..."
        cd frontend
        npm install 2>/dev/null || true
        cd ..
    fi
    
    echo -e "${GREEN}✓${NC} 依赖安装完成"
}

# 启动后端
start_backend() {
    echo ""
    echo "========================================"
    echo "  启动后端服务"
    echo "========================================"
    
    cd backend
    nohup python3 main.py > ../logs/backend.log 2>&1 &
    BACKEND_PID=$!
    cd ..
    
    sleep 3
    
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo -e "${GREEN}✓${NC} 后端已启动 (PID: $BACKEND_PID)"
        echo "  - API文档: http://localhost:8000/docs"
    else
        echo -e "${RED}✗${NC} 后端启动失败，查看日志: logs/backend.log"
    fi
}

# 启动前端
start_frontend() {
    if ! command -v node &> /dev/null; then
        echo ""
        echo -e "${YELLOW}!${NC} 跳过前端启动（Node.js未安装）"
        return
    fi
    
    echo ""
    echo "========================================"
    echo "  启动前端服务"
    echo "========================================"
    
    cd frontend
    nohup npm run dev > ../logs/frontend.log 2>&1 &
    FRONTEND_PID=$!
    cd ..
    
    sleep 5
    
    if curl -s http://localhost:3000 > /dev/null 2>&1; then
        echo -e "${GREEN}✓${NC} 前端已启动 (PID: $FRONTEND_PID)"
        echo "  - 访问地址: http://localhost:3000"
    else
        echo -e "${YELLOW}!${NC} 前端启动中，查看日志: logs/frontend.log"
    fi
}

# 启动Docker服务
start_docker() {
    if ! check_docker; then
        echo ""
        echo -e "${YELLOW}!${NC} 跳过Docker服务（Docker未安装）"
        return
    fi
    
    echo ""
    echo "========================================"
    echo "  启动Docker服务"
    echo "========================================"
    
    # 创建日志目录
    mkdir -p logs
    
    # 启动核心服务
    docker-compose up -d postgres redis
    
    echo -e "${GREEN}✓${NC} PostgreSQL 已启动 (端口: 5432)"
    echo -e "${GREEN}✓${NC} Redis 已启动 (端口: 6379)"
}

# 停止服务
stop_all() {
    echo ""
    echo "========================================"
    echo "  停止所有服务"
    echo "========================================"
    
    # 停止前端
    if lsof -ti:3000 > /dev/null 2>&1; then
        kill $(lsof -ti:3000) 2>/dev/null || true
        echo -e "${GREEN}✓${NC} 前端已停止"
    fi
    
    # 停止后端
    if lsof -ti:8000 > /dev/null 2>&1; then
        kill $(lsof -ti:8000) 2>/dev/null || true
        echo -e "${GREEN}✓${NC} 后端已停止"
    fi
    
    # 停止Docker
    if command -v docker &> /dev/null; then
        docker-compose down 2>/dev/null || true
        echo -e "${GREEN}✓${NC} Docker服务已停止"
    fi
}

# 显示状态
show_status() {
    echo ""
    echo "========================================"
    echo "  服务状态"
    echo "========================================"
    
    echo ""
    echo "后端服务:"
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo -e "  ${GREEN}✓${NC} 运行中 (http://localhost:8000)"
    else
        echo -e "  ${RED}✗${NC} 未运行"
    fi
    
    echo ""
    echo "前端服务:"
    if curl -s http://localhost:3000 > /dev/null 2>&1; then
        echo -e "  ${GREEN}✓${NC} 运行中 (http://localhost:3000)"
    else
        echo -e "  ${YELLOW}○${NC} 未运行"
    fi
    
    echo ""
    echo "Docker服务:"
    if command -v docker &> /dev/null; then
        docker-compose ps 2>/dev/null || echo "  无Docker服务"
    else
        echo "  Docker未安装"
    fi
    
    echo ""
    echo "默认账号:"
    echo "  用户名: admin"
    echo "  密码: admin123"
}

# 主函数
main() {
    echo "AI Platform 启动脚本"
    echo ""
    
    # 创建日志目录
    mkdir -p logs
    
    # 检查环境
    check_python
    check_node
    
    # 根据参数执行
    case "${1:-start}" in
        start)
            install_deps
            start_docker
            start_backend
            start_frontend
            show_status
            ;;
        stop)
            stop_all
            ;;
        restart)
            stop_all
            sleep 2
            install_deps
            start_backend
            start_frontend
            show_status
            ;;
        status)
            show_status
            ;;
        deps)
            install_deps
            ;;
        docker)
            start_docker
            ;;
        backend)
            start_backend
            ;;
        frontend)
            start_frontend
            ;;
        help|--help|-h)
            echo "用法: $0 [命令]"
            echo ""
            echo "命令:"
            echo "  start     - 启动所有服务（默认）"
            echo "  stop      - 停止所有服务"
            echo "  restart   - 重启所有服务"
            echo "  status    - 显示服务状态"
            echo "  deps      - 仅安装依赖"
            echo "  docker    - 仅启动Docker服务"
            echo "  backend   - 仅启动后端"
            echo "  frontend  - 仅启动前端"
            echo "  help      - 显示此帮助"
            ;;
        *)
            echo "未知命令: $1"
            echo "使用: $0 help 查看帮助"
            exit 1
            ;;
    esac
}

main "$@"
