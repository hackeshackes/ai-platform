#!/usr/bin/env python3
"""
Gateway API Test Script

测试网关API的所有端点。
"""
import sys
sys.path.insert(0, '.')

from gateway.router import router, Route
from gateway.ratelimit import limiter, RateLimitConfig, RateLimitAlgorithm
from gateway.quota import quota_manager, QuotaConfig, QuotaPeriod, QuotaRule
import time

def test_router():
    """测试路由器"""
    print("=== Testing Router ===")
    
    # 添加路由
    route1 = Route(
        path="/api/v1/users/*",
        target_url="http://user-service:8080",
        methods=["GET", "POST"],
        rate_limit=50,
        description="User service"
    )
    route_id = router.add_route(route1)
    print(f"Added route: {route_id}")
    
    route2 = Route(
        path="/api/v1/orders/*",
        target_url="http://order-service:8080",
        methods=["GET", "POST", "PUT", "DELETE"],
        quota_limit=1000,
        description="Order service"
    )
    route_id2 = router.add_route(route2)
    print(f"Added route 2: {route_id2}")
    
    # 列出路由
    routes = router.list_routes()
    print(f"Total routes: {len(routes)}")
    
    # 匹配路由
    matched = router.get_route("/api/v1/users/123", "GET")
    print(f"Matched route: {matched.path if matched else None}")
    
    print("Router tests passed!\n")

def test_ratelimit():
    """测试限流器"""
    print("=== Testing Rate Limiter ===")
    
    # 配置限流
    config = RateLimitConfig(
        requests=10,
        window_seconds=60,
        algorithm=RateLimitAlgorithm.TOKEN_BUCKET
    )
    limiter.create_limiter("user:1", config)
    
    # 消耗请求
    for i in range(12):
        result = limiter.check_rate_limit("user:1")
        print(f"Request {i+1}: allowed={result.allowed}, remaining={result.remaining}")
    
    # 重置
    limiter.reset("user:1")
    result = limiter.check_rate_limit("user:1")
    print(f"After reset: remaining={result.remaining}")
    
    print("Rate limiter tests passed!\n")

def test_quota():
    """测试配额管理器"""
    print("=== Testing Quota Manager ===")
    
    # 添加规则
    rule = QuotaRule(
        name="premium_users",
        key_pattern="premium:*",
        config=QuotaConfig(limit=100, period=QuotaPeriod.DAILY),
        priority=10
    )
    quota_manager.add_rule(rule)
    
    # 检查默认用户
    usage1 = quota_manager.get_usage("regular_user")
    print(f"Regular user: used={usage1.used}, limit={usage1.limit}, remaining={usage1.remaining}")
    
    # 检查匹配规则的用户
    usage2 = quota_manager.get_usage("premium:user1")
    print(f"Premium user: used={usage2.used}, limit={usage2.limit}, remaining={usage2.remaining}")
    
    # 消耗配额
    for i in range(5):
        quota_manager.consume("regular_user")
    
    usage3 = quota_manager.get_usage("regular_user")
    print(f"After consumption: used={usage3.used}, remaining={usage3.remaining}")
    
    # 列出规则
    rules = quota_manager.list_rules()
    print(f"Total quota rules: {len(rules)}")
    
    print("Quota manager tests passed!\n")

def test_middleware():
    """测试中间件"""
    print("=== Testing Middleware ===")
    
    from gateway.middleware import GatewayMiddleware
    
    middleware = GatewayMiddleware()
    print(f"Middleware class: {middleware.__class__.__name__}")
    
    # 测试工厂方法
    rate_limit_mw = middleware.add_rate_limit(default_requests=100)
    print(f"Rate limit middleware factory: {rate_limit_mw}")
    
    quota_mw = middleware.add_quota(default_quota=1000)
    print(f"Quota middleware factory: {quota_mw}")
    
    print("Middleware tests passed!\n")

def main():
    """运行所有测试"""
    print("Starting Gateway API Tests...\n")
    
    try:
        test_router()
        test_ratelimit()
        test_quota()
        test_middleware()
        
        print("=" * 50)
        print("All tests passed successfully!")
        print("=" * 50)
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
