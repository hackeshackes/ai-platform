"""
示例 - Usage Examples
自愈系统的使用示例
"""

import asyncio
from datetime import datetime, timedelta


async def basic_health_check_example():
    """基本健康检查示例"""
    from self_healing import run_health_check, get_health_checker
    
    print("=== 基本健康检查示例 ===")
    
    # 执行一次完整的健康检查
    report = await run_health_check()
    
    print(f"总体状态: {report.overall_status.value}")
    print(f"检查时间: {report.timestamp}")
    print(f"服务数量: {report.summary['total_services']}")
    print(f"资源数量: {report.summary['total_resources']}")
    
    # 检查是否有警告
    if report.recommendations:
        print("建议:")
        for rec in report.recommendations:
            print(f"  - {rec}")
    
    return report


async def incident_handling_example():
    """事件处理示例"""
    from self_healing import get_incident_manager, get_health_checker
    from self_healing.health_checker import HealthStatus
    
    print("\n=== 事件处理示例 ===")
    
    manager = get_incident_manager()
    
    # 创建测试事件
    from self_healing.incident_manager import Incident, IncidentSeverity, IncidentCategory, IncidentStatus
    
    test_incident = Incident(
        incident_id="test_001",
        title="Test Incident: High CPU Usage",
        description="CPU usage exceeded 90% threshold",
        severity=IncidentSeverity.P2_HIGH,
        status=IncidentStatus.DETECTED,
        category=IncidentCategory.PERFORMANCE_DEGRADATION,
        created_at=datetime.now(),
        updated_at=datetime.now(),
        detected_by="test"
    )
    
    # 分配事件
    manager.assign_incident("test_001", "on_call_engineer")
    print(f"事件已分配: {test_incident.assignee}")
    
    # 更新状态
    manager.update_incident_status(
        "test_001",
        IncidentStatus.INVESTIGATING,
        user="engineer",
        message="Starting investigation"
    )
    
    # 获取事件
    incident = manager.get_incident("test_001")
    print(f"事件状态: {incident.status.value}")
    
    # 获取活跃事件
    active = manager.get_active_incidents()
    print(f"活跃事件数量: {len(active)}")
    
    # 获取统计
    stats = manager.get_incident_stats()
    print(f"事件统计: {stats}")


async def auto_fix_example():
    """自动修复示例"""
    from self_healing import get_fix_engine
    from self_healing.incident_manager import (
        Incident, IncidentSeverity, IncidentCategory, IncidentStatus
    )
    
    print("\n=== 自动修复示例 ===")
    
    engine = get_fix_engine()
    
    # 创建测试事件
    incident = Incident(
        incident_id="fix_test_001",
        title="Service Down: web_server",
        description="Web server is not responding",
        severity=IncidentSeverity.P1_CRITICAL,
        status=IncidentStatus.DETECTED,
        category=IncidentCategory.SERVICE_DOWN,
        created_at=datetime.now(),
        updated_at=datetime.now(),
        detected_by="health_checker",
        affected_services=["web_server"]
    )
    
    # 触发自动修复
    record = await engine.process_incident(incident)
    
    print(f"修复记录ID: {record.record_id}")
    print(f"修复类型: {record.fix_type.value}")
    print(f"修复状态: {record.status.value}")
    print(f"执行动作数: {len(record.actions)}")
    
    for action in record.actions:
        print(f"  - {action.fix_type.value}: {action.status.value}")
    
    # 获取修复统计
    stats = engine.get_fix_stats()
    print(f"修复成功率: {stats['success_rate']:.1f}%")


async def runbook_example():
    """运维手册示例"""
    from self_healing import get_runbook_automation
    
    print("\n=== 运维手册示例 ===")
    
    automation = get_runbook_automation()
    
    # 获取所有手册
    runbooks = automation.get_all_runbooks()
    print(f"可用手册数量: {len(runbooks)}")
    
    for rb in runbooks:
        print(f"  - {rb.name} ({rb.runbook_id})")
        print(f"    步骤数: {len(rb.steps)}")
    
    # 执行手册
    print("\n执行服务重启手册...")
    execution = await automation.execute_runbook(
        runbook_id='service_restart',
        variables={'service_name': 'nginx'}
    )
    
    print(f"执行ID: {execution.execution_id}")
    print(f"执行状态: {execution.status.value}")
    print(f"已完成步骤: {len(execution.step_executions)}")
    
    # 获取执行历史
    history = automation.get_execution_history()
    print(f"历史执行数: {len(history)}")


async def full_self_healing_example():
    """完整自愈流程示例"""
    from self_healing import get_api
    
    print("\n=== 完整自愈流程示例 ===")
    
    api = get_api()
    
    # 1. 执行健康检查
    print("1. 执行健康检查...")
    health_result = await api.health_check_all()
    print(f"   成功: {health_result.success}")
    if health_result.data:
        print(f"   总体状态: {health_result.data.get('overall_status')}")
    
    # 2. 获取活跃事件
    print("\n2. 获取活跃事件...")
    incidents_result = api.get_active_incidents()
    print(f"   活跃事件: {incidents_result.data.get('count', 0)}")
    
    # 3. 完整诊断
    print("\n3. 执行完整诊断...")
    diagnosis = await api.full_diagnosis()
    print(f"   成功: {diagnosis.success}")
    if diagnosis.data:
        print(f"   检测到事件: {diagnosis.data.get('stats', {}).get('incidents_detected', 0)}")
    
    # 4. 获取仪表盘
    print("\n4. 获取仪表盘数据...")
    dashboard = api.get_dashboard()
    print(f"   成功: {dashboard.success}")


def configuration_example():
    """配置示例"""
    from self_healing.config import Config, get_config, reload_config
    
    print("\n=== 配置示例 ===")
    
    # 获取默认配置
    config = get_config()
    print(f"服务数量: {len(config.services)}")
    print(f"修复策略数量: {len(config.fix_strategies)}")
    
    # 获取特定策略
    strategy = config.get_strategy('service_down')
    if strategy:
        print(f"服务宕机策略: {strategy.strategy}")
        print(f"  自动修复: {strategy.auto_fix}")
        print(f"  最大尝试次数: {strategy.max_attempts}")
    
    # 获取服务配置
    service = config.get_service('web_server')
    if service:
        print(f"Web服务器配置: 健康检查间隔 {service.health_check_interval}s")


def main():
    """主函数 - 运行所有示例"""
    print("=" * 50)
    print("Self Healing System - Usage Examples")
    print("=" * 50)
    
    # 运行所有示例
    asyncio.run(configuration_example())
    asyncio.run(basic_health_check_example())
    asyncio.run(incident_handling_example())
    asyncio.run(auto_fix_example())
    asyncio.run(runbook_example())
    asyncio.run(full_self_healing_example())
    
    print("\n" + "=" * 50)
    print("All examples completed!")
    print("=" * 50)


if __name__ == "__main__":
    main()
