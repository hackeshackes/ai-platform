"""
深空探测系统示例 - Deep Space System Examples
=============================================

使用示例和演示代码

作者: AI Platform Team
版本: 1.0.0
"""

from deep_space.navigation import DeepSpaceNavigator, RouteMethod
from deep_space.planet_explorer import PlanetExplorer
from deep_space.seti_integration import SETIAnalyzer
from deep_space.communication import DeepSpaceCommunicator


def example_navigation():
    """导航系统示例"""
    print("=" * 60)
    print("深空导航系统示例")
    print("=" * 60)
    
    # 创建导航器
    navigator = DeepSpaceNavigator()
    
    # 规划地球到火星的航线
    print("\n1. 规划地球到火星的航线:")
    route = navigator.plan_route(
        origin="earth",
        destination="mars",
        method="optimal"
    )
    print(f"   出发: {route.origin}")
    print(f"   目标: {route.destination}")
    print(f"   方法: {route.method.value}")
    print(f"   总距离: {route.total_distance:.2f} AU")
    print(f"   预计时间: {route.total_time:.1f} 天")
    print(f"   所需燃料: {route.fuel_required:.0f} kg")
    print(f"   成功率: {route.estimated_success_rate*100:.1f}%")
    
    # 显示机动计划
    print("\n2. 机动计划:")
    for i, maneuver in enumerate(route.maneuvers, 1):
        print(f"   步骤 {i}: {maneuver['name']}")
        print(f"     时间: {maneuver['time']:.1f} 天")
        print(f"     Δv: {maneuver['delta_v']:.2f} km/s")
        print(f"     持续时间: {maneuver['duration']} 秒")
    
    # 规划木星航线
    print("\n3. 规划地球到木星的航线:")
    jupiter_route = navigator.plan_route(
        origin="earth",
        destination="jupiter",
        method="fastest"
    )
    print(f"   预计时间: {jupiter_route.total_time:.1f} 天")
    print(f"   所需燃料: {jupiter_route.fuel_required:.0f} kg")
    
    # 显示导航状态
    print("\n4. 导航系统状态:")
    status = navigator.getNavigationStatus()
    print(f"   已知障碍物数量: {status['known_obstacles']}")
    print(f"   系统就绪: {status['system_ready']}")


def example_planet_exploration():
    """行星探测示例"""
    print("\n" + "=" * 60)
    print("行星探测系统示例")
    print("=" * 60)
    
    # 创建火星探测器
    print("\n1. 初始化火星探测器:")
    explorer = PlanetExplorer("mars")
    print(f"   目标行星: mars")
    
    # 表面分析
    print("\n2. 执行表面地形分析:")
    terrain_result = explorer.analyze_surface()
    print(f"   分析时间: {terrain_result['analysis_time']}")
    print(f"   发现地形类型: {terrain_result['terrain_types_found']}")
    print(f"   检测到危险: {terrain_result['hazard_count']} 个")
    
    # 资源扫描
    print("\n3. 扫描资源:")
    resource_result = explorer.scan_resources()
    print(f"   发现资源类型: {resource_result['resource_types']}")
    print(f"   发现矿床: {resource_result['deposit_count']} 个")
    print(f"   估计总价值: ${resource_result['total_estimated_value']:,.0f}")
    
    # 着陆点选择
    print("\n4. 选择最佳着陆点:")
    site_result = explorer.select_site()
    if site_result['recommended_site']:
        site = site_result['recommended_site']
        print(f"   推荐着陆点: {site['name']}")
        print(f"   位置: ({site['latitude']:.2f}°, {site['longitude']:.2f}°)")
        print(f"   海拔: {site['elevation']:.0f} m")
        print(f"   质量等级: {site['quality']}")
        print(f"   安全分数: {site['safety_score']:.2f}")
        print(f"   推荐进场: {site['recommended_approach']}")
    
    # 采集样本
    print("\n5. 采集样本:")
    if site['recommended_site']:
        sample_result = explorer.collect_samples(
            location=(site['latitude'], site['longitude'])
        )
    else:
        sample_result = explorer.collect_samples((0.0, 0.0))
    print(f"   采集样本数: {sample_result['total_samples']}")
    print(f"   总质量: {sample_result['report']['total_mass']:.1f} g")
    print(f"   保存完好: {sample_result['report']['preserved_samples']} 个")
    
    # 完整探索
    print("\n6. 执行完整探索任务:")
    full_result = explorer.full_exploration()
    print(f"   任务状态: {full_result['mission_summary']['status']}")
    print(f"   下一步行动: {full_result['next_recommended_action']}")


def example_seti_integration():
    """SETI集成示例"""
    print("\n" + "=" * 60)
    print("SETI集成系统示例")
    print("=" * 60)
    
    # 创建SETI分析器
    print("\n1. 初始化SETI分析器:")
    seti = SETIAnalyzer()
    print("   SETI系统就绪")
    
    # 执行扫描
    print("\n2. 执行深空信号扫描:")
    scan_result = seti.scan()
    summary = scan_result['scan_summary']
    print(f"   扫描信号数: {summary['signals_scanned']}")
    print(f"   处理速率: {summary['processing_rate']} 信号/秒")
    print(f"   检测到异常: {summary['anomalies_detected']}")
    print(f"   发现模式: {summary['patterns_found']}")
    print(f"   评估文明: {summary['civilizations_assessed']}")
    
    # 信号分析详情
    print("\n3. 信号分析详情:")
    analysis = scan_result['signal_analysis']
    print(f"   总处理信号: {analysis['total_processed']}")
    print(f"   人工信号: {analysis['artificial_signals']}")
    print(f"   自然信号: {analysis['natural_signals']}")
    
    # 异常检测
    print("\n4. 异常检测结果:")
    anomaly_report = scan_result['anomaly_report']
    print(f"   总异常数: {anomaly_report['total_anomalies']}")
    if anomaly_report['anomalies']:
        for anomaly in anomaly_report['anomalies'][:3]:
            print(f"   - {anomaly['anomaly_type']}: {anomaly['description'][:50]}...")
    
    # 文明评估
    print("\n5. 外星文明评估:")
    civ_assessment = scan_result['civilization_assessment']
    threat = civ_assessment['threat_summary']
    print(f"   评估文明数: {threat['total_civilizations']}")
    print(f"   最高威胁: {threat['highest_threat']}")
    print(f"   建议: {threat['recommendation']}")
    
    # 系统状态
    print("\n6. SETI系统状态:")
    status = seti.get_system_status()
    print(f"   系统状态: {status['status']}")
    print(f"   处理速率: {status['processing_rate']} 信号/秒")


def example_communication():
    """通信系统示例"""
    print("\n" + "=" * 60)
    print("深空通信系统示例")
    print("=" * 60)
    
    # 创建通信器
    print("\n1. 初始化深空通信器:")
    comm = DeepSpaceCommunicator()
    print("   通信系统就绪")
    
    # 发送消息
    print("\n2. 发送测试消息到火星:")
    result = comm.send_message(
        content="Hello from Earth! This is a test transmission.",
        destination="mars",
        priority=4
    )
    print(f"   消息ID: {result['message_id']}")
    print(f"   目标: {result['destination']}")
    print(f"   通道: {result['channel']}")
    print(f"   单程延迟: {result['delay_info']['one_way_delay_s']:.1f} 秒")
    print(f"   往返延迟: {result['delay_info']['round_trip_delay_s']:.1f} 秒")
    print(f"   传输状态: {result['transmission']['status']}")
    print(f"   成功率: {result['transmission']['success_rate']*100:.1f}%")
    
    # 配置通信链路
    print("\n3. 配置地球-木星通信链路:")
    link_config = comm.configure_link(
        target_distance=778e9,  # 778 million km
        required_data_rate=1e6  # 1 Mbps
    )
    print(f"   建议配置:")
    print(f"     频率: {link_config['recommended_configuration']['frequency']}")
    print(f"     带宽: {link_config['recommended_configuration']['bandwidth_hz']/1e6:.1f} MHz")
    print(f"     调制: {link_config['recommended_configuration']['modulation']}")
    print(f"     编码: {link_config['recommended_configuration']['encoding']}")
    print(f"   链路裕度: {link_config['link_margin']/1e6:.1f} Mbps")
    
    # 通信状态
    print("\n4. 通信系统状态:")
    status = comm.get_communication_status()
    print(f"   系统状态: {status['status']}")
    print(f"   活跃通道: {status['active_channels']}")
    print(f"   消息队列: {status['message_queue_size']} 条")


def run_all_examples():
    """运行所有示例"""
    print("\n")
    print("#" * 60)
    print("#  深空探测系统 v1.0.0 - 完整示例演示")
    print("#" * 60)
    print()
    
    # 运行各个示例
    example_navigation()
    example_planet_exploration()
    example_seti_integration()
    example_communication()
    
    print("\n" + "#" * 60)
    print("#  所有示例运行完成")
    print("#" * 60)
    print()


if __name__ == '__main__':
    run_all_examples()
