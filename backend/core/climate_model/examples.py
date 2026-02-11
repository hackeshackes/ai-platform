"""
气候模型使用示例
"""

from climate_model import ClimateModel
from climate_model.api import ClimateModelAPI


def example_basic_usage():
    """基础使用示例"""
    print("=" * 60)
    print("气候模型 - 基础使用示例")
    print("=" * 60)
    
    # 创建模型
    print("\n1. 创建气候模型...")
    model = ClimateModel(resolution="1km")
    print("   ✓ 模型创建成功")
    
    # 设置情景
    print("\n2. 设置气候情景...")
    model.set_scenario("RCP8.5")
    print("   ✓ 情景设置为: RCP8.5 (高排放情景)")
    
    # 运行模拟
    print("\n3. 运行气候模拟 (2020-2100)...")
    result = model.run(start_year=2020, end_year=2100, verbose=False)
    print(f"   ✓ 模拟完成")
    
    # 获取预测
    print("\n4. 获取气候预测...")
    temp_pred = model.get_prediction(variable="temperature", year=2100)
    co2_pred = model.get_prediction(variable="co2", year=2100)
    sea_pred = model.get_prediction(variable="sea_level", year=2100)
    
    print(f"   2100年预测:")
    print(f"   - 全球平均气温: {temp_pred['value']:.2f}°C")
    print(f"   - CO2浓度: {co2_pred['value']:.1f} ppm")
    print(f"   - 海平面变化: {sea_pred['value']:.3f} m")
    
    return result


def example_scenario_comparison():
    """情景对比示例"""
    print("\n" + "=" * 60)
    print("气候模型 - 情景对比示例")
    print("=" * 60)
    
    scenarios = ["RCP2.6", "RCP4.5", "RCP6.0", "RCP8.5"]
    results = {}
    
    for scenario in scenarios:
        print(f"\n正在模拟: {scenario}")
        model = ClimateModel(resolution="1km")
        model.set_scenario(scenario)
        result = model.run(start_year=2020, end_year=2100, verbose=False)
        
        temp_change = result['history']['temperature'][-1] - result['history']['temperature'][0]
        co2_final = result['history']['co2'][-1]
        sea_final = result['history']['sea_level'][-1]
        
        results[scenario] = {
            'temp_change': temp_change,
            'co2': co2_final,
            'sea_level': sea_final
        }
        
        print(f"   温度变化: {temp_change:+.2f}°C")
        print(f"   CO2浓度: {co2_final:.1f} ppm")
        print(f"   海平面变化: {sea_final:.3f} m")
    
    print("\n" + "-" * 60)
    print("情景对比总结:")
    print("-" * 60)
    print(f"{'情景':<10} {'温度变化':<15} {'CO2(ppm)':<15} {'海平面(m)':<15}")
    print("-" * 60)
    for scenario, data in results.items():
        print(f"{scenario:<10} {data['temp_change']:+.2f}°C{'':<8} "
              f"{data['co2']:<15.1f} {data['sea_level']:<15.3f}")
    
    return results


def example_api_usage():
    """API使用示例"""
    print("\n" + "=" * 60)
    print("气候模型API - 使用示例")
    print("=" * 60)
    
    api = ClimateModelAPI()
    
    print("\n1. 创建模型...")
    resp = api.create_model(resolution="1km")
    print(f"   状态: {resp['status']}")
    
    print("\n2. 设置情景...")
    resp = api.set_scenario("RCP8.5")
    print(f"   消息: {resp['message']}")
    
    print("\n3. 运行模拟...")
    result = api.run_simulation(start_year=2020, end_year=2050, verbose=False)
    print(f"   模拟完成: {result['status']}")
    
    print("\n4. 获取预测...")
    prediction = api.get_prediction(variable="temperature", year=2050)
    print(f"   2050年气温预测: {prediction['value']:.2f}°C")
    
    print("\n5. 获取反馈分析...")
    feedback = api.get_feedback_analysis()
    print(f"   CO2辐射强迫: {feedback['co2_forcing']:.2f} W/m²")
    print(f"   气候敏感度: {feedback['climate_sensitivity']:.2f}")
    
    return api


if __name__ == "__main__":
    # 运行示例
    print("\n选择要运行的示例:")
    print("1. 基础使用")
    print("2. 情景对比")
    print("3. API使用")
    
    choice = input("\n请输入选项 (1-3): ")
    
    if choice == "1":
        example_basic_usage()
    elif choice == "2":
        example_scenario_comparison()
    elif choice == "3":
        example_api_usage()
    else:
        print("运行所有示例...")
        example_basic_usage()
        example_scenario_comparison()
        example_api_usage()
