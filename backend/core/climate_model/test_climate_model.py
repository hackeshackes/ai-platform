"""
气候模型测试用例
"""

import unittest
import numpy as np
import sys
import os

# 添加backend到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.climate_model import ClimateModel
from core.climate_model.atmosphere import AtmosphereModel
from core.climate_model.ocean import OceanModel
from core.climate_model.land import LandModel


class TestAtmosphereModel(unittest.TestCase):
    """大气模型测试"""
    
    def setUp(self):
        self.model = AtmosphereModel(resolution="1km", grid_size=36, lat_bands=18)
        
    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.model.resolution, "1km")
        self.assertEqual(self.model.grid_size, 36)
        self.assertEqual(self.model.lat_bands, 18)
        
    def test_temperature_shape(self):
        """测试气温数组形状"""
        state = self.model.get_state()
        self.assertEqual(state.temperature.shape, (18, 36))
        
    def test_co2_setter(self):
        """测试CO2设置"""
        self.model.set_co2_concentration(450)
        self.assertEqual(self.model.co2_concentration, 450)
        
    def test_step(self):
        """测试时间步进"""
        initial_time = self.model.time
        self.model.step(scenario="RCP8.5")
        self.assertGreater(self.model.time.year, initial_time.year)
        
    def test_statistics(self):
        """测试统计信息"""
        stats = self.model.get_statistics()
        self.assertIn('mean_temperature', stats)
        self.assertIn('global_co2', stats)
        self.assertIn('mean_cloud_cover', stats)


class TestOceanModel(unittest.TestCase):
    """海洋模型测试"""
    
    def setUp(self):
        self.model = OceanModel(resolution="1km", grid_size=36, lat_bands=18)
        
    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.model.resolution, "1km")
        
    def test_state_shape(self):
        """测试状态数组形状"""
        state = self.model.get_state()
        self.assertEqual(state.sea_temperature.shape, (18, 36))
        self.assertEqual(state.salinity.shape, (18, 36))
        
    def test_step(self):
        """测试时间步进"""
        initial_time = self.model.time
        self.model.step(scenario="RCP8.5")
        self.assertGreater(self.model.time.year, initial_time.year)
        
    def test_statistics(self):
        """测试统计信息"""
        stats = self.model.get_statistics()
        self.assertIn('mean_sea_temperature', stats)
        self.assertIn('mean_salinity', stats)


class TestLandModel(unittest.TestCase):
    """陆地模型测试"""
    
    def setUp(self):
        self.model = LandModel(resolution="1km", grid_size=36, lat_bands=18)
        
    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.model.resolution, "1km")
        
    def test_state_shape(self):
        """测试状态数组形状"""
        state = self.model.get_state()
        self.assertEqual(state.temperature.shape, (18, 36))
        self.assertEqual(state.vegetation.shape, (18, 36))
        
    def test_land_mask(self):
        """测试陆地掩码"""
        self.assertTrue(np.any(self.model.land_mask))
        self.assertTrue(np.any(~self.model.land_mask))
        
    def test_step(self):
        """测试时间步进"""
        initial_time = self.model.time
        self.model.step(scenario="RCP8.5")
        self.assertGreater(self.model.time.year, initial_time.year)
        
    def test_statistics(self):
        """测试统计信息"""
        stats = self.model.get_statistics()
        self.assertIn('mean_land_temperature', stats)
        self.assertIn('glacier_mass', stats)


class TestClimateModel(unittest.TestCase):
    """气候模型主测试"""
    
    def setUp(self):
        self.model = ClimateModel(resolution="1km", grid_size=36, lat_bands=18)
        
    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.model.resolution, "1km")
        self.assertIsNotNone(self.model.atmosphere)
        self.assertIsNotNone(self.model.ocean)
        self.assertIsNotNone(self.model.land)
        
    def test_scenario_setting(self):
        """测试情景设置"""
        self.model.set_scenario("RCP4.5")
        self.assertEqual(self.model.scenario, "RCP4.5")
        
    def test_invalid_scenario(self):
        """测试无效情景"""
        with self.assertRaises(ValueError):
            self.model.set_scenario("INVALID")
            
    def test_run_simulation(self):
        """测试运行模拟"""
        result = self.model.run(start_year=2020, end_year=2025, verbose=False)
        self.assertIn('scenario', result)
        self.assertIn('history', result)
        self.assertEqual(len(result['history']['years']), 6)
        
    def test_history_data(self):
        """测试历史数据"""
        self.model.run(start_year=2020, end_year=2030, verbose=False)
        self.assertGreater(len(self.model.history['years']), 0)
        self.assertGreater(len(self.model.history['temperature']), 0)
        self.assertGreater(len(self.model.history['co2']), 0)
        
    def test_prediction(self):
        """测试预测获取"""
        self.model.run(start_year=2020, end_year=2100, verbose=False)
        
        temp_pred = self.model.get_prediction(variable="temperature", year=2100)
        self.assertIn('value', temp_pred)
        self.assertIsNotNone(temp_pred['value'])
        
        co2_pred = self.model.get_prediction(variable="co2", year=2100)
        self.assertIn('value', co2_pred)
        
    def test_get_state(self):
        """测试获取状态"""
        state = self.model.get_state()
        self.assertIn('atmosphere', state)
        self.assertIn('ocean', state)
        self.assertIn('land', state)
        
    def test_feedback_analysis(self):
        """测试反馈分析"""
        analysis = self.model.get_feedback_analysis()
        self.assertIn('co2_forcing', analysis)
        self.assertIn('feedbacks', analysis)
        self.assertIn('climate_sensitivity', analysis)


class TestClimateModelAPI(unittest.TestCase):
    """API测试"""
    
    def setUp(self):
        from core.climate_model.api import ClimateModelAPI
        self.api = ClimateModelAPI()
        
    def test_create_model(self):
        """测试创建模型"""
        result = self.api.create_model(resolution="1km")
        self.assertEqual(result['status'], 'success')
        self.assertEqual(self.api.status, 'initialized')
        
    def test_set_scenario(self):
        """测试设置情景"""
        self.api.create_model()
        result = self.api.set_scenario("RCP2.6")
        self.assertEqual(result['status'], 'success')
        
    def test_run_simulation(self):
        """测试运行模拟"""
        self.api.create_model()
        result = self.api.run_simulation(start_year=2020, end_year=2025, verbose=False)
        self.assertEqual(result['status'], 'success')
        self.assertEqual(self.api.status, 'completed')
        
    def test_get_prediction(self):
        """测试获取预测"""
        self.api.create_model()
        self.api.run_simulation(start_year=2020, end_year=2100, verbose=False)
        pred = self.api.get_prediction(variable="temperature", year=2100)
        self.assertIn('value', pred)
        
    def test_list_scenarios(self):
        """测试列出情景"""
        scenarios = self.api.list_scenarios()
        self.assertIn('scenarios', scenarios)
        self.assertIn('RCP2.6', scenarios['scenarios'])
        self.assertIn('RCP8.5', scenarios['scenarios'])


class TestSimulationAccuracy(unittest.TestCase):
    """模拟精度测试"""
    
    def test_temperature_range(self):
        """测试温度在合理范围内"""
        model = ClimateModel(resolution="1km", grid_size=36, lat_bands=18)
        model.run(start_year=2020, end_year=2100, verbose=False)
        
        temps = model.history['temperature']
        self.assertTrue(all(-10 < t < 50 for t in temps),
                       f"温度超出合理范围: {min(temps):.2f} - {max(temps):.2f}")
        
    def test_co2_range(self):
        """测试CO2浓度在合理范围内"""
        model = ClimateModel(resolution="1km", grid_size=36, lat_bands=18)
        model.run(start_year=2020, end_year=2100, verbose=False)
        
        co2_values = model.history['co2']
        self.assertTrue(all(280 < c < 1000 for c in co2_values),
                       f"CO2浓度超出合理范围: {min(co2_values):.1f} - {max(co2_values):.1f}")
        
    def test_sea_level_change(self):
        """测试海平面变化方向正确"""
        model = ClimateModel(resolution="1km", grid_size=36, lat_bands=18)
        model.run(start_year=2020, end_year=2100, verbose=False)
        
        sea_levels = model.history['sea_level']
        # 海平面应该上升
        self.assertGreater(sea_levels[-1], sea_levels[0],
                          "海平面应该上升")
        
    def test_trend_consistency(self):
        """测试趋势一致性"""
        model = ClimateModel(resolution="1km", grid_size=36, lat_bands=18)
        model.run(start_year=2020, end_year=2100, verbose=False)
        
        # 温度趋势应该为正 (全球变暖)
        temps = model.history['temperature']
        slope = np.polyfit(range(len(temps)), temps, 1)[0]
        self.assertGreater(slope, 0, "温度趋势应该为正")


if __name__ == '__main__':
    # 运行测试
    unittest.main(verbosity=2)
