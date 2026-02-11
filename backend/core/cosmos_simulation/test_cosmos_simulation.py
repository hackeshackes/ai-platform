"""
Test Module - 测试模块

提供宇宙模拟器的单元测试和集成测试。
"""

import unittest
import numpy as np
import sys
import os

# 添加模块路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from big_bang import BigBang, InitialConditions, PhysicalConstants
from galaxy_formation import GalaxyFormation, DarkMatterHalo, GalaxyType
from stellar_evolution import StellarEvolution, StellarProperties, EvolutionaryStage
from cosmology import Cosmology, CosmologicalParameters
from config import SimulationConfig


class TestBigBang(unittest.TestCase):
    """大爆炸模块测试"""
    
    def setUp(self):
        self.big_bang = BigBang()
    
    def test_initial_conditions(self):
        """测试初始条件设置"""
        z = 1000
        conditions = self.big_bang.set_initial_conditions(z)
        
        self.assertEqual(conditions.redshift, z)
        self.assertGreater(conditions.temperature, 0)
        self.assertLess(conditions.scale_factor, 1)
    
    def test_temperature_redshift_relation(self):
        """测试温度-红移关系 T = T_cmb * (1+z)"""
        for z in [0, 1, 10, 100, 1000]:
            T = self.big_bang.get_temperature_at_redshift(z)
            expected_T = 2.72548 * (1 + z)
            self.assertAlmostEqual(T, expected_T, places=2)
    
    def test_inflation_simulation(self):
        """测试暴胀模拟"""
        result = self.big_bang.simulate_inflation(e_folds=60)
        
        self.assertEqual(result["phase"], "inflation")
        self.assertEqual(result["duration_e_folds"], 60)
        self.assertIn("primordial_perturbations", result)
    
    def test_nucleosynthesis(self):
        """测试核合成模拟"""
        result = self.big_bang.simulate_nucleosynthesis()
        
        self.assertEqual(result["phase"], "big_bang_nucleosynthesis")
        self.assertIn("abundances", result)
        self.assertAlmostEqual(result["abundances"]["H"], 0.75, places=1)
    
    def test_recombination(self):
        """测试复合期模拟"""
        result = self.big_bang.simulate_recombination(z=1100)
        
        self.assertEqual(result["phase"], "recombination")
        self.assertEqual(result["redshift"], 1100)
    
    def test_universe_age(self):
        """测试宇宙年龄计算"""
        for z in [0, 1, 10]:
            age = self.big_bang.get_age_at_redshift(z)
            self.assertGreater(age, 0)
            # 年龄应该随红移减小而增加
            if z > 0:
                age_z0 = self.big_bang.get_age_at_redshift(0)
                self.assertGreater(age_z0, age)


class TestGalaxyFormation(unittest.TestCase):
    """星系形成模块测试"""
    
    def setUp(self):
        self.galaxy_sim = GalaxyFormation()
    
    def test_create_halo(self):
        """测试暗物质晕创建"""
        halo = self.galaxy_sim.create_collapsed_halo(mass=1e12, z=0)
        
        self.assertIsInstance(halo, DarkMatterHalo)
        self.assertEqual(halo.mass, 1e12)
        self.assertGreater(halo.radius, 0)
    
    def test_gas_cooling(self):
        """测试气体冷却"""
        halo = self.galaxy_sim.create_collapsed_halo(mass=1e12, z=0)
        gas = self.galaxy_sim.simulate_gas_cooling(halo)
        
        self.assertGreater(gas.mass, 0)
        self.assertGreater(gas.temperature, 0)
    
    def test_galaxy_evolution(self):
        """测试星系演化"""
        result = self.galaxy_sim.simulate_galaxy_evolution(
            galaxy_id="test_galaxy",
            initial_mass=1e11,
            z_start=5,
            z_end=0,
            time_step="1Gyr"
        )
        
        self.assertIn("evolution", result)
        self.assertIn("final_state", result)
        self.assertGreater(len(result["evolution"]), 0)
    
    def test_galaxy_population(self):
        """测试星系群生成"""
        population = self.galaxy_sim.generate_population(z=0, n_galaxies=100)
        
        self.assertEqual(len(population), 100)
        for galaxy in population:
            self.assertIn("mass", galaxy)
            self.assertIn("type", galaxy)


class TestStellarEvolution(unittest.TestCase):
    """恒星演化模块测试"""
    
    def setUp(self):
        self.stellar_sim = StellarEvolution()
    
    def test_create_star(self):
        """测试恒星创建"""
        for mass in [0.5, 1.0, 5.0, 10.0]:
            star = self.stellar_sim.create_star(mass)
            
            self.assertIsInstance(star, StellarProperties)
            self.assertAlmostEqual(star.mass, mass, places=2)
            self.assertGreater(star.lifetime, 0)
    
    def test_star_properties(self):
        """测试恒星属性计算"""
        star = self.stellar_sim.create_star(1.0)
        
        # 太阳类型恒星
        self.assertAlmostEqual(star.luminosity, 1.0, delta=0.5)
        self.assertAlmostEqual(star.radius, 1.0, delta=0.2)
        self.assertAlmostEqual(star.temperature, 5778, delta=500)
    
    def test_main_sequence_evolution(self):
        """测试主序星演化"""
        star = self.stellar_sim.create_star(1.0)
        track = self.stellar_sim.compute_main_sequence(star)
        
        self.assertGreater(len(track.time), 0)
        self.assertEqual(len(track.time), len(track.radius))
    
    def test_full_evolution_low_mass(self):
        """测试低质量恒星完整演化"""
        star = self.stellar_sim.create_star(1.0)
        result = self.stellar_sim.evolve_star(
            id(star),
            end_stage=EvolutionaryStage.WHITE_DWARF,
            time_step=0.1
        )
        
        self.assertEqual(result["final_stage"], "white_dwarf")
    
    def test_supernova_high_mass(self):
        """测试高质量恒星超新星"""
        star = self.stellar_sim.create_star(15.0)
        result = self.stellar_sim.evolve_star(
            id(star),
            end_stage=EvolutionaryStage.BLACK_HOLE,
            time_step=0.1
        )
        
        self.assertIn(result["final_stage"], ["black_hole", "neutron_star"])
    
    def test_imf_generation(self):
        """测试初始质量函数"""
        masses = self.stellar_sim._generate_imf(
            mass_range=(0.1, 100),
            n=1000,
            alpha=-2.35
        )
        
        self.assertEqual(len(masses), 1000)
        self.assertTrue(all(masses >= 0.1))
        self.assertTrue(all(masses <= 100))


class TestCosmology(unittest.TestCase):
    """宇宙学模块测试"""
    
    def setUp(self):
        self.cosmology = Cosmology()
    
    def test_hubble_parameter(self):
        """测试哈勃参数"""
        for z in [0, 1, 10, 100]:
            H = self.cosmology.compute_H(z)
            
            self.assertGreater(H, 0)
            # H(z) 应该随红移增加
            H_z0 = self.cosmology.compute_H(0)
            self.assertGreater(H, H_z0)
    
    def test_critical_density(self):
        """测试临界密度"""
        rho_c = self.cosmology.rho_crit
        
        self.assertGreater(rho_c, 0)
        self.assertAlmostEqual(rho_c / 9.47e-27, 1.0, delta=0.1)  # 约 9.47e-27 kg/m³
    
    def test_distance_modulus(self):
        """测试距离模数"""
        mu = self.cosmology.compute_distance_modulus(0)
        self.assertAlmostEqual(mu, 0, places=2)  # z=0时，mu=0
        
        mu = self.cosmology.compute_distance_modulus(0.5)
        self.assertGreater(mu, 0)
    
    def test_luminosity_distance(self):
        """测试光度距离"""
        for z in [0, 0.5, 1.0, 2.0]:
            d_L = self.cosmology.compute_luminosity_distance(z)
            self.assertGreater(d_L, 0)
    
    def test_lookback_time(self):
        """测试回溯时间"""
        t = self.cosmology.compute_lookback_time(0)
        self.assertAlmostEqual(t, 0, places=2)  # z=0时，回溯时间=0
        
        t = self.cosmology.compute_lookback_time(1)
        self.assertGreater(t, 0)
    
    def test_age_universe(self):
        """测试宇宙年龄"""
        age_z0 = self.cosmology.compute_age(0)
        self.assertAlmostEqual(age_z0, 13.8, delta=1.0)  # 约13.8 Gyr
        
        age_z10 = self.cosmology.compute_age(10)
        self.assertLess(age_z10, age_z0)
    
    def test_growth_factor(self):
        """测试增长因子"""
        for z in [0, 1, 10, 100]:
            D = self.cosmology.compute_linear_growth_factor(z)
            self.assertGreaterEqual(D, 0)
            self.assertLessEqual(D, 1)
    
    def test_power_spectrum(self):
        """测试功率谱"""
        P = self.cosmology.compute_power_spectrum(z=0)
        
        self.assertEqual(len(P.k), 100)
        self.assertEqual(len(P.k), len(P.P_k))
        self.assertTrue(all(P.P_k > 0))
    
    def test_cmb_power_spectrum(self):
        """测试CMB功率谱"""
        cmb = self.cosmology.compute_cmb_power_spectrum()
        
        self.assertEqual(len(cmb.ell), 100)
        self.assertTrue(all(cmb.C_ell_tt > 0))
    
    def test_clustering_statistics(self):
        """测试聚类统计"""
        stats = self.cosmology.compute_clustering_statistics()
        
        self.assertIn("sigma_8", stats)
        self.assertAlmostEqual(stats["sigma_8"], 0.811, delta=0.1)


class TestConfig(unittest.TestCase):
    """配置模块测试"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = SimulationConfig()
        
        self.assertEqual(config.H0, 67.4)
        self.assertEqual(config.Omega_m, 0.315)
    
    def test_config_from_file(self):
        """测试从文件加载配置"""
        # 临时配置文件
        import tempfile
        import json
        
        config_data = {
            "H0": 70.0,
            "Omega_m": 0.3,
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name
        
        try:
            config = SimulationConfig.from_file(temp_path)
            self.assertEqual(config.H0, 70.0)
            self.assertEqual(config.Omega_m, 0.3)
        finally:
            os.unlink(temp_path)
    
    def test_config_presets(self):
        """测试配置预设"""
        planck = SimulationConfig.planck()
        self.assertEqual(planck.H0, 67.4)
        
        wmap = SimulationConfig.wmap()
        self.assertEqual(wmap.H0, 70.0)
    
    def test_time_step_conversion(self):
        """测试时间步长转换"""
        config = SimulationConfig()
        
        # 测试自动
        dt = config.get_time_step_seconds()
        self.assertGreater(dt, 0)


class TestIntegration(unittest.TestCase):
    """集成测试"""
    
    def test_full_simulation(self):
        """测试完整模拟流程"""
        from cosmos_simulation import CosmosSimulation
        
        cosmos = CosmosSimulation()
        initial = cosmos.set_initial_conditions(z=100)
        evolution = cosmos.evolve(end_redshift=0, time_step="1Gyr")
        state = evolution.get_state(redshift=0, scale="galaxy")
        
        self.assertIsNotNone(initial)
        self.assertIsNotNone(evolution)
        self.assertIsNotNone(state)
        self.assertGreater(state['age'], 0)
    
    def test_api_response(self):
        """测试API响应"""
        from api import CosmosAPI
        
        api = CosmosAPI()
        response = api.get_status()
        
        self.assertTrue(response.success)
        self.assertIn("running", response.data)


def run_tests():
    """运行所有测试"""
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加测试类
    suite.addTests(loader.loadTestsFromTestCase(TestBigBang))
    suite.addTests(loader.loadTestsFromTestCase(TestGalaxyFormation))
    suite.addTests(loader.loadTestsFromTestCase(TestStellarEvolution))
    suite.addTests(loader.loadTestsFromTestCase(TestCosmology))
    suite.addTests(loader.loadTestsFromTestCase(TestConfig))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == "__main__":
    run_tests()
