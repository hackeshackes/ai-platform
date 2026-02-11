"""
深空探测系统测试 - Deep Space System Tests
=============================================

单元测试和集成测试

作者: AI Platform Team
版本: 1.0.0
"""

import unittest
import math
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from deep_space.navigation import (
    DeepSpaceNavigator, 
    OrbitalCalculator, 
    ObstacleAvoider,
    RouteMethod,
    Position,
    Velocity,
    TrajectoryPoint
)
from deep_space.planet_explorer import (
    PlanetExplorer,
    TerrainAnalyzer,
    ResourceIdentifier,
    LandingSiteSelector,
    TerrainType,
    ResourceType
)
from deep_space.seti_integration import (
    SETIAnalyzer,
    SignalProcessor,
    AnomalyDetector,
    PatternRecognizer,
    CivilizationAssessor,
    RawSignal,
    SignalType
)
from deep_space.communication import (
    DeepSpaceCommunicator,
    SignalEncoder,
    ErrorCorrector,
    DelayCompensator,
    BandwidthOptimizer
)
from deep_space.config import get_config, PLANET_DATA


class TestOrbitalCalculator(unittest.TestCase):
    """轨道计算器测试"""
    
    def test_orbital_period(self):
        """测试轨道周期计算"""
        # 地球轨道周期应该约365.25天
        period = OrbitalCalculator.orbital_period(1.0)
        self.assertAlmostEqual(period, 365.25, delta=1)
    
    def test_hohmann_transfer(self):
        """测试霍曼转移轨道"""
        # 地球到火星的霍曼转移
        hohmann = OrbitalCalculator.hohmann_transfer(1.0, 1.524)
        self.assertGreater(hohmann['transfer_time'], 200)
        self.assertLess(hohmann['transfer_time'], 300)
        self.assertGreater(hohmann['total_delta_v'], 5)
        self.assertLess(hohmann['total_delta_v'], 10)
    
    def test_position_at_anomaly(self):
        """测试位置计算"""
        orbital_elements = OrbitalCalculator.kepler_orbital_elements(
            semi_major_axis=1.0,
            eccentricity=0.0167,
            inclination=0,
            longitude_ascending=0,
            argument_periapsis=0,
            true_anomaly=0
        )
        position = OrbitalCalculator.position_at_anomaly(orbital_elements, 0)
        self.assertIsInstance(position, Position)
        # 近地点应该接近1 AU
        self.assertAlmostEqual(position.x, 1.0, places=1)
    
    def test_position_distance(self):
        """测试位置距离计算"""
        pos1 = Position(0, 0, 0)
        pos2 = Position(1, 0, 0)
        self.assertAlmostEqual(pos1.distance_to(pos2), 1.0)


class TestObstacleAvoider(unittest.TestCase):
    """障碍规避器测试"""
    
    def setUp(self):
        self.avoider = ObstacleAvoider()
        self.test_trajectory = [
            TrajectoryPoint(
                position=Position(1, 0, 0),
                velocity=Velocity(0, 0, 0),
                time=0,
                fuel_consumed=0
            )
        ]
    
    def test_detect_collision_no_collision(self):
        """测试无碰撞检测"""
        from deep_space.navigation import Obstacle, ObstacleType
        self.avoider.add_obstacle(Obstacle(
            name="test",
            position=Position(10, 0, 0),
            radius=1,
            type=ObstacleType.ASTEROID
        ))
        collisions = self.avoider.detect_collision(self.test_trajectory)
        self.assertEqual(len(collisions), 0)
    
    def test_assess_route_safety(self):
        """测试航线安全评估"""
        safety = self.avoider.assess_route_safety(self.test_trajectory)
        self.assertEqual(safety, 1.0)


class TestNavigation(unittest.TestCase):
    """导航系统测试"""
    
    def setUp(self):
        self.navigator = DeepSpaceNavigator()
    
    def test_plan_route_earth_mars(self):
        """测试地球到火星航线规划"""
        route = self.navigator.plan_route(
            origin="earth",
            destination="mars",
            method="optimal"
        )
        self.assertEqual(route.origin, "earth")
        self.assertEqual(route.destination, "mars")
        self.assertEqual(route.method, RouteMethod.OPTIMAL)
        self.assertGreater(route.total_distance, 0)
        self.assertGreater(route.total_time, 0)
        self.assertGreater(route.estimated_success_rate, 0.9)
    
    def test_plan_route_fastest(self):
        """测试最快航线规划"""
        route = self.navigator.plan_route(
            origin="earth",
            destination="mars",
            method="fastest"
        )
        self.assertEqual(route.method, RouteMethod.FASTEST)
    
    def test_plan_route_unknown_planet(self):
        """测试未知行星"""
        with self.assertRaises(ValueError):
            self.navigator.plan_route(
                origin="earth",
                destination="unknown_planet"
            )
    
    def test_get_navigation_status(self):
        """测试获取导航状态"""
        status = self.navigator.getNavigationStatus()
        self.assertIn('position', status)
        self.assertIn('velocity', status)
        self.assertIn('known_obstacles', status)


class TestTerrainAnalyzer(unittest.TestCase):
    """地形分析器测试"""
    
    def setUp(self):
        self.analyzer = TerrainAnalyzer("mars")
    
    def test_analyze_surface(self):
        """测试表面分析"""
        terrain = self.analyzer.analyze_surface()
        self.assertIsInstance(terrain, list)
        self.assertGreater(len(terrain), 0)
        for td in terrain:
            self.assertIsInstance(td.terrain_type, TerrainType)
            self.assertIsInstance(td.slope, float)
    
    def test_generate_elevation_map(self):
        """测试高程图生成"""
        elevation_map = self.analyzer.generate_elevation_map(10, 10)
        self.assertEqual(len(elevation_map), 10)
        self.assertEqual(len(elevation_map[0]), 10)
    
    def test_detect_hazards(self):
        """测试危险检测"""
        from deep_space.planet_explorer import TerrainData
        test_terrain = [
            TerrainData(
                terrain_type=TerrainType.PLAIN,
                elevation=0,
                slope=50,  # 陡坡
                roughness=0.5,
                rock_density=0.1,
                thermal_stability=0.9,
                description="test"
            )
        ]
        hazards = self.analyzer.detect_hazards(test_terrain)
        self.assertGreater(len(hazards), 0)


class TestResourceIdentifier(unittest.TestCase):
    """资源识别器测试"""
    
    def setUp(self):
        self.identifier = ResourceIdentifier("mars")
    
    def test_scan_for_resources(self):
        """测试资源扫描"""
        resources = self.identifier.scan_for_resources()
        self.assertIsInstance(resources, list)
    
    def test_estimate_resource_value(self):
        """测试资源价值估算"""
        from deep_space.planet_explorer import ResourceDeposit
        deposit = ResourceDeposit(
            resource_type=ResourceType.WATER_ICE,
            concentration=0.5,
            depth=10,
            estimated_amount=1000000,
            accessibility=0.8,
            confidence=0.9,
            location=(0, 0)
        )
        value = self.identifier.estimate_resource_value(deposit)
        self.assertGreater(value, 0)


class TestLandingSiteSelector(unittest.TestCase):
    """着陆点选择器测试"""
    
    def setUp(self):
        self.selector = LandingSiteSelector("mars")
    
    def test_select_site(self):
        """测试着陆点选择"""
        from deep_space.planet_explorer import TerrainData
        
        # 创建测试地形数据
        test_terrain = [
            TerrainData(
                terrain_type=TerrainType.PLAIN,
                elevation=0,
                slope=5,
                roughness=0.3,
                rock_density=0.1,
                thermal_stability=0.9,
                description="flat area"
            )
            for _ in range(10)
        ]
        
        sites = self.selector.select_site(test_terrain)
        self.assertIsInstance(sites, list)
    
    def test_calculate_safety_score(self):
        """测试安全分数计算"""
        from deep_space.planet_explorer import TerrainData
        td = TerrainData(
            terrain_type=TerrainType.PLAIN,
            elevation=0,
            slope=5,
            roughness=0.3,
            rock_density=0.1,
            thermal_stability=0.9,
            description="safe terrain"
        )
        score = self.selector._calculate_safety_score(td)
        self.assertGreater(score, 0.7)


class TestPlanetExplorer(unittest.TestCase):
    """行星探测器测试"""
    
    def test_initialize_explorer(self):
        """测试探测器初始化"""
        explorer = PlanetExplorer("mars")
        self.assertEqual(explorer.planet, "mars")
    
    def test_full_exploration(self):
        """测试完整探索"""
        explorer = PlanetExplorer("mars")
        result = explorer.full_exploration()
        self.assertEqual(result['mission_summary']['status'], 'completed')
        self.assertIn('terrain_analysis', result)
        self.assertIn('resource_survey', result)
        self.assertIn('landing_site_selection', result)


class TestSignalProcessor(unittest.TestCase):
    """信号处理器测试"""
    
    def setUp(self):
        self.processor = SignalProcessor()
    
    def test_process_signal(self):
        """测试信号处理"""
        signal = RawSignal(
            signal_id="test-001",
            timestamp=1234567890.0,
            frequency=1.42e9,
            bandwidth=1.0,
            intensity=1.0,
            duration=1.0,
            source_direction=(0, 0),
            snr=10.0
        )
        processed = self.processor.process_signal(signal)
        self.assertIsInstance(processed.signal_type, SignalType)
    
    def test_process_batch(self):
        """测试批量处理"""
        signals = [
            RawSignal(
                signal_id=f"test-{i:03d}",
                timestamp=1234567890.0 + i,
                frequency=1.42e9,
                bandwidth=1.0,
                intensity=1.0,
                duration=1.0,
                source_direction=(0, 0),
                snr=10.0
            )
            for i in range(10)
        ]
        processed = self.processor.process_batch(signals)
        self.assertEqual(len(processed), 10)


class TestAnomalyDetector(unittest.TestCase):
    """异常检测器测试"""
    
    def setUp(self):
        self.detector = AnomalyDetector()
    
    def test_detect_anomaly(self):
        """测试异常检测"""
        from deep_space.seti_integration import ProcessedSignal, SignalClassification
        
        signal = RawSignal(
            signal_id="test-001",
            timestamp=1234567890.0,
            frequency=1.42e9,
            bandwidth=0.001,
            intensity=100,
            duration=1.0,
            source_direction=(0, 0),
            snr=50.0
        )
        
        processed = ProcessedSignal(
            raw_signal=signal,
            signal_type=SignalType.ARTIFICIAL,
            classification=SignalClassification.CANDIDATE,
            features={
                'frequency_stability': 0.99,
                'bandwidth_ratio': 1e-9,
                'spectral_purity': 0.98
            },
            confidence=0.95
        )
        
        anomalies = self.detector.detect_anomaly([processed])
        self.assertIsInstance(anomalies, list)


class TestSETIAnalyzer(unittest.TestCase):
    """SETI分析器测试"""
    
    def setUp(self):
        self.seti = SETIAnalyzer()
    
    def test_scan(self):
        """测试扫描"""
        result = self.seti.scan()
        self.assertIn('scan_summary', result)
        self.assertIn('signal_analysis', result)
        self.assertIn('anomaly_report', result)
        self.assertIn('civilization_assessment', result)
    
    def test_detect_anomaly(self):
        """测试异常检测"""
        result = self.seti.detect_anomaly()
        self.assertIn('anomalies_detected', result)
    
    def test_system_status(self):
        """测试系统状态"""
        status = self.seti.get_system_status()
        self.assertEqual(status['status'], 'operational')


class TestSignalEncoder(unittest.TestCase):
    """信号编码器测试"""
    
    def setUp(self):
        self.encoder = SignalEncoder()
    
    def test_encode_decode(self):
        """测试编码解码"""
        original = b"Hello, Deep Space!"
        encoded = self.encoder.encode(original)
        decoded = self.encoder.decode(encoded)
        self.assertEqual(original, decoded)
    
    def test_calculate_overhead(self):
        """测试开销计算"""
        overhead = self.encoder.calculate_overhead(1000)
        self.assertGreater(overhead, 0)


class TestErrorCorrector(unittest.TestCase):
    """错误纠正器测试"""
    
    def setUp(self):
        self.corrector = ErrorCorrector()
    
    def test_add_correct_errors(self):
        """测试添加和纠正错误"""
        original = b"Test data for error correction"
        corrected = self.corrector.add_error_correction(original)
        result = self.corrector.correct_errors(corrected)
        self.assertEqual(len(result), len(original))
    
    def test_interleave_deinterleave(self):
        """测试交织解交织"""
        original = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        interleaved = self.corrector.interleave(original)
        deinterleaved = self.corrector.deinterleave(interleaved)
        self.assertEqual(original, deinterleaved)


class TestDelayCompensator(unittest.TestCase):
    """延迟补偿器测试"""
    
    def setUp(self):
        self.compensator = DelayCompensator()
    
    def test_calculate_light_time(self):
        """测试光时计算"""
        # 地球到月球约1.28光秒
        light_time = self.compensator.calculate_light_time(384400e3)
        self.assertAlmostEqual(light_time, 1.28, places=1)
    
    def test_compensate_delay(self):
        """测试延迟补偿"""
        result = self.compensator.compensate_delay(b"test", 384400e3)
        self.assertIn('distance_m', result)
        self.assertIn('one_way_delay_s', result)
        self.assertIn('round_trip_delay_s', result)


class TestBandwidthOptimizer(unittest.TestCase):
    """带宽优化器测试"""
    
    def setUp(self):
        self.optimizer = BandwidthOptimizer()
    
    def test_optimize_bandwidth(self):
        """测试带宽优化"""
        result = self.optimizer.optimize_bandwidth(
            available_bandwidth=1e6,
            required_data_rate=500e3,
            channel_conditions={'snr': 15}
        )
        self.assertIn('allocated_bandwidth_hz', result)
        self.assertIn('actual_data_rate_bps', result)
        self.assertIn('modulation', result)
    
    def test_shannon_capacity(self):
        """测试香农容量"""
        capacity = self.optimizer._shannon_capacity(1e6, 10)
        self.assertGreater(capacity, 0)


class TestDeepSpaceCommunicator(unittest.TestCase):
    """深空通信器测试"""
    
    def setUp(self):
        self.comm = DeepSpaceCommunicator()
    
    def test_send_message(self):
        """测试发送消息"""
        result = self.comm.send_message(
            content="Test message",
            destination="mars",
            priority=3
        )
        self.assertIn('message_id', result)
        self.assertIn('channel', result)
        self.assertIn('delay_info', result)
    
    def test_configure_link(self):
        """测试配置链路"""
        result = self.comm.configure_link(
            target_distance=225e9,
            required_data_rate=1e6
        )
        self.assertIn('bandwidth_optimization', result)
        self.assertIn('delay_compensation', result)
    
    def test_communication_status(self):
        """测试通信状态"""
        status = self.comm.get_communication_status()
        self.assertEqual(status['status'], 'operational')
        self.assertIn('active_channels', status)


class TestConfig(unittest.TestCase):
    """配置测试"""
    
    def test_get_config(self):
        """测试获取配置"""
        config = get_config()
        self.assertIsNotNone(config.navigation)
        self.assertIsNotNone(config.exploration)
        self.assertIsNotNone(config.seti)
        self.assertIsNotNone(config.communication)
    
    def test_planet_data(self):
        """测试行星数据"""
        self.assertIn('mars', PLANET_DATA)
        self.assertIn('jupiter', PLANET_DATA)
        self.assertEqual(PLANET_DATA['mars']['name'], '火星')


def run_tests():
    """运行所有测试"""
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加测试类
    test_classes = [
        TestOrbitalCalculator,
        TestObstacleAvoider,
        TestNavigation,
        TestTerrainAnalyzer,
        TestResourceIdentifier,
        TestLandingSiteSelector,
        TestPlanetExplorer,
        TestSignalProcessor,
        TestAnomalyDetector,
        TestSETIAnalyzer,
        TestSignalEncoder,
        TestErrorCorrector,
        TestDelayCompensator,
        TestBandwidthOptimizer,
        TestDeepSpaceCommunicator,
        TestConfig
    ]
    
    for test_class in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(test_class))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 输出摘要
    print("\n" + "=" * 60)
    print("测试摘要")
    print("=" * 60)
    print(f"测试运行: {result.testsRun}")
    print(f"成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")
    print("=" * 60)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    import logging
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 运行测试
    success = run_tests()
    sys.exit(0 if success else 1)
