"""
Examples Module - 使用示例

提供宇宙模拟器的使用示例。
"""

from cosmos_simulation import CosmosSimulation
from big_bang import BigBang
from galaxy_formation import GalaxyFormation
from stellar_evolution import StellarEvolution
from cosmology import Cosmology
from api import CosmosAPI
from config import SimulationConfig


def example_1_basic_simulation():
    """示例1: 基础宇宙模拟"""
    print("=" * 50)
    print("Example 1: Basic Universe Simulation")
    print("=" * 50)
    
    # 创建模拟器
    cosmos = CosmosSimulation()
    
    # 设置初始条件 (z=1000)
    initial = cosmos.set_initial_conditions(z=1000)
    print(f"Initial redshift: {initial.redshift}")
    print(f"Initial temperature: {initial.temperature:.2e} K")
    
    # 运行演化
    evolution = cosmos.evolve(end_redshift=0, time_step="1Gyr")
    print(f"Evolution completed. Final redshift: {evolution.final_redshift}")
    
    # 获取最终状态
    state = evolution.get_state(redshift=0, scale="galaxy")
    print(f"Final age: {state['age']:.2f} Gyr")
    
    return cosmos


def example_2_galaxy_formation():
    """示例2: 星系形成模拟"""
    print("\n" + "=" * 50)
    print("Example 2: Galaxy Formation")
    print("=" * 50)
    
    galaxy_sim = GalaxyFormation()
    
    # 创建暗物质晕
    halo = galaxy_sim.create_collapsed_halo(mass=1e12, z=2)
    print(f"Halo mass: {halo.mass:.2e} M_sun")
    print(f"Halo radius: {halo.radius:.2f} kpc")
    
    # 模拟气体冷却
    gas = galaxy_sim.simulate_gas_cooling(halo)
    print(f"Gas mass: {gas.mass:.2e} M_sun")
    print(f"Gas temperature: {gas.temperature:.2e} K")
    
    # 演化单个星系
    evolution = galaxy_sim.simulate_galaxy_evolution(
        galaxy_id="milky_way",
        initial_mass=1e12,
        z_start=10,
        z_end=0,
        time_step="1Gyr"
    )
    
    print(f"\nGalaxy evolution:")
    print(f"  Initial stellar mass: {evolution['evolution'][0]['stellar_mass']:.2e}")
    print(f"  Final stellar mass: {evolution['final_state']['stellar_mass']:.2e}")
    print(f"  Final SFR: {evolution['final_state']['sfr']:.2f} M_sun/yr")
    
    return galaxy_sim


def example_3_stellar_evolution():
    """示例3: 恒星演化模拟"""
    print("\n" + "=" * 50)
    print("Example 3: Stellar Evolution")
    print("=" * 50)
    
    stellar_sim = StellarEvolution()
    
    # 创建不同质量的恒星
    masses = [1.0, 5.0, 10.0, 25.0]  # M_sun
    
    for mass in masses:
        star = stellar_sim.create_star(mass, metallicity=0.02)
        print(f"\nStar (M={mass} M_sun):")
        print(f"  Spectral type: {star.stellar_type.value}")
        print(f"  Radius: {star.radius:.2f} R_sun")
        print(f"  Luminosity: {star.luminosity:.2f} L_sun")
        print(f"  Temperature: {star.temperature:.0f} K")
        print(f"  Lifetime: {star.lifetime:.2f} Gyr")
        
        # 演化恒星
        result = stellar_sim.evolve_star(
            id(star),
            end_stage="white_dwarf" if mass < 8 else "supernova",
            time_step=0.1
        )
        
        print(f"  Final mass: {result['final_mass']:.2f} M_sun")
        print(f"  Final stage: {result['final_stage']}")
        
        if result['supernova']:
            print(f"  Supernova type: {result['supernova']['type']}")
    
    return stellar_sim


def example_4_cosmology():
    """示例4: 宇宙学计算"""
    print("\n" + "=" * 50)
    print("Example 4: Cosmology Calculations")
    print("=" * 50)
    
    cosmology = Cosmology()
    
    # 计算不同红移处的宇宙学参数
    redshifts = [0, 1, 3, 6, 1100]
    
    print("\nCosmological Parameters vs Redshift:")
    print("-" * 70)
    print(f"{'z':>6} {'H(z)':>12} {'D_L':>12} {'t_lookback':>12} {'Age':>10}")
    print("-" * 70)
    
    for z in redshifts:
        H = cosmology.compute_H(z)
        D_L = cosmology.compute_luminosity_distance(z)
        t_lookback = cosmology.compute_lookback_time(z)
        age = cosmology.compute_age(z)
        
        print(f"{z:>6.1f} {H:>10.2f} {D_L:>10.2f} {t_lookback:>10.2f} {age:>10.2f}")
    
    print("-" * 70)
    
    # 计算功率谱
    print("\nComputing matter power spectrum...")
    P = cosmology.compute_power_spectrum(z=0)
    print(f"  k range: {P.k[0]:.2e} - {P.k[-1]:.2e} 1/Mpc")
    print(f"  P(k) range: {P.P_k[0]:.2e} - {P.P_k[-1]:.2e}")
    
    # 计算CMB功率谱
    print("\nComputing CMB power spectrum...")
    cmb = cosmology.compute_cmb_power_spectrum()
    print(f"  l range: {cmb.ell[0]:.0f} - {cmb.ell[-1]:.0f}")
    print(f"  TT peak amplitude: {max(cmb.C_ell_tt):.2f} uK²")
    
    return cosmology


def example_5_api_usage():
    """示例5: API使用"""
    print("\n" + "=" * 50)
    print("Example 5: API Usage")
    print("=" * 50)
    
    api = CosmosAPI()
    
    # 创建模拟
    response = api.create_simulation()
    print(f"Create simulation: {response.message}")
    print(f"  Success: {response.success}")
    
    # 设置初始条件
    response = api.set_initial_conditions(z=1000)
    print(f"\nSet initial conditions: {response.message}")
    print(f"  Success: {response.success}")
    
    # 运行演化
    response = api.evolve(end_redshift=0, time_step="1Gyr")
    print(f"\nEvolve simulation: {response.message}")
    print(f"  Success: {response.success}")
    
    # 获取状态
    response = api.get_state(redshift=0, scale="galaxy")
    print(f"\nGet state: {response.message}")
    print(f"  Success: {response.success}")
    
    # 计算宇宙学距离
    response = api.compute_distance(z=0.5)
    print(f"\nCompute distances at z=0.5:")
    if response.success:
        data = response.data
        print(f"  Luminosity distance: {data['luminosity_distance']:.2f} Mpc")
        print(f"  Angular diameter distance: {data['angular_diameter_distance']:.2f} Mpc")
        print(f"  Lookback time: {data['lookback_time']:.2f} Gyr")
    
    return api


def example_6_full_workflow():
    """示例6: 完整工作流"""
    print("\n" + "=" * 50)
    print("Example 6: Full Simulation Workflow")
    print("=" * 50)
    
    # 配置
    config = SimulationConfig(
        z_start=100,
        z_end=0,
        H0=67.4,
        Omega_m=0.315,
    )
    
    # 1. 初始化模拟器
    cosmos = CosmosSimulation(config)
    print("1. Initialized Cosmos Simulation")
    
    # 2. 设置初始条件
    initial = cosmos.set_initial_conditions(z=100)
    print(f"2. Set initial conditions at z={initial.redshift}")
    
    # 3. 演化宇宙
    evolution = cosmos.evolve(end_redshift=0, time_step="0.5Gyr")
    print(f"3. Evolved to z=0")
    
    # 4. 获取状态
    state = evolution.get_state(redshift=0, scale="cosmic_web")
    print(f"4. Current age: {state['age']:.2f} Gyr")
    
    # 5. 创建一些星系
    galaxy_sim = cosmos.galaxy_formation
    for mass in [1e10, 1e11, 1e12]:
        halo = galaxy_sim.create_collapsed_halo(mass, z=0)
        print(f"5. Created galaxy halo: M={halo.mass:.2e} M_sun")
    
    # 6. 创建一些恒星
    stellar_sim = cosmos.stellar_evolution
    for mass in [1, 5, 10]:
        star = stellar_sim.create_star(mass, metallicity=0.02)
        print(f"6. Created star: M={star.mass} M_sun ({star.stellar_type.value})")
    
    # 7. 获取宇宙学参数
    params = cosmos.cosmology.get_cosmological_parameters()
    print(f"7. H0={params['H0']}, Ω_m={params['Omega_m']}, Ω_Λ={params['Omega_Lambda']}")
    
    # 8. 计算CMB
    cmb = cosmos.cosmology.compute_cmb_power_spectrum()
    print(f"8. CMB TT peak at l~{cmb.ell[np.argmax(cmb.C_ell_tt)]:.0f}")
    
    return cosmos


def run_all_examples():
    """运行所有示例"""
    examples = [
        ("Basic Simulation", example_1_basic_simulation),
        ("Galaxy Formation", example_2_galaxy_formation),
        ("Stellar Evolution", example_3_stellar_evolution),
        ("Cosmology", example_4_cosmology),
        ("API Usage", example_5_api_usage),
        ("Full Workflow", example_6_full_workflow),
    ]
    
    results = []
    for name, func in examples:
        try:
            result = func()
            results.append((name, True, None))
        except Exception as e:
            print(f"\nError in {name}: {e}")
            results.append((name, False, str(e)))
    
    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)
    for name, success, error in results:
        status = "✓" if success else "✗"
        print(f"  {status} {name}")
        if error:
            print(f"    Error: {error}")
    
    return results


if __name__ == "__main__":
    run_all_examples()
