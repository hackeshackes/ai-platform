"""
气候模型配置
"""

# 模拟分辨率
RESOLUTION = "1km"

# 模拟区域
DEFAULT_REGION = "global"

# 时间设置
DEFAULT_START_YEAR = 2020
DEFAULT_END_YEAR = 2100
TIME_STEP = 1  # 年

# 气候情景
SCENARIOS = {
    "RCP2.6": "强减排情景",
    "RCP4.5": "中等减排情景", 
    "RCP6.0": "中等排放情景",
    "RCP8.5": "高排放情景"
}

# 模拟精度目标
TARGET_ACCURACY = 0.95
PREDICTION_ACCURACY_TARGET = 0.90

# 物理常数
SOLAR_CONSTANT = 1361  # W/m²
STEFAN_BOLTZMANN = 5.67e-8  # W/(m²·K⁴)
EARTH_RADIUS = 6371000  # m
EARTH_SURFACE_AREA = 4 * 3.14159 * (EARTH_RADIUS ** 2)

# 空间参数
GRID_SIZE = 360  # 经度方向
LAT_BANDS = 180  # 纬度方向
CELL_AREA = EARTH_SURFACE_AREA / (GRID_SIZE * LAT_BANDS)

# 碳循环参数
CO2_CONVERSION_FACTOR = 2.12  # ppm to GtC
CARBON_RESERVOIRS = ['atmosphere', 'land', 'ocean']

# 反馈机制参数
WATER_VAPOR_FEEDBACK = 1.5
CLOUD_FEEDBACK = 0.5
ICE_ALBEDO_FEEDBACK = 0.3
LAPSED_RATE = 6.5  # °C/km
