/**
 * 宇宙模拟 API
 * Cosmos Simulation API
 */

// ==================== 类型定义 ====================

/**
 * 宇宙模拟请求
 */
export interface CosmosSimulationRequest {
  /** 模拟范围 */
  scale: 'galaxy' | 'cluster' | 'universe';
  /** 时间范围（年） */
  timeRange: {
    start: number;
    end: number;
  };
  /** 物理参数 */
  physics?: {
    darkMatter: boolean;
    darkEnergy: boolean;
    gravitationalConstant: number;
  };
  /** 初始条件 */
  initialConditions?: {
    numberOfGalaxies: number;
    galaxyTypeDistribution: Record<string, number>;
  };
}

/**
 * 宇宙模拟结果
 */
export interface CosmosSimulationResult {
  /** 模拟ID */
  id: string;
  /** 状态 */
  status: 'pending' | 'running' | 'completed' | 'failed';
  /** 模拟范围 */
  scale: string;
  /** 时间范围 */
  timeRange: {
    start: number;
    end: number;
  };
  /** 结果数据 */
  data?: {
    /** 星系演化 */
    galaxyEvolution: {
      time: number;
      galaxies: {
        id: string;
        type: string;
        mass: number;
        position: [number, number, number];
      }[];
    }[];
    /** 宇宙结构形成 */
    structureFormation: {
      time: number;
      structures: {
        type: 'filament' | 'void' | 'cluster';
        size: number;
        position: [number, number, number];
      }[];
    }[];
    /** 暗物质分布 */
    darkMatterDistribution: {
      time: number;
      density: number[][][];
    }[];
  };
  /** 错误信息 */
  error?: string;
}

/**
 * 星系数据
 */
export interface GalaxyData {
  /** 星系ID */
  id: string;
  /** 星系名称 */
  name: string;
  /** 星系类型 */
  type: 'spiral' | 'elliptical' | 'irregular' | 'dwarf';
  /** 距离（光年） */
  distance: number;
  /** 质量（太阳质量） */
  mass: number;
  /** 恒星数量 */
  starCount: number;
  /** 年龄（年） */
  age: number;
  /** 坐标 */
  coordinates: {
    ra: number;
    dec: number;
    distance: number;
  };
  /** 图像URL */
  imageUrl?: string;
}

/**
 * 宇宙射线数据
 */
export interface CosmicRayData {
  /** 检测时间 */
  timestamp: string;
  /** 能量（eV） */
  energy: number;
  /** 来源方向 */
  direction: {
    ra: number;
    dec: number;
  };
  /** 粒子类型 */
  particleType: 'proton' | 'helium' | 'electron' | 'heavy';
  /** 检测站 */
  detector: string;
}

// ==================== API 客户端 ====================

const COSMOS_API_BASE = '/api/v12/cosmos';

/**
 * 宇宙模拟 API 客户端
 */
export class CosmosAPI {
  /**
   * 创建宇宙模拟任务
   */
  static async createSimulation(
    params: CosmosSimulationRequest
  ): Promise<CosmosSimulationResult> {
    const response = await fetch(`${COSMOS_API_BASE}/simulations`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(params),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.message || `HTTP ${response.status}: 宇宙模拟创建失败`);
    }

    return response.json();
  }

  /**
   * 获取模拟任务状态
   */
  static async getSimulationStatus(id: string): Promise<CosmosSimulationResult> {
    const response = await fetch(`${COSMOS_API_BASE}/simulations/${id}`);

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.message || `HTTP ${response.status}: 获取模拟状态失败`);
    }

    return response.json();
  }

  /**
   * 获取星系列表
   */
  static async getGalaxies(
    page: number = 1,
    limit: number = 20,
    filters?: {
      type?: string;
      minMass?: number;
      maxDistance?: number;
    }
  ): Promise<{
    galaxies: GalaxyData[];
    total: number;
    page: number;
  }> {
    const params = new URLSearchParams();
    params.append('page', page.toString());
    params.append('limit', limit.toString());
    if (filters?.type) params.append('type', filters.type);
    if (filters?.minMass) params.append('minMass', filters.minMass.toString());
    if (filters?.maxDistance) params.append('maxDistance', filters.maxDistance.toString());
    const response = await fetch(`${COSMOS_API_BASE}/galaxies?${params}`);

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.message || `HTTP ${response.status}: 获取星系列表失败`);
    }

    return response.json();
  }

  /**
   * 获取宇宙射线数据
   */
  static async getCosmicRays(
    startTime: string,
    endTime: string,
    minEnergy?: number
  ): Promise<CosmicRayData[]> {
    const params = new URLSearchParams({ startTime, endTime });
    if (minEnergy) params.append('minEnergy', minEnergy.toString());

    const response = await fetch(`${COSMOS_API_BASE}/cosmic-rays?${params}`);

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.message || `HTTP ${response.status}: 获取宇宙射线数据失败`);
    }

    return response.json();
  }

  /**
   * 计算宇宙膨胀
   */
  static async calculateExpansion(
    redshift: number
  ): Promise<{
    distance: number;
    age: number;
    hubbleConstant: number;
    criticalDensity: number;
  }> {
    const params = new URLSearchParams({ redshift: redshift.toString() });
    const response = await fetch(`${COSMOS_API_BASE}/expansion?${params}`);

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.message || `HTTP ${response.status}: 计算宇宙膨胀失败`);
    }

    return response.json();
  }

  /**
   * 模拟黑洞合并
   */
  static async simulateBlackHoleMerge(
    mass1: number,
    mass2: number,
    distance: number
  ): Promise<{
    mergerTime: number;
    finalMass: number;
    gravitationalWaveAmplitude: number;
    peakFrequency: number;
  }> {
    const response = await fetch(`${COSMOS_API_BASE}/black-holes/merge`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ mass1, mass2, distance }),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.message || `HTTP ${response.status}: 黑洞合并模拟失败`);
    }

    return response.json();
  }
}

// 导出默认实例
export const cosmos = new CosmosAPI();
