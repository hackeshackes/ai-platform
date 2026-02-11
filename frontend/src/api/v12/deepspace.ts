/**
 * 深空探测 API
 * Deep Space Exploration API
 */

// ==================== 类型定义 ====================

/**
 * 深空探测任务
 */
export interface DeepSpaceMission {
  /** 任务ID */
  id: string;
  /** 任务名称 */
  name: string;
  /** 目标天体 */
  target: string;
  /** 任务类型 */
  type: 'flyby' | 'orbit' | 'landing' | 'rover' | 'sample-return';
  /** 发射日期 */
  launchDate: string;
  /** 预计到达日期 */
  arrivalDate?: string;
  /** 状态 */
  status: 'planned' | 'in-transit' | 'active' | 'completed' | 'failed';
  /** 航天器信息 */
  spacecraft?: {
    name: string;
    mass: number;
    powerSource: string;
    instruments: string[];
  };
  /** 科学目标 */
  objectives?: string[];
}

/**
 * 天体数据
 */
export interface CelestialBody {
  /** 天体ID */
  id: string;
  /** 名称 */
  name: string;
  /** 类型 */
  type: 'planet' | 'moon' | 'asteroid' | 'comet' | 'dwarf-planet' | 'star';
  /** 父天体（如果是卫星） */
  parent?: string;
  /** 半径（km） */
  radius: number;
  /** 质量（kg） */
  mass: number;
  /** 平均温度（K） */
  temperature: number;
  /** 轨道参数 */
  orbit?: {
    semiMajorAxis: number;
    eccentricity: number;
    period: number;
    inclination: number;
  };
  /** 物理特征 */
  features?: {
    atmosphere?: {
      composition: Record<string, number>;
      pressure: number;
    };
    surface?: {
      composition: Record<string, number>;
      terrain: string;
    };
  };
  /** 图像URL */
  imageUrl?: string;
}

/**
 * 深空探测数据
 */
export interface DeepSpaceData {
  /** 数据ID */
  id: string;
  /** 来源任务 */
  missionId: string;
  /** 数据类型 */
  dataType: 'image' | 'spectra' | 'telemetry' | 'scientific';
  /** 采集时间 */
  timestamp: string;
  /** 目标天体 */
  target: string;
  /** 数据内容 */
  content: Record<string, unknown>;
  /** 数据大小 */
  size: number;
  /** 下载URL */
  downloadUrl?: string;
}

/**
 * 深空通信状态
 */
export interface DeepSpaceCommunication {
  /** 通信ID */
  id: string;
  /** 任务ID */
  missionId: string;
  /** 信号强度（dBm） */
  signalStrength: number;
  /** 延迟（分钟） */
  latency: number;
  /** 数据传输速率（bps） */
  dataRate: number;
  /** 状态 */
  status: 'excellent' | 'good' | 'fair' | 'poor';
}

// ==================== API 客户端 ====================

const DEEPSPACE_API_BASE = '/api/v12/deepspace';

/**
 * 深空探测 API 客户端
 */
export class DeepSpaceAPI {
  /**
   * 获取深空任务列表
   */
  static async getMissions(
    filters?: {
      status?: string;
      target?: string;
      type?: string;
    }
  ): Promise<DeepSpaceMission[]> {
    const params = new URLSearchParams(filters || {});
    const response = await fetch(`${DEEPSPACE_API_BASE}/missions?${params}`);

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.message || `HTTP ${response.status}: 获取任务列表失败`);
    }

    return response.json();
  }

  /**
   * 获取任务详情
   */
  static async getMissionDetails(id: string): Promise<DeepSpaceMission> {
    const response = await fetch(`${DEEPSPACE_API_BASE}/missions/${id}`);

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.message || `HTTP ${response.status}: 获取任务详情失败`);
    }

    return response.json();
  }

  /**
   * 获取天体数据
   */
  static async getCelestialBody(
    id: string | 'all',
    filters?: {
      type?: string;
      parent?: string;
    }
  ): Promise<CelestialBody | CelestialBody[]> {
    const params = new URLSearchParams(filters || {});
    const response = await fetch(`${DEEPSPACE_API_BASE}/bodies/${id}?${params}`);

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.message || `HTTP ${response.status}: 获取天体数据失败`);
    }

    return response.json();
  }

  /**
   * 搜索天体
   */
  static async searchBodies(
    query: string,
    limit: number = 10
  ): Promise<CelestialBody[]> {
    const params = new URLSearchParams({ q: query, limit: limit.toString() });
    const response = await fetch(`${DEEPSPACE_API_BASE}/bodies/search?${params}`);

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.message || `HTTP ${response.status}: 搜索天体失败`);
    }

    return response.json();
  }

  /**
   * 获取探测数据
   */
  static async getMissionData(
    missionId: string,
    dataType?: string,
    page: number = 1,
    limit: number = 20
  ): Promise<{
    data: DeepSpaceData[];
    total: number;
  }> {
    const params = new URLSearchParams({
      page: page.toString(),
      limit: limit.toString(),
      ...(dataType && { dataType }),
    });
    const response = await fetch(`${DEEPSPACE_API_BASE}/missions/${missionId}/data?${params}`);

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.message || `HTTP ${response.status}: 获取探测数据失败`);
    }

    return response.json();
  }

  /**
   * 获取通信状态
   */
  static async getCommunicationStatus(missionId: string): Promise<DeepSpaceCommunication> {
    const response = await fetch(`${DEEPSPACE_API_BASE}/missions/${missionId}/communication`);

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.message || `HTTP ${response.status}: 获取通信状态失败`);
    }

    return response.json();
  }

  /**
   * 计算轨道
   */
  static async calculateOrbit(
    bodyId: string,
    targetDate: string
  ): Promise<{
    position: [number, number, number];
    velocity: [number, number, number];
    altitude: number;
  }> {
    const response = await fetch(`${DEEPSPACE_API_BASE}/bodies/${bodyId}/orbit`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ targetDate }),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.message || `HTTP ${response.status}: 计算轨道失败`);
    }

    return response.json();
  }
}

// 导出默认实例
export const deepspace = new DeepSpaceAPI();
