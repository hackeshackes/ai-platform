/**
 * 气候模型 API
 * Climate Model API
 */

// ==================== 类型定义 ====================

/**
 * 气候模拟请求
 */
export interface ClimateSimulationRequest {
  /** 模拟区域 */
  region: string;
  /** 起始时间 */
  startTime: string;
  /** 结束时间 */
  endTime: string;
  /** 气候模型类型 */
  modelType: 'global' | 'regional' | 'local';
  /** 分辨率 */
  resolution?: number;
  /** 变量列表 */
  variables?: string[];
}

/**
 * 气候模拟结果
 */
export interface ClimateSimulationResult {
  /** 模拟ID */
  id: string;
  /** 状态 */
  status: 'pending' | 'running' | 'completed' | 'failed';
  /** 模拟区域 */
  region: string;
  /** 时间范围 */
  timeRange: {
    start: string;
    end: string;
  };
  /** 结果数据 */
  data?: {
    /** 温度数据 */
    temperature: number[];
    /** 降水数据 */
    precipitation: number[];
    /** 风速数据 */
    windSpeed: number[];
    /** 湿度数据 */
    humidity: number[];
  };
  /** 错误信息 */
  error?: string;
}

/**
 * 气候预测数据
 */
export interface ClimateForecast {
  /** 预测ID */
  id: string;
  /** 预测时间 */
  forecastTime: string;
  /** 目标时间 */
  targetTime: string;
  /** 预测类型 */
  type: 'short-term' | 'medium-term' | 'long-term';
  /** 预测值 */
  predictions: {
    temperature: {
      mean: number;
      min: number;
      max: number;
    };
    precipitation: {
      mean: number;
      probability: number;
    };
    extremeWeather: {
      type: string;
      probability: number;
    }[];
  };
  /** 置信度 */
  confidence: number;
}

// ==================== API 客户端 ====================

const CLIMATE_API_BASE = '/api/v12/climate';

/**
 * 气候模型 API 客户端
 */
export class ClimateAPI {
  /**
   * 创建气候模拟任务
   */
  static async createSimulation(
    params: ClimateSimulationRequest
  ): Promise<ClimateSimulationResult> {
    const response = await fetch(`${CLIMATE_API_BASE}/simulations`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(params),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.message || `HTTP ${response.status}: 气候模拟创建失败`);
    }

    return response.json();
  }

  /**
   * 获取模拟任务状态
   */
  static async getSimulationStatus(id: string): Promise<ClimateSimulationResult> {
    const response = await fetch(`${CLIMATE_API_BASE}/simulations/${id}`);

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.message || `HTTP ${response.status}: 获取模拟状态失败`);
    }

    return response.json();
  }

  /**
   * 获取气候预测
   */
  static async getForecast(
    region: string,
    type: 'short-term' | 'medium-term' | 'long-term' = 'short-term'
  ): Promise<ClimateForecast[]> {
    const params = new URLSearchParams({ region, type });
    const response = await fetch(`${CLIMATE_API_BASE}/forecasts?${params}`);

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.message || `HTTP ${response.status}: 获取气候预测失败`);
    }

    return response.json();
  }

  /**
   * 获取历史气候数据
   */
  static async getHistoricalData(
    region: string,
    startTime: string,
    endTime: string
  ): Promise<{
    timestamps: string[];
    temperature: number[];
    precipitation: number[];
  }> {
    const params = new URLSearchParams({ region, startTime, endTime });
    const response = await fetch(`${CLIMATE_API_BASE}/historical?${params}`);

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.message || `HTTP ${response.status}: 获取历史数据失败`);
    }

    return response.json();
  }

  /**
   * 分析气候趋势
   */
  static async analyzeTrends(
    region: string,
    period: 'yearly' | 'monthly' | 'decade'
  ): Promise<{
    trend: 'warming' | 'cooling' | 'stable';
    rate: number;
    prediction: {
      year: number;
      temperature: number;
    }[];
  }> {
    const params = new URLSearchParams({ region, period });
    const response = await fetch(`${CLIMATE_API_BASE}/trends?${params}`);

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.message || `HTTP ${response.status}: 分析气候趋势失败`);
    }

    return response.json();
  }
}

// 导出默认实例
export const climate = new ClimateAPI();
