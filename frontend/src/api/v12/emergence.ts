/**
 * 涌现能力 API
 * Emergence Capability API
 */

// ==================== 类型定义 ====================

/**
 * 涌现模拟请求
 */
export interface EmergenceSimulationRequest {
  /** 系统类型 */
  systemType: 'neural' | 'multi-agent' | 'complex-adaptive' | 'swarm';
  /** 涌现目标 */
  emergenceGoal: string;
  /** 系统参数 */
  systemParams?: {
    agentCount: number;
    interactionRadius: number;
    learningRate: number;
    environmentSize: number;
  };
  /** 观测指标 */
  metrics?: string[];
}

/**
 * 涌现模拟结果
 */
export interface EmergenceSimulationResult {
  /** 模拟ID */
  id: string;
  /** 状态 */
  status: 'pending' | 'running' | 'completed' | 'failed';
  /** 涌现现象检测 */
  emergenceDetected: boolean;
  /** 涌现类型 */
  emergenceType?: string;
  /** 涌现强度 */
  emergenceIntensity?: number;
  /** 系统状态历史 */
  systemHistory: {
    timestamp: number;
    globalState: Record<string, number>;
    agentStates: Record<string, unknown>[];
  }[];
  /** 涌现指标 */
  metrics?: {
    name: string;
    values: number[];
    trend: 'increasing' | 'decreasing' | 'stable';
    criticalPoint?: number;
  }[];
  /** 解释 */
  explanation?: string;
}

/**
 * 复杂适应系统
 */
export interface ComplexAdaptiveSystem {
  /** 系统ID */
  id: string;
  /** 系统名称 */
  name: string;
  /** 系统类型 */
  type: string;
  /** 代理数量 */
  agentCount: number;
  /** 交互网络 */
  interactionNetwork: {
    nodes: string[];
    edges: [string, string, number][];
  };
  /** 适应规则 */
  adaptationRules: {
    trigger: string;
    action: string;
    conditions: Record<string, unknown>;
  }[];
  /** 涌现行为 */
  emergentBehaviors: {
    name: string;
    description: string;
    confidence: number;
  }[];
}

/**
 * 涌现模式
 */
export interface EmergencePattern {
  /** 模式ID */
  id: string;
  /** 模式名称 */
  name: string;
  /** 模式类别 */
  category: 'spatial' | 'temporal' | 'behavioral' | 'structural';
  /** 关键特征 */
  keyFeatures: string[];
  /** 检测条件 */
  detectionConditions: Record<string, unknown>;
  /** 相关案例 */
  relatedCases: string[];
  /** 发生概率 */
  probability: number;
}

/**
 * 自组织分析
 */
export interface SelfOrganizationAnalysis {
  /** 分析ID */
  id: string;
  /** 系统状态 */
  systemState: {
    orderParameter: number;
    entropy: number;
    correlationLength: number;
  };
  /** 自组织临界性 */
  criticality: {
    isCritical: boolean;
    criticalExponent: number;
    avalancheStatistics: {
      sizeDistribution: Record<string, number>;
      durationDistribution: Record<string, number>;
    };
  };
  /** 相变分析 */
  phaseTransition: {
    currentPhase: string;
    transitionPoint?: number;
    orderParameterTrajectory: number[];
  };
}

// ==================== API 客户端 ====================

const EMERGENCE_API_BASE = '/api/v12/emergence';

/**
 * 涌现能力 API 客户端
 */
export class EmergenceAPI {
  /**
   * 启动涌现模拟
   */
  static async startSimulation(
    request: EmergenceSimulationRequest
  ): Promise<EmergenceSimulationResult> {
    const response = await fetch(`${EMERGENCE_API_BASE}/simulate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.message || `HTTP ${response.status}: 涌现模拟启动失败`);
    }

    return response.json();
  }

  /**
   * 获取模拟状态
   */
  static async getSimulationStatus(id: string): Promise<EmergenceSimulationResult> {
    const response = await fetch(`${EMERGENCE_API_BASE}/simulate/${id}`);

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.message || `HTTP ${response.status}: 获取模拟状态失败`);
    }

    return response.json();
  }

  /**
   * 检测涌现现象
   */
  static async detectEmergence(
    systemType: string,
    stateHistory: Record<string, unknown>[],
    thresholds?: Record<string, number>
  ): Promise<{
    emergenceEvents: {
      type: string;
      timestamp: number;
      intensity: number;
      description: string;
    }[];
    emergenceScore: number;
  }> {
    const response = await fetch(`${EMERGENCE_API_BASE}/detect`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ systemType, stateHistory, thresholds }),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.message || `HTTP ${response.status}: 涌现检测失败`);
    }

    return response.json();
  }

  /**
   * 创建复杂适应系统
   */
  static async createSystem(
    config: Omit<ComplexAdaptiveSystem, 'id'>
  ): Promise<ComplexAdaptiveSystem> {
    const response = await fetch(`${EMERGENCE_API_BASE}/systems`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(config),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.message || `HTTP ${response.status}: 创建系统失败`);
    }

    return response.json();
  }

  /**
   * 获取系统详情
   */
  static async getSystemDetails(id: string): Promise<ComplexAdaptiveSystem> {
    const response = await fetch(`${EMERGENCE_API_BASE}/systems/${id}`);

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.message || `HTTP ${response.status}: 获取系统详情失败`);
    }

    return response.json();
  }

  /**
   * 分析自组织
   */
  static async analyzeSelfOrganization(
    systemId: string
  ): Promise<SelfOrganizationAnalysis> {
    const response = await fetch(`${EMERGENCE_API_BASE}/systems/${systemId}/self-organization`);

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.message || `HTTP ${response.status}: 自组织分析失败`);
    }

    return response.json();
  }

  /**
   * 搜索涌现模式
   */
  static async searchPatterns(
    query: string,
    category?: string
  ): Promise<EmergencePattern[]> {
    const params = new URLSearchParams({ q: query, ...(category && { category }) });
    const response = await fetch(`${EMERGENCE_API_BASE}/patterns?${params}`);

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.message || `HTTP ${response.status}: 搜索模式失败`);
    }

    return response.json();
  }

  /**
   * 模拟群体智能
   */
  static async simulateSwarmIntelligence(
    agentCount: number,
    environmentSize: number,
    taskType: 'foraging' | 'pathfinding' | 'clustering' | 'nest-building',
    iterations: number
  ): Promise<{
    finalCollectiveState: Record<string, unknown>;
    swarmPerformance: {
      efficiency: number;
      robustness: number;
      adaptability: number;
    };
    emergentStrategies: string[];
  }> {
    const response = await fetch(`${EMERGENCE_API_BASE}/swarm`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ agentCount, environmentSize, taskType, iterations }),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.message || `HTTP ${response.status}: 群体智能模拟失败`);
    }

    return response.json();
  }
}

// 导出默认实例
export const emergence = new EmergenceAPI();
