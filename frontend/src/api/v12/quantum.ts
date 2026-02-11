/**
 * 量子模块 API
 * Quantum Module API
 */

// ==================== 类型定义 ====================

/**
 * 量子计算请求
 */
export interface QuantumComputingRequest {
  /** 计算类型 */
  type: 'circuit' | 'annealing' | 'simulation';
  /** 量子比特数 */
  qubits: number;
  /** 量子电路（如果适用） */
  circuit?: {
    gates: {
      name: string;
      target: number[];
      parameters?: number[];
    }[];
    measurements: number[];
  };
  /** 优化参数 */
  parameters?: Record<string, unknown>;
}

/**
 * 量子计算结果
 */
export interface QuantumComputingResult {
  /** 计算ID */
  id: string;
  /** 状态 */
  status: 'pending' | 'running' | 'completed' | 'failed';
  /** 使用的量子比特数 */
  qubits: number;
  /** 结果数据 */
  result?: {
    /** 测量结果 */
    measurements: {
      state: string;
      probability: number;
      count: number;
    }[];
    /** 执行时间（微秒） */
    executionTime: number;
    /** 量子比特保真度 */
    fidelity: number;
  };
  /** 错误信息 */
  error?: string;
}

/**
 * 量子态数据
 */
export interface QuantumState {
  /** 态ID */
  id: string;
  /** 态名称 */
  name: string;
  /** 量子比特数 */
  qubits: number;
  /** 态向量 */
  stateVector: ComplexNumber[];
  /** 密度矩阵 */
  densityMatrix?: ComplexNumber[][];
  /** 纯态/混态 */
  isPure: boolean;
}

/**
 * 复数类型
 */
export interface ComplexNumber {
  /** 实部 */
  re: number;
  /** 虚部 */
  im: number;
}

/**
 * 量子纠缠数据
 */
export interface QuantumEntanglement {
  /** 纠缠ID */
  id: string;
  /** 纠缠粒子对 */
  particles: [string, string];
  /** 纠缠类型 */
  type: 'bell' | 'ghz' | 'w' | 'cluster';
  /** 纠缠态 */
  state: string;
  /** 纠缠度 */
  entanglementDegree: number;
  /** 创建时间 */
  createdAt: string;
  /** 存活时间（微秒） */
  lifetime: number;
}

/**
 * 量子密钥分发
 */
export interface QuantumKeyDistribution {
  /** 会话ID */
  id: string;
  /** 密钥长度（bits） */
  keyLength: number;
  /** 协议类型 */
  protocol: 'bb84' | 'b92' | 'e91';
  /** 分发状态 */
  status: 'preparing' | 'transmitting' | 'sifting' | 'error-correction' | 'privacy-amplification' | 'completed';
  /** 错误率（%） */
  errorRate: number;
  /** 密钥 */
  key?: string;
  /** 安全密钥长度 */
  secureKeyLength: number;
}

// ==================== API 客户端 ====================

const QUANTUM_API_BASE = '/api/v12/quantum';

/**
 * 量子模块 API 客户端
 */
export class QuantumAPI {
  /**
   * 提交量子计算任务
   */
  static async submitComputation(
    request: QuantumComputingRequest
  ): Promise<QuantumComputingResult> {
    const response = await fetch(`${QUANTUM_API_BASE}/compute`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.message || `HTTP ${response.status}: 量子计算提交失败`);
    }

    return response.json();
  }

  /**
   * 获取计算结果
   */
  static async getComputationResult(id: string): Promise<QuantumComputingResult> {
    const response = await fetch(`${QUANTUM_API_BASE}/compute/${id}`);

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.message || `HTTP ${response.status}: 获取计算结果失败`);
    }

    return response.json();
  }

  /**
   * 创建量子态
   */
  static async createState(
    name: string,
    qubits: number,
    stateType: 'random' | 'ghz' | 'w' | 'bell' | 'custom',
    customState?: number[]
  ): Promise<QuantumState> {
    const response = await fetch(`${QUANTUM_API_BASE}/states`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ name, qubits, stateType, customState }),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.message || `HTTP ${response.status}: 创建量子态失败`);
    }

    return response.json();
  }

  /**
   * 获取量子态
   */
  static async getState(id: string): Promise<QuantumState> {
    const response = await fetch(`${QUANTUM_API_BASE}/states/${id}`);

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.message || `HTTP ${response.status}: 获取量子态失败`);
    }

    return response.json();
  }

  /**
   * 创建纠缠
   */
  static async createEntanglement(
    type: 'bell' | 'ghz' | 'w' | 'cluster',
    qubits: number
  ): Promise<QuantumEntanglement> {
    const response = await fetch(`${QUANTUM_API_BASE}/entanglement`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ type, qubits }),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.message || `HTTP ${response.status}: 创建纠缠失败`);
    }

    return response.json();
  }

  /**
   * 启动量子密钥分发
   */
  static async startKeyDistribution(
    protocol: 'bb84' | 'b92' | 'e91',
    keyLength: number
  ): Promise<QuantumKeyDistribution> {
    const response = await fetch(`${QUANTUM_API_BASE}/qkd`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ protocol, keyLength }),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.message || `HTTP ${response.status}: 启动密钥分发失败`);
    }

    return response.json();
  }

  /**
   * 获取密钥分发状态
   */
  static async getKeyDistributionStatus(id: string): Promise<QuantumKeyDistribution> {
    const response = await fetch(`${QUANTUM_API_BASE}/qkd/${id}`);

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.message || `HTTP ${response.status}: 获取密钥分发状态失败`);
    }

    return response.json();
  }

  /**
   * 量子隐形传态
   */
  static async teleport(
    stateId: string,
    targetQubit: number
  ): Promise<{
    success: boolean;
    fidelity: number;
    BellMeasurement: {
      outcome: string;
      classicalBits: string;
    };
  }> {
    const response = await fetch(`${QUANTUM_API_BASE}/teleport`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ stateId, targetQubit }),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.message || `HTTP ${response.status}: 量子隐形传态失败`);
    }

    return response.json();
  }
}

// 导出默认实例
export const quantum = new QuantumAPI();
