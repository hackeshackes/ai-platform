/**
 * 生物模拟 API
 * Biological Simulation API
 */

// ==================== 类型定义 ====================

/**
 * 生物模拟请求
 */
export interface BioSimulationRequest {
  /** 模拟类型 */
  simulationType: 'evolution' | 'ecosystem' | 'genetics' | 'population';
  /** 生物种群 */
  species?: string[];
  /** 环境参数 */
  environment?: {
    temperature: number;
    humidity: number;
    resources: number;
    predators?: string[];
  };
  /** 模拟时长（代/时间单位） */
  duration: number;
  /** 初始数量 */
  initialPopulation?: number;
}

/**
 * 生物模拟结果
 */
export interface BioSimulationResult {
  /** 模拟ID */
  id: string;
  /** 状态 */
  status: 'pending' | 'running' | 'completed' | 'failed';
  /** 模拟类型 */
  type: string;
  /** 结果数据 */
  data?: {
    /** 种群数量变化 */
    populationHistory: {
      species: string;
      count: number[];
      timestamps: string[];
    }[];
    /** 进化历史 */
    evolutionHistory: {
      species: string;
      traits: {
        name: string;
        value: number;
        frequency: number;
      }[];
    }[];
    /** 生态关系 */
    ecologicalRelationships: {
      type: 'predation' | 'symbiosis' | 'competition';
      species1: string;
      species2: string;
      strength: number;
    }[];
    /** 灭绝事件 */
    extinctionEvents?: {
      species: string;
      time: string;
      reason: string;
    }[];
  };
  /** 统计摘要 */
  summary?: {
    finalPopulation: Record<string, number>;
    biodiversityIndex: number;
    dominantSpecies: string;
  };
  /** 错误信息 */
  error?: string;
}

/**
 * 基因序列数据
 */
export interface GeneticSequence {
  /** 序列ID */
  id: string;
  /** 物种名称 */
  species: string;
  /** 基因名称 */
  gene: string;
  /** 序列数据 */
  sequence: string;
  /** 序列长度 */
  length: number;
  /** 注释 */
  annotations?: {
    position: number;
    type: string;
    description: string;
  }[];
}

/**
 * 生态系统分析
 */
export interface EcosystemAnalysis {
  /** 分析ID */
  id: string;
  /** 生态系统名称 */
  name: string;
  /** 生物多样性指数 */
  biodiversityIndex: number;
  /** 食物网结构 */
  foodWeb: {
    nodes: {
      id: string;
      name: string;
      trophicLevel: number;
    }[];
    edges: {
      from: string;
      to: string;
      type: string;
    }[];
  };
  /** 能量流动 */
  energyFlow: {
    totalEnergy: number;
    efficiency: number;
    trophicTransfer: number[];
  };
  /** 稳定性评估 */
  stability: {
    resistance: number;
    resilience: number;
    overall: number;
  };
}

// ==================== API 客户端 ====================

const BIO_API_BASE = '/api/v12/bio';

/**
 * 生物模拟 API 客户端
 */
export class BioAPI {
  /**
   * 创建生物模拟任务
   */
  static async createSimulation(
    params: BioSimulationRequest
  ): Promise<BioSimulationResult> {
    const response = await fetch(`${BIO_API_BASE}/simulations`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(params),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.message || `HTTP ${response.status}: 生物模拟创建失败`);
    }

    return response.json();
  }

  /**
   * 获取模拟任务状态
   */
  static async getSimulationStatus(id: string): Promise<BioSimulationResult> {
    const response = await fetch(`${BIO_API_BASE}/simulations/${id}`);

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.message || `HTTP ${response.status}: 获取模拟状态失败`);
    }

    return response.json();
  }

  /**
   * 分析基因序列
   */
  static async analyzeGeneSequence(
    sequence: string,
    species: string
  ): Promise<{
    gcContent: number;
    codons: Record<string, number>;
    proteins: {
      start: number;
      end: number;
      sequence: string;
      translation: string;
    }[];
    mutations?: {
      position: number;
      original: string;
      variant: string;
      effect: string;
    }[];
  }> {
    const response = await fetch(`${BIO_API_BASE}/genetics/analyze`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ sequence, species }),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.message || `HTTP ${response.status}: 基因序列分析失败`);
    }

    return response.json();
  }

  /**
   * 获取生态系统分析
   */
  static async analyzeEcosystem(
    ecosystemId: string
  ): Promise<EcosystemAnalysis> {
    const response = await fetch(`${BIO_API_BASE}/ecosystems/${ecosystemId}`);

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.message || `HTTP ${response.status}: 生态系统分析失败`);
    }

    return response.json();
  }

  /**
   * 模拟进化过程
   */
  static async simulateEvolution(
    species: string[],
    generations: number,
    mutationRate: number = 0.001
  ): Promise<{
    generations: {
      id: number;
      population: number;
      fitness: number;
      adaptations: string[];
    }[];
    evolutionaryTree: {
      parent: string;
      child: string;
      mutation: string;
    }[];
  }> {
    const response = await fetch(`${BIO_API_BASE}/evolution`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ species, generations, mutationRate }),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.message || `HTTP ${response.status}: 进化模拟失败`);
    }

    return response.json();
  }
}

// 导出默认实例
export const bio = new BioAPI();
