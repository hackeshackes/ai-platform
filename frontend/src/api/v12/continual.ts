/**
 * 持续学习 API
 * Continual Learning API
 */

// ==================== 类型定义 ====================

/**
 * 持续学习任务
 */
export interface ContinualLearningTask {
  /** 任务ID */
  id: string;
  /** 任务名称 */
  name: string;
  /** 任务序列 */
  sequence: number;
  /** 数据集 */
  dataset: string;
  /** 任务类型 */
  taskType: 'classification' | 'regression' | 'generation' | 'rl';
  /** 类别数 */
  numClasses?: number;
  /** 训练样本数 */
  trainSamples: number;
  /** 测试样本数 */
  testSamples: number;
  /** 任务难度 */
  difficulty: 'easy' | 'medium' | 'hard';
}

/**
 * 持续学习配置
 */
export interface ContinualLearningConfig {
  /** 记忆策略 */
  memoryStrategy: 'replay' | 'regularization' | 'architecture' | 'knowledge-distillation';
  /** 回放缓冲区大小 */
  replayBufferSize?: number;
  /** 正则化强度 */
  regularizationStrength?: number;
  /** 知识蒸馏温度 */
  distillationTemperature?: number;
  /** EWC Fisher矩阵更新频率 */
  fisherUpdateFrequency?: number;
  /** 网络架构策略 */
  architectureStrategy?: 'progressive' | 'packnet' | 'lwf';
}

/**
 * 持续学习结果
 */
export interface ContinualLearningResult {
  /** 会话ID */
  id: string;
  /** 状态 */
  status: 'pending' | 'training' | 'completed' | 'failed';
  /** 使用策略 */
  strategy: string;
  /** 任务序列 */
  taskSequence: string[];
  /** 性能指标 */
  metrics: {
    /** 每任务准确率 */
    perTaskAccuracy: {
      task: string;
      accuracy: number;
    }[];
    /** 平均准确率 */
    averageAccuracy: number;
    /** 前向迁移 */
    forwardTransfer: number;
    /** 后向迁移 */
    backwardTransfer: number;
    /** 遗忘率 */
    forgetting: number;
  };
  /** 记忆使用情况 */
  memoryUsage?: {
    samples: number;
    percentage: number;
  };
  /** 灾难性遗忘指标 */
  forgettingMetrics?: {
    task: string;
    beforeAccuracy: number;
    afterAccuracy: number;
    forgettingAmount: number;
  }[];
}

/**
 * 遗忘分析
 */
export interface ForgettingAnalysis {
  /** 分析ID */
  id: string;
  /** 任务对 */
  taskPairs: {
    learnedTask: string;
    affectedTask: string;
  }[];
  /** 遗忘热点 */
  forgettingHotspots: {
    layer: number;
    neuron: number;
    forgettingScore: number;
  }[];
  /** 补救建议 */
  recommendations: {
    method: string;
    expectedImprovement: number;
    implementation: string;
  }[];
}

/**
 * 经验回放样本
 */
export interface ReplaySample {
  /** 样本ID */
  id: string;
  /** 任务来源 */
  task: string;
  /** 输入 */
  input: number[] | string;
  /** 标签 */
  label: string | number;
  /** 重要性分数 */
  importanceScore: number;
  /** 回放优先级 */
  priority: number;
  /** 最后回放时间 */
  lastReplay?: string;
}

// ==================== API 客户端 ====================

const CONTINUAL_API_BASE = '/api/v12/continual';

/**
 * 持续学习 API 客户端
 */
export class ContinualAPI {
  /**
   * 配置持续学习
   */
  static async configure(
    tasks: ContinualLearningTask[],
    config: ContinualLearningConfig
  ): Promise<{ sessionId: string }> {
    const response = await fetch(`${CONTINUAL_API_BASE}/configure`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ tasks, config }),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.message || `HTTP ${response.status}: 持续学习配置失败`);
    }

    return response.json();
  }

  /**
   * 训练单个任务
   */
  static async trainTask(
    sessionId: string,
    taskId: string,
    epochs: number = 10
  ): Promise<{
    taskId: string;
    accuracy: number;
    loss: number;
    trainingTime: number;
  }> {
    const response = await fetch(`${CONTINUAL_API_BASE}/sessions/${sessionId}/train`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ taskId, epochs }),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.message || `HTTP ${response.status}: 任务训练失败`);
    }

    return response.json();
  }

  /**
   * 获取训练结果
   */
  static async getTrainingResult(sessionId: string): Promise<ContinualLearningResult> {
    const response = await fetch(`${CONTINUAL_API_BASE}/sessions/${sessionId}`);

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.message || `HTTP ${response.status}: 获取训练结果失败`);
    }

    return response.json();
  }

  /**
   * 分析遗忘情况
   */
  static async analyzeForgetting(
    sessionId: string
  ): Promise<ForgettingAnalysis> {
    const response = await fetch(`${CONTINUAL_API_BASE}/sessions/${sessionId}/forgetting`);

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.message || `HTTP ${response.status}: 遗忘分析失败`);
    }

    return response.json();
  }

  /**
   * 获取经验回放样本
   */
  static async getReplaySamples(
    sessionId: string,
    count: number,
    strategy: 'random' | 'priority' | 'reservoir' = 'reservoir'
  ): Promise<ReplaySample[]> {
    const params = new URLSearchParams({ count: count.toString(), strategy });
    const response = await fetch(`${CONTINUAL_API_BASE}/sessions/${sessionId}/replay?${params}`);

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.message || `HTTP ${response.status}: 获取回放样本失败`);
    }

    return response.json();
  }

  /**
   * 更新EWC Fisher矩阵
   */
  static async updateFisherMatrix(
    sessionId: string,
    taskId: string
  ): Promise<{
    matrixSize: number;
    updateTime: number;
  }> {
    const response = await fetch(`${CONTINUAL_API_BASE}/sessions/${sessionId}/fisher`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ taskId }),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.message || `HTTP ${response.status}: Fisher矩阵更新失败`);
    }

    return response.json();
  }

  /**
   * 执行知识蒸馏
   */
  static async distillKnowledge(
    sessionId: string,
    taskId: string,
    temperature: number = 2.0
  ): Promise<{
    distillationLoss: number;
    studentAccuracy: number;
    teacherStudentGap: number;
  }> {
    const response = await fetch(`${CONTINUAL_API_BASE}/sessions/${sessionId}/distill`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ taskId, temperature }),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.message || `HTTP ${response.status}: 知识蒸馏失败`);
    }

    return response.json();
  }

  /**
   * 评估持续学习性能
   */
  static async evaluatePerformance(
    sessionId: string,
    evaluationTasks?: string[]
  ): Promise<{
    overallMetrics: {
      averageAccuracy: number;
      averageForgetting: number;
      transferEfficiency: number;
    };
    taskWiseMetrics: {
      task: string;
      accuracy: number;
      forgettingSince: Record<string, number>;
    }[];
    comparison: {
      baseline: number;
      ours: number;
      improvement: number;
    };
  }> {
    const response = await fetch(`${CONTINUAL_API_BASE}/sessions/${sessionId}/evaluate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ evaluationTasks }),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.message || `HTTP ${response.status}: 性能评估失败`);
    }

    return response.json();
  }
}

// 导出默认实例
export const continual = new ContinualAPI();
