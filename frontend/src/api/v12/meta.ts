/**
 * 元学习 API
 * Meta Learning API
 */

// ==================== 类型定义 ====================

/**
 * 元学习任务请求
 */
export interface MetaLearningRequest {
  /** 任务类型 */
  taskType: 'few-shot' | 'zero-shot' | 'multi-task' | 'continual';
  /** 支持集 */
  supportSet?: {
    samples: {
      input: number[] | string;
      label: string;
    }[];
    classes: string[];
  };
  /** 查询集 */
  querySet?: {
    samples: {
      input: number[] | string;
    }[];
  };
  /** 模型配置 */
  modelConfig?: {
    baseModel: string;
    learningRate: number;
    innerLoopSteps: number;
    outerLoopSteps: number;
  };
}

/**
 * 元学习结果
 */
export interface MetaLearningResult {
  /** 任务ID */
  id: string;
  /** 状态 */
  status: 'pending' | 'training' | 'completed' | 'failed';
  /** 元学习策略 */
  strategy: string;
  /** 支持集准确率 */
  supportAccuracy: number;
  /** 查询集准确率 */
  queryAccuracy: number;
  /** 学习到的初始化参数 */
  learnedInitialization?: number[];
  /** 元梯度 */
  metaGradients?: Record<string, number>;
}

/**
 * MAML配置
 */
export interface MAMLConfig {
  /** 内部学习率 */
  innerLearningRate: number;
  /** 外部学习率 */
  outerLearningRate: number;
  /** 内部循环步数 */
  innerLoopSteps: number;
  /** 外部循环步数 */
  outerLoopSteps: number;
  /** 支持集大小 */
  supportSize: number;
  /** 查询集大小 */
  querySize: number;
  /** 任务数量 */
  taskBatchSize: number;
  /** 是否一阶近似 */
  firstOrder: boolean;
}

/**
 * 元学习模型
 */
export interface MetaLearningModel {
  /** 模型ID */
  id: string;
  /** 模型名称 */
  name: string;
  /** 策略类型 */
  strategy: 'maml' | 'reptile' | 'prototypical' | 'matching' | 'relation';
  /** 基础架构 */
  baseArchitecture: string;
  /** 参数量 */
  parameters: number;
  /** 支持的类别数 */
  supportedClasses: number;
  /** 训练状态 */
  trainingStatus: 'not-started' | 'training' | 'completed';
  /** 性能指标 */
  metrics?: {
    fewShot5Way: number;
    fewShot5Shot: number;
    fewShot1Shot: number;
  };
}

/**
 * 少样本学习结果
 */
export interface FewShotResult {
  /** 预测类别 */
  predictions: {
    sample: number;
    class: string;
    probability: number;
  }[];
  /** 原型向量 */
  prototypes: {
    class: string;
    vector: number[];
  }[];
  /** 距离度量 */
  distanceMetric: 'euclidean' | 'cosine' | 'manhattan';
}

// ==================== API 客户端 ====================

const META_API_BASE = '/api/v12/meta';

/**
 * 元学习 API 客户端
 */
export class MetaAPI {
  /**
   * 训练元学习模型
   */
  static async trainModel(
    request: MetaLearningRequest
  ): Promise<MetaLearningResult> {
    const response = await fetch(`${META_API_BASE}/train`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.message || `HTTP ${response.status}: 元学习训练失败`);
    }

    return response.json();
  }

  /**
   * 少样本分类
   */
  static async fewShotClassify(
    supportSet: MetaLearningRequest['supportSet'],
    querySet: MetaLearningRequest['querySet'],
    way: number,
    shot: number,
    strategy: 'prototypical' | 'matching' | 'relation' = 'prototypical'
  ): Promise<FewShotResult> {
    const response = await fetch(`${META_API_BASE}/few-shot`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ supportSet, querySet, way, shot, strategy }),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.message || `HTTP ${response.status}: 少样本分类失败`);
    }

    return response.json();
  }

  /**
   * 配置MAML训练
   */
  static async configureMAML(
    config: MAMLConfig,
    tasks: string[]
  ): Promise<{ sessionId: string }> {
    const response = await fetch(`${META_API_BASE}/maml/configure`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ config, tasks }),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.message || `HTTP ${response.status}: MAML配置失败`);
    }

    return response.json();
  }

  /**
   * 获取元学习模型列表
   */
  static async getModels(): Promise<MetaLearningModel[]> {
    const response = await fetch(`${META_API_BASE}/models`);

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.message || `HTTP ${response.status}: 获取模型列表失败`);
    }

    return response.json();
  }

  /**
   * 获取模型详情
   */
  static async getModelDetails(id: string): Promise<MetaLearningModel> {
    const response = await fetch(`${META_API_BASE}/models/${id}`);

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.message || `HTTP ${response.status}: 获取模型详情失败`);
    }

    return response.json();
  }

  /**
   * 零样本分类
   */
  static async zeroShotClassify(
    inputs: (number[] | string)[],
    classDescriptions: string[],
    baseModel: string
  ): Promise<{
    predictions: {
      input: number;
      class: string;
      probability: number;
    }[];
  }> {
    const response = await fetch(`${META_API_BASE}/zero-shot`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ inputs, classDescriptions, baseModel }),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.message || `HTTP ${response.status}: 零样本分类失败`);
    }

    return response.json();
  }

  /**
   * 跨域元学习适应
   */
  static async crossDomainAdaptation(
    sourceDomain: string,
    targetDomain: string,
    numSteps: number
  ): Promise<{
    adaptedModel: string;
    transferMetrics: {
      accuracy: number;
      loss: number;
      domainDistance: number;
    };
  }> {
    const response = await fetch(`${META_API_BASE}/cross-domain`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ sourceDomain, targetDomain, numSteps }),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.message || `HTTP ${response.status}: 跨域适应失败`);
    }

    return response.json();
  }
}

// 导出默认实例
export const meta = new MetaAPI();
