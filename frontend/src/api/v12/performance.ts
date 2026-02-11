/**
 * 性能优化 API
 * Performance Optimization API
 */

// ==================== 类型定义 ====================

/**
 * 性能分析请求
 */
export interface PerformanceAnalysisRequest {
  /** 分析目标 */
  target: {
    type: 'model' | 'service' | 'pipeline' | 'system';
    id: string;
    name: string;
  };
  /** 分析类型 */
  analysisType: 'bottleneck' | 'latency' | 'throughput' | 'memory' | 'comprehensive';
  /** 分析配置 */
  config?: {
    /** 采样率 */
    samplingRate?: number;
    /** 跟踪时长（秒） */
    traceDuration?: number;
    /** 指标列表 */
    metrics?: string[];
    /** 启用profiling */
    enableProfiling?: boolean;
  };
}

/**
 * 性能分析结果
 */
export interface PerformanceAnalysisResult {
  /** 分析ID */
  id: string;
  /** 状态 */
  status: 'pending' | 'analyzing' | 'completed' | 'failed';
  /** 分析类型 */
  analysisType: string;
  /** 总体评分 */
  overallScore: number;
  /** 瓶颈列表 */
  bottlenecks: {
    id: string;
    component: string;
    severity: 'critical' | 'high' | 'medium' | 'low';
    description: string;
    metrics: {
      name: string;
      value: number;
      threshold: number;
      unit: string;
    }[];
    recommendations: {
      priority: number;
      suggestion: string;
      expectedImprovement: number;
    }[];
  }[];
  /** 性能指标 */
  metrics: {
    /** 延迟统计 */
    latency: {
      p50: number;
      p90: number;
      p95: number;
      p99: number;
      unit: string;
    };
    /** 吞吐量 */
    throughput: {
      current: number;
      max: number;
      unit: string;
    };
    /** 资源使用率 */
    resourceUtilization: {
      cpu: number;
      memory: number;
      gpu?: number;
      disk?: number;
      network?: number;
    };
  };
  /** 优化建议 */
  suggestions: {
    category: string;
    items: {
      suggestion: string;
      effort: 'low' | 'medium' | 'high';
      impact: 'low' | 'medium' | 'high';
      implementation: string;
    }[];
  }[];
}

/**
 * 模型性能基准
 */
export interface ModelPerformanceBenchmark {
  /** 模型ID */
  modelId: string;
  /** 模型版本 */
  version: string;
  /** 基准测试配置 */
  benchmarkConfig: {
    dataset: string;
    batchSize: number;
    hardware: string;
    precision: 'fp32' | 'fp16' | 'int8';
  };
  /** 性能指标 */
  performance: {
    /** 推理延迟（毫秒） */
    inferenceLatency: {
      min: number;
      max: number;
      mean: number;
      p95: number;
    };
    /** 吞吐量（样本/秒） */
    throughput: number;
    /** GPU内存使用（GB） */
    gpuMemory: number;
    /** 模型大小（MB） */
    modelSize: number;
    /** 准确率（如果适用） */
    accuracy?: number;
  };
  /** 对比历史 */
  history: {
    version: string;
    date: string;
    performance: {
      latency: number;
      throughput: number;
    };
  }[];
}

/**
 * 优化配置
 */
export interface OptimizationConfig {
  /** 配置ID */
  id: string;
  /** 优化目标 */
  objective: 'latency' | 'throughput' | 'memory' | 'balanced';
  /** 优化策略 */
  strategies: {
    strategy: 'quantization' | 'pruning' | 'distillation' | 'architecture' | 'compiler';
    enabled: boolean;
    parameters: Record<string, unknown>;
  }[];
  /** 硬件目标 */
  hardwareTarget: string;
  /** 精度约束 */
  accuracyConstraint: {
    minAccuracy: number;
    tolerance: number;
  };
  /** 状态 */
  status: 'draft' | 'optimizing' | 'completed' | 'failed';
}

/**
 * 系统性能指标
 */
export interface SystemPerformanceMetrics {
  /** 采集时间 */
  timestamp: string;
  /** CPU指标 */
  cpu: {
    usage: number;
    cores: number;
    frequency: number;
    temperature?: number;
  };
  /** 内存指标 */
  memory: {
    total: number;
    used: number;
    free: number;
    swap: number;
  };
  /** GPU指标（如果有） */
  gpu?: {
    utilization: number;
    memoryUsed: number;
    memoryTotal: number;
    temperature: number;
    power: number;
  }[];
  /** 磁盘指标 */
  disk: {
    readIOPS: number;
    writeIOPS: number;
    readThroughput: number;
    writeThroughput: number;
  };
  /** 网络指标 */
  network: {
    bytesIn: number;
    bytesOut: number;
    packetsIn: number;
    packetsOut: number;
    latency: number;
  };
}

/**
 * 性能追踪
 */
export interface PerformanceTrace {
  /** 追踪ID */
  id: string;
  /** 追踪名称 */
  name: string;
  /** 开始时间 */
  startTime: string;
  /** 结束时间 */
  endTime: string;
  /** 持续时间（毫秒） */
  duration: number;
  /** 追踪事件 */
  events: {
    id: string;
    name: string;
    category: string;
    timestamp: number;
    duration: number;
    pid: number;
    tid: number;
    args?: Record<string, unknown>;
  }[];
  /** 统计摘要 */
  summary: {
    totalEvents: number;
    eventCategories: Record<string, number>;
    topSlowEvents: {
      name: string;
      count: number;
      totalDuration: number;
    }[];
  };
  /** 下载URL */
  downloadUrl?: string;
}

// ==================== API 客户端 ====================

const PERFORMANCE_API_BASE = '/api/v12/performance';

/**
 * 性能优化 API 客户端
 */
export class PerformanceAPI {
  /**
   * 启动性能分析
   */
  static async startAnalysis(
    request: PerformanceAnalysisRequest
  ): Promise<PerformanceAnalysisResult> {
    const response = await fetch(`${PERFORMANCE_API_BASE}/analyze`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.message || `HTTP ${response.status}: 性能分析启动失败`);
    }

    return response.json();
  }

  /**
   * 获取分析结果
   */
  static async getAnalysisResult(id: string): Promise<PerformanceAnalysisResult> {
    const response = await fetch(`${PERFORMANCE_API_BASE}/analyze/${id}`);

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.message || `HTTP ${response.status}: 获取分析结果失败`);
    }

    return response.json();
  }

  /**
   * 获取模型性能基准
   */
  static async getModelBenchmark(
    modelId: string,
    version?: string
  ): Promise<ModelPerformanceBenchmark> {
    const params = new URLSearchParams(version ? { version } : {});
    const response = await fetch(`${PERFORMANCE_API_BASE}/benchmarks/${modelId}?${params}`);

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.message || `HTTP ${response.status}: 获取模型基准失败`);
    }

    return response.json();
  }

  /**
   * 运行模型基准测试
   */
  static async runModelBenchmark(
    modelId: string,
    config: {
      dataset: string;
      batchSize: number;
      hardware: string;
      precision: 'fp32' | 'fp16' | 'int8';
      iterations?: number;
    }
  ): Promise<ModelPerformanceBenchmark> {
    const response = await fetch(`${PERFORMANCE_API_BASE}/benchmarks/${modelId}/run`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(config),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.message || `HTTP ${response.status}: 基准测试运行失败`);
    }

    return response.json();
  }

  /**
   * 创建优化配置
   */
  static async createOptimizationConfig(
    config: Omit<OptimizationConfig, 'id' | 'status'>
  ): Promise<OptimizationConfig> {
    const response = await fetch(`${PERFORMANCE_API_BASE}/optimizations`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(config),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.message || `HTTP ${response.status}: 创建优化配置失败`);
    }

    return response.json();
  }

  /**
   * 执行模型优化
   */
  static async executeOptimization(
    optimizationId: string
  ): Promise<{
    success: boolean;
    optimizedModelId: string;
    improvementReport: {
      metric: string;
      before: number;
      after: number;
      improvement: number;
    }[];
  }> {
    const response = await fetch(`${PERFORMANCE_API_BASE}/optimizations/${optimizationId}/execute`, {
      method: 'POST',
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.message || `HTTP ${response.status}: 优化执行失败`);
    }

    return response.json();
  }

  /**
   * 获取系统性能指标
   */
  static async getSystemMetrics(
    duration: number = 300,
    interval: number = 5
  ): Promise<SystemPerformanceMetrics[]> {
    const params = new URLSearchParams({
      duration: duration.toString(),
      interval: interval.toString(),
    });
    const response = await fetch(`${PERFORMANCE_API_BASE}/metrics/system?${params}`);

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.message || `HTTP ${response.status}: 获取系统指标失败`);
    }

    return response.json();
  }

  /**
   * 开始性能追踪
   */
  static async startTrace(
    targetType: string,
    targetId: string,
    duration: number = 30
  ): Promise<PerformanceTrace> {
    const response = await fetch(`${PERFORMANCE_API_BASE}/traces`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ targetType, targetId, duration }),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.message || `HTTP ${response.status}: 追踪启动失败`);
    }

    return response.json();
  }

  /**
   * 获取追踪结果
   */
  static async getTraceResult(traceId: string): Promise<PerformanceTrace> {
    const response = await fetch(`${PERFORMANCE_API_BASE}/traces/${traceId}`);

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.message || `HTTP ${response.status}: 获取追踪结果失败`);
    }

    return response.json();
  }

  /**
   * 生成优化建议
   */
  static async generateOptimizationSuggestions(
    targetType: string,
    targetId: string,
    goals: string[]
  ): Promise<{
    suggestions: {
      area: string;
      currentIssue: string;
      suggestedAction: string;
      expectedGain: string;
      implementationEffort: string;
    }[];
    priorityOrder: string[];
  }> {
    const response = await fetch(`${PERFORMANCE_API_BASE}/suggestions`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ targetType, targetId, goals }),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.message || `HTTP ${response.status}: 生成优化建议失败`);
    }

    return response.json();
  }

  /**
   * 比较性能
   */
  static async comparePerformance(
    targetType: string,
    ids: string[],
    metrics: string[]
  ): Promise<{
    comparison: {
      id: string;
      name: string;
      values: Record<string, number>;
    }[];
    winner: string;
    analysis: string;
  }> {
    const response = await fetch(`${PERFORMANCE_API_BASE}/compare`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ targetType, ids, metrics }),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.message || `HTTP ${response.status}: 性能比较失败`);
    }

    return response.json();
  }
}

// 导出默认实例
export const performance = new PerformanceAPI();
