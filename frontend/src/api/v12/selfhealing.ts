/**
 * 自愈系统 API
 * Self-Healing System API
 */

// ==================== 类型定义 ====================

/**
 * 自愈事件
 */
export interface SelfHealingEvent {
  /** 事件ID */
  id: string;
  /** 事件类型 */
  type: 'anomaly-detected' | 'failure-predicted' | 'degradation-detected' | 'manual-trigger';
  /** 严重程度 */
  severity: 'critical' | 'warning' | 'info';
  /** 触发时间 */
  timestamp: string;
  /** 受影响实体 */
  affectedEntity: {
    type: string;
    id: string;
    name: string;
  };
  /** 问题描述 */
  description: string;
  /** 检测到的异常 */
  detectedAnomaly?: {
    metrics: {
      name: string;
      expected: number;
      actual: number;
      deviation: number;
    }[];
    anomalyScore: number;
  };
  /** 预测的故障 */
  predictedFailure?: {
    type: string;
    probability: number;
    estimatedTimeToFailure: number;
  };
  /** 自愈状态 */
  healingStatus: 'pending' | 'diagnosing' | 'applying-fix' | 'verifying' | 'completed' | 'failed';
  /** 应用的修复方案 */
  appliedFix?: {
    fixId: string;
    fixType: string;
    actions: string[];
    result: 'success' | 'partial' | 'failed';
  };
  /** 验证结果 */
  verification?: {
    passed: boolean;
    metrics: {
      name: string;
      before: number;
      after: number;
      improvement: number;
    }[];
  };
}

/**
 * 修复方案
 */
export interface FixSolution {
  /** 方案ID */
  id: string;
  /** 问题类型 */
  problemType: string;
  /** 方案名称 */
  name: string;
  /** 方案描述 */
  description: string;
  /** 自动化级别 */
  automationLevel: 'manual' | 'semi-automated' | 'fully-automated';
  /** 执行步骤 */
  steps: {
    step: number;
    action: string;
    command?: string;
    rollback?: string;
    estimatedTime: number;
  }[];
  /** 前置条件 */
  prerequisites: string[];
  /** 风险评估 */
  riskAssessment: {
    level: 'low' | 'medium' | 'high';
    impactAreas: string[];
    rollbackPlan: string;
  };
  /** 成功率统计 */
  successRate: number;
  /** 平均修复时间 */
  averageFixTime: number;
}

/**
 * 自愈策略
 */
export interface SelfHealingStrategy {
  /** 策略ID */
  id: string;
  /** 策略名称 */
  name: string;
  /** 策略描述 */
  description: string;
  /** 触发条件 */
  triggerConditions: {
    metric: string;
    operator: 'gt' | 'lt' | 'eq' | 'gte' | 'lte';
    threshold: number;
    duration: number;
  }[];
  /** 适用实体类型 */
  applicableEntities: string[];
  /** 修复方案 */
  fixSolutionId: string;
  /** 执行策略 */
  executionPolicy: {
    maxRetries: number;
    retryInterval: number;
    timeout: number;
    rollbackOnFailure: boolean;
  };
  /** 启用状态 */
  enabled: boolean;
  /** 最后执行时间 */
  lastExecutedAt?: string;
}

/**
 * 健康检查结果
 */
export interface HealthCheckResult {
  /** 检查ID */
  id: string;
  /** 检查时间 */
  timestamp: string;
  /** 检查目标 */
  target: {
    type: string;
    id: string;
    name: string;
  };
  /** 健康状态 */
  status: 'healthy' | 'degraded' | 'unhealthy' | 'unknown';
  /** 检查项目 */
  checks: {
    name: string;
    status: 'pass' | 'fail' | 'warn';
    message: string;
    details?: Record<string, unknown>;
  }[];
  /** 总体评分 */
  overallScore: number;
  /** 建议操作 */
  recommendations: string[];
}

/**
 * 回滚计划
 */
export interface RollbackPlan {
  /** 计划ID */
  id: string;
  /** 关联的修复ID */
  fixId: string;
  /** 步骤 */
  steps: {
    step: number;
    action: string;
    command: string;
    expectedOutcome: string;
    verification: string;
  }[];
  /** 创建时间 */
  createdAt: string;
  /** 最后更新时间 */
  updatedAt: string;
}

// ==================== API 客户端 ====================

const SELFHEALING_API_BASE = '/api/v12/selfhealing';

/**
 * 自愈系统 API 客户端
 */
export class SelfHealingAPI {
  /**
   * 获取自愈事件列表
   */
  static async getEvents(
    filters?: {
      status?: string;
      severity?: string;
      entityType?: string;
      startTime?: string;
      endTime?: string;
    }
  ): Promise<SelfHealingEvent[]> {
    const params = new URLSearchParams(filters || {});
    const response = await fetch(`${SELFHEALING_API_BASE}/events?${params}`);

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.message || `HTTP ${response.status}: 获取事件列表失败`);
    }

    return response.json();
  }

  /**
   * 获取事件详情
   */
  static async getEventDetails(id: string): Promise<SelfHealingEvent> {
    const response = await fetch(`${SELFHEALING_API_BASE}/events/${id}`);

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.message || `HTTP ${response.status}: 获取事件详情失败`);
    }

    return response.json();
  }

  /**
   * 手动触发自愈
   */
  static async triggerHealing(
    entityType: string,
    entityId: string,
    fixSolutionId?: string
  ): Promise<SelfHealingEvent> {
    const response = await fetch(`${SELFHEALING_API_BASE}/trigger`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ entityType, entityId, fixSolutionId }),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.message || `HTTP ${response.status}: 触发自愈失败`);
    }

    return response.json();
  }

  /**
   * 获取修复方案列表
   */
  static async getFixSolutions(
    problemType?: string
  ): Promise<FixSolution[]> {
    const params = new URLSearchParams(problemType ? { problemType } : {});
    const response = await fetch(`${SELFHEALING_API_BASE}/fix-solutions?${params}`);

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.message || `HTTP ${response.status}: 获取修复方案失败`);
    }

    return response.json();
  }

  /**
   * 获取修复方案详情
   */
  static async getFixSolutionDetails(id: string): Promise<FixSolution> {
    const response = await fetch(`${SELFHEALING_API_BASE}/fix-solutions/${id}`);

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.message || `HTTP ${response.status}: 获取方案详情失败`);
    }

    return response.json();
  }

  /**
   * 获取自愈策略列表
   */
  static async getStrategies(): Promise<SelfHealingStrategy[]> {
    const response = await fetch(`${SELFHEALING_API_BASE}/strategies`);

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.message || `HTTP ${response.status}: 获取策略列表失败`);
    }

    return response.json();
  }

  /**
   * 创建/更新自愈策略
   */
  static async saveStrategy(
    strategy: Omit<SelfHealingStrategy, 'id'>
  ): Promise<SelfHealingStrategy> {
    const response = await fetch(`${SELFHEALING_API_BASE}/strategies`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(strategy),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.message || `HTTP ${response.status}: 保存策略失败`);
    }

    return response.json();
  }

  /**
   * 执行健康检查
   */
  static async executeHealthCheck(
    entityType: string,
    entityId: string
  ): Promise<HealthCheckResult> {
    const response = await fetch(`${SELFHEALING_API_BASE}/health-check`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ entityType, entityId }),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.message || `HTTP ${response.status}: 健康检查失败`);
    }

    return response.json();
  }

  /**
   * 获取回滚计划
   */
  static async getRollbackPlan(fixId: string): Promise<RollbackPlan> {
    const response = await fetch(`${SELFHEALING_API_BASE}/rollback/${fixId}`);

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.message || `HTTP ${response.status}: 获取回滚计划失败`);
    }

    return response.json();
  }

  /**
   * 执行回滚
   */
  static async executeRollback(
    fixId: string,
    reason?: string
  ): Promise<{
    success: boolean;
    rollbackId: string;
    message: string;
  }> {
    const response = await fetch(`${SELFHEALING_API_BASE}/rollback/${fixId}/execute`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ reason }),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.message || `HTTP ${response.status}: 回滚执行失败`);
    }

    return response.json();
  }

  /**
   * 获取自愈统计
   */
  static async getStatistics(
    timeRange?: {
      start: string;
      end: string;
    }
  ): Promise<{
    totalEvents: number;
    healedAutomatically: number;
    healedManually: number;
    failedHealings: number;
    averageHealingTime: number;
    successRate: number;
    topProblems: {
      problem: string;
      count: number;
      successRate: number;
    }[];
  }> {
    const params = new URLSearchParams(timeRange || {});
    const response = await fetch(`${SELFHEALING_API_BASE}/statistics?${params}`);

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.message || `HTTP ${response.status}: 获取统计失败`);
    }

    return response.json();
  }
}

// 导出默认实例
export const selfhealing = new SelfHealingAPI();
