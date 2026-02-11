/**
 * AIOps API
 * AI Operations API
 */

// ==================== 类型定义 ====================

/**
 * AIOps 告警
 */
export interface AIOpsAlert {
  /** 告警ID */
  id: string;
  /** 告警标题 */
  title: string;
  /** 告警级别 */
  severity: 'critical' | 'warning' | 'info' | 'low';
  /** 告警来源 */
  source: string;
  /** 触发时间 */
  timestamp: string;
  /** 关联实体 */
  affectedEntity: {
    type: string;
    id: string;
    name: string;
  };
  /** 告警内容 */
  description: string;
  /** 根因分析 */
  rootCause?: {
    probability: number;
    description: string;
    relatedMetrics: string[];
  };
  /** 建议操作 */
  recommendedActions?: {
    action: string;
    priority: number;
    automation?: boolean;
  }[];
  /** 状态 */
  status: 'active' | 'acknowledged' | 'resolved' | 'suppressed';
}

/**
 * 异常检测结果
 */
export interface AnomalyDetectionResult {
  /** 检测ID */
  id: string;
  /** 检测时间 */
  timestamp: string;
  /** 实体类型 */
  entityType: string;
  /** 实体ID */
  entityId: string;
  /** 异常类型 */
  anomalyType: 'point' | 'contextual' | 'collective';
  /** 异常分数 */
  anomalyScore: number;
  /** 置信度 */
  confidence: number;
  /** 异常特征 */
  features: {
    name: string;
    expected: number;
    actual: number;
    deviation: number;
  }[];
  /** 预测值 */
  predictedValue?: number;
}

/**
 * 根因分析结果
 */
export interface RootCauseAnalysisResult {
  /** 分析ID */
  id: string;
  /** 问题描述 */
  problem: string;
  /** 相关告警 */
  relatedAlerts: string[];
  /** 根因假设 */
  hypotheses: {
    cause: string;
    probability: number;
    evidence: string[];
    affectedComponents: string[];
  }[];
  /** 推荐解决方案 */
  recommendedSolutions: {
    solution: string;
    effectiveness: number;
    effort: number;
    risk: 'low' | 'medium' | 'high';
  }[];
  /** 影响范围 */
  impactScope: {
    components: string[];
    users: number;
    services: string[];
  };
}

/**
 * 容量预测
 */
export interface CapacityForecast {
  /** 预测ID */
  id: string;
  /** 资源类型 */
  resourceType: string;
  /** 资源ID */
  resourceId: string;
  /** 预测时间范围 */
  forecastRange: {
    start: string;
    end: string;
  };
  /** 预测数据 */
  forecast: {
    timestamp: string;
    usage: number;
    capacity: number;
    utilizationPercent: number;
  }[];
  /** 瓶颈预测 */
  bottlenecks: {
    timestamp: string;
    resource: string;
    predictedShortage: number;
  }[];
  /** 扩容建议 */
  scalingRecommendations: {
    recommendedCapacity: number;
    estimatedCost: number;
    urgency: 'low' | 'medium' | 'high';
  }[];
}

/**
 * 智能告警分组
 */
export interface AlertGroup {
  /** 分组ID */
  id: string;
  /** 分组名称 */
  name: string;
  /** 相关告警数 */
  alertCount: number;
  /** 告警严重程度分布 */
  severityDistribution: Record<string, number>;
  /** 共同根因 */
  commonRootCause?: string;
  /** 首次告警时间 */
  firstAlertTime: string;
  /** 最近告警时间 */
  lastAlertTime: string;
  /** 状态 */
  status: 'active' | 'investigating' | 'resolved';
}

// ==================== API 客户端 ====================

const AIOPS_API_BASE = '/api/v12/aiops';

/**
 * AIOps API 客户端
 */
export class AIOpsAPI {
  /**
   * 获取告警列表
   */
  static async getAlerts(
    filters?: {
      severity?: string;
      status?: string;
      source?: string;
      startTime?: string;
      endTime?: string;
    }
  ): Promise<AIOpsAlert[]> {
    const params = new URLSearchParams(filters || {});
    const response = await fetch(`${AIOPS_API_BASE}/alerts?${params}`);

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.message || `HTTP ${response.status}: 获取告警列表失败`);
    }

    return response.json();
  }

  /**
   * 获取告警详情
   */
  static async getAlertDetails(id: string): Promise<AIOpsAlert> {
    const response = await fetch(`${AIOPS_API_BASE}/alerts/${id}`);

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.message || `HTTP ${response.status}: 获取告警详情失败`);
    }

    return response.json();
  }

  /**
   * 执行异常检测
   */
  static async detectAnomaly(
    entityType: string,
    entityId: string,
    metrics: {
      name: string;
      values: number[];
      threshold?: number;
    }[]
  ): Promise<AnomalyDetectionResult> {
    const response = await fetch(`${AIOPS_API_BASE}/anomaly-detection`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ entityType, entityId, metrics }),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.message || `HTTP ${response.status}: 异常检测失败`);
    }

    return response.json();
  }

  /**
   * 启动根因分析
   */
  static async startRootCauseAnalysis(
    problemDescription: string,
    relatedAlerts?: string[]
  ): Promise<RootCauseAnalysisResult> {
    const response = await fetch(`${AIOPS_API_BASE}/root-cause`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ problemDescription, relatedAlerts }),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.message || `HTTP ${response.status}: 根因分析启动失败`);
    }

    return response.json();
  }

  /**
   * 获取容量预测
   */
  static async getCapacityForecast(
    resourceType: string,
    resourceId: string,
    forecastDays: number = 30
  ): Promise<CapacityForecast> {
    const params = new URLSearchParams({
      resourceType,
      resourceId,
      forecastDays: forecastDays.toString(),
    });
    const response = await fetch(`${AIOPS_API_BASE}/capacity-forecast?${params}`);

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.message || `HTTP ${response.status}: 容量预测失败`);
    }

    return response.json();
  }

  /**
   * 智能告警分组
   */
  static async groupAlerts(
    alertIds: string[],
    strategy: 'time' | 'metric' | 'topology' | 'semantic'
  ): Promise<AlertGroup[]> {
    const response = await fetch(`${AIOPS_API_BASE}/alerts/group`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ alertIds, strategy }),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.message || `HTTP ${response.status}: 告警分组失败`);
    }

    return response.json();
  }

  /**
   * 执行自动化修复
   */
  static async executeAutoRemediation(
    alertId: string,
    action: string
  ): Promise<{
    success: boolean;
    executionId: string;
    message: string;
  }> {
    const response = await fetch(`${AIOPS_API_BASE}/auto-remediation`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ alertId, action }),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.message || `HTTP ${response.status}: 自动修复执行失败`);
    }

    return response.json();
  }

  /**
   * 获取运维洞察
   */
  static async getOperationalInsights(
    timeRange: {
      start: string;
      end: string;
    }
  ): Promise<{
    summary: {
      totalAlerts: number;
      resolvedAlerts: number;
      mttr: number;
      mtta: number;
    };
    trends: {
      alertTrend: number[];
      resolutionTrend: number[];
    };
    topIssues: {
      issue: string;
      count: number;
      impact: number;
    }[];
    recommendations: string[];
  }> {
    const response = await fetch(`${AIOPS_API_BASE}/insights`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ timeRange }),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.message || `HTTP ${response.status}: 获取运维洞察失败`);
    }

    return response.json();
  }
}

// 导出默认实例
export const aiops = new AIOpsAPI();
