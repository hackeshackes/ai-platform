/**
 * 调度系统 API
 * Scheduler System API
 */

// ==================== 类型定义 ====================

/**
 * 调度任务
 */
export interface SchedulerTask {
  /** 任务ID */
  id: string;
  /** 任务名称 */
  name: string;
  /** 任务类型 */
  type: 'training' | 'inference' | 'data-processing' | 'evaluation' | 'custom';
  /** 任务状态 */
  status: 'pending' | 'queued' | 'running' | 'completed' | 'failed' | 'cancelled';
  /** 优先级 */
  priority: 'low' | 'normal' | 'high' | 'critical';
  /** 资源需求 */
  resourceRequirements: {
    /** GPU数量 */
    gpus: number;
    /** CPU核心数 */
    cpus: number;
    /** 内存（GB） */
    memory: number;
    /** 存储（GB） */
    storage: number;
    /** 专用资源 */
    specialized?: string[];
  };
  /** 依赖任务 */
  dependencies?: string[];
  /** 调度策略 */
  schedulingPolicy?: {
    strategy: 'fifo' | 'priority' | 'fair-share' | 'deadline';
    maxRuntime?: number;
    preemptible?: boolean;
  };
  /** 创建时间 */
  createdAt: string;
  /** 开始时间 */
  startedAt?: string;
  /** 完成时间 */
  completedAt?: string;
  /** 错误信息 */
  error?: string;
}

/**
 * 调度策略配置
 */
export interface SchedulingPolicyConfig {
  /** 策略名称 */
  name: string;
  /** 策略类型 */
  type: 'fifo' | 'priority' | 'fair-share' | 'deadline' | 'custom';
  /** 权重配置 */
  weights?: Record<string, number>;
  /** 公平共享配置 */
  fairShareConfig?: {
    userWeights: Record<string, number>;
    groupWeights: Record<string, number>;
  };
  /** 截止时间配置 */
  deadlineConfig?: {
    softDeadline: string;
    hardDeadline: string;
  };
  /** 抢占配置 */
  preemptionConfig?: {
    enabled: boolean;
    evictionPolicy: 'terminate' | 'checkpoint';
    checkpointPath?: string;
  };
}

/**
 * 资源池信息
 */
export interface ResourcePool {
  /** 池ID */
  id: string;
  /** 池名称 */
  name: string;
  /** 资源类型 */
  resourceType: 'gpu' | 'cpu' | 'memory' | 'storage';
  /** 总资源量 */
  totalResources: number;
  /** 已分配资源 */
  allocatedResources: number;
  /** 可用资源 */
  availableResources: number;
  /** 节点列表 */
  nodes: {
    id: string;
    name: string;
    status: 'available' | 'busy' | 'maintenance';
    resources: Record<string, number>;
  }[];
  /** 使用率 */
  utilization: number;
}

/**
 * 调度决策
 */
export interface SchedulingDecision {
  /** 决策ID */
  id: string;
  /** 决策时间 */
  timestamp: string;
  /** 调度事件类型 */
  eventType: 'task-submission' | 'resource-available' | 'task-completion' | 'rebalancing';
  /** 任务决策 */
  taskDecisions: {
    taskId: string;
    action: 'schedule' | 'requeue' | 'preempt' | 'reject';
    assignedNode: string;
    reason: string;
  }[];
  /** 资源重分配 */
  resourceReallocations: {
    fromNode: string;
    toNode: string;
    resources: Record<string, number>;
  }[];
  /** 决策指标 */
  metrics: {
    makespan: number;
    fairness: number;
    utilization: number;
  };
}

/**
 * 任务队列
 */
export interface TaskQueue {
  /** 队列ID */
  id: string;
  /** 队列名称 */
  name: string;
  /** 排队任务数 */
  pendingTasks: number;
  /** 运行中任务数 */
  runningTasks: number;
  /** 任务列表 */
  tasks: SchedulerTask[];
  /** 队列优先级 */
  priority: number;
  /** 资源配额 */
  quota: {
    maxGpus: number;
    maxCpus: number;
    maxMemory: number;
  };
  /** 当前使用量 */
  currentUsage: {
    gpus: number;
    cpus: number;
    memory: number;
  };
}

// ==================== API 客户端 ====================

const SCHEDULER_API_BASE = '/api/v12/scheduler';

/**
 * 调度系统 API 客户端
 */
export class SchedulerAPI {
  /**
   * 提交调度任务
   */
  static async submitTask(
    task: Omit<SchedulerTask, 'id' | 'status' | 'createdAt'>
  ): Promise<SchedulerTask> {
    const response = await fetch(`${SCHEDULER_API_BASE}/tasks`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(task),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.message || `HTTP ${response.status}: 任务提交失败`);
    }

    return response.json();
  }

  /**
   * 获取任务详情
   */
  static async getTaskDetails(id: string): Promise<SchedulerTask> {
    const response = await fetch(`${SCHEDULER_API_BASE}/tasks/${id}`);

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.message || `HTTP ${response.status}: 获取任务详情失败`);
    }

    return response.json();
  }

  /**
   * 获取任务列表
   */
  static async getTasks(
    filters?: {
      status?: string;
      type?: string;
      priority?: string;
      node?: string;
    }
  ): Promise<SchedulerTask[]> {
    const params = new URLSearchParams(filters || {});
    const response = await fetch(`${SCHEDULER_API_BASE}/tasks?${params}`);

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.message || `HTTP ${response.status}: 获取任务列表失败`);
    }

    return response.json();
  }

  /**
   * 取消任务
   */
  static async cancelTask(
    id: string,
    reason?: string
  ): Promise<{ success: boolean }> {
    const response = await fetch(`${SCHEDULER_API_BASE}/tasks/${id}/cancel`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ reason }),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.message || `HTTP ${response.status}: 任务取消失败`);
    }

    return response.json();
  }

  /**
   * 获取资源池列表
   */
  static async getResourcePools(): Promise<ResourcePool[]> {
    const response = await fetch(`${SCHEDULER_API_BASE}/resource-pools`);

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.message || `HTTP ${response.status}: 获取资源池失败`);
    }

    return response.json();
  }

  /**
   * 获取资源池详情
   */
  static async getResourcePoolDetails(id: string): Promise<ResourcePool> {
    const response = await fetch(`${SCHEDULER_API_BASE}/resource-pools/${id}`);

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.message || `HTTP ${response.status}: 获取资源池详情失败`);
    }

    return response.json();
  }

  /**
   * 获取任务队列
   */
  static async getTaskQueues(): Promise<TaskQueue[]> {
    const response = await fetch(`${SCHEDULER_API_BASE}/queues`);

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.message || `HTTP ${response.status}: 获取任务队列失败`);
    }

    return response.json();
  }

  /**
   * 配置调度策略
   */
  static async configurePolicy(
    config: SchedulingPolicyConfig
  ): Promise<{ policyId: string }> {
    const response = await fetch(`${SCHEDULER_API_BASE}/policies`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(config),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.message || `HTTP ${response.status}: 策略配置失败`);
    }

    return response.json();
  }

  /**
   * 获取调度决策历史
   */
  static async getSchedulingDecisions(
    startTime: string,
    endTime: string
  ): Promise<SchedulingDecision[]> {
    const params = new URLSearchParams({ startTime, endTime });
    const response = await fetch(`${SCHEDULER_API_BASE}/decisions?${params}`);

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.message || `HTTP ${response.status}: 获取调度决策失败`);
    }

    return response.json();
  }

  /**
   * 手动触发调度
   */
  static async triggerScheduling(
    reason?: string
  ): Promise<{ triggered: boolean }> {
    const response = await fetch(`${SCHEDULER_API_BASE}/trigger`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ reason }),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.message || `HTTP ${response.status}: 触发调度失败`);
    }

    return response.json();
  }

  /**
   * 获取调度器统计
   */
  static async getSchedulerStats(): Promise<{
    totalTasks: number;
    runningTasks: number;
    pendingTasks: number;
    completedTasks: number;
    averageWaitTime: number;
    averageExecutionTime: number;
    utilization: Record<string, number>;
  }> {
    const response = await fetch(`${SCHEDULER_API_BASE}/stats`);

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.message || `HTTP ${response.status}: 获取调度统计失败`);
    }

    return response.json();
  }
}

// 导出默认实例
export const scheduler = new SchedulerAPI();
