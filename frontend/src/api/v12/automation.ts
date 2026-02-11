/**
 * 自动化运维 API
 * Automation API
 */

// ==================== 类型定义 ====================

/**
 * 自动化任务
 */
export interface AutomationTask {
  /** 任务ID */
  id: string;
  /** 任务名称 */
  name: string;
  /** 任务描述 */
  description: string;
  /** 任务类型 */
  type: 'script' | 'playbook' | 'workflow' | 'api-call' | 'custom';
  /** 任务状态 */
  status: 'pending' | 'queued' | 'running' | 'completed' | 'failed' | 'cancelled' | 'paused';
  /** 触发类型 */
  triggerType: 'manual' | 'scheduled' | 'event-driven' | 'webhook';
  /** 触发条件 */
  triggerCondition?: {
    event?: string;
    schedule?: string;
    webhookId?: string;
  };
  /** 执行参数 */
  parameters?: Record<string, unknown>;
  /** 执行配置 */
  executionConfig: {
    /** 超时时间（秒） */
    timeout: number;
    /** 重试次数 */
    retries: number;
    /** 并发策略 */
    concurrency: 'sequential' | 'parallel';
  };
  /** 目标主机 */
  targets?: {
    type: 'host' | 'group' | 'tag';
    identifiers: string[];
  }[];
  /** 创建时间 */
  createdAt: string;
  /** 最后执行时间 */
  lastExecutedAt?: string;
  /** 创建者 */
  createdBy: string;
}

/**
 * 执行历史
 */
export interface ExecutionHistory {
  /** 执行ID */
  id: string;
  /** 任务ID */
  taskId: string;
  /** 执行时间 */
  executedAt: string;
  /** 状态 */
  status: 'success' | 'failed' | 'timeout' | 'cancelled';
  /** 执行时长（秒） */
  duration: number;
  /** 执行用户 */
  executedBy: string;
  /** 输入参数 */
  inputParameters: Record<string, unknown>;
  /** 输出结果 */
  output?: Record<string, unknown>;
  /** 错误信息 */
  error?: {
    message: string;
    stackTrace?: string;
  };
  /** 日志 */
  logs: {
    timestamp: string;
    level: 'info' | 'warn' | 'error';
    message: string;
  }[];
}

/**
 * 自动化工作流
 */
export interface AutomationWorkflow {
  /** 工作流ID */
  id: string;
  /** 工作流名称 */
  name: string;
  /** 工作流描述 */
  description: string;
  /** 状态 */
  status: 'active' | 'inactive' | 'draft';
  /** 步骤 */
  steps: {
    stepId: string;
    name: string;
    type: 'task' | 'condition' | 'approval' | 'notification' | 'delay';
    config: Record<string, unknown>;
    nextSteps: {
      condition: string;
      nextStepId: string;
    }[];
    errorHandling: {
      onError: 'stop' | 'retry' | 'skip' | 'jump';
      errorStepId?: string;
      retryCount?: number;
    };
  }[];
  /** 触发器 */
  triggers: {
    type: 'manual' | 'scheduled' | 'event' | 'webhook';
    config: Record<string, unknown>;
  }[];
  /** 执行历史 */
  executions: {
    id: string;
    status: string;
    startTime: string;
    endTime?: string;
  }[];
}

/**
 * 变更管理
 */
export interface ChangeManagement {
  /** 变更ID */
  id: string;
  /** 变更标题 */
  title: string;
  /** 变更描述 */
  description: string;
  /** 变更类型 */
  type: 'normal' | 'standard' | 'emergency';
  /** 风险等级 */
  riskLevel: 'low' | 'medium' | 'high' | 'critical';
  /** 状态 */
  status: 'draft' | 'pending-approval' | 'approved' | 'in-progress' | 'completed' | 'rejected' | 'cancelled';
  /** 实施计划 */
  implementationPlan: {
    steps: {
      step: number;
      action: string;
      expectedOutcome: string;
      rollbackAction: string;
    }[];
    estimatedDuration: number;
    requiredApprovals: string[];
  };
  /** 影响范围 */
  impactScope: {
    systems: string[];
    services: string[];
    users: number;
  };
  /** 审批信息 */
  approvals: {
    approver: string;
    status: 'pending' | 'approved' | 'rejected';
    comment?: string;
    timestamp?: string;
  }[];
  /** 创建者 */
  createdBy: string;
  /** 创建时间 */
  createdAt: string;
}

/**
 * 配置项
 */
export interface ConfigurationItem {
  /** 配置ID */
  id: string;
  /** 配置名称 */
  name: string;
  /** 配置类型 */
  type: 'server' | 'application' | 'database' | 'network' | 'service';
  /** 配置键值对 */
  attributes: Record<string, unknown>;
  /** 关联配置 */
  relations: {
    type: 'depends-on' | 'connected-to' | 'parent-of' | 'member-of';
    targetId: string;
  }[];
  /** 版本信息 */
  version: string;
  /** 状态 */
  status: 'active' | 'inactive' | 'maintenance';
  /** 最后更新时间 */
  lastUpdated: string;
}

// ==================== API 客户端 ====================

const AUTOMATION_API_BASE = '/api/v12/automation';

/**
 * 自动化运维 API 客户端
 */
export class AutomationAPI {
  /**
   * 创建自动化任务
   */
  static async createTask(
    task: Omit<AutomationTask, 'id' | 'status' | 'createdAt'>
  ): Promise<AutomationTask> {
    const response = await fetch(`${AUTOMATION_API_BASE}/tasks`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(task),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.message || `HTTP ${response.status}: 创建任务失败`);
    }

    return response.json();
  }

  /**
   * 获取任务列表
   */
  static async getTasks(
    filters?: {
      type?: string;
      status?: string;
      triggerType?: string;
    }
  ): Promise<AutomationTask[]> {
    const params = new URLSearchParams(filters || {});
    const response = await fetch(`${AUTOMATION_API_BASE}/tasks?${params}`);

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.message || `HTTP ${response.status}: 获取任务列表失败`);
    }

    return response.json();
  }

  /**
   * 获取任务详情
   */
  static async getTaskDetails(id: string): Promise<AutomationTask> {
    const response = await fetch(`${AUTOMATION_API_BASE}/tasks/${id}`);

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.message || `HTTP ${response.status}: 获取任务详情失败`);
    }

    return response.json();
  }

  /**
   * 手动执行任务
   */
  static async executeTask(
    taskId: string,
    parameters?: Record<string, unknown>
  ): Promise<{
    executionId: string;
    status: string;
  }> {
    const response = await fetch(`${AUTOMATION_API_BASE}/tasks/${taskId}/execute`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ parameters }),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.message || `HTTP ${response.status}: 任务执行失败`);
    }

    return response.json();
  }

  /**
   * 获取执行历史
   */
  static async getExecutionHistory(
    taskId: string,
    page: number = 1,
    limit: number = 20
  ): Promise<{
    executions: ExecutionHistory[];
    total: number;
  }> {
    const params = new URLSearchParams({
      page: page.toString(),
      limit: limit.toString(),
    });
    const response = await fetch(`${AUTOMATION_API_BASE}/tasks/${taskId}/executions?${params}`);

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.message || `HTTP ${response.status}: 获取执行历史失败`);
    }

    return response.json();
  }

  /**
   * 获取执行详情
   */
  static async getExecutionDetails(executionId: string): Promise<ExecutionHistory> {
    const response = await fetch(`${AUTOMATION_API_BASE}/executions/${executionId}`);

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.message || `HTTP ${response.status}: 获取执行详情失败`);
    }

    return response.json();
  }

  /**
   * 创建工作流
   */
  static async createWorkflow(
    workflow: Omit<AutomationWorkflow, 'id' | 'executions'>
  ): Promise<AutomationWorkflow> {
    const response = await fetch(`${AUTOMATION_API_BASE}/workflows`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(workflow),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.message || `HTTP ${response.status}: 创建工作流失败`);
    }

    return response.json();
  }

  /**
   * 获取工作流列表
   */
  static async getWorkflows(
    status?: 'active' | 'inactive' | 'draft'
  ): Promise<AutomationWorkflow[]> {
    const params = new URLSearchParams(status ? { status } : {});
    const response = await fetch(`${AUTOMATION_API_BASE}/workflows?${params}`);

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.message || `HTTP ${response.status}: 获取工作流列表失败`);
    }

    return response.json();
  }

  /**
   * 执行工作流
   */
  static async executeWorkflow(
    workflowId: string,
    inputs?: Record<string, unknown>
  ): Promise<{
    executionId: string;
    status: string;
  }> {
    const response = await fetch(`${AUTOMATION_API_BASE}/workflows/${workflowId}/execute`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ inputs }),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.message || `HTTP ${response.status}: 工作流执行失败`);
    }

    return response.json();
  }

  /**
   * 获取变更列表
   */
  static async getChanges(
    filters?: {
      status?: string;
      type?: string;
      riskLevel?: string;
    }
  ): Promise<ChangeManagement[]> {
    const params = new URLSearchParams(filters || {});
    const response = await fetch(`${AUTOMATION_API_BASE}/changes?${params}`);

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.message || `HTTP ${response.status}: 获取变更列表失败`);
    }

    return response.json();
  }

  /**
   * 创建变更
   */
  static async createChange(
    change: Omit<ChangeManagement, 'id' | 'status' | 'createdAt'>
  ): Promise<ChangeManagement> {
    const response = await fetch(`${AUTOMATION_API_BASE}/changes`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(change),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.message || `HTTP ${response.status}: 创建变更失败`);
    }

    return response.json();
  }

  /**
   * 审批变更
   */
  static async approveChange(
    changeId: string,
    decision: 'approved' | 'rejected',
    comment?: string
  ): Promise<ChangeManagement> {
    const response = await fetch(`${AUTOMATION_API_BASE}/changes/${changeId}/approve`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ decision, comment }),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.message || `HTTP ${response.status}: 变更审批失败`);
    }

    return response.json();
  }

  /**
   * 获取配置项
   */
  static async getConfigurationItems(
    type?: string
  ): Promise<ConfigurationItem[]> {
    const params = new URLSearchParams(type ? { type } : {});
    const response = await fetch(`${AUTOMATION_API_BASE}/configuration?${params}`);

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.message || `HTTP ${response.status}: Failed to get configuration items`);
    }

    return response.json();
  }

  /**
   * 更新配置项
   */
  static async updateConfigurationItem(
    id: string,
    attributes: Record<string, unknown>
  ): Promise<ConfigurationItem> {
    const response = await fetch(`${AUTOMATION_API_BASE}/configuration/${id}`, {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ attributes }),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.message || `HTTP ${response.status}: 更新配置失败`);
    }

    return response.json();
  }
}

// 导出默认实例
export const automation = new AutomationAPI();
