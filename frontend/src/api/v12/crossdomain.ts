/**
 * 跨域推理 API
 * Cross-Domain Reasoning API
 */

// ==================== 类型定义 ====================

/**
 * 跨域推理请求
 */
export interface CrossDomainRequest {
  /** 源领域 */
  sourceDomain: string;
  /** 目标领域 */
  targetDomain: string;
  /** 问题描述 */
  problem: string;
  /** 可用知识库 */
  knowledgeBases: string[];
  /** 推理策略 */
  strategy?: 'analogical' | 'abstractive' | 'compositional' | 'hybrid';
}

/**
 * 跨域推理结果
 */
export interface CrossDomainResult {
  /** 推理ID */
  id: string;
  /** 状态 */
  status: 'pending' | 'reasoning' | 'completed' | 'failed';
  /** 源领域 */
  sourceDomain: string;
  /** 目标领域 */
  targetDomain: string;
  /** 推理结果 */
  result?: {
    /** 答案 */
    answer: string;
    /** 置信度 */
    confidence: number;
    /** 推理路径 */
    reasoningPath: {
      step: number;
      domain: string;
      inference: string;
      confidence: number;
    }[];
    /** 类比映射 */
    analogyMappings?: {
      sourceConcept: string;
      targetConcept: string;
      similarity: number;
    }[];
    /** 知识转移 */
    transferredKnowledge?: {
      concept: string;
      transferredFrom: string;
      adaptedFor: string;
      adaptationMethod: string;
    }[];
  };
  /** 错误信息 */
  error?: string;
}

/**
 * 领域知识图谱
 */
export interface DomainKnowledgeGraph {
  /** 图谱ID */
  id: string;
  /** 领域名称 */
  domain: string;
  /** 节点 */
  nodes: {
    id: string;
    label: string;
    type: string;
    properties: Record<string, unknown>;
  }[];
  /** 边 */
  edges: {
    source: string;
    target: string;
    relation: string;
    weight: number;
  }[];
  /** 核心概念 */
  coreConcepts: string[];
}

/**
 * 概念映射
 */
export interface ConceptMapping {
  /** 映射ID */
  id: string;
  /** 源概念 */
  sourceConcept: {
    domain: string;
    term: string;
    definition: string;
  };
  /** 目标概念 */
  targetConcept: {
    domain: string;
    term: string;
    definition: string;
  };
  /** 相似度 */
  similarity: number;
  /** 映射类型 */
  mappingType: 'exact' | 'partial' | 'analogous' | 'metaphorical';
  /** 对齐方法 */
  alignmentMethod: string;
}

/**
 * 跨域推理任务
 */
export interface CrossDomainTask {
  /** 任务ID */
  id: string;
  /** 任务名称 */
  name: string;
  /** 源领域 */
  sourceDomains: string[];
  /** 目标领域 */
  targetDomain: string;
  /** 任务类型 */
  type: 'classification' | 'prediction' | 'generation' | 'explanation';
  /** 难度 */
  difficulty: 'easy' | 'medium' | 'hard' | 'expert';
  /** 预期推理深度 */
  expectedDepth: number;
}

// ==================== API 客户端 ====================

const CROSSDOMAIN_API_BASE = '/api/v12/crossdomain';

/**
 * 跨域推理 API 客户端
 */
export class CrossDomainAPI {
  /**
   * 提交跨域推理请求
   */
  static async reason(
    request: CrossDomainRequest
  ): Promise<CrossDomainResult> {
    const response = await fetch(`${CROSSDOMAIN_API_BASE}/reason`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.message || `HTTP ${response.status}: 跨域推理失败`);
    }

    return response.json();
  }

  /**
   * 获取推理结果
   */
  static async getReasoningResult(id: string): Promise<CrossDomainResult> {
    const response = await fetch(`${CROSSDOMAIN_API_BASE}/reason/${id}`);

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.message || `HTTP ${response.status}: 获取推理结果失败`);
    }

    return response.json();
  }

  /**
   * 获取领域知识图谱
   */
  static async getKnowledgeGraph(
    domain: string
  ): Promise<DomainKnowledgeGraph> {
    const response = await fetch(`${CROSSDOMAIN_API_BASE}/knowledge/${domain}`);

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.message || `HTTP ${response.status}: 获取知识图谱失败`);
    }

    return response.json();
  }

  /**
   * 搜索概念映射
   */
  static async searchConceptMappings(
    concept: string,
    sourceDomain: string,
    targetDomain: string
  ): Promise<ConceptMapping[]> {
    const params = new URLSearchParams({
      concept,
      sourceDomain,
      targetDomain,
    });
    const response = await fetch(`${CROSSDOMAIN_API_BASE}/mappings?${params}`);

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.message || `HTTP ${response.status}: 搜索概念映射失败`);
    }

    return response.json();
  }

  /**
   * 计算概念相似度
   */
  static async calculateConceptSimilarity(
    concept1: { domain: string; term: string },
    concept2: { domain: string; term: string }
  ): Promise<{
    semanticSimilarity: number;
    structuralSimilarity: number;
    overallSimilarity: number;
  }> {
    const response = await fetch(`${CROSSDOMAIN_API_BASE}/similarity`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ concept1, concept2 }),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.message || `HTTP ${response.status}: 计算相似度失败`);
    }

    return response.json();
  }

  /**
   * 执行类比推理
   */
  static async analogicalReasoning(
    sourceProblem: string,
    targetProblem: string,
    sourceDomain: string,
    targetDomain: string
  ): Promise<{
    mappings: {
      sourceElement: string;
      targetElement: string;
      mappingType: string;
    }[];
    predictions: {
      targetPrediction: string;
      confidence: number;
    }[];
    validation: {
      consistency: number;
      plausibility: number;
    };
  }> {
    const response = await fetch(`${CROSSDOMAIN_API_BASE}/analogical`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ sourceProblem, targetProblem, sourceDomain, targetDomain }),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.message || `HTTP ${response.status}: 类比推理失败`);
    }

    return response.json();
  }

  /**
   * 获取跨域任务
   */
  static async getTasks(
    filters?: {
      sourceDomain?: string;
      targetDomain?: string;
      difficulty?: string;
      type?: string;
    }
  ): Promise<CrossDomainTask[]> {
    const params = new URLSearchParams(filters || {});
    const response = await fetch(`${CROSSDOMAIN_API_BASE}/tasks?${params}`);

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.message || `HTTP ${response.status}: 获取任务列表失败`);
    }

    return response.json();
  }

  /**
   * 评估跨域泛化能力
   */
  static async evaluateGeneralization(
    modelId: string,
    domainPairs: { source: string; target: string }[]
  ): Promise<{
    overallScore: number;
    perPairScores: {
      pair: string;
      accuracy: number;
      transferLoss: number;
    }[];
    recommendations: string[];
  }> {
    const response = await fetch(`${CROSSDOMAIN_API_BASE}/evaluate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ modelId, domainPairs }),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.message || `HTTP ${response.status}: 泛化能力评估失败`);
    }

    return response.json();
  }
}

// 导出默认实例
export const crossdomain = new CrossDomainAPI();
