/**
 * v12 模块类型定义
 * AI Platform v12 版本类型
 */

// v12 模块分组
export type V12ModuleGroup = 
  | 'democratization'    // AI民主化
  | 'hyperautomation'    // 超自动化
  | 'superintelligence'  // 超级智能
  | 'quantum'           // 量子AI
  | 'cosmos'            // 宇宙级AI

// AI民主化模块
export interface NLGenerator {
  type: 'nl-generator'
  path: '/v12/nl-generator'
  title: '自然语言生成器'
  description: '将自然语言转换为代码和工作流'
  icon: '💬'
}

export interface NoCodeBuilder {
  type: 'no-code'
  path: '/v12/no-code'
  title: '无代码构建器'
  description: '可视化拖拽构建AI应用'
  icon: '🧱'
}

export interface TemplateMarketplace {
  type: 'templates'
  path: '/v12/templates'
  title: '模板市场'
  description: '分享和使用AI应用模板'
  icon: '📦'
}

export interface EducationCenter {
  type: 'education'
  path: '/v12/education'
  title: '教育中心'
  description: 'AI学习和培训资源'
  icon: '🎓'
}

export interface SmartRecommender {
  type: 'recommender'
  path: '/v12/recommender'
  title: '智能推荐'
  description: '个性化AI解决方案推荐'
  icon: '💡'
}

export interface AutoDocumentation {
  type: 'auto-doc'
  path: '/v12/auto-doc'
  title: '自动文档'
  description: 'AI应用自动文档生成'
  icon: '📝'
}

export interface OneClickDeploy {
  type: 'deploy'
  path: '/v12/deploy'
  title: '一键部署'
  description: '快速部署AI应用到生产环境'
  icon: '🚀'
}

// 超自动化模块
export interface AIOpsDashboard {
  type: 'aiops'
  path: '/v12/aiops'
  title: 'AIOps仪表板'
  description: 'AI驱动的运维监控和告警'
  icon: '📊'
}

export interface SmartScheduler {
  type: 'scheduler'
  path: '/v12/scheduler'
  title: '智能调度'
  description: 'AI优化的任务调度系统'
  icon: '⏰'
}

export interface SelfHealing {
  type: 'self-healing'
  path: '/v12/self-healing'
  title: '自愈系统'
  description: '自动化故障检测和恢复'
  icon: '🔧'
}

export interface AutomationOps {
  type: 'automation'
  path: '/v12/automation'
  title: '自动化运维'
  description: '全面的IT自动化解决方案'
  icon: '⚙️'
}

export interface PerformanceTuner {
  type: 'performance'
  path: '/v12/performance'
  title: '性能调优'
  description: 'AI驱动的性能优化建议'
  icon: '📈'
}

// 超级智能模块
export interface MetaLearning {
  type: 'meta-learning'
  path: '/v12/meta-learning'
  title: '元学习'
  description: '学会学习的AI系统'
  icon: '🧠'
}

export interface EmergenceEngine {
  type: 'emergence'
  path: '/v12/emergence'
  title: '涌现引擎'
  description: '探索AI系统的涌现行为'
  icon: '✨'
}

export interface CrossDomainReasoning {
  type: 'cross-domain'
  path: '/v12/cross-domain'
  title: '跨域推理'
  description: '跨领域知识迁移和推理'
  icon: '🌐'
}

export interface ContinualLearning {
  type: 'continual'
  path: '/v12/continual'
  title: '持续学习'
  description: '增量学习和知识积累'
  icon: '🔄'
}

// 量子AI模块
export interface QuantumSimulator {
  type: 'quantum-sim'
  path: '/v12/quantum-sim'
  title: '量子模拟器'
  description: '量子电路模拟和测试'
  icon: '⚛️'
}

export interface QuantumOptimizer {
  type: 'quantum-opt'
  path: '/v12/quantum-opt'
  title: '量子优化'
  description: '量子优化算法实现'
  icon: '🎯'
}

export interface QuantumML {
  type: 'quantum-ml'
  path: '/v12/quantum-ml'
  title: '量子机器学习'
  description: '量子增强的机器学习'
  icon: '🔮'
}

export interface HybridCompute {
  type: 'hybrid'
  path: '/v12/hybrid'
  title: '混合计算'
  description: '经典和量子混合计算'
  icon: '🔀'
}

// 宇宙级AI模块
export interface ClimateModel {
  type: 'climate'
  path: '/v12/climate'
  title: '气候模型'
  description: 'AI驱动的气候预测和模拟'
  icon: '🌍'
}

export interface BioSimulation {
  type: 'bio'
  path: '/v12/bio'
  title: '生物模拟'
  description: '生物系统和分子模拟'
  icon: '🧬'
}

export interface CosmosSimulation {
  type: 'cosmos'
  path: '/v12/cosmos'
  title: '宇宙模拟'
  description: '宇宙演化和大尺度模拟'
  icon: '🌌'
}

export interface DeepSpace {
  type: 'deepspace'
  path: '/v12/deepspace'
  title: '深空探索'
  description: '深空数据分析和探索'
  icon: '🚀'
}

// 联合类型
export type V12Module = 
  | NLGenerator | NoCodeBuilder | TemplateMarketplace | EducationCenter
  | SmartRecommender | AutoDocumentation | OneClickDeploy
  | AIOpsDashboard | SmartScheduler | SelfHealing | AutomationOps | PerformanceTuner
  | MetaLearning | EmergenceEngine | CrossDomainReasoning | ContinualLearning
  | QuantumSimulator | QuantumOptimizer | QuantumML | HybridCompute
  | ClimateModel | BioSimulation | CosmosSimulation | DeepSpace

// v12 路由配置
export interface V12RouteConfig {
  path: string
  component: React.ComponentType<any>
  title: string
  icon?: string
  group: V12ModuleGroup
  description?: string
  permissions?: string[]
  breadcrumbs?: BreadcrumbItem[]
}

export interface BreadcrumbItem {
  title: string
  path?: string
}

// v12 权限配置
export interface V12Permission {
  resource: string
  action: 'view' | 'create' | 'edit' | 'delete' | 'admin'
  roles: string[]
}
