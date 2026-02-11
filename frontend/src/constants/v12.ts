/**
 * v12 常量定义
 * AI Platform v12 版本常量
 */

import { V12ModuleGroup, V12RouteConfig } from '../types/v12'

// v12 模块分组配置
export const V12_MODULE_GROUPS: Record<V12ModuleGroup, { 
  title: string
  icon: string
  description: string
  order: number
}> = {
  democratization: {
    title: 'AI民主化',
    icon: '🌐',
    description: '让AI技术触手可及',
    order: 1,
  },
  hyperautomation: {
    title: '超自动化',
    icon: '⚡',
    description: 'AI驱动的全面自动化',
    order: 2,
  },
  superintelligence: {
    title: '超级智能',
    icon: '🧠',
    description: '下一代AI能力',
    order: 3,
  },
  quantum: {
    title: '量子AI',
    icon: '⚛️',
    description: '量子计算与AI的融合',
    order: 4,
  },
  cosmos: {
    title: '宇宙级AI',
    icon: '🌌',
    description: '探索AI的极限边界',
    order: 5,
  },
}

// v12 路由配置
export const V12_ROUTES: V12RouteConfig[] = [
  // AI民主化模块
  {
    path: '/v12/nl-generator',
    component: () => import('../pages/v12/NLGenerator'),
    title: '自然语言生成器',
    group: 'democratization',
    icon: '💬',
    description: '将自然语言转换为代码和工作流',
  },
  {
    path: '/v12/no-code',
    component: () => import('../pages/v12/NoCodeBuilder'),
    title: '无代码构建器',
    group: 'democratization',
    icon: '🧱',
    description: '可视化拖拽构建AI应用',
  },
  {
    path: '/v12/templates',
    component: () => import('../pages/v12/TemplateMarketplace'),
    title: '模板市场',
    group: 'democratization',
    icon: '📦',
    description: '分享和使用AI应用模板',
  },
  {
    path: '/v12/education',
    component: () => import('../pages/v12/EducationCenter'),
    title: '教育中心',
    group: 'democratization',
    icon: '🎓',
    description: 'AI学习和培训资源',
  },
  {
    path: '/v12/recommender',
    component: () => import('../pages/v12/SmartRecommender'),
    title: '智能推荐',
    group: 'democratization',
    icon: '💡',
    description: '个性化AI解决方案推荐',
  },
  {
    path: '/v12/auto-doc',
    component: () => import('../pages/v12/AutoDocumentation'),
    title: '自动文档',
    group: 'democratization',
    icon: '📝',
    description: 'AI应用自动文档生成',
  },
  {
    path: '/v12/deploy',
    component: () => import('../pages/v12/OneClickDeploy'),
    title: '一键部署',
    group: 'democratization',
    icon: '🚀',
    description: '快速部署AI应用到生产环境',
  },
  
  // 超自动化模块
  {
    path: '/v12/aiops',
    component: () => import('../pages/v12/AIOpsDashboard'),
    title: 'AIOps仪表板',
    group: 'hyperautomation',
    icon: '📊',
    description: 'AI驱动的运维监控和告警',
  },
  {
    path: '/v12/scheduler',
    component: () => import('../pages/v12/SmartScheduler'),
    title: '智能调度',
    group: 'hyperautomation',
    icon: '⏰',
    description: 'AI优化的任务调度系统',
  },
  {
    path: '/v12/self-healing',
    component: () => import('../pages/v12/SelfHealing'),
    title: '自愈系统',
    group: 'hyperautomation',
    icon: '🔧',
    description: '自动化故障检测和恢复',
  },
  {
    path: '/v12/automation',
    component: () => import('../pages/v12/AutomationOps'),
    title: '自动化运维',
    group: 'hyperautomation',
    icon: '⚙️',
    description: '全面的IT自动化解决方案',
  },
  {
    path: '/v12/performance',
    component: () => import('../pages/v12/PerformanceTuner'),
    title: '性能调优',
    group: 'hyperautomation',
    icon: '📈',
    description: 'AI驱动的性能优化建议',
  },
  
  // 超级智能模块
  {
    path: '/v12/meta-learning',
    component: () => import('../pages/v12/MetaLearning'),
    title: '元学习',
    group: 'superintelligence',
    icon: '🧠',
    description: '学会学习的AI系统',
  },
  {
    path: '/v12/emergence',
    component: () => import('../pages/v12/EmergenceEngine'),
    title: '涌现引擎',
    group: 'superintelligence',
    icon: '✨',
    description: '探索AI系统的涌现行为',
  },
  {
    path: '/v12/cross-domain',
    component: () => import('../pages/v12/CrossDomainReasoning'),
    title: '跨域推理',
    group: 'superintelligence',
    icon: '🌐',
    description: '跨领域知识迁移和推理',
  },
  {
    path: '/v12/continual',
    component: () => import('../pages/v12/ContinualLearning'),
    title: '持续学习',
    group: 'superintelligence',
    icon: '🔄',
    description: '增量学习和知识积累',
  },
  
  // 量子AI模块
  {
    path: '/v12/quantum-sim',
    component: () => import('../pages/v12/QuantumSimulator'),
    title: '量子模拟器',
    group: 'quantum',
    icon: '⚛️',
    description: '量子电路模拟和测试',
  },
  {
    path: '/v12/quantum-opt',
    component: () => import('../pages/v12/QuantumOptimizer'),
    title: '量子优化',
    group: 'quantum',
    icon: '🎯',
    description: '量子优化算法实现',
  },
  {
    path: '/v12/quantum-ml',
    component: () => import('../pages/v12/QuantumML'),
    title: '量子机器学习',
    group: 'quantum',
    icon: '🔮',
    description: '量子增强的机器学习',
  },
  {
    path: '/v12/hybrid',
    component: () => import('../pages/v12/HybridCompute'),
    title: '混合计算',
    group: 'quantum',
    icon: '🔀',
    description: '经典和量子混合计算',
  },
  
  // 宇宙级AI模块
  {
    path: '/v12/climate',
    component: () => import('../pages/v12/ClimateModel'),
    title: '气候模型',
    group: 'cosmos',
    icon: '🌍',
    description: 'AI驱动的气候预测和模拟',
  },
  {
    path: '/v12/bio',
    component: () => import('../pages/v12/BioSimulation'),
    title: '生物模拟',
    group: 'cosmos',
    icon: '🧬',
    description: '生物系统和分子模拟',
  },
  {
    path: '/v12/cosmos',
    component: () => import('../pages/v12/CosmosSimulation'),
    title: '宇宙模拟',
    group: 'cosmos',
    icon: '🌌',
    description: '宇宙演化和大尺度模拟',
  },
  {
    path: '/v12/deepspace',
    component: () => import('../pages/v12/DeepSpace'),
    title: '深空探索',
    group: 'cosmos',
    icon: '🚀',
    description: '深空数据分析和探索',
  },
]

// 侧边栏菜单配置
export const V12_SIDEBAR_MENU = [
  {
    group: 'democratization',
    title: 'AI民主化',
    icon: '🌐',
    items: [
      { key: '/v12/nl-generator', label: '💬 自然语言生成器' },
      { key: '/v12/no-code', label: '🧱 无代码构建器' },
      { key: '/v12/templates', label: '📦 模板市场' },
      { key: '/v12/education', label: '🎓 教育中心' },
      { key: '/v12/recommender', label: '💡 智能推荐' },
      { key: '/v12/auto-doc', label: '📝 自动文档' },
      { key: '/v12/deploy', label: '🚀 一键部署' },
    ],
  },
  {
    group: 'hyperautomation',
    title: '超自动化',
    icon: '⚡',
    items: [
      { key: '/v12/aiops', label: '📊 AIOps仪表板' },
      { key: '/v12/scheduler', label: '⏰ 智能调度' },
      { key: '/v12/self-healing', label: '🔧 自愈系统' },
      { key: '/v12/automation', label: '⚙️ 自动化运维' },
      { key: '/v12/performance', label: '📈 性能调优' },
    ],
  },
  {
    group: 'superintelligence',
    title: '超级智能',
    icon: '🧠',
    items: [
      { key: '/v12/meta-learning', label: '🧠 元学习' },
      { key: '/v12/emergence', label: '✨ 涌现引擎' },
      { key: '/v12/cross-domain', label: '🌐 跨域推理' },
      { key: '/v12/continual', label: '🔄 持续学习' },
    ],
  },
  {
    group: 'quantum',
    title: '量子AI',
    icon: '⚛️',
    items: [
      { key: '/v12/quantum-sim', label: '⚛️ 量子模拟器' },
      { key: '/v12/quantum-opt', label: '🎯 量子优化' },
      { key: '/v12/quantum-ml', label: '🔮 量子机器学习' },
      { key: '/v12/hybrid', label: '🔀 混合计算' },
    ],
  },
  {
    group: 'cosmos',
    title: '宇宙级AI',
    icon: '🌌',
    items: [
      { key: '/v12/climate', label: '🌍 气候模型' },
      { key: '/v12/bio', label: '🧬 生物模拟' },
      { key: '/v12/cosmos', label: '🌌 宇宙模拟' },
      { key: '/v12/deepspace', label: '🚀 深空探索' },
    ],
  },
]

// 权限配置
export const V12_PERMISSIONS = {
  democratization: ['user', 'admin'],
  hyperautomation: ['admin', 'operator'],
  superintelligence: ['admin', 'researcher'],
  quantum: ['admin', 'researcher'],
  cosmos: ['admin', 'researcher'],
}

// v12 入口页面配置
export const V12_ENTRY_CONFIG = {
  title: '🚀 v12 新功能',
  subtitle: '探索下一代AI能力',
  quickStart: [
    {
      title: '快速开始',
      description: '5分钟内创建您的第一个AI应用',
      icon: '⚡',
      path: '/v12/no-code',
    },
    {
      title: '模板市场',
      description: '从模板库中选择合适的应用',
      icon: '📦',
      path: '/v12/templates',
    },
    {
      title: '学习路径',
      description: '系统化的AI学习资源',
      icon: '🎓',
      path: '/v12/education',
    },
  ],
}
