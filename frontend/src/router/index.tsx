/**
 * v12 路由配置
 * AI Platform v12 版本路由
 */
import React from 'react'
import { Navigate } from 'react-router-dom'

// 页面组件
import Dashboard from '../pages/Dashboard'
import Projects from '../pages/Projects'
import Experiments from '../pages/Experiments'
import Tasks from '../pages/Tasks'
import Training from '../pages/Training'
import Inference from '../pages/Inference'
import Datasets from '../pages/Datasets'
import Login from '../pages/Login'

// v2 页面
import { AutoMLPage } from '../pages/v2/AutoML'
import { FeatureStorePage } from '../pages/v2/FeatureStore'
import { NotebooksPage } from '../pages/v2/Notebooks'
import { RAGPage } from '../pages/v2/RAG'

// v8 页面
import { AgentFactoryPage } from '../pages/v8/AgentFactory'
import { KnowledgeGraphPage } from '../pages/v8/KnowledgeGraph'
import { EmbodiedAIPage } from '../pages/v8/EmbodiedAI'
import { AgentCollaborationPage } from '../pages/v8/AgentCollaboration'
import { SecurityPage } from '../pages/v8/Security'
import { PluginMarketplacePage } from '../pages/v8/PluginMarketplace'

// v9 页面
import { AdaptiveLearning } from '../pages/v9/AdaptiveLearning'
import { FederatedLearning } from '../pages/v9/FederatedLearning'
import { DecisionEngine } from '../pages/v9/DecisionEngine'

// v12 页面
import NLGenerator from '../pages/v12/NLGenerator'
import NoCodeBuilder from '../pages/v12/NoCodeBuilder'
import TemplateMarketplace from '../pages/v12/TemplateMarketplace'
import EducationCenter from '../pages/v12/EducationCenter'
import SmartRecommender from '../pages/v12/SmartRecommender'
import AutoDocumentation from '../pages/v12/AutoDocumentation'
import OneClickDeploy from '../pages/v12/OneClickDeploy'
import AIOpsDashboard from '../pages/v12/AIOpsDashboard'
import SmartScheduler from '../pages/v12/SmartScheduler'
import SelfHealing from '../pages/v12/SelfHealing'
import AutomationOps from '../pages/v12/AutomationOps'
import PerformanceTuner from '../pages/v12/PerformanceTuner'
import MetaLearning from '../pages/v12/MetaLearning'
import EmergenceEngine from '../pages/v12/EmergenceEngine'
import CrossDomainReasoning from '../pages/v12/CrossDomainReasoning'
import ContinualLearning from '../pages/v12/ContinualLearning'
import QuantumSimulator from '../pages/v12/QuantumSimulator'
import QuantumOptimizer from '../pages/v12/QuantumOptimizer'
import QuantumML from '../pages/v12/QuantumML'
import HybridCompute from '../pages/v12/HybridCompute'
import ClimateModel from '../pages/v12/ClimateModel'
import BioSimulation from '../pages/v12/BioSimulation'
import CosmosSimulation from '../pages/v12/CosmosSimulation'
import DeepSpace from '../pages/v12/DeepSpace'

// 路由守卫
export const PrivateRoute = ({ children }: { children: React.ReactNode }) => {
  const token = localStorage.getItem('access_token')
  if (!token) {
    return <Navigate to="/login" replace />
  }
  return <>{children}</>
}

// 所有路由配置
export const routes = [
  // 公开路由
  { path: '/login', element: <Login />, requiresAuth: false },
  
  // 根路由重定向
  { path: '/', element: <Navigate to="/dashboard" replace />, requiresAuth: true },
  
  // 核心功能
  { path: '/dashboard', element: <Dashboard />, name: '仪表板', requiresAuth: true },
  { path: '/projects', element: <Projects />, name: '项目', requiresAuth: true },
  { path: '/experiments', element: <Experiments />, name: '实验', requiresAuth: true },
  { path: '/tasks', element: <Tasks />, name: '任务', requiresAuth: true },
  { path: '/training', element: <Training />, name: '训练', requiresAuth: true },
  { path: '/inference', element: <Inference />, name: '推理', requiresAuth: true },
  { path: '/datasets', element: <Datasets />, name: '数据集', requiresAuth: true },
  
  // v2 模块
  { path: '/automl', element: <AutoMLPage />, name: 'AutoML', requiresAuth: true },
  { path: '/feature-store', element: <FeatureStorePage />, name: 'Feature Store', requiresAuth: true },
  { path: '/notebooks', element: <NotebooksPage />, name: 'Notebooks', requiresAuth: true },
  { path: '/rag', element: <RAGPage />, name: 'RAG', requiresAuth: true },
  
  // v8 模块
  { path: '/agent-factory', element: <AgentFactoryPage />, name: 'Agent工厂', requiresAuth: true },
  { path: '/knowledge-graph', element: <KnowledgeGraphPage />, name: '知识图谱', requiresAuth: true },
  { path: '/embodied-ai', element: <EmbodiedAIPage />, name: '具身AI', requiresAuth: true },
  { path: '/collaboration', element: <AgentCollaborationPage />, name: 'Agent协作', requiresAuth: true },
  { path: '/security', element: <SecurityPage />, name: '安全中心', requiresAuth: true },
  { path: '/plugin-marketplace', element: <PluginMarketplacePage />, name: 'Plugin市场', requiresAuth: true },
  
  // v9 模块
  { path: '/v9/adaptive', element: <AdaptiveLearning />, name: '自适应学习', requiresAuth: true },
  { path: '/v9/federated', element: <FederatedLearning />, name: '联邦学习', requiresAuth: true },
  { path: '/v9/decision', element: <DecisionEngine />, name: '决策引擎', requiresAuth: true },
  
  // v12 模块 - AI民主化
  { path: '/v12/nl-generator', element: <NLGenerator />, name: '自然语言生成器', requiresAuth: true, group: 'democratization' },
  { path: '/v12/no-code', element: <NoCodeBuilder />, name: '无代码构建器', requiresAuth: true, group: 'democratization' },
  { path: '/v12/templates', element: <TemplateMarketplace />, name: '模板市场', requiresAuth: true, group: 'democratization' },
  { path: '/v12/education', element: <EducationCenter />, name: '教育中心', requiresAuth: true, group: 'democratization' },
  { path: '/v12/recommender', element: <SmartRecommender />, name: '智能推荐', requiresAuth: true, group: 'democratization' },
  { path: '/v12/auto-doc', element: <AutoDocumentation />, name: '自动文档', requiresAuth: true, group: 'democratization' },
  { path: '/v12/deploy', element: <OneClickDeploy />, name: '一键部署', requiresAuth: true, group: 'democratization' },
  
  // v12 模块 - 超自动化
  { path: '/v12/aiops', element: <AIOpsDashboard />, name: 'AIOps仪表板', requiresAuth: true, group: 'hyperautomation' },
  { path: '/v12/scheduler', element: <SmartScheduler />, name: '智能调度', requiresAuth: true, group: 'hyperautomation' },
  { path: '/v12/self-healing', element: <SelfHealing />, name: '自愈系统', requiresAuth: true, group: 'hyperautomation' },
  { path: '/v12/automation', element: <AutomationOps />, name: '自动化运维', requiresAuth: true, group: 'hyperautomation' },
  { path: '/v12/performance', element: <PerformanceTuner />, name: '性能调优', requiresAuth: true, group: 'hyperautomation' },
  
  // v12 模块 - 超级智能
  { path: '/v12/meta-learning', element: <MetaLearning />, name: '元学习', requiresAuth: true, group: 'superintelligence' },
  { path: '/v12/emergence', element: <EmergenceEngine />, name: '涌现引擎', requiresAuth: true, group: 'superintelligence' },
  { path: '/v12/cross-domain', element: <CrossDomainReasoning />, name: '跨域推理', requiresAuth: true, group: 'superintelligence' },
  { path: '/v12/continual', element: <ContinualLearning />, name: '持续学习', requiresAuth: true, group: 'superintelligence' },
  
  // v12 模块 - 量子AI
  { path: '/v12/quantum-sim', element: <QuantumSimulator />, name: '量子模拟器', requiresAuth: true, group: 'quantum' },
  { path: '/v12/quantum-opt', element: <QuantumOptimizer />, name: '量子优化', requiresAuth: true, group: 'quantum' },
  { path: '/v12/quantum-ml', element: <QuantumML />, name: '量子机器学习', requiresAuth: true, group: 'quantum' },
  { path: '/v12/hybrid', element: <HybridCompute />, name: '混合计算', requiresAuth: true, group: 'quantum' },
  
  // v12 模块 - 宇宙级AI
  { path: '/v12/climate', element: <ClimateModel />, name: '气候模型', requiresAuth: true, group: 'cosmos' },
  { path: '/v12/bio', element: <BioSimulation />, name: '生物模拟', requiresAuth: true, group: 'cosmos' },
  { path: '/v12/cosmos', element: <CosmosSimulation />, name: '宇宙模拟', requiresAuth: true, group: 'cosmos' },
  { path: '/v12/deepspace', element: <DeepSpace />, name: '深空探索', requiresAuth: true, group: 'cosmos' },
]

// 面包屑生成函数
export function getBreadcrumbs(pathname: string) {
  const breadcrumbs = []
  
  // 添加首页
  breadcrumbs.push({ title: '首页', path: '/dashboard' })
  
  // 查找匹配的路由
  const route = routes.find(r => r.path === pathname)
  if (route) {
    breadcrumbs.push({ title: route.name, path: route.path })
  }
  
  return breadcrumbs
}

// v12 模块分组面包屑
export function getV12GroupBreadcrumbs(group: string) {
  const groupNames: Record<string, string> = {
    democratization: 'AI民主化',
    hyperautomation: '超自动化',
    superintelligence: '超级智能',
    quantum: '量子AI',
    cosmos: '宇宙级AI',
  }
  
  return [
    { title: '首页', path: '/dashboard' },
    { title: 'v12', path: '/v12' },
    { title: groupNames[group] || group, path: `/v12/${group}` },
  ]
}
