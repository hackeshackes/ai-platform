import { Routes, Route, useNavigate, useLocation, Navigate } from 'react-router-dom'
import { Layout as ALayout, Menu, Badge, Typography, Space } from 'antd'
import { useLang } from './locales'

// 页面组件
import Dashboard from './pages/Dashboard'
import Projects from './pages/Projects'
import Experiments from './pages/Experiments'
import Tasks from './pages/Tasks'
import Training from './pages/Training'
import Inference from './pages/Inference'
import Datasets from './pages/Datasets'
import Login from './pages/Login'

// v2 页面
import { AutoMLPage } from './pages/v2/AutoML'
import { FeatureStorePage } from './pages/v2/FeatureStore'
import { NotebooksPage } from './pages/v2/Notebooks'
import { RAGPage } from './pages/v2/RAG'

// v8 页面
import { AgentFactoryPage } from './pages/v8/AgentFactory'
import { KnowledgeGraphPage } from './pages/v8/KnowledgeGraph'
import { EmbodiedAIPage } from './pages/v8/EmbodiedAI'
import { AgentCollaborationPage } from './pages/v8/AgentCollaboration'
import { SecurityPage } from './pages/v8/Security'
import { PluginMarketplacePage } from './pages/v8/PluginMarketplace'

// v9 页面
import { AdaptiveLearning } from './pages/v9/AdaptiveLearning'
import { FederatedLearning } from './pages/v9/FederatedLearning'
import { DecisionEngine } from './pages/v9/DecisionEngine'

// v12 页面
import NLGenerator from './pages/v12/NLGenerator'
import NoCodeBuilder from './pages/v12/NoCodeBuilder'
import TemplateMarketplace from './pages/v12/TemplateMarketplace'
import EducationCenter from './pages/v12/EducationCenter'
import SmartRecommender from './pages/v12/SmartRecommender'
import AutoDocumentation from './pages/v12/AutoDocumentation'
import OneClickDeploy from './pages/v12/OneClickDeploy'
import AIOpsDashboard from './pages/v12/AIOpsDashboard'
import SmartScheduler from './pages/v12/SmartScheduler'
import SelfHealing from './pages/v12/SelfHealing'
import AutomationOps from './pages/v12/AutomationOps'
import PerformanceTuner from './pages/v12/PerformanceTuner'
import MetaLearning from './pages/v12/MetaLearning'
import EmergenceEngine from './pages/v12/EmergenceEngine'
import CrossDomainReasoning from './pages/v12/CrossDomainReasoning'
import ContinualLearning from './pages/v12/ContinualLearning'
import QuantumSimulator from './pages/v12/QuantumSimulator'
import QuantumOptimizer from './pages/v12/QuantumOptimizer'
import QuantumML from './pages/v12/QuantumML'
import HybridCompute from './pages/v12/HybridCompute'
import ClimateModel from './pages/v12/ClimateModel'
import BioSimulation from './pages/v12/BioSimulation'
import CosmosSimulation from './pages/v12/CosmosSimulation'
import DeepSpace from './pages/v12/DeepSpace'


// 路由守卫
const PrivateRoute = ({ children }: { children: React.ReactNode }) => {
  const token = localStorage.getItem('access_token')
  if (!token) {
    return <Navigate to="/login" replace />
  }
  return <>{children}</>
}

const { Text } = Typography
const { Sider, Content, Header } = ALayout

// v12 子菜单渲染
const renderV12SubMenu = (items: { key: string; label: string }[]) => {
  return items.map(item => ({
    key: item.key,
    label: item.label,
  }))
}

export default function App() {
  const { t, lang, setLang } = useLang()
  const navigate = useNavigate()
  const location = useLocation()
  
  const menuItems = [
    { key: '/dashboard', label: t('nav.dashboard') },
    { key: '/projects', label: t('nav.projects') },
    { key: '/experiments', label: t('nav.experiments') },
    { key: '/tasks', label: t('nav.tasks') },
    { key: '/training', label: t('nav.training') },
    { key: '/inference', label: t('nav.inference') },
    { key: '/datasets', label: t('nav.datasets') },
    { type: 'divider' },
    { key: '/automl', label: 'AutoML' },
    { key: '/feature-store', label: 'Feature Store' },
    { key: '/notebooks', label: 'Notebooks' },
    { key: '/rag', label: 'RAG' },
    { type: 'divider' },
    { key: '/agent-factory', label: '🤖 Agent工厂' },
    { key: '/knowledge-graph', label: '🧠 知识图谱' },
    { key: '/embodied-ai', label: '🦾 具身AI' },
    { key: '/collaboration', label: '👥 Agent协作' },
    { key: '/security', label: '🛡️ 安全中心' },
    { key: '/plugin-marketplace', label: '🧩 Plugin市场' },
    { key: '/v9/adaptive', label: '🧠 自适应学习' },
    { key: '/v9/federated', label: '🔗 联邦学习' },
    { key: '/v9/decision', label: '🎯 决策引擎' },
    { type: 'divider' },
    // v12 模块分组
    {
      key: 'v12-democratization',
      label: '🌐 AI民主化',
      icon: null,
      children: [
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
      key: 'v12-hyperautomation',
      label: '⚡ 超自动化',
      icon: null,
      children: [
        { key: '/v12/aiops', label: '📊 AIOps仪表板' },
        { key: '/v12/scheduler', label: '⏰ 智能调度' },
        { key: '/v12/self-healing', label: '🔧 自愈系统' },
        { key: '/v12/automation', label: '⚙️ 自动化运维' },
        { key: '/v12/performance', label: '📈 性能调优' },
      ],
    },
    {
      key: 'v12-superintelligence',
      label: '🧠 超级智能',
      icon: null,
      children: [
        { key: '/v12/meta-learning', label: '🧠 元学习' },
        { key: '/v12/emergence', label: '✨ 涌现引擎' },
        { key: '/v12/cross-domain', label: '🌐 跨域推理' },
        { key: '/v12/continual', label: '🔄 持续学习' },
      ],
    },
    {
      key: 'v12-quantum',
      label: '⚛️ 量子AI',
      icon: null,
      children: [
        { key: '/v12/quantum-sim', label: '⚛️ 量子模拟器' },
        { key: '/v12/quantum-opt', label: '🎯 量子优化' },
        { key: '/v12/quantum-ml', label: '🔮 量子机器学习' },
        { key: '/v12/hybrid', label: '🔀 混合计算' },
      ],
    },
    {
      key: 'v12-cosmos',
      label: '🌌 宇宙级AI',
      icon: null,
      children: [
        { key: '/v12/climate', label: '🌍 气候模型' },
        { key: '/v12/bio', label: '🧬 生物模拟' },
        { key: '/v12/cosmos', label: '🌌 宇宙模拟' },
        { key: '/v12/deepspace', label: '🚀 深空探索' },
      ],
    },
  ]

  const selectedKey = menuItems.find(item => {
    if (item.key && location.pathname.startsWith(item.key)) return true
    if (item.children) {
      return item.children.some(child => location.pathname.startsWith(child.key))
    }
    return false
  })?.key || '/dashboard'

  return (
    <ALayout style={{ minHeight: '100vh' }}>
      <Sider collapsible theme="dark" onCollapse={(collapsed) => collapsed}>
        <div className="logo" style={{ height: 32, margin: 16, color: '#fff', fontSize: 16, fontWeight: 'bold', textAlign: 'center', lineHeight: '32px', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8 }}>
          <span>🚀</span>
          <span>AI Platform</span>
        </div>
        <Menu 
          theme="dark" 
          mode="inline" 
          selectedKeys={[selectedKey]} 
          items={menuItems}
          onClick={({ key }) => navigate(key)}
          style={{ borderRight: 0 }}
        />
      </Sider>
      <ALayout>
        <Header style={{ background: '#fff', padding: '0 24px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid #f0f0f0' }}>
          <Space>
            <span style={{ fontSize: 18, fontWeight: 500 }}>{t('app.title')}</span>
            <Badge count="v12" style={{ backgroundColor: '#52c41a', marginLeft: 8 }} />
          </Space>
          <Space>
            <Text type="secondary">v12 新功能</Text>
            <span 
              style={{ cursor: 'pointer', padding: '8px 16px', borderRadius: 4, background: '#f5f5f5', userSelect: 'none' }}
              onClick={() => setLang(lang === 'zh' ? 'en' : 'zh')}
            >
              {lang === 'zh' ? '🇨🇳 中文' : '🇺🇸 English'}
            </span>
          </Space>
        </Header>
        <Content style={{ margin: 24 }}>
          <Routes>
            <Route path="/login" element={<Login />} />
            <Route path="/" element={<PrivateRoute><Dashboard /></PrivateRoute>} />
            <Route path="/dashboard" element={<PrivateRoute><Dashboard /></PrivateRoute>} />
            <Route path="/projects" element={<PrivateRoute><Projects /></PrivateRoute>} />
            <Route path="/experiments" element={<PrivateRoute><Experiments /></PrivateRoute>} />
            <Route path="/tasks" element={<PrivateRoute><Tasks /></PrivateRoute>} />
            <Route path="/training" element={<PrivateRoute><Training /></PrivateRoute>} />
            <Route path="/inference" element={<PrivateRoute><Inference /></PrivateRoute>} />
            <Route path="/datasets" element={<PrivateRoute><Datasets /></PrivateRoute>} />
            
            {/* v2 Pages */}
            <Route path="/automl" element={<PrivateRoute><AutoMLPage /></PrivateRoute>} />
            <Route path="/feature-store" element={<PrivateRoute><FeatureStorePage /></PrivateRoute>} />
            <Route path="/notebooks" element={<PrivateRoute><NotebooksPage /></PrivateRoute>} />
            <Route path="/rag" element={<PrivateRoute><RAGPage /></PrivateRoute>} />
            
            {/* v8 Pages */}
            <Route path="/agent-factory" element={<PrivateRoute><AgentFactoryPage /></PrivateRoute>} />
            <Route path="/knowledge-graph" element={<PrivateRoute><KnowledgeGraphPage /></PrivateRoute>} />
            <Route path="/embodied-ai" element={<PrivateRoute><EmbodiedAIPage /></PrivateRoute>} />
            <Route path="/collaboration" element={<PrivateRoute><AgentCollaborationPage /></PrivateRoute>} />
            <Route path="/security" element={<PrivateRoute><SecurityPage /></PrivateRoute>} />
            <Route path="/plugin-marketplace" element={<PrivateRoute><PluginMarketplacePage /></PrivateRoute>} />

            {/* v9 Pages */}
            <Route path="/v9/adaptive" element={<PrivateRoute><AdaptiveLearning /></PrivateRoute>} />
            <Route path="/v9/federated" element={<PrivateRoute><FederatedLearning /></PrivateRoute>} />
            <Route path="/v9/decision" element={<PrivateRoute><DecisionEngine /></PrivateRoute>} />

            {/* v12 Pages - AI民主化 */}
            <Route path="/v12/nl-generator" element={<PrivateRoute><NLGenerator /></PrivateRoute>} />
            <Route path="/v12/no-code" element={<PrivateRoute><NoCodeBuilder /></PrivateRoute>} />
            <Route path="/v12/templates" element={<PrivateRoute><TemplateMarketplace /></PrivateRoute>} />
            <Route path="/v12/education" element={<PrivateRoute><EducationCenter /></PrivateRoute>} />
            <Route path="/v12/recommender" element={<PrivateRoute><SmartRecommender /></PrivateRoute>} />
            <Route path="/v12/auto-doc" element={<PrivateRoute><AutoDocumentation /></PrivateRoute>} />
            <Route path="/v12/deploy" element={<PrivateRoute><OneClickDeploy /></PrivateRoute>} />

            {/* v12 Pages - 超自动化 */}
            <Route path="/v12/aiops" element={<PrivateRoute><AIOpsDashboard /></PrivateRoute>} />
            <Route path="/v12/scheduler" element={<PrivateRoute><SmartScheduler /></PrivateRoute>} />
            <Route path="/v12/self-healing" element={<PrivateRoute><SelfHealing /></PrivateRoute>} />
            <Route path="/v12/automation" element={<PrivateRoute><AutomationOps /></PrivateRoute>} />
            <Route path="/v12/performance" element={<PrivateRoute><PerformanceTuner /></PrivateRoute>} />

            {/* v12 Pages - 超级智能 */}
            <Route path="/v12/meta-learning" element={<PrivateRoute><MetaLearning /></PrivateRoute>} />
            <Route path="/v12/emergence" element={<PrivateRoute><EmergenceEngine /></PrivateRoute>} />
            <Route path="/v12/cross-domain" element={<PrivateRoute><CrossDomainReasoning /></PrivateRoute>} />
            <Route path="/v12/continual" element={<PrivateRoute><ContinualLearning /></PrivateRoute>} />

            {/* v12 Pages - 量子AI */}
            <Route path="/v12/quantum-sim" element={<PrivateRoute><QuantumSimulator /></PrivateRoute>} />
            <Route path="/v12/quantum-opt" element={<PrivateRoute><QuantumOptimizer /></PrivateRoute>} />
            <Route path="/v12/quantum-ml" element={<PrivateRoute><QuantumML /></PrivateRoute>} />
            <Route path="/v12/hybrid" element={<PrivateRoute><HybridCompute /></PrivateRoute>} />

            {/* v12 Pages - 宇宙级AI */}
            <Route path="/v12/climate" element={<PrivateRoute><ClimateModel /></PrivateRoute>} />
            <Route path="/v12/bio" element={<PrivateRoute><BioSimulation /></PrivateRoute>} />
            <Route path="/v12/cosmos" element={<PrivateRoute><CosmosSimulation /></PrivateRoute>} />
            <Route path="/v12/deepspace" element={<PrivateRoute><DeepSpace /></PrivateRoute>} />
          </Routes>
        </Content>
      </ALayout>
    </ALayout>
  )
}
