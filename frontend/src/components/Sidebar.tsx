import React from 'react'
import { Menu, Layout } from 'antd'
import { useNavigate, useLocation } from 'react-router-dom'
import { useLang } from '../locales'

const { Sider } = Layout

interface SidebarProps {
  collapsed?: boolean
  onCollapse?: (collapsed: boolean) => void
}

export default function Sidebar({ collapsed = false, onCollapse }: SidebarProps) {
  const navigate = useNavigate()
  const location = useLocation()
  const { t } = useLang()

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
    <Sider 
      collapsible 
      collapsed={collapsed} 
      onCollapse={onCollapse}
      theme="dark"
      style={{ overflow: 'auto', height: '100vh', position: 'fixed', left: 0, top: 0, bottom: 0 }}
    >
      <div 
        className="logo" 
        style={{ 
          height: 32, 
          margin: 16, 
          color: '#fff', 
          fontSize: collapsed ? 12 : 16, 
          fontWeight: 'bold', 
          textAlign: 'center', 
          lineHeight: '32px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          gap: 8,
          overflow: 'hidden',
          whiteSpace: 'nowrap',
        }}
      >
        <span>🚀</span>
        {!collapsed && <span>AI Platform</span>}
      </div>
      <Menu
        theme="dark"
        mode="inline"
        selectedKeys={[selectedKey]}
        items={menuItems}
        onClick={({ key }) => navigate(key)}
        style={{ borderRight: 0, marginTop: 8 }}
      />
    </Sider>
  )
}
