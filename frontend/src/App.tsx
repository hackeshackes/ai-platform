import { Routes, Route, useNavigate, useLocation, Navigate } from 'react-router-dom'
import { Layout as ALayout, Menu } from 'antd'
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

// 路由守卫
const PrivateRoute = ({ children }: { children: React.ReactNode }) => {
  const token = localStorage.getItem('access_token')
  if (!token) {
    return <Navigate to="/login" replace />
  }
  return <>{children}</>
}

const { Sider, Content, Header } = ALayout

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
  ]

  const selectedKey = menuItems.find(item => item.key && location.pathname.startsWith(item.key))?.key || '/dashboard'

  return (
    <ALayout style={{ minHeight: '100vh' }}>
      <Sider collapsible theme="dark" onCollapse={(collapsed) => collapsed}>
        <div className="logo" style={{ height: 32, margin: 16, color: '#fff', fontSize: 18, fontWeight: 'bold', textAlign: 'center', lineHeight: '32px' }}>
          AI Platform
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
          <span style={{ fontSize: 18, fontWeight: 500 }}>{t('app.title')}</span>
          <span 
            style={{ cursor: 'pointer', padding: '8px 16px', borderRadius: 4, background: '#f5f5f5', userSelect: 'none' }}
            onClick={() => setLang(lang === 'zh' ? 'en' : 'zh')}
          >
            {lang === 'zh' ? '🇨🇳 中文' : '🇺🇸 English'}
          </span>
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
          </Routes>
        </Content>
      </ALayout>
    </ALayout>
  )
}
