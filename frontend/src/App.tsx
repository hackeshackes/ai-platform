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
import DatasetVersions from './pages/DatasetVersions'
import DataQuality from './pages/DataQuality'
import Users from './pages/Users'
import Roles from './pages/Roles'
import Models from './pages/Models'
import Settings from './pages/Settings'
import Login from './pages/Login'
import GPU from './pages/GPU'
import LossChart from './pages/LossChart'

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
    { key: '/gpu', label: 'GPU监控' },
    { key: '/loss', label: 'Loss曲线' },
    { key: '/experiments', label: t('nav.experiments') },
    { key: '/tasks', label: t('nav.tasks') },
    { key: '/training', label: t('nav.training') },
    { key: '/inference', label: t('nav.inference') },
    { key: '/datasets', label: t('nav.datasets') },
    { key: '/dataset-versions', label: '版本管理' },
    { key: '/data-quality', label: '数据质量' },
    { key: '/users', label: '用户管理' },
    { key: '/roles', label: '权限管理' },
    { key: '/models', label: t('nav.models') },
    { key: '/settings', label: t('nav.settings') },
  ]

  const selectedKey = menuItems.find(item => location.pathname.startsWith(item.key))?.key || '/dashboard'

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
            <Route path="/gpu" element={<PrivateRoute><GPU /></PrivateRoute>} />
            <Route path="/loss" element={<PrivateRoute><LossChart /></PrivateRoute>} />
            <Route path="/experiments" element={<PrivateRoute><Experiments /></PrivateRoute>} />
            <Route path="/tasks" element={<PrivateRoute><Tasks /></PrivateRoute>} />
            <Route path="/training" element={<PrivateRoute><Training /></PrivateRoute>} />
            <Route path="/inference" element={<PrivateRoute><Inference /></PrivateRoute>} />
            <Route path="/datasets" element={<PrivateRoute><Datasets /></PrivateRoute>} />
            <Route path="/dataset-versions" element={<PrivateRoute><DatasetVersions /></PrivateRoute>} />
            <Route path="/data-quality" element={<PrivateRoute><DataQuality /></PrivateRoute>} />
            <Route path="/users" element={<PrivateRoute><Users /></PrivateRoute>} />
            <Route path="/roles" element={<PrivateRoute><Roles /></PrivateRoute>} />
            <Route path="/models" element={<PrivateRoute><Models /></PrivateRoute>} />
            <Route path="/settings" element={<PrivateRoute><Settings /></PrivateRoute>} />
          </Routes>
        </Content>
      </ALayout>
    </ALayout>
  )
}
