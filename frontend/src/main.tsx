import React from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import { ConfigProvider } from 'antd'
import App from './App'
import { LangProvider } from './locales'
import './index.css'

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <BrowserRouter>
      <LangProvider>
        <ConfigProvider theme={{ token: { colorPrimary: '#0066CC' } }}>
          <App />
        </ConfigProvider>
      </LangProvider>
    </BrowserRouter>
  </React.StrictMode>
)
