/**
 * AI Platform - 系统设置页面
 */

import { Card, Form, Input, Select, Switch, Button, Tabs, Row, Col, Divider, Space, Alert, Progress, message } from 'antd'
import { SettingOutlined, BellOutlined, SafetyOutlined, CloudOutlined, DatabaseOutlined } from '@ant-design/icons'
import { useState, useEffect } from 'react'

const { Option } = Select
const { TabPane } = Tabs

interface SystemInfo {
  site_name: string
  site_description: string
  version: string
  language: string
  theme: string
  features: Record<string, boolean>
}

interface StorageInfo {
  max_dataset_size_gb: number
  max_model_size_gb: number
  default_storage_path: string
  used_storage_gb: number
  total_storage_gb: number
}

export default function Settings() {
  const [loading, setLoading] = useState(false)
  const [systemInfo, setSystemInfo] = useState<SystemInfo | null>(null)
  const [storageInfo, setStorageInfo] = useState<StorageInfo | null>(null)
  const [form] = Form.useForm()

  const fetchSettings = async () => {
    try {
      setLoading(true)
      const token = localStorage.getItem('access_token')
      
      const [sysRes, storageRes] = await Promise.all([
        fetch('/api/v1/settings/system', { headers: { 'Authorization': `Bearer ${token}` } }),
        fetch('/api/v1/settings/storage', { headers: { 'Authorization': `Bearer ${token}` } })
      ])
      
      if (sysRes.ok) {
        const data = await sysRes.json()
        setSystemInfo(data)
        form.setFieldsValue({
          site_name: data.site_name,
          site_description: data.site_description,
          language: data.language,
          theme: data.theme
        })
      }
      
      if (storageRes.ok) {
        setStorageInfo(await storageRes.json())
      }
    } catch (error) {
      console.error('获取设置失败:', error)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchSettings()
  }, [])

  const handleSave = async () => {
    try {
      const values = await form.validateFields()
      message.success('设置已保存（模拟）')
    } catch (error) {
      message.error('保存失败')
    }
  }

  const storagePercent = storageInfo 
    ? Math.round((storageInfo.used_storage_gb / storageInfo.total_storage_gb) * 100) 
    : 0

  return (
    <div>
      <Card title={<><SettingOutlined /> 系统设置</>}>
        <Tabs defaultActiveKey="1">
          {/* 基本设置 */}
          <TabPane tab={<span><SettingOutlined /> 基本设置</span>} key="1">
            <Form form={form} layout="vertical" style={{ maxWidth: 600 }}>
              <Form.Item name="site_name" label="平台名称">
                <Input placeholder="AI Platform" />
              </Form.Item>
              
              <Form.Item name="site_description" label="平台描述">
                <Input.TextArea rows={3} placeholder="大模型全生命周期管理平台" />
              </Form.Item>
              
              <Form.Item name="language" label="语言">
                <Select>
                  <Option value="zh-CN">中文</Option>
                  <Option value="en-US">English</Option>
                </Select>
              </Form.Item>
              
              <Form.Item name="theme" label="主题">
                <Select>
                  <Option value="light">浅色</Option>
                  <Option value="dark">深色</Option>
                  <Option value="auto">跟随系统</Option>
                </Select>
              </Form.Item>
              
              <Form.Item name="timezone" label="时区">
                <Select defaultValue="Asia/Shanghai">
                  <Option value="Asia/Shanghai">Asia/Shanghai (UTC+8)</Option>
                  <Option value="UTC">UTC</Option>
                  <Option value="America/New_York">America/New_York (UTC-5)</Option>
                </Select>
              </Form.Item>
              
              <Form.Item>
                <Space>
                  <Button type="primary" onClick={handleSave} loading={loading}>
                    保存设置
                  </Button>
                  <Button>重置</Button>
                </Space>
              </Form.Item>
            </Form>
          </TabPane>

          {/* 存储设置 */}
          <TabPane tab={<span><DatabaseOutlined /> 存储设置</span>} key="2">
            <Row gutter={24}>
              <Col span={12}>
                <Card size="small" title="存储使用情况">
                  <div style={{ marginBottom: 16 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
                      <span>已使用</span>
                      <span>{storageInfo?.used_storage_gb || 0} GB / {storageInfo?.total_storage_gb || 0} GB</span>
                    </div>
                    <Progress 
                      percent={storagePercent} 
                      strokeColor={storagePercent > 80 ? '#ff4d4f' : storagePercent > 60 ? '#faad14' : '#52c41a'}
                    />
                  </div>
                  
                  <Divider />
                  
                  <Form layout="vertical" size="small">
                    <Form.Item label="数据集最大大小 (GB)">
                      <Input type="number" defaultValue={storageInfo?.max_dataset_size_gb || 10} />
                    </Form.Item>
                    <Form.Item label="模型最大大小 (GB)">
                      <Input type="number" defaultValue={storageInfo?.max_model_size_gb || 50} />
                    </Form.Item>
                    <Form.Item label="默认存储路径">
                      <Input defaultValue={storageInfo?.default_storage_path || '/data'} />
                    </Form.Item>
                    <Button type="primary">保存</Button>
                  </Form>
                </Card>
              </Col>
              
              <Col span={12}>
                <Card size="small" title="存储目录">
                  <div style={{ fontFamily: 'monospace', fontSize: 12 }}>
                    <div>/data/datasets</div>
                    <div style={{ color: '#888' }}>存储上传的数据集</div>
                    
                    <Divider />
                    
                    <div>/data/models</div>
                    <div style={{ color: '#888' }}>存储训练好的模型</div>
                    
                    <Divider />
                    
                    <div>/data/logs</div>
                    <div style={{ color: '#888' }}>存储训练日志</div>
                  </div>
                </Card>
              </Col>
            </Row>
          </TabPane>

          {/* GPU阈值 */}
          <TabPane tab={<span><CloudOutlined /> GPU监控设置</span>} key="3">
            <Alert
              type="info"
              message="GPU监控阈值设置"
              description="当GPU使用率或显存使用超过阈值时，系统会发出告警通知。"
              style={{ marginBottom: 24 }}
            />
            
            <Form layout="vertical" style={{ maxWidth: 400 }}>
              <Form.Item label="利用率警告阈值 (%)">
                <Input type="number" defaultValue={80} suffix="%" />
              </Form.Item>
              
              <Form.Item label="利用率严重阈值 (%)">
                <Input type="number" defaultValue={95} suffix="%" />
              </Form.Item>
              
              <Form.Item label="显存警告阈值 (%)">
                <Input type="number" defaultValue={80} suffix="%" />
              </Form.Item>
              
              <Form.Item label="显存严重阈值 (%)">
                <Input type="number" defaultValue={95} suffix="%" />
              </Form.Item>
              
              <Form.Item>
                <Space>
                  <Button type="primary">保存阈值</Button>
                  <Button>恢复默认</Button>
                </Space>
              </Form.Item>
            </Form>
          </TabPane>

          {/* 通知设置 */}
          <TabPane tab={<span><BellOutlined /> 通知设置</span>} key="4">
            <Form layout="vertical" style={{ maxWidth: 400 }}>
              <Form.Item label="邮件通知">
                <Switch checkedChildren="开" unCheckedChildren="关" defaultChecked />
              </Form.Item>
              
              <Form.Item label="GPU告警">
                <Switch checkedChildren="开" unCheckedChildren="关" defaultChecked />
              </Form.Item>
              
              <Form.Item label="任务完成通知">
                <Switch checkedChildren="开" unCheckedChildren="关" defaultChecked />
              </Form.Item>
              
              <Form.Item label="训练失败告警">
                <Switch checkedChildren="开" unCheckedChildren="关" defaultChecked />
              </Form.Item>
              
              <Form.Item label="新用户注册通知">
                <Switch checkedChildren="开" unCheckedChildren="关" />
              </Form.Item>
              
              <Divider />
              
              <Form.Item>
                <Button type="primary">保存通知设置</Button>
              </Form.Item>
            </Form>
          </TabPane>

          {/* 安全设置 */}
          <TabPane tab={<span><SafetyOutlined /> 安全设置</span>} key="5">
            <Alert
              type="warning"
              message="安全设置"
              description="请谨慎修改安全相关设置，确保系统安全。"
              style={{ marginBottom: 24 }}
            />
            
            <Form layout="vertical" style={{ maxWidth: 400 }}>
              <Form.Item label="API密钥轮换周期">
                <Select defaultValue="30">
                  <Option value="7">7天</Option>
                  <Option value="30">30天</Option>
                  <Option value="90">90天</Option>
                  <Option value="365">365天</Option>
                </Select>
              </Form.Item>
              
              <Form.Item label="会话超时 (分钟)">
                <Input type="number" defaultValue={60} />
              </Form.Item>
              
              <Form.Item label="密码最小长度">
                <Input type="number" defaultValue={8} />
              </Form.Item>
              
              <Form.Item label="需要多因素认证">
                <Switch checkedChildren="是" unCheckedChildren="否" />
              </Form.Item>
              
              <Divider />
              
              <Form.Item>
                <Space>
                  <Button type="primary">保存安全设置</Button>
                  <Button danger>重置所有密钥</Button>
                </Space>
              </Form.Item>
            </Form>
          </TabPane>

          {/* 系统信息 */}
          <TabPane tab={<span><SafetyOutlined /> 系统信息</span>} key="6">
            <Card size="small" title="系统信息">
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
                <div><strong>版本:</strong> {systemInfo?.version || '1.0.0'}</div>
                <div><strong>平台名称:</strong> {systemInfo?.site_name || 'AI Platform'}</div>
                <div><strong>语言:</strong> {systemInfo?.language || 'zh-CN'}</div>
                <div><strong>主题:</strong> {systemInfo?.theme || 'light'}</div>
              </div>
              
              <Divider />
              
              <div style={{ marginBottom: 16 }}><strong>已启用功能:</strong></div>
              <Space wrap>
                {systemInfo?.features && Object.entries(systemInfo.features).map(([key, enabled]) => (
                  <Tag key={key} color={enabled ? 'green' : 'default'}>
                    {key.replace('_', ' ')}
                  </Tag>
                ))}
              </Space>
            </Card>
          </TabPane>
        </Tabs>
      </Card>
    </div>
  )
}
