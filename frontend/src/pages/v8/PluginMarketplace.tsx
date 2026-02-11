// Plugin市场页面 - v8
import React, { useState, useEffect } from 'react'
import { Card, Table, Button, Modal, Form, Input, Select, Tag, message, Space, Tabs, Rate, Row, Col, Badge, Avatar, List, Empty } from 'antd'
import { PlusOutlined, SearchOutlined, DownloadOutlined, DeleteOutlined, StarOutlined, AppstoreOutlined, TagsOutlined } from '@ant-design/icons'

const { Option } = Select
const { TabPane } = Tabs
const { TextArea } = Input

interface PluginInfo {
  id: string
  name: string
  display_name: string
  description: string
  version: string
  author: string
  category: string
  tags: string[]
  downloads: number
  rating: number
  reviews_count: number
  installed: boolean
}

interface PluginDetail extends PluginInfo {
  permissions: string[]
  dependencies: Record<string, string>
  readme?: string
}

const categoryMap: Record<string, { color: string, icon: string }> = {
  tool: { color: 'blue', icon: '🔧' },
  agent: { color: 'green', icon: '🤖' },
  integration: { color: 'orange', icon: '🔗' },
  ui: { color: 'purple', icon: '🎨' },
  visualization: { color: 'cyan', icon: '📊' },
  data_source: { color: 'magenta', icon: '📁' }
}

export function PluginMarketplacePage() {
  const [plugins, setPlugins] = useState<PluginInfo[]>([])
  const [installedPlugins, setInstalledPlugins] = useState<PluginInfo[]>([])
  const [loading, setLoading] = useState(false)
  const [searchLoading, setSearchLoading] = useState(false)
  const [detailModal, setDetailModal] = useState(false)
  const [publishModal, setPublishModal] = useState(false)
  const [selectedPlugin, setSelectedPlugin] = useState<PluginDetail | null>(null)
  const [form] = Form.useForm()
  const [publishForm] = Form.useForm()

  // 获取Plugin列表
  const fetchPlugins = async () => {
    setLoading(true)
    try {
      const res = await fetch('http://localhost:8000/api/v1/plugins/marketplace')
      const data = await res.json()
      if (Array.isArray(data)) {
        setPlugins(data)
      }
    } catch (e) {
      message.error('获取Plugin列表失败')
    }
    setLoading(false)
  }

  // 获取已安装Plugin
  const fetchInstalledPlugins = async () => {
    try {
      const res = await fetch('http://localhost:8000/api/v1/plugins/marketplace/installed')
      const data = await res.json()
      if (Array.isArray(data)) {
        setInstalledPlugins(data)
      }
    } catch (e) {
      console.error(e)
    }
  }

  // 获取Plugin详情
  const fetchPluginDetail = async (pluginId: string) => {
    try {
      const res = await fetch(`http://localhost:8000/api/v1/plugins/marketplace/${pluginId}`)
      const data = await res.json()
      setSelectedPlugin(data)
      setDetailModal(true)
    } catch (e) {
      message.error('获取详情失败')
    }
  }

  // 搜索Plugin
  const handleSearch = async (values: any) => {
    setSearchLoading(true)
    try {
      const query = new URLSearchParams({
        q: values.keyword || '',
        category: values.category || '',
        tags: values.tags || ''
      }).toString()
      const res = await fetch(`http://localhost:8000/api/v1/plugins/marketplace/search?${query}`)
      const data = await res.json()
      if (data.plugins) {
        setPlugins(data.plugins)
      }
    } catch (e) {
      message.error('搜索失败')
    }
    setSearchLoading(false)
  }

  // 安装Plugin
  const handleInstall = async (pluginId: string) => {
    try {
      const res = await fetch('http://localhost:8000/api/v1/plugins/marketplace/install', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ plugin_id: pluginId })
      })
      const data = await res.json()
      if (data.success) {
        message.success(`Plugin安装成功`)
        fetchPlugins()
        fetchInstalledPlugins()
      } else {
        message.error(data.detail || '安装失败')
      }
    } catch (e) {
      message.error('安装失败')
    }
  }

  // 卸载Plugin
  const handleUninstall = async (pluginId: string) => {
    try {
      const res = await fetch('http://localhost:8000/api/v1/plugins/marketplace/uninstall', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ plugin_id: pluginId })
      })
      const data = await res.json()
      if (data.success) {
        message.success(`Plugin卸载成功`)
        fetchPlugins()
        fetchInstalledPlugins()
      } else {
        message.error(data.detail || '卸载失败')
      }
    } catch (e) {
      message.error('卸载失败')
    }
  }

  // 发布Plugin
  const handlePublish = async (values: any) => {
    try {
      const res = await fetch('http://localhost:8000/api/v1/plugins/marketplace/publish', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ...values,
          category: values.category?.value || values.category
        })
      })
      const data = await res.json()
      if (data.success) {
        message.success('Plugin发布成功')
        setPublishModal(false)
        publishForm.resetFields()
        fetchPlugins()
      } else {
        message.error(data.detail || '发布失败')
      }
    } catch (e) {
      message.error('发布失败')
    }
  }

  useEffect(() => {
    fetchPlugins()
    fetchInstalledPlugins()
  }, [])

  const getCategoryInfo = (category: string) => {
    return categoryMap[category] || { color: 'default', icon: '📦' }
  }

  const pluginColumns = [
    { 
      title: 'Plugin', 
      key: 'info',
      render: (_: any, record: PluginInfo) => (
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <Avatar 
            style={{ backgroundColor: getCategoryInfo(record.category).color }}
            icon={getCategoryInfo(record.category).icon}
          />
          <div>
            <div style={{ fontWeight: 500 }}>{record.display_name}</div>
            <div style={{ fontSize: 12, color: '#999' }}>by {record.author}</div>
          </div>
        </div>
      )
    },
    { title: '描述', dataIndex: 'description', key: 'description', ellipsis: true, width: 250 },
    { 
      title: '分类', 
      dataIndex: 'category', 
      key: 'category',
      render: (cat: string) => (
        <Tag color={getCategoryInfo(cat).color}>
          {getCategoryInfo(cat).icon} {cat}
        </Tag>
      )
    },
    { 
      title: '标签', 
      dataIndex: 'tags', 
      key: 'tags',
      render: (tags: string[]) => (
        <Space wrap>
          {tags?.slice(0, 3).map(t => <Tag key={t}>{t}</Tag>)}
        </Space>
      )
    },
    { 
      title: '评分', 
      dataIndex: 'rating', 
      key: 'rating',
      render: (r: number) => (
        <Space>
          <Rate disabled value={r} style={{ fontSize: 14 }} />
          <span>({r.toFixed(1)})</span>
        </Space>
      )
    },
    { title: '下载', dataIndex: 'downloads', key: 'downloads', render: (d: number) => d.toLocaleString() },
    { 
      title: '状态', 
      key: 'installed',
      render: (_: any, record: PluginInfo) => (
        record.installed ? <Tag color="green">已安装</Tag> : <Tag>未安装</Tag>
      )
    },
    {
      title: '操作',
      key: 'action',
      render: (_: any, record: PluginInfo) => (
        <Space>
          <Button type="link" onClick={() => fetchPluginDetail(record.id)}>详情</Button>
          {record.installed ? (
            <Button type="link" danger icon={<DeleteOutlined />} onClick={() => handleUninstall(record.id)}>
              卸载
            </Button>
          ) : (
            <Button type="primary" icon={<DownloadOutlined />} onClick={() => handleInstall(record.id)}>
              安装
            </Button>
          )}
        </Space>
      )
    }
  ]

  const installedColumns = [
    { title: '名称', dataIndex: 'display_name', key: 'name' },
    { title: '版本', dataIndex: 'version', key: 'version' },
    { title: '作者', dataIndex: 'author', key: 'author' },
    {
      title: '操作',
      key: 'action',
      render: (_: any, record: PluginInfo) => (
        <Button type="link" danger icon={<DeleteOutlined />} onClick={() => handleUninstall(record.id)}>
          卸载
        </Button>
      )
    }
  ]

  return (
    <div>
      <h2>🧩 Plugin市场</h2>
      
      <Tabs defaultActiveKey="browse">
        <TabPane tab={<span><AppstoreOutlined /> 浏览插件</span>} key="browse">
          {/* 搜索栏 */}
          <Card style={{ marginBottom: 16 }}>
            <Form layout="inline" onFinish={handleSearch}>
              <Form.Item name="keyword">
                <Input placeholder="搜索Plugin..." prefix={<SearchOutlined />} style={{ width: 300 }} />
              </Form.Item>
              <Form.Item name="category">
                <Select placeholder="分类" style={{ width: 150 }} allowClear>
                  {Object.entries(categoryMap).map(([key, val]) => (
                    <Option key={key} value={key}>{val.icon} {key}</Option>
                  ))}
                </Select>
              </Form.Item>
              <Form.Item name="tags">
                <Input placeholder="标签" style={{ width: 150 }} />
              </Form.Item>
              <Form.Item>
                <Button type="primary" htmlType="submit" loading={searchLoading}>
                  搜索
                </Button>
              </Form.Item>
            </Form>
          </Card>

          {/* Plugin列表 */}
          <Card extra={
            <Button type="primary" icon={<PlusOutlined />} onClick={() => setPublishModal(true)}>
              发布Plugin
            </Button>
          }>
            <Table 
              dataSource={plugins} 
              columns={pluginColumns} 
              rowKey="id"
              loading={loading}
              pagination={{ pageSize: 10 }}
            />
          </Card>
        </TabPane>

        <TabPane tab={<span><DownloadOutlined /> 已安装 ({installedPlugins.length})</span>} key="installed">
          <Card>
            {installedPlugins.length > 0 ? (
              <Table 
                dataSource={installedPlugins} 
                columns={installedColumns} 
                rowKey="id"
              />
            ) : (
              <Empty description="暂无已安装的Plugin" />
            )}
          </Card>
        </TabPane>

        <TabPane tab={<span><TagsOutlined /> 分类浏览</span>} key="categories">
          <Row gutter={16}>
            {Object.entries(categoryMap).map(([key, val]) => {
              const count = plugins.filter(p => p.category === key).length
              return (
                <Col span={8} key={key} style={{ marginBottom: 16 }}>
                  <Card hoverable onClick={() => {
                    setPlugins(plugins.filter(p => p.category === key))
                  }}>
                    <Card.Meta
                      avatar={<Avatar style={{ backgroundColor: val.color }}>{val.icon}</Avatar>}
                      title={key.toUpperCase()}
                      description={`${count} 个Plugin`}
                    />
                  </Card>
                </Col>
              )
            })}
          </Row>
        </TabPane>
      </Tabs>

      {/* Plugin详情弹窗 */}
      <Modal
        title={selectedPlugin?.display_name}
        open={detailModal}
        onCancel={() => setDetailModal(false)}
        width={600}
        footer={[
          <Button key="close" onClick={() => setDetailModal(false)}>关闭</Button>,
          selectedPlugin && !selectedPlugin.installed && (
            <Button key="install" type="primary" onClick={() => {
              handleInstall(selectedPlugin.id)
              setDetailModal(false)
            }}>
              安装
            </Button>
          )
        ]}
      >
        {selectedPlugin && (
          <div>
            <p><strong>作者:</strong> {selectedPlugin.author}</p>
            <p><strong>版本:</strong> {selectedPlugin.version}</p>
            <p><strong>描述:</strong> {selectedPlugin.description}</p>
            <p>
              <strong>分类:</strong> <Tag color={getCategoryInfo(selectedPlugin.category).color}>
                {getCategoryInfo(selectedPlugin.category).icon} {selectedPlugin.category}
              </Tag>
            </p>
            <p>
              <strong>标签:</strong> <Space wrap>
                {selectedPlugin.tags?.map(t => <Tag key={t}>{t}</Tag>)}
              </Space>
            </p>
            <p><strong>下载量:</strong> {selectedPlugin.downloads.toLocaleString()}</p>
            <p><strong>评分:</strong> <Rate disabled value={selectedPlugin.rating} /> ({selectedPlugin.reviews_count}条评价)</p>
            {selectedPlugin.permissions?.length > 0 && (
              <div>
                <strong>权限:</strong>
                <div><Space wrap>
                  {selectedPlugin.permissions.map(p => <Tag key={p}>{p}</Tag>)}
                </Space></div>
              </div>
            )}
          </div>
        )}
      </Modal>

      {/* 发布Plugin弹窗 */}
      <Modal
        title="发布Plugin"
        open={publishModal}
        onCancel={() => setPublishModal(false)}
        width={700}
        footer={null}
      >
        <Form form={publishForm} layout="vertical" onFinish={handlePublish}>
          <Row gutter={16}>
            <Col span={12}>
              <Form.Item name="name" label="名称" rules={[{ required: true }]}>
                <Input placeholder="英文名称，如: my-plugin" />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item name="display_name" label="显示名称" rules={[{ required: true }]}>
                <Input placeholder="中文名称，如: 我的插件" />
              </Form.Item>
            </Col>
          </Row>
          <Form.Item name="description" label="描述" rules={[{ required: true }]}>
            <TextArea rows={3} placeholder="简洁描述Plugin功能" />
          </Form.Item>
          <Row gutter={16}>
            <Col span={8}>
              <Form.Item name="version" label="版本号" rules={[{ required: true }]}>
                <Input placeholder="1.0.0" />
              </Form.Item>
            </Col>
            <Col span={8}>
              <Form.Item name="category" label="分类" rules={[{ required: true }]}>
                <Select placeholder="选择分类">
                  {Object.entries(categoryMap).map(([key, val]) => (
                    <Option key={key} value={key}>{val.icon} {key}</Option>
                  ))}
                </Select>
              </Form.Item>
            </Col>
            <Col span={8}>
              <Form.Item name="author" label="作者" rules={[{ required: true }]}>
                <Input placeholder="你的名字" />
              </Form.Item>
            </Col>
          </Row>
          <Form.Item name="tags" label="标签">
            <Select mode="tags" placeholder="添加标签">
              <Option value="tool">tool</Option>
              <Option value="agent">agent</Option>
              <Option value="integration">integration</Option>
              <Option value="ui">ui</Option>
            </Select>
          </Form.Item>
          <Form.Item name="content" label="Plugin代码" rules={[{ required: true }]}>
            <TextArea rows={8} placeholder="输入Plugin代码" />
          </Form.Item>
          <Button type="primary" htmlType="submit" block>发布</Button>
        </Form>
      </Modal>
    </div>
  )
}
