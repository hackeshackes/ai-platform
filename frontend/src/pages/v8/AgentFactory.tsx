// Agent工厂页面 - v8
import React, { useState, useEffect } from 'react'
import { Card, Table, Button, Modal, Form, Input, Select, Tag, message, Space, Tabs, List, Badge } from 'antd'
import { PlusOutlined, DeleteOutlined, EditOutlined, RocketOutlined, PlayCircleOutlined } from '@ant-design/icons'

const { Option } = Select
const { TabPane } = Tabs

interface AgentTemplate {
  id: string
  name: string
  description: string
  version: string
  capabilities: string[]
  created_at: string
}

interface CreatedAgent {
  id: string
  name: string
  template_id: string
  status: string
  created_at: string
}

export function AgentFactoryPage() {
  const [templates, setTemplates] = useState<AgentTemplate[]>([])
  const [agents, setAgents] = useState<CreatedAgent[]>([])
  const [loading, setLoading] = useState(false)
  const [createModal, setCreateModal] = useState(false)
  const [batchModal, setBatchModal] = useState(false)
  const [form] = Form.useForm()
  const [batchForm] = Form.useForm()

  // 获取模板列表
  const fetchTemplates = async () => {
    setLoading(true)
    try {
      const res = await fetch('http://localhost:8000/api/v1/agents/factory/templates')
      const data = await res.json()
      if (data.success) {
        setTemplates(data.templates)
      }
    } catch (e) {
      message.error('获取模板失败')
    }
    setLoading(false)
  }

  // 获取Agent列表
  const fetchAgents = async () => {
    try {
      const res = await fetch('http://localhost:8000/api/v1/agents/orchestration/sessions')
      const data = await res.json()
      if (data.sessions) {
        setAgents(data.sessions.map((s: any) => ({
          id: s.id,
          name: s.name,
          template_id: 'custom',
          status: s.status,
          created_at: s.created_at
        })))
      }
    } catch (e) {
      console.error(e)
    }
  }

  useEffect(() => {
    fetchTemplates()
    fetchAgents()
  }, [])

  // 创建Agent
  const handleCreate = async (values: any) => {
    try {
      const res = await fetch('http://localhost:8000/api/v1/agents/factory/create', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(values)
      })
      const data = await res.json()
      if (data.success) {
        message.success('Agent创建成功')
        setCreateModal(false)
        form.resetFields()
        fetchAgents()
      } else {
        message.error(data.detail || '创建失败')
      }
    } catch (e) {
      message.error('创建失败')
    }
  }

  // 批量创建
  const handleBatchCreate = async (values: any) => {
    try {
      const res = await fetch('http://localhost:8000/api/v1/agents/factory/batch', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(values)
      })
      const data = await res.json()
      if (data.success) {
        message.success(`成功创建 ${data.agents?.length || 0} 个Agent`)
        setBatchModal(false)
        batchForm.resetFields()
        fetchAgents()
      } else {
        message.error(data.detail || '批量创建失败')
      }
    } catch (e) {
      message.error('批量创建失败')
    }
  }

  // 部署Agent
  const handleDeploy = async (agentId: string) => {
    try {
      const res = await fetch(`http://localhost:8000/api/v1/agents/factory/deploy`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ agent_id: agentId })
      })
      const data = await res.json()
      if (data.success) {
        message.success('Agent部署成功')
      } else {
        message.error(data.detail || '部署失败')
      }
    } catch (e) {
      message.error('部署失败')
    }
  }

  const templateColumns = [
    { title: '名称', dataIndex: 'name', key: 'name' },
    { title: '描述', dataIndex: 'description', key: 'description', ellipsis: true },
    { title: '版本', dataIndex: 'version', key: 'version' },
    { 
      title: '能力', 
      dataIndex: 'capabilities', 
      key: 'capabilities',
      render: (caps: string[]) => (
        <Space>
          {caps.slice(0, 3).map(c => <Tag key={c}>{c}</Tag>)}
          {caps.length > 3 && <Tag>+{caps.length - 3}</Tag>}
        </Space>
      )
    },
    {
      title: '操作',
      key: 'action',
      render: (_: any, record: AgentTemplate) => (
        <Button type="primary" icon={<PlusOutlined />} onClick={() => {
          form.setFieldsValue({ template_id: record.id, name: `${record.name}-${Date.now()}` })
          setCreateModal(true)
        }}>
          创建
        </Button>
      )
    }
  ]

  const agentColumns = [
    { title: '名称', dataIndex: 'name', key: 'name' },
    { title: '模板', dataIndex: 'template_id', key: 'template_id' },
    { 
      title: '状态', 
      dataIndex: 'status', 
      key: 'status',
      render: (status: string) => (
        <Badge status={status === 'active' ? 'success' : 'default'} text={status} />
      )
    },
    { title: '创建时间', dataIndex: 'created_at', key: 'created_at', render: (t: string) => new Date(t).toLocaleString() },
    {
      title: '操作',
      key: 'action',
      render: (_: any, record: CreatedAgent) => (
        <Space>
          <Button icon={<RocketOutlined />} onClick={() => handleDeploy(record.id)}>部署</Button>
          <Button icon={<PlayCircleOutlined />}>启动</Button>
        </Space>
      )
    }
  ]

  return (
    <div>
      <h2>🤖 Agent工厂</h2>
      <Tabs defaultActiveKey="templates">
        <TabPane tab="模板市场" key="templates">
          <Card>
            <Table 
              dataSource={templates} 
              columns={templateColumns} 
              rowKey="id"
              loading={loading}
            />
          </Card>
        </TabPane>
        <TabPane tab="我的Agent" key="agents">
          <Card extra={
            <Space>
              <Button type="primary" icon={<PlusOutlined />} onClick={() => setBatchModal(true)}>
                批量创建
              </Button>
              <Button icon={<PlusOutlined />} onClick={() => setCreateModal(true)}>
                创建Agent
              </Button>
            </Space>
          }>
            <Table 
              dataSource={agents} 
              columns={agentColumns} 
              rowKey="id"
            />
          </Card>
        </TabPane>
      </Tabs>

      {/* 创建Agent弹窗 */}
      <Modal
        title="创建Agent"
        open={createModal}
        onCancel={() => setCreateModal(false)}
        footer={null}
      >
        <Form form={form} layout="vertical" onFinish={handleCreate}>
          <Form.Item name="template_id" label="选择模板" rules={[{ required: true }]}>
            <Select placeholder="选择模板">
              {templates.map(t => (
                <Option key={t.id} value={t.id}>{t.name} - {t.description}</Option>
              ))}
            </Select>
          </Form.Item>
          <Form.Item name="name" label="Agent名称" rules={[{ required: true }]}>
            <Input placeholder="输入Agent名称" />
          </Form.Item>
          <Form.Item>
            <Button type="primary" htmlType="submit" block>创建</Button>
          </Form.Item>
        </Form>
      </Modal>

      {/* 批量创建弹窗 */}
      <Modal
        title="批量创建Agent"
        open={batchModal}
        onCancel={() => setBatchModal(false)}
        footer={null}
      >
        <Form form={batchForm} layout="vertical" onFinish={handleBatchCreate}>
          <Form.Item name="template_id" label="选择模板" rules={[{ required: true }]}>
            <Select placeholder="选择模板">
              {templates.map(t => (
                <Option key={t.id} value={t.id}>{t.name}</Option>
              ))}
            </Select>
          </Form.Item>
          <Form.Item name="base_name" label="基础名称" rules={[{ required: true }]}>
            <Input placeholder="如: dev-agent，将生成 dev-agent-1, dev-agent-2..." />
          </Form.Item>
          <Form.Item name="count" label="数量" rules={[{ required: true }]}>
            <Select placeholder="选择数量">
              {[3, 5, 10, 20, 50].map(n => (
                <Option key={n} value={n}>{n}个</Option>
              ))}
            </Select>
          </Form.Item>
          <Form.Item>
            <Button type="primary" htmlType="submit" block>批量创建</Button>
          </Form.Item>
        </Form>
      </Modal>
    </div>
  )
}
