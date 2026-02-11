// Agent协作页面 - v8
import React, { useState, useEffect } from 'react'
import { Card, Table, Button, Modal, Form, Input, Select, Tag, message, Space, Tabs, Steps, List, Avatar, Badge } from 'antd'
import { PlusOutlined, TeamOutlined, PlayCircleOutlined, SyncOutlined, CheckCircleOutlined } from '@ant-design/icons'

const { Option } = Select
const { TabPane } = Tabs
const { Step } = Steps

interface CollaborationSession {
  id: string
  name: string
  mode: string
  status: string
  agents: any[]
  created_at: string
  progress: number
}

interface Task {
  id: string
  name: string
  assignee: string
  status: string
  result?: string
}

export function AgentCollaborationPage() {
  const [sessions, setSessions] = useState<CollaborationSession[]>([])
  const [loading, setLoading] = useState(false)
  const [createModal, setCreateModal] = useState(false)
  const [detailModal, setDetailModal] = useState(false)
  const [selectedSession, setSelectedSession] = useState<CollaborationSession | null>(null)
  const [form] = Form.useForm()

  // 获取会话列表
  const fetchSessions = async () => {
    setLoading(true)
    try {
      const res = await fetch('http://localhost:8000/api/v1/agents/collaboration/sessions')
      const data = await res.json()
      if (data.sessions) {
        setSessions(data.sessions.map((s: any) => ({
          id: s.id,
          name: s.name || '未命名',
          mode: s.mode || 'sequential',
          status: s.status || 'pending',
          agents: s.agents || [],
          created_at: s.created_at,
          progress: s.progress || 0
        })))
      }
    } catch (e) {
      message.error('获取会话列表失败')
    }
    setLoading(false)
  }

  useEffect(() => {
    fetchSessions()
  }, [])

  // 创建协作会话
  const handleCreate = async (values: any) => {
    try {
      const res = await fetch('http://localhost:8000/api/v1/agents/collaboration/session', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(values)
      })
      const data = await res.json()
      if (data.id) {
        message.success('协作会话创建成功')
        setCreateModal(false)
        form.resetFields()
        fetchSessions()
      } else {
        message.error(data.detail || '创建失败')
      }
    } catch (e) {
      message.error('创建失败')
    }
  }

  // 执行协作
  const handleExecute = async (sessionId: string) => {
    try {
      const res = await fetch(`http://localhost:8000/api/v1/agents/collaboration/session/${sessionId}/execute`, {
        method: 'POST'
      })
      const data = await res.json()
      if (data.success) {
        message.success('协作执行成功')
        fetchSessions()
      } else {
        message.error(data.detail || '执行失败')
      }
    } catch (e) {
      message.error('执行失败')
    }
  }

  // 查看详情
  const handleViewDetail = async (session: CollaborationSession) => {
    setSelectedSession(session)
    setDetailModal(true)
  }

  const getModeTag = (mode: string) => {
    const colors: Record<string, string> = {
      sequential: 'blue',
      parallel: 'green',
      hierarchical: 'purple',
      consensus: 'orange'
    }
    const labels: Record<string, string> = {
      sequential: '顺序执行',
      parallel: '并行执行',
      hierarchical: '层级协作',
      consensus: '共识决策'
    }
    return <Tag color={colors[mode] || 'default'}>{labels[mode] || mode}</Tag>
  }

  const getStatusTag = (status: string) => {
    const config: Record<string, { color: string, icon: any }> = {
      pending: { color: 'default', icon: null },
      running: { color: 'processing', icon: <SyncOutlined spin /> },
      completed: { color: 'success', icon: <CheckCircleOutlined /> },
      failed: { color: 'error', icon: null }
    }
    const c = config[status] || config.pending
    return <Badge status={c.color as any} text={status} icon={c.icon} />
  }

  const sessionColumns = [
    { title: '会话名称', dataIndex: 'name', key: 'name' },
    { title: '协作模式', dataIndex: 'mode', key: 'mode', render: (m: string) => getModeTag(m) },
    { title: '状态', dataIndex: 'status', key: 'status', render: (s: string) => getStatusTag(s) },
    { 
      title: 'Agent数量', 
      key: 'agents',
      render: (_: any, record: CollaborationSession) => record.agents?.length || 0
    },
    { 
      title: '进度', 
      key: 'progress',
      render: (_: any, record: CollaborationSession) => (
        <Steps size="small" current={Math.floor(record.progress / 25)} status={record.progress === 100 ? 'finish' : 'process'}>
          <Step /><Step /><Step /><Step />
        </Steps>
      )
    },
    { title: '创建时间', dataIndex: 'created_at', key: 'created_at', render: (t: string) => new Date(t).toLocaleString() },
    {
      title: '操作',
      key: 'action',
      render: (_: any, record: CollaborationSession) => (
        <Space>
          <Button type="link" icon={<PlayCircleOutlined />} onClick={() => handleExecute(record.id)}>
            执行
          </Button>
          <Button type="link" onClick={() => handleViewDetail(record)}>
            详情
          </Button>
        </Space>
      )
    }
  ]

  return (
    <div>
      <h2>👥 Agent协作网络</h2>
      <Card extra={
        <Button type="primary" icon={<PlusOutlined />} onClick={() => setCreateModal(true)}>
          创建协作会话
        </Button>
      }>
        <Table 
          dataSource={sessions} 
          columns={sessionColumns} 
          rowKey="id"
          loading={loading}
        />
      </Card>

      {/* 创建会话弹窗 */}
      <Modal
        title="创建协作会话"
        open={createModal}
        onCancel={() => setCreateModal(false)}
        footer={null}
      >
        <Form form={form} layout="vertical" onFinish={handleCreate}>
          <Form.Item name="name" label="会话名称" rules={[{ required: true }]}>
            <Input placeholder="输入会话名称" />
          </Form.Item>
          <Form.Item name="mode" label="协作模式" rules={[{ required: true }]}>
            <Select placeholder="选择协作模式">
              <Option value="sequential">顺序执行 - Agent依次执行任务</Option>
              <Option value="parallel">并行执行 - Agent同时执行任务</Option>
              <Option value="hierarchical">层级协作 - 主Agent协调子Agent</Option>
              <Option value="consensus">共识决策 - 多Agent投票决策</Option>
            </Select>
          </Form.Item>
          <Form.Item name="agents" label="参与Agent">
            <Select mode="tags" placeholder="输入Agent ID或名称">
              <Option value="researcher">researcher</Option>
              <Option value="analyst">analyst</Option>
              <Option value="writer">writer</Option>
              <Option value="coder">coder</Option>
            </Select>
          </Form.Item>
          <Button type="primary" htmlType="submit" block>创建</Button>
        </Form>
      </Modal>

      {/* 会话详情弹窗 */}
      <Modal
        title={`协作会话: ${selectedSession?.name}`}
        open={detailModal}
        onCancel={() => setDetailModal(false)}
        width={700}
        footer={[
          <Button key="execute" type="primary" icon={<PlayCircleOutlined />} onClick={() => selectedSession && handleExecute(selectedSession.id)}>
            执行协作
          </Button>,
          <Button key="close" onClick={() => setDetailModal(false)}>关闭</Button>
        ]}
      >
        {selectedSession && (
          <div>
            <Card title="基本信息" size="small">
              <Space>
                {getModeTag(selectedSession.mode)}
                {getStatusTag(selectedSession.status)}
              </Space>
            </Card>
            <Card title="参与Agent" size="small" style={{ marginTop: 16 }}>
              <List
                dataSource={selectedSession.agents || []}
                renderItem={(agent: any) => (
                  <List.Item>
                    <List.Item.Meta
                      avatar={<Avatar icon={<TeamOutlined />} />}
                      title={agent.name || agent.id}
                      description={agent.role || '参与者'}
                    />
                    <Tag color={agent.status === 'active' ? 'green' : 'default'}>
                      {agent.status || 'pending'}
                    </Tag>
                  </List.Item>
                )}
              />
            </Card>
            <Card title="协作流程" size="small" style={{ marginTop: 16 }}>
              <Steps direction="vertical" current={selectedSession.progress >= 100 ? 3 : Math.floor(selectedSession.progress / 33)}>
                <Step title="任务分解" description="将任务分解为子任务" />
                <Step title="Agent分配" description="分配Agent执行子任务" />
                <Step title="执行协作" description="Agent协作完成任务" />
                <Step title="结果汇总" description="汇总协作结果" />
              </Steps>
            </Card>
          </div>
        )}
      </Modal>
    </div>
  )
}
