/**
 * AI Platform - 任务管理页面
 */

import { Card, Table, Tag, Button, Space, Modal, Form, Input, Select, Progress, Timeline, Tabs } from 'antd'
import { PlayCircleOutlined, PauseCircleOutlined, DeleteOutlined, PlusOutlined, EyeOutlined, ConsoleSqlOutlined } from '@ant-design/icons'
import { useLang } from '../locales'
import { useState, useEffect } from 'react'
import { taskAPI } from '../api/client'

const { Option } = Select
const { TabPane } = Tabs

export default function Tasks() {
  const { t } = useLang()
  const [loading, setLoading] = useState(false)
  const [tasks, setTasks] = useState<any[]>([])
  const [selectedTask, setSelectedTask] = useState<any>(null)
  const [isModalOpen, setIsModalOpen] = useState(false)
  const [isDetailOpen, setIsDetailOpen] = useState(false)
  const [form] = Form.useForm()
  const [filterStatus, setFilterStatus] = useState<string | undefined>()

  const fetchTasks = async () => {
    try {
      setLoading(true)
      const response = await taskAPI.list({ status: filterStatus })
      if (response.tasks) {
        setTasks(response.tasks.map((t: any) => ({
          key: t.id,
          ...t
        })))
      }
    } catch (error: any) {
      console.error('获取任务列表失败:', error)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchTasks()
  }, [filterStatus])

  const handleCreate = async () => {
    try {
      const values = await form.validateFields()
      setLoading(true)
      await taskAPI.create(values)
      message.success('任务创建成功！')
      setIsModalOpen(false)
      form.resetFields()
      fetchTasks()
    } catch (error: any) {
      message.error(error.response?.data?.detail || '创建任务失败')
    } finally {
      setLoading(false)
    }
  }

  const handleDelete = async (id: string) => {
    try {
      setLoading(true)
      await taskAPI.delete(id)
      message.success('任务已删除')
      fetchTasks()
    } catch (error: any) {
      message.error(error.response?.data?.detail || '删除失败')
    } finally {
      setLoading(false)
    }
  }

  const getStatusTag = (status: string) => {
    const colors: Record<string, string> = {
      pending: 'orange',
      running: 'processing',
      completed: 'success',
      failed: 'error',
      stopped: 'default'
    }
    const labels: Record<string, string> = {
      pending: '等待中',
      running: '运行中',
      completed: '已完成',
      failed: '失败',
      stopped: '已停止'
    }
    return <Tag color={colors[status] || 'default'}>{labels[status] || status}</Tag>
  }

  const columns = [
    { title: 'ID', dataIndex: 'id', key: 'id', width: 80 },
    { title: '任务名称', dataIndex: 'name', key: 'name' },
    { 
      title: '类型', 
      dataIndex: 'type', 
      key: 'type',
      render: (type: string) => (
        <Tag>{type === 'training' ? '训练' : type === 'inference' ? '推理' : type}</Tag>
      )
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => getStatusTag(status)
    },
    {
      title: '进度',
      dataIndex: 'progress',
      key: 'progress',
      render: (progress: number) => (
        <Progress percent={Math.round(progress)} size="small" style={{ width: 100 }} />
      )
    },
    {
      title: '创建时间',
      dataIndex: 'created_at',
      key: 'created_at',
      render: (time: string) => new Date(time).toLocaleString()
    },
    {
      title: '操作',
      key: 'action',
      render: (_: any, record: any) => (
        <Space>
          <Button 
            type="link" 
            size="small" 
            icon={<EyeOutlined />}
            onClick={() => {
              setSelectedTask(record)
              setIsDetailOpen(true)
            }}
          >
            详情
          </Button>
          <Button 
            type="link" 
            size="small" 
            danger 
            icon={<DeleteOutlined />}
            onClick={() => handleDelete(record.id)}
          >
            删除
          </Button>
        </Space>
      )
    },
  ]

  return (
    <div>
      <Card
        title={
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span>📋 任务管理</span>
            <Space>
              <Select
                placeholder="状态筛选"
                allowClear
                style={{ width: 120 }}
                value={filterStatus}
                onChange={setFilterStatus}
                options={[
                  { value: 'pending', label: '等待中' },
                  { value: 'running', label: '运行中' },
                  { value: 'completed', label: '已完成' },
                  { value: 'failed', label: '失败' },
                ]}
              />
              <Button 
                type="primary" 
                icon={<PlusOutlined />}
                onClick={() => setIsModalOpen(true)}
              >
                创建任务
              </Button>
            </Space>
          </div>
        }
      >
        <Table
          columns={columns}
          dataSource={tasks}
          loading={loading}
          pagination={{ pageSize: 10 }}
        />
      </Card>

      {/* 创建任务弹窗 */}
      <Modal
        title="创建新任务"
        open={isModalOpen}
        onOk={handleCreate}
        onCancel={() => {
          setIsModalOpen(false)
          form.resetFields()
        }}
        confirmLoading={loading}
      >
        <Form form={form} layout="vertical">
          <Form.Item
            name="name"
            label="任务名称"
            rules={[{ required: true, message: '请输入任务名称' }]}
          >
            <Input placeholder="输入任务名称" />
          </Form.Item>
          <Form.Item
            name="project_id"
            label="项目ID"
            rules={[{ required: true, message: '请选择项目' }]}
          >
            <Select placeholder="选择项目">
              <Option value={1}>LLM Fine-tuning Demo</Option>
              <Option value={2}>测试项目</Option>
            </Select>
          </Form.Item>
          <Form.Item
            name="type"
            label="任务类型"
            rules={[{ required: true, message: '请选择任务类型' }]}
          >
            <Select placeholder="选择任务类型">
              <Option value="training">训练</Option>
              <Option value="inference">推理</Option>
              <Option value="evaluation">评估</Option>
            </Select>
          </Form.Item>
          <Form.Item
            name="config"
            label="配置 (JSON)"
          >
            <Input.TextArea 
              rows={4} 
              placeholder='{"model": "llama-2-7b", "epochs": 3}' 
            />
          </Form.Item>
        </Form>
      </Modal>

      {/* 任务详情弹窗 */}
      <Modal
        title={`任务详情: ${selectedTask?.name}`}
        open={isDetailOpen}
        onCancel={() => setIsDetailOpen(false)}
        footer={[
          <Button key="close" onClick={() => setIsDetailOpen(false)}>关闭</Button>
        ]}
        width={700}
      >
        {selectedTask && (
          <Tabs defaultActiveKey="1">
            <TabPane tab="基本信息" key="1">
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
                <div><strong>ID:</strong> {selectedTask.id}</div>
                <div><strong>类型:</strong> {selectedTask.type}</div>
                <div><strong>状态:</strong> {getStatusTag(selectedTask.status)}</div>
                <div><strong>进度:</strong> <Progress percent={selectedTask.progress} /></div>
                <div><strong>创建时间:</strong> {new Date(selectedTask.created_at).toLocaleString()}</div>
                <div><strong>开始时间:</strong> {selectedTask.started_at ? new Date(selectedTask.started_at).toLocaleString() : '-'}</div>
              </div>
            </TabPane>
            <TabPane tab="配置信息" key="2">
              <pre style={{ background: '#f5f5f5', padding: 16, borderRadius: 4, overflow: 'auto' }}>
                {JSON.stringify(selectedTask.config || {}, null, 2)}
              </pre>
            </TabPane>
            <TabPane tab="执行日志" key="3">
              <div style={{ 
                background: '#1e1e1e', 
                color: '#d4d4d4', 
                padding: 16, 
                borderRadius: 4,
                fontFamily: 'monospace',
                fontSize: 12,
                maxHeight: 300,
                overflow: 'auto'
              }}>
                {selectedTask.logs || '暂无日志'}
              </div>
            </TabPane>
          </Tabs>
        )}
      </Modal>
    </div>
  )
}
