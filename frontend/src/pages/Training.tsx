/**
 * AI Platform - 训练任务提交页面
 */

import { Card, Form, Input, Select, Button, Steps, Result, Table, Tag, Space, Progress, Timeline, message } from 'antd'
import { RocketOutlined, CloudServerOutlined, DataOutlined, ExperimentOutlined, CheckCircleOutlined, PlayCircleOutlined } from '@ant-design/icons'
import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'

const { Option } = Select
const { Step } = Steps

interface Model {
  id: string
  name: string
  provider: string
  size: string
  type: string
}

interface Dataset {
  id: number
  name: string
  size: string
  rows: string
}

interface Template {
  id: string
  name: string
  description: string
  min_gpu_memory: string
}

interface Job {
  job_id: string
  experiment_name: string
  model: string
  status: string
  progress: number
  eta: string
}

export default function Training() {
  const navigate = useNavigate()
  const [currentStep, setCurrentStep] = useState(0)
  const [loading, setLoading] = useState(false)
  const [form] = Form.useForm()
  
  const [models, setModels] = useState<Model[]>([])
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [templates, setTemplates] = useState<Template[]>([])
  const [recentJobs, setRecentJobs] = useState<Job[]>([])
  const [submitted, setSubmitted] = useState<any>(null)

  // 获取选项数据
  useEffect(() => {
    fetchOptions()
    fetchJobs()
  }, [])

  const fetchOptions = async () => {
    try {
      const token = localStorage.getItem('access_token')
      
      const [modelsRes, datasetsRes, templatesRes] = await Promise.all([
        fetch('/api/v1/training/models', { headers: { 'Authorization': `Bearer ${token}` } }),
        fetch('/api/v1/training/datasets', { headers: { 'Authorization': `Bearer ${token}` } }),
        fetch('/api/v1/training/templates', { headers: { 'Authorization': `Bearer ${token}` } })
      ])
      
      const modelsData = await modelsRes.json()
      const datasetsData = await datasetsRes.json()
      const templatesData = await templatesRes.json()
      
      setModels(modelsData.models || [])
      setDatasets(datasetsData.datasets || [])
      setTemplates(templatesData.templates || [])
    } catch (error) {
      console.error('获取选项数据失败:', error)
    }
  }

  const fetchJobs = async () => {
    try {
      const token = localStorage.getItem('access_token')
      const res = await fetch('/api/v1/training/jobs', { headers: { 'Authorization': `Bearer ${token}` } })
      const data = await res.json()
      setRecentJobs(data.jobs || [])
    } catch (error) {
      console.error('获取任务列表失败:', error)
    }
  }

  const handleSubmit = async () => {
    try {
      const values = await form.validateFields()
      setLoading(true)
      
      const token = localStorage.getItem('access_token')
      const response = await fetch('/api/v1/training/submit', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({
          ...values,
          experiment_name: values.name || `${values.model_id}-${Date.now()}`
        })
      })
      
      if (response.ok) {
        const data = await response.json()
        setSubmitted(data.job)
        setCurrentStep(3)
        message.success('训练任务提交成功！')
        fetchJobs()
      } else {
        throw new Error('提交失败')
      }
    } catch (error: any) {
      message.error(error.message || '提交失败')
    } finally {
      setLoading(false)
    }
  }

  const steps = [
    { title: '选择模型', icon: <CloudServerOutlined />, content: 0 },
    { title: '选择数据', icon: <DataOutlined />, content: 1 },
    { title: '配置训练', icon: <ExperimentOutlined />, content: 2 },
    { title: '确认提交', icon: <CheckCircleOutlined />, content: 3 },
  ]

  const getStatusTag = (status: string) => {
    const colors: Record<string, string> = {
      queued: 'orange',
      running: 'processing',
      completed: 'success',
      failed: 'error'
    }
    const labels: Record<string, string> = {
      queued: '排队中',
      running: '运行中',
      completed: '已完成',
      failed: '失败'
    }
    return <Tag color={colors[status]}>{labels[status]}</Tag>
  }

  return (
    <div>
      <Card
        title={<><RocketOutlined /> 提交训练任务</>}
        extra={
          <Button onClick={() => navigate('/tasks')}>查看所有任务</Button>
        }
      >
        {/* 步骤条 */}
        <Steps current={currentStep} style={{ marginBottom: 32 }}>
          {steps.map(s => (
            <Step key={s.title} title={s.title} icon={s.icon} />
          ))}
        </Steps>

        {/* 步骤1: 选择模型 */}
        {currentStep === 0 && (
          <Form form={form} layout="vertical">
            <Form.Item
              name="experiment_name"
              label="实验名称"
              rules={[{ required: true, message: '请输入实验名称' }]}
            >
              <Input placeholder="例如: Llama-2-7B-LoRA-Test" />
            </Form.Item>

            <Form.Item
              name="model_id"
              label="选择模型"
              rules={[{ required: true, message: '请选择模型' }]}
            >
              <Select placeholder="选择预训练模型" size="large">
                {models.map(m => (
                  <Option key={m.id} value={m.id}>
                    <Space>
                      <span>{m.name}</span>
                      <Tag>{m.size}</Tag>
                      <Tag color="blue">{m.provider}</Tag>
                    </Space>
                  </Option>
                ))}
              </Select>
            </Form.Item>

            <Form.Item
              name="project_id"
              label="关联项目"
              rules={[{ required: true, message: '请选择项目' }]}
            >
              <Select placeholder="选择关联项目">
                <Option value={1}>LLM Fine-tuning Demo</Option>
                <Option value={2}>测试项目</Option>
              </Select>
            </Form.Item>

            <div style={{ marginTop: 32 }}>
              <Button type="primary" onClick={() => setCurrentStep(1)}>
                下一步: 选择数据
              </Button>
            </div>
          </Form>
        )}

        {/* 步骤2: 选择数据 */}
        {currentStep === 1 && (
          <Form form={form} layout="vertical">
            <Form.Item
              name="dataset_id"
              label="选择数据集"
              rules={[{ required: true, message: '请选择数据集' }]}
            >
              <Select placeholder="选择训练数据集" size="large">
                {datasets.map(d => (
                  <Option key={d.id} value={d.id}>
                    <Space>
                      <span>{d.name}</span>
                      <Tag>{d.size}</Tag>
                      <Tag color="green">{d.rows} 行</Tag>
                    </Space>
                  </Option>
                ))}
              </Select>
            </Form.Item>

            <Form.Item label="数据预览">
              <Table
                size="small"
                pagination={false}
                dataSource={datasets}
                columns={[
                  { title: '名称', dataIndex: 'name', key: 'name' },
                  { title: '大小', dataIndex: 'size', key: 'size' },
                  { title: '行数', dataIndex: 'rows', key: 'rows' },
                ]}
              />
            </Form.Item>

            <div style={{ marginTop: 32 }}>
              <Button onClick={() => setCurrentStep(0)} style={{ marginRight: 16 }}>
                上一步
              </Button>
              <Button type="primary" onClick={() => setCurrentStep(2)}>
                下一步: 配置训练
              </Button>
            </div>
          </Form>
        )}

        {/* 步骤3: 配置训练 */}
        {currentStep === 2 && (
          <Form form={form} layout="vertical">
            <Form.Item
              name="template_id"
              label="训练模板"
              rules={[{ required: true, message: '请选择训练模板' }]}
            >
              <Select placeholder="选择训练配置模板" size="large">
                {templates.map(t => (
                  <Option key={t.id} value={t.id}>
                    <div>
                      <Space>
                        <span style={{ fontWeight: 'bold' }}>{t.name}</span>
                        <Tag color="purple">需 {t.min_gpu_memory}</Tag>
                      </Space>
                      <div style={{ color: '#888', fontSize: 12, marginTop: 4 }}>
                        {t.description}
                      </div>
                    </div>
                  </Option>
                ))}
              </Select>
            </Form.Item>

            <Form.Item
              name="learning_rate"
              label="学习率"
              initialValue={2e-4}
            >
              <Input type="number" step="0.00001" placeholder="学习率" />
            </Form.Item>

            <Form.Item
              name="epochs"
              label="训练轮数"
              initialValue={3}
            >
              <Select>
                <Option value={1}>1 epoch</Option>
                <Option value={2}>2 epochs</Option>
                <Option value={3}>3 epochs</Option>
                <Option value={5}>5 epochs</Option>
              </Select>
            </Form.Item>

            <Form.Item
              name="batch_size"
              label="批大小"
              initialValue={4}
            >
              <Select>
                <Option value={2}>2</Option>
                <Option value={4}>4</Option>
                <Option value={8}>8</Option>
                <Option value={16}>16</Option>
              </Select>
            </Form.Item>

            <div style={{ marginTop: 32 }}>
              <Button onClick={() => setCurrentStep(1)} style={{ marginRight: 16 }}>
                上一步
              </Button>
              <Button type="primary" onClick={handleSubmit} loading={loading}>
                <PlayCircleOutlined /> 提交训练任务
              </Button>
            </div>
          </Form>
        )}

        {/* 步骤4: 提交成功 */}
        {currentStep === 3 && submitted && (
          <Result
            status="success"
            title="训练任务已提交！"
            subTitle={`任务ID: ${submitted.job_id}`}
            extra={[
              <Button type="primary" key="console" onClick={() => navigate('/tasks')}>
                查看任务状态
              </Button>,
              <Button key="new" onClick={() => {
                setCurrentStep(0)
                setSubmitted(null)
                form.resetFields()
              }}>
                提交新任务
              </Button>,
            ]}
          >
            <Card size="small" style={{ marginTop: 24 }}>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
                <div><strong>实验名称:</strong> {submitted.experiment_name}</div>
                <div><strong>模型:</strong> {submitted.model?.name}</div>
                <div><strong>模板:</strong> {submitted.template?.name}</div>
                <div><strong>状态:</strong> {getStatusTag(submitted.status)}</div>
                <div><strong>预估启动:</strong> {submitted.estimated_start}</div>
                <div><strong>队列位置:</strong> #{submitted.queue_position}</div>
              </div>
            </Card>
          </Result>
        )}
      </Card>

      {/* 最近任务 */}
      <Card title="最近训练任务" style={{ marginTop: 24 }}>
        <Table
          size="small"
          pagination={false}
          dataSource={recentJobs}
          columns={[
            { title: '任务ID', dataIndex: 'job_id', key: 'job_id' },
            { title: '实验名称', dataIndex: 'experiment_name', key: 'name' },
            { title: '模型', dataIndex: 'model', key: 'model' },
            { title: '状态', dataIndex: 'status', key: 'status', render: (s: string) => getStatusTag(s) },
            { 
              title: '进度', 
              dataIndex: 'progress', 
              key: 'progress',
              render: (p: number) => <Progress percent={p} size="small" />
            },
            { title: '预计剩余', dataIndex: 'eta', key: 'eta' },
          ]}
        />
      </Card>
    </div>
  )
}
