/**
 * AI Platform - 模型管理页面
 */

import { Card, Table, Tag, Button, Space, Modal, Form, Input, Select, Progress, Descriptions, message, Popconfirm } from 'antd'
import { PlusOutlined, DeleteOutlined, EyeOutlined, DownloadOutlined, UploadOutlined } from '@ant-design/icons'
import { useLang } from '../locales'
import { useState, useEffect } from 'react'

interface Model {
  id: number
  name: string
  description?: string
  project_id: number
  base_model: string
  model_type?: string
  framework?: string
  version?: string
  stage?: string
  parameter_size?: string
  quantization?: string
  size?: number
  storage_path?: string
  metrics?: {
    chinese_bleu?: number
    response_quality?: number
  }
  created_at?: string
}

export default function Models() {
  const { t } = useLang()
  const [loading, setLoading] = useState(false)
  const [models, setModels] = useState<Model[]>([])
  const [selectedModel, setSelectedModel] = useState<Model | null>(null)
  const [isModalOpen, setIsModalOpen] = useState(false)
  const [form] = Form.useForm()

  const fetchModels = async () => {
    try {
      setLoading(true)
      const response = await fetch('/api/v1/models', {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('access_token')}`
        }
      })
      const data = await response.json()
      if (data.models) {
        setModels(data.models.map((m: any) => ({
          ...m,
          key: m.id
        })))
      }
    } catch (error: any) {
      console.error('获取模型列表失败:', error)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchModels()
  }, [])

  const formatSize = (bytes: number) => {
    if (!bytes) return 'N/A'
    for (let unit of ['B', 'KB', 'MB', 'GB', 'TB']) {
      if (bytes < 1024) return `${bytes.toFixed(2)} ${unit}`
      bytes /= 1024
    }
    return `${bytes.toFixed(2)} TB`
  }

  const getStageTag = (stage: string) => {
    const colors: Record<string, string> = {
      development: 'processing',
      testing: 'warning',
      production: 'success',
      archived: 'default'
    }
    const labels: Record<string, string> = {
      development: '开发中',
      testing: '测试中',
      production: '生产',
      archived: '归档'
    }
    return <Tag color={colors[stage] || 'default'}>{labels[stage] || stage}</Tag>
  }

  const columns = [
    { title: 'ID', dataIndex: 'id', key: 'id', width: 60 },
    { title: '模型名称', dataIndex: 'name', key: 'name' },
    { 
      title: '基座模型', 
      dataIndex: 'base_model', 
      key: 'base_model',
      render: (text: string) => <code>{text}</code>
    },
    { 
      title: '框架', 
      dataIndex: 'framework', 
      key: 'framework',
      render: (f: string) => <Tag>{f}</Tag>
    },
    { 
      title: '参数量', 
      dataIndex: 'parameter_size', 
      key: 'parameter_size',
      render: (s: string) => <Tag color="blue">{s}</Tag>
    },
    { 
      title: '大小', 
      dataIndex: 'size', 
      key: 'size',
      render: (s: number) => formatSize(s)
    },
    {
      title: '阶段',
      dataIndex: 'stage',
      key: 'stage',
      render: (s: string) => getStageTag(s)
    },
    {
      title: '创建时间',
      dataIndex: 'created_at',
      key: 'created_at',
      render: (time: string) => time ? new Date(time).toLocaleString() : '-'
    },
    {
      title: '操作',
      key: 'action',
      render: (_: any, record: Model) => (
        <Space>
          <Button 
            type="link" 
            size="small" 
            icon={<EyeOutlined />}
            onClick={() => {
              setSelectedModel(record)
              setIsModalOpen(true)
            }}
          >
            详情
          </Button>
          <Button 
            type="link" 
            size="small" 
            icon={<DownloadOutlined />}
          >
            下载
          </Button>
          <Popconfirm
            title="确认删除"
            description="确定要删除这个模型吗？"
            okText="确定"
            cancelText="取消"
          >
            <Button type="link" size="small" danger icon={<DeleteOutlined />}>
              删除
            </Button>
          </Popconfirm>
        </Space>
      )
    },
  ]

  return (
    <div>
      <Card
        title={
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span>🤖 模型管理</span>
            <Space>
              <Button icon={<UploadOutlined />}>导入模型</Button>
              <Button type="primary" icon={<PlusOutlined />}>新建模型</Button>
            </Space>
          </div>
        }
      >
        <Table
          columns={columns}
          dataSource={models}
          loading={loading}
          pagination={{ pageSize: 10 }}
        />
      </Card>

      {/* 模型详情弹窗 */}
      <Modal
        title={`模型详情: ${selectedModel?.name}`}
        open={isModalOpen}
        onCancel={() => {
          setIsModalOpen(false)
          setSelectedModel(null)
        }}
        footer={[
          <Button key="close" onClick={() => setIsModalOpen(false)}>关闭</Button>,
          <Button key="download" icon={<DownloadOutlined />}>下载模型</Button>
        ]}
        width={700}
      >
        {selectedModel && (
          <Descriptions bordered column={2}>
            <Descriptions.Item label="ID">{selectedModel.id}</Descriptions.Item>
            <Descriptions.Item label="名称">{selectedModel.name}</Descriptions.Item>
            <Descriptions.Item label="基座模型" span={2}>
              <code>{selectedModel.base_model}</code>
            </Descriptions.Item>
            <Descriptions.Item label="框架">{selectedModel.framework}</Descriptions.Item>
            <Descriptions.Item label="参数量">{selectedModel.parameter_size}</Descriptions.Item>
            <Descriptions.Item label="量化">{selectedModel.quantization}</Descriptions.Item>
            <Descriptions.Item label="大小">{formatSize(selectedModel.size || 0)}</Descriptions.Item>
            <Descriptions.Item label="阶段">{getStageTag(selectedModel.stage || 'development')}</Descriptions.Item>
            <Descriptions.Item label="存储路径" span={2}>
              <code>{selectedModel.storage_path}</code>
            </Descriptions.Item>
            <Descriptions.Item label="创建时间">
              {selectedModel.created_at ? new Date(selectedModel.created_at).toLocaleString() : '-'}
            </Descriptions.Item>
            <Descriptions.Item label="评估指标" span={2}>
              {selectedModel.metrics ? (
                <Space direction="vertical">
                  <div>中文BLEU: {selectedModel.metrics.chinese_bleu || 'N/A'}</div>
                  <div>回复质量: {selectedModel.metrics.response_quality || 'N/A'}</div>
                </Space>
              ) : '暂无评估数据'}
            </Descriptions.Item>
          </Descriptions>
        )}
      </Modal>
    </div>
  )
}
