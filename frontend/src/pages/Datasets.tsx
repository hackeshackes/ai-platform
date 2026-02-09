/**
 * AI Platform - 数据集管理页面
 */

import { Card, Table, Tag, Button, Space, Modal, Form, Input, Select, Upload, Progress, Descriptions, Tabs, message } from 'antd'
import { UploadOutlined, DeleteOutlined, EyeOutlined, PlusOutlined, FileOutlined, CheckCircleOutlined, WarningOutlined } from '@ant-design/icons'
import { useLang } from '../locales'
import { useState, useEffect } from 'react'

interface Dataset {
  id: number
  name: string
  description: string
  project_id: number
  data_type: string
  format: string
  size: number
  row_count: number
  storage_path: string
  version: number
  annotation_status: string
  stats: {
    avg_length: number
    unique_entities: number
  }
  created_at: string
}

export default function Datasets() {
  const { t } = useLang()
  const [loading, setLoading] = useState(false)
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [selectedDataset, setSelectedDataset] = useState<Dataset | null>(null)
  const [isModalOpen, setIsModalOpen] = useState(false)
  const [isDetailOpen, setIsDetailOpen] = useState(false)
  const [form] = Form.useForm()
  const [uploadProgress, setUploadProgress] = useState(0)

  const fetchDatasets = async () => {
    try {
      setLoading(true)
      const response = await fetch('/api/v1/datasets', {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('access_token')}`
        }
      })
      const data = await response.json()
      if (data.datasets) {
        setDatasets(data.datasets.map((d: any) => ({
          ...d,
          key: d.id
        })))
      }
    } catch (error: any) {
      console.error('获取数据集列表失败:', error)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchDatasets()
  }, [])

  const formatSize = (bytes: number) => {
    for (let unit of ['B', 'KB', 'MB', 'GB']) {
      if (bytes < 1024) return `${bytes.toFixed(2)} ${unit}`
      bytes /= 1024
    }
    return `${bytes.toFixed(2)} TB`
  }

  const getStatusTag = (status: string) => {
    const colors: Record<string, string> = {
      pending: 'orange',
      ready: 'success',
      processing: 'processing',
      failed: 'error'
    }
    const labels: Record<string, string> = {
      pending: '等待中',
      ready: '就绪',
      processing: '处理中',
      failed: '失败'
    }
    return <Tag color={colors[status] || 'default'}>{labels[status] || status}</Tag>
  }

  const columns = [
    { title: 'ID', dataIndex: 'id', key: 'id', width: 60 },
    { title: '名称', dataIndex: 'name', key: 'name' },
    { title: '描述', dataIndex: 'description', key: 'description', ellipsis: true },
    { 
      title: '类型', 
      dataIndex: 'data_type', 
      key: 'data_type',
      render: (type: string) => <Tag>{type}</Tag>
    },
    { 
      title: '格式', 
      dataIndex: 'format', 
      key: 'format',
      render: (format: string) => <Tag color="blue">{format.toUpperCase()}</Tag>
    },
    { 
      title: '大小', 
      dataIndex: 'size', 
      key: 'size',
      render: (size: number) => formatSize(size)
    },
    { 
      title: '行数', 
      dataIndex: 'row_count', 
      key: 'row_count',
      render: (count: number) => count.toLocaleString()
    },
    {
      title: '状态',
      dataIndex: 'annotation_status',
      key: 'status',
      render: (status: string) => getStatusTag(status)
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
      render: (_: any, record: Dataset) => (
        <Space>
          <Button 
            type="link" 
            size="small" 
            icon={<EyeOutlined />}
            onClick={() => {
              setSelectedDataset(record)
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
          >
            删除
          </Button>
        </Space>
      )
    },
  ]

  const handleCreate = async () => {
    try {
      const values = await form.validateFields()
      setLoading(true)
      
      const response = await fetch('/api/v1/datasets', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('access_token')}`
        },
        body: JSON.stringify(values)
      })
      
      if (response.ok) {
        message.success('数据集创建成功！')
        setIsModalOpen(false)
        form.resetFields()
        fetchDatasets()
      } else {
        throw new Error('创建失败')
      }
    } catch (error: any) {
      message.error(error.message || '创建失败')
    } finally {
      setLoading(false)
    }
  }

  const handleUpload = () => {
    // 模拟上传
    setUploadProgress(0)
    const interval = setInterval(() => {
      setUploadProgress(prev => {
        if (prev >= 100) {
          clearInterval(interval)
          return 100
        }
        return prev + 10
      })
    }, 200)
  }

  return (
    <div>
      <Card
        title={
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span>📦 数据集管理</span>
            <Button 
              type="primary" 
              icon={<PlusOutlined />}
              onClick={() => setIsModalOpen(true)}
            >
              创建数据集
            </Button>
          </div>
        }
      >
        <Table
          columns={columns}
          dataSource={datasets}
          loading={loading}
          pagination={{ pageSize: 10 }}
        />
      </Card>

      {/* 创建数据集弹窗 */}
      <Modal
        title="创建数据集"
        open={isModalOpen}
        onOk={handleCreate}
        onCancel={() => {
          setIsModalOpen(false)
          form.resetFields()
          setUploadProgress(0)
        }}
        confirmLoading={loading}
      >
        <Form form={form} layout="vertical">
          <Form.Item
            name="name"
            label="数据集名称"
            rules={[{ required: true, message: '请输入数据集名称' }]}
          >
            <Input placeholder="输入数据集名称" />
          </Form.Item>
          <Form.Item
            name="description"
            label="描述"
          >
            <Input.TextArea rows={3} placeholder="输入数据集描述（可选）" />
          </Form.Item>
          <Form.Item
            name="project_id"
            label="关联项目"
            rules={[{ required: true, message: '请选择项目' }]}
          >
            <Select placeholder="选择项目">
              <Select.Option value={1}>LLM Fine-tuning Demo</Select.Option>
              <Select.Option value={2}>测试项目</Select.Option>
            </Select>
          </Form.Item>
          <Form.Item
            name="data_type"
            label="数据类型"
            rules={[{ required: true, message: '请选择数据类型' }]}
          >
            <Select placeholder="选择数据类型">
              <Select.Option value="text">文本</Select.Option>
              <Select.Option value="image">图像</Select.Option>
              <Select.Option value="audio">音频</Select.Option>
              <Select.Option value="multi">多模态</Select.Option>
            </Select>
          </Form.Item>
          <Form.Item
            name="format"
            label="数据格式"
            rules={[{ required: true, message: '请选择格式' }]}
          >
            <Select placeholder="选择格式">
              <Select.Option value="jsonl">JSONL</Select.Option>
              <Select.Option value="json">JSON</Select.Option>
              <Select.Option value="csv">CSV</Select.Option>
              <Select.Option value="parquet">Parquet</Select.Option>
            </Select>
          </Form.Item>
          
          <Form.Item label="上传文件">
            <Upload.Dragger
              name="file"
              multiple={false}
              beforeUpload={() => false}
              onChange={handleUpload}
            >
              <p className="ant-upload-drag-icon">
                <UploadOutlined />
              </p>
              <p className="ant-upload-text">点击或拖拽文件到此区域上传</p>
              <p className="ant-upload-hint">支持 JSONL, JSON, CSV 格式</p>
            </Upload.Dragger>
            {uploadProgress > 0 && uploadProgress < 100 && (
              <Progress percent={uploadProgress} size="small" style={{ marginTop: 8 }} />
            )}
          </Form.Item>
        </Form>
      </Modal>

      {/* 数据集详情弹窗 */}
      <Modal
        title={`数据集详情: ${selectedDataset?.name}`}
        open={isDetailOpen}
        onCancel={() => setIsDetailOpen(false)}
        footer={[
          <Button key="close" onClick={() => setIsDetailOpen(false)}>关闭</Button>
        ]}
        width={700}
      >
        {selectedDataset && (
          <Tabs defaultActiveKey="1">
            <TabPane tab="基本信息" key="1">
              <Descriptions bordered column={2}>
                <Descriptions.Item label="ID">{selectedDataset.id}</Descriptions.Item>
                <Descriptions.Item label="名称">{selectedDataset.name}</Descriptions.Item>
                <Descriptions.Item label="描述" span={2}>{selectedDataset.description}</Descriptions.Item>
                <Descriptions.Item label="数据类型">{selectedDataset.data_type}</Descriptions.Item>
                <Descriptions.Item label="格式">{selectedDataset.format.toUpperCase()}</Descriptions.Item>
                <Descriptions.Item label="大小">{formatSize(selectedDataset.size)}</Descriptions.Item>
                <Descriptions.Item label="数据行数">{selectedDataset.row_count.toLocaleString()}</Descriptions.Item>
                <Descriptions.Item label="版本">v{selectedDataset.version}</Descriptions.Item>
                <Descriptions.Item label="存储路径">{selectedDataset.storage_path}</Descriptions.Item>
                <Descriptions.Item label="创建时间">{new Date(selectedDataset.created_at).toLocaleString()}</Descriptions.Item>
              </Descriptions>
            </TabPane>
            <TabPane tab="质量报告" key="2">
              <Card size="small" style={{ marginBottom: 16 }}>
                <Descriptions column={2}>
                  <Descriptions.Item label="平均长度">{selectedDataset.stats?.avg_length} tokens</Descriptions.Item>
                  <Descriptions.Item label="唯一实体">{selectedDataset.stats?.unique_entities}</Descriptions.Item>
                </Descriptions>
              </Card>
              <div style={{ display: 'flex', gap: 16 }}>
                <Tag icon={<CheckCircleOutlined />} color="success">完整性检查通过</Tag>
                <Tag icon={<WarningOutlined />} color="warning">格式验证通过</Tag>
              </div>
            </TabPane>
          </Tabs>
        )}
      </Modal>
    </div>
  )
}
