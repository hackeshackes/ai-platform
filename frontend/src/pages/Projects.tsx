/**
 * AI Platform - Projects 页面
 */

import { Card, Table, Button, Tag, Space, Input, Modal, Form, message, Popconfirm } from 'antd'
import { PlusOutlined, SearchOutlined, EditOutlined, DeleteOutlined } from '@ant-design/icons'
import { useLang } from '../locales'
import { useState, useEffect } from 'react'
import { projectAPI } from '../api/client'

export default function Projects() {
  const { t } = useLang()
  const [isModalOpen, setIsModalOpen] = useState(false)
  const [loading, setLoading] = useState(false)
  const [projects, setProjects] = useState<any[]>([])
  const [form] = Form.useForm()
  const [searchText, setSearchText] = useState('')

  // 获取项目列表
  const fetchProjects = async () => {
    try {
      setLoading(true)
      const response = await projectAPI.list()
      if (response.projects) {
        setProjects(response.projects.map((p: any) => ({
          key: p.id,
          id: p.id,
          name: p.name,
          description: p.description || '-',
          status: p.status || 'active',
          experiments: 0,
          created: new Date(p.created_at).toLocaleDateString(),
        })))
      }
    } catch (error: any) {
      message.error(error.response?.data?.detail || '获取项目列表失败')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchProjects()
  }, [])

  // 创建项目
  const handleCreate = async () => {
    try {
      const values = await form.validateFields()
      setLoading(true)
      await projectAPI.create(values)
      message.success('项目创建成功！')
      setIsModalOpen(false)
      form.resetFields()
      fetchProjects()
    } catch (error: any) {
      message.error(error.response?.data?.detail || '创建项目失败')
    } finally {
      setLoading(false)
    }
  }

  // 删除项目
  const handleDelete = async (id: string) => {
    try {
      setLoading(true)
      await projectAPI.delete(id)
      message.success('项目已删除')
      fetchProjects()
    } catch (error: any) {
      message.error(error.response?.data?.detail || '删除项目失败')
    } finally {
      setLoading(false)
    }
  }

  // 表格列
  const columns = [
    { title: 'ID', dataIndex: 'id', key: 'id', width: 60 },
    { title: t('projects.name'), dataIndex: 'name', key: 'name' },
    { title: t('common.description'), dataIndex: 'description', key: 'description' },
    {
      title: t('projects.status'),
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => (
        <Tag color={status === 'active' ? 'green' : 'default'}>
          {status === 'active' ? '活跃' : '已归档'}
        </Tag>
      )
    },
    { title: '创建时间', dataIndex: 'created', key: 'created' },
    {
      title: t('experiments.action'),
      key: 'action',
      render: (_: any, record: any) => (
        <Space>
          <Button 
            type="link" 
            size="small" 
            icon={<EditOutlined />}
          >
            编辑
          </Button>
          <Popconfirm
            title="确认删除"
            description="确定要删除这个项目吗？"
            onConfirm={() => handleDelete(record.id)}
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

  // 搜索过滤
  const filteredProjects = projects.filter(p => 
    p.name.toLowerCase().includes(searchText.toLowerCase()) ||
    p.description.toLowerCase().includes(searchText.toLowerCase())
  )

  return (
    <div>
      <Card
        title={
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span>📁 项目管理</span>
            <Space>
              <Input
                placeholder="搜索项目..."
                prefix={<SearchOutlined />}
                value={searchText}
                onChange={(e) => setSearchText(e.target.value)}
                style={{ width: 200 }}
              />
              <Button 
                type="primary" 
                icon={<PlusOutlined />}
                onClick={() => setIsModalOpen(true)}
              >
                新建项目
              </Button>
            </Space>
          </div>
        }
      >
        <Table
          columns={columns}
          dataSource={filteredProjects}
          loading={loading}
          pagination={{ pageSize: 10 }}
        />
      </Card>

      {/* 创建项目弹窗 */}
      <Modal
        title="新建项目"
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
            label="项目名称"
            rules={[{ required: true, message: '请输入项目名称' }]}
          >
            <Input placeholder="输入项目名称" />
          </Form.Item>
          <Form.Item
            name="description"
            label="项目描述"
          >
            <Input.TextArea rows={3} placeholder="输入项目描述（可选）" />
          </Form.Item>
        </Form>
      </Modal>
    </div>
  )
}
