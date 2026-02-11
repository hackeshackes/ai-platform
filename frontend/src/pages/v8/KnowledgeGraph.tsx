// 知识图谱2.0页面 - v8
import React, { useState, useEffect } from 'react'
import { Card, Table, Button, Modal, Form, Input, Select, Tag, message, Space, Tabs, Drawer, Descriptions, List } from 'antd'
import { PlusOutlined, SearchOutlined, ShareAltOutlined, ApiOutlined, NodeIndexOutlined } from '@ant-design/icons'

const { Option } = Select
const { TabPane } = Tabs
const { TextArea } = Input

interface KGEntity {
  id: string
  name: string
  type: string
  properties: Record<string, any>
  created_at: string
}

interface KGRelation {
  id: string
  source: string
  target: string
  relation: string
}

export function KnowledgeGraphPage() {
  const [entities, setEntities] = useState<KGEntity[]>([])
  const [relations, setRelations] = useState<KGRelation[]>([])
  const [loading, setLoading] = useState(false)
  const [entityModal, setEntityModal] = useState(false)
  const [relationModal, setRelationModal] = useState(false)
  const [reasoningModal, setReasoningModal] = useState(false)
  const [semanticModal, setSemanticModal] = useState(false)
  const [detailDrawer, setDetailDrawer] = useState(false)
  const [selectedEntity, setSelectedEntity] = useState<KGEntity | null>(null)
  const [form] = Form.useForm()
  const [relationForm] = Form.useForm()
  const [reasoningResult, setReasoningResult] = useState<any>(null)
  const [semanticResult, setSemanticResult] = useState<any>(null)

  // 获取实体列表
  const fetchEntities = async () => {
    setLoading(true)
    try {
      const res = await fetch('http://localhost:8000/api/v1/kg/v2/entities')
      const data = await res.json()
      if (Array.isArray(data)) {
        setEntities(data)
      }
    } catch (e) {
      message.error('获取实体失败')
    }
    setLoading(false)
  }

  // 获取关系列表
  const fetchRelations = async () => {
    try {
      const res = await fetch('http://localhost:8000/api/v1/kg/v2/relations')
      const data = await res.json()
      if (Array.isArray(data)) {
        setRelations(data)
      }
    } catch (e) {
      console.error(e)
    }
  }

  useEffect(() => {
    fetchEntities()
    fetchRelations()
  }, [])

  // 创建实体
  const handleCreateEntity = async (values: any) => {
    try {
      const res = await fetch('http://localhost:8000/api/v1/kg/v2/entities', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(values)
      })
      const data = await res.json()
      if (data.success) {
        message.success('实体创建成功')
        setEntityModal(false)
        form.resetFields()
        fetchEntities()
      } else {
        message.error(data.detail || '创建失败')
      }
    } catch (e) {
      message.error('创建失败')
    }
  }

  // 创建关系
  const handleCreateRelation = async (values: any) => {
    try {
      const res = await fetch('http://localhost:8000/api/v1/kg/v2/relations', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(values)
      })
      const data = await res.json()
      if (data.success) {
        message.success('关系创建成功')
        setRelationModal(false)
        relationForm.resetFields()
        fetchRelations()
      } else {
        message.error(data.detail || '创建失败')
      }
    } catch (e) {
      message.error('创建失败')
    }
  }

  // 知识推理
  const handleReasoning = async (values: any) => {
    try {
      const res = await fetch('http://localhost:8000/api/v1/kg/v2/reasoning', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(values)
      })
      const data = await res.json()
      setReasoningResult(data)
    } catch (e) {
      message.error('推理失败')
    }
  }

  // 语义搜索
  const handleSemanticSearch = async (values: any) => {
    try {
      const res = await fetch(`http://localhost:8000/api/v1/kg/v2/semantic-search?query=${encodeURIComponent(values.query)}`)
      const data = await res.json()
      setSemanticResult(data)
    } catch (e) {
      message.error('搜索失败')
    }
  }

  const entityColumns = [
    { title: '名称', dataIndex: 'name', key: 'name' },
    { title: '类型', dataIndex: 'type', key: 'type', render: (t: string) => <Tag color="blue">{t}</Tag> },
    { 
      title: '属性', 
      key: 'properties',
      render: (_: any, record: KGEntity) => (
        <span>{Object.keys(record.properties || {}).length} 个属性</span>
      )
    },
    { title: '创建时间', dataIndex: 'created_at', key: 'created_at', render: (t: string) => new Date(t).toLocaleString() },
    {
      title: '操作',
      key: 'action',
      render: (_: any, record: KGEntity) => (
        <Button type="link" onClick={() => {
          setSelectedEntity(record)
          setDetailDrawer(true)
        }}>
          查看详情
        </Button>
      )
    }
  ]

  const relationColumns = [
    { title: '源实体', dataIndex: 'source', key: 'source' },
    { title: '关系', dataIndex: 'relation', key: 'relation', render: (r: string) => <Tag>{r}</Tag> },
    { title: '目标实体', dataIndex: 'target', key: 'target' },
  ]

  return (
    <div>
      <h2>🧠 知识图谱2.0</h2>
      <Tabs defaultActiveKey="entities">
        <TabPane tab="实体管理" key="entities" icon={<NodeIndexOutlined />}>
          <Card extra={
            <Button type="primary" icon={<PlusOutlined />} onClick={() => setEntityModal(true)}>
              添加实体
            </Button>
          }>
            <Table 
              dataSource={entities} 
              columns={entityColumns} 
              rowKey="id"
              loading={loading}
            />
          </Card>
        </TabPane>
        <TabPane tab="关系管理" key="relations" icon={<ShareAltOutlined />}>
          <Card extra={
            <Button type="primary" icon={<PlusOutlined />} onClick={() => setRelationModal(true)}>
              添加关系
            </Button>
          }>
            <Table 
              dataSource={relations} 
              columns={relationColumns} 
              rowKey="id"
            />
          </Card>
        </TabPane>
        <TabPane tab="知识推理" key="reasoning" icon={<ApiOutlined />}>
          <Card title="推理引擎">
            <Form layout="vertical" onFinish={handleReasoning}>
              <Form.Item name="entity_id" label="实体ID" rules={[{ required: true }]}>
                <Input placeholder="输入实体ID进行推理" />
              </Form.Item>
              <Form.Item name="type" label="推理类型" initialValue="rule">
                <Select>
                  <Option value="rule">规则推理</Option>
                  <Option value="neural">神经网络推理</Option>
                  <Option value="hybrid">混合推理</Option>
                </Select>
              </Form.Item>
              <Button type="primary" htmlType="submit">开始推理</Button>
            </Form>
            {reasoningResult && (
              <Card title="推理结果" style={{ marginTop: 16 }}>
                <pre>{JSON.stringify(reasoningResult, null, 2)}</pre>
              </Card>
            )}
          </Card>
        </TabPane>
        <TabPane tab="语义搜索" key="semantic">
          <Card title="混合语义搜索">
            <Form layout="inline" onFinish={handleSemanticSearch}>
              <Form.Item name="query" rules={[{ required: true }]}>
                <Input placeholder="输入搜索关键词" style={{ width: 300 }} />
              </Form.Item>
              <Button type="primary" htmlType="submit" icon={<SearchOutlined />}>
                搜索
              </Button>
            </Form>
            {semanticResult && (
              <Card title="搜索结果" style={{ marginTop: 16 }}>
                <List
                  dataSource={Array.isArray(semanticResult) ? semanticResult : []}
                  renderItem={(item: any) => (
                    <List.Item>
                      <List.Item.Meta title={item.name} description={item.description} />
                    </List.Item>
                  )}
                />
              </Card>
            )}
          </Card>
        </TabPane>
      </Tabs>

      {/* 添加实体弹窗 */}
      <Modal
        title="添加实体"
        open={entityModal}
        onCancel={() => setEntityModal(false)}
        footer={null}
      >
        <Form form={form} layout="vertical" onFinish={handleCreateEntity}>
          <Form.Item name="name" label="名称" rules={[{ required: true }]}>
            <Input placeholder="实体名称" />
          </Form.Item>
          <Form.Item name="type" label="类型" rules={[{ required: true }]}>
            <Select placeholder="选择类型">
              <Option value="Person">人物</Option>
              <Option value="Organization">组织</Option>
              <Option value="Concept">概念</Option>
              <Option value="Product">产品</Option>
              <Option value="Event">事件</Option>
              <Option value="Location">地点</Option>
            </Select>
          </Form.Item>
          <Form.Item name="properties" label="属性 (JSON)">
            <TextArea rows={4} placeholder='{"key": "value"}' />
          </Form.Item>
          <Button type="primary" htmlType="submit" block>创建</Button>
        </Form>
      </Modal>

      {/* 添加关系弹窗 */}
      <Modal
        title="添加关系"
        open={relationModal}
        onCancel={() => setRelationModal(false)}
        footer={null}
      >
        <Form form={relationForm} layout="vertical" onFinish={handleCreateRelation}>
          <Form.Item name="source" label="源实体" rules={[{ required: true }]}>
            <Input placeholder="源实体ID" />
          </Form.Item>
          <Form.Item name="relation" label="关系类型" rules={[{ required: true }]}>
            <Select placeholder="选择关系">
              <Option value="RELATED_TO">相关</Option>
              <Option value="PART_OF">部分</Option>
              <Option value="KNOWS">认识</Option>
              <Option value="WORKS_AT">工作于</Option>
              <Option value="LOCATED_IN">位于</Option>
              <Option value="INCLUDES">包含</Option>
            </Select>
          </Form.Item>
          <Form.Item name="target" label="目标实体" rules={[{ required: true }]}>
            <Input placeholder="目标实体ID" />
          </Form.Item>
          <Button type="primary" htmlType="submit" block>创建</Button>
        </Form>
      </Modal>

      {/* 实体详情抽屉 */}
      <Drawer
        title="实体详情"
        open={detailDrawer}
        onClose={() => setDetailDrawer(false)}
        width={400}
      >
        {selectedEntity && (
          <Descriptions column={1}>
            <Descriptions.Item label="ID">{selectedEntity.id}</Descriptions.Item>
            <Descriptions.Item label="名称">{selectedEntity.name}</Descriptions.Item>
            <Descriptions.Item label="类型">
              <Tag>{selectedEntity.type}</Tag>
            </Descriptions.Item>
            <Descriptions.Item label="创建时间">
              {new Date(selectedEntity.created_at).toLocaleString()}
            </Descriptions.Item>
            <Descriptions.Item label="属性">
              <pre>{JSON.stringify(selectedEntity.properties, null, 2)}</pre>
            </Descriptions.Item>
          </Descriptions>
        )}
      </Drawer>
    </div>
  )
}
