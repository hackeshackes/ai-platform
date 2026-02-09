import { Card, Table, Tag, Button, Space, Input, Modal, Form, Select, Row, Col } from 'antd'
import { PlusOutlined, PlayCircleOutlined, PauseCircleOutlined, ApiOutlined, CloudUploadOutlined } from '@ant-design/icons'
import { useLang } from '../locales'
import { useState } from 'react'
import { TextArea } from 'antd/lib/input/TextArea'

export default function Inference() {
  const { t } = useLang()
  const [isModalOpen, setIsModalOpen] = useState(false)
  const [apiTestOpen, setApiTestOpen] = useState(false)
  const [form] = Form.useForm()

  const services = [
    { key: 1, name: 'qwen-chat-service', model: 'Qwen-7B-Lora-v1', replicas: 2, status: 'running', endpoint: 'api/v1/inference/qwen-chat' },
    { key: 2, name: 'llama2-completion', model: 'Llama-2-13B-Full', replicas: 1, status: 'stopped', endpoint: 'api/v1/inference/llama2' },
    { key: 3, name: 'baichuan-embedding', model: 'Baichuan-7B-Base', replicas: 3, status: 'running', endpoint: 'api/v1/inference/baichuan-embed' },
  ]

  const columns = [
    { title: t('inference.name'), dataIndex: 'name', key: 'name' },
    { title: t('inference.model'), dataIndex: 'model', key: 'model' },
    { title: t('inference.replicas'), dataIndex: 'replicas', key: 'replicas' },
    {
      title: t('experiments.status'),
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => (
        <Tag color={status === 'running' ? 'green' : 'default'}>{status === 'running' ? t('inference.running') : t('inference.stopped')}</Tag>
      )
    },
    { title: t('common.apiEndpoint'), dataIndex: 'endpoint', key: 'endpoint', render: (v: string) => <code style={{ fontSize: 12 }}>{v}</code> },
    {
      title: t('inference.action'),
      key: 'action',
      render: (_: any, record: any) => (
        <Space>
          {record.status === 'running' ? (
            <Button type="link" size="small" icon={<PauseCircleOutlined />}>{t('tasks.stop')}</Button>
          ) : (
            <Button type="primary" size="small" icon={<PlayCircleOutlined />}>{t('tasks.start')}</Button>
          )}
          <Button type="link" size="small" icon={<ApiOutlined />} onClick={() => setApiTestOpen(true)}>{t('inference.test')}</Button>
          <Button type="link" size="small" danger>{t('common.delete')}</Button>
        </Space>
      )
    },
  ]

  const handleDeploy = () => {
    setIsModalOpen(false)
  }

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 24 }}>
        <h1 style={{ fontSize: 24, margin: 0 }}>{t('inference.title')}</h1>
        <Button type="primary" icon={<PlusOutlined />} onClick={() => setIsModalOpen(true)}>
          {t('inference.deploy')}
        </Button>
      </div>

      <Card>
        <Table columns={columns} dataSource={services} pagination={{ pageSize: 10 }} />
      </Card>

      <Modal title={t('inference.deploy')} open={isModalOpen} onCancel={() => setIsModalOpen(false)} onOk={handleDeploy} width={600}>
        <Form form={form} layout="vertical">
          <Form.Item name="name" label={t('inference.name')} rules={[{ required: true }]}>
            <Input placeholder={t('common.inputPlaceholder')} />
          </Form.Item>
          <Form.Item name="model" label={t('inference.model')} rules={[{ required: true }]}>
            <Select placeholder={t('common.selectPlaceholder')} options={[
              { value: 'qwen-7b', label: 'Qwen-7B-Lora-v1' },
              { value: 'llama-13b', label: 'Llama-2-13B-Full' },
              { value: 'baichuan-7b', label: 'Baichuan-7B-Base' },
            ]} />
          </Form.Item>
          <Form.Item name="replicas" label={t('inference.replicas')} initialValue={1}>
            <Select options={[
              { value: 1, label: '1 replica' },
              { value: 2, label: '2 replicas' },
              { value: 3, label: '3 replicas' },
              { value: 5, label: '5 replicas' },
            ]} />
          </Form.Item>
        </Form>
      </Modal>

      <Modal title={t('inference.test')} open={apiTestOpen} onCancel={() => setApiTestOpen(false)} footer={null} width={700}>
        <Row gutter={16}>
          <Col span={12}>
            <TextArea rows={4} placeholder={t('common.inputPlaceholder')} defaultValue="你好，请介绍一下你自己。" />
            <Button type="primary" style={{ marginTop: 8 }} block>{t('training.start')}</Button>
          </Col>
          <Col span={12}>
            <div style={{ background: '#f5f5f5', padding: 12, borderRadius: 4, minHeight: 100 }}>
              <pre style={{ margin: 0, fontSize: 12 }}>Response will appear here...</pre>
            </div>
          </Col>
        </Row>
      </Modal>
    </div>
  )
}
