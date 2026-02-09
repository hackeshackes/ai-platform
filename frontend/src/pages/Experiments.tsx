import { Card, Table, Button, Tag, Space, Select, Progress } from 'antd'
import { PlusOutlined, EyeOutlined } from '@ant-design/icons'
import { useLang } from '../locales'
import { useState } from 'react'

export default function Experiments() {
  const { t } = useLang()

  const experiments = [
    { key: 1, name: 'Qwen-7B-Lora-v1', baseModel: 'Qwen-1.5-7B', type: t('experiments.fineTuning'), status: 'completed', loss: 0.23, created: '2026-02-07' },
    { key: 2, name: 'Llama-2-13B-Full', baseModel: 'Llama-2-13b-hf', type: t('experiments.training'), status: 'running', loss: 0.45, created: '2026-02-06' },
    { key: 3, name: 'Baichuan-7B-Distill', baseModel: 'Baichuan-13B', type: t('experiments.distillation'), status: 'pending', loss: null, created: '2026-02-05' },
    { key: 4, name: 'ChatGLM-6B-PT', baseModel: 'ChatGLM-6B', type: t('experiments.fineTuning'), status: 'completed', loss: 0.18, created: '2026-02-04' },
  ]

  const columns = [
    { title: t('experiments.name'), dataIndex: 'name', key: 'name' },
    { title: t('experiments.baseModel'), dataIndex: 'baseModel', key: 'baseModel' },
    { title: t('experiments.type'), dataIndex: 'type', key: 'type' },
    {
      title: t('experiments.loss'),
      dataIndex: 'loss',
      key: 'loss',
      render: (loss: number | null) => loss ? loss.toFixed(4) : '-'
    },
    {
      title: t('experiments.status'),
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => {
        const colors: Record<string, string> = { completed: 'green', running: 'blue', pending: 'orange', failed: 'red' }
        const labels: Record<string, string> = { 
          completed: t('experiments.completed'), 
          running: t('experiments.running'), 
          pending: t('experiments.pending'),
          failed: t('common.failed')
        }
        return <Tag color={colors[status]}>{labels[status]}</Tag>
      }
    },
    {
      title: t('experiments.action'),
      key: 'action',
      render: () => (
        <Space>
          <Button type="link" size="small" icon={<EyeOutlined />}>{t('experiments.details')}</Button>
          <Button type="link" size="small" danger>{t('experiments.stop')}</Button>
        </Space>
      )
    },
  ]

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 24 }}>
        <h1 style={{ fontSize: 24, margin: 0 }}>{t('experiments.title')}</h1>
        <Button type="primary" icon={<PlusOutlined />}>{t('experiments.create')}</Button>
      </div>

      <Card>
        <Space style={{ marginBottom: 16 }}>
          <Select placeholder={t('experiments.baseModel')} style={{ width: 200 }} options={[
            { value: 'qwen', label: 'Qwen-1.5-7B' },
            { value: 'llama', label: 'Llama-2-13b-hf' },
            { value: 'baichuan', label: 'Baichuan-13B' },
          ]} />
          <Select placeholder={t('experiments.type')} style={{ width: 150 }} options={[
            { value: 'finetuning', label: t('experiments.fineTuning') },
            { value: 'training', label: t('experiments.training') },
            { value: 'distillation', label: t('experiments.distillation') },
          ]} />
        </Space>
        <Table columns={columns} dataSource={experiments} pagination={{ pageSize: 10 }} />
      </Card>
    </div>
  )
}
