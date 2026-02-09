import { Card, Row, Col, Statistic, Table, Tag, Progress } from 'antd'
import { ProjectOutlined, ExperimentOutlined, RocketOutlined, CloudServerOutlined } from '@ant-design/icons'
import { useLang } from '../locales'

export default function Dashboard() {
  const { t } = useLang()

  const stats = [
    { title: t('projects.title'), value: 12, icon: <ProjectOutlined />, color: '#1890ff' },
    { title: t('experiments.title'), value: 28, icon: <ExperimentOutlined />, color: '#52c41a' },
    { title: 'Tasks', value: 156, icon: <RocketOutlined />, color: '#faad14' },
    { title: 'Inference', value: 8, icon: <CloudServerOutlined />, color: '#722ed1' },
  ]

  const recentTasks = [
    { key: 1, name: 'Qwen-7B Fine-tuning', type: t('experiments.training'), progress: 75, status: 'running', gpu: '2/4' },
    { key: 2, name: 'Llama-2-13B Eval', type: t('inference.title'), progress: 100, status: 'completed', gpu: '-' },
    { key: 3, name: 'Data Preprocessing', type: t('tasks.data'), progress: 45, status: 'running', gpu: '1/4' },
    { key: 4, name: 'Baichuan-7B蒸馏', type: t('experiments.distillation'), progress: 0, status: 'pending', gpu: '-' },
    { key: 5, name: 'Model Quantization', type: 'Optimization', progress: 100, status: 'completed', gpu: '-' },
  ]

  const columns = [
    { title: t('tasks.name'), dataIndex: 'name', key: 'name' },
    { title: t('tasks.type'), dataIndex: 'type', key: 'type' },
    { 
      title: t('tasks.progress'), 
      key: 'progress',
      render: (_: any, record: any) => <Progress percent={record.progress} size="small" />
    },
    {
      title: t('experiments.status'),
      key: 'status',
      render: (_: any, record: any) => {
        const colors: Record<string, string> = { running: 'blue', completed: 'green', pending: 'orange', failed: 'red' }
        const labels: Record<string, string> = { 
          running: t('experiments.running'), 
          completed: t('experiments.completed'), 
          pending: t('experiments.pending'),
          failed: t('common.failed')
        }
        return <Tag color={colors[record.status]}>{labels[record.status]}</Tag>
      }
    },
    { title: t('tasks.gpu'), dataIndex: 'gpu', key: 'gpu' },
  ]

  return (
    <div>
      <h1 style={{ fontSize: 24, marginBottom: 24 }}>{t('dashboard.title')}</h1>
      
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        {stats.map((stat, index) => (
          <Col xs={24} sm={12} md={6} key={index}>
            <Card>
              <Statistic
                title={stat.title}
                value={stat.value}
                prefix={stat.icon}
                valueStyle={{ color: stat.color }}
              />
            </Card>
          </Col>
        ))}
      </Row>

      <Card title={t('dashboard.recent') || 'Recent Tasks'}>
        <Table columns={columns} dataSource={recentTasks} pagination={false} size="small" />
      </Card>
    </div>
  )
}
