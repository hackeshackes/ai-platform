import React, { useState, useEffect } from 'react';
import { Card, Table, Button, Modal, Form, Input, Select, InputNumber, Space, Tag, Progress, Timeline, Descriptions, Alert } from 'antd';
import { PlayCircleOutlined, PlusOutlined, EyeOutlined, DeleteOutlined, RocketOutlined } from '@ant-design/icons';
import { api } from '../../api/client';

const { Option } = Select;
const { TextArea } = Input;

interface HPOParam {
  name: string;
  type: 'categorical' | 'continuous' | 'integer';
  values: number[];
  log_scale: boolean;
}

interface HPOTask {
  id: string;
  name: string;
  method: string;
  status: 'running' | 'completed' | 'failed';
  progress: number;
  best_value: number;
  best_params: Record<string, any>;
  trials: number;
}

export const AutoMLPage: React.FC = () => {
  const [tasks, setTasks] = useState<HPOTask[]>([]);
  const [methods, setMethods] = useState<{id: string, name: string}[]>([]);
  const [modalVisible, setModalVisible] = useState(false);
  const [form] = Form.useForm();

  useEffect(() => {
    loadMethods();
    loadTasks();
  }, []);

  const loadMethods = async () => {
    try {
      const res = await api.get('/automl/hpo/methods');
      setMethods(res.data.methods);
    } catch (e) {
      console.error('Failed to load methods:', e);
    }
  };

  const loadTasks = async () => {
    try {
      // 模拟数据
      setTasks([
        {
          id: '1',
          name: 'BERT超参数优化',
          method: 'bayesian',
          status: 'completed',
          progress: 100,
          best_value: 0.95,
          best_params: { lr: 0.001, batch_size: 32 },
          trials: 50
        },
        {
          id: '2',
          name: 'GPT微调优化',
          method: 'hyperband',
          status: 'running',
          progress: 65,
          best_value: 0.88,
          best_params: {},
          trials: 32
        }
      ]);
    } catch (e) {
      console.error('Failed to load tasks:', e);
    }
  };

  const startHPO = async (values: any) => {
    try {
      const params = values.params.map((p: any) => ({
        name: p.name,
        type: p.type,
        values: p.type === 'categorical' 
          ? p.categorical_values 
          : [p.min, p.max],
        log_scale: p.log_scale
      }));

      await api.post('/automl/hpo/start', {
        name: values.name,
        method: values.method,
        max_trials: values.max_trials,
        params: params
      });

      setModalVisible(false);
      form.resetFields();
      loadTasks();
    } catch (e) {
      console.error('Failed to start HPO:', e);
    }
  };

  const columns = [
    {
      title: '任务',
      dataIndex: 'name',
      key: 'name',
    },
    {
      title: '方法',
      dataIndex: 'method',
      key: 'method',
      render: (v: string) => <Tag>{v}</Tag>
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (v: string) => (
        <Tag color={v === 'completed' ? 'green' : v === 'running' ? 'blue' : 'red'}>
          {v}
        </Tag>
      )
    },
    {
      title: '进度',
      dataIndex: 'progress',
      key: 'progress',
      render: (v: number) => <Progress percent={v} size="small" />
    },
    {
      title: '最佳值',
      dataIndex: 'best_value',
      key: 'best_value',
      render: (v: number) => v.toFixed(4)
    },
    {
      title: '试验数',
      dataIndex: 'trials',
      key: 'trials',
    },
    {
      title: '操作',
      key: 'action',
      render: (_: any, record: HPOTask) => (
        <Space>
          <Button size="small" icon={<EyeOutlined />}>查看</Button>
          {record.status === 'running' && (
            <Button size="small" danger icon={<DeleteOutlined />}>停止</Button>
          )}
        </Space>
      )
    }
  ];

  return (
    <div className="automl-page">
      <Card
        title={
          <Space>
            <RocketOutlined />
            AutoML - 超参数优化
          </Space>
        }
        extra={
          <Button type="primary" icon={<PlusOutlined />} onClick={() => setModalVisible(true)}>
            新建优化任务
          </Button>
        }
      >
        <Alert
          message="AutoML功能"
          description="支持贝叶斯优化、Hyperband等超参数优化方法，自动搜索最优模型参数。"
          type="info"
          showIcon
          style={{ marginBottom: 16 }}
        />

        <Table
          columns={columns}
          dataSource={tasks}
          rowKey="id"
          pagination={false}
        />
      </Card>

      <Modal
        title="新建超参数优化任务"
        open={modalVisible}
        onCancel={() => setModalVisible(false)}
        footer={null}
        width={720}
      >
        <Form form={form} layout="vertical" onFinish={startHPO}>
          <Form.Item name="name" label="任务名称" rules={[{ required: true }]}>
            <Input placeholder="输入任务名称" />
          </Form.Item>

          <Form.Item name="method" label="优化方法" rules={[{ required: true }]}>
            <Select placeholder="选择优化方法">
              {methods.map(m => (
                <Option key={m.id} value={m.id}>{m.name}</Option>
              ))}
            </Select>
          </Form.Item>

          <Form.Item name="max_trials" label="最大试验次数" initialValue={100}>
            <InputNumber min={1} max={1000} style={{ width: '100%' }} />
          </Form.Item>

          <Form.Item>
            <Button type="primary" htmlType="submit" icon={<PlayCircleOutlined />}>
              启动优化
            </Button>
          </Form.Item>
        </Form>
      </Modal>
    </div>
  );
};

export default AutoMLPage;
