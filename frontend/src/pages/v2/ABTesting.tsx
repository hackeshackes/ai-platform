import React, { useState, useEffect } from 'react';
import { Card, Table, Button, Modal, Form, Input, Select, Space, Tag, Tabs, Statistic, Progress, Badge, Typography, message } from 'antd';
import { PlusOutlined, PlayCircleOutlined, PauseCircleOutlined, TrophyOutlined, BarChartOutlined } from '@ant-design/icons';
import { api } from '../../api/client';

const { Text, Title } = Typography;

interface Experiment {
  experiment_id: string;
  name: string;
  description: string;
  status: string;
  variants_count: number;
  created_by: string;
}

interface Result {
  variant_id: string;
  sample_size: number;
  conversions: number;
  conversion_rate: number;
  p_value: number;
}

export const ABTestingPage: React.FC = () => {
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [results, setResults] = useState<Record<string, Result[]>>({});
  const [loading, setLoading] = useState(false);
  const [createModalVisible, setCreateModalVisible] = useState(false);
  const [selectedExperiment, setSelectedExperiment] = useState<Experiment | null>(null);
  const [form] = Form.useForm();

  useEffect(() => {
    loadExperiments();
  }, []);

  const loadExperiments = async () => {
    setLoading(true);
    try {
      const res = await api.get('/abtesting/experiments');
      setExperiments(res.data.experiments || []);
    } catch (e) {
      console.error('Failed to load experiments:', e);
    } finally {
      setLoading(false);
    }
  };

  const createExperiment = async (values: any) => {
    try {
      await api.post('/abtesting/experiments', {
        name: values.name,
        description: values.description,
        variants: [
          { variant_id: 'control', name: 'Control' },
          { variant_id: 'treatment', name: 'Treatment' }
        ]
      });
      setCreateModalVisible(false);
      form.resetFields();
      loadExperiments();
      message.success('Experiment created');
    } catch (e) {
      message.error('Failed to create experiment');
    }
  };

  const startExperiment = async (id: string) => {
    try {
      await api.post(`/abtesting/experiments/${id}/start`);
      loadExperiments();
      message.success('Experiment started');
    } catch (e) {
      message.error('Failed to start');
    }
  };

  const completeExperiment = async (id: string) => {
    try {
      await api.post(`/abtesting/experiments/${id}/complete`);
      loadExperiments();
      message.success('Experiment completed');
    } catch (e) {
      message.error('Failed to complete');
    }
  };

  const viewResults = async (experiment: Experiment) => {
    setSelectedExperiment(experiment);
    try {
      const res = await api.get(`/abtesting/experiments/${experiment.experiment_id}/results`);
      setResults({ [experiment.experiment_id]: res.data.results || [] });
    } catch (e) {
      message.error('Failed to load results');
    }
  };

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'running': return <Badge status="processing" text="Running" />;
      case 'completed': return <Badge status="success" text="Completed" />;
      case 'paused': return <Badge status="warning" text="Paused" />;
      default: return <Badge status="default" text={status} />;
    }
  };

  const columns = [
    { title: 'Name', dataIndex: 'name', key: 'name' },
    { title: 'Status', dataIndex: 'status', key: 'status', render: (s: string) => getStatusBadge(s) },
    { title: 'Variants', dataIndex: 'variants_count', key: 'variants' },
    { title: 'Created By', dataIndex: 'created_by', key: 'created_by' },
    {
      title: 'Actions',
      key: 'actions',
      render: (_: any, record: Experiment) => (
        <Space>
          {record.status === 'draft' && (
            <Button type="link" icon={<PlayCircleOutlined />} onClick={() => startExperiment(record.experiment_id)}>Start</Button>
          )}
          {record.status === 'running' && (
            <>
              <Button type="link" onClick={() => viewResults(record)}>Results</Button>
              <Button type="link" icon={<PauseCircleOutlined />} onClick={() => completeExperiment(record.experiment_id)}>Complete</Button>
            </>
          )}
          {record.status === 'completed' && (
            <Button type="link" icon={<TrophyOutlined />} onClick={() => viewResults(record)}>View Results</Button>
          )}
        </Space>
      )
    },
  ];

  const resultColumns = [
    { title: 'Variant', dataIndex: 'variant_id', key: 'variant' },
    { title: 'Sample Size', dataIndex: 'sample_size', key: 'sample' },
    { title: 'Conversions', dataIndex: 'conversions', key: 'conversions' },
    { 
      title: 'Rate', 
      dataIndex: 'conversion_rate', 
      key: 'rate',
      render: (r: number) => `${(r * 100).toFixed(2)}%`
    },
    { title: 'P-Value', dataIndex: 'p_value', key: 'pvalue', render: (p: number) => p?.toFixed(4) || '-' },
  ];

  return (
    <div style={{ padding: 24 }}>
      <Title level={2}><BarChartOutlined /> A/B Testing</Title>

      <Card>
        <div style={{ marginBottom: 16 }}>
          <Button type="primary" icon={<PlusOutlined />} onClick={() => setCreateModalVisible(true)}>
            Create Experiment
          </Button>
        </div>
        <Table columns={columns} dataSource={experiments} rowKey="experiment_id" loading={loading} />
      </Card>

      {selectedExperiment && results[selectedExperiment.experiment_id] && (
        <Card title={`Results: ${selectedExperiment.name}`} style={{ marginTop: 16 }}>
          <Table columns={resultColumns} dataSource={results[selectedExperiment.experiment_id]} rowKey="variant_id" pagination={false} />
        </Card>
      )}

      <Modal title="Create Experiment" open={createModalVisible} onCancel={() => setCreateModalVisible(false)} footer={null}>
        <Form form={form} onFinish={createExperiment} layout="vertical">
          <Form.Item name="name" label="Name" rules={[{ required: true }]}><Input /></Form.Item>
          <Form.Item name="description" label="Description"><Input.TextArea /></Form.Item>
          <Form.Item><Button type="primary" htmlType="submit">Create</Button></Form.Item>
        </Form>
      </Modal>
    </div>
  );
};
