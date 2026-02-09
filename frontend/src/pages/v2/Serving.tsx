import React, { useState, useEffect } from 'react';
import { Card, Table, Button, Modal, Form, Input, InputNumber, Space, Tag, Tabs, Badge, Typography, message } from 'antd';
import { PlusOutlined, PlayCircleOutlined, StopOutlined, DeleteOutlined, ControlOutlined, SwapOutlined } from '@ant-design/icons';
import { api } from '../../api/client';

const { Text, Title } = Typography;

interface Endpoint {
  endpoint_id: string;
  name: string;
  model_id: string;
  version: string;
  status: string;
  replicas: number;
  url: string;
}

export const ServingPage: React.FC = () => {
  const [endpoints, setEndpoints] = useState<Endpoint[]>([]);
  const [loading, setLoading] = useState(false);
  const [createModalVisible, setCreateModalVisible] = useState(false);
  const [form] = Form.useForm();

  useEffect(() => {
    loadEndpoints();
  }, []);

  const loadEndpoints = async () => {
    setLoading(true);
    try {
      const res = await api.get('/serving/endpoints');
      setEndpoints(res.data.endpoints || []);
    } catch (e) {
      console.error('Failed to load endpoints:', e);
    } finally {
      setLoading(false);
    }
  };

  const createEndpoint = async (values: any) => {
    try {
      await api.post('/serving/endpoints', values);
      setCreateModalVisible(false);
      form.resetFields();
      loadEndpoints();
      message.success('Endpoint created');
    } catch (e) {
      console.error('Failed to create endpoint:', e);
    }
  };

  const startEndpoint = async (id: string) => {
    try {
      await api.post(`/serving/endpoints/${id}/start`);
      loadEndpoints();
      message.success('Endpoint started');
    } catch (e) {
      message.error('Failed to start endpoint');
    }
  };

  const stopEndpoint = async (id: string) => {
    try {
      await api.post(`/serving/endpoints/${id}/stop`);
      loadEndpoints();
      message.success('Endpoint stopped');
    } catch (e) {
      message.error('Failed to stop endpoint');
    }
  };

  const deleteEndpoint = async (id: string) => {
    try {
      await api.delete(`/serving/endpoints/${id}`);
      loadEndpoints();
      message.success('Endpoint deleted');
    } catch (e) {
      message.error('Failed to delete endpoint');
    }
  };

  const columns = [
    { title: 'Name', dataIndex: 'name', key: 'name' },
    { title: 'Model', dataIndex: 'model_id', key: 'model' },
    { title: 'Version', dataIndex: 'version', key: 'version' },
    { 
      title: 'Status', 
      dataIndex: 'status', 
      key: 'status',
      render: (status: string) => (
        <Badge status={status === 'running' ? 'success' : status === 'stopped' ? 'default' : 'processing'} text={status} />
      )
    },
    { title: 'Replicas', dataIndex: 'replicas', key: 'replicas' },
    {
      title: 'Actions',
      key: 'actions',
      render: (_: any, record: Endpoint) => (
        <Space>
          {record.status === 'running' ? (
            <Button type="link" icon={<StopOutlined />} onClick={() => stopEndpoint(record.endpoint_id)}>Stop</Button>
          ) : (
            <Button type="link" icon={<PlayCircleOutlined />} onClick={() => startEndpoint(record.endpoint_id)}>Start</Button>
          )}
          <Button type="link" danger icon={<DeleteOutlined />} onClick={() => deleteEndpoint(record.endpoint_id)}>Delete</Button>
        </Space>
      )
    },
  ];

  return (
    <div style={{ padding: 24 }}>
      <Title level={2}><ControlOutlined /> Model Serving v2</Title>

      <Card>
        <div style={{ marginBottom: 16 }}>
          <Button type="primary" icon={<PlusOutlined />} onClick={() => setCreateModalVisible(true)}>
            Create Endpoint
          </Button>
        </div>
        <Table columns={columns} dataSource={endpoints} rowKey="endpoint_id" loading={loading} />
      </Card>

      <Modal title="Create Endpoint" open={createModalVisible} onCancel={() => setCreateModalVisible(false)} footer={null}>
        <Form form={form} onFinish={createEndpoint} layout="vertical">
          <Form.Item name="name" label="Name" rules={[{ required: true }]}><Input /></Form.Item>
          <Form.Item name="model_id" label="Model ID" rules={[{ required: true }]}><Input placeholder="e.g., gpt-4" /></Form.Item>
          <Form.Item name="model_version" label="Version" rules={[{ required: true }]}><Input placeholder="e.g., v1" /></Form.Item>
          <Form.Item name="replicas" label="Replicas" initialValue={1}><InputNumber min={1} max={10} style={{ width: '100%' }} /></Form.Item>
          <Form.Item><Button type="primary" htmlType="submit">Create</Button></Form.Item>
        </Form>
      </Modal>
    </div>
  );
};
