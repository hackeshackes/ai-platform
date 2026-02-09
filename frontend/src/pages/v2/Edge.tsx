import React, { useState, useEffect } from 'react';
import { Card, Table, Button, Modal, Form, Input, Select, Space, Tag, Tabs, Statistic, Typography, message, Progress } from 'antd';
import { PlusOutlined, CloudUploadOutlined, MobileOutlined, DownloadOutlined, CheckCircleOutlined } from '@ant-design/icons';
import { api } from '../../api/client';

const { Text, Title } = Typography;

interface Deployment {
  deployment_id: string;
  name: string;
  device_type: string;
  status: string;
}

interface ExportConfig {
  config_id: string;
  format: string;
  device: string;
  quantized: boolean;
}

export const EdgePage: React.FC = () => {
  const [deployments, setDeployments] = useState<Deployment[]>([]);
  const [loading, setLoading] = useState(false);
  const [createModalVisible, setCreateModalVisible] = useState(false);
  const [exportModalVisible, setExportModalVisible] = useState(false);
  const [form] = Form.useForm();
  const [exportForm] = Form.useForm();

  useEffect(() => {
    loadDeployments();
  }, []);

  const loadDeployments = async () => {
    setLoading(true);
    try {
      const res = await api.get('/edge/deployments');
      setDeployments(res.data.deployments || []);
    } catch (e) {
      console.error('Failed to load deployments:', e);
    } finally {
      setLoading(false);
    }
  };

  const createDeployment = async (values: any) => {
    try {
      await api.post('/edge/deployments', {
        name: values.name,
        model_id: values.model_id,
        export_config_id: values.export_config_id,
        device_type: values.device_type,
        device_url: values.device_url
      });
      setCreateModalVisible(false);
      form.resetFields();
      loadDeployments();
      message.success('Deployment created');
    } catch (e) {
      message.error('Failed to create deployment');
    }
  };

  const deployToDevice = async (id: string) => {
    try {
      await api.post(`/edge/deployments/${id}/deploy`);
      loadDeployments();
      message.success('Deploying to device...');
    } catch (e) {
      message.error('Failed to deploy');
    }
  };

  const quickExport = async () => {
    try {
      const res = await api.post('/edge/export/quick', {
        model_id: 'resnet50',
        export_format: 'onnx',
        quantize: true
      });
      message.success(`Export complete: ${res.data.output_path}`);
    } catch (e) {
      message.error('Failed to export');
    }
  };

  const checkCompatibility = async (values: any) => {
    try {
      const res = await api.get('/edge/export/compatibility', {
        params: {
          model_id: values.model_id,
          export_format: values.export_format,
          device: values.device
        }
      });
      message.info(`Compatible: ${res.data.compatible}`);
    } catch (e) {
      message.error('Failed to check compatibility');
    }
  };

  const columns = [
    { title: 'Name', dataIndex: 'name', key: 'name' },
    { 
      title: 'Device', 
      dataIndex: 'device_type', 
      key: 'device',
      render: (d: string) => <Tag icon={<MobileOutlined />}>{d}</Tag>
    },
    { 
      title: 'Status', 
      dataIndex: 'status', 
      key: 'status',
      render: (s: string) => (
        <Tag color={s === 'running' ? 'green' : s === 'pending' ? 'orange' : 'default'}>{s}</Tag>
      )
    },
    {
      title: 'Actions',
      key: 'actions',
      render: (_: any, record: Deployment) => (
        <Space>
          {record.status === 'pending' && (
            <Button type="link" icon={<CloudUploadOutlined />} onClick={() => deployToDevice(record.deployment_id)}>
              Deploy
            </Button>
          )}
        </Space>
      )
    },
  ];

  return (
    <div style={{ padding: 24 }}>
      <Title level={2}><MobileOutlined /> Edge Inference</Title>

      <Tabs defaultActiveKey="deployments">
        <TabPane tab="Deployments" key="deployments">
          <Card>
            <div style={{ marginBottom: 16, display: 'flex', gap: 8 }}>
              <Button type="primary" icon={<PlusOutlined />} onClick={() => setCreateModalVisible(true)}>
                Create Deployment
              </Button>
              <Button icon={<DownloadOutlined />} onClick={quickExport}>
                Quick Export (ONNX)
              </Button>
            </div>
            <Table columns={columns} dataSource={deployments} rowKey="deployment_id" loading={loading} />
          </Card>
        </TabPane>
        
        <TabPane tab="Devices" key="devices">
          <Card title="Supported Devices">
            <Table
              dataSource={[
                { device: 'CPU', formats: 'ONNX, TFLite, OpenVINO' },
                { device: 'GPU', formats: 'ONNX, TensorRT' },
                { device: 'Edge TPU', formats: 'TFLite (quantized)' },
                { device: 'Mobile', formats: 'CoreML, TFLite' },
                { device: 'Neural Compute', formats: 'ONNX' },
              ]}
              columns={[
                { title: 'Device', dataIndex: 'device', key: 'device' },
                { title: 'Supported Formats', dataIndex: 'formats', key: 'formats' },
              ]}
              rowKey="device"
              pagination={false}
            />
          </Card>
        </TabPane>
      </Tabs>

      <Modal title="Create Deployment" open={createModalVisible} onCancel={() => setCreateModalVisible(false)} footer={null}>
        <Form form={form} onFinish={createDeployment} layout="vertical">
          <Form.Item name="name" label="Name" rules={[{ required: true }]}><Input /></Form.Item>
          <Form.Item name="model_id" label="Model ID" rules={[{ required: true }]}><Input placeholder="e.g., resnet50" /></Form.Item>
          <Form.Item name="export_config_id" label="Export Config ID"><Input /></Form.Item>
          <Form.Item name="device_type" label="Device Type" rules={[{ required: true }]}>
            <Select>
              <Select.Option value="cpu">CPU</Select.Option>
              <Select.Option value="gpu">GPU</Select.Option>
              <Select.Option value="edge_tpu">Edge TPU</Select.Option>
              <Select.Option value="mobile">Mobile</Select.Option>
            </Select>
          </Form.Item>
          <Form.Item name="device_url" label="Device URL"><Input placeholder="e.g., 192.168.1.100:8080" /></Form.Item>
          <Form.Item><Button type="primary" htmlType="submit">Create</Button></Form.Item>
        </Form>
      </Modal>
    </div>
  );
};
