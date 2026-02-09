import React, { useState, useEffect } from 'react';
import { Card, Table, Button, Modal, Form, Input, Select, Space, Tag, Tabs, Statistic, Typography, message, Row, Col, List, Switch } from 'antd';
import { CloudOutlined, AmazonOutlined, GoogleOutlined, WindowsOutlined, ApiOutlined } from '@ant-design/icons';
import { api } from '../../api/client';

const { Text, Title, Paragraph } = Typography;

interface Credential {
  credential_id: string;
  provider: string;
  name: string;
}

interface Registry {
  registry_id: string;
  provider: string;
  name: string;
  url: string;
}

export const CloudSettingsPage: React.FC = () => {
  const [credentials, setCredentials] = useState<Credential[]>([]);
  const [registries, setRegistries] = useState<Registry[]>([]);
  const [loading, setLoading] = useState(false);
  const [createCredModal, setCreateCredModal] = useState(false);
  const [form] = Form.useForm();

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    setLoading(true);
    try {
      const [credsRes, registriesRes] = await Promise.all([
        api.get('/cloud/credentials'),
        api.get('/cloud/registries')
      ]);
      setCredentials(credsRes.data.credentials || []);
      setRegistries(registriesRes.data.registries || []);
    } catch (e) {
      console.error('Failed to load data:', e);
    } finally {
      setLoading(false);
    }
  };

  const createCredential = async (values: any) => {
    try {
      await api.post('/cloud/credentials', values);
      setCreateCredModal(false);
      form.resetFields();
      loadData();
      message.success('Credential created');
    } catch (e) {
      message.error('Failed to create credential');
    }
  };

  const validateCredential = async (id: string) => {
    try {
      const res = await api.post(`/cloud/credentials/${id}/validate`);
      message.success(res.data.message);
    } catch (e) {
      message.error('Validation failed');
    }
  };

  const deleteCredential = async (id: string) => {
    try {
      await api.delete(`/cloud/credentials/${id}`);
      loadData();
      message.success('Credential deleted');
    } catch (e) {
      message.error('Failed to delete');
    }
  };

  const credColumns = [
    { title: 'Provider', dataIndex: 'provider', key: 'provider', render: (p: string) => {
      const icon = p === 'aws' ? <AmazonOutlined style={{ color: '#FF9900' }} /> : 
                   p === 'gcp' ? <GoogleOutlined style={{ color: '#4285F4' }} /> : 
                   <WindowsOutlined style={{ color: '#0078D4' }} />;
      return <Space>{icon} {p.toUpperCase()}</Space>;
    }},
    { title: 'Name', dataIndex: 'name', key: 'name' },
    { 
      title: 'Actions',
      key: 'actions',
      render: (_: any, record: Credential) => (
        <Space>
          <Button type="link" size="small" onClick={() => validateCredential(record.credential_id)}>Validate</Button>
          <Button type="link" danger size="small" onClick={() => deleteCredential(record.credential_id)}>Delete</Button>
        </Space>
      )
    },
  ];

  const registryColumns = [
    { title: 'Provider', dataIndex: 'provider', key: 'provider', render: (p: string) => <Tag>{p.toUpperCase()}</Tag> },
    { title: 'Name', dataIndex: 'name', key: 'name' },
    { title: 'URL', dataIndex: 'url', key: 'url', render: (u: string) => <Text copyable>{u}</Text> },
  ];

  return (
    <div style={{ padding: 24 }}>
      <Title level={2}><CloudOutlined /> Cloud Integration</Title>

      <Row gutter={16} style={{ marginBottom: 24 }}>
        <Col span={8}>
          <Card><Statistic title="Credentials" value={credentials.length} prefix={<ApiOutlined />} /></Card>
        </Col>
        <Col span={8}>
          <Card><Statistic title="Registries" value={registries.length} prefix={<CloudOutlined />} /></Card>
        </Col>
        <Col span={8}>
          <Card>
            <Statistic title="Providers" value={3} />
            <Space size="large" style={{ marginTop: 8 }}>
              <AmazonOutlined style={{ fontSize: 24, color: '#FF9900' }} />
              <GoogleOutlined style={{ fontSize: 24, color: '#4285F4' }} />
              <WindowsOutlined style={{ fontSize: 24, color: '#0078D4' }} />
            </Space>
          </Card>
        </Col>
      </Row>

      <Tabs defaultActiveKey="credentials">
        <TabPane tab="Credentials" key="credentials">
          <Card>
            <div style={{ marginBottom: 16 }}>
              <Button type="primary" onClick={() => setCreateCredModal(true)}>Add Credential</Button>
            </div>
            <Table columns={credColumns} dataSource={credentials} rowKey="credential_id" loading={loading} />
          </Card>
        </TabPane>
        
        <TabPane tab="Container Registries" key="registries">
          <Card>
            <Table columns={registryColumns} dataSource={registries} rowKey="registry_id" loading={loading} />
          </Card>
        </TabPane>
        
        <TabPane tab="Resources" key="resources">
          <Card title="Supported Resources">
            <List
              dataSource={[
                { type: 'AWS S3', features: ['list_buckets', 'sync', 'upload'] },
                { type: 'GCP GCS', features: ['list_buckets', 'sync'] },
                { type: 'Azure Blob', features: ['list_containers', 'sync'] },
                { type: 'ECR', features: ['push', 'pull'] },
                { type: 'GCR', features: ['push', 'pull'] },
                { type: 'ACR', features: ['push', 'pull'] },
              ]}
              renderItem={(item: any) => (
                <List.Item>
                  <List.Item.Meta title={item.type} description={item.features.join(', ')} />
                </List.Item>
              )}
            />
          </Card>
        </TabPane>
      </Tabs>

      <Modal title="Add Cloud Credential" open={createCredModal} onCancel={() => setCreateCredModal(false)} footer={null}>
        <Form form={form} onFinish={createCredential} layout="vertical">
          <Form.Item name="provider" label="Provider" rules={[{ required: true }]}>
            <Select>
              <Select.Option value="aws">AWS</Select.Option>
              <Select.Option value="gcp">GCP</Select.Option>
              <Select.Option value="azure">Azure</Select.Option>
            </Select>
          </Form.Item>
          <Form.Item name="name" label="Name" rules={[{ required: true }]}><Input /></Form.Item>
          <Form.Item name="config" label="Config (JSON)"><Input.TextArea rows={3} placeholder='{"region": "us-east-1"}' /></Form.Item>
          <Form.Item><Button type="primary" htmlType="submit">Create</Button></Form.Item>
        </Form>
      </Modal>
    </div>
  );
};
