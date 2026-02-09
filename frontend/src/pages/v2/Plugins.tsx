import React, { useState, useEffect } from 'react';
import { Card, Table, Button, Modal, Form, Input, Select, Space, Tag, Tabs, Statistic, Typography, message, Row, Col, Rate, List } from 'antd';
import { AppstoreOutlined, ShopOutlined, StarOutlined, DownloadOutlined, PlusOutlined } from '@ant-design/icons';
import { api } from '../../api/client';

const { Text, Title, Paragraph } = Typography;

interface Plugin {
  plugin_id: string;
  name: string;
  description: string;
  category: string;
  version: string;
  status: string;
  author: string;
  rating: number;
  downloads: number;
}

interface Installation {
  installation_id: string;
  plugin_id: string;
  version: string;
  status: string;
  enabled: boolean;
}

export const PluginsPage: React.FC = () => {
  const [plugins, setPlugins] = useState<Plugin[]>([]);
  const [installed, setInstalled] = useState<Installation[]>([]);
  const [loading, setLoading] = useState(false);
  const [installModal, setInstallModal] = useState(false);
  const [selectedPlugin, setSelectedPlugin] = useState<Plugin | null>(null);
  const [form] = Form.useForm();

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    setLoading(true);
    try {
      const [pluginsRes, installedRes] = await Promise.all([
        api.get('/plugins/plugins'),
        api.get('/plugins/installed')
      ]);
      setPlugins(pluginsRes.data.plugins || []);
      setInstalled(installedRes.data.plugins || []);
    } catch (e) {
      console.error('Failed to load data:', e);
    } finally {
      setLoading(false);
    }
  };

  const installPlugin = async (values: any) => {
    if (!selectedPlugin) return;
    try {
      await api.post(`/plugins/${selectedPlugin.plugin_id}/install`, { version: values.version });
      setInstallModal(false);
      form.resetFields();
      loadData();
      message.success('Plugin installed');
    } catch (e) {
      message.error('Failed to install plugin');
    }
  };

  const uninstallPlugin = async (id: string) => {
    try {
      await api.post(`/plugins/${id}/uninstall`);
      loadData();
      message.success('Plugin uninstalled');
    } catch (e) {
      message.error('Failed to uninstall');
    }
  };

  const togglePlugin = async (id: string, enabled: boolean) => {
    try {
      await api.put(`/plugins/${id}/enable`, { enabled });
      loadData();
      message.success(`Plugin ${enabled ? 'enabled' : 'disabled'}`);
    } catch (e) {
      message.error('Failed to update plugin');
    }
  };

  const submitForReview = async (values: any) => {
    try {
      await api.post('/plugins', values);
      message.success('Plugin submitted for review');
    } catch (e) {
      message.error('Failed to submit plugin');
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'approved': return 'success';
      case 'pending_review': return 'processing';
      case 'draft': return 'default';
      case 'rejected': return 'error';
      default: return 'default';
    }
  };

  const pluginColumns = [
    { title: 'Name', dataIndex: 'name', key: 'name' },
    { title: 'Category', dataIndex: 'category', key: 'category', render: (c: string) => <Tag>{c}</Tag> },
    { title: 'Version', dataIndex: 'version', key: 'version' },
    { title: 'Author', dataIndex: 'author', key: 'author' },
    { title: 'Rating', dataIndex: 'rating', key: 'rating', render: (r: number) => <Rate disabled value={Math.round(r)} /> },
    { title: 'Downloads', dataIndex: 'downloads', key: 'downloads' },
    { 
      title: 'Status', 
      dataIndex: 'status', 
      key: 'status',
      render: (s: string) => <Tag color={getStatusColor(s)}>{s.replace('_', ' ')}</Tag>
    },
    {
      title: 'Actions',
      key: 'actions',
      render: (_: any, record: Plugin) => (
        <Space>
          {record.status === 'approved' && (
            <Button type="link" onClick={() => {
              setSelectedPlugin(record);
              setInstallModal(true);
            }}>Install</Button>
          )}
        </Space>
      )
    },
  ];

  const installedColumns = [
    { title: 'Plugin', dataIndex: 'plugin_id', key: 'plugin' },
    { title: 'Version', dataIndex: 'version', key: 'version' },
    { 
      title: 'Status', 
      dataIndex: 'status', 
      key: 'status',
      render: (s: string) => <Tag color={s === 'installed' ? 'green' : 'orange'}>{s}</Tag>
    },
    {
      title: 'Enabled',
      dataIndex: 'enabled',
      key: 'enabled',
      render: (enabled: boolean, record: Installation) => (
        <Switch checked={enabled} onChange={(checked) => togglePlugin(record.plugin_id, checked)} />
      )
    },
    {
      title: 'Actions',
      key: 'actions',
      render: (_: any, record: Installation) => (
        <Button type="link" danger onClick={() => uninstallPlugin(record.plugin_id)}>Uninstall</Button>
      )
    },
  ];

  return (
    <div style={{ padding: 24 }}>
      <Title level={2}><AppstoreOutlined /> Plugin Marketplace</Title>

      <Row gutter={16} style={{ marginBottom: 24 }}>
        <Col span={8}>
          <Card><Statistic title="Available Plugins" value={plugins.length} prefix={<ShopOutlined />} /></Card>
        </Col>
        <Col span={8}>
          <Card><Statistic title="Installed" value={installed.length} prefix={<DownloadOutlined />} /></Card>
        </Col>
        <Col span={8}>
          <Card><Statistic title="Avg Rating" value={plugins.length > 0 ? (plugins.reduce((a, p) => a + p.rating, 0) / plugins.length).toFixed(1) : 0} prefix={<StarOutlined />} /></Card>
        </Col>
      </Row>

      <Tabs defaultActiveKey="marketplace">
        <TabPane tab="Marketplace" key="marketplace">
          <Card>
            <Table columns={pluginColumns} dataSource={plugins} rowKey="plugin_id" loading={loading} />
          </Card>
        </TabPane>
        
        <TabPane tab="Installed" key="installed">
          <Card>
            <Table columns={installedColumns} dataSource={installed} rowKey="installation_id" loading={loading} />
          </Card>
        </TabPane>
        
        <TabPane tab="My Plugins" key="my_plugins">
          <Card title="Submit New Plugin">
            <Form onFinish={submitForReview} layout="vertical">
              <Form.Item name="name" label="Plugin Name" rules={[{ required: true }]}><Input /></Form.Item>
              <Form.Item name="description" label="Description" rules={[{ required: true }]}><Input.TextArea /></Form.Item>
              <Form.Item name="category" label="Category" rules={[{ required: true }]}>
                <Select>
                  <Select.Option value="integration">Integration</Select.Option>
                  <Select.Option value="visualization">Visualization</Select.Option>
                  <Select.Option value="notification">Notification</Select.Option>
                  <Select.Option value="monitoring">Monitoring</Select.Option>
                  <Select.Option value="security">Security</Select.Option>
                </Select>
              </Form.Item>
              <Form.Item name="author" label="Author" rules={[{ required: true }]}><Input /></Form.Item>
              <Form.Item name="version" label="Version"><Input placeholder="1.0.0" /></Form.Item>
              <Form.Item><Button type="primary" htmlType="submit" icon={<PlusOutlined />}>Submit for Review</Button></Form.Item>
            </Form>
          </Card>
        </TabPane>
      </Tabs>

      <Modal title={`Install: ${selectedPlugin?.name}`} open={installModal} onCancel={() => setInstallModal(false)} footer={null}>
        <Form form={form} onFinish={installPlugin} layout="vertical">
          <Paragraph>{selectedPlugin?.description}</Paragraph>
          <Form.Item name="version" label="Version" initialValue={selectedPlugin?.version}>
            <Input />
          </Form.Item>
          <Form.Item><Button type="primary" htmlType="submit">Install</Button></Form.Item>
        </Form>
      </Modal>
    </div>
  );
};
