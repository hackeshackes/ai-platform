import React, { useState, useEffect } from 'react';
import { Card, Table, Button, Modal, Form, Input, Select, Space, Tag, Tabs, Statistic, Typography, message, Row, Col } from 'antd';
import { ConsoleSqlOutlined, HistoryOutlined, CodeOutlined, RocketOutlined } from '@ant-design/icons';
import { api } from '../../api/client';

const { Text, Title, Paragraph } = Typography;

interface Command {
  command_id: string;
  name: string;
  category: string;
  description: string;
}

interface ScriptTemplate {
  template_id: string;
  name: string;
  category: string;
}

export const CLISettingsPage: React.FC = () => {
  const [commands, setCommands] = useState<Command[]>([]);
  const [templates, setTemplates] = useState<ScriptTemplate[]>([]);
  const [history, setHistory] = useState<Record<string, any>[]>([]);
  const [loading, setLoading] = useState(false);
  const [form] = Form.useForm();

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    setLoading(true);
    try {
      const [commandsRes, templatesRes, historyRes] = await Promise.all([
        api.get('/cli/commands'),
        api.get('/cli/templates'),
        api.get('/cli/history')
      ]);
      setCommands(commandsRes.data.commands || []);
      setTemplates(templatesRes.data.templates || []);
      setHistory(historyRes.data.history || []);
    } catch (e) {
      console.error('Failed to load data:', e);
    } finally {
      setLoading(false);
    }
  };

  const executeCommand = async (values: any) => {
    try {
      const res = await api.post(`/cli/commands/${values.command_id}/execute`, {
        options: {}
      });
      message.success(`Executed: ${res.data.command}`);
      loadData();
    } catch (e) {
      message.error('Failed to execute command');
    }
  };

  const generateScript = async (values: any) => {
    try {
      const res = await api.post(`/cli/templates/${values.template_id}/generate`, {
        variables: {}
      });
      message.success('Script generated');
    } catch (e) {
      message.error('Failed to generate script');
    }
  };

  const clearHistory = async () => {
    try {
      await api.delete('/cli/history');
      setHistory([]);
      message.success('History cleared');
    } catch (e) {
      message.error('Failed to clear history');
    }
  };

  const commandColumns = [
    { title: 'Command', dataIndex: 'name', key: 'name', render: (n: string) => <Tag color="blue">{n}</Tag> },
    { title: 'Category', dataIndex: 'category', key: 'category' },
    { title: 'Description', dataIndex: 'description', key: 'description', ellipsis: true },
  ];

  const templateColumns = [
    { title: 'Name', dataIndex: 'name', key: 'name' },
    { title: 'Category', dataIndex: 'category', key: 'category' },
  ];

  const historyColumns = [
    { title: 'Command', dataIndex: ['result', 'command'], key: 'command' },
    { title: 'Output', dataIndex: ['result', 'output'], key: 'output', ellipsis: true },
    { title: 'Time', dataIndex: 'timestamp', key: 'time', render: (t: string) => new Date(t).toLocaleString() },
  ];

  return (
    <div style={{ padding: 24 }}>
      <Title level={2}><ConsoleSqlOutlined /> CLI Settings</Title>

      <Row gutter={16} style={{ marginBottom: 24 }}>
        <Col span={8}>
          <Card><Statistic title="Commands" value={commands.length} prefix={<CodeOutlined />} /></Card>
        </Col>
        <Col span={8}>
          <Card><Statistic title="Templates" value={templates.length} prefix={<CodeOutlined />} /></Card>
        </Col>
        <Col span={8}>
          <Card><Statistic title="History" value={history.length} prefix={<HistoryOutlined />} /></Card>
        </Col>
      </Row>

      <Tabs defaultActiveKey="commands">
        <TabPane tab="Commands" key="commands">
          <Card>
            <Table columns={commandColumns} dataSource={commands} rowKey="command_id" loading={loading} />
          </Card>
        </TabPane>
        
        <TabPane tab="Templates" key="templates">
          <Card>
            <Table columns={templateColumns} dataSource={templates} rowKey="template_id" loading={loading} />
          </Card>
        </TabPane>
        
        <TabPane tab="History" key="history">
          <Card>
            <div style={{ marginBottom: 16 }}>
              <Button onClick={clearHistory}>Clear History</Button>
            </div>
            <Table columns={historyColumns} dataSource={history} rowKey="timestamp" loading={loading} pagination={{ pageSize: 10 }} />
          </Card>
        </TabPane>
      </Tabs>

      <Card title="Quick Execute" style={{ marginTop: 24 }}>
        <Form form={form} layout="inline" onFinish={executeCommand}>
          <Form.Item name="command_id" rules={[{ required: true }]}>
            <Select style={{ width: 300 }} placeholder="Select command">
              {commands.map(c => (
                <Select.Option key={c.command_id} value={c.command_id}>
                  {c.name}
                </Select.Option>
              ))}
            </Select>
          </Form.Item>
          <Form.Item><Button type="primary" htmlType="submit" icon={<RocketOutlined />}>Execute</Button></Form.Item>
        </Form>
      </Card>
    </div>
  );
};
