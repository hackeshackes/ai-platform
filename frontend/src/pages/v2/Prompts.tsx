import React, { useState, useEffect } from 'react';
import { Card, Table, Button, Modal, Form, Input, Select, InputNumber, Space, Tag, Tabs, Badge, Progress, Timeline, Typography } from 'antd';
import { PlusOutlined, EditOutlined, DeleteOutlined, PlayCircleOutlined, HistoryOutlined, CopyOutlined } from '@ant-design/icons';
import { api } from '../../api/client';

const { TextArea } = Input;
const { Text, Title } = Typography;
const { TabPane } = Tabs;

interface PromptTemplate {
  template_id: string;
  name: string;
  description: string;
  type: string;
  parameters_count: number;
  examples_count: number;
  created_by: string;
}

interface Prompt {
  prompt_id: string;
  name: string;
  description: string;
  type: string;
  status: string;
  version: number;
  tags: string[];
  metrics: Record<string, any>;
}

interface PromptTestResult {
  result_id: string;
  version: number;
  test_input: Record<string, any>;
  test_output: string;
  score?: number;
  latency_ms: number;
}

export const PromptsPage: React.FC = () => {
  const [templates, setTemplates] = useState<PromptTemplate[]>([]);
  const [prompts, setPrompts] = useState<Prompt[]>([]);
  const [loading, setLoading] = useState(false);
  const [createModalVisible, setCreateModalVisible] = useState(false);
  const [testModalVisible, setTestModalVisible] = useState(false);
  const [selectedPrompt, setSelectedPrompt] = useState<Prompt | null>(null);
  const [testResults, setTestResults] = useState<PromptTestResult[]>([]);
  const [form] = Form.useForm();
  const [testForm] = Form.useForm();

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    setLoading(true);
    try {
      const [templatesRes, promptsRes] = await Promise.all([
        api.get('/prompts/templates'),
        api.get('/prompts')
      ]);
      setTemplates(templatesRes.data.templates || []);
      setPrompts(promptsRes.data.prompts || []);
    } catch (e) {
      console.error('Failed to load data:', e);
    } finally {
      setLoading(false);
    }
  };

  const createPrompt = async (values: any) => {
    try {
      await api.post('/prompts', values);
      setCreateModalVisible(false);
      form.resetFields();
      loadData();
    } catch (e) {
      console.error('Failed to create prompt:', e);
    }
  };

  const testPrompt = async (values: any) => {
    if (!selectedPrompt) return;
    
    try {
      const testInput = JSON.parse(values.test_input);
      await api.post(`/prompts/${selectedPrompt.prompt_id}/test`, {
        test_input: testInput
      });
      
      // Load test results
      const resultsRes = await api.get(`/prompts/${selectedPrompt.prompt_id}/results`);
      setTestResults(resultsRes.data.results || []);
      
      testForm.resetFields();
    } catch (e) {
      console.error('Failed to test prompt:', e);
    }
  };

  const templateColumns = [
    { title: 'Name', dataIndex: 'name', key: 'name' },
    { title: 'Type', dataIndex: 'type', key: 'type', render: (type: string) => (
      <Tag color="blue">{type}</Tag>
    )},
    { title: 'Parameters', dataIndex: 'parameters_count', key: 'params' },
    { title: 'Created By', dataIndex: 'created_by', key: 'created_by' },
  ];

  const promptColumns = [
    { title: 'Name', dataIndex: 'name', key: 'name' },
    { title: 'Type', dataIndex: 'type', key: 'type', render: (type: string) => (
      <Tag color="green">{type}</Tag>
    )},
    { title: 'Status', dataIndex: 'status', key: 'status', render: (status: string) => (
      <Badge status={status === 'active' ? 'success' : 'default'} text={status} />
    )},
    { title: 'Version', dataIndex: 'version', key: 'version', render: (v: number) => `v${v}` },
    { title: 'Tags', dataIndex: 'tags', key: 'tags', render: (tags: string[]) => (
      <Space>
        {tags.map(t => <Tag key={t}>{t}</Tag>)}
      </Space>
    )},
    {
      title: 'Actions',
      key: 'actions',
      render: (_: any, record: Prompt) => (
        <Space>
          <Button
            type="link"
            icon={<PlayCircleOutlined />}
            onClick={() => {
              setSelectedPrompt(record);
              setTestModalVisible(true);
            }}
          >
            Test
          </Button>
        </Space>
      )
    },
  ];

  const testResultColumns = [
    { title: 'Input', dataIndex: ['test_input', 'text'], key: 'input' },
    { title: 'Output', dataIndex: 'test_output', key: 'output', ellipsis: true },
    { title: 'Score', dataIndex: 'score', key: 'score', render: (s: number) => s ? `${(s * 100).toFixed(1)}%` : '-' },
    { title: 'Latency', dataIndex: 'latency_ms', key: 'latency', render: (ms: number) => `${ms.toFixed(0)}ms` },
  ];

  return (
    <div style={{ padding: 24 }}>
      <Title level={2}>Prompt Management</Title>
      
      <Tabs defaultActiveKey="prompts">
        <TabPane tab="Prompts" key="prompts">
          <Card>
            <div style={{ marginBottom: 16 }}>
              <Button
                type="primary"
                icon={<PlusOutlined />}
                onClick={() => setCreateModalVisible(true)}
              >
                Create Prompt
              </Button>
            </div>
            
            <Table
              columns={promptColumns}
              dataSource={prompts}
              rowKey="prompt_id"
              loading={loading}
            />
          </Card>
        </TabPane>
        
        <TabPane tab="Templates" key="templates">
          <Card>
            <Table
              columns={templateColumns}
              dataSource={templates}
              rowKey="template_id"
              loading={loading}
            />
          </Card>
        </TabPane>
      </Tabs>

      {/* Create Prompt Modal */}
      <Modal
        title="Create Prompt"
        open={createModalVisible}
        onCancel={() => setCreateModalVisible(false)}
        footer={null}
      >
        <Form form={form} onFinish={createPrompt} layout="vertical">
          <Form.Item name="name" label="Name" rules={[{ required: true }]}>
            <Input />
          </Form.Item>
          <Form.Item name="description" label="Description">
            <Input.TextArea />
          </Form.Item>
          <Form.Item name="template_id" label="Template" rules={[{ required: true }]}>
            <Select>
              {templates.map(t => (
                <Select.Option key={t.template_id} value={t.template_id}>
                  {t.name} ({t.type})
                </Select.Option>
              ))}
            </Select>
          </Form.Item>
          <Form.Item name="prompt_type" label="Type" rules={[{ required: true }]}>
            <Select>
              <Select.Option value="chat">Chat</Select.Option>
              <Select.Option value="completion">Completion</Select.Option>
              <Select.Option value="rag">RAG</Select.Option>
            </Select>
          </Form.Item>
          <Form.Item name="tags" label="Tags">
            <Select mode="tags" placeholder="Add tags" />
          </Form.Item>
          <Form.Item>
            <Button type="primary" htmlType="submit">Create</Button>
          </Form.Item>
        </Form>
      </Modal>

      {/* Test Prompt Modal */}
      <Modal
        title={`Test: ${selectedPrompt?.name}`}
        open={testModalVisible}
        onCancel={() => {
          setTestModalVisible(false);
          setSelectedPrompt(null);
          setTestResults([]);
        }}
        width={900}
        footer={null}
      >
        <Space direction="vertical" style={{ width: '100%' }} size="large">
          <Form form={testForm} onFinish={testPrompt} layout="vertical">
            <Form.Item name="test_input" label="Test Input (JSON)" rules={[{ required: true }]}>
              <TextArea rows={4} placeholder='{"text": "Your test text here"}' />
            </Form.Item>
            <Form.Item>
              <Button type="primary" htmlType="submit" icon={<PlayCircleOutlined />}>
                Run Test
              </Button>
            </Form.Item>
          </Form>
          
          {testResults.length > 0 && (
            <Card title="Test Results">
              <Table
                columns={testResultColumns}
                dataSource={testResults}
                rowKey="result_id"
                pagination={false}
              />
            </Card>
          )}
        </Space>
      </Modal>
    </div>
  );
};
