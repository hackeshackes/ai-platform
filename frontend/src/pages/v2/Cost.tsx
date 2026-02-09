import React, { useState, useEffect } from 'react';
import { Card, Table, Button, Modal, Form, Input, InputNumber, Select, Space, Tag, Tabs, Statistic, Progress, DatePicker, Row, Col, Typography, List } from 'antd';
import { DollarOutlined, LineChartOutlined, WalletOutlined, BulbOutlined, WarningOutlined } from '@ant-design/icons';
import { api } from '../../api/client';
import { Line } from 'react-chartjs-2';

const { RangePicker } = DatePicker;
const { Text, Title, Paragraph } = Typography;
const { TabPane } = Tabs;

interface Budget {
  budget_id: string;
  name: string;
  limit: number;
  used: number;
  remaining: number;
  usage_percent: number;
  type: string;
  period: string;
  enabled: boolean;
}

interface CostSummary {
  total_cost: number;
  by_type: Record<string, number>;
  by_provider: Record<string, number>;
  entry_count: number;
}

interface TokenSummary {
  total_cost: number;
  total_tokens: number;
  request_count: number;
  by_model: Record<string, number>;
  by_provider: Record<string, number>;
}

interface Forecast {
  current_spend: number;
  predicted_daily: number;
  predicted_weekly: number;
  predicted_monthly: number;
  trend: string;
  confidence: number;
}

export const CostPage: React.FC = () => {
  const [budgets, setBudgets] = useState<Budget[]>([]);
  const [costSummary, setCostSummary] = useState<CostSummary | null>(null);
  const [tokenSummary, setTokenSummary] = useState<TokenSummary | null>(null);
  const [forecast, setForecast] = useState<Forecast | null>(null);
  const [suggestions, setSuggestions] = useState<Record<string, any>[]>([]);
  const [loading, setLoading] = useState(false);
  const [createModalVisible, setCreateModalVisible] = useState(false);
  const [form] = Form.useForm();

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    setLoading(true);
    try {
      const [budgetsRes, costRes, tokenRes, forecastRes, suggestionsRes] = await Promise.all([
        api.get('/cost/budgets'),
        api.get('/cost/summary'),
        api.get('/cost/tokens'),
        api.get('/cost/forecast'),
        api.get('/cost/suggestions')
      ]);
      setBudgets(budgetsRes.data.budgets || []);
      setCostSummary(costRes.data || null);
      setTokenSummary(tokenRes.data || null);
      setForecast(forecastRes.data || null);
      setSuggestions(suggestionsRes.data.suggestions || []);
    } catch (e) {
      console.error('Failed to load data:', e);
    } finally {
      setLoading(false);
    }
  };

  const createBudget = async (values: any) => {
    try {
      await api.post('/cost/budgets', values);
      setCreateModalVisible(false);
      form.resetFields();
      loadData();
    } catch (e) {
      console.error('Failed to create budget:', e);
    }
  };

  const trackToken = async () => {
    try {
      await api.post('/cost/track/token', {
        provider: 'openai',
        model: 'gpt-4o',
        prompt_tokens: 100,
        completion_tokens: 200,
        latency_ms: 1500
      });
      loadData();
    } catch (e) {
      console.error('Failed to track token:', e);
    }
  };

  const budgetColumns = [
    { title: 'Name', dataIndex: 'name', key: 'name' },
    { title: 'Limit', dataIndex: 'limit', key: 'limit', render: (v: number) => `$${v.toFixed(2)}` },
    { 
      title: 'Usage', 
      key: 'usage',
      render: (_: any, record: Budget) => (
        <Progress 
          percent={record.usage_percent} 
          status={record.usage_percent > 90 ? 'exception' : record.usage_percent > 70 ? 'active' : 'success'}
        />
      )
    },
    { title: 'Remaining', dataIndex: 'remaining', key: 'remaining', render: (v: number) => `$${v.toFixed(2)}` },
    { title: 'Period', dataIndex: 'period', key: 'period' },
    { 
      title: 'Status', 
      dataIndex: 'enabled', 
      key: 'enabled',
      render: (enabled: boolean) => (
        <Tag color={enabled ? 'green' : 'red'}>{enabled ? 'Active' : 'Inactive'}</Tag>
      )
    },
  ];

  const getTrendIcon = (trend: string) => {
    if (trend === 'increasing') return <WarningOutlined style={{ color: '#ff4d4f' }} />;
    if (trend === 'decreasing') return <BulbOutlined style={{ color: '#52c41a' }} />;
    return <LineChartOutlined style={{ color: '#1890ff' }} />;
  };

  return (
    <div style={{ padding: 24 }}>
      <Title level={2}>
        <DollarOutlined /> Cost Intelligence
      </Title>

      {forecast && (
        <Row gutter={16} style={{ marginBottom: 24 }}>
          <Col span={6}>
            <Card>
              <Statistic
                title="Current Spend"
                value={forecast.current_spend}
                prefix="$"
                precision={2}
              />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic
                title="Predicted Monthly"
                value={forecast.predicted_monthly}
                prefix="$"
                precision={2}
              />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic
                title="Trend"
                value={forecast.trend}
                prefix={getTrendIcon(forecast.trend)}
              />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic
                title="Confidence"
                value={forecast.confidence}
                suffix="%"
                valueStyle={{ color: forecast.confidence > 0.7 ? '#52c41a' : '#faad14' }}
              />
            </Card>
          </Col>
        </Row>
      )}

      <Tabs defaultActiveKey="overview">
        <TabPane tab="Overview" key="overview">
          <Row gutter={16}>
            <Col span={12}>
              <Card title="Cost by Provider">
                {costSummary && (
                  <div>
                    {Object.entries(costSummary.by_provider || {}).map(([provider, cost]) => (
                      <div key={provider} style={{ marginBottom: 8 }}>
                        <Text>{provider.toUpperCase()}</Text>
                        <Progress percent={Math.min(100, (cost as number) / costSummary.total_cost * 100)} 
                          format={() => `$${(cost as number).toFixed(2)}`} />
                      </div>
                    ))}
                  </div>
                )}
              </Card>
            </Col>
            <Col span={12}>
              <Card title="Token Usage">
                {tokenSummary && (
                  <div>
                    <Statistic title="Total Tokens" value={tokenSummary.total_tokens} />
                    <Statistic title="Total Cost" value={tokenSummary.total_cost} prefix="$" />
                    <Statistic title="API Calls" value={tokenSummary.request_count} />
                  </div>
                )}
              </Card>
            </Col>
          </Row>
        </TabPane>
        
        <TabPane tab="Budgets" key="budgets">
          <Card>
            <div style={{ marginBottom: 16 }}>
              <Button
                type="primary"
                icon={<WalletOutlined />}
                onClick={() => setCreateModalVisible(true)}
              >
                Create Budget
              </Button>
              <Button style={{ marginLeft: 8 }} onClick={trackToken}>
                Track Token
              </Button>
            </div>
            <Table
              columns={budgetColumns}
              dataSource={budgets}
              rowKey="budget_id"
              loading={loading}
            />
          </Card>
        </TabPane>
        
        <TabPane tab="Models" key="models">
          <Card title="Token Usage by Model">
            {tokenSummary && (
              <Table
                dataSource={Object.entries(tokenSummary.by_model || {}).map(([model, tokens]) => ({
                  model,
                  tokens
                }))}
                columns={[
                  { title: 'Model', dataIndex: 'model', key: 'model' },
                  { title: 'Tokens', dataIndex: 'tokens', key: 'tokens' },
                ]}
                rowKey="model"
              />
            )}
          </Card>
        </TabPane>
        
        <TabPane tab="Suggestions" key="suggestions">
          <Card title="Cost Optimization Suggestions">
            {suggestions.length > 0 ? (
              <List
                dataSource={suggestions}
                renderItem={(item: any) => (
                  <List.Item>
                    <List.Item.Meta
                      title={
                        <Space>
                          <Tag color={item.priority === 'high' ? 'red' : item.priority === 'medium' ? 'orange' : 'green'}>
                            {item.priority.toUpperCase()}
                          </Tag>
                          {item.title}
                        </Space>
                      }
                      description={item.description}
                    />
                    <Text type="secondary">Savings: {item.savings_estimate}</Text>
                  </List.Item>
                )}
              />
            ) : (
              <Paragraph type="secondary">No suggestions available yet. Start tracking costs to get insights.</Paragraph>
            )}
          </Card>
        </TabPane>
      </Tabs>

      {/* Create Budget Modal */}
      <Modal
        title="Create Budget"
        open={createModalVisible}
        onCancel={() => setCreateModalVisible(false)}
        footer={null}
      >
        <Form form={form} onFinish={createBudget} layout="vertical">
          <Form.Item name="name" label="Budget Name" rules={[{ required: true }]}>
            <Input />
          </Form.Item>
          <Form.Item name="total_limit" label="Limit ($)" rules={[{ required: true }]}>
            <InputNumber min={0} step={10} style={{ width: '100%' }} />
          </Form.Item>
          <Form.Item name="cost_type" label="Type" rules={[{ required: true }]}>
            <Select>
              <Select.Option value="api_call">API Calls</Select.Option>
              <Select.Option value="token">Tokens</Select.Option>
              <Select.Option value="storage">Storage</Select.Option>
            </Select>
          </Form.Item>
          <Form.Item name="period" label="Period" rules={[{ required: true }]}>
            <Select>
              <Select.Option value="daily">Daily</Select.Option>
              <Select.Option value="weekly">Weekly</Select.Option>
              <Select.Option value="monthly">Monthly</Select.Option>
              <Select.Option value="yearly">Yearly</Select.Option>
            </Select>
          </Form.Item>
          <Form.Item name="alert_threshold" label="Alert Threshold (%)">
            <InputNumber min={0.1} max={1} step={0.1} defaultValue={0.8} style={{ width: '100%' }} />
          </Form.Item>
          <Form.Item>
            <Button type="primary" htmlType="submit">Create Budget</Button>
          </Form.Item>
        </Form>
      </Modal>
    </div>
  );
};
