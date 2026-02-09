import React, { useState, useEffect } from 'react';
import { Card, Table, Button, Modal, Form, Input, Select, Space, Tag, Tabs, Statistic, Switch, Alert, List } from 'antd';
import { PlusOutlined, SafetyCertificateOutlined, CheckCircleOutlined, CloseCircleOutlined, WarningOutlined } from '@ant-design/icons';
import { api } from '../../api/client';

const { Text, Title, Paragraph } = Typography;

interface GuardrailRule {
  rule_id: string;
  name: string;
  description: string;
  type: string;
  severity: string;
  action: string;
  enabled: boolean;
  keywords_count: number;
}

interface Violation {
  result_id: string;
  rule_name: string;
  type: string;
  passed: boolean;
  action: string;
  message: string;
  checked_at: string;
}

export const GuardrailsPage: React.FC = () => {
  const [rules, setRules] = useState<GuardrailRule[]>([]);
  const [violations, setViolations] = useState<Violation[]>([]);
  const [stats, setStats] = useState<Record<string, any>>({});
  const [loading, setLoading] = useState(false);
  const [checkModalVisible, setCheckModalVisible] = useState(false);
  const [checkResult, setCheckResult] = useState<Record<string, any> | null>(null);
  const [checkText, setCheckText] = useState('');
  const [checkType, setCheckType] = useState('input');
  const [form] = Form.useForm();

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    setLoading(true);
    try {
      const [rulesRes, violationsRes, statsRes] = await Promise.all([
        api.get('/guardrails/rules'),
        api.get('/guardrails/violations'),
        api.get('/guardrails/stats')
      ]);
      setRules(rulesRes.data.rules || []);
      setViolations(violationsRes.data.violations || []);
      setStats(statsRes.data || {});
    } catch (e) {
      console.error('Failed to load data:', e);
    } finally {
      setLoading(false);
    }
  };

  const toggleRule = async (ruleId: string, enabled: boolean) => {
    try {
      await api.put(`/guardrails/rules/${ruleId}`, { enabled });
      loadData();
    } catch (e) {
      console.error('Failed to toggle rule:', e);
    }
  };

  const runCheck = async () => {
    try {
      const endpoint = checkType === 'input' ? '/guardrails/check/input' : '/guardrails/check/output';
      const res = await api.post(endpoint, { text: checkText });
      setCheckResult(res.data);
    } catch (e) {
      console.error('Failed to run check:', e);
    }
  };

  const ruleColumns = [
    { title: 'Name', dataIndex: 'name', key: 'name' },
    { title: 'Type', dataIndex: 'type', key: 'type', render: (type: string) => (
      <Tag color={type === 'input' ? 'blue' : type === 'output' ? 'green' : 'orange'}>{type}</Tag>
    )},
    { title: 'Severity', dataIndex: 'severity', key: 'severity', render: (sev: string) => {
      const color = sev === 'critical' ? 'red' : sev === 'high' ? 'orange' : sev === 'medium' ? 'yellow' : 'green';
      return <Tag color={color}>{sev}</Tag>;
    }},
    { title: 'Action', dataIndex: 'action', key: 'action' },
    { title: 'Keywords', dataIndex: 'keywords_count', key: 'keywords' },
    {
      title: 'Enabled',
      dataIndex: 'enabled',
      key: 'enabled',
      render: (enabled: boolean, record: GuardrailRule) => (
        <Switch
          checked={enabled}
          onChange={(checked) => toggleRule(record.rule_id, checked)}
        />
      )
    },
  ];

  const violationColumns = [
    { title: 'Rule', dataIndex: 'rule_name', key: 'rule' },
    { title: 'Type', dataIndex: 'type', key: 'type' },
    { 
      title: 'Result', 
      dataIndex: 'passed', 
      key: 'passed', 
      render: (passed: boolean) => (
        passed ? 
          <CheckCircleOutlined style={{ color: '#52c41a' }} /> : 
          <CloseCircleOutlined style={{ color: '#ff4d4f' }} />
      )
    },
    { title: 'Message', dataIndex: 'message', key: 'message', ellipsis: true },
    { title: 'Time', dataIndex: 'checked_at', key: 'time', render: (t: string) => new Date(t).toLocaleString() },
  ];

  return (
    <div style={{ padding: 24 }}>
      <Title level={2}>
        <ShieldOutlined /> LLM Guardrails
      </Title>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16, marginBottom: 24 }}>
        <Card>
          <Statistic title="Total Rules" value={stats.total_rules || 0} />
        </Card>
        <Card>
          <Statistic title="Enabled" value={stats.enabled_rules || 0} />
        </Card>
        <Card>
          <Statistic title="Total Violations" value={stats.total_violations || 0} />
        </Card>
        <Card>
          <Statistic title="Configs" value={stats.total_configs || 0} />
        </Card>
      </div>

      <Tabs defaultActiveKey="rules">
        <TabPane tab="Rules" key="rules">
          <Card>
            <Table
              columns={ruleColumns}
              dataSource={rules}
              rowKey="rule_id"
              loading={loading}
            />
          </Card>
        </TabPane>
        
        <TabPane tab="Violations" key="violations">
          <Card>
            <Alert
              message="Recent Violations"
              description="All detected guardrail violations are logged here for review."
              type="info"
              showIcon
              style={{ marginBottom: 16 }}
            />
            <Table
              columns={violationColumns}
              dataSource={violations}
              rowKey="result_id"
              loading={loading}
              pagination={{ pageSize: 10 }}
            />
          </Card>
        </TabPane>
        
        <TabPane tab="Test" key="test">
          <Card>
            <Space direction="vertical" style={{ width: '100%' }} size="large">
              <div>
                <Text strong>Check Type: </Text>
                <Select value={checkType} onChange={setCheckType} style={{ width: 200 }}>
                  <Select.Option value="input">Input Check</Select.Option>
                  <Select.Option value="output">Output Check</Select.Option>
                </Select>
              </div>
              
              <Form layout="vertical">
                <Form.Item label="Text to Check">
                  <TextArea
                    rows={4}
                    value={checkText}
                    onChange={(e) => setCheckText(e.target.value)}
                    placeholder="Enter text to check against guardrails..."
                  />
                </Form.Item>
                <Button type="primary" icon={<ShieldOutlined />} onClick={runCheck}>
                  Run Check
                </Button>
              </Form>

              {checkResult && (
                <Card title="Check Result" style={{ marginTop: 16 }}>
                  {checkResult.passed ? (
                    <Alert
                      message="Passed"
                      description="No violations detected."
                      type="success"
                      showIcon
                    />
                  ) : (
                    <Alert
                      message="Violations Detected"
                      description={`${checkResult.violations_count} violation(s) found.`}
                      type="error"
                      showIcon
                    />
                  )}
                  
                  {checkResult.results && checkResult.results.length > 0 && (
                    <List
                      size="small"
                      dataSource={checkResult.results}
                      renderItem={(item: any) => (
                        <List.Item>
                          <List.Item.Meta
                            title={item.rule_name}
                            description={item.message}
                          />
                          <Tag color={item.action === 'block' ? 'red' : 'orange'}>
                            {item.action.toUpperCase()}
                          </Tag>
                        </List.Item>
                      )}
                    />
                  )}
                </Card>
              )}
            </Space>
          </Card>
        </TabPane>
      </Tabs>
    </div>
  );
};
