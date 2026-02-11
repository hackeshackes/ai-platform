import React from 'react';
import { Card, Row, Col, Statistic, Table, Tag, Progress, Alert } from 'antd';
import { Brain, TrendingUp, AlertTriangle, CheckCircle, Target } from 'lucide-react';

export const DecisionEngine: React.FC = () => {
  const [decisions] = useState([
    { id: 'd1', type: 'pricing', action: '提高10%', confidence: 0.87, risk: 'LOW', timestamp: '2026-02-10 15:30' },
    { id: 'd2', type: 'investment', action: '保持现状', confidence: 0.82, risk: 'MEDIUM', timestamp: '2026-02-10 14:20' },
  ]);

  const columns = [
    { title: 'ID', dataIndex: 'id', key: 'id', render: (text: string) => <code>{text}</code> },
    { title: '类型', dataIndex: 'type', key: 'type', render: (v: string) => <Tag>{v}</Tag> },
    { title: '决策', dataIndex: 'action', key: 'action' },
    { title: '置信度', dataIndex: 'confidence', key: 'confidence', render: (v: number) => <Progress percent={v * 100} size="small" /> },
    { title: '风险', dataIndex: 'risk', key: 'risk', render: (v: string) => <Tag color={v === 'LOW' ? 'green' : 'orange'}>{v}</Tag> },
    { title: '时间', dataIndex: 'timestamp', key: 'timestamp' },
  ];

  return (
    <div>
      <h1><Brain size={32} /> 自主决策引擎</h1>
      <Row gutter={[16, 16]} style={{ marginTop: 16 }}>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic title="今日决策" value={12} prefix={<Target />} />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic title="平均置信度" value={86.5} suffix="%" prefix={<TrendingUp />} valueStyle={{ color: '#52c41a' }} />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic title="风险预警" value={2} prefix={<AlertTriangle />} valueStyle={{ color: '#faad14' }} />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic title="建议采纳率" value={78} suffix="%" />
          </Card>
        </Col>
      </Row>
      <Card title="决策历史" style={{ marginTop: 16 }}>
        <Table dataSource={decisions} columns={columns} rowKey="id" />
      </Card>
      <Row gutter={[16, 16]} style={{ marginTop: 16 }}>
        <Col span={12}>
          <Card title="实时决策建议">
            <Alert message="市场策略优化" description="基于当前数据分析，建议调整定价策略以提升收益。" type="success" showIcon icon={<CheckCircle />} />
          </Card>
        </Col>
        <Col span={12}>
          <Card title="风险评估">
            <p>财务风险: <Progress percent={30} size="small" /></p>
            <p>市场风险: <Progress percent={25} size="small" /></p>
            <p>运营风险: <Progress percent={15} size="small" /></p>
          </Card>
        </Col>
      </Row>
    </div>
  );
};

import { useState } from 'react';
