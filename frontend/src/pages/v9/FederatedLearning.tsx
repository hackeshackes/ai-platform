import React, { useState } from 'react';
import { Card, Row, Col, Statistic, Table, Tag, Button, Progress, Steps } from 'antd';
import { Share2, Users, Lock, Activity } from 'lucide-react';

export const FederatedLearning: React.FC = () => {
  const [sessions] = useState([
    { id: 'fl-001', name: '销售预测模型', model: 'regression', rounds: 10, participants: 5, status: 'training', progress: 60 },
    { id: 'fl-002', name: '客户分类', model: 'classifier', rounds: 5, participants: 3, status: 'completed', progress: 100 },
  ]);

  const columns = [
    { title: '会话ID', dataIndex: 'id', key: 'id', render: (text: string) => <code>{text}</code> },
    { title: '名称', dataIndex: 'name', key: 'name' },
    { title: '模型', dataIndex: 'model', key: 'model', render: (v: string) => <Tag>{v}</Tag> },
    { title: '参与方', dataIndex: 'participants', key: 'participants' },
    { title: '状态', dataIndex: 'status', key: 'status', render: (v: string) => <Tag color={v === 'completed' ? 'green' : 'blue'}>{v}</Tag> },
    { title: '进度', dataIndex: 'progress', key: 'progress', render: (v: number) => <Progress percent={v} size="small" /> },
  ];

  return (
    <div>
      <h1><Share2 size={32} /> 联邦学习平台</h1>
      <Row gutter={[16, 16]} style={{ marginTop: 16 }}>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic title="活跃会话" value={3} prefix={<Activity />} />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic title="总参与方" value={12} prefix={<Users />} />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic title="隐私保护" value="ε=1.0" prefix={<Lock />} />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic title="聚合算法" value={4} suffix="种" />
          </Card>
        </Col>
      </Row>
      <Card title="联邦训练会话" extra={<Button type="primary">创建会话</Button>} style={{ marginTop: 16 }}>
        <Table dataSource={sessions} columns={columns} rowKey="id" />
      </Card>
      <Row gutter={[16, 16]} style={{ marginTop: 16 }}>
        <Col span={12}>
          <Card title="训练流程">
            <Steps current={2} size="small">
              <Steps.Step title="创建会话" description="配置参数" />
              <Steps.Step title="加入参与方" description="4个客户端" />
              <Steps.Step title="本地训练" description="数据本地化" />
              <Steps.Step title="模型聚合" description="FedAvg" />
            </Steps>
          </Card>
        </Col>
        <Col span={12}>
          <Card title="隐私保护配置">
            <p>✅ 差分隐私: 高斯噪声 ε=1.0</p>
            <p>✅ 梯度裁剪: L2范数 ≤ 1.0</p>
            <p>⏳ 安全聚合: TLS加密传输</p>
          </Card>
        </Col>
      </Row>
    </div>
  );
};
