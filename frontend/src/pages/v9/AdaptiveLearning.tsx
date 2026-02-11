import React, { useState, useEffect } from 'react';
import { Card, Row, Col, Statistic, List, Tag, Progress, Button, Input } from 'antd';
import { Brain, TrendingUp, Activity, Clock } from 'lucide-react';

export const AdaptiveLearning: React.FC = () => {
  const [intents] = useState([
    { type: 'QUERY', confidence: 0.92, count: 156 },
    { type: 'CREATION', confidence: 0.88, count: 89 },
    { type: 'ANALYSIS', confidence: 0.85, count: 67 },
  ]);

  return (
    <div>
      <h1><Brain size={32} /> Agent自适应学习</h1>
      <Row gutter={[16, 16]} style={{ marginTop: 16 }}>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic title="总交互数" value={591} prefix={<Activity />} />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic title="学习成功率" value={85.0} suffix="%" prefix={<TrendingUp />} valueStyle={{ color: '#52c41a' }} />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic title="意图类型" value={5} prefix={<Brain />} />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic title="平均响应" value={0.3} suffix="s" prefix={<Clock />} />
          </Card>
        </Col>
      </Row>
      <Card title="意图识别分析" style={{ marginTop: 16 }}>
        <List
          dataSource={intents}
          renderItem={(item) => (
            <List.Item>
              <Tag color="blue">{item.type}</Tag>
              <span>置信度: {(item.confidence * 100).toFixed(1)}%</span>
              <Progress percent={item.confidence * 100} size="small" style={{ width: 120 }} />
            </List.Item>
          )}
        />
      </Card>
      <Card title="实时学习测试" style={{ marginTop: 16 }}>
        <Input.Search placeholder="输入文本进行意图识别..." enterButton="分析" size="large" />
      </Card>
    </div>
  );
};
