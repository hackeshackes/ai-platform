import React, { useState, useEffect } from 'react';
import { Card, Table, Button, Modal, Form, Input, Select, Space, Tag, Tabs, Statistic, Badge, Typography, List, message, Row, Col } from 'antd';
import { PlusOutlined, TeamOutlined, MessageOutlined, BellOutlined, AuditOutlined, CheckCircleOutlined, CloseCircleOutlined } from '@ant-design/icons';
import { api } from '../../api/client';

const { Text, Title } = Typography;

interface Team {
  team_id: string;
  name: string;
  description: string;
  members_count: number;
}

interface Review {
  review_id: string;
  title: string;
  type: string;
  status: string;
  requested_by: string;
}

interface Notification {
  notification_id: string;
  type: string;
  title: string;
  message: string;
  read: boolean;
  created_at: string;
}

export const CollaborationPage: React.FC = () => {
  const [teams, setTeams] = useState<Team[]>([]);
  const [reviews, setReviews] = useState<Review[]>([]);
  const [notifications, setNotifications] = useState<Notification[]>([]);
  const [loading, setLoading] = useState(false);
  const [createTeamModal, setCreateTeamModal] = useState(false);
  const [form] = Form.useForm();

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    setLoading(true);
    try {
      const [teamsRes, reviewsRes, notificationsRes] = await Promise.all([
        api.get('/collaboration/teams'),
        api.get('/collaboration/reviews'),
        api.get('/collaboration/notifications', { params: { user_id: 'user1' } })
      ]);
      setTeams(teamsRes.data.teams || []);
      setReviews(reviewsRes.data.reviews || []);
      setNotifications(notificationsRes.data.notifications || []);
    } catch (e) {
      console.error('Failed to load data:', e);
    } finally {
      setLoading(false);
    }
  };

  const createTeam = async (values: any) => {
    try {
      await api.post('/collaboration/teams', values);
      setCreateTeamModal(false);
      form.resetFields();
      loadData();
      message.success('Team created');
    } catch (e) {
      message.error('Failed to create team');
    }
  };

  const updateReviewStatus = async (id: string, status: string) => {
    try {
      await api.post(`/collaboration/reviews/${id}/status`, { status });
      loadData();
      message.success('Status updated');
    } catch (e) {
      message.error('Failed to update status');
    }
  };

  const markNotificationRead = async (id: string) => {
    try {
      await api.post(`/collaboration/notifications/${id}/read`);
      loadData();
    } catch (e) {
      console.error('Failed to mark as read');
    }
  };

  const teamColumns = [
    { title: 'Name', dataIndex: 'name', key: 'name' },
    { title: 'Description', dataIndex: 'description', key: 'desc', ellipsis: true },
    { title: 'Members', dataIndex: 'members_count', key: 'members' },
  ];

  const reviewColumns = [
    { title: 'Title', dataIndex: 'title', key: 'title' },
    { title: 'Type', dataIndex: 'type', key: 'type', render: (t: string) => <Tag>{t}</Tag> },
    { 
      title: 'Status', 
      dataIndex: 'status', 
      key: 'status',
      render: (s: string) => {
        const color = s === 'approved' ? 'green' : s === 'rejected' ? 'red' : s === 'pending' ? 'orange' : 'default';
        return <Tag color={color}>{s}</Tag>;
      }
    },
    { title: 'Requested By', dataIndex: 'requested_by', key: 'requested_by' },
    {
      title: 'Actions',
      key: 'actions',
      render: (_: any, record: Review) => (
        <Space>
          {record.status === 'pending' && (
            <>
              <Button type="link" icon={<CheckCircleOutlined />} onClick={() => updateReviewStatus(record.review_id, 'approved')}>Approve</Button>
              <Button type="link" danger icon={<CloseCircleOutlined />} onClick={() => updateReviewStatus(record.review_id, 'rejected')}>Reject</Button>
            </>
          )}
        </Space>
      )
    },
  ];

  const getNotificationIcon = (type: string) => {
    switch (type) {
      case 'review_requested': return <MessageOutlined style={{ color: '#1890ff' }} />;
      case 'comment': return <MessageOutlined style={{ color: '#52c41a' }} />;
      default: return <BellOutlined style={{ color: '#faad14' }} />;
    }
  };

  return (
    <div style={{ padding: 24 }}>
      <Title level={2}><TeamOutlined /> Collaboration</Title>

      <Row gutter={16} style={{ marginBottom: 24 }}>
        <Col span={8}>
          <Card><Statistic title="Teams" value={teams.length} prefix={<TeamOutlined />} /></Card>
        </Col>
        <Col span={8}>
          <Card><Statistic title="Pending Reviews" value={reviews.filter(r => r.status === 'pending').length} prefix={<MessageOutlined />} /></Card>
        </Col>
        <Col span={8}>
          <Card><Statistic title="Unread Notifications" value={notifications.filter(n => !n.read).length} prefix={<BellOutlined />} /></Card>
        </Col>
      </Row>

      <Tabs defaultActiveKey="teams">
        <TabPane tab="Teams" key="teams">
          <Card>
            <div style={{ marginBottom: 16 }}>
              <Button type="primary" icon={<PlusOutlined />} onClick={() => setCreateTeamModal(true)}>
                Create Team
              </Button>
            </div>
            <Table columns={teamColumns} dataSource={teams} rowKey="team_id" loading={loading} />
          </Card>
        </TabPane>
        
        <TabPane tab="Reviews" key="reviews">
          <Card>
            <Table columns={reviewColumns} dataSource={reviews} rowKey="review_id" loading={loading} />
          </Card>
        </TabPane>
        
        <TabPane tab="Notifications" key="notifications">
          <Card>
            <List
              dataSource={notifications}
              renderItem={(item: Notification) => (
                <List.Item
                  actions={[
                    !item.read && <Button type="link" size="small" onClick={() => markNotificationRead(item.notification_id)}>Mark Read</Button>
                  ]}
                >
                  <List.Item.Meta
                    avatar={getNotificationIcon(item.type)}
                    title={item.title}
                    description={item.message}
                  />
                  <Text type="secondary">{new Date(item.created_at).toLocaleString()}</Text>
                </List.Item>
              )}
            />
          </Card>
        </TabPane>
      </Tabs>

      <Modal title="Create Team" open={createTeamModal} onCancel={() => setCreateTeamModal(false)} footer={null}>
        <Form form={form} onFinish={createTeam} layout="vertical">
          <Form.Item name="name" label="Team Name" rules={[{ required: true }]}><Input /></Form.Item>
          <Form.Item name="description" label="Description"><Input.TextArea /></Form.Item>
          <Form.Item><Button type="primary" htmlType="submit">Create</Button></Form.Item>
        </Form>
      </Modal>
    </div>
  );
};
