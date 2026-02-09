/**
 * 权限管理页面 v1.1
 */
import React, { useState, useEffect } from 'react';
import { Table, Card, Button, Modal, Form, Input, Select, Tag, Space, message } from 'antd';
import { PlusOutlined, DeleteOutlined, SafetyCertificateOutlined } from '@ant-design/icons';
import { api } from '../api/client';

interface Role {
  id: number;
  name: string;
  description: string | null;
  permissions: string[];
  created_at: string;
}

const PERMISSIONS = [
  { label: '项目读取', value: 'projects:read' },
  { label: '项目写入', value: 'projects:write' },
  { label: '任务读取', value: 'tasks:read' },
  { label: '任务写入', value: 'tasks:write' },
  { label: '数据集读取', value: 'datasets:read' },
  { label: '数据集写入', value: 'datasets:write' },
  { label: '模型读取', value: 'models:read' },
  { label: '模型写入', value: 'models:write' },
];

export default function Roles() {
  const [roles, setRoles] = useState<Role[]>([]);
  const [loading, setLoading] = useState(false);
  const [modalVisible, setModalVisible] = useState(false);
  const [form] = Form.useForm();

  useEffect(() => { fetchRoles(); }, []);

  const fetchRoles = async () => {
    setLoading(true);
    try {
      const res = await api.roles?.list?.() || [];
      setRoles(res.map((r: any) => ({
        ...r,
        permissions: typeof r.permissions === 'string' ? r.permissions.split(',') : r.permissions
      })));
    } catch (error) {
      message.error('获取角色列表失败');
    } finally {
      setLoading(false);
    }
  };

  const handleCreate = async (values: any) => {
    try {
      await api.roles?.create?.(values);
      message.success('角色创建成功');
      setModalVisible(false);
      form.resetFields();
      fetchRoles();
    } catch (error: any) {
      message.error(error?.detail || '创建失败');
    }
  };

  const handleInit = async () => {
    try {
      await api.roles?.init?.();
      message.success('默认角色已初始化');
      fetchRoles();
    } catch (error) {
      message.error('初始化失败');
    }
  };

  const columns = [
    { title: 'ID', dataIndex: 'id', key: 'id', width: 60 },
    { title: '角色名', dataIndex: 'name', key: 'name' },
    { title: '描述', dataIndex: 'description', key: 'description', ellipsis: true },
    { 
      title: '权限', dataIndex: 'permissions', key: 'permissions',
      render: (perms: string[]) => (
        <Space wrap>
          {perms?.slice(0, 3).map((p: string) => <Tag key={p} color="blue">{p}</Tag>)}
          {perms?.length > 3 && <Tag>+{perms.length - 3}</Tag>}
        </Space>
      )
    },
    {
      title: '操作', key: 'action',
      render: (_: any, record: Role) => (
        <Button danger size="small" icon={<DeleteOutlined />}>删除</Button>
      ),
    },
  ];

  return (
    <div className="roles">
      <Card
        title={<Space><SafetyCertificateOutlined />权限管理</Space>}
        extra={
          <Space>
            <Button onClick={handleInit}>初始化默认</Button>
            <Button type="primary" icon={<PlusOutlined />} onClick={() => setModalVisible(true)}>新建角色</Button>
          </Space>
        }
      >
        <Table dataSource={roles} columns={columns} rowKey="id" loading={loading} pagination={{ pageSize: 10 }} />
      </Card>

      <Modal title="新建角色" open={modalVisible} onCancel={() => setModalVisible(false)} footer={null}>
        <Form form={form} layout="vertical" onFinish={handleCreate}>
          <Form.Item name="name" label="角色名" rules={[{ required: true }]}><Input placeholder="角色名" /></Form.Item>
          <Form.Item name="description" label="描述"><Input placeholder="角色描述" /></Form.Item>
          <Form.Item name="permissions" label="权限" rules={[{ required: true }]}>
            <Select mode="multiple" placeholder="选择权限" options={PERMISSIONS} />
          </Form.Item>
          <Form.Item><Button type="primary" htmlType="submit" block>创建</Button></Form.Item>
        </Form>
      </Modal>
    </div>
  );
}
