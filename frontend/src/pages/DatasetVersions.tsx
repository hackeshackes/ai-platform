/**
 * 数据集版本管理页面 v1.1
 */
import React, { useState, useEffect } from 'react';
import { Table, Card, Button, Modal, Form, Input, message, Tag, Space, Select } from 'antd';
import { PlusOutlined, DeleteOutlined, HistoryOutlined, SwapOutlined, RollbackOutlined } from '@ant-design/icons';
import { api } from '../../api/client';

interface DatasetVersion {
  id: number;
  dataset_id: number;
  version: string;
  commit_message: string | null;
  row_count: number;
  file_size: number;
  created_at: string;
}

interface Dataset {
  id: number;
  name: string;
}

export default function DatasetVersions() {
  const [versions, setVersions] = useState<DatasetVersion[]>([]);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [loading, setLoading] = useState(false);
  const [createModal, setCreateModal] = useState(false);
  const [rollbackModal, setRollbackModal] = useState(false);
  const [form] = Form.useForm();
  const [rollbackForm] = Form.useForm();

  useEffect(() => {
    fetchDatasets();
    fetchVersions();
  }, []);

  const fetchDatasets = async () => {
    try {
      const res = await api.datasets.list();
      setDatasets(res.datasets || []);
    } catch (error) {
      console.error('获取数据集失败:', error);
    }
  };

  const fetchVersions = async () => {
    setLoading(true);
    try {
      const res = await api.versions.list();
      setVersions(res.versions || []);
    } catch (error) {
      message.error('获取版本列表失败');
    } finally {
      setLoading(false);
    }
  };

  const handleCreate = async (values: any) => {
    try {
      await api.versions.create(values);
      message.success('版本创建成功');
      setCreateModal(false);
      form.resetFields();
      fetchVersions();
    } catch (error) {
      message.error('创建失败');
    }
  };

  const handleDelete = async (id: number) => {
    try {
      await api.versions.delete(id);
      message.success('版本已删除');
      fetchVersions();
    } catch (error) {
      message.error('删除失败');
    }
  };

  const handleRollback = async (values: any) => {
    try {
      const targetVersion = versions.find(v => v.id === values.target_version);
      if (!targetVersion) {
        message.error('目标版本不存在');
        return;
      }
      
      await api.versions.create({
        dataset_id: targetVersion.dataset_id,
        version: `v${Date.now()}`,
        commit_message: `回滚到 ${targetVersion.version}`,
        row_count: targetVersion.row_count,
        file_size: targetVersion.file_size,
      });
      
      message.success('回滚成功');
      setRollbackModal(false);
      rollbackForm.resetFields();
      fetchVersions();
    } catch (error) {
      message.error('回滚失败');
    }
  };

  const columns = [
    { title: 'ID', dataIndex: 'id', key: 'id', width: 60 },
    { 
      title: '版本', dataIndex: 'version', key: 'version',
      render: (text: string) => <Tag color="blue">{text}</Tag>,
    },
    {
      title: '数据集',
      dataIndex: 'dataset_id',
      key: 'dataset_id',
      render: (id: number) => {
        const ds = datasets.find(d => d.id === id);
        return ds?.name || `ID: ${id}`;
      },
    },
    { title: '提交信息', dataIndex: 'commit_message', key: 'commit_message', ellipsis: true },
    { title: '行数', dataIndex: 'row_count', key: 'row_count' },
    {
      title: '大小',
      dataIndex: 'file_size',
      key: 'file_size',
      render: (size: number) => `${(size / 1024 / 1024).toFixed(2)} MB`,
    },
    {
      title: '创建时间',
      dataIndex: 'created_at',
      key: 'created_at',
      render: (time: string) => new Date(time).toLocaleString(),
    },
    {
      title: '操作',
      key: 'action',
      render: (_: any, record: DatasetVersion) => (
        <Space>
          <Button 
            size="small"
            icon={<RollbackOutlined />}
            onClick={() => setRollbackModal(true)}
          >
            回滚
          </Button>
          <Button 
            danger 
            size="small"
            icon={<DeleteOutlined />}
            onClick={() => handleDelete(record.id)}
          >
            删除
          </Button>
        </Space>
      ),
    },
  ];

  return (
    <div className="dataset-versions">
      <Card
        title={<Space><HistoryOutlined />数据集版本管理</Space>}
        extra={
          <Button type="primary" icon={<PlusOutlined />} onClick={() => setCreateModal(true)}>
            新建版本
          </Button>
        }
      >
        <Table
          dataSource={versions}
          columns={columns}
          rowKey="id"
          loading={loading}
          pagination={{ pageSize: 10 }}
        />
      </Card>

      <Modal
        title="新建数据集版本"
        open={createModal}
        onCancel={() => setCreateModal(false)}
        footer={null}
      >
        <Form form={form} layout="vertical" onFinish={handleCreate}>
          <Form.Item name="dataset_id" label="数据集" rules={[{ required: true }]}>
            <Select placeholder="选择数据集">
              {datasets.map(ds => (
                <Select.Option key={ds.id} value={ds.id}>{ds.name}</Select.Option>
              ))}
            </Select>
          </Form.Item>
          <Form.Item name="version" label="版本号" rules={[{ required: true }]}>
            <Input placeholder="例如: v1.0" />
          </Form.Item>
          <Form.Item name="commit_message" label="提交信息">
            <Input.TextArea rows={3} placeholder="描述变更..." />
          </Form.Item>
          <Form.Item>
            <Button type="primary" htmlType="submit" block>创建版本</Button>
          </Form.Item>
        </Form>
      </Modal>

      <Modal
        title="回滚到指定版本"
        open={rollbackModal}
        onCancel={() => setRollbackModal(false)}
        footer={null}
      >
        <Form form={rollbackForm} layout="vertical" onFinish={handleRollback}>
          <Form.Item name="target_version" label="选择目标版本" rules={[{ required: true }]}>
            <Select placeholder="选择要回滚的版本">
              {versions.map(v => (
                <Select.Option key={v.id} value={v.id}>
                  {v.version} - {v.commit_message || '无描述'}
                </Select.Option>
              ))}
            </Select>
          </Form.Item>
          <Form.Item>
            <Button type="primary" danger htmlType="submit" block>确认回滚</Button>
          </Form.Item>
        </Form>
      </Modal>
    </div>
  );
}
