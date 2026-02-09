import React, { useState, useEffect } from 'react';
import { Card, Table, Button, Modal, Form, Input, Space, Tag, List, Typography, Tooltip, Popconfirm } from 'antd';
import { PlusOutlined, FileOutlined, PlayCircleOutlined, DeleteOutlined, ShareAltOutlined, ExportOutlined } from '@ant-design/icons';
import { api } from '../../api/client';

const { TextArea } = Input;
const { Text } = Typography;

interface Notebook {
  notebook_id: string;
  name: string;
  description: string;
  cells_count: number;
  created_at: string;
  updated_at: string;
}

export const NotebooksPage: React.FC = () => {
  const [notebooks, setNotebooks] = useState<Notebook[]>([]);
  const [createModalVisible, setCreateModalVisible] = useState(false);
  const [detailModalVisible, setDetailModalVisible] = useState(false);
  const [selectedNotebook, setSelectedNotebook] = useState<Notebook | null>(null);
  const [form] = Form.useForm();

  useEffect(() => {
    loadNotebooks();
  }, []);

  const loadNotebooks = async () => {
    try {
      const res = await api.get('/notebooks');
      setNotebooks(res.data.notebooks || []);
    } catch (e) {
      console.error('Failed to load notebooks:', e);
    }
  };

  const createNotebook = async (values: any) => {
    try {
      await api.post('/notebooks', {
        name: values.name,
        description: values.description
      });
      setCreateModalVisible(false);
      form.resetFields();
      loadNotebooks();
    } catch (e) {
      console.error('Failed to create notebook:', e);
    }
  };

  const deleteNotebook = async (notebookId: string) => {
    try {
      await api.delete(`/notebooks/${notebookId}`);
      loadNotebooks();
    } catch (e) {
      console.error('Failed to delete notebook:', e);
    }
  };

  const runNotebook = async (notebookId: string) => {
    try {
      await api.post(`/notebooks/${notebookId}/run`);
      // Refresh
      loadNotebooks();
    } catch (e) {
      console.error('Failed to run notebook:', e);
    }
  };

  const columns = [
    {
      title: '名称',
      dataIndex: 'name',
      key: 'name',
      render: (name: string) => (
        <Space>
          <FileOutlined />
          <span>{name}</span>
        </Space>
      )
    },
    {
      title: 'Cells',
      dataIndex: 'cells_count',
      key: 'cells_count',
      render: (count: number) => <Tag>{count} cells</Tag>
    },
    {
      title: '创建时间',
      dataIndex: 'created_at',
      key: 'created_at',
      render: (v: string) => new Date(v).toLocaleString()
    },
    {
      title: '更新时间',
      dataIndex: 'updated_at',
      key: 'updated_at',
      render: (v: string) => new Date(v).toLocaleString()
    },
    {
      title: '操作',
      key: 'action',
      render: (_: any, record: Notebook) => (
        <Space>
          <Tooltip title="运行">
            <Button
              size="small"
              icon={<PlayCircleOutlined />}
              onClick={() => runNotebook(record.notebook_id)}
            />
          </Tooltip>
          <Tooltip title="查看">
            <Button
              size="small"
              icon={<FileOutlined />}
              onClick={() => {
                setSelectedNotebook(record);
                setDetailModalVisible(true);
              }}
            />
          </Tooltip>
          <Popconfirm
            title="确认删除?"
            onConfirm={() => deleteNotebook(record.notebook_id)}
          >
            <Button size="small" danger icon={<DeleteOutlined />} />
          </Popconfirm>
        </Space>
      )
    }
  ];

  return (
    <div className="notebooks-page">
      <Card
        title={
          <Space>
            <FileOutlined />
            Notebooks - 交互式笔记本
          </Space>
        }
        extra={
          <Button type="primary" icon={<PlusOutlined />} onClick={() => setCreateModalVisible(true)}>
            新建Notebook
          </Button>
        }
      >
        <Table
          columns={columns}
          dataSource={notebooks}
          rowKey="notebook_id"
          pagination={false}
        />
      </Card>

      <Modal
        title="新建Notebook"
        open={createModalVisible}
        onCancel={() => setCreateModalVisible(false)}
        footer={null}
      >
        <Form form={form} layout="vertical" onFinish={createNotebook}>
          <Form.Item name="name" label="名称" rules={[{ required: true }]}>
            <Input placeholder="输入Notebook名称" />
          </Form.Item>

          <Form.Item name="description" label="描述">
            <TextArea rows={3} placeholder="输入描述" />
          </Form.Item>

          <Form.Item>
            <Button type="primary" htmlType="submit">
              创建
            </Button>
          </Form.Item>
        </Form>
      </Modal>

      <Modal
        title={
          <Space>
            <FileOutlined />
            {selectedNotebook?.name}
          </Space>
        }
        open={detailModalVisible}
        onCancel={() => setDetailModalVisible(false)}
        footer={null}
        width={720}
      >
        {selectedNotebook && (
          <div>
            <p><Text strong>描述:</Text> {selectedNotebook.description}</p>
            <p><Text strong>Cells:</Text> {selectedNotebook.cells_count}</p>
            <p><Text strong>创建时间:</Text> {new Date(selectedNotebook.created_at).toLocaleString()}</p>
            <p><Text strong>更新时间:</Text> {new Date(selectedNotebook.updated_at).toLocaleString()}</p>
            
            <Space style={{ marginTop: 16 }}>
              <Button icon={<PlayCircleOutlined />} onClick={() => runNotebook(selectedNotebook.notebook_id)}>
                运行所有Cells
              </Button>
              <Button icon={<ExportOutlined />}>
                导出
              </Button>
              <Button icon={<ShareAltOutlined />}>
                分享
              </Button>
            </Space>
          </div>
        )}
      </Modal>
    </div>
  );
};

export default NotebooksPage;
