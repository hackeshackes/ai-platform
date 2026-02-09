import React, { useState, useEffect } from 'react';
import { Card, Table, Button, Modal, Form, Input, Select, InputNumber, Space, Tag, List, Avatar, Typography, Empty } from 'antd';
import { PlusOutlined, RobotOutlined, FileSearchOutlined, DeleteOutlined, SendOutlined } from '@ant-design/icons';
import { api } from '../../api/client';

const { TextArea } = Input;
const { Text, Title } = Typography;

interface RAGCollection {
  id: string;
  name: string;
  description: string;
  document_count: number;
  chunk_count: number;
  created_at: string;
}

interface RAGQueryResult {
  answer: string;
  sources: { chunk_id: string; score: number }[];
  metadata: Record<string, any>;
}

export const RAGPage: React.FC = () => {
  const [collections, setCollections] = useState<RAGCollection[]>([]);
  const [createModalVisible, setCreateModalVisible] = useState(false);
  const [queryModalVisible, setQueryModalVisible] = useState(false);
  const [selectedCollection, setSelectedCollection] = useState<string | null>(null);
  const [queryResult, setQueryResult] = useState<RAGQueryResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [form] = Form.useForm();
  const [queryForm] = Form.useForm();

  useEffect(() => {
    loadCollections();
  }, []);

  const loadCollections = async () => {
    try {
      const res = await api.get('/rag/collections');
      setCollections(res.data.collections || []);
    } catch (e) {
      console.error('Failed to load collections:', e);
    }
  };

  const createCollection = async (values: any) => {
    try {
      await api.post('/rag/collections', values);
      setCreateModalVisible(false);
      form.resetFields();
      loadCollections();
    } catch (e) {
      console.error('Failed to create collection:', e);
    }
  };

  const addDocuments = async (values: any) => {
    if (!selectedCollection) return;
    
    try {
      const documents = values.documents.split('\n').filter((d: string) => d.trim());
      await api.post(`/rag/collections/${selectedCollection}/documents`, {
        documents
      });
      setQueryModalVisible(false);
      queryForm.resetFields();
      loadCollections();
    } catch (e) {
      console.error('Failed to add documents:', e);
    }
  };

  const executeQuery = async (values: any) => {
    if (!selectedCollection) return;
    
    setLoading(true);
    try {
      const res = await api.post(`/rag/collections/${selectedCollection}/query`, {
        question: values.question,
        top_k: values.top_k || 5
      });
      setQueryResult(res.data);
    } catch (e) {
      console.error('Failed to query:', e);
    } finally {
      setLoading(false);
    }
  };

  const columns = [
    {
      title: '名称',
      dataIndex: 'name',
      key: 'name',
      render: (name: string) => (
        <Space>
          <Avatar icon={<RobotOutlined />} />
          <Text strong>{name}</Text>
        </Space>
      )
    },
    {
      title: '文档数',
      dataIndex: 'document_count',
      key: 'document_count',
    },
    {
      title: '块数',
      dataIndex: 'chunk_count',
      key: 'chunk_count',
    },
    {
      title: '创建时间',
      dataIndex: 'created_at',
      key: 'created_at',
      render: (v: string) => new Date(v).toLocaleString()
    },
    {
      title: '操作',
      key: 'action',
      render: (_: any, record: RAGCollection) => (
        <Space>
          <Button 
            size="small" 
            icon={<FileSearchOutlined />}
            onClick={() => {
              setSelectedCollection(record.id);
              setQueryModalVisible(true);
            }}
          >
            查询
          </Button>
          <Button 
            size="small"
            type="primary"
            onClick={() => {
              setSelectedCollection(record.id);
              setQueryModalVisible(true);
            }}
          >
            添加文档
          </Button>
        </Space>
      )
    }
  ];

  return (
    <div className="rag-page">
      <Card
        title={
          <Space>
            <RobotOutlined />
            RAG - 检索增强生成
          </Space>
        }
        extra={
          <Button type="primary" icon={<PlusOutlined />} onClick={() => setCreateModalVisible(true)}>
            新建知识库
          </Button>
        }
      >
        {collections.length === 0 ? (
          <Empty description="暂无知识库，点击新建创建一个" />
        ) : (
          <Table
            columns={columns}
            dataSource={collections}
            rowKey="id"
            pagination={false}
          />
        )}
      </Card>

      {/* 新建知识库 */}
      <Modal
        title="新建知识库"
        open={createModalVisible}
        onCancel={() => setCreateModalVisible(false)}
        footer={null}
      >
        <Form form={form} layout="vertical" onFinish={createCollection}>
          <Form.Item name="name" label="名称" rules={[{ required: true }]}>
            <Input placeholder="输入知识库名称" />
          </Form.Item>
          <Form.Item name="description" label="描述">
            <TextArea rows={3} placeholder="输入描述" />
          </Form.Item>
          <Form.Item>
            <Button type="primary" htmlType="submit">创建</Button>
          </Form.Item>
        </Form>
      </Modal>

      {/* 查询/添加文档 */}
      <Modal
        title="RAG查询"
        open={queryModalVisible}
        onCancel={() => {
          setQueryModalVisible(false);
          setQueryResult(null);
        }}
        footer={null}
        width={800}
      >
        <Space direction="vertical" style={{ width: '100%' }} size="large">
          {/* 添加文档 */}
          <Card title="添加文档" size="small">
            <Form form={queryForm} onFinish={addDocuments}>
              <Form.Item name="documents" rules={[{ required: true }]}>
                <TextArea rows={4} placeholder="每行一个文档" />
              </Form.Item>
              <Button type="primary" htmlType="submit">添加文档</Button>
            </Form>
          </Card>

          {/* 查询 */}
          <Card title="向知识库提问" size="small">
            <Form layout="inline" onFinish={executeQuery}>
              <Form.Item name="question" rules={[{ required: true }]} style={{ flex: 1 }}>
                <TextArea rows={2} placeholder="输入您的问题" />
              </Form.Item>
              <Form.Item name="top_k" initialValue={5}>
                <InputNumber min={1} max={20} placeholder="top_k" />
              </Form.Item>
              <Form.Item>
                <Button type="primary" htmlType="submit" loading={loading} icon={<SendOutlined />}>
                  查询
                </Button>
              </Form.Item>
            </Form>
          </Card>

          {/* 查询结果 */}
          {queryResult && (
            <Card title="查询结果" size="small">
              <div style={{ marginBottom: 16 }}>
                <Title level={5}>答案</Title>
                <Text>{queryResult.answer}</Text>
              </div>
              
              <div>
                <Title level={5}>参考来源</Title>
                <List
                  size="small"
                  dataSource={queryResult.sources}
                  renderItem={(item) => (
                    <List.Item>
                      <Text code>{item.chunk_id}</Text>
                      <Text type="secondary">相似度: {(item.score * 100).toFixed(2)}%</Text>
                    </List.Item>
                  )}
                />
              </div>
            </Card>
          )}
        </Space>
      </Modal>
    </div>
  );
};

export default RAGPage;
