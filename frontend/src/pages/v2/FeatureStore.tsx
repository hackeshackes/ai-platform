import React, { useState, useEffect } from 'react';
import { Card, Table, Button, Modal, Form, Input, Select, InputNumber, Space, Tag, Descriptions, Progress, Tabs } from 'antd';
import { PlusOutlined, DatabaseOutlined, EyeOutlined, DeleteOutlined } from '@ant-design/icons';
import { api } from '../../api/client';

const { Option } = Select;
const { TextArea } = Input;

interface FeatureGroup {
  group_id: string;
  name: string;
  description: string;
  features_count: number;
  source_type: string;
  created_at: string;
}

export const FeatureStorePage: React.FC = () => {
  const [groups, setGroups] = useState<FeatureGroup[]>([]);
  const [modalVisible, setModalVisible] = useState(false);
  const [detailVisible, setDetailVisible] = useState(false);
  const [selectedGroup, setSelectedGroup] = useState<FeatureGroup | null>(null);
  const [form] = Form.useForm();

  useEffect(() => {
    loadGroups();
  }, []);

  const loadGroups = async () => {
    try {
      const res = await api.get('/feature-store/groups');
      setGroups(res.data.groups || []);
    } catch (e) {
      console.error('Failed to load groups:', e);
    }
  };

  const createGroup = async (values: any) => {
    try {
      await api.post('/feature-store/groups', {
        name: values.name,
        description: values.description,
        source_type: values.source_type,
        features: values.features.map((f: any, i: number) => ({
          name: f.name,
          dtype: f.dtype,
          description: f.description
        }))
      });
      setModalVisible(false);
      form.resetFields();
      loadGroups();
    } catch (e) {
      console.error('Failed to create group:', e);
    }
  };

  const columns = [
    {
      title: '名称',
      dataIndex: 'name',
      key: 'name',
      render: (name: string) => (
        <Space>
          <DatabaseOutlined />
          <span>{name}</span>
        </Space>
      )
    },
    {
      title: '特征数',
      dataIndex: 'features_count',
      key: 'features_count',
    },
    {
      title: '类型',
      dataIndex: 'source_type',
      key: 'source_type',
      render: (type: string) => (
        <Tag color={type === 'batch' ? 'blue' : 'green'}>{type}</Tag>
      )
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
      render: (_: any, record: FeatureGroup) => (
        <Space>
          <Button 
            size="small" 
            icon={<EyeOutlined />}
            onClick={() => {
              setSelectedGroup(record);
              setDetailVisible(true);
            }}
          >
            查看
          </Button>
        </Space>
      )
    }
  ];

  return (
    <div className="feature-store-page">
      <Card
        title={
          <Space>
            <DatabaseOutlined />
            Feature Store - 特征存储
          </Space>
        }
        extra={
          <Button type="primary" icon={<PlusOutlined />} onClick={() => setModalVisible(true)}>
            新建特征组
          </Button>
        }
      >
        <Table
          columns={columns}
          dataSource={groups}
          rowKey="group_id"
          pagination={false}
        />
      </Card>

      <Modal
        title="新建特征组"
        open={modalVisible}
        onCancel={() => setModalVisible(false)}
        footer={null}
        width={720}
      >
        <Form form={form} layout="vertical" onFinish={createGroup}>
          <Form.Item name="name" label="名称" rules={[{ required: true }]}>
            <Input placeholder="输入特征组名称" />
          </Form.Item>

          <Form.Item name="description" label="描述">
            <TextArea rows={2} placeholder="输入描述" />
          </Form.Item>

          <Form.Item name="source_type" label="类型" initialValue="batch">
            <Select>
              <Option value="batch">Batch (离线)</Option>
              <Option value="stream">Stream (实时)</Option>
            </Select>
          </Form.Item>

          <Form.List name="features" initialValue={[{ name: '', dtype: 'float64', description: '' }]}>
            {(fields, { add, remove }) => (
              <>
                {fields.map((field) => (
                  <Space key={field.key} style={{ display: 'flex', marginBottom: 8 }} align="baseline">
                    <Form.Item
                      {...field}
                      name={[field.name, 'name']}
                      label="特征名"
                      rules={[{ required: true }]}
                    >
                      <Input placeholder="特征名" />
                    </Form.Item>
                    <Form.Item
                      {...field}
                      name={[field.name, 'dtype']}
                      label="数据类型"
                    >
                      <Select style={{ width: 120 }}>
                        <Option value="int32">int32</Option>
                        <Option value="float64">float64</Option>
                        <Option value="string">string</Option>
                        <Option value="bool">bool</Option>
                      </Select>
                    </Form.Item>
                    <Form.Item
                      {...field}
                      name={[field.name, 'description']}
                      label="描述"
                    >
                      <Input placeholder="描述" />
                    </Form.Item>
                    <Button type="link" danger onClick={() => remove(field.name)}>
                      删除
                    </Button>
                  </Space>
                ))}
                <Form.Item>
                  <Button type="dashed" onClick={() => add()} block icon={<PlusOutlined />}>
                    添加特征
                  </Button>
                </Form.Item>
              </>
            )}
          </Form.List>

          <Form.Item>
            <Button type="primary" htmlType="submit">
              创建
            </Button>
          </Form.Item>
        </Form>
      </Modal>

      <Modal
        title="特征组详情"
        open={detailVisible}
        onCancel={() => setDetailVisible(false)}
        footer={null}
        width={600}
      >
        {selectedGroup && (
          <Descriptions bordered column={1}>
            <Descriptions.Item label="名称">{selectedGroup.name}</Descriptions.Item>
            <Descriptions.Item label="描述">{selectedGroup.description}</Descriptions.Item>
            <Descriptions.Item label="类型">
              <Tag color={selectedGroup.source_type === 'batch' ? 'blue' : 'green'}>
                {selectedGroup.source_type}
              </Tag>
            </Descriptions.Item>
            <Descriptions.Item label="创建时间">
              {new Date(selectedGroup.created_at).toLocaleString()}
            </Descriptions.Item>
          </Descriptions>
        )}
      </Modal>
    </div>
  );
};

export default FeatureStorePage;
