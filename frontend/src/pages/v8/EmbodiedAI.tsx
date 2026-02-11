// 具身AI页面 - v8
import React, { useState, useEffect } from 'react'
import { Card, Table, Button, Modal, Form, Input, Select, Tag, message, Space, Tabs, Descriptions, Statistic, Row, Col } from 'antd'
import { PlusOutlined, RobotOutlined, ApiOutlined, ControlOutlined, ThunderboltOutlined } from '@ant-design/icons'

const { Option } = Select
const { TabPane } = Tabs

interface Device {
  id: string
  name: string
  type: string
  protocol: string
  status: string
  capabilities: string[]
  last_seen: string
}

interface DeviceStats {
  total: number
  online: number
  offline: number
  by_type: Record<string, number>
}

export function EmbodiedAIPage() {
  const [devices, setDevices] = useState<Device[]>([])
  const [stats, setStats] = useState<DeviceStats>({ total: 0, online: 0, offline: 0, by_type: {} })
  const [loading, setLoading] = useState(false)
  const [registerModal, setRegisterModal] = useState(false)
  const [controlModal, setControlModal] = useState(false)
  const [selectedDevice, setSelectedDevice] = useState<Device | null>(null)
  const [form] = Form.useForm()
  const [controlForm] = Form.useForm()

  // 获取设备列表
  const fetchDevices = async () => {
    setLoading(true)
    try {
      const res = await fetch('http://localhost:8000/api/v1/embodied/devices')
      const data = await res.json()
      if (Array.isArray(data)) {
        setDevices(data)
        // 计算统计数据
        const online = data.filter((d: Device) => d.status === 'online').length
        setStats({
          total: data.length,
          online,
          offline: data.length - online,
          by_type: data.reduce((acc: Record<string, number>, d: Device) => {
            acc[d.type] = (acc[d.type] || 0) + 1
            return acc
          }, {})
        })
      }
    } catch (e) {
      message.error('获取设备列表失败')
    }
    setLoading(false)
  }

  useEffect(() => {
    fetchDevices()
  }, [])

  // 注册设备
  const handleRegister = async (values: any) => {
    try {
      const res = await fetch('http://localhost:8000/api/v1/embodied/devices/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(values)
      })
      const data = await res.json()
      if (data.success || data.id) {
        message.success('设备注册成功')
        setRegisterModal(false)
        form.resetFields()
        fetchDevices()
      } else {
        message.error(data.detail || '注册失败')
      }
    } catch (e) {
      message.error('注册失败')
    }
  }

  // 控制设备
  const handleControl = async (values: any) => {
    if (!selectedDevice) return
    try {
      const res = await fetch(`http://localhost:8000/api/v1/embodied/devices/${selectedDevice.id}/control`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          device_id: selectedDevice.id,
          action: values.action,
          params: values.params || {}
        })
      })
      const data = await res.json()
      if (data.success) {
        message.success('控制命令发送成功')
        setControlModal(false)
        controlForm.resetFields()
      } else {
        message.error(data.detail || '控制失败')
      }
    } catch (e) {
      message.error('控制失败')
    }
  }

  const deviceColumns = [
    { title: '名称', dataIndex: 'name', key: 'name' },
    { title: '类型', dataIndex: 'type', key: 'type', render: (t: string) => <Tag color={getTypeColor(t)}>{t}</Tag> },
    { title: '协议', dataIndex: 'protocol', key: 'protocol' },
    { 
      title: '状态', 
      dataIndex: 'status', 
      key: 'status',
      render: (status: string) => (
        <Tag color={status === 'online' ? 'green' : 'red'}>
          {status === 'online' ? '在线' : '离线'}
        </Tag>
      )
    },
    { 
      title: '能力', 
      dataIndex: 'capabilities', 
      key: 'capabilities',
      render: (caps: string[]) => caps.slice(0, 2).map(c => <Tag key={c}>{c}</Tag>)
    },
    { title: '最后活跃', dataIndex: 'last_seen', key: 'last_seen', render: (t: string) => t ? new Date(t).toLocaleString() : 'N/A' },
    {
      title: '操作',
      key: 'action',
      render: (_: any, record: Device) => (
        <Space>
          <Button type="link" onClick={() => {
            setSelectedDevice(record)
            setControlModal(true)
          }}>
            控制
          </Button>
          <Button type="link">详情</Button>
        </Space>
      )
    }
  ]

  const getTypeColor = (type: string) => {
    const colors: Record<string, string> = {
      ROBOT: 'blue',
      SENSOR: 'green',
      IOT: 'orange',
      CAMERA: 'purple',
      AR_VR: 'cyan'
    }
    return colors[type] || 'default'
  }

  return (
    <div>
      <h2>🦾 具身AI</h2>
      
      {/* 统计卡片 */}
      <Row gutter={16} style={{ marginBottom: 24 }}>
        <Col span={6}>
          <Card>
            <Statistic title="设备总数" value={stats.total} prefix={<RobotOutlined />} />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic title="在线设备" value={stats.online} valueStyle={{ color: '#3f8600' }} prefix={<ThunderboltOutlined />} />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic title="离线设备" value={stats.offline} valueStyle={{ color: '#cf1322' }} />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic title="设备类型" value={Object.keys(stats.by_type).length} prefix={<ApiOutlined />} />
          </Card>
        </Col>
      </Row>

      <Tabs defaultActiveKey="devices">
        <TabPane tab="设备管理" key="devices" icon={<ControlOutlined />}>
          <Card extra={
            <Button type="primary" icon={<PlusOutlined />} onClick={() => setRegisterModal(true)}>
              注册设备
            </Button>
          }>
            <Table 
              dataSource={devices} 
              columns={deviceColumns} 
              rowKey="id"
              loading={loading}
            />
          </Card>
        </TabPane>
        <TabPane tab="机器人控制" key="robots">
          <Card title="🤖 机器人控制台">
            <Row gutter={16}>
              <Col span={12}>
                <Form layout="vertical">
                  <Form.Item label="选择机器人">
                    <Select placeholder="选择要控制的机器人">
                      {devices.filter(d => d.type === 'ROBOT').map(d => (
                        <Option key={d.id} value={d.id}>{d.name}</Option>
                      ))}
                    </Select>
                  </Form.Item>
                  <Form.Item label="移动命令">
                    <Space>
                      <Button>前进</Button>
                      <Button>后退</Button>
                      <Button>左转</Button>
                      <Button>右转</Button>
                      <Button>停止</Button>
                    </Space>
                  </Form.Item>
                </Form>
              </Col>
              <Col span={12}>
                <Card title="状态监控" size="small">
                  <Descriptions column={1}>
                    <Descriptions.Item label="位置">X: 0.00, Y: 0.00</Descriptions.Item>
                    <Descriptions.Item label="姿态">Roll: 0°, Pitch: 0°, Yaw: 0°</Descriptions.Item>
                    <Descriptions.Item label="速度">0.0 m/s</Descriptions.Item>
                    <Descriptions.Item label="电池">85%</Descriptions.Item>
                  </Descriptions>
                </Card>
              </Col>
            </Row>
          </Card>
        </TabPane>
        <TabPane tab="传感器数据" key="sensors">
          <Card title="📡 传感器监控">
            <Row gutter={16}>
              {devices.filter(d => d.type === 'SENSOR').map(device => (
                <Col span={8} key={device.id}>
                  <Card size="small" title={device.name}>
                    <Statistic title="状态" value="正常" valueStyle={{ color: '#52c41a' }} />
                    <p>最后更新: {device.last_seen ? new Date(device.last_seen).toLocaleString() : 'N/A'}</p>
                  </Card>
                </Col>
              ))}
              {devices.filter(d => d.type === 'SENSOR').length === 0 && (
                <Col span={24}>
                  <Card>暂无传感器设备</Card>
                </Col>
              )}
            </Row>
          </Card>
        </TabPane>
      </Tabs>

      {/* 注册设备弹窗 */}
      <Modal
        title="注册设备"
        open={registerModal}
        onCancel={() => setRegisterModal(false)}
        footer={null}
      >
        <Form form={form} layout="vertical" onFinish={handleRegister}>
          <Form.Item name="name" label="设备名称" rules={[{ required: true }]}>
            <Input placeholder="输入设备名称" />
          </Form.Item>
          <Form.Item name="type" label="设备类型" rules={[{ required: true }]}>
            <Select placeholder="选择类型">
              <Option value="ROBOT">机器人</Option>
              <Option value="SENSOR">传感器</Option>
              <Option value="IOT">IoT设备</Option>
              <Option value="CAMERA">摄像头</Option>
              <Option value="AR_VR">AR/VR设备</Option>
            </Select>
          </Form.Item>
          <Form.Item name="protocol" label="通信协议" rules={[{ required: true }]}>
            <Select placeholder="选择协议">
              <Option value="MQTT">MQTT</Option>
              <Option value="REST">REST API</Option>
              <Option value="WebSocket">WebSocket</Option>
              <Option value="ROS">ROS</Option>
              <Option value="Modbus">Modbus</Option>
            </Select>
          </Form.Item>
          <Form.Item name="capabilities" label="设备能力">
            <Select mode="tags" placeholder="输入能力标签">
              <Option value="motion_control">运动控制</Option>
              <Option value="sensing">感知</Option>
              <Option value="vision">视觉</Option>
              <Option value="navigation">导航</Option>
            </Select>
          </Form.Item>
          <Button type="primary" htmlType="submit" block>注册</Button>
        </Form>
      </Modal>

      {/* 控制设备弹窗 */}
      <Modal
        title={`控制设备: ${selectedDevice?.name}`}
        open={controlModal}
        onCancel={() => setControlModal(false)}
        footer={null}
      >
        <Form form={controlForm} layout="vertical" onFinish={handleControl}>
          <Form.Item name="action" label="控制命令" rules={[{ required: true }]}>
            <Select placeholder="选择命令">
              <Option value="turn_on">开启</Option>
              <Option value="turn_off">关闭</Option>
              <Option value="reset">重置</Option>
              <Option value="calibrate">校准</Option>
              <Option value="set_mode">设置模式</Option>
            </Select>
          </Form.Item>
          <Form.Item name="params" label="参数 (JSON)">
            <Form.Item noStyle>
              <Input.TextArea rows={3} placeholder='{"mode": "auto"}' />
            </Form.Item>
          </Form.Item>
          <Button type="primary" htmlType="submit" block>发送命令</Button>
        </Form>
      </Modal>
    </div>
  )
}
