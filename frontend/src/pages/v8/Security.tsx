// 安全中心页面 - v8
import React, { useState, useEffect } from 'react'
import { Card, Table, Button, Modal, Form, Input, Select, Tag, message, Space, Tabs,Statistic, Row, Col, Timeline, Descriptions } from 'antd'
import { SafetyOutlined, AuditOutlined, LockOutlined, EyeOutlined, DeleteOutlined, SecurityScanOutlined } from '@ant-design/icons'

const { Option } = Select
const { TabPane } = Tabs
const { TextArea } = Input

interface AuditLog {
  id: string
  user_id: string
  action: string
  resource: string
  result: string
  timestamp: string
  ip_address: string
}

interface SecurityStats {
  total_logs: number
  success_count: number
  failed_count: number
  blocked_count: number
}

export function SecurityPage() {
  const [auditLogs, setAuditLogs] = useState<AuditLog[]>([])
  const [stats, setStats] = useState<SecurityStats>({ total_logs: 0, success_count: 0, failed_count: 0, blocked_count: 0 })
  const [loading, setLoading] = useState(false)
  const [maskModal, setMaskModal] = useState(false)
  const [encryptModal, setEncryptModal] = useState(false)
  const [maskResult, setMaskResult] = useState<any>(null)
  const [encryptResult, setEncryptResult] = useState<any>(null)
  const [maskForm] = Form.useForm()
  const [encryptForm] = Form.useForm()

  // 获取审计日志
  const fetchAuditLogs = async () => {
    setLoading(true)
    try {
      const res = await fetch('http://localhost:8000/api/v1/security/audit/logs')
      const data = await res.json()
      if (Array.isArray(data)) {
        setAuditLogs(data.slice(0, 50)) // 取前50条
        const success = data.filter((l: AuditLog) => l.result === 'success').length
        const failed = data.filter((l: AuditLog) => l.result === 'failed').length
        setStats({
          total_logs: data.length,
          success_count: success,
          failed_count: failed,
          blocked_count: 0
        })
      }
    } catch (e) {
      message.error('获取审计日志失败')
    }
    setLoading(false)
  }

  useEffect(() => {
    fetchAuditLogs()
  }, [])

  // 数据脱敏
  const handleMask = async (values: any) => {
    try {
      const res = await fetch('http://localhost:8000/api/v1/security/mask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          data: JSON.parse(values.data),
          type: values.type
        })
      })
      const data = await res.json()
      setMaskResult(data)
    } catch (e) {
      message.error('脱敏失败')
    }
  }

  // 数据加密
  const handleEncrypt = async (values: any) => {
    try {
      const res = await fetch('http://localhost:8000/api/v1/security/encrypt', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ data: values.data })
      })
      const data = await res.json()
      setEncryptResult(data)
    } catch (e) {
      message.error('加密失败')
    }
  }

  const getActionTag = (action: string) => {
    const colors: Record<string, string> = {
      login: 'green',
      logout: 'default',
      read: 'blue',
      write: 'orange',
      delete: 'red',
      admin: 'purple'
    }
    const labels: Record<string, string> = {
      login: '登录',
      logout: '登出',
      read: '读取',
      write: '写入',
      delete: '删除',
      admin: '管理'
    }
    return <Tag color={colors[action] || 'default'}>{labels[action] || action}</Tag>
  }

  const logColumns = [
    { title: '时间', dataIndex: 'timestamp', key: 'timestamp', render: (t: string) => new Date(t).toLocaleString() },
    { title: '用户', dataIndex: 'user_id', key: 'user_id' },
    { title: '操作', dataIndex: 'action', key: 'action', render: (a: string) => getActionTag(a) },
    { title: '资源', dataIndex: 'resource', key: 'resource', ellipsis: true },
    { 
      title: '结果', 
      dataIndex: 'result', 
      key: 'result',
      render: (r: string) => <Tag color={r === 'success' ? 'green' : 'red'}>{r === 'success' ? '成功' : '失败'}</Tag>
    },
    { title: 'IP', dataIndex: 'ip_address', key: 'ip_address' },
  ]

  return (
    <div>
      <h2>🛡️ 安全中心</h2>
      
      {/* 统计卡片 */}
      <Row gutter={16} style={{ marginBottom: 24 }}>
        <Col span={6}>
          <Card>
            <Statistic title="审计日志" value={stats.total_logs} prefix={<AuditOutlined />} />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic title="成功操作" value={stats.success_count} valueStyle={{ color: '#3f8600' }} prefix={<SafetyOutlined />} />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic title="失败操作" value={stats.failed_count} valueStyle={{ color: '#cf1322' }} prefix={<DeleteOutlined />} />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic title="已阻断" value={stats.blocked_count} valueStyle={{ color: '#faad14' }} prefix={<SecurityScanOutlined />} />
          </Card>
        </Col>
      </Row>

      <Tabs defaultActiveKey="audit">
        <TabPane tab="审计日志" key="audit" icon={<AuditOutlined />}>
          <Card>
            <Table 
              dataSource={auditLogs} 
              columns={logColumns} 
              rowKey="id"
              loading={loading}
              pagination={{ pageSize: 10 }}
            />
          </Card>
        </TabPane>
        <TabPane tab="数据脱敏" key="masking" icon={<EyeOutlined />}>
          <Row gutter={16}>
            <Col span={12}>
              <Card title="脱敏工具">
                <Form form={maskForm} layout="vertical" onFinish={handleMask}>
                  <Form.Item name="type" label="数据类型" rules={[{ required: true }]}>
                    <Select placeholder="选择脱敏类型">
                      <Option value="email">邮箱脱敏</Option>
                      <Option value="phone">手机号脱敏</Option>
                      <Option value="id_card">身份证脱敏</Option>
                      <Option value="name">姓名脱敏</Option>
                      <Option value="credit_card">信用卡脱敏</Option>
                    </Select>
                  </Form.Item>
                  <Form.Item name="data" label="原始数据 (JSON)" rules={[{ required: true }]}>
                    <TextArea rows={4} placeholder='{"email": "user@example.com"}' />
                  </Form.Item>
                  <Button type="primary" htmlType="submit">脱敏</Button>
                </Form>
              </Card>
            </Col>
            <Col span={12}>
              <Card title="脱敏结果">
                {maskResult ? (
                  <pre>{JSON.stringify(maskResult, null, 2)}</pre>
                ) : (
                  <p style={{ color: '#999' }}>输入数据后点击脱敏按钮查看结果</p>
                )}
              </Card>
            </Col>
          </Row>
        </TabPane>
        <TabPane tab="数据加密" key="encryption" icon={<LockOutlined />}>
          <Row gutter={16}>
            <Col span={12}>
              <Card title="加密工具 (AES-256)">
                <Form form={encryptForm} layout="vertical" onFinish={handleEncrypt}>
                  <Form.Item name="data" label="待加密数据" rules={[{ required: true }]}>
                    <TextArea rows={4} placeholder="输入要加密的文本" />
                  </Form.Item>
                  <Space>
                    <Button type="primary" htmlType="submit">加密</Button>
                  </Space>
                </Form>
              </Card>
            </Col>
            <Col span={12}>
              <Card title="加密结果">
                {encryptResult ? (
                  <div>
                    <Descriptions column={1} size="small">
                      <Descriptions.Item label="算法">AES-256-GCM</Descriptions.Item>
                      <Descriptions.Item label="密文">
                        <Input.TextArea rows={3} value={encryptResult.encrypted_data} readOnly />
                      </Descriptions.Item>
                      <Descriptions.Item label="IV">
                        <Input value={encryptResult.iv} readOnly />
                      </Descriptions.Item>
                    </Descriptions>
                  </div>
                ) : (
                  <p style={{ color: '#999' }}>输入数据后点击加密按钮查看结果</p>
                )}
              </Card>
            </Col>
          </Row>
        </TabPane>
        <TabPane tab="安全策略" key="policy">
          <Card title="当前安全策略">
            <Descriptions column={2}>
              <Descriptions.Item label="密码策略">
                <Tag color="green">已启用</Tag> 长度≥8, 必须包含数字和字母
              </Descriptions.Item>
              <Descriptions.Item label="双因素认证">
                <Tag color="orange">可选</Tag>
              </Descriptions.Item>
              <Descriptions.Item label="会话超时">
                <Tag color="green">30分钟</Tag>
              </Descriptions.Item>
              <Descriptions.Item label="IP白名单">
                <Tag color="default">未配置</Tag>
              </Descriptions.Item>
              <Descriptions.Item label="数据脱敏">
                <Tag color="green">已启用</Tag>
              </Descriptions.Item>
              <Descriptions.Item label="审计日志">
                <Tag color="green">已启用</Tag> 保留90天
              </Descriptions.Item>
            </Descriptions>
          </Card>
        </TabPane>
      </Tabs>
    </div>
  )
}
