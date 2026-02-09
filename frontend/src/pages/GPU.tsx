/**
 * AI Platform - GPU监控页面
 */

import { Card, Row, Col, Statistic, Progress, Spin, Tag, List } from 'antd'
import { ThunderboltOutlined, RiseOutlined, ThermometerOutlined, DashboardOutlined } from '@ant-design/icons'
import { useState, useEffect } from 'react'
import { metricsAPI } from '../api/client'

interface GPUMetric {
  gpu_id: number
  name: string
  total_memory_mb: number
  used_memory_mb: number
  utilization_percent: number
  temperature_c: number
  power_watts?: number
}

export default function GPUMonitor() {
  const [loading, setLoading] = useState(true)
  const [data, setData] = useState<any>(null)

  const fetchGPU = async () => {
    try {
      setLoading(true)
      const result = await metricsAPI.gpu()
      setData(result)
    } catch (error: any) {
      console.error('获取GPU数据失败:', error)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchGPU()
    // 每5秒刷新
    const interval = setInterval(fetchGPU, 5000)
    return () => clearInterval(interval)
  }, [])

  if (loading && !data) {
    return (
      <div style={{ textAlign: 'center', padding: 100 }}>
        <Spin size="large" tip="加载GPU监控数据..." />
      </div>
    )
  }

  const gpuList = data?.metrics || []

  return (
    <div>
      <Card title={<><DashboardOutlined /> GPU 实时监控</>}>
        {/* 总览卡片 */}
        <Row gutter={16} style={{ marginBottom: 24 }}>
          <Col span={6}>
            <Card>
              <Statistic
                title="GPU数量"
                value={data?.total_gpus || 0}
                prefix={<ThunderboltOutlined />}
              />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic
                title="总显存"
                value={data?.total_memory_mb || 0}
                suffix="MB"
                prefix={<RiseOutlined />}
              />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic
                title="已用显存"
                value={data?.used_memory_mb || 0}
                suffix="MB"
              />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic
                title="平均利用率"
                value={data?.avg_utilization || 0}
                suffix="%"
                precision={1}
              />
            </Card>
          </Col>
        </Row>

        {/* 单个GPU详情 */}
        {gpuList.map((gpu: GPUMetric) => {
          const memoryPercent = Math.round((gpu.used_memory_mb / gpu.total_memory_mb) * 100)
          const utilColor = gpu.utilization_percent > 80 ? 'red' : gpu.utilization_percent > 50 ? 'orange' : 'green'
          const tempColor = gpu.temperature_c > 80 ? 'red' : gpu.temperature_c > 60 ? 'orange' : 'green'

          return (
            <Card 
              key={gpu.gpu_id} 
              style={{ marginBottom: 16 }}
              title={
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <span>GPU {gpu.gpu_id}: {gpu.name}</span>
                  <Tag color={utilColor}>{gpu.utilization_percent}% 利用</Tag>
                  <Tag color={tempColor}>{gpu.temperature_c}°C</Tag>
                </div>
              }
            >
              <Row gutter={24}>
                {/* 显存使用 */}
                <Col span={12}>
                  <div style={{ marginBottom: 16 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                      <span>显存使用</span>
                      <span>{gpu.used_memory_mb} MB / {gpu.total_memory_mb} MB</span>
                    </div>
                    <Progress 
                      percent={memoryPercent} 
                      strokeColor={memoryPercent > 80 ? '#ff4d4f' : '#52c41a'}
                      size="small"
                    />
                  </div>
                </Col>

                {/* GPU利用率 */}
                <Col span={12}>
                  <div style={{ marginBottom: 16 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                      <span>GPU计算利用率</span>
                      <span>{gpu.utilization_percent}%</span>
                    </div>
                    <Progress 
                      percent={gpu.utilization_percent} 
                      strokeColor={utilColor === 'red' ? '#ff4d4f' : utilColor === 'orange' ? '#faad14' : '#52c41a'}
                      size="small"
                    />
                  </div>
                </Col>
              </Row>

              {/* 详细信息列表 */}
              <List
                size="small"
                bordered
                dataSource={[
                  { label: 'GPU ID', value: gpu.gpu_id },
                  { label: 'GPU名称', value: gpu.name },
                  { label: '总显存', value: `${gpu.total_memory_mb} MB` },
                  { label: '已用显存', value: `${gpu.used_memory_mb} MB` },
                  { label: '利用率', value: `${gpu.utilization_percent}%` },
                  { label: '温度', value: `${gpu.temperature_c}°C` },
                  { label: '功耗', value: gpu.power_watts ? `${gpu.power_watts} W` : 'N/A' },
                ]}
                renderItem={(item: any) => (
                  <List.Item>
                    <span style={{ color: '#888' }}>{item.label}</span>
                    <span style={{ fontWeight: 500 }}>{item.value}</span>
                  </List.Item>
                )}
              />
            </Card>
          )
        })}

        {/* 底部说明 */}
        <Card size="small" style={{ background: '#f5f5f5' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ color: '#666' }}>
              💡 数据每5秒自动刷新 | 显示{ gpuList.length }个GPU设备
            </span>
            <span style={{ color: '#999', fontSize: 12 }}>
              {data ? new Date().toLocaleTimeString() : '-'}
            </span>
          </div>
        </Card>
      </Card>
    </div>
  )
}
