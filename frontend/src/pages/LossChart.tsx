/**
 * AI Platform - Loss曲线图页面 (ECharts)
 */

import { Card, Row, Col, Select, Button, Statistic, Space, Spin } from 'antd'
import { ReloadOutlined, FullscreenOutlined } from '@ant-design/icons'
import { useState, useEffect, useRef } from 'react'
import * as echarts from 'echarts'
import { metricsAPI } from '../api/client'

interface LossData {
  step: number
  loss: number
  epoch?: number
  timestamp: string
}

export default function LossChart() {
  const chartRef = useRef<HTMLDivElement>(null)
  const chartInstance = useRef<echarts.ECharts | null>(null)
  
  const [loading, setLoading] = useState(true)
  const [data, setData] = useState<LossData[]>([])
  const [experimentId, setExperimentId] = useState('demo-exp-001')
  const [metrics, setMetrics] = useState<any>({})

  const fetchData = async () => {
    try {
      setLoading(true)
      const response = await metricsAPI.loss(experimentId)
      setData(response.data)
      setMetrics(response.metrics || {})
    } catch (error: any) {
      console.error('获取Loss数据失败:', error)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchData()
  }, [experimentId])

  // 初始化ECharts
  useEffect(() => {
    if (!chartRef.current) return

    chartInstance.current = echarts.init(chartRef.current)
    
    const resizeHandler = () => chartInstance.current?.resize()
    window.addEventListener('resize', resizeHandler)
    
    return () => {
      window.removeEventListener('resize', resizeHandler)
      chartInstance.current?.dispose()
    }
  }, [])

  // 更新图表
  useEffect(() => {
    if (!chartInstance.current || data.length === 0) return

    const option: echarts.EChartsOption = {
      title: {
        text: `Loss曲线 - ${experimentId}`,
        left: 'center',
        textStyle: { fontSize: 16 }
      },
      tooltip: {
        trigger: 'axis',
        formatter: (params: any) => {
          const p = params[0]
          return `Step: ${p.value[0]}<br/>Loss: ${p.value[1].toFixed(4)}`
        }
      },
      xAxis: {
        type: 'value',
        name: 'Step',
        nameLocation: 'middle',
        nameGap: 30,
        min: 'dataMin',
        max: 'dataMax'
      },
      yAxis: {
        type: 'value',
        name: 'Loss',
        nameLocation: 'middle',
        nameGap: 40,
        scale: true
      },
      grid: {
        left: 60,
        right: 40,
        top: 60,
        bottom: 60
      },
      toolbox: {
        feature: {
          dataZoom: { yAxisIndex: 'none' },
          restore: {},
          saveAsImage: {}
        }
      },
      dataZoom: [
        { type: 'inside', start: 0, end: 100 },
        { type: 'slider', start: 0, end: 100 }
      ],
      series: [
        {
          name: 'Loss',
          type: 'line',
          data: data.map(d => [d.step, d.loss]),
          smooth: true,
          symbol: 'none',
          lineStyle: {
            width: 2,
            color: '#1890ff'
          },
          areaStyle: {
            color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
              { offset: 0, color: 'rgba(24, 144, 255, 0.3)' },
              { offset: 1, color: 'rgba(24, 144, 255, 0.05)' }
            ])
          },
          markPoint: {
            data: [
              { type: 'min', name: '最低Loss', itemStyle: { color: '#52c41a' } },
              { type: 'max', name: '最高Loss', itemStyle: { color: '#ff4d4f' } }
            ]
          },
          markLine: {
            data: [
              { type: 'average', name: '平均Loss', lineStyle: { color: '#faad14' } }
            ]
          }
        }
      ]
    }

    chartInstance.current.setOption(option)
  }, [data, experimentId])

  return (
    <div>
      <Card
        title={<span>📈 Loss曲线可视化</span>}
        extra={
          <Space>
            <Select
              value={experimentId}
              onChange={setExperimentId}
              style={{ width: 200 }}
              options={[
                { value: 'demo-exp-001', label: '实验 1: Llama Fine-tuning' },
                { value: 'demo-exp-002', label: '实验 2: Qwen SFT' },
              ]}
            />
            <Button icon={<ReloadOutlined />} onClick={fetchData}>刷新</Button>
          </Space>
        }
      >
        {/* 统计卡片 */}
        <Row gutter={16} style={{ marginBottom: 24 }}>
          <Col span={6}>
            <Card size="small">
              <Statistic title="初始Loss" value={metrics.initial_loss || 0} precision={4} />
            </Card>
          </Col>
          <Col span={6}>
            <Card size="small">
              <Statistic title="当前Loss" value={metrics.final_loss || 0} precision={4} valueStyle={{ color: '#52c41a' }} />
            </Card>
          </Col>
          <Col span={6}>
            <Card size="small">
              <Statistic title="最低Loss" value={metrics.best_loss || 0} precision={4} />
            </Card>
          </Col>
          <Col span={6}>
            <Card size="small">
              <Statistic title="数据点数" value={data.length} />
            </Card>
          </Col>
        </Row>

        {/* 图表区域 */}
        <Spin spinning={loading}>
          <div 
            ref={chartRef} 
            style={{ height: 400, width: '100%' }} 
          />
        </Spin>

        {/* 底部说明 */}
        <Card size="small" style={{ marginTop: 16, background: '#f5f5f5' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ color: '#666' }}>
              💡 支持缩放/拖拽 | 点击数据点查看详情 | 可切换实验对比
            </span>
            <Button type="link" icon={<FullscreenOutlined />}>全屏查看</Button>
          </div>
        </Card>
      </Card>
    </div>
  )
}
