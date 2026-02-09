import React, { useState, useEffect } from 'react';
import { Card, Table, Button, Modal, Form, Input, Select, Space, Tag, Tabs, Statistic, Typography, message, Row, Col } from 'antd';
import { PlusOutlined, BarChartOutlined, LineChartOutlined, PieChartOutlined, DashboardOutlined } from '@ant-design/icons';
import { api } from '../../api/client';

const { Text, Title } = Typography;

interface Chart {
  chart_id: string;
  name: string;
  type: string;
  data_source: string;
}

interface Report {
  report_id: string;
  name: string;
  type: string;
  created_by: string;
}

interface Dashboard {
  dashboard_id: string;
  name: string;
  widgets_count: number;
}

export const VisualizationsPage: React.FC = () => {
  const [charts, setCharts] = useState<Chart[]>([]);
  const [reports, setReports] = useState<Report[]>([]);
  const [dashboards, setDashboards] = useState<Dashboard[]>([]);
  const [loading, setLoading] = useState(false);
  const [createChartModal, setCreateChartModal] = useState(false);
  const [createReportModal, setCreateReportModal] = useState(false);
  const [form] = Form.useForm();

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    setLoading(true);
    try {
      const [chartsRes, reportsRes, dashboardsRes] = await Promise.all([
        api.get('/visualization/charts'),
        api.get('/visualization/reports'),
        api.get('/visualization/dashboards')
      ]);
      setCharts(chartsRes.data.charts || []);
      setReports(reportsRes.data.reports || []);
      setDashboards(dashboardsRes.data.dashboards || []);
    } catch (e) {
      console.error('Failed to load data:', e);
    } finally {
      setLoading(false);
    }
  };

  const createChart = async (values: any) => {
    try {
      await api.post('/visualization/charts', values);
      setCreateChartModal(false);
      form.resetFields();
      loadData();
      message.success('Chart created');
    } catch (e) {
      message.error('Failed to create chart');
    }
  };

  const createReport = async (values: any) => {
    try {
      await api.post('/visualization/reports', values);
      setCreateReportModal(false);
      form.resetFields();
      loadData();
      message.success('Report created');
    } catch (e) {
      message.error('Failed to create report');
    }
  };

  const generateReport = async (id: string) => {
    try {
      const res = await api.post(`/visualization/reports/${id}/generate`);
      message.success('Report generated');
    } catch (e) {
      message.error('Failed to generate report');
    }
  };

  const chartColumns = [
    { title: 'Name', dataIndex: 'name', key: 'name' },
    { title: 'Type', dataIndex: 'type', key: 'type', render: (t: string) => <Tag>{t}</Tag> },
    { title: 'Data Source', dataIndex: 'data_source', key: 'source' },
  ];

  const reportColumns = [
    { title: 'Name', dataIndex: 'name', key: 'name' },
    { title: 'Type', dataIndex: 'type', key: 'type' },
    { title: 'Created By', dataIndex: 'created_by', key: 'created_by' },
    {
      title: 'Actions',
      key: 'actions',
      render: (_: any, record: Report) => (
        <Button type="link" onClick={() => generateReport(record.report_id)}>Generate</Button>
      )
    },
  ];

  const dashboardColumns = [
    { title: 'Name', dataIndex: 'name', key: 'name' },
    { title: 'Widgets', dataIndex: 'widgets_count', key: 'widgets' },
  ];

  return (
    <div style={{ padding: 24 }}>
      <Title level={2}><DashboardOutlined /> Visualization</Title>

      <Row gutter={16} style={{ marginBottom: 24 }}>
        <Col span={8}>
          <Card><Statistic title="Charts" value={charts.length} prefix={<BarChartOutlined />} /></Card>
        </Col>
        <Col span={8}>
          <Card><Statistic title="Reports" value={reports.length} prefix={<LineChartOutlined />} /></Card>
        </Col>
        <Col span={8}>
          <Card><Statistic title="Dashboards" value={dashboards.length} prefix={<PieChartOutlined />} /></Card>
        </Col>
      </Row>

      <Tabs defaultActiveKey="charts">
        <TabPane tab="Charts" key="charts">
          <Card>
            <div style={{ marginBottom: 16 }}>
              <Button type="primary" icon={<PlusOutlined />} onClick={() => setCreateChartModal(true)}>
                Create Chart
              </Button>
            </div>
            <Table columns={chartColumns} dataSource={charts} rowKey="chart_id" loading={loading} />
          </Card>
        </TabPane>
        
        <TabPane tab="Reports" key="reports">
          <Card>
            <div style={{ marginBottom: 16 }}>
              <Button type="primary" icon={<PlusOutlined />} onClick={() => setCreateReportModal(true)}>
                Create Report
              </Button>
            </div>
            <Table columns={reportColumns} dataSource={reports} rowKey="report_id" loading={loading} />
          </Card>
        </TabPane>
        
        <TabPane tab="Dashboards" key="dashboards">
          <Card>
            <Table columns={dashboardColumns} dataSource={dashboards} rowKey="dashboard_id" loading={loading} />
          </Card>
        </TabPane>
      </Tabs>

      <Modal title="Create Chart" open={createChartModal} onCancel={() => setCreateChartModal(false)} footer={null}>
        <Form form={form} onFinish={createChart} layout="vertical">
          <Form.Item name="name" label="Name" rules={[{ required: true }]}><Input /></Form.Item>
          <Form.Item name="chart_type" label="Type" rules={[{ required: true }]}>
            <Select>
              <Select.Option value="line">Line Chart</Select.Option>
              <Select.Option value="bar">Bar Chart</Select.Option>
              <Select.Option value="scatter">Scatter Plot</Select.Option>
              <Select.Option value="histogram">Histogram</Select.Option>
              <Select.Option value="heatmap">Heatmap</Select.Option>
            </Select>
          </Form.Item>
          <Form.Item name="data_source" label="Data Source" rules={[{ required: true }]}>
            <Select>
              <Select.Option value="experiments">Experiments</Select.Option>
              <Select.Option value="models">Models</Select.Option>
              <Select.Option value="cost">Cost</Select.Option>
            </Select>
          </Form.Item>
          <Form.Item name="x_axis" label="X Axis"><Input /></Form.Item>
          <Form.Item name="y_axis" label="Y Axis"><Input /></Form.Item>
          <Form.Item><Button type="primary" htmlType="submit">Create</Button></Form.Item>
        </Form>
      </Modal>

      <Modal title="Create Report" open={createReportModal} onCancel={() => setCreateReportModal(false)} footer={null}>
        <Form form={form} onFinish={createReport} layout="vertical">
          <Form.Item name="name" label="Name" rules={[{ required: true }]}><Input /></Form.Item>
          <Form.Item name="description" label="Description"><Input.TextArea /></Form.Item>
          <Form.Item name="report_type" label="Type" rules={[{ required: true }]}>
            <Select>
              <Select.Option value="experiment_comparison">Experiment Comparison</Select.Option>
              <Select.Option value="model_performance">Model Performance</Select.Option>
              <Select.Option value="cost_analysis">Cost Analysis</Select.Option>
            </Select>
          </Form.Item>
          <Form.Item><Button type="primary" htmlType="submit">Create</Button></Form.Item>
        </Form>
      </Modal>
    </div>
  );
};
