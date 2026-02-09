/**
 * 数据质量检查页面 v1.1
 */
import React, { useState, useEffect } from 'react';
import { Card, Row, Col, Statistic, Progress, Button, Upload, message, Table, Tag, List, Typography, Space } from 'antd';
import { CheckCircleOutlined, WarningOutlined, CloseCircleOutlined, UploadOutlined, FileExcelOutlined } from '@ant-design/icons';
import { api } from '../../api/client';

const { Title, Text } = Typography;

interface QualityReport {
  dataset_id: number;
  total_rows: number;
  total_columns: number;
  null_quality_score: number;
  duplicate_quality_score: number;
  format_quality_score: number;
  overall_score: number;
  issues: string[];
  recommendations: string[];
}

export default function DataQuality() {
  const [loading, setLoading] = useState(false);
  const [report, setReport] = useState<QualityReport | null>(null);
  const [fileList, setFileList] = useState<any[]>([]);

  const handleFileUpload = async (file: any) => {
    setLoading(true);
    const formData = new FormData();
    formData.append('file', file);
    formData.append('dataset_id', '0'); // 临时
    
    try {
      const res = await api.quality.checkFile(formData);
      setReport(res);
      message.success('质量检查完成');
    } catch (error) {
      message.error('检查失败');
    } finally {
      setLoading(false);
    }
    return false; // 阻止默认上传
  };

  const getScoreColor = (score: number) => {
    if (score >= 90) return '#52c41a';
    if (score >= 70) return '#faad14';
    return '#ff4d4f';
  };

  const getScoreIcon = (score: number) => {
    if (score >= 90) return <CheckCircleOutlined style={{ color: '#52c41a' }} />;
    if (score >= 70) return <WarningOutlined style={{ color: '#faad14' }} />;
    return <CloseCircleOutlined style={{ color: '#ff4d4f' }} />;
  };

  return (
    <div className="data-quality">
      <Title level={3}>🧪 数据质量检查</Title>
      
      <Row gutter={[16, 16]}>
        {/* 上传区域 */}
        <Col span={24}>
          <Card title="上传数据集检查质量">
            <Upload.Dragger
              name="file"
              beforeUpload={handleFileUpload}
              fileList={fileList}
              onChange={({ fileList }) => setFileList(fileList)}
              accept=".csv,.json,.jsonl"
            >
              <p className="ant-upload-drag-icon">
                <FileExcelOutlined />
              </p>
              <p className="ant-upload-text">点击或拖拽文件到此区域上传</p>
              <p className="ant-upload-hint">支持 CSV, JSON, JSONL 格式</p>
            </Upload.Dragger>
          </Card>
        </Col>

        {/* 质量评分 */}
        {report && (
          <>
            <Col span={24}>
              <Card title="📊 质量评分">
                <Row gutter={16}>
                  <Col span={6}>
                    <Statistic
                      title="总体评分"
                      value={report.overall_score.toFixed(1)}
                      prefix={getScoreIcon(report.overall_score)}
                      suffix="/ 100"
                    />
                    <Progress 
                      percent={report.overall_score} 
                      showInfo={false}
                      strokeColor={getScoreColor(report.overall_score)}
                    />
                  </Col>
                  <Col span={6}>
                    <Statistic
                      title="空值检测"
                      value={report.null_quality_score.toFixed(1)}
                      prefix={getScoreIcon(report.null_quality_score)}
                      suffix="/ 100"
                    />
                    <Progress 
                      percent={report.null_quality_score} 
                      showInfo={false}
                      strokeColor={getScoreColor(report.null_quality_score)}
                    />
                  </Col>
                  <Col span={6}>
                    <Statistic
                      title="重复检测"
                      value={report.duplicate_quality_score.toFixed(1)}
                      prefix={getScoreIcon(report.duplicate_quality_score)}
                      suffix="/ 100"
                    />
                    <Progress 
                      percent={report.duplicate_quality_score} 
                      showInfo={false}
                      strokeColor={getScoreColor(report.duplicate_quality_score)}
                    />
                  </Col>
                  <Col span={6}>
                    <Statistic
                      title="格式检测"
                      value={report.format_quality_score.toFixed(1)}
                      prefix={getScoreIcon(report.format_quality_score)}
                      suffix="/ 100"
                    />
                    <Progress 
                      percent={report.format_quality_score} 
                      showInfo={false}
                      strokeColor={getScoreColor(report.format_quality_score)}
                    />
                  </Col>
                </Row>
              </Card>
            </Col>

            {/* 基础统计 */}
            <Col span={12}>
              <Card title="📈 数据统计">
                <Row gutter={16}>
                  <Col span={12}>
                    <Statistic title="总行数" value={report.total_rows} />
                  </Col>
                  <Col span={12}>
                    <Statistic title="总列数" value={report.total_columns} />
                  </Col>
                </Row>
              </Card>
            </Col>

            {/* 问题和建议 */}
            <Col span={12}>
              <Card title="💡 分析结果">
                <List
                  size="small"
                  dataSource={report.issues}
                  renderItem={(item, index) => (
                    <List.Item>
                      <Tag color="red" key={index}>{item}</Tag>
                    </List.Item>
                  )}
                />
                <List
                  size="small"
                  dataSource={report.recommendations}
                  renderItem={(item, index) => (
                    <List.Item>
                      <Tag color="blue" key={index}>{item}</Tag>
                    </List.Item>
                  )}
                />
              </Card>
            </Col>
          </>
        )}
      </Row>
    </div>
  );
}
