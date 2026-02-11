/**
 * AIOps Dashboard - Hyperautomation Page
 * AIOps仪表盘 - 超自动化页面
 */

import React, { useState, useCallback } from 'react';
import {
  Activity, Server, Database, Cloud, Shield, Zap, AlertTriangle,
  CheckCircle, Clock, TrendingUp, TrendingDown, Cpu, Memory,
  HardDrive, Network, Terminal, Settings, RefreshCw, Play, Pause,
  Bell, MessageSquare, FileText, Users, Globe, Globe2,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Switch } from '@/components/ui/switch';
import { Progress } from '@/components/ui/progress';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Separator } from '@/components/ui/separator';
import { useToast } from '@/hooks/use-toast';
import { cn } from '@/lib/utils';

interface Metric {
  id: string; name: string; value: number; unit: string;
  change: number; trend: 'up' | 'down' | 'stable';
  status: 'healthy' | 'warning' | 'critical';
}

interface Alert {
  id: string; title: string; severity: 'low' | 'medium' | 'high' | 'critical';
  source: string; timestamp: Date; acknowledged: boolean;
}

interface Service {
  id: string; name: string; status: 'running' | 'stopped' | 'degraded';
  uptime: string; requests: number; latency: number; errorRate: number;
}

const SAMPLE_METRICS: Metric[] = [
  { id: '1', name: 'CPU Usage', value: 45, unit: '%', change: -5, trend: 'down', status: 'healthy' },
  { id: '2', name: 'Memory Usage', value: 68, unit: '%', change: 3, trend: 'up', status: 'warning' },
  { id: '3', name: 'Disk I/O', value: 32, unit: '%', change: -2, trend: 'stable', status: 'healthy' },
  { id: '4', name: 'Network In', value: 1250, unit: 'MB/s', change: 15, trend: 'up', status: 'healthy' },
  { id: '5', name: 'Request Latency', value: 145, unit: 'ms', change: 10, trend: 'up', status: 'warning' },
];

const SAMPLE_ALERTS: Alert[] = [
  { id: '1', title: 'High memory usage detected', severity: 'high', source: 'api-server-01', timestamp: new Date(), acknowledged: false },
  { id: '2', title: 'Database connection pool near capacity', severity: 'medium', source: 'db-primary', timestamp: new Date(Date.now() - 300000), acknowledged: false },
  { id: '3', title: 'SSL certificate expires in 7 days', severity: 'low', source: 'gateway', timestamp: new Date(Date.now() - 600000), acknowledged: true },
];

const SAMPLE_SERVICES: Service[] = [
  { id: '1', name: 'API Gateway', status: 'running', uptime: '99.99%', requests: 125000, latency: 45, errorRate: 0.01 },
  { id: '2', name: 'Auth Service', status: 'running', uptime: '99.95%', requests: 45000, latency: 32, errorRate: 0.05 },
  { id: '3', name: 'Data Processing', status: 'degraded', uptime: '98.50%', requests: 12000, latency: 234, errorRate: 2.1 },
  { id: '4', name: 'ML Inference', status: 'running', uptime: '99.90%', requests: 5600, latency: 156, errorRate: 0.3 },
];

const styles = {
  container: 'min-h-screen bg-gradient-to-br from-slate-50 via-white to-gray-50 dark:from-gray-900 dark:via-gray-800 dark:to-gray-900 p-6',
  header: 'flex items-center justify-between mb-8',
  title: 'text-4xl font-bold bg-gradient-to-r from-slate-600 to-gray-600 bg-clip-text text-transparent',
  subtitle: 'text-gray-600 dark:text-gray-300 mt-2',
  mainGrid: 'grid grid-cols-1 lg:grid-cols-4 gap-6',
  content: 'lg:col-span-3',
  card: 'bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm shadow-xl rounded-2xl border-0',
  metricCard: 'bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm shadow-xl rounded-2xl border-0 hover:shadow-2xl transition-all',
  button: 'bg-gradient-to-r from-slate-600 to-gray-600 hover:from-slate-700 hover:to-gray-700 text-white shadow-lg',
};

export default function AIOpsDashboard() {
  const { toast } = useToast();
  const [metrics] = useState<Metric[]>(SAMPLE_METRICS);
  const [alerts, setAlerts] = useState<Alert[]>(SAMPLE_ALERTS);
  const [services] = useState<Service[]>(SAMPLE_SERVICES);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [selectedTimeRange, setSelectedTimeRange] = useState('1h');

  const refreshData = useCallback(async () => {
    setIsRefreshing(true);
    try { await new Promise(resolve => setTimeout(resolve, 1000)); } finally { setIsRefreshing(false); }
  }, []);

  const handleAcknowledgeAlert = useCallback((alertId: string) => {
    setAlerts(prev => prev.map(a => a.id === alertId ? { ...a, acknowledged: true } : a));
    toast({ title: 'Alert Acknowledged' });
  }, [toast]);

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical': return 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400';
      case 'high': return 'bg-orange-100 text-orange-700 dark:bg-orange-900/30 dark:text-orange-400';
      case 'medium': return 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400';
      case 'low': return 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400';
      default: return 'bg-gray-100 text-gray-700';
    }
  };

  const criticalAlerts = alerts.filter(a => a.severity === 'critical' && !a.acknowledged);

  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <div>
          <h1 className={styles.title}><Activity className="inline mr-2 h-8 w-8" />AIOps Dashboard</h1>
          <p className={styles.subtitle}>Monitor and manage your AI infrastructure with intelligent automation</p>
        </div>
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <Switch checked={autoRefresh} onCheckedChange={setAutoRefresh} />
            <span className="text-sm text-gray-500">Auto-refresh</span>
          </div>
          <Select value={selectedTimeRange} onValueChange={setSelectedTimeRange}>
            <SelectTrigger className="w-[120px]"><SelectValue /></SelectTrigger>
            <SelectContent>
              <SelectItem value="15m">Last 15 min</SelectItem>
              <SelectItem value="1h">Last 1 hour</SelectItem>
              <SelectItem value="24h">Last 24 hours</SelectItem>
            </SelectContent>
          </Select>
          <Button variant="outline" onClick={refreshData} disabled={isRefreshing}>
            <RefreshCw className={`w-4 h-4 mr-2 ${isRefreshing ? 'animate-spin' : ''}`} />Refresh
          </Button>
          <Button className={styles.button}><Settings className="w-4 h-4 mr-2" />Settings</Button>
        </div>
      </div>

      {criticalAlerts.length > 0 && (
        <div className="mb-6 p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-xl flex items-center justify-between">
          <div className="flex items-center gap-3">
            <AlertTriangle className="w-6 h-6 text-red-500" />
            <div>
              <p className="font-semibold text-red-700 dark:text-red-400">{criticalAlerts.length} Critical Alert{criticalAlerts.length > 1 ? 's' : ''}</p>
              <p className="text-sm text-red-600 dark:text-red-300">{criticalAlerts[0].title}</p>
            </div>
          </div>
          <Button variant="destructive" size="sm" onClick={() => handleAcknowledgeAlert(criticalAlerts[0].id)}>Acknowledge</Button>
        </div>
      )}

      <div className={styles.mainGrid}>
        <div className={styles.content}>
          <Tabs defaultValue="overview" className="space-y-6">
            <TabsList className="bg-white/60 dark:bg-gray-800/60 backdrop-blur-sm p-1 rounded-xl">
              <TabsTrigger value="overview" className="px-4 py-2 rounded-lg data-[state=active]:bg-slate-100 data-[state=active]:text-slate-700">
                <Activity className="w-4 h-4 mr-2" />Overview
              </TabsTrigger>
              <TabsTrigger value="infrastructure" className="px-4 py-2 rounded-lg data-[state=active]:bg-slate-100 data-[state=active]:text-slate-700">
                <Server className="w-4 h-4 mr-2" />Infrastructure
              </TabsTrigger>
              <TabsTrigger value="services" className="px-4 py-2 rounded-lg data-[state=active]:bg-slate-100 data-[state=active]:text-slate-700">
                <Globe className="w-4 h-4 mr-2" />Services
              </TabsTrigger>
            </TabsList>

            <TabsContent value="overview">
              <div className="grid grid-cols-2 md:grid-cols-3 gap-4 mb-6">
                {metrics.map((metric) => (
                  <Card key={metric.id} className={styles.metricCard}>
                    <CardContent className="p-4">
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-sm text-gray-500">{metric.name}</span>
                        <Badge className={cn(metric.status === 'healthy' ? 'bg-green-100 text-green-700' : 'bg-yellow-100 text-yellow-700', 'text-xs')}>
                          {metric.status}
                        </Badge>
                      </div>
                      <div className="flex items-end justify-between">
                        <span className="text-2xl font-bold">{metric.value}{metric.unit}</span>
                        <div className={cn('flex items-center gap-1 text-sm', metric.trend === 'up' ? 'text-green-500' : metric.trend === 'down' ? 'text-red-500' : 'text-gray-400')}>
                          {metric.trend === 'up' ? <TrendingUp className="w-4 h-4" /> : metric.trend === 'down' ? <TrendingDown className="w-4 h-4" /> : null}
                          {metric.change > 0 ? '+' : ''}{metric.change}%
                        </div>
                      </div>
                      <Progress value={metric.value} className="mt-2 h-1.5" />
                    </CardContent>
                  </Card>
                ))}
              </div>

              <Card className={styles.card}>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2"><Globe2 className="w-5 h-5 text-slate-500" />Services Status</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    {services.map((service) => (
                      <div key={service.id} className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-900/50 rounded-lg">
                        <div className="flex items-center gap-3">
                          <div className={cn('w-2 h-2 rounded-full', service.status === 'running' ? 'bg-green-500' : service.status === 'degraded' ? 'bg-yellow-500' : 'bg-red-500')} />
                          <div>
                            <p className="font-medium">{service.name}</p>
                            <p className="text-sm text-gray-500">Uptime: {service.uptime}</p>
                          </div>
                        </div>
                        <div className="flex items-center gap-4 text-sm text-gray-500">
                          <span>{service.requests.toLocaleString()} reqs</span>
                          <span>{service.latency}ms</span>
                          <span className={service.errorRate > 1 ? 'text-red-500' : 'text-green-500'}>{service.errorRate}%</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="infrastructure">
              <Card className={styles.card}>
                <CardContent className="p-8 text-center">
                  <Server className="w-16 h-16 mx-auto mb-4 text-slate-300" />
                  <h3 className="text-xl font-semibold mb-2">Infrastructure Overview</h3>
                  <p className="text-gray-500">View detailed infrastructure metrics</p>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="services">
              <Card className={styles.card}>
                <CardContent className="p-8 text-center">
                  <Globe className="w-16 h-16 mx-auto mb-4 text-slate-300" />
                  <h3 className="text-xl font-semibold mb-2">Services Management</h3>
                  <p className="text-gray-500">Manage your deployed services</p>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </div>

        <div className="space-y-6">
          <Card className={styles.card}>
            <CardHeader>
              <CardTitle className="flex items-center gap-2"><Bell className="w-5 h-5 text-amber-500" />Recent Alerts</CardTitle>
            </CardHeader>
            <CardContent>
              <ScrollArea className="h-[300px]">
                <div className="space-y-3">
                  {alerts.map((alert) => (
                    <div key={alert.id} className={cn('p-3 rounded-lg', alert.acknowledged ? 'bg-gray-50 dark:bg-gray-900/50' : 'bg-amber-50 dark:bg-amber-900/20')}>
                      <div className="flex items-start justify-between">
                        <div>
                          <Badge className={cn('text-xs mb-1', getSeverityColor(alert.severity))}>{alert.severity}</Badge>
                          <p className="text-sm font-medium">{alert.title}</p>
                          <p className="text-xs text-gray-500">{alert.source} • {new Date(alert.timestamp).toLocaleTimeString()}</p>
                        </div>
                        {!alert.acknowledged && <Button variant="ghost" size="sm" onClick={() => handleAcknowledgeAlert(alert.id)}><CheckCircle className="w-4 h-4" /></Button>}
                      </div>
                    </div>
                  ))}
                </div>
              </ScrollArea>
            </CardContent>
          </Card>

          <Card className={styles.card}>
            <CardHeader>
              <CardTitle className="flex items-center gap-2"><Shield className="w-5 h-5 text-green-500" />System Health</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-500">Overall Status</span>
                <Badge className="bg-green-100 text-green-700">Healthy</Badge>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-500">Active Services</span>
                <span className="font-semibold">{services.filter(s => s.status === 'running').length}/{services.length}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-500">Pending Alerts</span>
                <span className="font-semibold text-amber-500">{alerts.filter(a => !a.acknowledged).length}</span>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
