/**
 * Self-Healing - Hyperautomation Page
 * 自愈系统 - 超自动化页面
 */

import React, { useState, useCallback, useEffect } from 'react';
import {
  Activity, Heart, Shield, AlertTriangle, CheckCircle, XCircle,
  Zap, RefreshCw, Settings, Cpu, Memory, HardDrive, Network,
  Terminal, Clock, TrendingUp, TrendingDown, Play, Pause,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Switch } from '@/components/ui/switch';
import { Progress } from '@/components/ui/progress';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Separator } from '@/components/ui/separator';
import { useToast } from '@/hooks/use-toast';
import { cn } from '@/lib/utils';

interface Incident {
  id: string;
  title: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  status: 'detected' | 'investigating' | 'auto-fixing' | 'resolved' | 'failed';
  component: string;
  timestamp: Date;
  metrics: { name: string; value: number; threshold: number }[];
  actions: { action: string; timestamp: Date; result: 'success' | 'failed' }[];
}

interface HealthMetric {
  id: string; name: string; value: number; unit: string;
  status: 'healthy' | 'degraded' | 'critical';
  trend: 'up' | 'down' | 'stable';
  threshold: { warning: number; critical: number };
}

const SAMPLE_INCIDENTS: Incident[] = [
  { id: '1', title: 'API Response Time Spike', severity: 'medium', status: 'auto-fixing', component: 'API Gateway', timestamp: new Date(), metrics: [{ name: 'Latency', value: 450, threshold: 200 }], actions: [{ action: 'Restarting service', timestamp: new Date(), result: 'success' }] },
  { id: '2', title: 'Memory Leak Detected', severity: 'high', status: 'investigating', component: 'ML Processor', timestamp: new Date(Date.now() - 300000), metrics: [{ name: 'Memory', value: 92, threshold: 80 }], actions: [] },
  { id: '3', title: 'Database Connection Pool Exhausted', severity: 'critical', status: 'resolved', component: 'Database', timestamp: new Date(Date.now() - 600000), metrics: [{ name: 'Connections', value: 100, threshold: 100 }], actions: [{ action: 'Connection pool reset', timestamp: new Date(Date.now() - 600000), result: 'success' }] },
];

const SAMPLE_METRICS: HealthMetric[] = [
  { id: '1', name: 'CPU Usage', value: 45, unit: '%', status: 'healthy', trend: 'down', threshold: { warning: 70, critical: 90 } },
  { id: '2', name: 'Memory Usage', value: 78, unit: '%', status: 'degraded', trend: 'up', threshold: { warning: 70, critical: 90 } },
  { id: '3', name: 'Disk Usage', value: 65, unit: '%', status: 'healthy', trend: 'stable', threshold: { warning: 80, critical: 95 } },
  { id: '4', name: 'Network Latency', value: 25, unit: 'ms', status: 'healthy', trend: 'down', threshold: { warning: 100, critical: 200 } },
  { id: '5', name: 'Error Rate', value: 0.5, unit: '%', status: 'healthy', trend: 'stable', threshold: { warning: 2, critical: 5 } },
];

const styles = {
  container: 'min-h-screen bg-gradient-to-br from-red-50 via-white to-orange-50 dark:from-gray-900 dark:via-gray-800 dark:to-gray-900 p-6',
  header: 'flex items-center justify-between mb-8',
  title: 'text-4xl font-bold bg-gradient-to-r from-red-600 to-orange-600 bg-clip-text text-transparent',
  subtitle: 'text-gray-600 dark:text-gray-300 mt-2',
  mainGrid: 'grid grid-cols-1 lg:grid-cols-3 gap-6',
  content: 'lg:col-span-2',
  card: 'bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm shadow-xl rounded-2xl border-0',
  incidentCard: 'bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm shadow-xl rounded-2xl border-0 hover:shadow-2xl transition-all',
  button: 'bg-gradient-to-r from-red-600 to-orange-600 hover:from-red-700 hover:to-orange-700 text-white shadow-lg',
};

export default function SelfHealing() {
  const { toast } = useToast();
  const [incidents, setIncidents] = useState<Incident[]>(SAMPLE_INCIDENTS);
  const [metrics] = useState<HealthMetric[]>(SAMPLE_METRICS);
  const [autoHeal, setAutoHeal] = useState(true);
  const [isHealing, setIsHealing] = useState(false);

  const handleHeal = useCallback(async (incidentId: string) => {
    setIncidents(prev => prev.map(i => i.id === incidentId ? { ...i, status: 'auto-fixing' } : i));
    setIsHealing(true);
    try {
      await new Promise(resolve => setTimeout(resolve, 3000));
      setIncidents(prev => prev.map(i => i.id === incidentId ? { ...i, status: 'resolved' } : i));
      toast({ title: 'Incident Resolved', description: 'Self-healing action completed successfully' });
    } catch (error) {
      setIncidents(prev => prev.map(i => i.id === incidentId ? { ...i, status: 'failed' } : i));
      toast({ title: 'Healing Failed', variant: 'destructive' });
    } finally {
      setIsHealing(false);
    }
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

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'detected': return <AlertTriangle className="w-4 h-4 text-red-500" />;
      case 'investigating': return <Activity className="w-4 h-4 text-yellow-500 animate-pulse" />;
      case 'auto-fixing': return <RefreshCw className="w-4 h-4 text-blue-500 animate-spin" />;
      case 'resolved': return <CheckCircle className="w-4 h-4 text-green-500" />;
      case 'failed': return <XCircle className="w-4 h-4 text-red-500" />;
      default: return null;
    }
  };

  const activeIncidents = incidents.filter(i => !['resolved', 'failed'].includes(i.status));
  const resolvedToday = incidents.filter(i => i.status === 'resolved').length;

  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <div>
          <h1 className={styles.title}><Heart className="inline mr-2 h-8 w-8" />Self-Healing System</h1>
          <p className={styles.subtitle}>AI-powered automatic detection and recovery from system issues</p>
        </div>
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <Switch checked={autoHeal} onCheckedChange={setAutoHeal} />
            <span className="text-sm text-gray-500">Auto-Heal</span>
          </div>
          <Button variant="outline"><Settings className="w-4 h-4 mr-2" />Configure</Button>
          <Button className={styles.button}><Zap className="w-4 h-4 mr-2" />Run Diagnostics</Button>
        </div>
      </div>

      {activeIncidents.length > 0 && (
        <div className="mb-6 p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-xl">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <AlertTriangle className="w-6 h-6 text-red-500" />
              <div>
                <p className="font-semibold text-red-700 dark:text-red-400">{activeIncidents.length} Active Incident{activeIncidents.length > 1 ? 's' : ''}</p>
                <p className="text-sm text-red-600 dark:text-red-300">{activeIncidents[0].title}</p>
              </div>
            </div>
            <Button variant="destructive" size="sm" onClick={() => handleHeal(activeIncidents[0].id)} disabled={isHealing}>
              {isHealing ? <RefreshCw className="w-4 h-4 mr-2 animate-spin" /> : <Zap className="w-4 h-4 mr-2" />}
              Auto-Heal Now
            </Button>
          </div>
        </div>
      )}

      <div className={styles.mainGrid}>
        <div className={styles.content}>
          <Tabs defaultValue="incidents" className="space-y-6">
            <TabsList className="bg-white/60 dark:bg-gray-800/60 backdrop-blur-sm p-1 rounded-xl">
              <TabsTrigger value="incidents" className="px-4 py-2 rounded-lg data-[state=active]:bg-red-100 data-[state=active]:text-red-700">
                <AlertTriangle className="w-4 h-4 mr-2" />Incidents
              </TabsTrigger>
              <TabsTrigger value="health" className="px-4 py-2 rounded-lg data-[state=active]:bg-red-100 data-[state=active]:text-red-700">
                <Activity className="w-4 h-4 mr-2" />Health
              </TabsTrigger>
              <TabsTrigger value="actions" className="px-4 py-2 rounded-lg data-[state=active]:bg-red-100 data-[state=active]:text-red-700">
                <Terminal className="w-4 h-4 mr-2" />Actions
              </TabsTrigger>
            </TabsList>

            <TabsContent value="incidents">
              <div className="grid gap-4">
                {incidents.map((incident) => (
                  <Card key={incident.id} className={styles.incidentCard}>
                    <CardContent className="p-4">
                      <div className="flex items-start justify-between mb-3">
                        <div className="flex items-start gap-3">
                          {getStatusIcon(incident.status)}
                          <div>
                            <h3 className="font-semibold">{incident.title}</h3>
                            <p className="text-sm text-gray-500">{incident.component} • {new Date(incident.timestamp).toLocaleString()}</p>
                          </div>
                        </div>
                        <div className="flex items-center gap-2">
                          <Badge className={getSeverityColor(incident.severity)}>{incident.severity}</Badge>
                          <Badge variant="outline">{incident.status}</Badge>
                        </div>
                      </div>
                      <div className="grid grid-cols-2 gap-4 mb-3">
                        {incident.metrics.map((m, i) => (
                          <div key={i} className="bg-gray-50 dark:bg-gray-900/50 rounded-lg p-3">
                            <p className="text-sm text-gray-500">{m.name}</p>
                            <div className="flex items-center justify-between mt-1">
                              <span className="text-xl font-bold">{m.value}{m.unit}</span>
                              <span className="text-xs text-gray-400">Threshold: {m.threshold}</span>
                            </div>
                            <Progress value={(m.value / m.threshold) * 100} className="mt-2 h-1.5" />
                          </div>
                        ))}
                      </div>
                      <div className="flex items-center justify-between pt-3 border-t">
                        <span className="text-sm text-gray-500">{incident.actions.length} automatic action{incident.actions.length !== 1 ? 's' : ''} taken</span>
                        {incident.status !== 'resolved' && incident.status !== 'failed' && (
                          <Button size="sm" variant="outline" onClick={() => handleHeal(incident.id)} disabled={isHealing}>
                            <Zap className="w-3 h-3 mr-1" />Trigger Healing
                          </Button>
                        )}
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            </TabsContent>

            <TabsContent value="health">
              <div className="grid grid-cols-2 gap-4">
                {metrics.map((metric) => (
                  <Card key={metric.id} className={styles.card}>
                    <CardContent className="p-4">
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-sm text-gray-500">{metric.name}</span>
                        <Badge className={metric.status === 'healthy' ? 'bg-green-100 text-green-700' : metric.status === 'degraded' ? 'bg-yellow-100 text-yellow-700' : 'bg-red-100 text-red-700'}>
                          {metric.status}
                        </Badge>
                      </div>
                      <div className="flex items-end justify-between">
                        <span className="text-3xl font-bold">{metric.value}{metric.unit}</span>
                        <div className={cn('flex items-center gap-1 text-sm', metric.trend === 'up' ? 'text-red-500' : metric.trend === 'down' ? 'text-green-500' : 'text-gray-400')}>
                          {metric.trend === 'up' ? <TrendingUp className="w-4 h-4" /> : metric.trend === 'down' ? <TrendingDown className="w-4 h-4" /> : null}
                        </div>
                      </div>
                      <Progress value={metric.value} className="mt-2 h-2" />
                    </CardContent>
                  </Card>
                ))}
              </div>
            </TabsContent>

            <TabsContent value="actions">
              <Card className={styles.card}>
                <CardContent className="p-8 text-center">
                  <Terminal className="w-16 h-16 mx-auto mb-4 text-red-300" />
                  <h3 className="text-xl font-semibold mb-2">Automatic Healing Actions</h3>
                  <p className="text-gray-500">View and manage automatic remediation workflows</p>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </div>

        <div className="space-y-6">
          <Card className={styles.card}>
            <CardHeader><CardTitle className="flex items-center gap-2"><Shield className="w-5 h-5 text-green-500" />System Health</CardTitle></CardHeader>
            <CardContent className="space-y-4">
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-500">Overall Status</span>
                <Badge className="bg-green-100 text-green-700">Healthy</Badge>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-500">Active Incidents</span>
                <span className="font-semibold text-red-500">{activeIncidents.length}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-500">Resolved Today</span>
                <span className="font-semibold text-green-500">{resolvedToday}</span>
              </div>
              <Separator />
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-500">Auto-Heal Enabled</span>
                <Switch checked={autoHeal} onCheckedChange={setAutoHeal} />
              </div>
            </CardContent>
          </Card>

          <Card className={styles.card}>
            <CardHeader><CardTitle className="flex items-center gap-2"><Zap className="w-5 h-5 text-amber-500" />Recent Actions</CardTitle></CardHeader>
            <CardContent>
              <ScrollArea className="h-[200px]">
                <div className="space-y-3">
                  {incidents.flatMap(i => i.actions).slice(0, 5).map((action, i) => (
                    <div key={i} className="flex items-center gap-2 text-sm">
                      {action.result === 'success' ? <CheckCircle className="w-4 h-4 text-green-500" /> : <XCircle className="w-4 h-4 text-red-500" />}
                      <span>{action.action}</span>
                      <span className="text-gray-400 ml-auto">{new Date(action.timestamp).toLocaleTimeString()}</span>
                    </div>
                  ))}
                </div>
              </ScrollArea>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
