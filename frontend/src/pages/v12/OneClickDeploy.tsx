/**
 * One-Click Deploy - AI Democratization Page
 * 一键部署 - AI民主化页面
 */

import React, { useState, useCallback } from 'react';
import {
  Rocket, Play, Pause, RefreshCw, Settings, Cloud, Server,
  Database, Globe, Shield, Zap, Clock, CheckCircle, XCircle,
  ChevronRight, Download, Upload, Terminal, Activity, Cpu, HardDrive,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
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

interface Deployment {
  id: string;
  name: string;
  status: 'running' | 'stopped' | 'deploying' | 'error';
  type: 'web' | 'api' | 'worker' | 'database';
  region: string;
  url?: string;
  startedAt: Date;
  resources: { cpu: number; memory: number; storage: number };
  logs: string[];
}

const REGIONS = [
  { id: 'us-east', name: 'US East (N. Virginia)', flag: '🇺🇸' },
  { id: 'us-west', name: 'US West (Oregon)', flag: '🇺🇸' },
  { id: 'eu-west', name: 'EU West (Ireland)', flag: '🇪🇺' },
];

const INSTANCE_TYPES = [
  { id: 'nano', name: 'Nano', cpu: '0.5 vCPU', memory: '0.5 GB', price: '$5/mo' },
  { id: 'small', name: 'Small', cpu: '1 vCPU', memory: '2 GB', price: '$15/mo' },
  { id: 'medium', name: 'Medium', cpu: '2 vCPU', memory: '4 GB', price: '$30/mo' },
];

const SAMPLE_DEPLOYMENTS: Deployment[] = [
  {
    id: '1', name: 'production-api', status: 'running', type: 'api',
    region: 'us-east', url: 'https://api.example.com', startedAt: new Date('2024-02-01'),
    resources: { cpu: 45, memory: 62, storage: 78 },
    logs: ['[2024-02-12 10:00:00] Server started successfully', '[2024-02-12 10:00:01] Connected to database'],
  },
  {
    id: '2', name: 'staging-web', status: 'running', type: 'web',
    region: 'us-west', url: 'https://staging.example.com', startedAt: new Date('2024-02-10'),
    resources: { cpu: 12, memory: 28, storage: 34 },
    logs: ['[2024-02-12 09:00:00] Build completed', '[2024-02-12 09:00:01] Deploying assets'],
  },
];

const styles = {
  container: 'min-h-screen bg-gradient-to-br from-violet-50 via-white to-purple-50 dark:from-gray-900 dark:via-gray-800 dark:to-gray-900 p-6',
  header: 'flex items-center justify-between mb-8',
  title: 'text-4xl font-bold bg-gradient-to-r from-violet-600 to-purple-600 bg-clip-text text-transparent',
  subtitle: 'text-gray-600 dark:text-gray-300 mt-2',
  mainGrid: 'grid grid-cols-1 lg:grid-cols-3 gap-6',
  content: 'lg:col-span-2',
  card: 'bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm shadow-xl rounded-2xl border-0',
  deployCard: 'bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm shadow-xl rounded-2xl border-0 hover:shadow-2xl transition-all cursor-pointer',
  button: 'bg-gradient-to-r from-violet-600 to-purple-600 hover:from-violet-700 hover:to-purple-700 text-white shadow-lg',
  badge: 'px-3 py-1 rounded-full text-xs font-medium',
};

export default function OneClickDeploy() {
  const { toast } = useToast();
  const [deployments, setDeployments] = useState<Deployment[]>(SAMPLE_DEPLOYMENTS);
  const [selectedDeploy, setSelectedDeploy] = useState<Deployment | null>(null);
  const [isDeploying, setIsDeploying] = useState(false);
  const [deployProgress, setDeployProgress] = useState(0);
  const [config, setConfig] = useState({
    name: '', type: 'web' as const, region: 'us-east', instance: 'small',
    autoScale: true, monitoring: true, backup: true,
  });

  const handleDeploy = useCallback(async () => {
    if (!config.name.trim()) {
      toast({ title: 'Warning', description: 'Please enter a deployment name', variant: 'destructive' });
      return;
    }

    setIsDeploying(true);
    setDeployProgress(0);

    try {
      const progressInterval = setInterval(() => {
        setDeployProgress(prev => Math.min(prev + 10, 90));
      }, 500);

      await new Promise(resolve => setTimeout(resolve, 10000));

      const newDeployment: Deployment = {
        id: `deploy-${Date.now()}`,
        name: config.name, status: 'running', type: config.type,
        region: config.region,
        url: `https://${config.name.replace(/\s+/g, '-')}.example.com`,
        startedAt: new Date(),
        resources: { cpu: 10, memory: 15, storage: 20 },
        logs: ['[Deployment started]', '[Container built successfully]', '[Service started]', '[Health check passed]'],
      };

      setDeployments(prev => [newDeployment, ...prev]);
      setSelectedDeploy(newDeployment);
      setDeployProgress(100);
      clearInterval(progressInterval);

      toast({ title: 'Deployment Successful', description: `${config.name} is now running` });
    } catch (error) {
      toast({ title: 'Deployment Failed', description: 'There was an error during deployment', variant: 'destructive' });
    } finally {
      setIsDeploying(false);
      setTimeout(() => setDeployProgress(0), 1000);
    }
  }, [config, toast]);

  const handleStop = useCallback(async (deployId: string) => {
    setDeployments(prev => prev.map(d => d.id === deployId ? { ...d, status: 'stopped' } : d));
    toast({ title: 'Deployment Stopped', description: 'Your deployment has been stopped' });
  }, [toast]);

  const handleRestart = useCallback(async (deployId: string) => {
    setDeployments(prev => prev.map(d => d.id === deployId ? { ...d, status: 'deploying' } : d));
    await new Promise(resolve => setTimeout(resolve, 3000));
    setDeployments(prev => prev.map(d => d.id === deployId ? { ...d, status: 'running' } : d));
    toast({ title: 'Deployment Restarted', description: 'Your deployment is running again' });
  }, [toast]);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'running': return 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400';
      case 'stopped': return 'bg-gray-100 text-gray-700 dark:bg-gray-900/30 dark:text-gray-400';
      case 'deploying': return 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400';
      case 'error': return 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400';
      default: return 'bg-gray-100 text-gray-700';
    }
  };

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'web': return '🌐'; case 'api': return '🔌'; case 'worker': return '⚙️'; case 'database': return '🗄️';
      default: return '📦';
    }
  };

  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <div>
          <h1 className={styles.title}><Rocket className="inline mr-2 h-8 w-8" />One-Click Deploy</h1>
          <p className={styles.subtitle}>Deploy your AI applications to the cloud with a single click</p>
        </div>
        <div className="flex items-center gap-3">
          <Button variant="outline"><Upload className="w-4 h-4 mr-2" />Import Config</Button>
          <Button variant="outline"><Download className="w-4 h-4 mr-2" />Export Config</Button>
          <Button className={styles.button}><Rocket className="w-4 h-4 mr-2" />New Deployment</Button>
        </div>
      </div>

      <div className={styles.mainGrid}>
        <div className={styles.content}>
          <Tabs defaultValue="deployments" className="space-y-6">
            <TabsList className="bg-white/60 dark:bg-gray-800/60 backdrop-blur-sm p-1 rounded-xl">
              <TabsTrigger value="deployments" className="px-4 py-2 rounded-lg data-[state=active]:bg-violet-100 data-[state=active]:text-violet-700">
                <Server className="w-4 h-4 mr-2" />Deployments
              </TabsTrigger>
              <TabsTrigger value="new" className="px-4 py-2 rounded-lg data-[state=active]:bg-violet-100 data-[state=active]:text-violet-700">
                <Zap className="w-4 h-4 mr-2" />New Deploy
              </TabsTrigger>
              <TabsTrigger value="monitoring" className="px-4 py-2 rounded-lg data-[state=active]:bg-violet-100 data-[state=active]:text-violet-700">
                <Activity className="w-4 h-4 mr-2" />Monitoring
              </TabsTrigger>
            </TabsList>

            <TabsContent value="deployments">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {deployments.map((deploy) => (
                  <Card key={deploy.id} className={cn(styles.deployCard, selectedDeploy?.id === deploy.id && 'ring-2 ring-violet-500')} onClick={() => setSelectedDeploy(deploy)}>
                    <CardHeader className="pb-2">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <span className="text-2xl">{getTypeIcon(deploy.type)}</span>
                          <div>
                            <CardTitle className="text-lg">{deploy.name}</CardTitle>
                            <CardDescription>{deploy.region}</CardDescription>
                          </div>
                        </div>
                        <Badge className={cn(getStatusColor(deploy.status), 'flex items-center gap-1')}>
                          {deploy.status === 'running' && <Activity className="w-3 h-3" />}
                          {deploy.status === 'deploying' && <RefreshCw className="w-3 h-3 animate-spin" />}
                          {deploy.status}
                        </Badge>
                      </div>
                    </CardHeader>
                    <CardContent>
                      {deploy.status === 'running' && (
                        <div className="space-y-2 mb-4">
                          <div className="flex items-center gap-2 text-sm">
                            <Cpu className="w-4 h-4 text-gray-400" />
                            <Progress value={deploy.resources.cpu} className="flex-1 h-1.5" />
                            <span className="w-10 text-right">{deploy.resources.cpu}%</span>
                          </div>
                          <div className="flex items-center gap-2 text-sm">
                            <HardDrive className="w-4 h-4 text-gray-400" />
                            <Progress value={deploy.resources.memory} className="flex-1 h-1.5" />
                            <span className="w-10 text-right">{deploy.resources.memory}%</span>
                          </div>
                        </div>
                      )}
                      {deploy.url && <p className="text-sm text-violet-600 truncate">{deploy.url}</p>}
                      <div className="flex gap-2 mt-4">
                        {deploy.status === 'running' ? (
                          <Button variant="outline" size="sm" onClick={(e) => { e.stopPropagation(); handleStop(deploy.id); }}>
                            <Pause className="w-3 h-3 mr-1" />Stop
                          </Button>
                        ) : (
                          <Button variant="outline" size="sm" onClick={(e) => { e.stopPropagation(); handleRestart(deploy.id); }}>
                            <Play className="w-3 h-3 mr-1" />Start
                          </Button>
                        )}
                        <Button variant="ghost" size="sm"><Settings className="w-3 h-3" /></Button>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            </TabsContent>

            <TabsContent value="new">
              <Card className={styles.card}>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2"><Zap className="w-5 h-5 text-violet-500" />Configure Your Deployment</CardTitle>
                  <CardDescription>Set up your deployment configuration</CardDescription>
                </CardHeader>
                <CardContent className="space-y-6">
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className="text-sm font-medium mb-2 block">Deployment Name</label>
                      <Input placeholder="my-app" value={config.name} onChange={(e) => setConfig({ ...config, name: e.target.value })} />
                    </div>
                    <div>
                      <label className="text-sm font-medium mb-2 block">Type</label>
                      <Select value={config.type} onValueChange={(v: any) => setConfig({ ...config, type: v })}>
                        <SelectTrigger><SelectValue /></SelectTrigger>
                        <SelectContent>
                          <SelectItem value="web">Web Application</SelectItem>
                          <SelectItem value="api">API Service</SelectItem>
                          <SelectItem value="worker">Background Worker</SelectItem>
                          <SelectItem value="database">Database</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                  </div>

                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className="text-sm font-medium mb-2 block">Region</label>
                      <Select value={config.region} onValueChange={(v) => setConfig({ ...config, region: v })}>
                        <SelectTrigger><SelectValue /></SelectTrigger>
                        <SelectContent>
                          {REGIONS.map(r => <SelectItem key={r.id} value={r.id}>{r.flag} {r.name}</SelectItem>)}
                        </SelectContent>
                      </Select>
                    </div>
                    <div>
                      <label className="text-sm font-medium mb-2 block">Instance Type</label>
                      <Select value={config.instance} onValueChange={(v) => setConfig({ ...config, instance: v })}>
                        <SelectTrigger><SelectValue /></SelectTrigger>
                        <SelectContent>
                          {INSTANCE_TYPES.map(t => <SelectItem key={t.id} value={t.id}>{t.name} - {t.cpu} / {t.memory} ({t.price})</SelectItem>)}
                        </SelectContent>
                      </Select>
                    </div>
                  </div>

                  <Separator />

                  <div className="space-y-4">
                    <div className="flex items-center justify-between">
                      <span className="text-sm">Auto-scale instances</span>
                      <Switch checked={config.autoScale} onCheckedChange={(v) => setConfig({ ...config, autoScale: v })} />
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-sm">Enable monitoring</span>
                      <Switch checked={config.monitoring} onCheckedChange={(v) => setConfig({ ...config, monitoring: v })} />
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-sm">Daily backups</span>
                      <Switch checked={config.backup} onCheckedChange={(v) => setConfig({ ...config, backup: v })} />
                    </div>
                  </div>

                  {isDeploying ? (
                    <div className="space-y-2">
                      <Progress value={deployProgress} className="h-2" />
                      <p className="text-sm text-gray-500">Deploying... {deployProgress}%</p>
                    </div>
                  ) : (
                    <Button className={styles.button + ' w-full'} size="lg" onClick={handleDeploy}>
                      <Rocket className="w-4 h-4 mr-2" />Deploy Now
                    </Button>
                  )}
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="monitoring">
              <Card className={styles.card}>
                <CardContent className="p-8 text-center">
                  <Activity className="w-16 h-16 mx-auto mb-4 text-violet-300" />
                  <h3 className="text-xl font-semibold mb-2">Monitoring Dashboard</h3>
                  <p className="text-gray-500">View real-time metrics and logs</p>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </div>

        <div className="space-y-6">
          <Card className={styles.card}>
            <CardHeader>
              <CardTitle className="flex items-center gap-2"><Shield className="w-5 h-5 text-green-500" />Quick Stats</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-500">Total Deployments</span>
                <span className="font-semibold">{deployments.length}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-500">Running</span>
                <span className="font-semibold text-green-500">{deployments.filter(d => d.status === 'running').length}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-500">Avg CPU Usage</span>
                <span className="font-semibold">{Math.round(deployments.reduce((a, d) => a + d.resources.cpu, 0) / deployments.length)}%</span>
              </div>
            </CardContent>
          </Card>

          {selectedDeploy && (
            <Card className={styles.card}>
              <CardHeader>
                <CardTitle className="flex items-center gap-2"><Terminal className="w-5 h-5 text-gray-500" />Logs</CardTitle>
              </CardHeader>
              <CardContent>
                <ScrollArea className="h-[200px]">
                  <div className="space-y-1">
                    {selectedDeploy.logs.map((log, i) => (
                      <p key={i} className="text-xs font-mono text-gray-600">{log}</p>
                    ))}
                  </div>
                </ScrollArea>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}
