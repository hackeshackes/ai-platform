/**
 * HybridCompute - Quantum AI Page
 * 混合计算 - 量子AI页面
 */

import React, { useState, useCallback } from 'react';
import {
  Cpu, Zap, RefreshCw, Settings, Play, Activity,
  Layers, Globe, Server, Database, Network, ChevronRight,
  Gauge, TrendingUp, Clock, CheckCircle,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Progress } from '@/components/ui/progress';
import { ScrollArea } from '@/components/ui/scroll-area';
import { useToast } from '@/hooks/use-toast';
import { cn } from '@/lib/utils';

interface ComputeTask {
  id: string; name: string; status: 'queued' | 'running' | 'completed';
  computeType: 'classical' | 'quantum' | 'hybrid';
  progress: number; timeEstimate: string; resources: { classical: number; quantum: number };
}

const SAMPLE_TASKS: ComputeTask[] = [
  { id: '1', name: 'Large Language Model Inference', status: 'completed', computeType: 'classical', progress: 100, timeEstimate: '2.3s', resources: { classical: 100, quantum: 0 } },
  { id: '2', name: 'Quantum Chemistry Simulation', status: 'running', computeType: 'quantum', progress: 67, timeEstimate: '5m 30s', resources: { classical: 20, quantum: 80 } },
  { id: '3', name: 'Hybrid Optimization Loop', status: 'running', computeType: 'hybrid', progress: 45, timeEstimate: '12m', resources: { classical: 50, quantum: 50 } },
  { id: '4', name: 'Monte Carlo Simulation', status: 'queued', computeType: 'classical', progress: 0, timeEstimate: '8m', resources: { classical: 100, quantum: 0 } },
  { id: '5', name: 'QAOA Parameter Optimization', status: 'completed', computeType: 'hybrid', progress: 100, timeEstimate: '15m', resources: { classical: 30, quantum: 70 } },
];

const styles = {
  container: 'min-h-screen bg-gradient-to-br from-slate-50 via-white to-gray-50 dark:from-gray-900 dark:via-gray-800 dark:to-gray-900 p-6',
  header: 'flex items-center justify-between mb-8',
  title: 'text-4xl font-bold bg-gradient-to-r from-slate-600 to-gray-600 bg-clip-text text-transparent',
  subtitle: 'text-gray-600 dark:text-gray-300 mt-2',
  mainGrid: 'grid grid-cols-1 lg:grid-cols-3 gap-6',
  content: 'lg:col-span-2',
  card: 'bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm shadow-xl rounded-2xl border-0',
  taskCard: 'bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm shadow-xl rounded-2xl border-0 hover:shadow-2xl transition-all',
  button: 'bg-gradient-to-r from-slate-600 to-gray-600 hover:from-slate-700 hover:to-gray-700 text-white shadow-lg',
};

export default function HybridCompute() {
  const { toast } = useToast();
  const [tasks, setTasks] = useState<ComputeTask[]>(SAMPLE_TASKS);
  const [isProcessing, setIsProcessing] = useState(false);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'running': return 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400';
      case 'completed': return 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400';
      case 'queued': return 'bg-gray-100 text-gray-700 dark:bg-gray-900/30 dark:text-gray-400';
      default: return 'bg-gray-100 text-gray-700';
    }
  };

  const getTypeColor = (type: string) => {
    switch (type) {
      case 'quantum': return 'bg-purple-100 text-purple-700 dark:bg-purple-900/30 dark:text-purple-400';
      case 'hybrid': return 'bg-cyan-100 text-cyan-700 dark:bg-cyan-900/30 dark:text-cyan-400';
      case 'classical': return 'bg-slate-100 text-slate-700 dark:bg-slate-900/30 dark:text-slate-400';
      default: return 'bg-gray-100 text-gray-700';
    }
  };

  const runningTasks = tasks.filter(t => t.status === 'running');
  const completedTasks = tasks.filter(t => t.status === 'completed');
  const avgQuantumUsage = Math.round(tasks.reduce((a, t) => a + t.resources.quantum, 0) / tasks.length);

  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <div>
          <h1 className={styles.title}><Cpu className="inline mr-2 h-8 w-8" />Hybrid Compute</h1>
          <p className={styles.subtitle}>Seamlessly blend classical and quantum computing resources</p>
        </div>
        <div className="flex items-center gap-3">
          <Button variant="outline"><Settings className="w-4 h-4 mr-2" />Configure</Button>
          <Button className={styles.button} disabled={isProcessing}>
            <Zap className="w-4 h-4 mr-2" />New Task
          </Button>
        </div>
      </div>

      <div className={styles.mainGrid}>
        <div className={styles.content}>
          <Tabs defaultValue="tasks" className="space-y-6">
            <TabsList className="bg-white/60 dark:bg-gray-800/60 backdrop-blur-sm p-1 rounded-xl">
              <TabsTrigger value="tasks" className="px-4 py-2 rounded-lg data-[state=active]:bg-slate-100 data-[state=active]:text-slate-700">
                <Activity className="w-4 h-4 mr-2" />Tasks
              </TabsTrigger>
              <TabsTrigger value="resources" className="px-4 py-2 rounded-lg data-[state=active]:bg-slate-100 data-[state=active]:text-slate-700">
                <Layers className="w-4 h-4 mr-2" />Resources
              </TabsTrigger>
              <TabsTrigger value="scheduler" className="px-4 py-2 rounded-lg data-[state=active]:bg-slate-100 data-[state=active]:text-slate-700">
                <Clock className="w-4 h-4 mr-2" />Scheduler
              </TabsTrigger>
            </TabsList>

            <TabsContent value="tasks">
              <div className="grid gap-4">
                {tasks.map((task) => (
                  <Card key={task.id} className={styles.taskCard}>
                    <CardContent className="p-4">
                      <div className="flex items-center justify-between mb-3">
                        <div className="flex items-center gap-3">
                          <div className={cn('w-10 h-10 rounded-xl flex items-center justify-center', getTypeColor(task.computeType))}>
                            {task.computeType === 'quantum' ? <Zap className="w-5 h-5" /> : task.computeType === 'hybrid' ? <Layers className="w-5 h-5" /> : <Server className="w-5 h-5" />}
                          </div>
                          <div>
                            <h3 className="font-semibold">{task.name}</h3>
                            <p className="text-sm text-gray-500">Est. {task.timeEstimate}</p>
                          </div>
                        </div>
                        <div className="flex items-center gap-2">
                          <Badge className={getTypeColor(task.computeType)}>{task.computeType}</Badge>
                          <Badge className={getStatusColor(task.status)}>{task.status}</Badge>
                        </div>
                      </div>
                      <div className="flex items-center gap-2 mb-3">
                        <span className="text-xs text-gray-500 w-20">Classical</span>
                        <Progress value={task.resources.classical} className="flex-1 h-1.5" />
                        <span className="text-xs w-8 text-right">{task.resources.classical}%</span>
                      </div>
                      <div className="flex items-center gap-2 mb-3">
                        <span className="text-xs text-gray-500 w-20">Quantum</span>
                        <Progress value={task.resources.quantum} className="flex-1 h-1.5" />
                        <span className="text-xs w-8 text-right">{task.resources.quantum}%</span>
                      </div>
                      <Progress value={task.progress} className="h-2" />
                    </CardContent>
                  </Card>
                ))}
              </div>
            </TabsContent>

            <TabsContent value="resources">
              <Card className={styles.card}>
                <CardHeader><CardTitle>Compute Resource Allocation</CardTitle></CardHeader>
                <CardContent>
                  <div className="grid grid-cols-2 gap-4">
                    <div className="p-4 bg-slate-100 dark:bg-slate-900/50 rounded-lg">
                      <div className="flex items-center gap-2 mb-2">
                        <Server className="w-5 h-5 text-slate-600" />
                        <span className="font-medium">Classical</span>
                      </div>
                      <Progress value={75} className="h-2 mb-2" />
                      <p className="text-sm text-gray-500">75% utilized</p>
                    </div>
                    <div className="p-4 bg-purple-100 dark:bg-purple-900/30 rounded-lg">
                      <div className="flex items-center gap-2 mb-2">
                        <Zap className="w-5 h-5 text-purple-600" />
                        <span className="font-medium">Quantum</span>
                      </div>
                      <Progress value={avgQuantumUsage} className="h-2 mb-2" />
                      <p className="text-sm text-gray-500">{avgQuantumUsage}% utilized</p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="scheduler">
              <Card className={styles.card}>
                <CardContent className="p-8 text-center">
                  <Clock className="w-16 h-16 mx-auto mb-4 text-slate-300" />
                  <h3 className="text-xl font-semibold mb-2">Task Scheduler</h3>
                  <p className="text-gray-500">Intelligent task distribution across compute types</p>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </div>

        <div className="space-y-6">
          <Card className={styles.card}>
            <CardHeader><CardTitle className="flex items-center gap-2"><Activity className="w-5 h-5 text-slate-500" />System Status</CardTitle></CardHeader>
            <CardContent className="space-y-4">
              <div className="flex justify-between items-center"><span className="text-sm text-gray-500">Running</span><span className="font-semibold">{runningTasks.length}</span></div>
              <div className="flex justify-between items-center"><span className="text-sm text-gray-500">Completed</span><span className="font-semibold text-green-500">{completedTasks.length}</span></div>
              <div className="flex justify-between items-center"><span className="text-sm text-gray-500">Queued</span><span className="font-semibold">{tasks.filter(t => t.status === 'queued').length}</span></div>
            </CardContent>
          </Card>

          <Card className={styles.card}>
            <CardHeader><CardTitle className="flex items-center gap-2"><Gauge className="w-5 h-5 text-amber-500" />Efficiency Score</CardTitle></CardHeader>
            <CardContent className="text-center">
              <div className="relative inline-block">
                <svg className="w-24 h-24 transform -rotate-90">
                  <circle cx="48" cy="48" r="40" stroke="#e5e7eb" strokeWidth="8" fill="none" />
                  <circle cx="48" cy="48" r="40" stroke="#64748b" strokeWidth="8" fill="none" strokeDasharray="251" strokeDashoffset="40" />
                </svg>
                <span className="absolute inset-0 flex items-center justify-center text-2xl font-bold">84</span>
              </div>
              <p className="mt-2 text-sm text-gray-500">Hybrid Efficiency</p>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
