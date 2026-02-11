/**
 * QuantumOptimizer - Quantum AI Page
 * 量子优化 - 量子AI页面
 */

import React, { useState, useCallback } from 'react';
import {
  Zap, Target, Settings, RefreshCw, Play, Pause, Cpu,
  Activity, TrendingUp, ChevronRight, Layers, Terminal,
  Gauge, Bolt, Clock, CheckCircle,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Progress } from '@/components/ui/progress';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { useToast } from '@/hooks/use-toast';
import { cn } from '@/lib/utils';

interface OptimizationTask {
  id: string; name: string; problem: string; status: 'pending' | 'running' | 'completed';
  progress: number; speedup: number; iterations: number; optimality: number;
}

interface Benchmark {
  name: string; classical: number; quantum: number; speedup: number;
}

const SAMPLE_TASKS: OptimizationTask[] = [
  { id: '1', name: 'Portfolio Optimization', problem: 'Maximize returns with risk constraints', status: 'completed', progress: 100, speedup: 15, iterations: 500, optimality: 0.98 },
  { id: '2', name: 'Supply Chain Routing', problem: 'Minimize delivery costs', status: 'running', progress: 67, speedup: 0, iterations: 234, optimality: 0 },
  { id: '3', name: 'Protein Folding', problem: 'Find minimum energy conformation', status: 'pending', progress: 0, speedup: 0, iterations: 0, optimality: 0 },
  { id: '4', name: 'Traffic Optimization', problem: 'Reduce congestion in network', status: 'completed', progress: 100, speedup: 22, iterations: 750, optimality: 0.95 },
];

const SAMPLE_BENCHMARKS: Benchmark[] = [
  { name: 'MaxCut (n=50)', classical: 1000, quantum: 45, speedup: 22.2 },
  { name: 'TSP (n=20)', classical: 500, quantum: 35, speedup: 14.3 },
  { name: 'Knapsack (n=100)', classical: 200, quantum: 18, speedup: 11.1 },
  { name: 'QAOA (layers=3)', classical: 800, quantum: 52, speedup: 15.4 },
];

const PROBLEM_TYPES = ['Combinatorial', 'Continuous', 'Machine Learning', 'Financial', 'Logistics'];

const styles = {
  container: 'min-h-screen bg-gradient-to-br from-cyan-50 via-white to-blue-50 dark:from-gray-900 dark:via-gray-800 dark:to-gray-900 p-6',
  header: 'flex items-center justify-between mb-8',
  title: 'text-4xl font-bold bg-gradient-to-r from-cyan-600 to-blue-600 bg-clip-text text-transparent',
  subtitle: 'text-gray-600 dark:text-gray-300 mt-2',
  mainGrid: 'grid grid-cols-1 lg:grid-cols-3 gap-6',
  content: 'lg:col-span-2',
  card: 'bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm shadow-xl rounded-2xl border-0',
  taskCard: 'bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm shadow-xl rounded-2xl border-0 hover:shadow-2xl transition-all cursor-pointer',
  button: 'bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-700 hover:to-blue-700 text-white shadow-lg',
};

export default function QuantumOptimizer() {
  const { toast } = useToast();
  const [tasks, setTasks] = useState<OptimizationTask[]>(SAMPLE_TASKS);
  const [benchmarks] = useState<Benchmark[]>(SAMPLE_BENCHMARKS);
  const [isOptimizing, setIsOptimizing] = useState(false);
  const [selectedTask, setSelectedTask] = useState<OptimizationTask | null>(null);
  const [problemType, setProblemType] = useState('Combinatorial');

  const handleOptimize = useCallback(async (taskId: string) => {
    setTasks(prev => prev.map(t => t.id === taskId ? { ...t, status: 'running' } : t));
    setIsOptimizing(true);
    toast({ title: 'Optimization Started', description: 'Running quantum optimization algorithm' });
    
    let progress = 0;
    const interval = setInterval(() => {
      progress += 5;
      setTasks(prev => prev.map(t => t.id === taskId ? { ...t, progress: Math.min(progress, 100) } : t));
      if (progress >= 100) {
        clearInterval(interval);
        setTasks(prev => prev.map(t => t.id === taskId ? { ...t, status: 'completed', speedup: Math.round(Math.random() * 20 + 10), optimality: 0.95 } : t));
        setIsOptimizing(false);
        toast({ title: 'Optimization Complete', description: 'Quantum speedup achieved' });
      }
    }, 250);
  }, [toast]);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'running': return 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400';
      case 'completed': return 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400';
      case 'pending': return 'bg-gray-100 text-gray-700 dark:bg-gray-900/30 dark:text-gray-400';
      default: return 'bg-gray-100 text-gray-700';
    }
  };

  const completedTasks = tasks.filter(t => t.status === 'completed');
  const avgSpeedup = completedTasks.length > 0 ? Math.round(completedTasks.reduce((a, t) => a + t.speedup, 0) / completedTasks.length) : 0;

  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <div>
          <h1 className={styles.title}><Target className="inline mr-2 h-8 w-8" />Quantum Optimizer</h1>
          <p className={styles.subtitle}>Solve complex optimization problems with quantum algorithms</p>
        </div>
        <div className="flex items-center gap-3">
          <Select value={problemType} onValueChange={setProblemType}>
            <SelectTrigger className="w-[180px]"><SelectValue /></SelectTrigger>
            <SelectContent>
              {PROBLEM_TYPES.map(type => <SelectItem key={type} value={type}>{type}</SelectItem>)}
            </SelectContent>
          </Select>
          <Button className={styles.button} disabled={isOptimizing}>
            <Zap className="w-4 h-4 mr-2" />New Problem
          </Button>
        </div>
      </div>

      <div className={styles.mainGrid}>
        <div className={styles.content}>
          <Tabs defaultValue="problems" className="space-y-6">
            <TabsList className="bg-white/60 dark:bg-gray-800/60 backdrop-blur-sm p-1 rounded-xl">
              <TabsTrigger value="problems" className="px-4 py-2 rounded-lg data-[state=active]:bg-cyan-100 data-[state=active]:text-cyan-700">
                <Target className="w-4 h-4 mr-2" />Problems
              </TabsTrigger>
              <TabsTrigger value="benchmarks" className="px-4 py-2 rounded-lg data-[state=active]:bg-cyan-100 data-[state=active]:text-cyan-700">
                <Activity className="w-4 h-4 mr-2" />Benchmarks
              </TabsTrigger>
              <TabsTrigger value="algorithms" className="px-4 py-2 rounded-lg data-[state=active]:bg-cyan-100 data-[state=active]:text-cyan-700">
                <Cpu className="w-4 h-4 mr-2" />Algorithms
              </TabsTrigger>
            </TabsList>

            <TabsContent value="problems">
              <div className="grid gap-4">
                {tasks.map((task) => (
                  <Card key={task.id} className={cn(styles.taskCard, selectedTask?.id === task.id && 'ring-2 ring-cyan-500')} onClick={() => setSelectedTask(task)}>
                    <CardContent className="p-4">
                      <div className="flex items-start justify-between mb-3">
                        <div>
                          <h3 className="font-semibold">{task.name}</h3>
                          <p className="text-sm text-gray-500">{task.problem}</p>
                        </div>
                        <Badge className={getStatusColor(task.status)}>{task.status}</Badge>
                      </div>
                      {task.status !== 'pending' && (
                        <div className="grid grid-cols-4 gap-4 mb-3">
                          <div className="text-center">
                            <p className="text-lg font-bold text-cyan-600">{task.speedup}x</p>
                            <p className="text-xs text-gray-500">Speedup</p>
                          </div>
                          <div className="text-center">
                            <p className="text-lg font-bold">{task.iterations}</p>
                            <p className="text-xs text-gray-500">Iterations</p>
                          </div>
                          <div className="text-center">
                            <p className="text-lg font-bold">{task.optimality.toFixed(2)}</p>
                            <p className="text-xs text-gray-500">Optimality</p>
                          </div>
                          <div className="text-center">
                            <p className="text-lg font-bold">{task.progress}%</p>
                            <p className="text-xs text-gray-500">Progress</p>
                          </div>
                        </div>
                      )}
                      <Progress value={task.progress} className="h-2" />
                      {task.status === 'pending' && (
                        <Button className="w-full mt-3" onClick={(e) => { e.stopPropagation(); handleOptimize(task.id); }}>
                          <Play className="w-4 h-4 mr-2" />Start Optimization
                        </Button>
                      )}
                    </CardContent>
                  </Card>
                ))}
              </div>
            </TabsContent>

            <TabsContent value="benchmarks">
              <Card className={styles.card}>
                <CardHeader><CardTitle>Classical vs Quantum Performance</CardTitle></CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    {benchmarks.map((b, i) => (
                      <div key={i} className="p-4 bg-gray-50 dark:bg-gray-900/50 rounded-lg">
                        <div className="flex items-center justify-between mb-2">
                          <span className="font-medium">{b.name}</span>
                          <Badge className="bg-green-100 text-green-700">{b.speedup}x speedup</Badge>
                        </div>
                        <div className="flex items-center gap-4 text-sm">
                          <div className="flex-1">
                            <span className="text-gray-500">Classical: {b.classical}ms</span>
                            <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2 mt-1">
                              <div className="bg-gray-500 h-2 rounded-full" style={{ width: '100%' }} />
                            </div>
                          </div>
                          <div className="flex-1">
                            <span className="text-cyan-600">Quantum: {b.quantum}ms</span>
                            <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2 mt-1">
                              <div className="bg-cyan-500 h-2 rounded-full" style={{ width: `${(b.quantum / b.classical) * 100}%` }} />
                            </div>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="algorithms">
              <Card className={styles.card}>
                <CardContent className="p-8 text-center">
                  <Cpu className="w-16 h-16 mx-auto mb-4 text-cyan-300" />
                  <h3 className="text-xl font-semibold mb-2">Quantum Algorithms</h3>
                  <p className="text-gray-500">QAOA, VQE, Quantum Annealing, and more</p>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </div>

        <div className="space-y-6">
          <Card className={styles.card}>
            <CardHeader><CardTitle className="flex items-center gap-2"><Zap className="w-5 h-5 text-cyan-500" />Performance</CardTitle></CardHeader>
            <CardContent className="space-y-4">
              <div className="flex justify-between items-center"><span className="text-sm text-gray-500">Completed</span><span className="font-semibold text-green-500">{completedTasks.length}</span></div>
              <div className="flex justify-between items-center"><span className="text-sm text-gray-500">Avg Speedup</span><span className="font-semibold">{avgSpeedup}x</span></div>
              <div className="flex justify-between items-center"><span className="text-sm text-gray-500">Problems Solved</span><span className="font-semibold">{completedTasks.length * 156}</span></div>
            </CardContent>
          </Card>

          <Card className={styles.card}>
            <CardHeader><CardTitle className="flex items-center gap-2"><Gauge className="w-5 h-5 text-amber-500" />Optimization Index</CardTitle></CardHeader>
            <CardContent className="text-center">
              <div className="relative inline-block">
                <svg className="w-24 h-24 transform -rotate-90">
                  <circle cx="48" cy="48" r="40" stroke="#e5e7eb" strokeWidth="8" fill="none" />
                  <circle cx="48" cy="48" r="40" stroke="#06b6d4" strokeWidth="8" fill="none" strokeDasharray="251" strokeDashoffset="35" />
                </svg>
                <span className="absolute inset-0 flex items-center justify-center text-2xl font-bold">86</span>
              </div>
              <p className="mt-2 text-sm text-gray-500">Optimization Efficiency</p>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
