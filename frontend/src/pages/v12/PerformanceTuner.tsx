/**
 * PerformanceTuner - Hyperautomation Page
 * 性能优化 - 超自动化页面
 */

import React, { useState, useCallback } from 'react';
import {
  Zap, Activity, Cpu, Memory, HardDrive, Network, Gauge,
  TrendingUp, TrendingDown, RefreshCw, Play, Pause, Settings,
  CheckCircle, AlertTriangle, ChevronRight, BarChart3, Clock,
  Database, Server, Globe, Terminal, Download, Upload,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Progress } from '@/components/ui/progress';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Separator } from '@/components/ui/separator';
import { useToast } from '@/hooks/use-toast';
import { cn } from '@/lib/utils';

interface OptimizationSuggestion {
  id: string;
  title: string;
  description: string;
  impact: 'high' | 'medium' | 'low';
  category: string;
  estimatedGain: string;
  effort: 'easy' | 'medium' | 'hard';
  selected: boolean;
}

interface PerformanceMetric {
  id: string; name: string; value: number; unit: string;
  status: 'optimal' | 'good' | 'degraded' | 'critical';
  trend: 'up' | 'down' | 'stable';
  history: number[];
}

const SAMPLE_SUGGESTIONS: OptimizationSuggestion[] = [
  { id: '1', title: 'Enable GPU Acceleration', description: 'Use GPU for ML inference to improve throughput', impact: 'high', category: 'compute', estimatedGain: '+45% faster', effort: 'easy', selected: false },
  { id: '2', title: 'Optimize Database Queries', description: 'Add indexes and optimize slow queries', impact: 'high', category: 'database', estimatedGain: '-60% latency', effort: 'medium', selected: true },
  { id: '3', title: 'Enable Caching Layer', description: 'Add Redis cache for frequently accessed data', impact: 'medium', category: 'cache', estimatedGain: '+30% response', effort: 'medium', selected: false },
  { id: '4', title: 'Compress Model Weights', description: 'Quantize ML models for faster loading', impact: 'medium', category: 'ml', estimatedGain: '-40% size', effort: 'hard', selected: false },
  { id: '5', title: 'Scale Horizontally', description: 'Add more instances to handle increased load', impact: 'high', category: 'infrastructure', estimatedGain: '+100% capacity', effort: 'easy', selected: true },
];

const SAMPLE_METRICS: PerformanceMetric[] = [
  { id: '1', name: 'Response Time', value: 145, unit: 'ms', status: 'good', trend: 'down', history: [180, 165, 155, 150, 145] },
  { id: '2', name: 'Throughput', value: 2500, unit: 'req/s', status: 'optimal', trend: 'up', history: [2000, 2100, 2300, 2400, 2500] },
  { id: '3', name: 'Error Rate', value: 0.5, unit: '%', status: 'optimal', trend: 'stable', history: [0.6, 0.5, 0.5, 0.5, 0.5] },
  { id: '4', name: 'CPU Usage', value: 72, unit: '%', status: 'degraded', trend: 'up', history: [60, 62, 65, 68, 72] },
  { id: '5', name: 'Memory Usage', value: 68, unit: '%', status: 'good', trend: 'stable', history: [70, 69, 68, 68, 68] },
];

const styles = {
  container: 'min-h-screen bg-gradient-to-br from-emerald-50 via-white to-teal-50 dark:from-gray-900 dark:via-gray-800 dark:to-gray-900 p-6',
  header: 'flex items-center justify-between mb-8',
  title: 'text-4xl font-bold bg-gradient-to-r from-emerald-600 to-teal-600 bg-clip-text text-transparent',
  subtitle: 'text-gray-600 dark:text-gray-300 mt-2',
  mainGrid: 'grid grid-cols-1 lg:grid-cols-3 gap-6',
  content: 'lg:col-span-2',
  card: 'bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm shadow-xl rounded-2xl border-0',
  suggestionCard: 'bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm shadow-xl rounded-2xl border-0 hover:shadow-2xl transition-all',
  button: 'bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-700 hover:to-teal-700 text-white shadow-lg',
};

export default function PerformanceTuner() {
  const { toast } = useToast();
  const [suggestions, setSuggestions] = useState<OptimizationSuggestion[]>(SAMPLE_SUGGESTIONS);
  const [metrics] = useState<PerformanceMetric[]>(SAMPLE_METRICS);
  const [isOptimizing, setIsOptimizing] = useState(false);
  const [selectedCategory, setSelectedCategory] = useState('all');

  const handleApplySuggestion = useCallback((id: string) => {
    setIsOptimizing(true);
    setTimeout(() => {
      setSuggestions(prev => prev.filter(s => s.id !== id));
      setIsOptimizing(false);
      toast({ title: 'Optimization Applied', description: 'Performance improvement has been applied' });
    }, 2000);
  }, [toast]);

  const handleApplyAll = useCallback(() => {
    const selected = suggestions.filter(s => s.selected);
    setIsOptimizing(true);
    setTimeout(() => {
      setSuggestions(prev => prev.filter(s => !s.selected.map(ss => ss.id).includes(s.id)));
      setIsOptimizing(false);
      toast({ title: 'Optimizations Applied', description: `${selected.length} optimizations have been applied` });
    }, 3000);
  }, [suggestions, toast]);

  const getImpactColor = (impact: string) => {
    switch (impact) {
      case 'high': return 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400';
      case 'medium': return 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400';
      case 'low': return 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400';
      default: return 'bg-gray-100 text-gray-700';
    }
  };

  const getEffortColor = (effort: string) => {
    switch (effort) {
      case 'easy': return 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400';
      case 'medium': return 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400';
      case 'hard': return 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400';
      default: return 'bg-gray-100 text-gray-700';
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'optimal': return 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400';
      case 'good': return 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400';
      case 'degraded': return 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400';
      case 'critical': return 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400';
      default: return 'bg-gray-100 text-gray-700';
    }
  };

  const filteredSuggestions = selectedCategory === 'all' ? suggestions : suggestions.filter(s => s.category === selectedCategory);
  const selectedCount = suggestions.filter(s => s.selected).length;

  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <div>
          <h1 className={styles.title}><Gauge className="inline mr-2 h-8 w-8" />Performance Tuner</h1>
          <p className={styles.subtitle}>AI-powered performance optimization and tuning</p>
        </div>
        <div className="flex items-center gap-3">
          <Button variant="outline"><RefreshCw className="w-4 h-4 mr-2" />Scan</Button>
          <Button variant="outline"><Download className="w-4 h-4 mr-2" />Export Report</Button>
          <Button className={styles.button} onClick={handleApplyAll} disabled={selectedCount === 0 || isOptimizing}>
            {isOptimizing ? <RefreshCw className="w-4 h-4 mr-2 animate-spin" /> : <Zap className="w-4 h-4 mr-2" />}
            Apply Selected ({selectedCount})
          </Button>
        </div>
      </div>

      <div className={styles.mainGrid}>
        <div className={styles.content}>
          <Tabs defaultValue="suggestions" className="space-y-6">
            <TabsList className="bg-white/60 dark:bg-gray-800/60 backdrop-blur-sm p-1 rounded-xl">
              <TabsTrigger value="suggestions" className="px-4 py-2 rounded-lg data-[state=active]:bg-emerald-100 data-[state=active]:text-emerald-700">
                <Zap className="w-4 h-4 mr-2" />Suggestions
              </TabsTrigger>
              <TabsTrigger value="metrics" className="px-4 py-2 rounded-lg data-[state=active]:bg-emerald-100 data-[state=active]:text-emerald-700">
                <Activity className="w-4 h-4 mr-2" />Metrics
              </TabsTrigger>
              <TabsTrigger value="benchmarks" className="px-4 py-2 rounded-lg data-[state=active]:bg-emerald-100 data-[state=active]:text-emerald-700">
                <BarChart3 className="w-4 h-4 mr-2" />Benchmarks
              </TabsTrigger>
            </TabsList>

            <TabsContent value="suggestions">
              <div className="flex gap-2 mb-4">
                {['all', 'compute', 'database', 'cache', 'ml', 'infrastructure'].map(cat => (
                  <Button key={cat} variant={selectedCategory === cat ? 'default' : 'outline'} size="sm" onClick={() => setSelectedCategory(cat)}>
                    {cat.charAt(0).toUpperCase() + cat.slice(1)}
                  </Button>
                ))}
              </div>
              <div className="grid gap-4">
                {filteredSuggestions.map((suggestion) => (
                  <Card key={suggestion.id} className={styles.suggestionCard}>
                    <CardContent className="p-4">
                      <div className="flex items-start justify-between">
                        <div className="flex items-start gap-4">
                          <div className="w-12 h-12 rounded-xl bg-emerald-100 dark:bg-emerald-900/30 flex items-center justify-center">
                            <Zap className="w-6 h-6 text-emerald-600" />
                          </div>
                          <div>
                            <h3 className="font-semibold">{suggestion.title}</h3>
                            <p className="text-sm text-gray-500 mt-1">{suggestion.description}</p>
                            <div className="flex items-center gap-3 mt-2">
                              <Badge className={getImpactColor(suggestion.impact)}>{suggestion.impact} impact</Badge>
                              <Badge className={getEffortColor(suggestion.effort)}>{suggestion.effort}</Badge>
                              <span className="text-sm text-green-600">{suggestion.estimatedGain}</span>
                            </div>
                          </div>
                        </div>
                        <div className="flex items-center gap-2">
                          <Button variant="outline" size="sm" onClick={() => handleApplySuggestion(suggestion.id)} disabled={isOptimizing}>
                            Apply
                          </Button>
                          <Button variant={suggestion.selected ? 'default' : 'outline'} size="sm" onClick={() => setSuggestions(prev => prev.map(s => s.id === suggestion.id ? { ...s, selected: !s.selected } : s))}>
                            {suggestion.selected ? 'Selected' : 'Select'}
                          </Button>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            </TabsContent>

            <TabsContent value="metrics">
              <div className="grid grid-cols-2 gap-4">
                {metrics.map((metric) => (
                  <Card key={metric.id} className={styles.card}>
                    <CardContent className="p-4">
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-sm text-gray-500">{metric.name}</span>
                        <Badge className={getStatusColor(metric.status)}>{metric.status}</Badge>
                      </div>
                      <div className="flex items-end justify-between">
                        <span className="text-3xl font-bold">{metric.value}{metric.unit}</span>
                        <div className={cn('flex items-center gap-1 text-sm', metric.trend === 'up' ? 'text-green-500' : metric.trend === 'down' ? 'text-red-500' : 'text-gray-400')}>
                          {metric.trend === 'up' ? <TrendingUp className="w-4 h-4" /> : metric.trend === 'down' ? <TrendingDown className="w-4 h-4" /> : null}
                        </div>
                      </div>
                      <div className="flex gap-0.5 mt-2">
                        {metric.history.map((v, i) => (
                          <div key={i} className="flex-1 bg-gray-200 dark:bg-gray-700 rounded-sm" style={{ height: 4 }}>
                            <div className="bg-emerald-500 rounded-sm" style={{ height: '100%', width: `${(v / Math.max(...metric.history)) * 100}%` }} />
                          </div>
                        ))}
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            </TabsContent>

            <TabsContent value="benchmarks">
              <Card className={styles.card}>
                <CardContent className="p-8 text-center">
                  <BarChart3 className="w-16 h-16 mx-auto mb-4 text-emerald-300" />
                  <h3 className="text-xl font-semibold mb-2">Performance Benchmarks</h3>
                  <p className="text-gray-500">Compare your performance against industry standards</p>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </div>

        <div className="space-y-6">
          <Card className={styles.card}>
            <CardHeader><CardTitle className="flex items-center gap-2"><Activity className="w-5 h-5 text-emerald-500" />Overall Score</CardTitle></CardHeader>
            <CardContent className="text-center">
              <div className="relative inline-block">
                <svg className="w-32 h-32 transform -rotate-90">
                  <circle cx="64" cy="64" r="56" stroke="#e5e7eb" strokeWidth="12" fill="none" />
                  <circle cx="64" cy="64" r="56" stroke="#10b981" strokeWidth="12" fill="none" strokeDasharray="352" strokeDashoffset="70" />
                </svg>
                <span className="absolute inset-0 flex items-center justify-center text-3xl font-bold">85</span>
              </div>
              <p className="mt-2 text-sm text-gray-500">Good Performance</p>
            </CardContent>
          </Card>

          <Card className={styles.card}>
            <CardHeader><CardTitle className="flex items-center gap-2"><Zap className="w-5 h-5 text-amber-500" />Quick Stats</CardTitle></CardHeader>
            <CardContent className="space-y-4">
              <div className="flex justify-between items-center"><span className="text-sm text-gray-500">Suggestions</span><span className="font-semibold">{suggestions.length}</span></div>
              <div className="flex justify-between items-center"><span className="text-sm text-gray-500">High Impact</span><span className="font-semibold text-green-500">{suggestions.filter(s => s.impact === 'high').length}</span></div>
              <div className="flex justify-between items-center"><span className="text-sm text-gray-500">Selected</span><span className="font-semibold">{selectedCount}</span></div>
              <Separator />
              <div className="flex justify-between items-center"><span className="text-sm text-gray-500">Last Scan</span><span className="text-sm">2 hours ago</span></div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
