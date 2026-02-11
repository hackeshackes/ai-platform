/**
 * CrossDomainReasoning - Super Intelligence Page
 * 跨域推理 - 超级智能页面
 */

import React, { useState, useCallback } from 'react';
import {
  Network, GitBranch, Layers, Zap, Brain, Target,
  Globe, Cpu, Database, Lightbulb, RefreshCw, Settings,
  Play, Pause, ChevronRight, TrendingUp, Eye,
  BookOpen, Code, Calculator, Music, Palette,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Progress } from '@/components/ui/progress';
import { ScrollArea } from '@/components/ui/scroll-area';
import { useToast } from '@/hooks/use-toast';
import { cn } from '@/lib/utils';

interface Domain {
  id: string; name: string; icon: string; color: string;
  connections: number; transferScore: number; status: 'active' | 'inactive';
}

interface ReasoningTask {
  id: string; prompt: string; domains: string[];
  status: 'pending' | 'processing' | 'completed';
  confidence: number; reasoning: string;
}

const SAMPLE_DOMAINS: Domain[] = [
  { id: '1', name: 'Mathematics', icon: '📐', color: 'bg-red-100 text-red-700', connections: 12, transferScore: 95, status: 'active' },
  { id: '2', name: 'Physics', icon: '⚛️', color: 'bg-blue-100 text-blue-700', connections: 10, transferScore: 88, status: 'active' },
  { id: '3', name: 'Biology', icon: '🧬', color: 'bg-green-100 text-green-700', connections: 8, transferScore: 82, status: 'active' },
  { id: '4', name: 'Medicine', icon: '⚕️', color: 'bg-cyan-100 text-cyan-700', connections: 6, transferScore: 79, status: 'active' },
  { id: '5', name: 'Economics', icon: '📊', color: 'bg-amber-100 text-amber-700', connections: 7, transferScore: 75, status: 'active' },
  { id: '6', name: 'Art', icon: '🎨', color: 'bg-pink-100 text-pink-700', connections: 5, transferScore: 68, status: 'inactive' },
  { id: '7', name: 'Music', icon: '🎵', color: 'bg-purple-100 text-purple-700', connections: 4, transferScore: 62, status: 'inactive' },
  { id: '8', name: 'Computer Science', icon: '💻', color: 'bg-indigo-100 text-indigo-700', connections: 15, transferScore: 92, status: 'active' },
];

const SAMPLE_TASKS: ReasoningTask[] = [
  { id: '1', prompt: 'How can optimization principles from calculus be applied to maximize art composition effectiveness?', domains: ['Mathematics', 'Art'], status: 'completed', confidence: 87, reasoning: 'Applied gradient descent concepts to visual balance.' },
  { id: '2', prompt: 'Draw parallels between neural networks and immune system learning', domains: ['Computer Science', 'Biology'], status: 'completed', confidence: 92, reasoning: 'Identified similarity between pattern recognition and antibody production.' },
  { id: '3', prompt: 'Use economic game theory to optimize resource allocation in emergency medicine', domains: ['Economics', 'Medicine'], status: 'processing', confidence: 65, reasoning: 'Analyzing triage optimization models.' },
];

const styles = {
  container: 'min-h-screen bg-gradient-to-br from-blue-50 via-white to-indigo-50 dark:from-gray-900 dark:via-gray-800 dark:to-gray-900 p-6',
  header: 'flex items-center justify-between mb-8',
  title: 'text-4xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent',
  subtitle: 'text-gray-600 dark:text-gray-300 mt-2',
  mainGrid: 'grid grid-cols-1 lg:grid-cols-3 gap-6',
  content: 'lg:col-span-2',
  card: 'bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm shadow-xl rounded-2xl border-0',
  domainCard: 'bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm shadow-xl rounded-2xl border-0 hover:shadow-2xl transition-all cursor-pointer',
  button: 'bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white shadow-lg',
};

export default function CrossDomainReasoning() {
  const { toast } = useToast();
  const [domains] = useState<Domain[]>(SAMPLE_DOMAINS);
  const [tasks, setTasks] = useState<ReasoningTask[]>(SAMPLE_TASKS);
  const [isProcessing, setIsProcessing] = useState(false);
  const [selectedTask, setSelectedTask] = useState<ReasoningTask | null>(null);

  const handleProcess = useCallback(async () => {
    setIsProcessing(true);
    toast({ title: 'Processing', description: 'Analyzing cross-domain connections' });
    setTimeout(() => {
      setIsProcessing(false);
      toast({ title: 'Complete', description: 'New insights discovered' });
    }, 4000);
  }, [toast]);

  const getDomainIcon = (icon: string) => {
    switch (icon) {
      case '📐': return <Calculator className="w-5 h-5" />;
      case '💻': return <Code className="w-5 h-5" />;
      case '🎨': return <Palette className="w-5 h-5" />;
      case '🎵': return <Music className="w-5 h-5" />;
      default: return <Globe className="w-5 h-5" />;
    }
  };

  const activeDomains = domains.filter(d => d.status === 'active');
  const avgTransferScore = Math.round(domains.reduce((a, d) => a + d.transferScore, 0) / domains.length);

  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <div>
          <h1 className={styles.title}><Network className="inline mr-2 h-8 w-8" />Cross-Domain Reasoning</h1>
          <p className={styles.subtitle}>Connect knowledge across disciplines for innovative insights</p>
        </div>
        <div className="flex items-center gap-3">
          <Button variant="outline"><Settings className="w-4 h-4 mr-2" />Configure Domains</Button>
          <Button className={styles.button} onClick={handleProcess} disabled={isProcessing}>
            {isProcessing ? <RefreshCw className="w-4 h-4 mr-2 animate-spin" /> : <Zap className="w-4 h-4 mr-2" />}
            {isProcessing ? 'Processing...' : 'Analyze Connections'}
          </Button>
        </div>
      </div>

      <div className={styles.mainGrid}>
        <div className={styles.content}>
          <Tabs defaultValue="domains" className="space-y-6">
            <TabsList className="bg-white/60 dark:bg-gray-800/60 backdrop-blur-sm p-1 rounded-xl">
              <TabsTrigger value="domains" className="px-4 py-2 rounded-lg data-[state=active]:bg-blue-100 data-[state=active]:text-blue-700">
                <Globe className="w-4 h-4 mr-2" />Domains
              </TabsTrigger>
              <TabsTrigger value="reasoning" className="px-4 py-2 rounded-lg data-[state=active]:bg-blue-100 data-[state=active]:text-blue-700">
                <Brain className="w-4 h-4 mr-2" />Reasoning
              </TabsTrigger>
              <TabsTrigger value="insights" className="px-4 py-2 rounded-lg data-[state=active]:bg-blue-100 data-[state=active]:text-blue-700">
                <Lightbulb className="w-4 h-4 mr-2" />Insights
              </TabsTrigger>
            </TabsList>

            <TabsContent value="domains">
              <div className="grid grid-cols-2 gap-4">
                {domains.map((domain) => (
                  <Card key={domain.id} className={cn(styles.domainCard, domain.status === 'inactive' && 'opacity-60')}>
                    <CardContent className="p-4">
                      <div className="flex items-center justify-between mb-3">
                        <div className="flex items-center gap-3">
                          <div className={cn('w-10 h-10 rounded-xl flex items-center justify-center', domain.color)}>
                            {getDomainIcon(domain.icon)}
                          </div>
                          <div>
                            <h3 className="font-semibold">{domain.name}</h3>
                            <p className="text-xs text-gray-500">{domain.connections} connections</p>
                          </div>
                        </div>
                        <Badge variant={domain.status === 'active' ? 'default' : 'secondary'}>{domain.status}</Badge>
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-sm text-gray-500">Transfer Score</span>
                        <span className="text-sm font-medium">{domain.transferScore}%</span>
                      </div>
                      <Progress value={domain.transferScore} className="mt-2 h-1.5" />
                    </CardContent>
                  </Card>
                ))}
              </div>
            </TabsContent>

            <TabsContent value="reasoning">
              <div className="space-y-4">
                {tasks.map((task) => (
                  <Card key={task.id} className={cn(styles.card, selectedTask?.id === task.id && 'ring-2 ring-blue-500')} onClick={() => setSelectedTask(task)}>
                    <CardContent className="p-4">
                      <div className="flex items-start justify-between mb-2">
                        <div className="flex items-center gap-2">
                          <span className="text-lg">🔗</span>
                          <h3 className="font-semibold">{task.prompt}</h3>
                        </div>
                        <Badge variant={task.status === 'completed' ? 'default' : task.status === 'processing' ? 'secondary' : 'outline'}>
                          {task.status}
                        </Badge>
                      </div>
                      <div className="flex flex-wrap gap-2 mb-3">
                        {task.domains.map((d) => (
                          <Badge key={d} variant="outline">{d}</Badge>
                        ))}
                      </div>
                      {task.status === 'completed' && (
                        <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-3">
                          <p className="text-sm text-gray-600 dark:text-gray-300">{task.reasoning}</p>
                          <div className="flex items-center justify-end mt-2">
                            <span className="text-xs text-gray-500">Confidence: </span>
                            <span className="text-sm font-medium text-green-600 ml-1">{task.confidence}%</span>
                          </div>
                        </div>
                      )}
                    </CardContent>
                  </Card>
                ))}
              </div>
            </TabsContent>

            <TabsContent value="insights">
              <Card className={styles.card}>
                <CardContent className="p-8 text-center">
                  <Lightbulb className="w-16 h-16 mx-auto mb-4 text-blue-300" />
                  <h3 className="text-xl font-semibold mb-2">Cross-Domain Insights</h3>
                  <p className="text-gray-500">Discover novel connections between domains</p>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </div>

        <div className="space-y-6">
          <Card className={styles.card}>
            <CardHeader><CardTitle className="flex items-center gap-2"><Network className="w-5 h-5 text-blue-500" />Network Status</CardTitle></CardHeader>
            <CardContent className="space-y-4">
              <div className="flex justify-between items-center"><span className="text-sm text-gray-500">Active Domains</span><span className="font-semibold">{activeDomains.length}</span></div>
              <div className="flex justify-between items-center"><span className="text-sm text-gray-500">Total Connections</span><span className="font-semibold">{domains.reduce((a, d) => a + d.connections, 0)}</span></div>
              <div className="flex justify-between items-center"><span className="text-sm text-gray-500">Avg Transfer</span><span className="font-semibold text-green-500">{avgTransferScore}%</span></div>
            </CardContent>
          </Card>

          <Card className={styles.card}>
            <CardHeader><CardTitle className="flex items-center gap-2"><TrendingUp className="w-5 h-5 text-green-500" />Reasoning Index</CardTitle></CardHeader>
            <CardContent className="text-center">
              <div className="relative inline-block">
                <svg className="w-24 h-24 transform -rotate-90">
                  <circle cx="48" cy="48" r="40" stroke="#e5e7eb" strokeWidth="8" fill="none" />
                  <circle cx="48" cy="48" r="40" stroke="#3b82f6" strokeWidth="8" fill="none" strokeDasharray="251" strokeDashoffset="35" />
                </svg>
                <span className="absolute inset-0 flex items-center justify-center text-2xl font-bold">86</span>
              </div>
              <p className="mt-2 text-sm text-gray-500">Cross-Domain Score</p>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
