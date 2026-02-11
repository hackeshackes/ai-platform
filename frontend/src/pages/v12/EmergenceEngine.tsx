/**
 * EmergenceEngine - Super Intelligence Page
 * 涌现能力 - 超级智能页面
 */

import React, { useState, useCallback } from 'react';
import {
  Brain, Sparkles, Zap, Network, Layers, GitBranch, Cpu,
  Activity, Target, TrendingUp, RefreshCw, Settings, Play,
  Eye, ChevronRight, Hexagon, DNA, Atom, Lightbulb,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Progress } from '@/components/ui/progress';
import { ScrollArea } from '@/components/ui/scroll-area';
import { useToast } from '@/hooks/use-toast';
import { cn } from '@/lib/utils';

interface EmergenceCapability {
  id: string; name: string; description: string;
  category: 'reasoning' | 'creativity' | 'adaptation' | 'social';
  level: number; threshold: number;
  status: 'developing' | 'active' | 'mastered';
  metrics: { name: string; value: number }[];
}

interface EmergenceEvent {
  id: string; timestamp: Date; capability: string;
  description: string; impact: 'low' | 'medium' | 'high';
}

const SAMPLE_CAPABILITIES: EmergenceCapability[] = [
  { id: '1', name: 'Abstract Reasoning', description: 'Ability to understand abstract concepts', category: 'reasoning', level: 85, threshold: 80, status: 'active', metrics: [{ name: 'Score', value: 92 }, { name: 'Consistency', value: 88 }] },
  { id: '2', name: 'Creative Synthesis', description: 'Generate novel ideas from combinations', category: 'creativity', level: 72, threshold: 75, status: 'developing', metrics: [{ name: 'Novelty', value: 78 }, { name: 'Usefulness', value: 65 }] },
  { id: '3', name: 'Self-Adaptation', description: 'Modify behavior based on feedback', category: 'adaptation', level: 90, threshold: 85, status: 'mastered', metrics: [{ name: 'Speed', value: 95 }, { name: 'Accuracy', value: 88 }] },
  { id: '4', name: 'Theory of Mind', description: 'Understand mental states of others', category: 'social', level: 68, threshold: 70, status: 'developing', metrics: [{ name: 'Inference', value: 72 }, { name: 'Prediction', value: 64 }] },
  { id: '5', name: 'Cross-Domain Transfer', description: 'Apply knowledge across domains', category: 'adaptation', level: 78, threshold: 80, status: 'active', metrics: [{ name: 'Transfer Rate', value: 82 }, { name: 'Retention', value: 74 }] },
];

const SAMPLE_EVENTS: EmergenceEvent[] = [
  { id: '1', timestamp: new Date(), capability: 'Self-Adaptation', description: 'New adaptation pattern discovered', impact: 'high' },
  { id: '2', timestamp: new Date(Date.now() - 3600000), capability: 'Abstract Reasoning', description: 'Solved novel abstraction task', impact: 'medium' },
  { id: '3', timestamp: new Date(Date.now() - 7200000), capability: 'Creative Synthesis', description: 'Generated breakthrough idea combination', impact: 'high' },
];

const styles = {
  container: 'min-h-screen bg-gradient-to-br from-amber-50 via-white to-orange-50 dark:from-gray-900 dark:via-gray-800 dark:to-gray-900 p-6',
  header: 'flex items-center justify-between mb-8',
  title: 'text-4xl font-bold bg-gradient-to-r from-amber-600 to-orange-600 bg-clip-text text-transparent',
  subtitle: 'text-gray-600 dark:text-gray-300 mt-2',
  mainGrid: 'grid grid-cols-1 lg:grid-cols-3 gap-6',
  content: 'lg:col-span-2',
  card: 'bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm shadow-xl rounded-2xl border-0',
  capCard: 'bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm shadow-xl rounded-2xl border-0 hover:shadow-2xl transition-all',
  button: 'bg-gradient-to-r from-amber-600 to-orange-600 hover:from-amber-700 hover:to-orange-700 text-white shadow-lg',
};

export default function EmergenceEngine() {
  const { toast } = useToast();
  const [capabilities, setCapabilities] = useState<EmergenceCapability[]>(SAMPLE_CAPABILITIES);
  const [events] = useState<EmergenceEvent[]>(SAMPLE_EVENTS);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  const handleAnalyze = useCallback(async () => {
    setIsAnalyzing(true);
    toast({ title: 'Analysis Started', description: 'Scanning for new emergent capabilities' });
    setTimeout(() => {
      setIsAnalyzing(false);
      toast({ title: 'Analysis Complete', description: 'No new emergence detected' });
    }, 5000);
  }, [toast]);

  const getCategoryIcon = (category: string) => {
    switch (category) {
      case 'reasoning': return '🧠'; case 'creativity': return '💡';
      case 'adaptation': return '🔄'; case 'social': return '👥';
      default: return '✨';
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'mastered': return 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400';
      case 'active': return 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400';
      case 'developing': return 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400';
      default: return 'bg-gray-100 text-gray-700';
    }
  };

  const getImpactColor = (impact: string) => {
    switch (impact) {
      case 'high': return 'text-red-500'; case 'medium': return 'text-yellow-500'; case 'low': return 'text-green-500';
      default: return 'text-gray-500';
    }
  };

  const masteredCount = capabilities.filter(c => c.status === 'mastered').length;
  const activeCount = capabilities.filter(c => c.status === 'active').length;
  const developingCount = capabilities.filter(c => c.status === 'developing').length;

  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <div>
          <h1 className={styles.title}><Sparkles className="inline mr-2 h-8 w-8" />Emergence Engine</h1>
          <p className={styles.subtitle}>Discover and nurture emergent AI capabilities</p>
        </div>
        <div className="flex items-center gap-3">
          <Button variant="outline"><Settings className="w-4 h-4 mr-2" />Configure</Button>
          <Button className={styles.button} onClick={handleAnalyze} disabled={isAnalyzing}>
            {isAnalyzing ? <RefreshCw className="w-4 h-4 mr-2 animate-spin" /> : <Network className="w-4 h-4 mr-2" />}
            {isAnalyzing ? 'Analyzing...' : 'Scan for Emergence'}
          </Button>
        </div>
      </div>

      <div className={styles.mainGrid}>
        <div className={styles.content}>
          <Tabs defaultValue="capabilities" className="space-y-6">
            <TabsList className="bg-white/60 dark:bg-gray-800/60 backdrop-blur-sm p-1 rounded-xl">
              <TabsTrigger value="capabilities" className="px-4 py-2 rounded-lg data-[state=active]:bg-amber-100 data-[state=active]:text-amber-700">
                <Brain className="w-4 h-4 mr-2" />Capabilities
              </TabsTrigger>
              <TabsTrigger value="events" className="px-4 py-2 rounded-lg data-[state=active]:bg-amber-100 data-[state=active]:text-amber-700">
                <Zap className="w-4 h-4 mr-2" />Emergence Events
              </TabsTrigger>
              <TabsTrigger value="patterns" className="px-4 py-2 rounded-lg data-[state=active]:bg-amber-100 data-[state=active]:text-amber-700">
                <Layers className="w-4 h-4 mr-2" />Patterns
              </TabsTrigger>
            </TabsList>

            <TabsContent value="capabilities">
              <div className="grid gap-4">
                {capabilities.map((cap) => (
                  <Card key={cap.id} className={styles.capCard}>
                    <CardContent className="p-4">
                      <div className="flex items-start justify-between mb-3">
                        <div className="flex items-start gap-3">
                          <span className="text-3xl">{getCategoryIcon(cap.category)}</span>
                          <div>
                            <h3 className="font-semibold">{cap.name}</h3>
                            <p className="text-sm text-gray-500">{cap.description}</p>
                          </div>
                        </div>
                        <Badge className={getStatusColor(cap.status)}>{cap.status}</Badge>
                      </div>
                      <div className="grid grid-cols-2 gap-4 mb-3">
                        {cap.metrics.map((m, i) => (
                          <div key={i} className="bg-gray-50 dark:bg-gray-900/50 rounded-lg p-2">
                            <p className="text-xs text-gray-500">{m.name}</p>
                            <div className="flex items-center justify-between mt-1">
                              <span className="text-lg font-bold">{m.value}%</span>
                              <Progress value={m.value} className="w-16 h-1" />
                            </div>
                          </div>
                        ))}
                      </div>
                      <div className="flex items-center justify-between pt-3 border-t">
                        <span className="text-sm text-gray-500">Development Level</span>
                        <Progress value={(cap.level / 100) * 100} className="w-32 h-2" />
                        <span className="text-sm font-medium">{cap.level}%</span>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            </TabsContent>

            <TabsContent value="events">
              <div className="space-y-3">
                {events.map((event) => (
                  <Card key={event.id} className={styles.card}>
                    <CardContent className="p-4">
                      <div className="flex items-start justify-between">
                        <div className="flex items-start gap-3">
                          <div className={cn('w-2 h-2 rounded-full mt-2', event.impact === 'high' ? 'bg-red-500' : event.impact === 'medium' ? 'bg-yellow-500' : 'bg-green-500')} />
                          <div>
                            <h3 className="font-semibold">{event.capability}</h3>
                            <p className="text-sm text-gray-500">{event.description}</p>
                            <p className="text-xs text-gray-400 mt-1">{new Date(event.timestamp).toLocaleString()}</p>
                          </div>
                        </div>
                        <Badge className={getImpactColor(event.impact)}>{event.impact} impact</Badge>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            </TabsContent>

            <TabsContent value="patterns">
              <Card className={styles.card}>
                <CardContent className="p-8 text-center">
                  <Layers className="w-16 h-16 mx-auto mb-4 text-amber-300" />
                  <h3 className="text-xl font-semibold mb-2">Emergence Patterns</h3>
                  <p className="text-gray-500">Analyze patterns that lead to emergent behavior</p>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </div>

        <div className="space-y-6">
          <Card className={styles.card}>
            <CardHeader><CardTitle className="flex items-center gap-2"><Activity className="w-5 h-5 text-amber-500" />Development Status</CardTitle></CardHeader>
            <CardContent className="space-y-4">
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-500">Mastered</span>
                <Badge className="bg-green-100 text-green-700">{masteredCount}</Badge>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-500">Active</span>
                <Badge className="bg-blue-100 text-blue-700">{activeCount}</Badge>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-500">Developing</span>
                <Badge className="bg-yellow-100 text-yellow-700">{developingCount}</Badge>
              </div>
            </CardContent>
          </Card>

          <Card className={styles.card}>
            <CardHeader><CardTitle className="flex items-center gap-2"><TrendingUp className="w-5 h-5 text-green-500" />Emergence Index</CardTitle></CardHeader>
            <CardContent className="text-center">
              <div className="relative inline-block">
                <svg className="w-24 h-24 transform -rotate-90">
                  <circle cx="48" cy="48" r="40" stroke="#e5e7eb" strokeWidth="8" fill="none" />
                  <circle cx="48" cy="48" r="40" stroke="#f59e0b" strokeWidth="8" fill="none" strokeDasharray="251" strokeDashoffset="50" />
                </svg>
                <span className="absolute inset-0 flex items-center justify-center text-2xl font-bold">80</span>
              </div>
              <p className="mt-2 text-sm text-gray-500">Emergence Score</p>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
