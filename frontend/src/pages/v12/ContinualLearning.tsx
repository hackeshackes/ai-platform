/**
 * ContinualLearning - Super Intelligence Page
 * 持续学习 - 超级智能页面
 */

import React, { useState, useCallback } from 'react';
import {
  Brain, RefreshCw, BookOpen, Target, TrendingUp, Clock,
  CheckCircle, AlertCircle, Play, Pause, Settings, Zap,
  Layers, History, GraduationCap, ChevronRight,
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

interface LearningSession {
  id: string; topic: string; startDate: Date;
  status: 'active' | 'completed' | 'paused';
  progress: number; tasksCompleted: number; totalTasks: number;
  accuracy: number; retention: number;
}

interface KnowledgeUpdate {
  id: string; timestamp: Date; type: 'addition' | 'modification' | 'refinement';
  description: string; impact: 'high' | 'medium' | 'low';
  source: string;
}

const SAMPLE_SESSIONS: LearningSession[] = [
  { id: '1', topic: 'Advanced Natural Language Processing', startDate: new Date(Date.now() - 604800000), status: 'active', progress: 72, tasksCompleted: 36, totalTasks: 50, accuracy: 94.5, retention: 88 },
  { id: '2', topic: 'Reinforcement Learning Fundamentals', startDate: new Date(Date.now() - 1209600000), status: 'completed', progress: 100, tasksCompleted: 40, totalTasks: 40, accuracy: 91.2, retention: 92 },
  { id: '3', topic: 'Computer Vision Techniques', startDate: new Date(Date.now() - 259200000), status: 'active', progress: 35, tasksCompleted: 14, totalTasks: 40, accuracy: 87.8, retention: 75 },
  { id: '4', topic: 'Graph Neural Networks', startDate: new Date(Date.now() - 86400000), status: 'paused', progress: 20, tasksCompleted: 4, totalTasks: 20, accuracy: 89.0, retention: 82 },
];

const SAMPLE_UPDATES: KnowledgeUpdate[] = [
  { id: '1', timestamp: new Date(), type: 'refinement', description: 'Improved understanding of attention mechanisms', impact: 'high', source: 'NLP Learning' },
  { id: '2', timestamp: new Date(Date.now() - 1800000), type: 'addition', description: 'New knowledge: Transformer architecture variants', impact: 'medium', source: 'Paper Review' },
  { id: '3', timestamp: new Date(Date.now() - 3600000), type: 'modification', description: 'Updated probability estimates for Bayesian models', impact: 'low', source: 'Data Analysis' },
];

const styles = {
  container: 'min-h-screen bg-gradient-to-br from-green-50 via-white to-emerald-50 dark:from-gray-900 dark:via-gray-800 dark:to-gray-900 p-6',
  header: 'flex items-center justify-between mb-8',
  title: 'text-4xl font-bold bg-gradient-to-r from-green-600 to-emerald-600 bg-clip-text text-transparent',
  subtitle: 'text-gray-600 dark:text-gray-300 mt-2',
  mainGrid: 'grid grid-cols-1 lg:grid-cols-3 gap-6',
  content: 'lg:col-span-2',
  card: 'bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm shadow-xl rounded-2xl border-0',
  sessionCard: 'bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm shadow-xl rounded-2xl border-0 hover:shadow-2xl transition-all',
  button: 'bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-700 hover:to-emerald-700 text-white shadow-lg',
};

export default function ContinualLearning() {
  const { toast } = useToast();
  const [sessions] = useState<LearningSession[]>(SAMPLE_SESSIONS);
  const [updates] = useState<KnowledgeUpdate[]>(SAMPLE_UPDATES);
  const [isLearning, setIsLearning] = useState(false);
  const [selectedSession, setSelectedSession] = useState<LearningSession | null>(null);

  const handleStartLearning = useCallback(async () => {
    setIsLearning(true);
    toast({ title: 'Learning Started', description: 'Continuing with next learning session' });
    setTimeout(() => {
      setIsLearning(false);
      toast({ title: 'Session Complete', description: 'Knowledge has been consolidated' });
    }, 5000);
  }, [toast]);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400';
      case 'completed': return 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400';
      case 'paused': return 'bg-gray-100 text-gray-700 dark:bg-gray-900/30 dark:text-gray-400';
      default: return 'bg-gray-100 text-gray-700';
    }
  };

  const getImpactColor = (impact: string) => {
    switch (impact) {
      case 'high': return 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400';
      case 'medium': return 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400';
      case 'low': return 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400';
      default: return 'bg-gray-100 text-gray-700';
    }
  };

  const activeSessions = sessions.filter(s => s.status === 'active');
  const completedSessions = sessions.filter(s => s.status === 'completed');
  const avgAccuracy = Math.round(sessions.reduce((a, s) => a + s.accuracy, 0) / sessions.length);

  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <div>
          <h1 className={styles.title}><Brain className="inline mr-2 h-8 w-8" />Continual Learning</h1>
          <p className={styles.subtitle}>AI that continuously learns and adapts without forgetting</p>
        </div>
        <div className="flex items-center gap-3">
          <Button variant="outline"><Settings className="w-4 h-4 mr-2" />Configure</Button>
          <Button className={styles.button} onClick={handleStartLearning} disabled={isLearning}>
            {isLearning ? <RefreshCw className="w-4 h-4 mr-2 animate-spin" /> : <Zap className="w-4 h-4 mr-2" />}
            {isLearning ? 'Learning...' : 'Continue Learning'}
          </Button>
        </div>
      </div>

      <div className={styles.mainGrid}>
        <div className={styles.content}>
          <Tabs defaultValue="sessions" className="space-y-6">
            <TabsList className="bg-white/60 dark:bg-gray-800/60 backdrop-blur-sm p-1 rounded-xl">
              <TabsTrigger value="sessions" className="px-4 py-2 rounded-lg data-[state=active]:bg-green-100 data-[state=active]:text-green-700">
                <BookOpen className="w-4 h-4 mr-2" />Sessions
              </TabsTrigger>
              <TabsTrigger value="knowledge" className="px-4 py-2 rounded-lg data-[state=active]:bg-green-100 data-[state=active]:text-green-700">
                <Layers className="w-4 h-4 mr-2" />Knowledge
              </TabsTrigger>
              <TabsTrigger value="history" className="px-4 py-2 rounded-lg data-[state=active]:bg-green-100 data-[state=active]:text-green-700">
                <History className="w-4 h-4 mr-2" />History
              </TabsTrigger>
            </TabsList>

            <TabsContent value="sessions">
              <div className="grid gap-4">
                {sessions.map((session) => (
                  <Card key={session.id} className={cn(styles.sessionCard, selectedSession?.id === session.id && 'ring-2 ring-green-500')} onClick={() => setSelectedSession(session)}>
                    <CardContent className="p-4">
                      <div className="flex items-start justify-between mb-3">
                        <div>
                          <h3 className="font-semibold">{session.topic}</h3>
                          <p className="text-sm text-gray-500">Started {new Date(session.startDate).toLocaleDateString()}</p>
                        </div>
                        <Badge className={getStatusColor(session.status)}>{session.status}</Badge>
                      </div>
                      <div className="grid grid-cols-4 gap-4 mb-3">
                        <div className="text-center">
                          <p className="text-xl font-bold text-green-600">{session.progress}%</p>
                          <p className="text-xs text-gray-500">Progress</p>
                        </div>
                        <div className="text-center">
                          <p className="text-xl font-bold">{session.tasksCompleted}/{session.totalTasks}</p>
                          <p className="text-xs text-gray-500">Tasks</p>
                        </div>
                        <div className="text-center">
                          <p className="text-xl font-bold">{session.accuracy}%</p>
                          <p className="text-xs text-gray-500">Accuracy</p>
                        </div>
                        <div className="text-center">
                          <p className="text-xl font-bold">{session.retention}%</p>
                          <p className="text-xs text-gray-500">Retention</p>
                        </div>
                      </div>
                      <Progress value={session.progress} className="h-2" />
                    </CardContent>
                  </Card>
                ))}
              </div>
            </TabsContent>

            <TabsContent value="knowledge">
              <div className="space-y-3">
                {updates.map((update) => (
                  <Card key={update.id} className={styles.card}>
                    <CardContent className="p-4">
                      <div className="flex items-start justify-between">
                        <div className="flex items-start gap-3">
                          {update.type === 'addition' ? <CheckCircle className="w-5 h-5 text-green-500 mt-0.5" /> : update.type === 'modification' ? <RefreshCw className="w-5 h-5 text-blue-500 mt-0.5" /> : <Target className="w-5 h-5 text-purple-500 mt-0.5" />}
                          <div>
                            <h3 className="font-semibold">{update.description}</h3>
                            <p className="text-sm text-gray-500">{update.source} • {new Date(update.timestamp).toLocaleString()}</p>
                          </div>
                        </div>
                        <Badge className={getImpactColor(update.impact)}>{update.impact} impact</Badge>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            </TabsContent>

            <TabsContent value="history">
              <Card className={styles.card}>
                <CardContent className="p-8 text-center">
                  <History className="w-16 h-16 mx-auto mb-4 text-green-300" />
                  <h3 className="text-xl font-semibold mb-2">Learning History</h3>
                  <p className="text-gray-500">View your complete learning trajectory</p>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </div>

        <div className="space-y-6">
          <Card className={styles.card}>
            <CardHeader><CardTitle className="flex items-center gap-2"><TrendingUp className="w-5 h-5 text-green-500" />Learning Stats</CardTitle></CardHeader>
            <CardContent className="space-y-4">
              <div className="flex justify-between items-center"><span className="text-sm text-gray-500">Active Sessions</span><span className="font-semibold">{activeSessions.length}</span></div>
              <div className="flex justify-between items-center"><span className="text-sm text-gray-500">Completed</span><span className="font-semibold text-green-500">{completedSessions.length}</span></div>
              <div className="flex justify-between items-center"><span className="text-sm text-gray-500">Avg Accuracy</span><span className="font-semibold">{avgAccuracy}%</span></div>
              <Separator />
              <div className="flex justify-between items-center"><span className="text-sm text-gray-500">Total Knowledge</span><span className="font-semibold">1.2M items</span></div>
            </CardContent>
          </Card>

          <Card className={styles.card}>
            <CardHeader><CardTitle className="flex items-center gap-2"><GraduationCap className="w-5 h-5 text-amber-500" />Knowledge Index</CardTitle></CardHeader>
            <CardContent className="text-center">
              <div className="relative inline-block">
                <svg className="w-24 h-24 transform -rotate-90">
                  <circle cx="48" cy="48" r="40" stroke="#e5e7eb" strokeWidth="8" fill="none" />
                  <circle cx="48" cy="48" r="40" stroke="#10b981" strokeWidth="8" fill="none" strokeDasharray="251" strokeDashoffset="40" />
                </svg>
                <span className="absolute inset-0 flex items-center justify-center text-2xl font-bold">84</span>
              </div>
              <p className="mt-2 text-sm text-gray-500">Learning Efficiency</p>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
