/**
 * MetaLearning - Super Intelligence Page
 * 元学习框架 - 超级智能页面
 */

import React, { useState, useCallback } from 'react';
import {
  Brain, Sparkles, Cpu, GitBranch, Layers, Zap, Target,
  TrendingUp, RefreshCw, Settings, Play, Pause, Eye,
  BookOpen, Code, Database, Network, ChevronRight, CheckCircle,
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

interface Model {
  id: string; name: string; type: string; status: 'training' | 'ready' | 'evaluating';
  accuracy: number; loss: number; tasks: number; iterations: number;
  metaParams: { learningRate: number; batchSize: number; epochs: number };
}

interface Task {
  id: string; name: string; domain: string; difficulty: 'easy' | 'medium' | 'hard';
  status: 'pending' | 'training' | 'completed'; accuracy: number; samples: number;
}

const SAMPLE_MODELS: Model[] = [
  { id: '1', name: 'Meta-Learner v1.0', type: 'MAML', status: 'ready', accuracy: 94.5, loss: 0.12, tasks: 50, iterations: 10000, metaParams: { learningRate: 0.001, batchSize: 32, epochs: 100 } },
  { id: '2', name: 'ProtoNet Baseline', type: 'Prototypical', status: 'training', accuracy: 87.2, loss: 0.34, tasks: 25, iterations: 4500, metaParams: { learningRate: 0.01, batchSize: 16, epochs: 50 } },
  { id: '3', name: 'Reptile Advanced', type: 'Reptile', status: 'ready', accuracy: 92.1, loss: 0.18, tasks: 40, iterations: 8000, metaParams: { learningRate: 0.005, batchSize: 64, epochs: 75 } },
];

const SAMPLE_TASKS: Task[] = [
  { id: '1', name: 'Image Classification', domain: 'Computer Vision', difficulty: 'medium', status: 'completed', accuracy: 96.2, samples: 1000 },
  { id: '2', name: 'Sentiment Analysis', domain: 'NLP', difficulty: 'easy', status: 'completed', accuracy: 94.8, samples: 500 },
  { id: '3', name: 'Object Detection', domain: 'Computer Vision', difficulty: 'hard', status: 'training', accuracy: 82.5, samples: 2000 },
  { id: '4', name: 'Machine Translation', domain: 'NLP', difficulty: 'hard', status: 'pending', accuracy: 0, samples: 1500 },
];

const styles = {
  container: 'min-h-screen bg-gradient-to-br from-indigo-50 via-white to-purple-50 dark:from-gray-900 dark:via-gray-800 dark:to-gray-900 p-6',
  header: 'flex items-center justify-between mb-8',
  title: 'text-4xl font-bold bg-gradient-to-r from-indigo-600 to-purple-600 bg-clip-text text-transparent',
  subtitle: 'text-gray-600 dark:text-gray-300 mt-2',
  mainGrid: 'grid grid-cols-1 lg:grid-cols-3 gap-6',
  content: 'lg:col-span-2',
  card: 'bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm shadow-xl rounded-2xl border-0',
  modelCard: 'bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm shadow-xl rounded-2xl border-0 hover:shadow-2xl transition-all',
  button: 'bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-700 hover:to-purple-700 text-white shadow-lg',
};

export default function MetaLearning() {
  const { toast } = useToast();
  const [models, setModels] = useState<Model[]>(SAMPLE_MODELS);
  const [tasks, setTasks] = useState<Task[]>(SAMPLE_TASKS);
  const [isTraining, setIsTraining] = useState(false);
  const [selectedModel, setSelectedModel] = useState<Model | null>(null);

  const handleTrain = useCallback(async () => {
    setIsTraining(true);
    toast({ title: 'Training Started', description: 'Meta-learning training has begun' });
    setTimeout(() => {
      setIsTraining(false);
      toast({ title: 'Training Complete', description: 'Model has been updated' });
    }, 5000);
  }, [toast]);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'ready': return 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400';
      case 'training': return 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400';
      case 'evaluating': return 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400';
      default: return 'bg-gray-100 text-gray-700';
    }
  };

  const getDifficultyColor = (difficulty: string) => {
    switch (difficulty) {
      case 'easy': return 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400';
      case 'medium': return 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400';
      case 'hard': return 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400';
      default: return 'bg-gray-100 text-gray-700';
    }
  };

  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <div>
          <h1 className={styles.title}><Brain className="inline mr-2 h-8 w-8" />Meta-Learning Framework</h1>
          <p className={styles.subtitle}>Learn to learn - AI that improves its learning capabilities</p>
        </div>
        <div className="flex items-center gap-3">
          <Button variant="outline"><Settings className="w-4 h-4 mr-2" />Configure</Button>
          <Button className={styles.button} onClick={handleTrain} disabled={isTraining}>
            {isTraining ? <RefreshCw className="w-4 h-4 mr-2 animate-spin" /> : <Zap className="w-4 h-4 mr-2" />}
            {isTraining ? 'Training...' : 'Train Meta-Learner'}
          </Button>
        </div>
      </div>

      <div className={styles.mainGrid}>
        <div className={styles.content}>
          <Tabs defaultValue="models" className="space-y-6">
            <TabsList className="bg-white/60 dark:bg-gray-800/60 backdrop-blur-sm p-1 rounded-xl">
              <TabsTrigger value="models" className="px-4 py-2 rounded-lg data-[state=active]:bg-indigo-100 data-[state=active]:text-indigo-700">
                <Brain className="w-4 h-4 mr-2" />Models
              </TabsTrigger>
              <TabsTrigger value="tasks" className="px-4 py-2 rounded-lg data-[state=active]:bg-indigo-100 data-[state=active]:text-indigo-700">
                <Target className="w-4 h-4 mr-2" />Tasks
              </TabsTrigger>
              <TabsTrigger value="analysis" className="px-4 py-2 rounded-lg data-[state=active]:bg-indigo-100 data-[state=active]:text-indigo-700">
                <Eye className="w-4 h-4 mr-2" />Analysis
              </TabsTrigger>
            </TabsList>

            <TabsContent value="models">
              <div className="grid gap-4">
                {models.map((model) => (
                  <Card key={model.id} className={cn(styles.modelCard, selectedModel?.id === model.id && 'ring-2 ring-indigo-500')} onClick={() => setSelectedModel(model)}>
                    <CardContent className="p-4">
                      <div className="flex items-center justify-between mb-3">
                        <div className="flex items-center gap-3">
                          <div className="w-10 h-10 rounded-xl bg-indigo-100 dark:bg-indigo-900/30 flex items-center justify-center">
                            <Brain className="w-5 h-5 text-indigo-600" />
                          </div>
                          <div>
                            <h3 className="font-semibold">{model.name}</h3>
                            <p className="text-sm text-gray-500">{model.type}</p>
                          </div>
                        </div>
                        <Badge className={getStatusColor(model.status)}>{model.status}</Badge>
                      </div>
                      <div className="grid grid-cols-4 gap-4 mb-3">
                        <div className="text-center">
                          <p className="text-2xl font-bold text-green-600">{model.accuracy}%</p>
                          <p className="text-xs text-gray-500">Accuracy</p>
                        </div>
                        <div className="text-center">
                          <p className="text-2xl font-bold">{model.loss}</p>
                          <p className="text-xs text-gray-500">Loss</p>
                        </div>
                        <div className="text-center">
                          <p className="text-2xl font-bold">{model.tasks}</p>
                          <p className="text-xs text-gray-500">Tasks</p>
                        </div>
                        <div className="text-center">
                          <p className="text-2xl font-bold">{model.iterations.toLocaleString()}</p>
                          <p className="text-xs text-gray-500">Iterations</p>
                        </div>
                      </div>
                      <Progress value={model.status === 'training' ? 45 : 100} className="h-1" />
                    </CardContent>
                  </Card>
                ))}
              </div>
            </TabsContent>

            <TabsContent value="tasks">
              <div className="grid gap-4">
                {tasks.map((task) => (
                  <Card key={task.id} className={styles.card}>
                    <CardContent className="p-4">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-3">
                          <div className="w-10 h-10 rounded-xl bg-purple-100 dark:bg-purple-900/30 flex items-center justify-center">
                            <Target className="w-5 h-5 text-purple-600" />
                          </div>
                          <div>
                            <h3 className="font-semibold">{task.name}</h3>
                            <p className="text-sm text-gray-500">{task.domain} • {task.samples.toLocaleString()} samples</p>
                          </div>
                        </div>
                        <div className="flex items-center gap-3">
                          <Badge className={getDifficultyColor(task.difficulty)}>{task.difficulty}</Badge>
                          {task.status === 'completed' && <Badge className="bg-green-100 text-green-700">{task.accuracy}%</Badge>}
                          {task.status === 'training' && <div className="w-20"><Progress value={65} className="h-1" /></div>}
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            </TabsContent>

            <TabsContent value="analysis">
              <Card className={styles.card}>
                <CardContent className="p-8 text-center">
                  <Eye className="w-16 h-16 mx-auto mb-4 text-indigo-300" />
                  <h3 className="text-xl font-semibold mb-2">Meta-Learning Analysis</h3>
                  <p className="text-gray-500">View detailed learning curves and generalization analysis</p>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </div>

        <div className="space-y-6">
          <Card className={styles.card}>
            <CardHeader><CardTitle className="flex items-center gap-2"><Zap className="w-5 h-5 text-indigo-500" />Meta-Parameters</CardTitle></CardHeader>
            <CardContent className="space-y-4">
              {selectedModel ? (
                <>
                  <div><label className="text-sm text-gray-500">Learning Rate</label><p className="font-mono">{selectedModel.metaParams.learningRate}</p></div>
                  <div><label className="text-sm text-gray-500">Batch Size</label><p className="font-mono">{selectedModel.metaParams.batchSize}</p></div>
                  <div><label className="text-sm text-gray-500">Epochs</label><p className="font-mono">{selectedModel.metaParams.epochs}</p></div>
                </>
              ) : (
                <p className="text-gray-500 text-sm">Select a model to view parameters</p>
              )}
            </CardContent>
          </Card>

          <Card className={styles.card}>
            <CardHeader><CardTitle className="flex items-center gap-2"><TrendingUp className="w-5 h-5 text-green-500" />Performance</CardTitle></CardHeader>
            <CardContent className="space-y-4">
              <div className="flex justify-between items-center"><span className="text-sm text-gray-500">Avg Accuracy</span><span className="font-semibold text-green-600">91.3%</span></div>
              <div className="flex justify-between items-center"><span className="text-sm text-gray-500">Tasks Learned</span><span className="font-semibold">{tasks.filter(t => t.status === 'completed').length}</span></div>
              <div className="flex justify-between items-center"><span className="text-sm text-gray-500">In Progress</span><span className="font-semibold text-blue-500">{tasks.filter(t => t.status === 'training').length}</span></div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
