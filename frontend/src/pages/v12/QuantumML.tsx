/**
 * QuantumML - Quantum AI Page
 * 量子机器学习 - 量子AI页面
 */

import React, { useState, useCallback } from 'react';
import {
  Brain, Cpu, Zap, Settings, RefreshCw, Play, Activity,
  Target, TrendingUp, ChevronRight, Layers, Terminal,
  Gauge, Eye, Network, Database,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Progress } from '@/components/ui/progress';
import { ScrollArea } from '@/components/ui/scroll-area';
import { useToast } from '@/hooks/use-toast';
import { cn } from '@/lib/utils';

interface QuantumModel {
  id: string; name: string; type: string; status: 'training' | 'ready' | 'evaluating';
  accuracy: number; qubits: number; layers: number; trainingTime: string;
}

interface QKernalExperiment {
  id: string; dataset: string; accuracy: number; classicalAcc: number; improvement: number;
}

const SAMPLE_MODELS: QuantumModel[] = [
  { id: '1', name: 'Quantum Neural Network', type: 'Classification', status: 'ready', accuracy: 94.5, qubits: 4, layers: 3, trainingTime: '2h 15m' },
  { id: '2', name: 'Quantum Support Vector Machine', type: 'Classification', status: 'training', accuracy: 0, qubits: 6, layers: 2, trainingTime: '45m' },
  { id: '3', name: 'Quantum Boltzmann Machine', type: 'Generation', status: 'ready', accuracy: 91.2, qubits: 8, layers: 4, trainingTime: '5h 30m' },
  { id: '4', name: 'Quantum Transformer', type: 'NLP', status: 'evaluating', accuracy: 88.7, qubits: 12, layers: 6, trainingTime: '8h 12m' },
];

const SAMPLE_EXPERIMENTS: QKernalExperiment[] = [
  { id: '1', dataset: 'MNIST Subset', accuracy: 96.2, classicalAcc: 94.1, improvement: 2.1 },
  { id: '2', dataset: 'Iris', accuracy: 98.0, classicalAcc: 97.5, improvement: 0.5 },
  { id: '3', dataset: 'Wine Quality', accuracy: 92.3, classicalAcc: 89.8, improvement: 2.5 },
];

const styles = {
  container: 'min-h-screen bg-gradient-to-br from-violet-50 via-white to-purple-50 dark:from-gray-900 dark:via-gray-800 dark:to-gray-900 p-6',
  header: 'flex items-center justify-between mb-8',
  title: 'text-4xl font-bold bg-gradient-to-r from-violet-600 to-purple-600 bg-clip-text text-transparent',
  subtitle: 'text-gray-600 dark:text-gray-300 mt-2',
  mainGrid: 'grid grid-cols-1 lg:grid-cols-3 gap-6',
  content: 'lg:col-span-2',
  card: 'bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm shadow-xl rounded-2xl border-0',
  modelCard: 'bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm shadow-xl rounded-2xl border-0 hover:shadow-2xl transition-all cursor-pointer',
  button: 'bg-gradient-to-r from-violet-600 to-purple-600 hover:from-violet-700 hover:to-purple-700 text-white shadow-lg',
};

export default function QuantumML() {
  const { toast } = useToast();
  const [models, setModels] = useState<QuantumModel[]>(SAMPLE_MODELS);
  const [experiments] = useState<QKernalExperiment[]>(SAMPLE_EXPERIMENTS);
  const [isTraining, setIsTraining] = useState(false);
  const [selectedModel, setSelectedModel] = useState<QuantumModel | null>(null);

  const handleTrain = useCallback(async (modelId: string) => {
    setModels(prev => prev.map(m => m.id === modelId ? { ...m, status: 'training' } : m));
    setIsTraining(true);
    toast({ title: 'Training Started', description: 'Training quantum machine learning model' });
    
    setTimeout(() => {
      setModels(prev => prev.map(m => m.id === modelId ? { ...m, status: 'ready', accuracy: Math.random() * 10 + 85 } : m));
      setIsTraining(false);
      toast({ title: 'Training Complete', description: 'Model has been trained successfully' });
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

  const avgAccuracy = models.filter(m => m.status === 'ready').reduce((a, m) => a + m.accuracy, 0) / Math.max(models.filter(m => m.status === 'ready').length, 1);
  const totalQubits = models.reduce((a, m) => a + m.qubits, 0);

  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <div>
          <h1 className={styles.title}><Brain className="inline mr-2 h-8 w-8" />Quantum Machine Learning</h1>
          <p className={styles.subtitle}>Combine quantum computing with machine learning algorithms</p>
        </div>
        <div className="flex items-center gap-3">
          <Button variant="outline"><Settings className="w-4 h-4 mr-2" />Configure</Button>
          <Button className={styles.button} disabled={isTraining}>
            <Zap className="w-4 h-4 mr-2" />New Model
          </Button>
        </div>
      </div>

      <div className={styles.mainGrid}>
        <div className={styles.content}>
          <Tabs defaultValue="models" className="space-y-6">
            <TabsList className="bg-white/60 dark:bg-gray-800/60 backdrop-blur-sm p-1 rounded-xl">
              <TabsTrigger value="models" className="px-4 py-2 rounded-lg data-[state=active]:bg-violet-100 data-[state=active]:text-violet-700">
                <Brain className="w-4 h-4 mr-2" />Models
              </TabsTrigger>
              <TabsTrigger value="experiments" className="px-4 py-2 rounded-lg data-[state=active]:bg-violet-100 data-[state=active]:text-violet-700">
                <Target className="w-4 h-4 mr-2" />Experiments
              </TabsTrigger>
              <TabsTrigger value="kernels" className="px-4 py-2 rounded-lg data-[state=active]:bg-violet-100 data-[state=active]:text-violet-700">
                <Layers className="w-4 h-4 mr-2" />QKernels
              </TabsTrigger>
            </TabsList>

            <TabsContent value="models">
              <div className="grid gap-4">
                {models.map((model) => (
                  <Card key={model.id} className={cn(styles.modelCard, selectedModel?.id === model.id && 'ring-2 ring-violet-500')} onClick={() => setSelectedModel(model)}>
                    <CardContent className="p-4">
                      <div className="flex items-center justify-between mb-3">
                        <div className="flex items-center gap-3">
                          <div className="w-10 h-10 rounded-xl bg-violet-100 dark:bg-violet-900/30 flex items-center justify-center">
                            <Brain className="w-5 h-5 text-violet-600" />
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
                          <p className="text-lg font-bold text-violet-600">{model.accuracy.toFixed(1)}%</p>
                          <p className="text-xs text-gray-500">Accuracy</p>
                        </div>
                        <div className="text-center">
                          <p className="text-lg font-bold">{model.qubits}</p>
                          <p className="text-xs text-gray-500">Qubits</p>
                        </div>
                        <div className="text-center">
                          <p className="text-lg font-bold">{model.layers}</p>
                          <p className="text-xs text-gray-500">Layers</p>
                        </div>
                        <div className="text-center">
                          <p className="text-lg font-bold">{model.trainingTime}</p>
                          <p className="text-xs text-gray-500">Training</p>
                        </div>
                      </div>
                      {model.status === 'training' && <Progress value={67} className="h-1.5" />}
                      {model.status === 'ready' && <Progress value={model.accuracy} className="h-1.5" />}
                      {model.status === 'pending' && (
                        <Button className="w-full mt-3" onClick={(e) => { e.stopPropagation(); handleTrain(model.id); }}>
                          <Play className="w-4 h-4 mr-2" />Train
                        </Button>
                      )}
                    </CardContent>
                  </Card>
                ))}
              </div>
            </TabsContent>

            <TabsContent value="experiments">
              <Card className={styles.card}>
                <CardHeader><CardTitle>Quantum Advantage Experiments</CardTitle></CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    {experiments.map((exp) => (
                      <div key={exp.id} className="p-4 bg-gray-50 dark:bg-gray-900/50 rounded-lg">
                        <div className="flex items-center justify-between mb-2">
                          <span className="font-medium">{exp.dataset}</span>
                          <Badge className={exp.improvement > 1 ? 'bg-green-100 text-green-700' : 'bg-gray-100 text-gray-700'}>
                            +{exp.improvement}% vs classical
                          </Badge>
                        </div>
                        <div className="flex items-center gap-4 text-sm">
                          <span className="text-gray-500">Quantum: <strong>{exp.accuracy}%</strong></span>
                          <span className="text-gray-500">Classical: <strong>{exp.classicalAcc}%</strong></span>
                          <Progress value={exp.accuracy} className="flex-1 h-1.5" />
                        </div>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="kernels">
              <Card className={styles.card}>
                <CardContent className="p-8 text-center">
                  <Layers className="w-16 h-16 mx-auto mb-4 text-violet-300" />
                  <h3 className="text-xl font-semibold mb-2">Quantum Kernels</h3>
                  <p className="text-gray-500">Design and test quantum kernels for ML</p>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </div>

        <div className="space-y-6">
          <Card className={styles.card}>
            <CardHeader><CardTitle className="flex items-center gap-2"><Activity className="w-5 h-5 text-violet-500" />Model Statistics</CardTitle></CardHeader>
            <CardContent className="space-y-4">
              <div className="flex justify-between items-center"><span className="text-sm text-gray-500">Ready Models</span><span className="font-semibold">{models.filter(m => m.status === 'ready').length}</span></div>
              <div className="flex justify-between items-center"><span className="text-sm text-gray-500">Avg Accuracy</span><span className="font-semibold">{avgAccuracy.toFixed(1)}%</span></div>
              <div className="flex justify-between items-center"><span className="text-sm text-gray-500">Total Qubits</span><span className="font-semibold">{totalQubits}</span></div>
            </CardContent>
          </Card>

          <Card className={styles.card}>
            <CardHeader><CardTitle className="flex items-center gap-2"><Zap className="w-5 h-5 text-amber-500" />Quantum Advantage</CardTitle></CardHeader>
            <CardContent className="text-center">
              <div className="relative inline-block">
                <svg className="w-24 h-24 transform -rotate-90">
                  <circle cx="48" cy="48" r="40" stroke="#e5e7eb" strokeWidth="8" fill="none" />
                  <circle cx="48" cy="48" r="40" stroke="#8b5cf6" strokeWidth="8" fill="none" strokeDasharray="251" strokeDashoffset="25" />
                </svg>
                <span className="absolute inset-0 flex items-center justify-center text-2xl font-bold">90</span>
              </div>
              <p className="mt-2 text-sm text-gray-500">Advantage Score</p>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
