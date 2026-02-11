/**
 * QuantumSimulator - Quantum AI Page
 * 量子模拟器 - 量子AI页面
 */

import React, { useState, useCallback } from 'react';
import {
  Atom, Zap, RefreshCw, Play, Pause, Settings, Cpu,
  Activity, TrendingUp, ChevronRight, Hexagon, Layers,
  Terminal, Database, Globe, Eye,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Progress } from '@/components/ui/progress';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { useToast } from '@/hooks/use-toast';
import { cn } from '@/lib/utils';

interface QuantumState {
  id: string; name: string; qubits: number; status: 'idle' | 'running' | 'completed';
  progress: number; fidelity: number; entanglement: number;
  circuit: { gate: string; target: number; depth: number }[];
}

interface QuantumResult {
  id: string; state: string; probability: number; amplitude: { real: number; imag: number };
}

const SAMPLE_STATES: QuantumState[] = [
  { id: '1', name: 'Grover Search', qubits: 4, status: 'completed', progress: 100, fidelity: 0.98, entanglement: 0.95, circuit: [{ gate: 'H', target: 0, depth: 1 }, { gate: 'Oracle', target: 1, depth: 2 }, { gate: 'Diffusion', target: 2, depth: 3 }] },
  { id: '2', name: 'Quantum Teleportation', qubits: 3, status: 'running', progress: 67, fidelity: 0.94, entanglement: 0.89, circuit: [{ gate: 'H', target: 0, depth: 1 }, { gate: 'CNOT', target: 1, depth: 2 }] },
  { id: '3', name: 'Variational Quantum Eigensolver', qubits: 8, status: 'idle', progress: 0, fidelity: 0.0, entanglement: 0.0, circuit: [] },
  { id: '4', name: 'QAOA Optimization', qubits: 6, status: 'idle', progress: 0, fidelity: 0.0, entanglement: 0.0, circuit: [] },
];

const SAMPLE_RESULTS: QuantumResult[] = [
  { id: '1', state: '|0000⟩', probability: 0.125, amplitude: { real: 0.353, imag: 0 } },
  { id: '2', state: '|0001⟩', probability: 0.125, amplitude: { real: 0.353, imag: 0 } },
  { id: '3', state: '|0010⟩', probability: 0.125, amplitude: { real: 0.353, imag: 0 } },
  { id: '4', state: '|1101⟩', probability: 0.062, amplitude: { real: -0.250, imag: 0 } },
  { id: '5', state: '|1111⟩', probability: 0.062, amplitude: { real: -0.250, imag: 0 } },
];

const GATES = ['H', 'X', 'Y', 'Z', 'CNOT', 'RZ', 'RX', 'RY', 'SWAP', 'T', 'S'];

const styles = {
  container: 'min-h-screen bg-gradient-to-br from-purple-50 via-white to-indigo-50 dark:from-gray-900 dark:via-gray-800 dark:to-gray-900 p-6',
  header: 'flex items-center justify-between mb-8',
  title: 'text-4xl font-bold bg-gradient-to-r from-purple-600 to-indigo-600 bg-clip-text text-transparent',
  subtitle: 'text-gray-600 dark:text-gray-300 mt-2',
  mainGrid: 'grid grid-cols-1 lg:grid-cols-3 gap-6',
  content: 'lg:col-span-2',
  card: 'bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm shadow-xl rounded-2xl border-0',
  stateCard: 'bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm shadow-xl rounded-2xl border-0 hover:shadow-2xl transition-all cursor-pointer',
  button: 'bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-700 hover:to-indigo-700 text-white shadow-lg',
};

export default function QuantumSimulator() {
  const { toast } = useToast();
  const [states, setStates] = useState<QuantumState[]>(SAMPLE_STATES);
  const [results] = useState<QuantumResult[]>(SAMPLE_RESULTS);
  const [isSimulating, setIsSimulating] = useState(false);
  const [selectedState, setSelectedState] = useState<QuantumState | null>(null);
  const [selectedGate, setSelectedGate] = useState('H');

  const handleSimulate = useCallback(async (stateId: string) => {
    setStates(prev => prev.map(s => s.id === stateId ? { ...s, status: 'running' } : s));
    setIsSimulating(true);
    toast({ title: 'Simulation Started', description: 'Running quantum circuit simulation' });
    
    let progress = 0;
    const interval = setInterval(() => {
      progress += 10;
      setStates(prev => prev.map(s => s.id === stateId ? { ...s, progress: Math.min(progress, 100) } : s));
      if (progress >= 100) {
        clearInterval(interval);
        setStates(prev => prev.map(s => s.id === stateId ? { ...s, status: 'completed', fidelity: 0.96, entanglement: 0.92 } : s));
        setIsSimulating(false);
        toast({ title: 'Simulation Complete', description: 'Quantum state has been simulated' });
      }
    }, 500);
  }, [toast]);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'running': return 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400';
      case 'completed': return 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400';
      case 'idle': return 'bg-gray-100 text-gray-700 dark:bg-gray-900/30 dark:text-gray-400';
      default: return 'bg-gray-100 text-gray-700';
    }
  };

  const runningStates = states.filter(s => s.status === 'running').length;
  const completedStates = states.filter(s => s.status === 'completed').length;
  const avgFidelity = states.filter(s => s.status === 'completed').reduce((a, s) => a + s.fidelity, 0) / Math.max(completedStates, 1);

  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <div>
          <h1 className={styles.title}><Atom className="inline mr-2 h-8 w-8" />Quantum Simulator</h1>
          <p className={styles.subtitle}>Simulate quantum circuits and explore quantum algorithms</p>
        </div>
        <div className="flex items-center gap-3">
          <Button variant="outline"><Settings className="w-4 h-4 mr-2" />Configure</Button>
          <Button className={styles.button} disabled={isSimulating}>
            <Zap className="w-4 h-4 mr-2" />New Circuit
          </Button>
        </div>
      </div>

      <div className={styles.mainGrid}>
        <div className={styles.content}>
          <Tabs defaultValue="circuits" className="space-y-6">
            <TabsList className="bg-white/60 dark:bg-gray-800/60 backdrop-blur-sm p-1 rounded-xl">
              <TabsTrigger value="circuits" className="px-4 py-2 rounded-lg data-[state=active]:bg-purple-100 data-[state=active]:text-purple-700">
                <Layers className="w-4 h-4 mr-2" />Circuits
              </TabsTrigger>
              <TabsTrigger value="composer" className="px-4 py-2 rounded-lg data-[state=active]:bg-purple-100 data-[state=active]:text-purple-700">
                <Terminal className="w-4 h-4 mr-2" />Circuit Composer
              </TabsTrigger>
              <TabsTrigger value="results" className="px-4 py-2 rounded-lg data-[state=active]:bg-purple-100 data-[state=active]:text-purple-700">
                <Activity className="w-4 h-4 mr-2" />Results
              </TabsTrigger>
            </TabsList>

            <TabsContent value="circuits">
              <div className="grid gap-4">
                {states.map((state) => (
                  <Card key={state.id} className={cn(styles.stateCard, selectedState?.id === state.id && 'ring-2 ring-purple-500')} onClick={() => setSelectedState(state)}>
                    <CardContent className="p-4">
                      <div className="flex items-center justify-between mb-3">
                        <div>
                          <h3 className="font-semibold">{state.name}</h3>
                          <p className="text-sm text-gray-500">{state.qubits} qubits</p>
                        </div>
                        <Badge className={getStatusColor(state.status)}>{state.status}</Badge>
                      </div>
                      {state.status !== 'idle' && (
                        <div className="grid grid-cols-3 gap-4 mb-3">
                          <div className="text-center">
                            <p className="text-lg font-bold text-purple-600">{state.progress}%</p>
                            <p className="text-xs text-gray-500">Progress</p>
                          </div>
                          <div className="text-center">
                            <p className="text-lg font-bold">{state.fidelity.toFixed(2)}</p>
                            <p className="text-xs text-gray-500">Fidelity</p>
                          </div>
                          <div className="text-center">
                            <p className="text-lg font-bold">{state.entanglement.toFixed(2)}</p>
                            <p className="text-xs text-gray-500">Entanglement</p>
                          </div>
                        </div>
                      )}
                      <Progress value={state.progress} className="h-2" />
                      {state.status === 'idle' && (
                        <Button className="w-full mt-3" onClick={(e) => { e.stopPropagation(); handleSimulate(state.id); }}>
                          <Play className="w-4 h-4 mr-2" />Run Simulation
                        </Button>
                      )}
                    </CardContent>
                  </Card>
                ))}
              </div>
            </TabsContent>

            <TabsContent value="composer">
              <Card className={styles.card}>
                <CardHeader>
                  <CardTitle>Circuit Composer</CardTitle>
                  <CardDescription>Build your quantum circuit</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="flex gap-2 overflow-x-auto pb-2">
                    {GATES.map((gate) => (
                      <Button key={gate} variant={selectedGate === gate ? 'default' : 'outline'} size="sm" onClick={() => setSelectedGate(gate)}>
                        {gate}
                      </Button>
                    ))}
                  </div>
                  <div className="bg-gray-100 dark:bg-gray-900 rounded-lg p-4 min-h-[200px] flex items-center justify-center">
                    <p className="text-gray-500">Drag and drop gates to build your circuit</p>
                  </div>
                  <Button className="w-full" disabled={isSimulating}>
                    <Play className="w-4 h-4 mr-2" />Run Circuit
                  </Button>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="results">
              <Card className={styles.card}>
                <CardHeader><CardTitle>Measurement Results</CardTitle></CardHeader>
                <CardContent>
                  <div className="space-y-2">
                    {results.map((result) => (
                      <div key={result.id} className="flex items-center justify-between p-2 bg-gray-50 dark:bg-gray-900/50 rounded-lg">
                        <span className="font-mono">{result.state}</span>
                        <div className="flex items-center gap-4">
                          <span className="text-sm text-gray-500">p = {result.probability}</span>
                          <Progress value={result.probability * 100} className="w-24 h-1.5" />
                        </div>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </div>

        <div className="space-y-6">
          <Card className={styles.card}>
            <CardHeader><CardTitle className="flex items-center gap-2"><Cpu className="w-5 h-5 text-purple-500" />Quantum Status</CardTitle></CardHeader>
            <CardContent className="space-y-4">
              <div className="flex justify-between items-center"><span className="text-sm text-gray-500">Running</span><span className="font-semibold">{runningStates}</span></div>
              <div className="flex justify-between items-center"><span className="text-sm text-gray-500">Completed</span><span className="font-semibold text-green-500">{completedStates}</span></div>
              <div className="flex justify-between items-center"><span className="text-sm text-gray-500">Avg Fidelity</span><span className="font-semibold">{avgFidelity.toFixed(2)}</span></div>
            </CardContent>
          </Card>

          <Card className={styles.card}>
            <CardHeader><CardTitle className="flex items-center gap-2"><Zap className="w-5 h-5 text-amber-500" />Performance</CardTitle></CardHeader>
            <CardContent className="text-center">
              <div className="relative inline-block">
                <svg className="w-24 h-24 transform -rotate-90">
                  <circle cx="48" cy="48" r="40" stroke="#e5e7eb" strokeWidth="8" fill="none" />
                  <circle cx="48" cy="48" r="40" stroke="#8b5cf6" strokeWidth="8" fill="none" strokeDasharray="251" strokeDashoffset="30" />
                </svg>
                <span className="absolute inset-0 flex items-center justify-center text-2xl font-bold">88</span>
              </div>
              <p className="mt-2 text-sm text-gray-500">Simulation Efficiency</p>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
