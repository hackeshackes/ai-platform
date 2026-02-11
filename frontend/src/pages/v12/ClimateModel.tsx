/**
 * ClimateModel - Cosmic AI Page
 * 气候模型 - 宇宙级AI页面
 */

import React, { useState, useCallback } from 'react';
import {
  Globe, Thermometer, Cloud, Wind, Droplets, Sun,
  Activity, TrendingUp, RefreshCw, Settings, Play, AlertTriangle,
  Map, Calendar, Clock, Target,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Progress } from '@/components/ui/progress';
import { ScrollArea } from '@/components/ui/scroll-area';
import { useToast } from '@/hooks/use-toast';
import { cn } from '@/lib/utils';

interface ClimateMetric {
  id: string; name: string; value: number; unit; status: 'normal' | 'warning' | 'critical';
  trend: 'up' | 'down' | 'stable'; historical: number[];
}

interface Prediction {
  id: string; region: string; timeframe: string; temperatureChange: number;
  precipitationChange: number; confidence: number; risk: 'low' | 'medium' | 'high';
}

const SAMPLE_METRICS: ClimateMetric[] = [
  { id: '1', name: 'Global Temperature', value: 1.2, unit: '°C above baseline', status: 'warning', trend: 'up', historical: [0.5, 0.7, 0.9, 1.0, 1.2] },
  { id: '2', name: 'CO₂ Concentration', value: 421, unit: 'ppm', status: 'critical', trend: 'up', historical: [400, 408, 415, 419, 421] },
  { id: '3', name: 'Sea Level Rise', value: 3.4, unit: 'mm/year', status: 'warning', trend: 'up', historical: [2.8, 3.0, 3.2, 3.3, 3.4] },
  { id: '4', name: 'Arctic Ice Extent', value: 4.2, unit: 'million km²', status: 'normal', trend: 'down', historical: [6.0, 5.5, 5.0, 4.5, 4.2] },
  { id: '5', name: 'Ocean Acidity', value: 8.1, unit: 'pH', status: 'warning', trend: 'down', historical: [8.2, 8.15, 8.12, 8.11, 8.1] },
];

const SAMPLE_PREDICTIONS: Prediction[] = [
  { id: '1', region: 'North America', timeframe: '2050', temperatureChange: 2.5, precipitationChange: -5, confidence: 85, risk: 'medium' },
  { id: '2', region: 'Europe', timeframe: '2100', temperatureChange: 4.1, precipitationChange: 10, confidence: 75, risk: 'high' },
  { id: '3', region: 'Southeast Asia', timeframe: '2070', temperatureChange: 2.0, precipitationChange: 15, confidence: 80, risk: 'high' },
  { id: '4', region: 'Australia', timeframe: '2050', temperatureChange: 3.0, precipitationChange: -15, confidence: 82, risk: 'high' },
];

const styles = {
  container: 'min-h-screen bg-gradient-to-br from-blue-50 via-cyan-50 to-teal-50 dark:from-gray-900 dark:via-gray-800 dark:to-gray-900 p-6',
  header: 'flex items-center justify-between mb-8',
  title: 'text-4xl font-bold bg-gradient-to-r from-blue-600 to-cyan-600 bg-clip-text text-transparent',
  subtitle: 'text-gray-600 dark:text-gray-300 mt-2',
  mainGrid: 'grid grid-cols-1 lg:grid-cols-3 gap-6',
  content: 'lg:col-span-2',
  card: 'bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm shadow-xl rounded-2xl border-0',
  metricCard: 'bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm shadow-xl rounded-2xl border-0 hover:shadow-2xl transition-all',
  button: 'bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-700 hover:to-cyan-700 text-white shadow-lg',
};

export default function ClimateModel() {
  const { toast } = useToast();
  const [metrics] = useState<ClimateMetric[]>(SAMPLE_METRICS);
  const [predictions] = useState<Prediction[]>(SAMPLE_PREDICTIONS);
  const [isSimulating, setIsSimulating] = useState(false);

  const handleSimulate = useCallback(async () => {
    setIsSimulating(true);
    toast({ title: 'Simulation Started', description: 'Running climate model simulation' });
    setTimeout(() => {
      setIsSimulating(false);
      toast({ title: 'Simulation Complete', description: 'Climate predictions updated' });
    }, 5000);
  }, [toast]);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'critical': return 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400';
      case 'warning': return 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400';
      case 'normal': return 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400';
      default: return 'bg-gray-100 text-gray-700';
    }
  };

  const getRiskColor = (risk: string) => {
    switch (risk) {
      case 'high': return 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400';
      case 'medium': return 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400';
      case 'low': return 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400';
      default: return 'bg-gray-100 text-gray-700';
    }
  };

  const criticalMetrics = metrics.filter(m => m.status === 'critical').length;
  const warningMetrics = metrics.filter(m => m.status === 'warning').length;

  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <div>
          <h1 className={styles.title}><Globe className="inline mr-2 h-8 w-8" />Climate Model</h1>
          <p className={styles.subtitle}>AI-powered global climate simulation and prediction</p>
        </div>
        <div className="flex items-center gap-3">
          <Button variant="outline"><Settings className="w-4 h-4 mr-2" />Configure</Button>
          <Button className={styles.button} onClick={handleSimulate} disabled={isSimulating}>
            {isSimulating ? <RefreshCw className="w-4 h-4 mr-2 animate-spin" /> : <Play className="w-4 h-4 mr-2" />}
            {isSimulating ? 'Simulating...' : 'Run Simulation'}
          </Button>
        </div>
      </div>

      {(criticalMetrics > 0 || warningMetrics > 0) && (
        <div className="mb-6 p-4 bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800 rounded-xl flex items-center gap-3">
          <AlertTriangle className="w-6 h-6 text-amber-500" />
          <div>
            <p className="font-semibold text-amber-700 dark:text-amber-400">Climate Alert</p>
            <p className="text-sm text-amber-600 dark:text-amber-300">{criticalMetrics} critical, {warningMetrics} warning indicators</p>
          </div>
        </div>
      )}

      <div className={styles.mainGrid}>
        <div className={styles.content}>
          <Tabs defaultValue="metrics" className="space-y-6">
            <TabsList className="bg-white/60 dark:bg-gray-800/60 backdrop-blur-sm p-1 rounded-xl">
              <TabsTrigger value="metrics" className="px-4 py-2 rounded-lg data-[state=active]:bg-blue-100 data-[state=active]:text-blue-700">
                <Thermometer className="w-4 h-4 mr-2" />Metrics
              </TabsTrigger>
              <TabsTrigger value="predictions" className="px-4 py-2 rounded-lg data-[state=active]:bg-blue-100 data-[state=active]:text-blue-700">
                <Target className="w-4 h-4 mr-2" />Predictions
              </TabsTrigger>
              <TabsTrigger value="scenarios" className="px-4 py-2 rounded-lg data-[state=active]:bg-blue-100 data-[state=active]:text-blue-700">
                <Globe className="w-4 h-4 mr-2" />Scenarios
              </TabsTrigger>
            </TabsList>

            <TabsContent value="metrics">
              <div className="grid grid-cols-2 gap-4">
                {metrics.map((metric) => (
                  <Card key={metric.id} className={styles.metricCard}>
                    <CardContent className="p-4">
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-sm text-gray-500">{metric.name}</span>
                        <Badge className={getStatusColor(metric.status)}>{metric.status}</Badge>
                      </div>
                      <div className="flex items-end justify-between">
                        <span className="text-2xl font-bold">{metric.value}{metric.unit}</span>
                        <div className={cn('flex items-center gap-1 text-sm', metric.trend === 'up' ? 'text-red-500' : metric.trend === 'down' ? 'text-green-500' : 'text-gray-400')}>
                          {metric.trend === 'up' ? <TrendingUp className="w-4 h-4" /> : metric.trend === 'down' ? <TrendingDown className="w-4 h-4" /> : null}
                        </div>
                      </div>
                      <div className="flex gap-0.5 mt-2">
                        {metric.historical.map((v, i) => (
                          <div key={i} className="flex-1 bg-gray-200 dark:bg-gray-700 rounded-sm" style={{ height: 4 }}>
                            <div className="bg-blue-500 rounded-sm" style={{ height: '100%', width: `${(v / Math.max(...metric.historical)) * 100}%` }} />
                          </div>
                        ))}
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            </TabsContent>

            <TabsContent value="predictions">
              <div className="grid gap-4">
                {predictions.map((pred) => (
                  <Card key={pred.id} className={styles.card}>
                    <CardContent className="p-4">
                      <div className="flex items-start justify-between mb-3">
                        <div>
                          <h3 className="font-semibold">{pred.region}</h3>
                          <p className="text-sm text-gray-500">{pred.timeframe}</p>
                        </div>
                        <Badge className={getRiskColor(pred.risk)}>{pred.risk} risk</Badge>
                      </div>
                      <div className="grid grid-cols-3 gap-4 mb-3">
                        <div className="text-center">
                          <p className="text-lg font-bold text-red-600">+{pred.temperatureChange}°C</p>
                          <p className="text-xs text-gray-500">Temperature</p>
                        </div>
                        <div className="text-center">
                          <p className="text-lg font-bold">{pred.precipitationChange}%</p>
                          <p className="text-xs text-gray-500">Precipitation</p>
                        </div>
                        <div className="text-center">
                          <p className="text-lg font-bold">{pred.confidence}%</p>
                          <p className="text-xs text-gray-500">Confidence</p>
                        </div>
                      </div>
                      <Progress value={pred.confidence} className="h-1.5" />
                    </CardContent>
                  </Card>
                ))}
              </div>
            </TabsContent>

            <TabsContent value="scenarios">
              <Card className={styles.card}>
                <CardContent className="p-8 text-center">
                  <Globe className="w-16 h-16 mx-auto mb-4 text-blue-300" />
                  <h3 className="text-xl font-semibold mb-2">Climate Scenarios</h3>
                  <p className="text-gray-500">Explore different climate scenarios and interventions</p>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </div>

        <div className="space-y-6">
          <Card className={styles.card}>
            <CardHeader><CardTitle className="flex items-center gap-2"><Activity className="w-5 h-5 text-blue-500" />Climate Status</CardTitle></CardHeader>
 className="space-y-4">
              <div className="flex justify-between items-center"><span className="text-sm text-gray            <CardContent-500">Critical</span><span className="font-semibold text-red-500">{criticalMetrics}</span></div>
              <div className="flex justify-between items-center"><span className="text-sm text-gray-500">Warning</span><span className="font-semibold text-yellow-500">{warningMetrics}</span></div>
              <div className="flex justify-between items-center"><span className="text-sm text-gray-500">Normal</span><span className="font-semibold text-green-500">{metrics.length - criticalMetrics - warningMetrics}</span></div>
            </CardContent>
          </Card>

          <Card className={styles.card}>
            <CardHeader><CardTitle className="flex items-center gap-2"><Target className="w-5 h-5 text-amber-500" />Climate Index</CardTitle></CardHeader>
            <CardContent className="text-center">
              <div className="relative inline-block">
                <svg className="w-24 h-24 transform -rotate-90">
                  <circle cx="48" cy="48" r="40" stroke="#e5e7eb" strokeWidth="8" fill="none" />
                  <circle cx="48" cy="48" r="40" stroke="#f59e0b" strokeWidth="8" fill="none" strokeDasharray="251" strokeDashoffset="60" />
                </svg>
                <span className="absolute inset-0 flex items-center justify-center text-2xl font-bold">60</span>
              </div>
              <p className="mt-2 text-sm text-gray-500">Climate Health</p>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
