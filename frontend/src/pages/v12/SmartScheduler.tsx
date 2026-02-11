/**
 * Smart Scheduler - Hyperautomation Page
 * 智能调度 - 超自动化页面
 */

import React, { useState, useCallback } from 'react';
import {
  Calendar, Clock, Play, Pause, RefreshCw, Settings, Plus, Trash2,
  Zap, Sun, Moon, Repeat, Bell, Timer, CalendarDays, ChevronLeft,
  ChevronRight, MoreHorizontal, CheckCircle, AlertCircle,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Switch } from '@/components/ui/switch';
import { Input } from '@/components/ui/input';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Separator } from '@/components/ui/separator';
import { useToast } from '@/hooks/use-toast';
import { cn } from '@/lib/utils';

interface ScheduledTask {
  id: string; name: string; description: string;
  type: 'job' | 'workflow' | 'backup' | 'report' | 'cleanup';
  schedule: string; nextRun: Date; lastRun?: Date;
  status: 'active' | 'paused' | 'running' | 'failed';
  duration?: string;
  history: { status: 'success' | 'failed' | 'skipped'; timestamp: Date }[];
}

const SAMPLE_TASKS: ScheduledTask[] = [
  { id: '1', name: 'Daily Data Backup', description: 'Automatically backup all data stores', type: 'backup', schedule: '0 2 * * *', nextRun: new Date(Date.now() + 3600000), lastRun: new Date(Date.now() - 86400000), status: 'active', duration: '45m', history: [{ status: 'success', timestamp: new Date(Date.now() - 86400000) }] },
  { id: '2', name: 'ML Model Retraining', description: 'Retrain recommendation models weekly', type: 'job', schedule: '0 3 * * 0', nextRun: new Date(Date.now() + 432000000), lastRun: new Date(Date.now() - 604800000), status: 'active', duration: '2h 30m', history: [{ status: 'success', timestamp: new Date(Date.now() - 604800000) }] },
  { id: '3', name: 'Performance Report', description: 'Generate weekly performance summary', type: 'report', schedule: '0 8 * * 1', nextRun: new Date(Date.now() + 259200000), lastRun: new Date(Date.now() - 604800000), status: 'active', duration: '5m', history: [{ status: 'success', timestamp: new Date(Date.now() - 604800000) }] },
  { id: '4', name: 'Cache Cleanup', description: 'Clear expired cache entries', type: 'cleanup', schedule: '*/30 * * * *', nextRun: new Date(Date.now() + 1800000), status: 'active', duration: '2m', history: [{ status: 'success', timestamp: new Date(Date.now() - 1800000) }] },
  { id: '5', name: 'Data Sync', description: 'Sync data with external services', type: 'workflow', schedule: '*/15 * * * *', nextRun: new Date(Date.now() + 900000), lastRun: new Date(Date.now() - 900000), status: 'failed', duration: '3m', history: [{ status: 'failed', timestamp: new Date(Date.now() - 900000) }] },
];

const DAYS = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];

const styles = {
  container: 'min-h-screen bg-gradient-to-br from-cyan-50 via-white to-teal-50 dark:from-gray-900 dark:via-gray-800 dark:to-gray-900 p-6',
  header: 'flex items-center justify-between mb-8',
  title: 'text-4xl font-bold bg-gradient-to-r from-cyan-600 to-teal-600 bg-clip-text text-transparent',
  subtitle: 'text-gray-600 dark:text-gray-300 mt-2',
  mainGrid: 'grid grid-cols-1 lg:grid-cols-4 gap-6',
  content: 'lg:col-span-3',
  card: 'bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm shadow-xl rounded-2xl border-0',
  taskCard: 'bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm shadow-xl rounded-2xl border-0 hover:shadow-2xl transition-all cursor-pointer',
  button: 'bg-gradient-to-r from-cyan-600 to-teal-600 hover:from-cyan-700 hover:to-teal-700 text-white shadow-lg',
};

export default function SmartScheduler() {
  const { toast } = useToast();
  const [tasks, setTasks] = useState<ScheduledTask[]>(SAMPLE_TASKS);
  const [selectedTask, setSelectedTask] = useState<ScheduledTask | null>(null);
  const [selectedDate, setSelectedDate] = useState(new Date());

  const handleToggleTask = useCallback((taskId: string) => {
    setTasks(prev => prev.map(t => t.id === taskId ? { ...t, status: t.status === 'active' ? 'paused' : 'active' } : t));
    toast({ title: 'Task Updated', description: 'Task status has been changed' });
  }, [toast]);

  const handleRunNow = useCallback((taskId: string) => {
    setTasks(prev => prev.map(t => t.id === taskId ? { ...t, status: 'running' } : t));
    setTimeout(() => {
      setTasks(prev => prev.map(t => t.id === taskId ? { ...t, status: 'active', lastRun: new Date() } : t));
      toast({ title: 'Task Completed', description: 'Task ran successfully' });
    }, 2000);
  }, [toast]);

  const handleDeleteTask = useCallback((taskId: string) => {
    setTasks(prev => prev.filter(t => t.id !== taskId));
    if (selectedTask?.id === taskId) setSelectedTask(null);
    toast({ title: 'Task Deleted' });
  }, [selectedTask, toast]);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400';
      case 'paused': return 'bg-gray-100 text-gray-700 dark:bg-gray-900/30 dark:text-gray-400';
      case 'running': return 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400';
      case 'failed': return 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400';
      default: return 'bg-gray-100 text-gray-700';
    }
  };

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'backup': return '💾'; case 'job': return '⚙️'; case 'report': return '📊';
      case 'cleanup': return '🧹'; case 'workflow': return '🔄'; default: return '📋';
    }
  };

  const formatNextRun = (date: Date) => {
    const now = new Date();
    const diff = date.getTime() - now.getTime();
    if (diff < 3600000) return `In ${Math.round(diff / 60000)} minutes`;
    if (diff < 86400000) return `In ${Math.round(diff / 3600000)} hours`;
    return date.toLocaleDateString();
  };

  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <div>
          <h1 className={styles.title}><Calendar className="inline mr-2 h-8 w-8" />Smart Scheduler</h1>
          <p className={styles.subtitle}>Automate your tasks with intelligent scheduling and optimization</p>
        </div>
        <div className="flex items-center gap-3">
          <Button variant="outline"><RefreshCw className="w-4 h-4 mr-2" />Sync Now</Button>
          <Button className={styles.button}><Plus className="w-4 h-4 mr-2" />New Schedule</Button>
        </div>
      </div>

      <div className={styles.mainGrid}>
        <div className={styles.content}>
          <Tabs defaultValue="tasks" className="space-y-6">
            <TabsList className="bg-white/60 dark:bg-gray-800/60 backdrop-blur-sm p-1 rounded-xl">
              <TabsTrigger value="tasks" className="px-4 py-2 rounded-lg data-[state=active]:bg-cyan-100 data-[state=active]:text-cyan-700">
                <CalendarDays className="w-4 h-4 mr-2" />Tasks
              </TabsTrigger>
              <TabsTrigger value="calendar" className="px-4 py-2 rounded-lg data-[state=active]:bg-cyan-100 data-[state=active]:text-cyan-700">
                <Calendar className="w-4 h-4 mr-2" />Calendar
              </TabsTrigger>
            </TabsList>

            <TabsContent value="tasks">
              <div className="grid gap-4">
                {tasks.map((task) => (
                  <Card key={task.id} className={cn(styles.taskCard, selectedTask?.id === task.id && 'ring-2 ring-cyan-500')} onClick={() => setSelectedTask(task)}>
                    <CardContent className="p-4">
                      <div className="flex items-start justify-between">
                        <div className="flex items-start gap-4">
                          <span className="text-3xl">{getTypeIcon(task.type)}</span>
                          <div>
                            <div className="flex items-center gap-2">
                              <h3 className="font-semibold">{task.name}</h3>
                              <Badge className={cn('text-xs', getStatusColor(task.status))}>{task.status}</Badge>
                            </div>
                            <p className="text-sm text-gray-500 mt-1">{task.description}</p>
                            <div className="flex items-center gap-4 mt-2 text-sm text-gray-500">
                              <span className="flex items-center gap-1"><Clock className="w-3 h-3" />{task.schedule}</span>
                              <span className="flex items-center gap-1 text-cyan-600"><Timer className="w-3 h-3" />{formatNextRun(task.nextRun)}</span>
                            </div>
                          </div>
                        </div>
                        <div className="flex items-center gap-2">
                          {task.status === 'active' ? (
                            <Button variant="outline" size="sm" onClick={(e) => { e.stopPropagation(); handleToggleTask(task.id); }}><Pause className="w-3 h-3" /></Button>
                          ) : (
                            <Button variant="outline" size="sm" onClick={(e) => { e.stopPropagation(); handleToggleTask(task.id); }}><Play className="w-3 h-3" /></Button>
                          )}
                          <Button variant="outline" size="sm" onClick={(e) => { e.stopPropagation(); handleRunNow(task.id); }}><RefreshCw className="w-3 h-3" /></Button>
                          <Button variant="ghost" size="sm" onClick={(e) => { e.stopPropagation(); handleDeleteTask(task.id); }}><Trash2 className="w-3 h-3 text-red-500" /></Button>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            </TabsContent>

            <TabsContent value="calendar">
              <Card className={styles.card}>
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <CardTitle>Schedule Calendar</CardTitle>
                    <div className="flex items-center gap-2">
                      <Button variant="ghost" size="sm" onClick={() => setSelectedDate(d => new Date(d.setDate(d.getDate() - 7)))}><ChevronLeft className="w-4 h-4" /></Button>
                      <span className="font-medium">{selectedDate.toLocaleDateString('en-US', { month: 'long', year: 'numeric' })}</span>
                      <Button variant="ghost" size="sm" onClick={() => setSelectedDate(d => new Date(d.setDate(d.getDate() + 7)))}><ChevronRight className="w-4 h-4" /></Button>
                    </div>
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-7 gap-2 mb-4">
                    {DAYS.map(day => <div key={day} className="text-center text-sm font-medium text-gray-500">{day}</div>)}
                  </div>
                  <div className="grid grid-cols-7 gap-2">
                    {Array.from({ length: 31 }, (_, i) => (
                      <div key={i} className={cn('p-2 text-center rounded-lg text-sm', i + 1 === selectedDate.getDate() ? 'bg-cyan-100 text-cyan-700 font-medium' : 'hover:bg-gray-100 dark:hover:bg-gray-700 cursor-pointer')}>
                        {i + 1}
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </div>

        <div className="space-y-6">
          {selectedTask ? (
            <Card className={styles.card}>
              <CardHeader><CardTitle className="flex items-center gap-2"><Settings className="w-5 h-5 text-cyan-500" />Task Details</CardTitle></CardHeader>
              <CardContent className="space-y-4">
                <div><label className="text-sm font-medium text-gray-500">Schedule</label><p className="font-mono mt-1">{selectedTask.schedule}</p></div>
                <div><label className="text-sm font-medium text-gray-500">Last Run</label><p>{selectedTask.lastRun ? selectedTask.lastRun.toLocaleString() : 'Never'}</p></div>
                <div><label className="text-sm font-medium text-gray-500">Duration</label><p>{selectedTask.duration || 'N/A'}</p></div>
                <Separator />
                <div>
                  <label className="text-sm font-medium text-gray-500 mb-2 block">Recent History</label>
                  <ScrollArea className="h-[150px]">
                    <div className="space-y-2">
                      {selectedTask.history.map((h, i) => (
                        <div key={i} className="flex items-center gap-2 text-sm">
                          {h.status === 'success' ? <CheckCircle className="w-4 h-4 text-green-500" /> : <AlertCircle className="w-4 h-4 text-red-500" />}
                          <span>{new Date(h.timestamp).toLocaleString()}</span>
                        </div>
                      ))}
                    </div>
                  </ScrollArea>
                </div>
              </CardContent>
            </Card>
          ) : (
            <Card className={styles.card}><CardContent className="p-8 text-center"><CalendarDays className="w-16 h-16 mx-auto mb-4 text-gray-300" /><p className="text-gray-500">Select a task to view details</p></CardContent></Card>
          )}

          <Card className={styles.card}>
            <CardHeader><CardTitle className="flex items-center gap-2"><Zap className="w-5 h-5 text-amber-500" />Quick Stats</CardTitle></CardHeader>
            <CardContent className="space-y-4">
              <div className="flex justify-between items-center"><span className="text-sm text-gray-500">Total Tasks</span><span className="font-semibold">{tasks.length}</span></div>
              <div className="flex justify-between items-center"><span className="text-sm text-gray-500">Active</span><span className="font-semibold text-green-500">{tasks.filter(t => t.status === 'active').length}</span></div>
              <div className="flex justify-between items-center"><span className="text-sm text-gray-500">Failed</span><span className="font-semibold text-red-500">{tasks.filter(t => t.status === 'failed').length}</span></div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
