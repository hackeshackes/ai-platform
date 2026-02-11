/**
 * No-Code Builder - AI Democratization Page
 * 零代码工作流 - AI民主化页面
 */

import React, { useState, useCallback, useEffect, useRef } from 'react';
import {
  Plus,
  Play,
  Pause,
  Save,
  Trash2,
  Copy,
  Settings,
  Move,
  Zap,
  GitBranch,
  Layers,
  ArrowRight,
  Download,
  Upload,
  Eye,
  EyeOff,
  Check,
  X,
  GripVertical,
  MoreHorizontal,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Switch } from '@/components/ui/switch';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Separator } from '@/components/ui/separator';
import { useToast } from '@/hooks/use-toast';
import { cn } from '@/lib/utils';

// Types
interface WorkflowNode {
  id: string;
  type: 'trigger' | 'action' | 'condition' | 'loop' | 'transform';
  name: string;
  config: Record<string, any>;
  position: { x: number; y: number };
  enabled: boolean;
}

interface WorkflowEdge {
  id: string;
  source: string;
  target: string;
  condition?: string;
}

interface Workflow {
  id: string;
  name: string;
  description: string;
  nodes: WorkflowNode[];
  edges: WorkflowEdge[];
  status: 'draft' | 'active' | 'paused';
  createdAt: Date;
  updatedAt: Date;
}

// API Functions
const saveWorkflow = async (workflow: Workflow): Promise<Workflow> => {
  const response = await apiClient.post<Workflow>('/api/v12/no-code-builder/workflows', workflow);
  return response.data;
};

const getWorkflows = async (): Promise<Workflow[]> => {
  const response = await apiClient.get<Workflow[]>('/api/v12/no-code-builder/workflows');
  return response.data;
};

const executeWorkflow = async (workflowId: string): Promise<void> => {
  await apiClient.post(`/api/v12/no-code-builder/workflows/${workflowId}/execute`);
};

// Node Templates
const NODE_TEMPLATES = [
  { type: 'trigger', name: 'HTTP Request', icon: '🌐' },
  { type: 'trigger', name: 'Timer', icon: '⏰' },
  { type: 'trigger', name: 'Webhook', icon: '🪝' },
  { type: 'trigger', name: 'Event Listener', icon: '👂' },
  { type: 'action', name: 'Send Email', icon: '📧' },
  { type: 'action', name: 'API Call', icon: '📡' },
  { type: 'action', name: 'Database Query', icon: '🗄️' },
  { type: 'action', name: 'Send Message', icon: '💬' },
  { type: 'condition', name: 'If/Else', icon: '❓' },
  { type: 'condition', name: 'Switch', icon: '🔀' },
  { type: 'loop', name: 'For Each', icon: '🔁' },
  { type: 'loop', name: 'While', icon: '⏳' },
  { type: 'transform', name: 'Data Transform', icon: '🔄' },
  { type: 'transform', name: 'Filter', icon: '🔍' },
];

// Styles
const styles = {
  container: 'min-h-screen bg-gradient-to-br from-emerald-50 via-white to-teal-50 dark:from-gray-900 dark:via-gray-800 dark:to-gray-900 p-6',
  header: 'flex items-center justify-between mb-8',
  title: 'text-4xl font-bold bg-gradient-to-r from-emerald-600 to-teal-600 bg-clip-text text-transparent',
  subtitle: 'text-gray-600 dark:text-gray-300 mt-2',
  mainGrid: 'grid grid-cols-1 lg:grid-cols-4 gap-6',
  sidebar: 'bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm rounded-2xl shadow-xl p-4',
  canvas: 'lg:col-span-3 bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm rounded-2xl shadow-xl min-h-[600px] relative overflow-hidden',
  nodeCard: 'absolute p-4 rounded-xl shadow-lg border-2 cursor-move transition-all hover:shadow-xl',
  nodeTrigger: 'border-emerald-500 bg-emerald-50 dark:bg-emerald-900/20',
  nodeAction: 'border-blue-500 bg-blue-50 dark:bg-blue-900/20',
  nodeCondition: 'border-amber-500 bg-amber-50 dark:bg-amber-900/20',
  nodeLoop: 'border-purple-500 bg-purple-50 dark:bg-purple-900/20',
  nodeTransform: 'border-pink-500 bg-pink-50 dark:bg-pink-900/20',
  toolbar: 'flex items-center gap-2 p-2 bg-white dark:bg-gray-800 rounded-xl shadow-lg',
  button: 'bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-700 hover:to-teal-700 text-white shadow-lg',
  iconButton: 'p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors',
};

export default function NoCodeBuilder() {
  const { toast } = useToast();
  const canvasRef = useRef<HTMLDivElement>(null);
  
  const [workflows, setWorkflows] = useState<Workflow[]>([]);
  const [currentWorkflow, setCurrentWorkflow] = useState<Workflow | null>(null);
  const [selectedNode, setSelectedNode] = useState<WorkflowNode | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [showGrid, setShowGrid] = useState(true);

  useEffect(() => {
    loadWorkflows();
  }, []);

  const loadWorkflows = async () => {
    try {
      const data = await getWorkflows();
      setWorkflows(data);
      if (data.length > 0) {
        setCurrentWorkflow(data[0]);
      } else {
        createNewWorkflow();
      }
    } catch (error) {
      console.error('Failed to load workflows:', error);
      createNewWorkflow();
    }
  };

  const createNewWorkflow = () => {
    const newWorkflow: Workflow = {
      id: `wf-${Date.now()}`,
      name: 'New Workflow',
      description: '',
      nodes: [],
      edges: [],
      status: 'draft',
      createdAt: new Date(),
      updatedAt: new Date(),
    };
    setCurrentWorkflow(newWorkflow);
    setWorkflows(prev => [...prev, newWorkflow]);
  };

  const addNode = useCallback((template: typeof NODE_TEMPLATES[0]) => {
    if (!currentWorkflow) return;

    const newNode: WorkflowNode = {
      id: `node-${Date.now()}`,
      type: template.type as any,
      name: template.name,
      config: {},
      position: { x: 100, y: 100 + (currentWorkflow.nodes.length * 80) },
      enabled: true,
    };

    setCurrentWorkflow({
      ...currentWorkflow,
      nodes: [...currentWorkflow.nodes, newNode],
      updatedAt: new Date(),
    });

    toast({
      title: 'Node Added',
      description: `${template.name} added to workflow`,
    });
  }, [currentWorkflow, toast]);

  const deleteNode = useCallback((nodeId: string) => {
    if (!currentWorkflow) return;

    setCurrentWorkflow({
      ...currentWorkflow,
      nodes: currentWorkflow.nodes.filter(n => n.id !== nodeId),
      edges: currentWorkflow.edges.filter(e => e.source !== nodeId && e.target !== nodeId),
      updatedAt: new Date(),
    });
    setSelectedNode(null);
  }, [currentWorkflow]);

  const updateNode = useCallback((nodeId: string, updates: Partial<WorkflowNode>) => {
    if (!currentWorkflow) return;

    setCurrentWorkflow({
      ...currentWorkflow,
      nodes: currentWorkflow.nodes.map(n =>
        n.id === nodeId ? { ...n, ...updates } : n
      ),
      updatedAt: new Date(),
    });
  }, [currentWorkflow]);

  const handleRun = useCallback(async () => {
    if (!currentWorkflow) return;

    setIsRunning(true);
    try {
      await executeWorkflow(currentWorkflow.id);
      toast({
        title: 'Workflow Executed',
        description: 'Your workflow ran successfully',
      });
    } catch (error) {
      toast({
        title: 'Execution Failed',
        description: 'There was an error running your workflow',
        variant: 'destructive',
      });
    } finally {
      setIsRunning(false);
    }
  }, [currentWorkflow, toast]);

  const handleSave = useCallback(async () => {
    if (!currentWorkflow) return;

    try {
      await saveWorkflow(currentWorkflow);
      toast({
        title: 'Saved',
        description: 'Workflow saved successfully',
      });
    } catch (error) {
      toast({
        title: 'Save Failed',
        description: 'There was an error saving your workflow',
        variant: 'destructive',
      });
    }
  }, [currentWorkflow, toast]);

  const getNodeClassName = (type: string) => {
    switch (type) {
      case 'trigger': return styles.nodeTrigger;
      case 'action': return styles.nodeAction;
      case 'condition': return styles.nodeCondition;
      case 'loop': return styles.nodeLoop;
      case 'transform': return styles.nodeTransform;
      default: return '';
    }
  };

  const getNodeIcon = (type: string) => {
    const template = NODE_TEMPLATES.find(t => t.type === type);
    return template?.icon || '📦';
  };

  return (
    <div className={styles.container}>
      {/* Header */}
      <div className={styles.header}>
        <div>
          <h1 className={styles.title}>
            <Layers className="inline mr-2 h-8 w-8" />
            No-Code Builder
          </h1>
          <p className={styles.subtitle}>
            Build powerful workflows without writing a single line of code
          </p>
        </div>
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <Switch checked={showGrid} onCheckedChange={setShowGrid} />
            <span className="text-sm text-gray-600 dark:text-gray-300">Show Grid</span>
          </div>
          <Button variant="outline">
            <Upload className="w-4 h-4 mr-2" />
            Import
          </Button>
          <Button variant="outline">
            <Download className="w-4 h-4 mr-2" />
            Export
          </Button>
          <Button className={styles.button} onClick={handleSave}>
            <Save className="w-4 h-4 mr-2" />
            Save
          </Button>
          <Button 
            className={styles.button} 
            onClick={handleRun}
            disabled={isRunning || !currentWorkflow?.nodes.length}
          >
            {isRunning ? (
              <>
                <Pause className="w-4 h-4 mr-2 animate-pulse" />
                Running...
              </>
            ) : (
              <>
                <Play className="w-4 h-4 mr-2" />
                Run
              </>
            )}
          </Button>
        </div>
      </div>

      {/* Main Content */}
      <div className={styles.mainGrid}>
        {/* Node Templates Sidebar */}
        <div className={styles.sidebar}>
          <h3 className="font-semibold mb-4 flex items-center gap-2">
            <Zap className="w-4 h-4 text-emerald-500" />
            Add Nodes
          </h3>
          <ScrollArea className="h-[calc(100vh-250px)]">
            <div className="space-y-2">
              {NODE_TEMPLATES.map((template) => (
                <button
                  key={`${template.type}-${template.name}`}
                  className="w-full flex items-center gap-3 p-3 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors text-left"
                  onClick={() => addNode(template)}
                >
                  <span className="text-2xl">{template.icon}</span>
                  <div>
                    <p className="font-medium text-sm">{template.name}</p>
                    <Badge variant="secondary" className="text-xs">
                      {template.type}
                    </Badge>
                  </div>
                  <Plus className="w-4 h-4 ml-auto text-gray-400" />
                </button>
              ))}
            </div>
          </ScrollArea>
        </div>

        {/* Canvas */}
        <div className={styles.canvas} ref={canvasRef}>
          <div className={cn(
            'absolute inset-0 p-8',
            showGrid && 'bg-[radial-gradient(circle,rgba(0,0,0,0.1)_1px,transparent_1px)] bg-[size:20px_20px]'
          )}>
            {currentWorkflow?.nodes.map((node) => (
              <div
                key={node.id}
                className={cn(
                  styles.nodeCard,
                  getNodeClassName(node.type),
                  !node.enabled && 'opacity-50',
                  selectedNode?.id === node.id && 'ring-2 ring-offset-2 ring-emerald-500'
                )}
                style={{
                  left: node.position.x,
                  top: node.position.y,
                  minWidth: '180px',
                }}
                onClick={() => setSelectedNode(node)}
              >
                <div className="flex items-center gap-2">
                  <GripVertical className="w-4 h-4 text-gray-400 cursor-grab" />
                  <span className="text-xl">{getNodeIcon(node.type)}</span>
                  <div className="flex-1">
                    <p className="font-medium text-sm">{node.name}</p>
                    <p className="text-xs text-gray-500">{node.type}</p>
                  </div>
                  <button
                    className="p-1 hover:bg-gray-200 dark:hover:bg-gray-600 rounded"
                    onClick={(e) => {
                      e.stopPropagation();
                      deleteNode(node.id);
                    }}
                  >
                    <X className="w-3 h-3" />
                  </button>
                </div>
                {node.type === 'condition' && (
                  <div className="mt-2 pt-2 border-t border-gray-200 dark:border-gray-600">
                    <p className="text-xs text-gray-500">No conditions set</p>
                  </div>
                )}
              </div>
            ))}

            {currentWorkflow?.nodes.length === 0 && (
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="text-center text-gray-400">
                  <Layers className="w-16 h-16 mx-auto mb-4 opacity-50" />
                  <p className="text-lg font-medium">No nodes yet</p>
                  <p className="text-sm">Add nodes from the sidebar to build your workflow</p>
                </div>
              </div>
            )}
          </div>

          {/* Floating Toolbar */}
          <div className="absolute bottom-4 left-1/2 -translate-x-1/2">
            <div className={styles.toolbar}>
              <Button variant="ghost" size="sm">
                <Move className="w-4 h-4" />
              </Button>
              <Button variant="ghost" size="sm">
                <GitBranch className="w-4 h-4" />
              </Button>
              <Separator orientation="vertical" className="h-6" />
              <Button variant="ghost" size="sm">
                <Eye className="w-4 h-4" />
              </Button>
              <Button variant="ghost" size="sm">
                <Settings className="w-4 h-4" />
              </Button>
            </div>
          </div>
        </div>

        {/* Properties Panel */}
        <div className="bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm rounded-2xl shadow-xl p-4">
          <h3 className="font-semibold mb-4 flex items-center gap-2">
            <Settings className="w-4 h-4 text-gray-500" />
            {selectedNode ? 'Node Properties' : 'Workflow Settings'}
          </h3>
          
          {selectedNode ? (
            <div className="space-y-4">
              <div>
                <label className="text-sm font-medium">Node Name</label>
                <Input
                  value={selectedNode.name}
                  onChange={(e) => updateNode(selectedNode.id, { name: e.target.value })}
                  className="mt-1"
                />
              </div>
              <div>
                <label className="text-sm font-medium">Enabled</label>
                <div className="flex items-center gap-2 mt-1">
                  <Switch
                    checked={selectedNode.enabled}
                    onCheckedChange={(v) => updateNode(selectedNode.id, { enabled: v })}
                  />
                  <span className="text-sm text-gray-500">
                    {selectedNode.enabled ? 'Enabled' : 'Disabled'}
                  </span>
                </div>
              </div>
              
              <Separator />
              
              <div>
                <label className="text-sm font-medium">Configuration</label>
                <p className="text-xs text-gray-500 mt-1">
                  Configure {selectedNode.name} settings
                </p>
                <Button variant="outline" className="w-full mt-2">
                  Open Config Editor
                </Button>
              </div>

              <Separator />

              <div className="flex gap-2">
                <Button variant="outline" size="sm" className="flex-1">
                  <Copy className="w-3 h-3 mr-1" />
                  Duplicate
                </Button>
                <Button 
                  variant="destructive" 
                  size="sm" 
                  className="flex-1"
                  onClick={() => deleteNode(selectedNode.id)}
                >
                  <Trash2 className="w-3 h-3 mr-1" />
                  Delete
                </Button>
              </div>
            </div>
          ) : (
            <div className="space-y-4">
              <div>
                <label className="text-sm font-medium">Workflow Name</label>
                <Input
                  value={currentWorkflow?.name || ''}
                  onChange={(e) => currentWorkflow && setCurrentWorkflow({
                    ...currentWorkflow,
                    name: e.target.value,
                  })}
                  className="mt-1"
                />
              </div>
              <div>
                <label className="text-sm font-medium">Description</label>
                <Input
                  value={currentWorkflow?.description || ''}
                  onChange={(e) => currentWorkflow && setCurrentWorkflow({
                    ...currentWorkflow,
                    description: e.target.value,
                  })}
                  className="mt-1"
                />
              </div>
              <div>
                <label className="text-sm font-medium">Status</label>
                <Select
                  value={currentWorkflow?.status}
                  onValueChange={(v) => currentWorkflow && setCurrentWorkflow({
                    ...currentWorkflow,
                    status: v as any,
                  })}
                >
                  <SelectTrigger className="mt-1">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="draft">Draft</SelectItem>
                    <SelectItem value="active">Active</SelectItem>
                    <SelectItem value="paused">Paused</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
