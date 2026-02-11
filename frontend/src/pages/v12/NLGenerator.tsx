/**
 * Natural Language Generator - AI Democratization Page
 * 自然语言生成器 - AI民主化页面
 */

import React, { useState, useCallback, useEffect } from 'react';
import { 
  Sparkles, 
  Copy, 
  RefreshCw, 
  Download, 
  History,
  Settings,
  Send,
  Wand2,
  Zap
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Textarea } from '@/components/ui/textarea';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Progress } from '@/components/ui/progress';
import { useToast } from '@/hooks/use-toast';
import { apiClient } from '@/lib/api';
import { useTranslation } from 'react-i18next';

// Types
interface GenerationRequest {
  prompt: string;
  type: 'code' | 'text' | 'sql' | 'config' | 'documentation';
  language: string;
  tone: 'formal' | 'casual' | 'technical';
  maxLength: number;
}

interface GenerationResponse {
  id: string;
  result: string;
  tokens: number;
  model: string;
  timestamp: Date;
}

interface HistoryItem {
  id: string;
  prompt: string;
  result: string;
  timestamp: Date;
  type: string;
}

// API Functions
const generateContent = async (request: GenerationRequest): Promise<GenerationResponse> => {
  const response = await apiClient.post<GenerationResponse>('/api/v12/nl-generator/generate', request);
  return response.data;
};

const getHistory = async (): Promise<HistoryItem[]> => {
  const response = await apiClient.get<HistoryItem[]>('/api/v12/nl-generator/history');
  return response.data;
};

// Styles
const styles = {
  container: 'min-h-screen bg-gradient-to-br from-purple-50 via-white to-blue-50 dark:from-gray-900 dark:via-gray-800 dark:to-gray-900 p-6',
  header: 'flex items-center justify-between mb-8',
  title: 'text-4xl font-bold bg-gradient-to-r from-purple-600 to-blue-600 bg-clip-text text-transparent',
  subtitle: 'text-gray-600 dark:text-gray-300 mt-2',
  mainGrid: 'grid grid-cols-1 lg:grid-cols-3 gap-6',
  inputSection: 'lg:col-span-2',
  outputSection: 'lg:col-span-1',
  card: 'bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm shadow-xl rounded-2xl border-0',
  inputArea: 'min-h-[200px] resize-none rounded-xl border-2 border-purple-200 focus:border-purple-500 dark:border-gray-700 dark:focus:border-purple-500',
  button: 'bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700 text-white shadow-lg',
  iconButton: 'p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors',
  resultArea: 'min-h-[400px] p-4 bg-gray-50 dark:bg-gray-900 rounded-xl font-mono text-sm overflow-auto',
  historyItem: 'p-3 border-b border-gray-100 dark:border-gray-700 last:border-0 hover:bg-gray-50 dark:hover:bg-gray-700/50 cursor-pointer transition-colors',
  badge: 'px-3 py-1 rounded-full text-xs font-medium',
  loadingBar: 'h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden',
  loadingProgress: 'h-full bg-gradient-to-r from-purple-500 to-blue-500 animate-pulse',
};

export default function NLGenerator() {
  const { t } = useTranslation();
  const { toast } = useToast();
  
  const [prompt, setPrompt] = useState('');
  const [result, setResult] = useState('');
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [progress, setProgress] = useState(0);
  const [settings, setSettings] = useState<Partial<GenerationRequest>>({
    type: 'text',
    language: 'zh-CN',
    tone: 'technical',
    maxLength: 2000,
  });

  // Load history on mount
  useEffect(() => {
    loadHistory();
  }, []);

  const loadHistory = async () => {
    try {
      const data = await getHistory();
      setHistory(data);
    } catch (error) {
      console.error('Failed to load history:', error);
    }
  };

  const handleGenerate = useCallback(async () => {
    if (!prompt.trim()) {
      toast({
        title: 'Warning',
        description: 'Please enter a prompt',
        variant: 'destructive',
      });
      return;
    }

    setIsGenerating(true);
    setProgress(0);

    try {
      // Simulate progress
      const progressInterval = setInterval(() => {
        setProgress(prev => Math.min(prev + 10, 90));
      }, 200);

      const response = await generateContent({
        prompt,
        ...settings,
      } as GenerationRequest);

      clearInterval(progressInterval);
      setProgress(100);
      setResult(response.result);

      // Add to history
      const newItem: HistoryItem = {
        id: response.id,
        prompt,
        result: response.result,
        timestamp: new Date(),
        type: settings.type || 'text',
      };
      setHistory(prev => [newItem, ...prev]);

      toast({
        title: 'Success',
        description: 'Content generated successfully',
      });
    } catch (error) {
      toast({
        title: 'Error',
        description: 'Failed to generate content',
        variant: 'destructive',
      });
    } finally {
      setIsGenerating(false);
      setTimeout(() => setProgress(0), 500);
    }
  }, [prompt, settings, toast]);

  const handleCopy = useCallback(() => {
    navigator.clipboard.writeText(result);
    toast({
      title: 'Copied',
      description: 'Result copied to clipboard',
    });
  }, [result, toast]);

  const handleClear = useCallback(() => {
    setPrompt('');
    setResult('');
  }, []);

  const handleDownload = useCallback(() => {
    const blob = new Blob([result], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `generated-${Date.now()}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  }, [result]);

  return (
    <div className={styles.container}>
      {/* Header */}
      <div className={styles.header}>
        <div>
          <h1 className={styles.title}>
            <Sparkles className="inline mr-2 h-8 w-8" />
            {t('v12.nlGenerator.title', 'Natural Language Generator')}
          </h1>
          <p className={styles.subtitle}>
            {t('v12.nlGenerator.subtitle', 'Transform your ideas into code, documents, and more with AI')}
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="outline" onClick={loadHistory}>
            <History className="w-4 h-4 mr-2" />
            History
          </Button>
          <Button variant="outline">
            <Settings className="w-4 h-4" />
          </Button>
        </div>
      </div>

      {/* Main Content */}
      <div className={styles.mainGrid}>
        {/* Input Section */}
        <div className={styles.inputSection}>
          <Card className={styles.card}>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Wand2 className="w-5 h-5 text-purple-500" />
                {t('v12.nlGenerator.inputTitle', 'Enter Your Prompt')}
              </CardTitle>
              <CardDescription>
                Describe what you want to generate in natural language
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Tabs value={settings.type} onValueChange={(v) => setSettings(s => ({ ...s, type: v as any }))}>
                <TabsList className="mb-4">
                  <TabsTrigger value="code">Code</TabsTrigger>
                  <TabsTrigger value="text">Text</TabsTrigger>
                  <TabsTrigger value="sql">SQL</TabsTrigger>
                  <TabsTrigger value="config">Config</TabsTrigger>
                  <TabsTrigger value="documentation">Docs</TabsTrigger>
                </TabsList>

                <Textarea
                  className={styles.inputArea}
                  placeholder="Describe what you want to generate..."
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                />

                <div className="flex flex-wrap gap-4 mt-4">
                  <Select
                    value={settings.language}
                    onValueChange={(v) => setSettings(s => ({ ...s, language: v }))}
                  >
                    <SelectTrigger className="w-[140px]">
                      <SelectValue placeholder="Language" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="zh-CN">Chinese</SelectItem>
                      <SelectItem value="en-US">English</SelectItem>
                      <SelectItem value="ja-JP">Japanese</SelectItem>
                      <SelectItem value="ko-KR">Korean</SelectItem>
                    </SelectContent>
                  </Select>

                  <Select
                    value={settings.tone}
                    onValueChange={(v) => setSettings(s => ({ ...s, tone: v as any }))}
                  >
                    <SelectTrigger className="w-[140px]">
                      <SelectValue placeholder="Tone" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="formal">Formal</SelectItem>
                      <SelectItem value="casual">Casual</SelectItem>
                      <SelectItem value="technical">Technical</SelectItem>
                    </SelectContent>
                  </Select>

                  <Select
                    value={settings.maxLength?.toString()}
                    onValueChange={(v) => setSettings(s => ({ ...s, maxLength: parseInt(v) }))}
                  >
                    <SelectTrigger className="w-[140px]">
                      <SelectValue placeholder="Max Length" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="500">500 tokens</SelectItem>
                      <SelectItem value="1000">1000 tokens</SelectItem>
                      <SelectItem value="2000">2000 tokens</SelectItem>
                      <SelectItem value="4000">4000 tokens</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="flex gap-3 mt-6">
                  <Button
                    className={styles.button}
                    onClick={handleGenerate}
                    disabled={isGenerating || !prompt.trim()}
                  >
                    {isGenerating ? (
                      <>
                        <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                        Generating...
                      </>
                    ) : (
                      <>
                        <Send className="w-4 h-4 mr-2" />
                        Generate
                      </>
                    )}
                  </Button>
                  <Button variant="outline" onClick={handleClear}>
                    Clear
                  </Button>
                </div>

                {isGenerating && (
                  <div className="mt-4">
                    <Progress value={progress} className={styles.loadingBar} />
                    <p className="text-sm text-gray-500 mt-1">Generating your content...</p>
                  </div>
                )}
              </Tabs>
            </CardContent>
          </Card>

          {/* Result Section */}
          {result && (
            <Card className={`${styles.card} mt-6`}>
              <CardHeader>
                <div className="flex items-center justify-between">
                  <CardTitle className="flex items-center gap-2">
                    <Zap className="w-5 h-5 text-yellow-500" />
                    Generated Result
                  </CardTitle>
                  <div className="flex gap-2">
                    <Button variant="ghost" size="sm" onClick={handleCopy}>
                      <Copy className="w-4 h-4 mr-1" />
                      Copy
                    </Button>
                    <Button variant="ghost" size="sm" onClick={handleDownload}>
                      <Download className="w-4 h-4 mr-1" />
                      Download
                    </Button>
                  </div>
                </div>
              </CardHeader>
              <CardContent>
                <pre className={styles.resultArea}>{result}</pre>
              </CardContent>
            </Card>
          )}
        </div>

        {/* History Section */}
        <div className={styles.outputSection}>
          <Card className={styles.card}>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <History className="w-5 h-5 text-blue-500" />
                Generation History
              </CardTitle>
              <CardDescription>
                Your recent generations
              </CardDescription>
            </CardHeader>
            <CardContent>
              {history.length === 0 ? (
                <div className="text-center py-8 text-gray-500">
                  <Sparkles className="w-12 h-12 mx-auto mb-3 opacity-50" />
                  <p>No history yet</p>
                  <p className="text-sm">Start generating to see your history</p>
                </div>
              ) : (
                <div className="space-y-2 max-h-[600px] overflow-auto">
                  {history.map((item) => (
                    <div
                      key={item.id}
                      className={styles.historyItem}
                      onClick={() => setResult(item.result)}
                    >
                      <div className="flex items-center justify-between mb-1">
                        <Badge className={styles.badge}>
                          {item.type}
                        </Badge>
                        <span className="text-xs text-gray-500">
                          {new Date(item.timestamp).toLocaleTimeString()}
                        </span>
                      </div>
                      <p className="text-sm font-medium truncate">{item.prompt}</p>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
