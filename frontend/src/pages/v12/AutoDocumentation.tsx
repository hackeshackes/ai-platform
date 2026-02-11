/**
 * Auto Documentation - AI Democratization Page
 * 自动文档 - AI民主化页面
 */

import React, { useState, useCallback, useEffect } from 'react';
import {
  FileText, Download, Copy, RefreshCw, Settings, Upload,
  Eye, Edit, Trash2, Plus, Sparkles, BookOpen, Code,
  Globe, Clock, User, Tag, ChevronDown, ChevronRight,
  Search, Filter, MoreHorizontal,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Switch } from '@/components/ui/switch';
import { Textarea } from '@/components/ui/textarea';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Separator } from '@/components/ui/separator';
import { useToast } from '@/hooks/use-toast';
import { cn } from '@/lib/utils';

interface Document {
  id: string;
  title: string;
  description: string;
  type: 'api' | 'guide' | 'tutorial' | 'reference';
  content: string;
  language: string;
  status: 'draft' | 'published' | 'archived';
  author: string;
  createdAt: Date;
  updatedAt: Date;
  tags: string[];
  views: number;
  versions: number;
}

const SAMPLE_DOCUMENTS: Document[] = [
  {
    id: '1',
    title: 'API Reference',
    description: 'Complete API documentation with endpoints, parameters, and examples',
    type: 'api',
    content: '# API Reference\n\n## Authentication\n\nAll API requests require...',
    language: 'markdown',
    status: 'published',
    author: 'System',
    createdAt: new Date('2024-01-15'),
    updatedAt: new Date('2024-02-10'),
    tags: ['api', 'reference', 'authentication'],
    views: 15420,
    versions: 12,
  },
  {
    id: '2',
    title: 'Getting Started Guide',
    description: 'Quick start guide for new developers',
    type: 'guide',
    content: '# Getting Started\n\nWelcome to our platform...',
    language: 'markdown',
    status: 'published',
    author: 'Documentation Team',
    createdAt: new Date('2024-02-01'),
    updatedAt: new Date('2024-02-08'),
    tags: ['guide', 'beginner', 'tutorial'],
    views: 8930,
    versions: 5,
  },
  {
    id: '3',
    title: 'Advanced Usage Tutorial',
    description: 'Learn advanced features and best practices',
    type: 'tutorial',
    content: '# Advanced Tutorial\n\nThis guide covers...',
    language: 'markdown',
    status: 'draft',
    author: 'Senior Developer',
    createdAt: new Date('2024-02-05'),
    updatedAt: new Date('2024-02-12'),
    tags: ['advanced', 'tutorial', 'best-practices'],
    views: 2340,
    versions: 3,
  },
];

const styles = {
  container: 'min-h-screen bg-gradient-to-br from-teal-50 via-white to-cyan-50 dark:from-gray-900 dark:via-gray-800 dark:to-gray-900 p-6',
  header: 'flex items-center justify-between mb-8',
  title: 'text-4xl font-bold bg-gradient-to-r from-teal-600 to-cyan-600 bg-clip-text text-transparent',
  subtitle: 'text-gray-600 dark:text-gray-300 mt-2',
  mainGrid: 'grid grid-cols-1 lg:grid-cols-4 gap-6',
  sidebar: 'space-y-6',
  content: 'lg:col-span-3',
  card: 'bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm shadow-xl rounded-2xl border-0',
  docCard: 'bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm shadow-xl rounded-2xl border-0 hover:shadow-2xl transition-all cursor-pointer',
  previewArea: 'min-h-[500px] p-6 bg-white dark:bg-gray-900 rounded-xl prose dark:prose-invert max-w-none',
  button: 'bg-gradient-to-r from-teal-600 to-cyan-600 hover:from-teal-700 hover:to-cyan-700 text-white shadow-lg',
  iconButton: 'p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors',
  badge: 'px-3 py-1 rounded-full text-xs font-medium',
  tabTrigger: 'px-4 py-2 rounded-lg data-[state=active]:bg-teal-100 data-[state=active]:text-teal-700',
};

export default function AutoDocumentation() {
  const { toast } = useToast();
  const [documents, setDocuments] = useState<Document[]>(SAMPLE_DOCUMENTS);
  const [selectedDoc, setSelectedDoc] = useState<Document | null>(null);
  const [sourceCode, setSourceCode] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedType, setSelectedType] = useState('all');
  const [previewMode, setPreviewMode] = useState<'split' | 'preview' | 'edit'>('split');
  const [autoGenerate, setAutoGenerate] = useState(true);
  const [aiAssistance, setAiAssistance] = useState(true);

  useEffect(() => {
    if (documents.length > 0 && !selectedDoc) {
      setSelectedDoc(documents[0]);
    }
  }, []);

  const handleGenerate = useCallback(async () => {
    if (!sourceCode.trim()) {
      toast({
        title: 'Warning',
        description: 'Please enter source code to generate documentation',
        variant: 'destructive',
      });
      return;
    }

    setIsGenerating(true);
    try {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 2000));
      const newDoc: Document = {
        id: `doc-${Date.now()}`,
        title: 'Generated Documentation',
        description: 'Auto-generated from your source code',
        type: 'api',
        content: `# Generated Documentation\n\n${sourceCode.slice(0, 500)}...`,
        language: 'markdown',
        status: 'draft',
        author: 'AI Generator',
        createdAt: new Date(),
        updatedAt: new Date(),
        tags: ['auto-generated', 'documentation'],
        views: 0,
        versions: 1,
      };
      
      setDocuments(prev => [newDoc, ...prev]);
      setSelectedDoc(newDoc);
      
      toast({
        title: 'Documentation Generated',
        description: 'Your documentation has been created successfully',
      });
    } catch (error) {
      toast({
        title: 'Generation Failed',
        description: 'Could not generate documentation',
        variant: 'destructive',
      });
    } finally {
      setIsGenerating(false);
    }
  }, [sourceCode, aiAssistance, toast]);

  const handleCopy = useCallback(() => {
    if (selectedDoc) {
      navigator.clipboard.writeText(selectedDoc.content);
      toast({
        title: 'Copied',
        description: 'Documentation copied to clipboard',
      });
    }
  }, [selectedDoc, toast]);

  const handleDownload = useCallback(() => {
    if (selectedDoc) {
      const blob = new Blob([selectedDoc.content], { type: 'text/markdown' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${selectedDoc.title.replace(/\s+/g, '_')}.md`;
      a.click();
      URL.revokeObjectURL(url);
    }
  }, [selectedDoc]);

  const filteredDocuments = documents.filter(doc => {
    if (searchQuery) {
      const query = searchQuery.toLowerCase();
      return (
        doc.title.toLowerCase().includes(query) ||
        doc.description.toLowerCase().includes(query) ||
        doc.tags.some(tag => tag.toLowerCase().includes(query))
      );
    }
    if (selectedType !== 'all' && doc.type !== selectedType) return false;
    return true;
  });

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'published': return 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400';
      case 'draft': return 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400';
      case 'archived': return 'bg-gray-100 text-gray-700 dark:bg-gray-900/30 dark:text-gray-400';
      default: return 'bg-gray-100 text-gray-700';
    }
  };

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'api': return '🔌';
      case 'guide': return '📖';
      case 'tutorial': return '🎓';
      case 'reference': return '📋';
      default: return '📄';
    }
  };

  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <div>
          <h1 className={styles.title}>
            <BookOpen className="inline mr-2 h-8 w-8" />
            Auto Documentation
          </h1>
          <p className={styles.subtitle}>
            Generate beautiful, comprehensive documentation automatically with AI
          </p>
        </div>
        <div className="flex items-center gap-3">
          <Button variant="outline">
            <Upload className="w-4 h-4 mr-2" />
            Import Code
          </Button>
          <Button variant="outline">
            <Settings className="w-4 h-4 mr-2" />
            Settings
          </Button>
          <Button className={styles.button}>
            <Plus className="w-4 h-4 mr-2" />
            New Document
          </Button>
        </div>
      </div>

      <div className={styles.mainGrid}>
        <div className={styles.sidebar}>
          <Card className={styles.card}>
            <CardHeader>
              <CardTitle className="flex items-center justify-between">
                <span className="flex items-center gap-2">
                  <FileText className="w-5 h-5 text-teal-500" />
                  Documents
                </span>
                <Badge>{documents.length}</Badge>
              </CardTitle>
              <CardDescription>Your documentation projects</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
                <Input
                  className="pl-9 h-10"
                  placeholder="Search documents..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                />
              </div>

              <Select value={selectedType} onValueChange={setSelectedType}>
                <SelectTrigger>
                  <SelectValue placeholder="Type" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Types</SelectItem>
                  <SelectItem value="api">API</SelectItem>
                  <SelectItem value="guide">Guide</SelectItem>
                  <SelectItem value="tutorial">Tutorial</SelectItem>
                  <SelectItem value="reference">Reference</SelectItem>
                </SelectContent>
              </Select>

              <ScrollArea className="h-[400px]">
                <div className="space-y-2">
                  {filteredDocuments.map((doc) => (
                    <div
                      key={doc.id}
                      className={cn(
                        'p-3 rounded-lg cursor-pointer transition-colors',
                        selectedDoc?.id === doc.id
                          ? 'bg-teal-100 dark:bg-teal-900/30'
                          : 'hover:bg-gray-100 dark:hover:bg-gray-700'
                      )}
                      onClick={() => setSelectedDoc(doc)}
                    >
                      <div className="flex items-start justify-between mb-1">
                        <div className="flex items-center gap-2">
                          <span>{getTypeIcon(doc.type)}</span>
                          <span className="font-medium text-sm">{doc.title}</span>
                        </div>
                        <Badge className={cn('text-xs', getStatusColor(doc.status))}>
                          {doc.status}
                        </Badge>
                      </div>
                      <p className="text-xs text-gray-500 line-clamp-2">{doc.description}</p>
                      <div className="flex items-center gap-2 mt-2 text-xs text-gray-400">
                        <Clock className="w-3 h-3" />
                        {new Date(doc.updatedAt).toLocaleDateString()}
                        <Eye className="w-3 h-3 ml-2" />
                        {doc.views}
                      </div>
                    </div>
                  ))}
                </div>
              </ScrollArea>
            </CardContent>
          </Card>
        </div>

        <div className={styles.content}>
          <Tabs defaultValue="generate" className="space-y-6">
            <TabsList className="bg-white/60 dark:bg-gray-800/60 backdrop-blur-sm p-1 rounded-xl">
              <TabsTrigger value="generate" className={styles.tabTrigger}>
                <Sparkles className="w-4 h-4 mr-2" />
                Generate
              </TabsTrigger>
              <TabsTrigger value="preview" className={styles.tabTrigger}>
                <Eye className="w-4 h-4 mr-2" />
                Preview
              </TabsTrigger>
              <TabsTrigger value="settings" className={styles.tabTrigger}>
                <Settings className="w-4 h-4 mr-2" />
                Settings
              </TabsTrigger>
            </TabsList>

            <TabsContent value="generate">
              <Card className={styles.card}>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Code className="w-5 h-5 text-teal-500" />
                    Generate Documentation from Code
                  </CardTitle>
                  <CardDescription>
                    Paste your source code below and let AI generate comprehensive documentation
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <Textarea
                    className="min-h-[200px] font-mono"
                    placeholder="Paste your source code here..."
                    value={sourceCode}
                    onChange={(e) => setSourceCode(e.target.value)}
                  />

                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-4">
                      <div className="flex items-center gap-2">
                        <Switch checked={autoGenerate} onCheckedChange={setAutoGenerate} />
                        <span className="text-sm">Auto-generate on save</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <Switch checked={aiAssistance} onCheckedChange={setAiAssistance} />
                        <span className="text-sm">AI assistance</span>
                      </div>
                    </div>
                    <Button
                      className={styles.button}
                      onClick={handleGenerate}
                      disabled={isGenerating || !sourceCode.trim()}
                    >
                      {isGenerating ? (
                        <>
                          <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                          Generating...
                        </>
                      ) : (
                        <>
                          <Sparkles className="w-4 h-4 mr-2" />
                          Generate Documentation
                        </>
                      )}
                    </Button>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="preview">
              {selectedDoc ? (
                <Card className={styles.card}>
                  <CardHeader>
                    <div className="flex items-center justify-between">
                      <div>
                        <CardTitle className="flex items-center gap-2">
                          {getTypeIcon(selectedDoc.type)}
                          {selectedDoc.title}
                        </CardTitle>
                        <CardDescription>{selectedDoc.description}</CardDescription>
                      </div>
                      <div className="flex items-center gap-2">
                        <Button variant="ghost" size="sm" onClick={handleCopy}>
                          <Copy className="w-4 h-4 mr-1" />
                          Copy
                        </Button>
                        <Button variant="ghost" size="sm" onClick={handleDownload}>
                          <Download className="w-4 h-4 mr-1" />
                          Download
                        </Button>
                        <Button variant="ghost" size="sm">
                          <MoreHorizontal className="w-4 h-4" />
                        </Button>
                      </div>
                    </div>
                  </CardHeader>
                  <CardContent>
                    <div className="flex flex-wrap gap-2 mb-4">
                      {selectedDoc.tags.map((tag) => (
                        <Badge key={tag} variant="secondary">{tag}</Badge>
                      ))}
                    </div>
                    <Separator className="mb-4" />
                    <div className={styles.previewArea}>
                      <pre className="whitespace-pre-wrap">{selectedDoc.content}</pre>
                    </div>
                  </CardContent>
                </Card>
              ) : (
                <Card className={styles.card}>
                  <CardContent className="p-8 text-center">
                    <FileText className="w-16 h-16 mx-auto mb-4 text-gray-300" />
                    <p className="text-gray-500">Select a document to preview</p>
                  </CardContent>
                </Card>
              )}
            </TabsContent>

            <TabsContent value="settings">
              <Card className={styles.card}>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Settings className="w-5 h-5 text-teal-500" />
                    Documentation Settings
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-6">
                  <div>
                    <label className="text-sm font-medium mb-2 block">Default Language</label>
                    <Select defaultValue="markdown">
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="markdown">Markdown</SelectItem>
                        <SelectItem value="rst">reStructuredText</SelectItem>
                        <SelectItem value="html">HTML</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  <div>
                    <label className="text-sm font-medium mb-2 block">AI Model</label>
                    <Select defaultValue="gpt-4">
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="gpt-4">GPT-4</SelectItem>
                        <SelectItem value="claude-3">Claude 3</SelectItem>
                        <SelectItem value="gemini">Gemini Pro</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="space-y-4">
                    <div className="flex items-center justify-between">
                      <span className="text-sm">Include code examples</span>
                      <Switch defaultChecked />
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-sm">Auto-generate TOC</span>
                      <Switch defaultChecked />
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-sm">Include API references</span>
                      <Switch defaultChecked />
                    </div>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </div>
      </div>
    </div>
  );
}
