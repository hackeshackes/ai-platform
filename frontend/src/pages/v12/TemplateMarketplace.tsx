/**
 * Template Marketplace - AI Democratization Page
 * AI模板市场 - AI民主化页面
 */

import React, { useState, useCallback, useEffect } from 'react';
import {
  Search,
  Star,
  Download,
  Eye,
  Heart,
  Share2,
  Filter,
  Grid,
  List,
  Clock,
  User,
  Tag,
  TrendingUp,
  Sparkles,
  ChevronRight,
  ExternalLink,
  Check,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Slider } from '@/components/ui/slider';
import { Checkbox } from '@/components/ui/checkbox';
import { Progress } from '@/components/ui/progress';
import { useToast } from '@/hooks/use-toast';
import { cn } from '@/lib/utils';

// Types
interface Template {
  id: string;
  name: string;
  description: string;
  category: string;
  tags: string[];
  author: string;
  downloads: number;
  likes: number;
  views: number;
  rating: number;
  thumbnail: string;
  isPremium: boolean;
  isNew: boolean;
  isFeatured: boolean;
  createdAt: Date;
  updatedAt: Date;
}

interface Category {
  id: string;
  name: string;
  icon: string;
  count: number;
}

// API Functions
const getTemplates = async (filters?: Record<string, any>): Promise<Template[]> => {
  const response = await apiClient.get<Template[]>('/api/v12/template-marketplace/templates', { params: filters });
  return response.data;
};

const getCategories = async (): Promise<Category[]> => {
  const response = await apiClient.get<Category[]>('/api/v12/template-marketplace/categories');
  return response.data;
};

const downloadTemplate = async (templateId: string): Promise<Blob> => {
  const response = await apiClient.get(`/api/v12/template-marketplace/templates/${templateId}/download`, {
    responseType: 'blob',
  });
  return response.data;
};

const likeTemplate = async (templateId: string): Promise<void> => {
  await apiClient.post(`/api/v12/template-marketplace/templates/${templateId}/like`);
};

// Sample Categories
const SAMPLE_CATEGORIES: Category[] = [
  { id: 'all', name: 'All Templates', icon: '📦', count: 156 },
  { id: 'workflow', name: 'Workflows', icon: '🔄', count: 45 },
  { id: 'chatbot', name: 'Chatbots', icon: '💬', count: 32 },
  { id: 'analytics', name: 'Analytics', icon: '📊', count: 28 },
  { id: 'automation', name: 'Automation', icon: '⚡', count: 24 },
  { id: 'integration', name: 'Integrations', icon: '🔗', count: 18 },
  { id: 'document', name: 'Documents', icon: '📄', count: 9 },
];

// Sample Templates
const SAMPLE_TEMPLATES: Template[] = [
  {
    id: '1',
    name: 'Customer Support Bot',
    description: 'Advanced customer support chatbot with sentiment analysis and ticket routing',
    category: 'chatbot',
    tags: ['support', 'sentiment', 'tickets'],
    author: 'AI Labs',
    downloads: 12500,
    likes: 892,
    views: 45000,
    rating: 4.8,
    thumbnail: '🤖',
    isPremium: false,
    isNew: false,
    isFeatured: true,
    createdAt: new Date('2024-01-15'),
    updatedAt: new Date('2024-02-01'),
  },
  {
    id: '2',
    name: 'Data Processing Pipeline',
    description: 'Automated ETL pipeline for data transformation and analysis',
    category: 'workflow',
    tags: ['ETL', 'data', 'pipeline'],
    author: 'Data Team',
    downloads: 8300,
    likes: 654,
    views: 32000,
    rating: 4.6,
    thumbnail: '📈',
    isPremium: true,
    isNew: true,
    isFeatured: false,
    createdAt: new Date('2024-02-01'),
    updatedAt: new Date('2024-02-10'),
  },
  {
    id: '3',
    name: 'Content Generator',
    description: 'AI-powered content generation for blogs, social media, and marketing',
    category: 'automation',
    tags: ['content', 'marketing', 'AI'],
    author: 'Marketing Pro',
    downloads: 15600,
    likes: 1200,
    views: 78000,
    rating: 4.9,
    thumbnail: '✍️',
    isPremium: false,
    isNew: false,
    isFeatured: true,
    createdAt: new Date('2023-12-01'),
    updatedAt: new Date('2024-01-20'),
  },
];

// Styles
const styles = {
  container: 'min-h-screen bg-gradient-to-br from-amber-50 via-white to-orange-50 dark:from-gray-900 dark:via-gray-800 dark:to-gray-900 p-6',
  header: 'flex items-center justify-between mb-8',
  title: 'text-4xl font-bold bg-gradient-to-r from-amber-600 to-orange-600 bg-clip-text text-transparent',
  subtitle: 'text-gray-600 dark:text-gray-300 mt-2',
  mainGrid: 'grid grid-cols-1 lg:grid-cols-4 gap-6',
  sidebar: 'space-y-6',
  content: 'lg:col-span-3',
  searchBar: 'relative flex-1',
  searchIcon: 'absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400',
  searchInput: 'pl-10 pr-4 h-12 rounded-xl border-2 border-amber-200 focus:border-amber-500 dark:border-gray-700 dark:focus:border-amber-500',
  card: 'bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm shadow-xl rounded-2xl border-0 overflow-hidden hover:shadow-2xl transition-shadow cursor-pointer',
  cardGrid: 'grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6',
  cardThumbnail: 'h-40 flex items-center justify-center text-6xl bg-gradient-to-br from-amber-100 to-orange-100 dark:from-amber-900/30 dark:to-orange-900/30',
  cardBadge: 'absolute top-3 right-3',
  statIcon: 'w-4 h-4',
  statText: 'text-sm text-gray-500',
  filterSection: 'bg-white/60 dark:bg-gray-800/60 backdrop-blur-sm rounded-xl p-4',
  categoryItem: 'flex items-center gap-3 p-3 rounded-lg hover:bg-amber-100 dark:hover:bg-amber-900/30 cursor-pointer transition-colors',
  categoryItemActive: 'bg-amber-100 dark:bg-amber-900/30',
  button: 'bg-gradient-to-r from-amber-600 to-orange-600 hover:from-amber-700 hover:to-orange-700 text-white shadow-lg',
  iconButton: 'p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors',
};

export default function TemplateMarketplace() {
  const { toast } = useToast();
  
  const [templates, setTemplates] = useState<Template[]>(SAMPLE_TEMPLATES);
  const [categories] = useState<Category[]>(SAMPLE_CATEGORIES);
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid');
  const [sortBy, setSortBy] = useState('popular');
  const [priceFilter, setPriceFilter] = useState<'all' | 'free' | 'premium'>('all');
  const [likedTemplates, setLikedTemplates] = useState<Set<string>>(new Set());
  const [isDownloading, setIsDownloading] = useState<string | null>(null);

  useEffect(() => {
    // In real app, load templates from API
    loadTemplates();
  }, [selectedCategory, searchQuery, sortBy, priceFilter]);

  const loadTemplates = async () => {
    try {
      const filters = {
        category: selectedCategory !== 'all' ? selectedCategory : undefined,
        search: searchQuery,
        sort: sortBy,
        premium: priceFilter === 'premium' ? true : priceFilter === 'free' ? false : undefined,
      };
      const data = await getTemplates(filters);
      setTemplates(data);
    } catch (error) {
      console.error('Failed to load templates:', error);
    }
  };

  const handleDownload = useCallback(async (template: Template) => {
    setIsDownloading(template.id);
    try {
      const blob = await downloadTemplate(template.id);
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${template.name.replace(/\s+/g, '_')}.zip`;
      a.click();
      URL.revokeObjectURL(url);

      toast({
        title: 'Download Started',
        description: `${template.name} is being downloaded`,
      });
    } catch (error) {
      toast({
        title: 'Download Failed',
        description: 'There was an error downloading the template',
        variant: 'destructive',
      });
    } finally {
      setIsDownloading(null);
    }
  }, [toast]);

  const handleLike = useCallback(async (templateId: string) => {
    if (likedTemplates.has(templateId)) {
      setLikedTemplates(prev => {
        const next = new Set(prev);
        next.delete(templateId);
        return next;
      });
    } else {
      setLikedTemplates(prev => new Set(prev).add(templateId));
      try {
        await likeTemplate(templateId);
      } catch (error) {
        console.error('Failed to like template:', error);
      }
    }
  }, [likedTemplates]);

  const handleShare = useCallback((template: Template) => {
    navigator.clipboard.writeText(`${window.location.origin}/templates/${template.id}`);
    toast({
      title: 'Link Copied',
      description: 'Template link copied to clipboard',
    });
  }, [toast]);

  const formatNumber = (num: number): string => {
    if (num >= 1000) {
      return (num / 1000).toFixed(1) + 'K';
    }
    return num.toString();
  };

  const filteredTemplates = templates.filter(t => {
    if (searchQuery) {
      const query = searchQuery.toLowerCase();
      return (
        t.name.toLowerCase().includes(query) ||
        t.description.toLowerCase().includes(query) ||
        t.tags.some(tag => tag.toLowerCase().includes(query))
      );
    }
    return true;
  });

  return (
    <div className={styles.container}>
      {/* Header */}
      <div className={styles.header}>
        <div>
          <h1 className={styles.title}>
            <Sparkles className="inline mr-2 h-8 w-8" />
            Template Marketplace
          </h1>
          <p className={styles.subtitle}>
            Discover and share AI-powered templates for every use case
          </p>
        </div>
        <div className="flex items-center gap-3">
          <Button variant="outline">
            <Share2 className="w-4 h-4 mr-2" />
            Share
          </Button>
          <Button className={styles.button}>
            <Download className="w-4 h-4 mr-2" />
            My Templates
          </Button>
        </div>
      </div>

      {/* Main Content */}
      <div className={styles.mainGrid}>
        {/* Sidebar */}
        <div className={styles.sidebar}>
          {/* Search */}
          <div className={styles.searchBar}>
            <Search className={styles.searchIcon} />
            <Input
              className={styles.searchInput}
              placeholder="Search templates..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
          </div>

          {/* Categories */}
          <Card className={styles.filterSection}>
            <h3 className="font-semibold mb-3 flex items-center gap-2">
              <Filter className="w-4 h-4" />
              Categories
            </h3>
            <div className="space-y-1">
              {categories.map((category) => (
                <div
                  key={category.id}
                  className={cn(
                    styles.categoryItem,
                    selectedCategory === category.id && styles.categoryItemActive
                  )}
                  onClick={() => setSelectedCategory(category.id)}
                >
                  <span className="text-xl">{category.icon}</span>
                  <span className="flex-1">{category.name}</span>
                  <Badge variant="secondary">{category.count}</Badge>
                </div>
              ))}
            </div>
          </Card>

          {/* Filters */}
          <Card className={styles.filterSection}>
            <h3 className="font-semibold mb-3 flex items-center gap-2">
              <Filter className="w-4 h-4" />
              Filters
            </h3>
            
            <div className="space-y-4">
              <div>
                <label className="text-sm font-medium mb-2 block">Price</label>
                <Select value={priceFilter} onValueChange={(v: any) => setPriceFilter(v)}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">All Prices</SelectItem>
                    <SelectItem value="free">Free</SelectItem>
                    <SelectItem value="premium">Premium</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div>
                <label className="text-sm font-medium mb-2 block">Rating</label>
                <div className="space-y-2">
                  {[4, 3, 2, 1].map((rating) => (
                    <div key={rating} className="flex items-center gap-2">
                      <Checkbox id={`rating-${rating}`} />
                      <label htmlFor={`rating-${rating}`} className="flex items-center gap-1 text-sm">
                        {'⭐'.repeat(rating)}
                        {'☆'.repeat(5 - rating)}
                        <span className="text-gray-500">& up</span>
                      </label>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </Card>
        </div>

        {/* Content */}
        <div className={styles.content}>
          {/* Toolbar */}
          <div className="flex items-center justify-between mb-6">
            <p className="text-gray-600 dark:text-gray-300">
              Showing {filteredTemplates.length} templates
            </p>
            <div className="flex items-center gap-3">
              <Select value={sortBy} onValueChange={setSortBy}>
                <SelectTrigger className="w-[160px]">
                  <SelectValue placeholder="Sort by" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="popular">
                    <span className="flex items-center gap-2">
                      <TrendingUp className="w-4 h-4" /> Popular
                    </span>
                  </SelectItem>
                  <SelectItem value="newest">
                    <span className="flex items-center gap-2">
                      <Clock className="w-4 h-4" /> Newest
                    </span>
                  </SelectItem>
                  <SelectItem value="rating">
                    <span className="flex items-center gap-2">
                      <Star className="w-4 h-4" /> Top Rated
                    </span>
                  </SelectItem>
                  <SelectItem value="downloads">
                    <span className="flex items-center gap-2">
                      <Download className="w-4 h-4" /> Most Downloaded
                    </span>
                  </SelectItem>
                </SelectContent>
              </Select>

              <div className="flex items-center border rounded-lg">
                <Button
                  variant={viewMode === 'grid' ? 'secondary' : 'ghost'}
                  size="sm"
                  onClick={() => setViewMode('grid')}
                >
                  <Grid className="w-4 h-4" />
                </Button>
                <Button
                  variant={viewMode === 'list' ? 'secondary' : 'ghost'}
                  size="sm"
                  onClick={() => setViewMode('list')}
                >
                  <List className="w-4 h-4" />
                </Button>
              </div>
            </div>
          </div>

          {/* Templates Grid */}
          <div className={cn(styles.cardGrid, viewMode === 'list' && 'grid-cols-1')}>
            {filteredTemplates.map((template) => (
              <Card
                key={template.id}
                className={cn(styles.card, viewMode === 'list' && 'flex-row')}
              >
                <div className={cn(styles.cardThumbnail, viewMode === 'list' && 'w-40 h-auto')}>
                  <span className="text-8xl">{template.thumbnail}</span>
                </div>
                <div className="flex-1">
                  <CardHeader className="pb-2">
                    <div className="flex items-start justify-between">
                      <div>
                        <CardTitle className="text-lg flex items-center gap-2">
                          {template.name}
                          {template.isNew && (
                            <Badge className="bg-green-500">New</Badge>
                          )}
                          {template.isPremium && (
                            <Badge className="bg-amber-500">Premium</Badge>
                          )}
                        </CardTitle>
                        <CardDescription className="mt-1">
                          {template.description}
                        </CardDescription>
                      </div>
                      <div className={styles.cardBadge}>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={(e) => {
                            e.stopPropagation();
                            handleLike(template.id);
                          }}
                        >
                          <Heart
                            className={cn(
                              'w-5 h-5',
                              likedTemplates.has(template.id) && 'fill-red-500 text-red-500'
                            )}
                          />
                        </Button>
                      </div>
                    </div>
                  </CardHeader>
                  <CardContent className="pb-2">
                    <div className="flex flex-wrap gap-1 mb-3">
                      {template.tags.slice(0, 3).map((tag) => (
                        <Badge key={tag} variant="secondary" className="text-xs">
                          {tag}
                        </Badge>
                      ))}
                    </div>
                    <div className="flex items-center gap-4 text-sm text-gray-500">
                      <span className="flex items-center gap-1">
                        <Star className="w-4 h-4 text-yellow-500" />
                        {template.rating}
                      </span>
                      <span className="flex items-center gap-1">
                        <Download className="w-4 h-4" />
                        {formatNumber(template.downloads)}
                      </span>
                      <span className="flex items-center gap-1">
                        <Eye className="w-4 h-4" />
                        {formatNumber(template.views)}
                      </span>
                    </div>
                  </CardContent>
                  <CardFooter className="pt-2 flex items-center justify-between">
                    <span className="text-sm text-gray-500 flex items-center gap-1">
                      <User className="w-4 h-4" />
                      {template.author}
                    </span>
                    <div className="flex gap-2">
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={(e) => {
                          e.stopPropagation();
                          handleShare(template);
                        }}
                      >
                        <Share2 className="w-4 h-4" />
                      </Button>
                      <Button
                        className={styles.button}
                        size="sm"
                        onClick={(e) => {
                          e.stopPropagation();
                          handleDownload(template);
                        }}
                        disabled={isDownloading === template.id}
                      >
                        {isDownloading === template.id ? (
                          <>
                            <Progress value={45} className="w-12 h-1" />
                          </>
                        ) : (
                          <>
                            <Download className="w-4 h-4 mr-1" />
                            Use
                          </>
                        )}
                      </Button>
                    </div>
                  </CardFooter>
                </div>
              </Card>
            ))}
          </div>

          {filteredTemplates.length === 0 && (
            <div className="text-center py-16">
              <Sparkles className="w-16 h-16 mx-auto mb-4 text-gray-300" />
              <p className="text-xl font-medium text-gray-500">No templates found</p>
              <p className="text-gray-400">Try adjusting your search or filters</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
