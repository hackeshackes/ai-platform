/**
 * Smart Recommender - AI Democratization Page
 * 智能推荐 - AI民主化页面
 */

import React, { useState, useCallback, useEffect } from 'react';
import {
  Sparkles,
  ThumbsUp,
  ThumbsDown,
  RefreshCw,
  History,
  Star,
  Zap,
  Target,
  TrendingUp,
  Clock,
  Eye,
  Bookmark,
  Share2,
  ChevronRight,
  Lightbulb,
  Brain,
  Cpu,
  Database,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Switch } from '@/components/ui/switch';
import { Slider } from '@/components/ui/slider';
import { Progress } from '@/components/ui/progress';
import { useToast } from '@/hooks/use-toast';
import { cn } from '@/lib/utils';

// Types
interface Recommendation {
  id: string;
  title: string;
  description: string;
  type: 'course' | 'tool' | 'workflow' | 'template' | 'article';
  relevanceScore: number;
  reason: string;
  tags: string[];
  thumbnail?: string;
  actions: {
    label: string;
    type: 'primary' | 'secondary';
    onClick: () => void;
  }[];
}

interface UserPreferences {
  interests: string[];
  skillLevel: 'beginner' | 'intermediate' | 'advanced';
  goals: string[];
  preferredFormats: string[];
  dailyTime: number;
}

// API Functions
const getRecommendations = async (preferences?: Partial<UserPreferences>): Promise<Recommendation[]> => {
  const response = await apiClient.post<Recommendation[]>('/api/v12/recommender/recommendations', preferences);
  return response.data;
};

const sendFeedback = async (recommendationId: string, feedback: 'like' | 'dislike'): Promise<void> => {
  await apiClient.post(`/api/v12/recommender/recommendations/${recommendationId}/feedback`, { feedback });
};

const updatePreferences = async (preferences: Partial<UserPreferences>): Promise<void> => {
  await apiClient.put('/api/v12/recommender/preferences', preferences);
};

// Sample Data
const SAMPLE_RECOMMENDATIONS: Recommendation[] = [
  {
    id: '1',
    title: 'Advanced Prompt Engineering',
    description: 'Master the art of crafting effective prompts for better AI responses',
    type: 'course',
    relevanceScore: 98,
    reason: 'Based on your recent NLP course progress',
    tags: ['Prompt Engineering', 'LLMs', 'Best Practices'],
    thumbnail: '💡',
    actions: [
      { label: 'Start Learning', type: 'primary', onClick: () => {} },
      { label: 'Save for Later', type: 'secondary', onClick: () => {} },
    ],
  },
  {
    id: '2',
    title: 'AutoML Workflow Builder',
    description: 'Automate your ML pipeline with this powerful no-code builder',
    type: 'tool',
    relevanceScore: 95,
    reason: 'Popular among users with your profile',
    tags: ['AutoML', 'Automation', 'Workflow'],
    thumbnail: '⚡',
    actions: [
      { label: 'Try It Now', type: 'primary', onClick: () => {} },
      { label: 'Learn More', type: 'secondary', onClick: () => {} },
    ],
  },
  {
    id: '3',
    title: 'Data Processing Template',
    description: 'Ready-to-use ETL pipeline for your data processing needs',
    type: 'template',
    relevanceScore: 92,
    reason: 'Matches your recent workflow usage',
    tags: ['Data', 'ETL', 'Template'],
    thumbnail: '📊',
    actions: [
      { label: 'Use Template', type: 'primary', onClick: () => {} },
      { label: 'Preview', type: 'secondary', onClick: () => {} },
    ],
  },
  {
    id: '4',
    title: 'Understanding Neural Networks',
    description: 'A comprehensive guide to how neural networks work under the hood',
    type: 'article',
    relevanceScore: 89,
    reason: 'Recommended based on your learning path',
    tags: ['Deep Learning', 'Neural Networks', 'Theory'],
    thumbnail: '🧠',
    actions: [
      { label: 'Read Article', type: 'primary', onClick: () => {} },
      { label: 'Bookmark', type: 'secondary', onClick: () => {} },
    ],
  },
  {
    id: '5',
    title: 'Customer Support Automation',
    description: 'Build an intelligent customer support system with AI',
    type: 'workflow',
    relevanceScore: 87,
    reason: 'Trending in your industry',
    tags: ['Support', 'Automation', 'Chatbot'],
    thumbnail: '💬',
    actions: [
      { label: 'Deploy Now', type: 'primary', onClick: () => {} },
      { label: 'View Demo', type: 'secondary', onClick: () => {} },
    ],
  },
];

// Styles
const styles = {
  container: 'min-h-screen bg-gradient-to-br from-pink-50 via-white to-rose-50 dark:from-gray-900 dark:via-gray-800 dark:to-gray-900 p-6',
  header: 'flex items-center justify-between mb-8',
  title: 'text-4xl font-bold bg-gradient-to-r from-pink-600 to-rose-600 bg-clip-text text-transparent',
  subtitle: 'text-gray-600 dark:text-gray-300 mt-2',
  mainGrid: 'grid grid-cols-1 lg:grid-cols-3 gap-6',
  content: 'lg:col-span-2',
  sidebar: 'space-y-6',
  card: 'bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm shadow-xl rounded-2xl border-0',
  recommendationCard: 'bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm shadow-xl rounded-2xl border-0 hover:shadow-2xl transition-all duration-300',
  badge: 'px-3 py-1 rounded-full text-xs font-medium',
  button: 'bg-gradient-to-r from-pink-600 to-rose-600 hover:from-pink-700 hover:to-rose-700 text-white shadow-lg',
  iconButton: 'p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors',
  scoreBar: 'h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden',
  scoreFill: 'h-full bg-gradient-to-r from-pink-500 to-rose-500',
  typeIcon: 'w-8 h-8 rounded-lg flex items-center justify-center text-xl',
};

// Contexts
const INTEREST_TAGS = [
  'Machine Learning', 'Deep Learning', 'NLP', 'Computer Vision',
  'Automation', 'Data Science', 'Robotics', 'Cloud Computing',
  'DevOps', 'API Development', 'Database', 'Analytics',
];

const GOALS = [
  'Learn New Skills', 'Improve Productivity', 'Build Projects',
  'Career Advancement', 'Research', 'Business Solutions',
];

export default function SmartRecommender() {
  const { toast } = useToast();
  
  const [recommendations, setRecommendations] = useState<Recommendation[]>(SAMPLE_RECOMMENDATIONS);
  const [preferences, setPreferences] = useState<UserPreferences>({
    interests: ['Machine Learning', 'NLP'],
    skillLevel: 'intermediate',
    goals: ['Learn New Skills', 'Build Projects'],
    preferredFormats: ['course', 'tool'],
    dailyTime: 30,
  });
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [feedbackGiven, setFeedbackGiven] = useState<Set<string>>(new Set());

  useEffect(() => {
    loadRecommendations();
  }, [preferences]);

  const loadRecommendations = async () => {
    try {
      const data = await getRecommendations(preferences);
      setRecommendations(data);
    } catch (error) {
      console.error('Failed to load recommendations:', error);
    }
  };

  const handleRefresh = useCallback(async () => {
    setIsRefreshing(true);
    try {
      await new Promise(resolve => setTimeout(resolve, 1000));
      loadRecommendations();
      toast({
        title: 'Recommendations Updated',
        description: 'Based on your latest activity',
      });
    } finally {
      setIsRefreshing(false);
    }
  }, [toast]);

  const handleFeedback = useCallback(async (id: string, feedback: 'like' | 'dislike') => {
    if (feedbackGiven.has(id)) return;

    setFeedbackGiven(prev => new Set(prev).add(id));
    try {
      await sendFeedback(id, feedback);
      toast({
        title: feedback === 'like' ? 'Thanks for the thumbs up!' : 'We\'ll improve recommendations',
        description: feedback === 'like' ? 'We\'ll show more like this' : 'We\'ll learn from your feedback',
      });
    } catch (error) {
      console.error('Failed to send feedback:', error);
    }
  }, [feedbackGiven, toast]);

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'course': return '📚';
      case 'tool': return '🛠️';
      case 'workflow': return '🔄';
      case 'template': return '📋';
      case 'article': return '📰';
      default: return '✨';
    }
  };

  const getTypeColor = (type: string) => {
    switch (type) {
      case 'course': return 'bg-blue-100 text-blue-600';
      case 'tool': return 'bg-green-100 text-green-600';
      case 'workflow': return 'bg-purple-100 text-purple-600';
      case 'template': return 'bg-amber-100 text-amber-600';
      case 'article': return 'bg-rose-100 text-rose-600';
      default: return 'bg-gray-100 text-gray-600';
    }
  };

  return (
    <div className={styles.container}>
      {/* Header */}
      <div className={styles.header}>
        <div>
          <h1 className={styles.title}>
            <Brain className="inline mr-2 h-8 w-8" />
            Smart Recommender
          </h1>
          <p className={styles.subtitle}>
            Personalized AI-powered recommendations based on your interests and goals
          </p>
        </div>
        <div className="flex items-center gap-3">
          <Button variant="outline" onClick={handleRefresh} disabled={isRefreshing}>
            <RefreshCw className={`w-4 h-4 mr-2 ${isRefreshing ? 'animate-spin' : ''}`} />
            Refresh
          </Button>
          <Button variant="outline">
            <History className="w-4 h-4 mr-2" />
            History
          </Button>
          <Button className={styles.button}>
            <Target className="w-4 h-4 mr-2" />
            Customize
          </Button>
        </div>
      </div>

      {/* Main Content */}
      <div className={styles.mainGrid}>
        {/* Recommendations */}
        <div className={styles.content}>
          <Tabs defaultValue="for-you" className="space-y-6">
            <TabsList className="bg-white/60 dark:bg-gray-800/60 backdrop-blur-sm p-1 rounded-xl">
              <TabsTrigger value="for-you" className="px-4 py-2 rounded-lg data-[state=active]:bg-pink-100 data-[state=active]:text-pink-700">
                <Sparkles className="w-4 h-4 mr-2" />
                For You
              </TabsTrigger>
              <TabsTrigger value="trending" className="px-4 py-2 rounded-lg data-[state=active]:bg-pink-100 data-[state=active]:text-pink-700">
                <TrendingUp className="w-4 h-4 mr-2" />
                Trending
              </TabsTrigger>
              <TabsTrigger value="new" className="px-4 py-2 rounded-lg data-[state=active]:bg-pink-100 data-[state=active]:text-pink-700">
                <Zap className="w-4 h-4 mr-2" />
                New for You
              </TabsTrigger>
            </TabsList>

            <TabsContent value="for-you" className="space-y-4">
              {recommendations.map((rec) => (
                <Card key={rec.id} className={styles.recommendationCard}>
                  <CardContent className="p-6">
                    <div className="flex gap-4">
                      <div className={cn('w-16 h-16 rounded-xl flex items-center justify-center text-3xl', getTypeColor(rec.type))}>
                        {rec.thumbnail || getTypeIcon(rec.type)}
                      </div>
                      <div className="flex-1">
                        <div className="flex items-start justify-between mb-2">
                          <div>
                            <h3 className="text-lg font-semibold flex items-center gap-2">
                              {rec.title}
                              {rec.relevanceScore >= 95 && (
                                <Badge className="bg-green-500">Top Match</Badge>
                              )}
                            </h3>
                            <p className="text-gray-500 text-sm">{rec.description}</p>
                          </div>
                          <div className="flex items-center gap-1 text-sm text-gray-500">
                            <Target className="w-4 h-4 text-pink-500" />
                            <span>{rec.relevanceScore}% match</span>
                          </div>
                        </div>

                        <div className="flex flex-wrap gap-2 mb-4">
                          {rec.tags.map((tag) => (
                            <Badge key={tag} variant="secondary">{tag}</Badge>
                          ))}
                        </div>

                        <div className="flex items-center gap-2 text-sm text-gray-500 mb-4">
                          <Lightbulb className="w-4 h-4 text-yellow-500" />
                          {rec.reason}
                        </div>

                        <div className="flex items-center justify-between">
                          <div className="flex gap-2">
                            {rec.actions.map((action, idx) => (
                              <Button
                                key={idx}
                                variant={action.type === 'primary' ? 'default' : 'outline'}
                                size="sm"
                                onClick={action.onClick}
                              >
                                {action.label}
                              </Button>
                            ))}
                          </div>
                          <div className="flex items-center gap-2">
                            <Button
                              variant="ghost"
                              size="sm"
                              onClick={() => handleFeedback(rec.id, 'like')}
                              disabled={feedbackGiven.has(rec.id)}
                            >
                              <ThumbsUp className={cn('w-4 h-4', feedbackGiven.has(rec.id) && 'text-green-500')} />
                            </Button>
                            <Button
                              variant="ghost"
                              size="sm"
                              onClick={() => handleFeedback(rec.id, 'dislike')}
                              disabled={feedbackGiven.has(rec.id)}
                            >
                              <ThumbsDown className={cn('w-4 h-4', feedbackGiven.has(rec.id) && 'text-red-500')} />
                            </Button>
                          </div>
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </TabsContent>

            <TabsContent value="trending">
              <Card className={styles.card}>
                <CardContent className="p-8 text-center">
                  <TrendingUp className="w-16 h-16 mx-auto mb-4 text-pink-300" />
                  <h3 className="text-xl font-semibold mb-2">Trending Recommendations</h3>
                  <p className="text-gray-500">See what's popular among users like you</p>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="new">
              <Card className={styles.card}>
                <CardContent className="p-8 text-center">
                  <Zap className="w-16 h-16 mx-auto mb-4 text-yellow-400" />
                  <h3 className="text-xl font-semibold mb-2">New Arrivals</h3>
                  <p className="text-gray-500">Fresh recommendations based on your interests</p>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </div>

        {/* Sidebar - Preferences */}
        <div className={styles.sidebar}>
          <Card className={styles.card}>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Cpu className="w-5 h-5 text-pink-500" />
                Your Preferences
              </CardTitle>
              <CardDescription>
                Help us improve recommendations
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div>
                <label className="text-sm font-medium mb-2 block">Skill Level</label>
                <Select
                  value={preferences.skillLevel}
                  onValueChange={(v: any) => setPreferences(p => ({ ...p, skillLevel: v }))}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="beginner">Beginner</SelectItem>
                    <SelectItem value="intermediate">Intermediate</SelectItem>
                    <SelectItem value="advanced">Advanced</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div>
                <label className="text-sm font-medium mb-2 block">Daily Learning Time</label>
                <div className="space-y-2">
                  <Slider
                    value={[preferences.dailyTime]}
                    onValueChange={([v]) => setPreferences(p => ({ ...p, dailyTime: v }))}
                    min={5}
                    max={120}
                    step={5}
                  />
                  <p className="text-sm text-gray-500">{preferences.dailyTime} minutes/day</p>
                </div>
              </div>

              <div>
                <label className="text-sm font-medium mb-2 block">Interests</label>
                <div className="flex flex-wrap gap-2">
                  {INTEREST_TAGS.map((tag) => (
                    <Button
                      key={tag}
                      variant={preferences.interests.includes(tag) ? 'default' : 'outline'}
                      size="sm"
                      className="rounded-full"
                      onClick={() => {
                        setPreferences(p => ({
                          ...p,
                          interests: p.interests.includes(tag)
                            ? p.interests.filter(i => i !== tag)
                            : [...p.interests, tag],
                        }));
                      }}
                    >
                      {tag}
                    </Button>
                  ))}
                </div>
              </div>

              <div>
                <label className="text-sm font-medium mb-2 block">Goals</label>
                <div className="space-y-2">
                  {GOALS.map((goal) => (
                    <div key={goal} className="flex items-center gap-2">
                      <Switch
                        checked={preferences.goals.includes(goal)}
                        onCheckedChange={(v) => {
                          setPreferences(p => ({
                            ...p,
                            goals: v
                              ? [...p.goals, goal]
                              : p.goals.filter(g => g !== goal),
                          }));
                        }}
                      />
                      <span className="text-sm">{goal}</span>
                    </div>
                  ))}
                </div>
              </div>

              <Button className="w-full" onClick={() => updatePreferences(preferences)}>
                Save Preferences
              </Button>
            </CardContent>
          </Card>

          {/* Stats */}
          <Card className={styles.card}>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Star className="w-5 h-5 text-yellow-500" />
                Recommendation Stats
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-500">Likes Given</span>
                <span className="font-semibold">24</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-500">Items Saved</span>
                <span className="font-semibold">12</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-500">Items Viewed</span>
                <span className="font-semibold">156</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-500">Accuracy Score</span>
                <span className="font-semibold text-green-500">94%</span>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
