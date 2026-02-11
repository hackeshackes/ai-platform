export interface Course {
  id: string;
  title: string;
  description: string;
  duration: string; // e.g., "5分钟", "30分钟"
  durationMinutes: number; // actual minutes for sorting
  level: 'beginner' | 'intermediate' | 'advanced';
  category: string;
  thumbnail?: string;
  videoUrl?: string;
  chapters: Chapter[];
  prerequisites?: string[];
  tags: string[];
}

export interface Chapter {
  id: string;
  title: string;
  content: string;
  duration: string;
  type: 'video' | 'reading' | 'exercise' | 'quiz';
  completed?: boolean;
}

export interface UserProgress {
  courseId: string;
  completedChapters: string[];
  totalChapters: number;
  percentComplete: number;
  lastAccessed: Date;
  startedAt: Date;
}

export interface CourseReview {
  id: string;
  userId: string;
  userName: string;
  rating: number; // 1-5
  comment: string;
  createdAt: Date;
  helpful: number;
}

export interface EducationState {
  courses: Course[];
  userProgress: Record<string, UserProgress>;
  reviews: Record<string, CourseReview[]>;
  currentCourse: Course | null;
  currentChapter: Chapter | null;
}

// Course list configuration
export const COURSES: Course[] = [
  {
    id: 'quickstart',
    title: '快速入门',
    description: '5分钟内了解AI Platform，创建并运行第一个Agent和Pipeline',
    duration: '5分钟',
    durationMinutes: 5,
    level: 'beginner',
    category: 'getting-started',
    chapters: [
      {
        id: 'what-is-ai-platform',
        title: '什么是AI Platform',
        content: 'ai-platform-intro',
        duration: '2分钟',
        type: 'video'
      },
      {
        id: 'create-first-agent',
        title: '创建第一个Agent',
        content: 'first-agent-creation',
        duration: '2分钟',
        type: 'exercise'
      },
      {
        id: 'run-first-pipeline',
        title: '运行第一个Pipeline',
        content: 'first-pipeline-run',
        duration: '1分钟',
        type: 'exercise'
      }
    ],
    tags: ['入门', 'Agent', 'Pipeline', '基础']
  },
  {
    id: 'agent-creation',
    title: 'Agent创建',
    description: '深入学习Agent类型选择、技能配置、记忆配置和测试方法',
    duration: '30分钟',
    durationMinutes: 30,
    level: 'intermediate',
    category: 'agents',
    chapters: [
      {
        id: 'agent-types',
        title: 'Agent类型选择',
        content: 'agent-types-guide',
        duration: '5分钟',
        type: 'video'
      },
      {
        id: 'skills-configuration',
        title: '技能配置',
        content: 'skills-config-guide',
        duration: '10分钟',
        type: 'video'
      },
      {
        id: 'memory-configuration',
        title: '记忆配置',
        content: 'memory-config-guide',
        duration: '8分钟',
        type: 'video'
      },
      {
        id: 'testing-agents',
        title: '测试Agent',
        content: 'agent-testing-guide',
        duration: '7分钟',
        type: 'exercise'
      }
    ],
    prerequisites: ['quickstart'],
    tags: ['Agent', '技能', '记忆', '测试']
  },
  {
    id: 'pipeline-building',
    title: 'Pipeline构建',
    description: '学习Pipeline设计理念、节点连接、条件分支和错误处理',
    duration: '1小时',
    durationMinutes: 60,
    level: 'intermediate',
    category: 'pipelines',
    chapters: [
      {
        id: 'pipeline-concepts',
        title: 'Pipeline概念',
        content: 'pipeline-concepts',
        duration: '10分钟',
        type: 'video'
      },
      {
        id: 'node-connections',
        title: '节点连接',
        content: 'node-connections',
        duration: '15分钟',
        type: 'video'
      },
      {
        id: 'conditional-branching',
        title: '条件分支',
        content: 'conditional-branching',
        duration: '20分钟',
        type: 'video'
      },
      {
        id: 'error-handling',
        title: '错误处理',
        content: 'error-handling-pipelines',
        duration: '15分钟',
        type: 'video'
      }
    ],
    prerequisites: ['quickstart', 'agent-creation'],
    tags: ['Pipeline', '节点', '条件', '错误处理']
  },
  {
    id: 'advanced-tips',
    title: '高级技巧',
    description: '掌握性能优化、调试技巧和开发最佳实践',
    duration: '2小时',
    durationMinutes: 120,
    level: 'advanced',
    category: 'advanced',
    chapters: [
      {
        id: 'performance-optimization',
        title: '性能优化',
        content: 'performance-tuning',
        duration: '45分钟',
        type: 'video'
      },
      {
        id: 'debugging-techniques',
        title: '调试技巧',
        content: 'debugging-guide',
        duration: '40分钟',
        type: 'video'
      },
      {
        id: 'best-practices-advanced',
        title: '最佳实践',
        content: 'advanced-best-practices',
        duration: '35分钟',
        type: 'reading'
      }
    ],
    prerequisites: ['agent-creation', 'pipeline-building'],
    tags: ['性能', '调试', '最佳实践', '高级']
  },
  {
    id: 'best-practices',
    title: '最佳实践',
    description: '学习项目结构、代码规范和测试策略',
    duration: '2小时',
    durationMinutes: 120,
    level: 'intermediate',
    category: 'development',
    chapters: [
      {
        id: 'project-structure',
        title: '项目结构',
        content: 'project-structure-guide',
        duration: '30分钟',
        type: 'video'
      },
      {
        id: 'code-standards',
        title: '代码规范',
        content: 'coding-standards',
        duration: '40分钟',
        type: 'video'
      },
      {
        id: 'testing-strategies',
        title: '测试策略',
        content: 'testing-strategies',
        duration: '50分钟',
        type: 'video'
      }
    ],
    prerequisites: ['quickstart'],
    tags: ['项目结构', '代码规范', '测试', '开发']
  },
  {
    id: 'troubleshooting',
    title: '故障排除',
    description: '解决常见问题、分析日志和使用调试工具',
    duration: '1小时',
    durationMinutes: 60,
    level: 'intermediate',
    category: 'support',
    chapters: [
      {
        id: 'common-issues',
        title: '常见问题',
        content: 'common-problems-solutions',
        duration: '20分钟',
        type: 'reading'
      },
      {
        id: 'log-analysis',
        title: '日志分析',
        content: 'log-analysis-guide',
        duration: '25分钟',
        type: 'video'
      },
      {
        id: 'debugging-tools',
        title: '调试工具',
        content: 'debugging-tools-reference',
        duration: '15分钟',
        type: 'video'
      }
    ],
    prerequisites: ['quickstart'],
    tags: ['故障', '日志', '调试', '问题解决']
  },
  {
    id: 'performance',
    title: '性能优化',
    description: '深入学习缓存策略、并发优化和资源管理',
    duration: '2小时',
    durationMinutes: 120,
    level: 'advanced',
    category: 'optimization',
    chapters: [
      {
        id: 'caching-strategies',
        title: '缓存策略',
        content: 'caching-deep-dive',
        duration: '40分钟',
        type: 'video'
      },
      {
        id: 'concurrency-optimization',
        title: '并发优化',
        content: 'concurrency-optimization',
        duration: '45分钟',
        type: 'video'
      },
      {
        id: 'resource-management',
        title: '资源管理',
        content: 'resource-management-guide',
        duration: '35分钟',
        type: 'video'
      }
    ],
    prerequisites: ['advanced-tips', 'best-practices'],
    tags: ['缓存', '并发', '资源', '性能']
  },
  {
    id: 'templates',
    title: '模板使用',
    description: '选择合适模板、自定义配置和部署运行',
    duration: '30分钟',
    durationMinutes: 30,
    level: 'beginner',
    category: 'templates',
    chapters: [
      {
        id: 'choosing-templates',
        title: '选择模板',
        content: 'template-selection-guide',
        duration: '10分钟',
        type: 'video'
      },
      {
        id: 'custom-config',
        title: '自定义配置',
        content: 'template-customization',
        duration: '12分钟',
        type: 'exercise'
      },
      {
        id: 'deployment',
        title: '部署运行',
        content: 'template-deployment',
        duration: '8分钟',
        type: 'exercise'
      }
    ],
    prerequisites: ['quickstart'],
    tags: ['模板', '配置', '部署', '快速开始']
  },
  {
    id: 'integration',
    title: '集成开发',
    description: '学习API集成、Webhook和第三方服务集成',
    duration: '3小时',
    durationMinutes: 180,
    level: 'advanced',
    category: 'integration',
    chapters: [
      {
        id: 'api-integration',
        title: 'API集成',
        content: 'api-integration-guide',
        duration: '60分钟',
        type: 'video'
      },
      {
        id: 'webhook-setup',
        title: 'Webhook',
        content: 'webhook-development',
        duration: '50分钟',
        type: 'video'
      },
      {
        id: 'third-party-integrations',
        title: '第三方集成',
        content: 'third-party-services',
        duration: '70分钟',
        type: 'video'
      }
    ],
    prerequisites: ['agent-creation', 'pipeline-building'],
    tags: ['API', 'Webhook', '集成', '第三方']
  },
  {
    id: 'project',
    title: '实战项目',
    description: '完整的项目规划、实现、测试和部署流程',
    duration: '4小时',
    durationMinutes: 240,
    level: 'advanced',
    category: 'project',
    chapters: [
      {
        id: 'project-planning',
        title: '项目规划',
        content: 'project-planning-guide',
        duration: '45分钟',
        type: 'video'
      },
      {
        id: 'complete-implementation',
        title: '完整实现',
        content: 'full-implementation-walkthrough',
        duration: '150分钟',
        type: 'video'
      },
      {
        id: 'testing-deployment',
        title: '测试部署',
        content: 'testing-deployment-guide',
        duration: '45分钟',
        type: 'exercise'
      }
    ],
    prerequisites: ['agent-creation', 'pipeline-building', 'best-practices', 'integration'],
    tags: ['实战', '项目', '完整流程', '综合']
  }
];

export const COURSE_CATEGORIES = [
  { id: 'getting-started', name: '快速入门', icon: '🚀' },
  { id: 'agents', name: 'Agent开发', icon: '🤖' },
  { id: 'pipelines', name: 'Pipeline构建', icon: '🔗' },
  { id: 'advanced', name: '高级技巧', icon: '⚡' },
  { id: 'development', name: '开发实践', icon: '📝' },
  { id: 'support', name: '故障排除', icon: '🔧' },
  { id: 'optimization', name: '性能优化', icon: '💨' },
  { id: 'templates', name: '模板使用', icon: '📋' },
  { id: 'integration', name: '集成开发', icon: '🔌' },
  { id: 'project', name: '实战项目', icon: '🎯' }
];

export const LEVEL_LABELS = {
  beginner: '入门',
  intermediate: '中级',
  advanced: '高级'
};

export const LEVEL_COLORS = {
  beginner: '#10B981',
  intermediate: '#F59E0B',
  advanced: '#EF4444'
};
