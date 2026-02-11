import React, { useState, useMemo, useCallback } from 'react';
import { Course, COURSES, COURSE_CATEGORIES, LEVEL_LABELS, UserProgress, EducationState } from './types';
import { CourseCard } from './components/CourseCard';
import { VideoPlayer } from './components/VideoPlayer';
import { CodeEditor } from './components/CodeEditor';
import { ProgressBar, StepProgress, CircularProgress } from './components/ProgressBar';
import './styles.css';

// 主教育中心组件
export const EducationCenter: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'courses' | 'my-learning' | 'achievements'>('courses');
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedCourse, setSelectedCourse] = useState<Course | null>(null);
  const [currentChapter, setCurrentChapter] = useState<number>(0);
  
  // 用户进度状态
  const [userProgress, setUserProgress] = useState<Record<string, UserProgress>>({});

  // 过滤课程
  const filteredCourses = useMemo(() => {
    return COURSES.filter(course => {
      const matchesCategory = !selectedCategory || course.category === selectedCategory;
      const matchesSearch = !searchQuery || 
        course.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
        course.description.toLowerCase().includes(searchQuery.toLowerCase()) ||
        course.tags.some(tag => tag.toLowerCase().includes(searchQuery.toLowerCase()));
      return matchesCategory && matchesSearch;
    });
  }, [selectedCategory, searchQuery]);

  // 计算总学习进度
  const totalProgress = useMemo(() => {
    const total = COURSES.length;
    if (total === 0) return 0;
    const completed = Object.values(userProgress).filter(p => p.percentComplete === 100).length;
    return (completed / total) * 100;
  }, [userProgress]);

  // 处理课程选择
  const handleCourseSelect = useCallback((course: Course) => {
    setSelectedCourse(course);
    setCurrentChapter(0);
  }, []);

  // 处理章节完成
  const handleChapterComplete = useCallback((courseId: string, chapterId: string) => {
    setUserProgress(prev => {
      const courseProgress = prev[courseId] || {
        courseId,
        completedChapters: [],
        totalChapters: COURSES.find(c => c.id === courseId)?.chapters.length || 0,
        percentComplete: 0,
        lastAccessed: new Date(),
        startedAt: new Date()
      };

      if (!courseProgress.completedChapters.includes(chapterId)) {
        courseProgress.completedChapters.push(chapterId);
        courseProgress.percentComplete = (courseProgress.completedChapters.length / courseProgress.totalChapters) * 100;
        courseProgress.lastAccessed = new Date();
      }

      return { ...prev, [courseId]: courseProgress };
    });
  }, []);

  // 进入课程详情
  const enterCourse = useCallback((course: Course) => {
    setUserProgress(prev => {
      if (prev[course.id]) return prev;
      
      return {
        ...prev,
        [course.id]: {
          courseId: course.id,
          completedChapters: [],
          totalChapters: course.chapters.length,
          percentComplete: 0,
          lastAccessed: new Date(),
          startedAt: new Date()
        }
      };
    });
    handleCourseSelect(course);
  }, [handleCourseSelect]);

  // 渲染课程卡片网格
  const renderCourseGrid = () => (
    <div className="course-grid">
      {filteredCourses.map(course => {
        const progress = userProgress[course.id];
        const isLocked = course.prerequisites?.some(
          prereq => !userProgress[prereq] || userProgress[prereq].percentComplete < 100
        );

        return (
          <CourseCard
            key={course.id}
            course={course}
            progress={progress?.percentComplete || 0}
            isLocked={isLocked}
            onClick={enterCourse}
            onContinue={handleCourseSelect}
          />
        );
      })}
    </div>
  );

  // 渲染课程详情页
  const renderCourseDetail = () => {
    if (!selectedCourse) return null;

    const progress = userProgress[selectedCourse.id];
    const chapter = selectedCourse.chapters[currentChapter];

    return (
      <div className="course-detail">
        <button 
          className="back-btn"
          onClick={() => setSelectedCourse(null)}
        >
          ← 返回课程列表
        </button>

        <div className="course-header">
          <h1>{selectedCourse.title}</h1>
          <p>{selectedCourse.description}</p>
          
          <div className="course-meta">
            <span className="meta-item">
              ⏱️ {selectedCourse.duration}
            </span>
            <span className="meta-item">
              📚 {selectedCourse.chapters.length}章节
            </span>
            <span className="level-badge">
              {LEVEL_LABELS[selectedCourse.level]}
            </span>
          </div>
        </div>

        {/* 学习进度 */}
        <div className="learning-progress">
          <CircularProgress 
            value={progress?.percentComplete || 0}
            size={100}
            label="完成进度"
          />
          <StepProgress
            steps={selectedCourse.chapters.map(c => ({ label: c.title }))}
            currentStep={currentChapter}
            onStepClick={setCurrentChapter}
          />
        </div>

        {/* 当前章节内容 */}
        <div className="chapter-content">
          <h2>{chapter?.title}</h2>
          <p>时长: {chapter?.duration}</p>

          {chapter?.type === 'video' && (
            <VideoPlayer
              src={`/videos/${selectedCourse.id}/${chapter.id}.mp4`}
              title={chapter.title}
              onProgress={(p, t) => console.log(`Progress: ${p}%, Time: ${t}s`)}
              onComplete={() => handleChapterComplete(selectedCourse.id, chapter.id)}
            />
          )}

          {chapter?.type === 'exercise' && (
            <div className="exercise-section">
              <CodeEditor
                initialCode="// 在这里编写你的代码\n"
                language="javascript"
                onRun={(code) => console.log('Running code:', code)}
              />
              <button 
                className="submit-btn"
                onClick={() => handleChapterComplete(selectedCourse.id, chapter.id)}
              >
                提交答案
              </button>
            </div>
          )}

          {chapter?.type === 'reading' && (
            <div className="reading-section">
              <p>阅读内容加载中...</p>
            </div>
          )}
        </div>

        {/* 导航按钮 */}
        <div className="chapter-nav">
          <button
            className="nav-btn"
            disabled={currentChapter === 0}
            onClick={() => setCurrentChapter(prev => prev - 1)}
          >
            上一章
          </button>
          
          <button
            className="nav-btn primary"
            disabled={currentChapter === selectedCourse.chapters.length - 1}
            onClick={() => {
              handleChapterComplete(selectedCourse.id, chapter.id);
              setCurrentChapter(prev => prev + 1);
            }}
          >
            下一章
          </button>
        </div>
      </div>
    );
  };

  return (
    <div className="education-center">
      {/* 头部 */}
      <header className="education-header">
        <div className="header-content">
          <h1>🎓 AI Platform 学习中心</h1>
          <p>系统化学习AI Platform平台开发</p>
        </div>
        
        <div className="header-stats">
          <div className="stat-item">
            <span className="stat-value">{COURSES.length}</span>
            <span className="stat-label">课程总数</span>
          </div>
          <div className="stat-item">
            <span className="stat-value">
              {Object.values(userProgress).filter(p => p.percentComplete === 100).length}
            </span>
            <span className="stat-label">已完成</span>
          </div>
          <div className="stat-item">
            <span className="stat-value">{totalProgress.toFixed(0)}%</span>
            <span className="stat-label">总体进度</span>
          </div>
        </div>
      </header>

      {/* 标签页导航 */}
      <nav className="tab-navigation">
        <button 
          className={`tab-btn ${activeTab === 'courses' ? 'active' : ''}`}
          onClick={() => setActiveTab('courses')}
        >
          📚 全部课程
        </button>
        <button 
          className={`tab-btn ${activeTab === 'my-learning' ? 'active' : ''}`}
          onClick={() => setActiveTab('my-learning')}
        >
          📖 我的学习
        </button>
        <button 
          className={`tab-btn ${activeTab === 'achievements' ? 'active' : ''}`}
          onClick={() => setActiveTab('achievements')}
        >
          🏆 成就
        </button>
      </nav>

      {/* 搜索和筛选 */}
      <div className="filter-section">
        <input
          type="text"
          className="search-input"
          placeholder="搜索课程..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
        />
        
        <div className="category-filter">
          <button
            className={`category-btn ${!selectedCategory ? 'active' : ''}`}
            onClick={() => setSelectedCategory(null)}
          >
            全部
          </button>
          {COURSE_CATEGORIES.map(cat => (
            <button
              key={cat.id}
              className={`category-btn ${selectedCategory === cat.id ? 'active' : ''}`}
              onClick={() => setSelectedCategory(cat.id)}
            >
              {cat.icon} {cat.name}
            </button>
          ))}
        </div>
      </div>

      {/* 主要内容区 */}
      <main className="education-content">
        {selectedCourse ? (
          renderCourseDetail()
        ) : (
          <>
            <div className="section-header">
              <h2>
                {selectedCategory 
                  ? COURSE_CATEGORIES.find(c => c.id === selectedCategory)?.name 
                  : '全部课程'}
              </h2>
              <span className="course-count">{filteredCourses.length}个课程</span>
            </div>
            {renderCourseGrid()}
          </>
        )}
      </main>

      {/* 页脚 */}
      <footer className="education-footer">
        <p>© 2024 AI Platform 教育中心 | 让学习更简单</p>
      </footer>
    </div>
  );
};

export default EducationCenter;
