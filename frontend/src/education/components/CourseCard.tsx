import React, { useState, useCallback, useRef, useEffect } from 'react';
import { Course, Chapter, LEVEL_LABELS, LEVEL_COLORS } from '../types';
import './styles.css';

interface CourseCardProps {
  course: Course;
  progress?: number;
  isLocked?: boolean;
  onClick?: (course: Course) => void;
  onContinue?: (course: Course) => void;
}

export const CourseCard: React.FC<CourseCardProps> = ({
  course,
  progress = 0,
  isLocked = false,
  onClick,
  onContinue
}) => {
  const [isHovered, setIsHovered] = useState(false);
  const cardRef = useRef<HTMLDivElement>(null);

  const handleClick = useCallback(() => {
    if (!isLocked && onClick) {
      onClick(course);
    }
  }, [course, isLocked, onClick]);

  const handleContinue = useCallback((e: React.MouseEvent) => {
    e.stopPropagation();
    if (onContinue) {
      onContinue(course);
    }
  }, [course, onContinue]);

  const formatDuration = (minutes: number): string => {
    if (minutes < 60) {
      return `${minutes}分钟`;
    }
    const hours = Math.floor(minutes / 60);
    const remainingMinutes = minutes % 60;
    if (remainingMinutes === 0) {
      return `${hours}小时`;
    }
    return `${hours}小时${remainingMinutes}分钟`;
  };

  return (
    <div
      ref={cardRef}
      className={`course-card ${isLocked ? 'locked' : ''} ${isHovered ? 'hovered' : ''}`}
      onClick={handleClick}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      {/* 课程封面 */}
      <div className="course-thumbnail">
        {course.thumbnail ? (
          <img src={course.thumbnail} alt={course.title} className="thumbnail-image" />
        ) : (
          <div className="thumbnail-placeholder">
            <span className="course-icon">
              {course.category === 'getting-started' && '🚀'}
              {course.category === 'agents' && '🤖'}
              {course.category === 'pipelines' && '🔗'}
              {course.category === 'advanced' && '⚡'}
              {course.category === 'development' && '📝'}
              {course.category === 'support' && '🔧'}
              {course.category === 'optimization' && '💨'}
              {course.category === 'templates' && '📋'}
              {course.category === 'integration' && '🔌'}
              {course.category === 'project' && '🎯'}
            </span>
          </div>
        )}
        
        {/* 难度标签 */}
        <div 
          className="level-badge"
          style={{ backgroundColor: LEVEL_COLORS[course.level] }}
        >
          {LEVEL_LABELS[course.level]}
        </div>

        {/* 锁定覆盖层 */}
        {isLocked && (
          <div className="locked-overlay">
            <span className="lock-icon">🔒</span>
            <span className="lock-text">请先完成前置课程</span>
          </div>
        )}
      </div>

      {/* 课程内容 */}
      <div className="course-content">
        <h3 className="course-title">{course.title}</h3>
        <p className="course-description">{course.description}</p>

        {/* 课程信息 */}
        <div className="course-meta">
          <div className="meta-item">
            <span className="meta-icon">⏱️</span>
            <span className="meta-text">{formatDuration(course.durationMinutes)}</span>
          </div>
          <div className="meta-item">
            <span className="meta-icon">📚</span>
            <span className="meta-text">{course.chapters.length}章节</span>
          </div>
        </div>

        {/* 标签 */}
        <div className="course-tags">
          {course.tags.slice(0, 3).map((tag, index) => (
            <span key={index} className="tag">
              {tag}
            </span>
          ))}
        </div>

        {/* 进度条 */}
        {progress > 0 && (
          <div className="course-progress">
            <div className="progress-header">
              <span className="progress-label">学习进度</span>
              <span className="progress-percent">{progress}%</span>
            </div>
            <div className="progress-bar-container">
              <div 
                className="progress-bar-fill"
                style={{ width: `${progress}%` }}
              />
            </div>
          </div>
        )}

        {/* 操作按钮 */}
        <div className="course-actions">
          {progress > 0 && progress < 100 ? (
            <button className="action-btn continue-btn" onClick={handleContinue}>
              <span className="btn-icon">▶️</span>
              继续学习
            </button>
          ) : progress === 100 ? (
            <button className="action-btn completed-btn">
              <span className="btn-icon">✅</span>
              完成学习
            </button>
          ) : (
            <button className="action-btn start-btn">
              <span className="btn-icon">🚀</span>
              开始学习
            </button>
          )}
        </div>
      </div>
    </div>
  );
};

export default CourseCard;
