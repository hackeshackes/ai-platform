import React, { useState, useCallback, useMemo } from 'react';
import './styles.css';

interface ProgressBarProps {
  value: number;
  max?: number;
  label?: string;
  showPercentage?: boolean;
  showValue?: boolean;
  size?: 'small' | 'medium' | 'large';
  variant?: 'default' | 'success' | 'warning' | 'danger' | 'gradient';
  animated?: boolean;
  striped?: boolean;
  milestones?: number[];
  onMilestoneReached?: (milestone: number) => void;
  displayValue?: string;
  color?: string;
}

export const ProgressBar: React.FC<ProgressBarProps> = ({
  value,
  max = 100,
  label,
  showPercentage = true,
  showValue = false,
  size = 'medium',
  variant = 'default',
  animated = true,
  striped = false,
  milestones = [],
  onMilestoneReached,
  displayValue,
  color
}) => {
  const [previousValue, setPreviousValue] = useState(value);
  const percentage = Math.min(Math.max((value / max) * 100, 0), 100);

  // 检测里程碑
  useMemo(() => {
    if (onMilestoneReached) {
      milestones.forEach(milestone => {
        if (previousValue < milestone && value >= milestone) {
          onMilestoneReached(milestone);
        }
      });
    }
  }, [value, milestones, onMilestoneReached, previousValue]);

  // 更新前一个值
  useState(() => {
    setPreviousValue(value);
  });

  // 获取变体颜色
  const getVariantColor = (): string => {
    const colors: Record<string, string> = {
      default: '#3B82F6',
      success: '#10B981',
      warning: '#F59E0B',
      danger: '#EF4444',
      gradient: 'linear-gradient(90deg, #3B82F6 0%, #8B5CF6 100%)'
    };
    return colors[variant] || colors.default;
  };

  // 获取尺寸样式
  const getSizeStyle = (): React.CSSProperties => {
    const sizes: Record<string, React.CSSProperties> = {
      small: { height: '6px', fontSize: '12px' },
      medium: { height: '10px', fontSize: '14px' },
      large: { height: '16px', fontSize: '16px' }
    };
    return sizes[size] || sizes.medium;
  };

  const sizeStyle = getSizeStyle();
  const fillColor = color || getVariantColor();

  // 计算里程碑状态
  const getMilestoneStatus = (milestone: number): 'completed' | 'current' | 'upcoming' => {
    if (percentage >= milestone) return 'completed';
    if (percentage >= milestone - 10 && percentage < milestone) return 'current';
    return 'upcoming';
  };

  return (
    <div className={`progress-bar-wrapper ${animated ? 'animated' : ''}`}>
      {/* 标签 */}
      {label && (
        <div className="progress-label-row">
          <span className="progress-label-text">{label}</span>
          {showValue && displayValue && (
            <span className="progress-display-value">{displayValue}</span>
          )}
        </div>
      )}

      {/* 进度条容器 */}
      <div 
        className={`progress-bar-track ${striped ? 'striped' : ''} ${size}`}
        style={{ 
          ...sizeStyle,
          backgroundColor: `${fillColor}20`
        }}
      >
        {/* 进度填充 */}
        <div 
          className="progress-bar-fill"
          style={{ 
            width: `${percentage}%`,
            background: fillColor
          }}
        >
          {/* 动画效果 */}
          {animated && (
            <div className="progress-shimmer" />
          )}
        </div>

        {/* 里程碑标记 */}
        {milestones.map((milestone, index) => (
          <div
            key={index}
            className={`milestone-marker ${getMilestoneStatus(milestone)}`}
            style={{ left: `${milestone}%` }}
          >
            <div className="milestone-dot" />
            <span className="milestone-label">{milestone}%</span>
          </div>
        ))}
      </div>

      {/* 百分比显示 */}
      {showPercentage && !label && (
        <div className="progress-percentage">
          {percentage.toFixed(1)}%
        </div>
      )}
    </div>
  );
};

// 环形进度条组件
interface CircularProgressProps {
  value: number;
  max?: number;
  size?: number;
  strokeWidth?: number;
  variant?: 'default' | 'success' | 'warning' | 'danger';
  showLabel?: boolean;
  label?: string;
  animated?: boolean;
}

export const CircularProgress: React.FC<CircularProgressProps> = ({
  value,
  max = 100,
  size = 120,
  strokeWidth = 8,
  variant = 'default',
  showLabel = true,
  label,
  animated = true
}) => {
  const percentage = Math.min(Math.max((value / max) * 100, 0), 100);
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (percentage / 100) * circumference;

  const colors: Record<string, string> = {
    default: '#3B82F6',
    success: '#10B981',
    warning: '#F59E0B',
    danger: '#EF4444'
  };

  const getColor = (): string => colors[variant] || colors.default;

  return (
    <div className="circular-progress-container">
      <svg 
        className={`circular-progress ${animated ? 'animated' : ''}`}
        width={size} 
        height={size}
      >
        {/* 背景圆 */}
        <circle
          className="circular-bg"
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          strokeWidth={strokeWidth}
        />
        {/* 进度圆 */}
        <circle
          className="circular-progress-bar"
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          strokeWidth={strokeWidth}
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          strokeLinecap="round"
          style={{ 
            stroke: getColor(),
            transition: animated ? 'stroke-dashoffset 0.5s ease' : 'none'
          }}
        />
      </svg>
      
      {/* 中心标签 */}
      {showLabel && (
        <div 
          className="circular-label"
          style={{ 
            width: size - strokeWidth * 4,
            height: size - strokeWidth * 4
          }}
        >
          <span className="circular-value">{percentage.toFixed(0)}%</span>
          {label && <span className="circular-text">{label}</span>}
        </div>
      )}
    </div>
  );
};

// 步骤进度组件
interface StepProgressProps {
  steps: { label: string; description?: string }[];
  currentStep: number;
  onStepClick?: (step: number) => void;
  variant?: 'default' | 'bullet' | 'number';
}

export const StepProgress: React.FC<StepProgressProps> = ({
  steps,
  currentStep,
  onStepClick,
  variant = 'default'
}) => {
  const getStepStatus = (index: number): 'completed' | 'current' | 'upcoming' => {
    if (index < currentStep) return 'completed';
    if (index === currentStep) return 'current';
    return 'upcoming';
  };

  return (
    <div className={`step-progress-container ${variant}`}>
      {steps.map((step, index) => (
        <div 
          key={index}
          className={`step-item ${getStepStatus(index)}`}
          onClick={() => onStepClick && index <= currentStep && onStepClick(index)}
        >
          <div className="step-indicator">
            {getStepStatus(index) === 'completed' ? (
              <span className="step-check">✓</span>
            ) : (
              <span className="step-number">{index + 1}</span>
            )}
          </div>
          
          <div className="step-content">
            <span className="step-label">{step.label}</span>
            {step.description && (
              <span className="step-description">{step.description}</span>
            )}
          </div>

          {/* 连接线 */}
          {index < steps.length - 1 && (
            <div className={`step-connector ${getStepStatus(index)}`} />
          )}
        </div>
      ))}
    </div>
  );
};

export default ProgressBar;
