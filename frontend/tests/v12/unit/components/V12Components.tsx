import React from 'react';
import { ButtonHTMLAttributes } from 'react';

interface V12ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'danger';
  size?: 'small' | 'medium' | 'large';
  children: React.ReactNode;
}

export const V12Button: React.FC<V12ButtonProps> = ({
  variant = 'primary',
  size = 'medium',
  children,
  className = '',
  disabled,
  onClick,
  ...props
}) => {
  return (
    <button
      data-testid="v12-button"
      className={`v12-button v12-button-${variant} v12-button-${size} ${className}`}
      disabled={disabled}
      onClick={onClick}
      {...props}
    >
      {children}
    </button>
  );
};

interface V12CardProps {
  title?: string;
  children: React.ReactNode;
  shadow?: 'none' | 'small' | 'medium' | 'large';
  className?: string;
}

export const V12Card: React.FC<V12CardProps> = ({
  title,
  children,
  shadow = 'medium',
  className = '',
}) => {
  return (
    <div data-testid="v12-card" className={`v12-card v12-card-shadow-${shadow} ${className}`}>
      {title && <div className="v12-card-title">{title}</div>}
      <div className="v12-card-content">{children}</div>
    </div>
  );
};

import React, { InputHTMLAttributes } from 'react';

interface V12InputProps extends InputHTMLAttributes<HTMLInputElement> {
  label?: string;
  error?: string;
  size?: 'small' | 'medium' | 'large';
}

export const V12Input: React.FC<V12InputProps> = ({
  label,
  error,
  size = 'medium',
  className = '',
  id,
  ...props
}) => {
  const inputId = id || `v12-input-${Math.random().toString(36).substr(2, 9)}`;
  
  return (
    <div className={`v12-input-wrapper v12-input-${size} ${className}`}>
      {label && (
        <label htmlFor={inputId} className="v12-input-label">
          {label}
        </label>
      )}
      <input
        id={inputId}
        data-testid="v12-input"
        className="v12-input"
        {...props}
      />
      {error && (
        <span className="v12-input-error" data-testid="v12-input-error">
          {error}
        </span>
      )}
    </div>
  );
};
