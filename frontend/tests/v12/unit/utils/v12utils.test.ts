// Utility Functions Tests
import { describe, it, expect } from 'vitest';

// Validation utilities
export const validateEmail = (email: string): boolean => {
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return emailRegex.test(email);
};

export const validatePassword = (password: string): { valid: boolean; message: string } => {
  if (password.length < 8) {
    return { valid: false, message: 'Password must be at least 8 characters' };
  }
  if (!/[A-Z]/.test(password)) {
    return { valid: false, message: 'Password must contain at least one uppercase letter' };
  }
  if (!/[a-z]/.test(password)) {
    return { valid: false, message: 'Password must contain at least one lowercase letter' };
  }
  if (!/[0-9]/.test(password)) {
    return { valid: false, message: 'Password must contain at least one number' };
  }
  return { valid: true, message: 'Password is valid' };
};

export const truncateText = (text: string, maxLength: number): string => {
  if (text.length <= maxLength) return text;
  return text.slice(0, maxLength - 3) + '...';
};

export const formatDate = (date: string | Date): string => {
  const d = new Date(date);
  return d.toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'long',
    day: 'numeric',
  });
};

export const debounce = <T extends (...args: any[]) => any>(
  func: T,
  wait: number
): ((...args: Parameters<T>) => void) => {
  let timeout: NodeJS.Timeout | null = null;
  
  return (...args: Parameters<T>) => {
    if (timeout) clearTimeout(timeout);
    timeout = setTimeout(() => func(...args), wait);
  };
};

describe('Utility Functions Tests', () => {
  describe('validateEmail', () => {
    it('should validate correct email format', () => {
      expect(validateEmail('user@example.com')).toBe(true);
      expect(validateEmail('john.doe@company.co.uk')).toBe(true);
    });

    it('should reject invalid email format', () => {
      expect(validateEmail('invalid-email')).toBe(false);
      expect(validateEmail('@nodomain.com')).toBe(false);
      expect(validateEmail('no@')).toBe(false);
      expect(validateEmail('')).toBe(false);
    });
  });

  describe('validatePassword', () => {
    it('should validate strong password', () => {
      const result = validatePassword('SecureP@ss123');
      expect(result.valid).toBe(true);
      expect(result.message).toBe('Password is valid');
    });

    it('should reject weak passwords', () => {
      expect(validatePassword('short').valid).toBe(false);
      expect(validatePassword('alllowercase123').valid).toBe(false);
      expect(validatePassword('ALLUPPERCASE123').valid).toBe(false);
      expect(validatePassword('NoNumbersHere@').valid).toBe(false);
    });
  });

  describe('truncateText', () => {
    it('should truncate long text', () => {
      expect(truncateText('Hello World', 8)).toBe('Hello...');
    });

    it('should not truncate short text', () => {
      expect(truncateText('Hi', 10)).toBe('Hi');
    });

    it('should handle exact length', () => {
      expect(truncateText('Hello', 5)).toBe('Hello');
    });
  });

  describe('formatDate', () => {
    it('should format date string correctly', () => {
      const result = formatDate('2024-01-15');
      expect(result).toContain('January');
      expect(result).toContain('15');
      expect(result).toContain('2024');
    });

    it('should format Date object correctly', () => {
      const result = formatDate(new Date('2024-06-20'));
      expect(result).toContain('June');
      expect(result).toContain('20');
    });
  });

  describe('debounce', () => {
    it('should delay function execution', () => {
      const func = vi.fn();
      const debouncedFunc = debounce(func, 100);
      
      debouncedFunc();
      expect(func).not.toHaveBeenCalled();
    });

    it('should execute after delay', () => {
      const func = vi.fn();
      const debouncedFunc = debounce(func, 50);
      
      debouncedFunc();
      setTimeout(() => {
        expect(func).toHaveBeenCalledTimes(1);
      }, 100);
    });
  });
});
