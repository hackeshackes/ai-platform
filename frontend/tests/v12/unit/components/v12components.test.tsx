import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { V12Button, V12Card, V12Input } from './V12Components';

describe('V12 Component Tests', () => {
  describe('V12Button', () => {
    it('should render button with correct text', () => {
      render(<V12Button>Click Me</V12Button>);
      expect(screen.getByRole('button')).toHaveTextContent('Click Me');
    });

    it('should call onClick when clicked', () => {
      const handleClick = vi.fn();
      render(<V12Button onClick={handleClick}>Click Me</V12Button>);
      
      fireEvent.click(screen.getByRole('button'));
      
      expect(handleClick).toHaveBeenCalledTimes(1);
    });

    it('should be disabled when disabled prop is true', () => {
      render(<V12Button disabled>Disabled Button</V12Button>);
      expect(screen.getByRole('button')).toBeDisabled();
    });

    it('should have correct data-testid', () => {
      render(<V12Button>Test</V12Button>);
      expect(screen.getByTestId('v12-button')).toBeInTheDocument();
    });
  });

  describe('V12Card', () => {
    it('should render card with title', () => {
      render(<V12Card title="Test Card">Card Content</V12Card>);
      expect(screen.getByText('Test Card')).toBeInTheDocument();
    });

    it('should render children content', () => {
      render(<V12Card>Card Content</V12Card>);
      expect(screen.getByText('Card Content')).toBeInTheDocument();
    });

    it('should have correct data-testid', () => {
      render(<V12Card>Test</V12Card>);
      expect(screen.getByTestId('v12-card')).toBeInTheDocument();
    });
  });

  describe('V12Input', () => {
    it('should render input with label', () => {
      render(<V12Input label="Username" />);
      expect(screen.getByLabelText('Username')).toBeInTheDocument();
    });

    it('should update value on change', () => {
      render(<V12Input />);
      const input = screen.getByRole('textbox');
      
      fireEvent.change(input, { target: { value: 'test value' } });
      
      expect(input).toHaveValue('test value');
    });

    it('should be disabled when disabled prop is true', () => {
      render(<V12Input disabled />);
      expect(screen.getByRole('textbox')).toBeDisabled();
    });

    it('should have correct data-testid', () => {
      render(<V12Input />);
      expect(screen.getByTestId('v12-input')).toBeInTheDocument();
    });
  });
});
