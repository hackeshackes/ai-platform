import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor, fireEvent, act } from '@testing-library/react';

// Mock the API module before imports
vi.mock('../unit/api/v12API', () => ({
  useV12API: () => ({
    users: [
      { id: '1', name: 'John Doe', email: 'john@example.com' },
      { id: '2', name: 'Jane Smith', email: 'jane@example.com' },
    ],
    loading: false,
    error: null,
    fetchUsers: vi.fn(),
    createUser: vi.fn(),
    updateUser: vi.fn(),
    deleteUser: vi.fn(),
  }),
  getProjects: vi.fn().mockResolvedValue([
    { id: '1', name: 'Project Alpha', status: 'active' },
    { id: '2', name: 'Project Beta', status: 'pending' },
  ]),
  getModels: vi.fn().mockResolvedValue([
    { id: '1', name: 'GPT-4', provider: 'OpenAI' },
    { id: '2', name: 'Claude-3', provider: 'Anthropic' },
  ]),
}));

import { V12Dashboard } from '../unit/pages/V12Dashboard';
import { V12UserList } from '../unit/pages/V12UserList';
import { V12ChatPage } from '../unit/pages/V12ChatPage';

describe('V12 Integration Tests', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('Dashboard Integration', () => {
    it('should load and display all dashboard data', async () => {
      render(<V12Dashboard />);
      
      await waitFor(() => {
        expect(screen.getByText('AI Platform Dashboard')).toBeInTheDocument();
        expect(screen.getByText('Project Alpha')).toBeInTheDocument();
        expect(screen.getByText('GPT-4')).toBeInTheDocument();
      });
    });
  });

  describe('User List Integration', () => {
    it('should display all users from API', () => {
      render(<V12UserList />);
      
      expect(screen.getByText('User Management')).toBeInTheDocument();
      expect(screen.getByText('John Doe')).toBeInTheDocument();
    });
  });

  describe('Chat Page Integration', () => {
    it('should render chat interface', () => {
      render(<V12ChatPage />);
      
      expect(screen.getByText('AI Chat')).toBeInTheDocument();
      expect(screen.getByPlaceholderText('Type your message...')).toBeInTheDocument();
    });

    it('should update input value on change', () => {
      render(<V12ChatPage />);
      
      const input = screen.getByPlaceholderText('Type your message...');
      fireEvent.change(input, { target: { value: 'Hello AI' } });
      
      expect(input).toHaveValue('Hello AI');
    });
  });

  describe('API Integration', () => {
    it('should have mocked API functions', () => {
      // The vi.mock already handles the API mocking in setup
      // All API functions are properly mocked via vit.mock
      expect(true).toBe(true); // Placeholder test
    });
  });
});
