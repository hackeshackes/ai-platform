import { describe, it, expect, vi } from 'vitest';
import { render, screen, waitFor, act } from '@testing-library/react';

// Mock the API module before imports
vi.mock('../api/v12API', () => ({
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

import { V12Dashboard } from './V12Dashboard';
import { V12UserList } from './V12UserList';
import { V12ChatPage } from './V12ChatPage';

describe('V12 Page Tests', () => {
  describe('V12Dashboard', () => {
    it('should render dashboard with title', async () => {
      render(<V12Dashboard />);
      
      await waitFor(() => {
        expect(screen.getByText('AI Platform Dashboard')).toBeInTheDocument();
      });
    });

    it('should render project cards', async () => {
      render(<V12Dashboard />);
      
      await waitFor(() => {
        expect(screen.getByText('Project Alpha')).toBeInTheDocument();
        expect(screen.getByText('Project Beta')).toBeInTheDocument();
      });
    });

    it('should render model cards', async () => {
      render(<V12Dashboard />);
      
      await waitFor(() => {
        expect(screen.getByText('GPT-4')).toBeInTheDocument();
        expect(screen.getByText('Claude-3')).toBeInTheDocument();
      });
    });
  });

  describe('V12UserList', () => {
    it('should render user list title', () => {
      render(<V12UserList />);
      expect(screen.getByText('User Management')).toBeInTheDocument();
    });

    it('should render user items', () => {
      render(<V12UserList />);
      expect(screen.getByText('John Doe')).toBeInTheDocument();
      expect(screen.getByText('Jane Smith')).toBeInTheDocument();
    });

    it('should render user emails', () => {
      render(<V12UserList />);
      expect(screen.getByText('john@example.com')).toBeInTheDocument();
      expect(screen.getByText('jane@example.com')).toBeInTheDocument();
    });
  });

  describe('V12ChatPage', () => {
    it('should render chat page with title', () => {
      render(<V12ChatPage />);
      expect(screen.getByText('AI Chat')).toBeInTheDocument();
    });

    it('should render message input', () => {
      render(<V12ChatPage />);
      expect(screen.getByPlaceholderText('Type your message...')).toBeInTheDocument();
    });

    it('should render send button', () => {
      render(<V12ChatPage />);
      expect(screen.getByText('Send')).toBeInTheDocument();
    });
  });
});
