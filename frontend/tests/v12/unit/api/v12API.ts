import { vi, describe, it, expect } from 'vitest';

// API Client Functions
export const getUsers = async (): Promise<any[]> => {
  const response = await fetch('/api/v12/users');
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || 'Failed to fetch users');
  return data.data;
};

export const getUserById = async (id: string): Promise<any> => {
  const response = await fetch(`/api/v12/users/${id}`);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || 'Failed to fetch user');
  return data.data;
};

export const createUser = async (user: { name: string; email: string }): Promise<any> => {
  const response = await fetch('/api/v12/users', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(user),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || 'Failed to create user');
  return data.data;
};

export const updateUser = async (id: string, user: any): Promise<any> => {
  const response = await fetch(`/api/v12/users/${id}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(user),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || 'Failed to update user');
  return data.data;
};

export const deleteUser = async (id: string): Promise<void> => {
  const response = await fetch(`/api/v12/users/${id}`, {
    method: 'DELETE',
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || 'Failed to delete user');
};

export const getProjects = async (): Promise<any[]> => {
  const response = await fetch('/api/v12/projects');
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || 'Failed to fetch projects');
  return data.data;
};

export const getModels = async (): Promise<any[]> => {
  const response = await fetch('/api/v12/models');
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || 'Failed to fetch models');
  return data.data;
};

export const sendChatMessage = async (message: { role: string; content: string }): Promise<any> => {
  const response = await fetch('/api/v12/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(message),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || 'Failed to send message');
  return data.data;
};

// React Hook for API
import { useState, useCallback } from 'react';

export const useV12API = () => {
  const [users, setUsers] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchUsers = useCallback(async () => {
    setLoading(true);
    try {
      const data = await getUsers();
      setUsers(data);
      setError(null);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, []);

  const createUser = useCallback(async (user: { name: string; email: string }) => {
    const newUser = await createUser(user);
    setUsers(prev => [...prev, newUser]);
    return newUser;
  }, []);

  const updateUser = useCallback(async (id: string, user: any) => {
    const updated = await updateUser(id, user);
    setUsers(prev => prev.map(u => u.id === id ? updated : u));
    return updated;
  }, []);

  const deleteUser = useCallback(async (id: string) => {
    await deleteUser(id);
    setUsers(prev => prev.filter(u => u.id !== id));
  }, []);

  return {
    users,
    loading,
    error,
    fetchUsers,
    createUser,
    updateUser,
    deleteUser,
  };
};
