import React from 'react';
import { useV12API } from '../api/v12API';

export const V12UserList: React.FC = () => {
  const { users, loading, error, fetchUsers, createUser, updateUser, deleteUser } = useV12API();

  React.useEffect(() => {
    fetchUsers();
  }, [fetchUsers]);

  if (loading) {
    return <div>Loading users...</div>;
  }

  if (error) {
    return <div>Error: {error}</div>;
  }

  return (
    <div data-testid="v12-user-list" className="v12-user-list">
      <h1 data-testid="user-list-title">User Management</h1>
      <div className="user-actions">
        <button onClick={() => createUser({ name: 'New User', email: 'new@example.com' })}>
          Add User
        </button>
      </div>
      <div className="user-grid">
        {users.map((user) => (
          <div key={user.id} className="user-card" data-testid={`user-${user.id}`}>
            <h3>{user.name}</h3>
            <p>{user.email}</p>
            <div className="user-actions">
              <button onClick={() => updateUser(user.id, { name: user.name })}>
                Edit
              </button>
              <button onClick={() => deleteUser(user.id)}>Delete</button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
