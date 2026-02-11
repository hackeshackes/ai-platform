/// <reference types="cypress" />

describe('V12 E2E Tests', () => {
  beforeEach(() => {
    // Mock API calls before each test
    cy.intercept('GET', '/api/v12/users', {
      statusCode: 200,
      body: {
        success: true,
        data: [
          { id: '1', name: 'John Doe', email: 'john@example.com' },
          { id: '2', name: 'Jane Smith', email: 'jane@example.com' },
        ],
      },
    }).as('getUsers');

    cy.intercept('GET', '/api/v12/projects', {
      statusCode: 200,
      body: {
        success: true,
        data: [
          { id: '1', name: 'Project Alpha', status: 'active' },
          { id: '2', name: 'Project Beta', status: 'pending' },
        ],
      },
    }).as('getProjects');

    cy.intercept('POST', '/api/v12/chat', {
      statusCode: 200,
      body: {
        success: true,
        data: {
          id: 'msg-1',
          role: 'assistant',
          content: 'This is a test response from the AI.',
          timestamp: new Date().toISOString(),
        },
      },
    }).as('sendMessage');
  });

  describe('Dashboard Flow', () => {
    it('should load dashboard and display projects', () => {
      cy.visit('/dashboard');
      cy.wait('@getProjects');
      
      cy.contains('AI Platform Dashboard').should('be.visible');
      cy.contains('Project Alpha').should('be.visible');
      cy.contains('Project Beta').should('be.visible');
    });

    it('should display model cards', () => {
      cy.intercept('GET', '/api/v12/models', {
        statusCode: 200,
        body: {
          success: true,
          data: [
            { id: '1', name: 'GPT-4', provider: 'OpenAI' },
            { id: '2', name: 'Claude-3', provider: 'Anthropic' },
          ],
        },
      }).as('getModels');

      cy.visit('/dashboard');
      cy.wait(['@getProjects', '@getModels']);
      
      cy.contains('GPT-4').should('be.visible');
      cy.contains('Claude-3').should('be.visible');
    });
  });

  describe('User Management Flow', () => {
    it('should navigate to user list page', () => {
      cy.visit('/users');
      cy.wait('@getUsers');
      
      cy.contains('User Management').should('be.visible');
      cy.contains('John Doe').should('be.visible');
    });

    it('should add a new user', () => {
      cy.intercept('POST', '/api/v12/users', {
        statusCode: 200,
        body: {
          success: true,
          data: { id: '3', name: 'New User', email: 'new@example.com' },
        },
      }).as('createUser');

      cy.visit('/users');
      cy.contains('Add User').click();
      cy.wait('@createUser');
      
      cy.contains('New User').should('be.visible');
    });

    it('should delete a user', () => {
      cy.intercept('DELETE', '/api/v12/users/1', {
        statusCode: 200,
        body: { success: true, message: 'User deleted' },
      }).as('deleteUser');

      cy.visit('/users');
      cy.contains('Delete').first().click();
      cy.wait('@deleteUser');
      
      cy.contains('John Doe').should('not.exist');
    });
  });

  describe('Chat Flow', () => {
    it('should send message and receive response', () => {
      cy.visit('/chat');
      
      cy.get('[data-testid="chat-input"]').type('Hello, AI!');
      cy.get('[data-testid="send-button"]').click();
      cy.wait('@sendMessage');
      
      cy.contains('Hello, AI!').should('be.visible');
      cy.contains('This is a test response from the AI.').should('be.visible');
    });

    it('should show loading state while sending', () => {
      cy.intercept('POST', '/api/v12/cy.intercept', {
        delay: 1000,
        statusCode: 200,
        body: {
          success: true,
          data: {
            id: 'msg-1',
            role: 'assistant',
            content: 'Response',
            timestamp: new Date().toISOString(),
          },
        },
      }).as('slowMessage');

      cy.visit('/chat');
      cy.get('[data-testid="chat-input"]').type('Slow message');
      cy.get('[data-testid="send-button"]').click();
      
      cy.get('[data-testid="send-button"]').should('be.disabled');
      cy.contains('Sending...').should('be.visible');
    });
  });

  describe('Navigation Flow', () => {
    it('should navigate between pages', () => {
      // Visit dashboard
      cy.visit('/dashboard');
      cy.contains('AI Platform Dashboard').should('be.visible');
      
      // Navigate to users
      cy.contains('User Management').click();
      cy.url().should('include', '/users');
      
      // Navigate to chat
      cy.contains('AI Chat').click();
      cy.url().should('include', '/chat');
    });

    it('should preserve state when navigating back', () => {
      cy.visit('/users');
      cy.wait('@getUsers');
      cy.contains('John Doe').should('be.visible');
      
      cy.visit('/dashboard');
      cy.contains('AI Platform Dashboard').should('be.visible');
      
      cy.go('back');
      cy.contains('User Management').should('be.visible');
    });
  });

  describe('Form Validation', () => {
    it('should disable send button for empty message', () => {
      cy.visit('/chat');
      
      cy.get('[data-testid="send-button"]').should('be.disabled');
    });

    it('should enable send button after typing', () => {
      cy.visit('/chat');
      
      cy.get('[data-testid="chat-input"]').type('Hello');
      cy.get('[data-testid="send-button"]').should('not.be.disabled');
    });
  });

  describe('Error Handling', () => {
    it('should show error message when API fails', () => {
      cy.intercept('GET', '/api/v12/users', {
        statusCode: 500,
        body: { success: false, error: 'Internal server error' },
      }).as('getUsersError');

      cy.visit('/users');
      cy.wait('@getUsersError');
      
      cy.contains('Error: Internal server error').should('be.visible');
    });

    it('should retry on failed request', () => {
      let callCount = 0;
      cy.intercept('GET', '/api/v12/users', (req) => {
        callCount++;
        if (callCount === 1) {
          req.reply({ statusCode: 500, body: { error: 'Failed' } });
        } else {
          req.reply({
            statusCode: 200,
            body: {
              success: true,
              data: [{ id: '1', name: 'John Doe', email: 'john@example.com' }],
            },
          });
        }
      }).as('getUsersRetry');

      cy.visit('/users');
      
      // First attempt fails
      cy.contains('Error: Failed').should('be.visible');
      
      // Retry button should appear
      cy.contains('Retry').click();
      
      // Second attempt succeeds
      cy.wait('@getUsersRetry');
      cy.contains('John Doe').should('be.visible');
    });
  });
});
