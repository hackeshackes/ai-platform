import { http, HttpResponse } from 'msw';

export const handlers = [
  // API Handlers
  http.get('/api/v12/users', () => {
    return HttpResponse.json({
      success: true,
      data: [
        { id: '1', name: 'John Doe', email: 'john@example.com' },
        { id: '2', name: 'Jane Smith', email: 'jane@example.com' },
      ],
    });
  }),
  
  http.get('/api/v12/users/:id', ({ params }) => {
    const { id } = params;
    return HttpResponse.json({
      success: true,
      data: { id, name: 'John Doe', email: 'john@example.com' },
    });
  }),
  
  http.post('/api/v12/users', async ({ request }) => {
    const body = await request.json();
    return HttpResponse.json({
      success: true,
      data: { id: '3', ...body },
    });
  }),
  
  http.put('/api/v12/users/:id', async ({ request, params }) => {
    const body = await request.json();
    return HttpResponse.json({
      success: true,
      data: { id: params.id, ...body },
    });
  }),
  
  http.delete('/api/v12/users/:id', ({ params }) => {
    return HttpResponse.json({
      success: true,
      message: 'User deleted successfully',
    });
  }),
  
  // AI Platform API Handlers
  http.get('/api/v12/projects', () => {
    return HttpResponse.json({
      success: true,
      data: [
        { id: '1', name: 'Project Alpha', status: 'active' },
        { id: '2', name: 'Project Beta', status: 'pending' },
      ],
    });
  }),
  
  http.get('/api/v12/models', () => {
    return HttpResponse.json({
      success: true,
      data: [
        { id: '1', name: 'GPT-4', provider: 'OpenAI' },
        { id: '2', name: 'Claude-3', provider: 'Anthropic' },
      ],
    });
  }),
  
  http.post('/api/v12/chat', async ({ request }) => {
    const body = await request.json();
    return HttpResponse.json({
      success: true,
      data: {
        id: 'msg-1',
        role: 'assistant',
        content: 'This is a test response from the AI.',
        timestamp: new Date().toISOString(),
      },
    });
  }),
];
