import React, { useState } from 'react';
import { sendChatMessage } from '../api/v12API';

interface Message {
  id: string;
  role: string;
  content: string;
  timestamp: string;
}

export const V12ChatPage: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSend = async () => {
    if (!inputValue.trim() || loading) return;

    const userMessage: Message = {
      id: `msg-${Date.now()}`,
      role: 'user',
      content: inputValue,
      timestamp: new Date().toISOString(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInputValue('');
    setLoading(true);

    try {
      const response = await sendChatMessage({
        role: 'user',
        content: userMessage.content,
      });

      const assistantMessage: Message = {
        id: response.id,
        role: response.role,
        content: response.content,
        timestamp: response.timestamp,
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (error) {
      console.error('Failed to send message:', error);
      // Add error message
      const errorMessage: Message = {
        id: `error-${Date.now()}`,
        role: 'assistant',
        content: 'Failed to send message. Please try again.',
        timestamp: new Date().toISOString(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div data-testid="v12-chat-page" className="v12-chat-page">
      <h1 data-testid="chat-title">AI Chat</h1>
      <div className="chat-container">
        <div className="chat-messages">
          {messages.length === 0 ? (
            <p className="empty-state">Start a conversation with the AI...</p>
          ) : (
            messages.map((message) => (
              <div
                key={message.id}
                className={`message message-${message.role}`}
                data-testid={`message-${message.id}`}
              >
                <div className="message-content">{message.content}</div>
                <div className="message-timestamp">
                  {new Date(message.timestamp).toLocaleTimeString()}
                </div>
              </div>
            ))
          )}
        </div>
        <div className="chat-input-container">
          <input
            type="text"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Type your message..."
            data-testid="chat-input"
            disabled={loading}
          />
          <button
            onClick={handleSend}
            disabled={loading || !inputValue.trim()}
            data-testid="send-button"
          >
            {loading ? 'Sending...' : 'Send'}
          </button>
        </div>
      </div>
    </div>
  );
};
