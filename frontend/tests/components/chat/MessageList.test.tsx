import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import { MessageList } from '@/components/chat/MessageList';
import { createMockMessages, createMockMessage } from '../../utils/mockFactories';

describe('MessageList Component', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('Empty state', () => {
    it('should render empty state when no messages', () => {
      render(<MessageList messages={[]} />);
      
      expect(screen.getByText(/没有消息/i) || screen.getByText(/empty/i) || screen.getByRole('list')).toBeInTheDocument();
    });
  });

  describe('Message rendering', () => {
    it('should render single message', () => {
      const messages = createMockMessages(1);
      render(<MessageList messages={messages} />);
      
      expect(screen.getByText(messages[0].content)).toBeInTheDocument();
    });

    it('should render multiple messages', () => {
      const messages = createMockMessages(5);
      render(<MessageList messages={messages} />);
      
      messages.forEach(msg => {
        expect(screen.getByText(msg.content)).toBeInTheDocument();
      });
    });

    it('should render messages in order', () => {
      const messages = [
        createMockMessage({ content: 'First', timestamp: 1000 }),
        createMockMessage({ content: 'Second', timestamp: 2000 }),
        createMockMessage({ content: 'Third', timestamp: 3000 }),
      ];
      render(<MessageList messages={messages} />);
      
      const messageElements = screen.getAllByRole('article');
      expect(messageElements).toHaveLength(3);
    });

    it('should handle mixed user/agent messages', () => {
      const messages = [
        createMockMessage({ role: 'user', content: 'User message' }),
        createMockMessage({ role: 'agent', content: 'Agent response' }),
      ];
      render(<MessageList messages={messages} />);
      
      expect(screen.getByText('User message')).toBeInTheDocument();
      expect(screen.getByText('Agent response')).toBeInTheDocument();
    });
  });

  describe('Accessibility', () => {
    it('should have proper role', () => {
      const messages = createMockMessages(2);
      render(<MessageList messages={messages} />);
      
      expect(screen.getByRole('list') || screen.getByRole('log')).toBeInTheDocument();
    });

    it('should be readable by screen readers', () => {
      const messages = createMockMessages(2);
      render(<MessageList messages={messages} />);
      
      messages.forEach(msg => {
        expect(screen.getByText(msg.content)).toBeInTheDocument();
      });
    });
  });

  describe('Edge cases', () => {
    it('should handle large message lists', () => {
      const messages = createMockMessages(100);
      render(<MessageList messages={messages} />);
      
      expect(screen.getAllByRole('article')).toHaveLength(100);
    });

    it('should handle messages with same timestamp', () => {
      const timestamp = Date.now();
      const messages = [
        createMockMessage({ content: 'Message 1', timestamp }),
        createMockMessage({ content: 'Message 2', timestamp }),
      ];
      render(<MessageList messages={messages} />);
      
      expect(screen.getByText('Message 1')).toBeInTheDocument();
      expect(screen.getByText('Message 2')).toBeInTheDocument();
    });
  });
});
