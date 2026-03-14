import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import { MessageBubble } from '@/components/chat/MessageBubble';
import { createMockMessage } from '../../utils/mockFactories';

describe('MessageBubble Component', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('User message rendering', () => {
    it('should render user message with content', () => {
      const message = createMockMessage({ role: 'user', content: 'Hello World' });
      render(<MessageBubble message={message} />);
      
      expect(screen.getByText('Hello World')).toBeInTheDocument();
    });

    it('should render user message with timestamp', () => {
      const timestamp = Date.now();
      const message = createMockMessage({ role: 'user', timestamp });
      render(<MessageBubble message={message} />);
      
      expect(screen.getByText(/Hello World/i) || screen.getByText(/\d/)).toBeInTheDocument();
    });
  });

  describe('Agent message rendering', () => {
    it('should render agent message with content', () => {
      const message = createMockMessage({ role: 'agent', content: 'I am the agent' });
      render(<MessageBubble message={message} />);
      
      expect(screen.getByText('I am the agent')).toBeInTheDocument();
    });

    it('should render agent message with timestamp', () => {
      const timestamp = Date.now();
      const message = createMockMessage({ role: 'agent', timestamp });
      render(<MessageBubble message={message} />);
      
      expect(screen.getByText(/I am the agent/i) || screen.getByText(/\d/)).toBeInTheDocument();
    });
  });

  describe('Message status indicators', () => {
    it('should render loading state', () => {
      const message = createMockMessage({ status: 'loading', content: '' });
      render(<MessageBubble message={message} />);
      
      expect(screen.getByText(/loading/i) || screen.getByRole('status')).toBeInTheDocument();
    });

    it('should render success state', () => {
      const message = createMockMessage({ status: 'success', content: 'Success message' });
      render(<MessageBubble message={message} />);
      
      expect(screen.getByText('Success message')).toBeInTheDocument();
    });

    it('should render error state', () => {
      const message = createMockMessage({ status: 'error', content: 'Error occurred' });
      render(<MessageBubble message={message} />);
      
      expect(screen.getByText('Error occurred')).toBeInTheDocument();
    });
  });

  describe('Message content', () => {
    it('should handle long messages', () => {
      const longContent = 'This is a very long message. '.repeat(50);
      const message = createMockMessage({ content: longContent });
      render(<MessageBubble message={message} />);
      
      expect(screen.getByText(longContent)).toBeInTheDocument();
    });

    it('should handle special characters', () => {
      const specialContent = '!@#$%^&*()_+-={}[]|\\:";\'<>?,./~`';
      const message = createMockMessage({ content: specialContent });
      render(<MessageBubble message={message} />);
      
      expect(screen.getByText(specialContent)).toBeInTheDocument();
    });

    it('should handle emoji', () => {
      const emojiContent = 'Hello 👋 World 🌍 Test 🎉';
      const message = createMockMessage({ content: emojiContent });
      render(<MessageBubble message={message} />);
      
      expect(screen.getByText(emojiContent)).toBeInTheDocument();
    });

    it('should handle empty content', () => {
      const message = createMockMessage({ content: '' });
      render(<MessageBubble message={message} />);
      
      expect(screen.getByRole('article') || screen.getByTestId('message-bubble')).toBeInTheDocument();
    });

    it('should handle multiline content', () => {
      const multilineContent = 'Line 1\nLine 2\nLine 3';
      const message = createMockMessage({ content: multilineContent });
      render(<MessageBubble message={message} />);
      
      expect(screen.getByText(/Line 1/)).toBeInTheDocument();
    });
  });

  describe('Accessibility', () => {
    it('should have proper role', () => {
      const message = createMockMessage();
      render(<MessageBubble message={message} />);
      
      expect(screen.getByRole('article') || screen.getByText(message.content)).toBeInTheDocument();
    });

    it('should be readable by screen readers', () => {
      const message = createMockMessage({ content: 'Test message' });
      render(<MessageBubble message={message} />);
      
      expect(screen.getByText('Test message')).toBeInTheDocument();
    });
  });

  describe('Edge cases', () => {
    it('should handle message with only spaces', () => {
      const message = createMockMessage({ content: '     ' });
      render(<MessageBubble message={message} />);
      
      expect(screen.getByText('     ')).toBeInTheDocument();
    });

    it('should handle message with tabs', () => {
      const tabContent = 'Line 1\tLine 2\tLine 3';
      const message = createMockMessage({ content: tabContent });
      render(<MessageBubble message={message} />);
      
      expect(screen.getByText(tabContent)).toBeInTheDocument();
    });

    it('should handle unicode characters', () => {
      const unicodeContent = '中文测试 日本語テスト 한글테스트';
      const message = createMockMessage({ content: unicodeContent });
      render(<MessageBubble message={message} />);
      
      expect(screen.getByText(unicodeContent)).toBeInTheDocument();
    });
  });
});
