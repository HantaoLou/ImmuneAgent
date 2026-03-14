import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { SessionList } from '@/components/sidebar/SessionList';
import { Session } from '@/types';

const createMockSession = (id: string, title: string, messageCount: number = 0): Session => ({
  id,
  title,
  messages: Array(messageCount).fill(null).map((_, i) => ({
    id: `msg-${i}`,
    role: 'user' as const,
    content: 'Message',
    timestamp: Date.now(),
    status: 'success' as const,
  })),
  createdAt: Date.now(),
});

describe('SessionList Component', () => {
  const mockOnSelect = vi.fn();
  const mockOnDelete = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('Empty state', () => {
    it('should render empty state when no sessions', () => {
      render(
        <SessionList
          sessions={[]}
          activeSessionId={null}
          onSessionSelect={mockOnSelect}
          onSessionDelete={mockOnDelete}
        />
      );

      expect(screen.getByText('暂无会话，点击新建会话开始聊天')).toBeInTheDocument();
    });

    it('should show empty state icon', () => {
      render(
        <SessionList
          sessions={[]}
          activeSessionId={null}
          onSessionSelect={mockOnSelect}
          onSessionDelete={mockOnDelete}
        />
      );

      expect(screen.getByText('💬')).toBeInTheDocument();
    });
  });

  describe('Session list rendering', () => {
    it('should render single session', () => {
      const sessions = [createMockSession('session-1', 'Test Session')];

      render(
        <SessionList
          sessions={sessions}
          activeSessionId={null}
          onSessionSelect={mockOnSelect}
          onSessionDelete={mockOnDelete}
        />
      );

      expect(screen.getByText('Test Session')).toBeInTheDocument();
    });

    it('should render multiple sessions', () => {
      const sessions = [
        createMockSession('session-1', 'Session 1'),
        createMockSession('session-2', 'Session 2'),
        createMockSession('session-3', 'Session 3'),
      ];

      render(
        <SessionList
          sessions={sessions}
          activeSessionId={null}
          onSessionSelect={mockOnSelect}
          onSessionDelete={mockOnDelete}
        />
      );

      expect(screen.getByText('Session 1')).toBeInTheDocument();
      expect(screen.getByText('Session 2')).toBeInTheDocument();
      expect(screen.getByText('Session 3')).toBeInTheDocument();
    });

    it('should display message count for each session', () => {
      const sessions = [
        createMockSession('session-1', 'Session 1', 5),
        createMockSession('session-2', 'Session 2', 10),
      ];

      render(
        <SessionList
          sessions={sessions}
          activeSessionId={null}
          onSessionSelect={mockOnSelect}
          onSessionDelete={mockOnDelete}
        />
      );

      expect(screen.getByText(/5 条消息/)).toBeInTheDocument();
      expect(screen.getByText(/10 条消息/)).toBeInTheDocument();
    });
  });

  describe('Active session highlighting', () => {
    it('should highlight active session', () => {
      const sessions = [
        createMockSession('session-1', 'Session 1'),
        createMockSession('session-2', 'Session 2'),
      ];

      const { container } = render(
        <SessionList
          sessions={sessions}
          activeSessionId="session-1"
          onSessionSelect={mockOnSelect}
          onSessionDelete={mockOnDelete}
        />
      );

      const activeItems = container.querySelectorAll('.active');
      expect(activeItems).toHaveLength(1);
    });

    it('should not highlight any session when activeSessionId is null', () => {
      const sessions = [
        createMockSession('session-1', 'Session 1'),
        createMockSession('session-2', 'Session 2'),
      ];

      const { container } = render(
        <SessionList
          sessions={sessions}
          activeSessionId={null}
          onSessionSelect={mockOnSelect}
          onSessionDelete={mockOnDelete}
        />
      );

      const activeItems = container.querySelectorAll('.active');
      expect(activeItems).toHaveLength(0);
    });
  });

  describe('Session selection', () => {
    it('should call onSessionSelect when session is clicked', async () => {
      const user = userEvent.setup();
      const sessions = [createMockSession('session-1', 'Test Session')];

      render(
        <SessionList
          sessions={sessions}
          activeSessionId={null}
          onSessionSelect={mockOnSelect}
          onSessionDelete={mockOnDelete}
        />
      );

      await user.click(screen.getByText('Test Session'));

      expect(mockOnSelect).toHaveBeenCalledWith('session-1');
    });

    it('should not call onSessionSelect for empty state', () => {
      render(
        <SessionList
          sessions={[]}
          activeSessionId={null}
          onSessionSelect={mockOnSelect}
          onSessionDelete={mockOnDelete}
        />
      );

      expect(mockOnSelect).not.toHaveBeenCalled();
    });
  });

  describe('Session deletion', () => {
    it('should pass onDelete handler to each SessionItem', () => {
      const sessions = [createMockSession('session-1', 'Test Session')];

      render(
        <SessionList
          sessions={sessions}
          activeSessionId={null}
          onSessionSelect={mockOnSelect}
          onSessionDelete={mockOnDelete}
        />
      );

      expect(screen.getByRole('button', { name: /delete/i })).toBeInTheDocument();
    });
  });

  describe('Edge cases', () => {
    it('should handle many sessions', () => {
      const sessions = Array(50)
        .fill(null)
        .map((_, i) => createMockSession(`session-${i}`, `Session ${i}`));

      render(
        <SessionList
          sessions={sessions}
          activeSessionId={null}
          onSessionSelect={mockOnSelect}
          onSessionDelete={mockOnDelete}
        />
      );

      expect(screen.getByText('Session 0')).toBeInTheDocument();
      expect(screen.getByText('Session 49')).toBeInTheDocument();
    });

    it('should handle sessions with same title', () => {
      const sessions = [
        createMockSession('session-1', 'Same Title'),
        createMockSession('session-2', 'Same Title'),
      ];

      render(
        <SessionList
          sessions={sessions}
          activeSessionId={null}
          onSessionSelect={mockOnSelect}
          onSessionDelete={mockOnDelete}
        />
      );

      const sameTitles = screen.getAllByText('Same Title');
      expect(sameTitles).toHaveLength(2);
    });

    it('should handle rapid re-renders', () => {
      const sessions = [createMockSession('session-1', 'Test Session')];

      const { rerender } = render(
        <SessionList
          sessions={sessions}
          activeSessionId={null}
          onSessionSelect={mockOnSelect}
          onSessionDelete={mockOnDelete}
        />
      );

      rerender(
        <SessionList
          sessions={sessions}
          activeSessionId="session-1"
          onSessionSelect={mockOnSelect}
          onSessionDelete={mockOnDelete}
        />
      );

      expect(screen.getByText('Test Session')).toBeInTheDocument();
    });
  });
});
