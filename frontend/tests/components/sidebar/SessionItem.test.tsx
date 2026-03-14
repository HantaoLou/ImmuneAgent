import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { SessionItem } from '@/components/sidebar/SessionItem';
import { Session } from '@/types';

const createMockSession = (overrides?: Partial<Session>): Session => ({
  id: 'session-1',
  title: 'Test Session',
  messages: [],
  createdAt: Date.now(),
  ...overrides,
});

describe('SessionItem Component', () => {
  const mockSession = createMockSession();
  const mockOnSelect = vi.fn();
  const mockOnDelete = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('Basic rendering', () => {
    it('should render session title', () => {
      render(
        <SessionItem
          session={mockSession}
          isActive={false}
          onSelect={mockOnSelect}
          onDelete={mockOnDelete}
        />
      );

      expect(screen.getByText('Test Session')).toBeInTheDocument();
    });

    it('should render message count', () => {
      const sessionWithMessages = createMockSession({
        messages: [
          { id: 'msg-1', role: 'user', content: 'Hello', timestamp: Date.now(), status: 'success' },
          { id: 'msg-2', role: 'agent', content: 'Hi', timestamp: Date.now(), status: 'success' },
        ],
      });

      render(
        <SessionItem
          session={sessionWithMessages}
          isActive={false}
          onSelect={mockOnSelect}
          onDelete={mockOnDelete}
        />
      );

      expect(screen.getByText(/2 条消息/)).toBeInTheDocument();
    });

    it('should render zero messages correctly', () => {
      render(
        <SessionItem
          session={mockSession}
          isActive={false}
          onSelect={mockOnSelect}
          onDelete={mockOnDelete}
        />
      );

      expect(screen.getByText(/0 条消息/)).toBeInTheDocument();
    });
  });

  describe('Active state', () => {
    it('should apply active class when active', () => {
      const { container } = render(
        <SessionItem
          session={mockSession}
          isActive={true}
          onSelect={mockOnSelect}
          onDelete={mockOnDelete}
        />
      );

      expect(container.querySelector('.active')).toBeInTheDocument();
    });

    it('should not apply active class when inactive', () => {
      const { container } = render(
        <SessionItem
          session={mockSession}
          isActive={false}
          onSelect={mockOnSelect}
          onDelete={mockOnDelete}
        />
      );

      expect(container.querySelector('.active')).not.toBeInTheDocument();
    });
  });

  describe('Click handling', () => {
    it('should call onSelect when clicked', async () => {
      const user = userEvent.setup();

      render(
        <SessionItem
          session={mockSession}
          isActive={false}
          onSelect={mockOnSelect}
          onDelete={mockOnDelete}
        />
      );

      await user.click(screen.getByText('Test Session'));

      expect(mockOnSelect).toHaveBeenCalledWith('session-1');
    });

    it('should call onSelect once', async () => {
      const user = userEvent.setup();

      render(
        <SessionItem
          session={mockSession}
          isActive={false}
          onSelect={mockOnSelect}
          onDelete={mockOnDelete}
        />
      );

      await user.click(screen.getByText('Test Session'));

      expect(mockOnSelect).toHaveBeenCalledTimes(1);
    });
  });

  describe('Delete functionality', () => {
    it('should render delete button', () => {
      render(
        <SessionItem
          session={mockSession}
          isActive={false}
          onSelect={mockOnSelect}
          onDelete={mockOnDelete}
        />
      );

      expect(screen.getByRole('button', { name: /delete/i })).toBeInTheDocument();
    });

    it('should show confirmation when delete is clicked', async () => {
      const user = userEvent.setup();

      render(
        <SessionItem
          session={mockSession}
          isActive={false}
          onSelect={mockOnSelect}
          onDelete={mockOnDelete}
        />
      );

      const deleteButton = screen.getByRole('button', { name: /delete/i });
      await user.click(deleteButton);

      expect(screen.getByText('确定删除此会话吗？')).toBeInTheDocument();
    });

    it('should call onDelete when confirmed', async () => {
      const user = userEvent.setup();

      render(
        <SessionItem
          session={mockSession}
          isActive={false}
          onSelect={mockOnSelect}
          onDelete={mockOnDelete}
        />
      );

      const deleteButton = screen.getByRole('button', { name: /delete/i });
      await user.click(deleteButton);

      const confirmButton = screen.getByText('确定');
      await user.click(confirmButton);

      expect(mockOnDelete).toHaveBeenCalledWith('session-1');
    });

    it('should not call onDelete when cancelled', async () => {
      const user = userEvent.setup();

      render(
        <SessionItem
          session={mockSession}
          isActive={false}
          onSelect={mockOnSelect}
          onDelete={mockOnDelete}
        />
      );

      const deleteButton = screen.getByRole('button', { name: /delete/i });
      await user.click(deleteButton);

      const cancelButton = screen.getByText('取消');
      await user.click(cancelButton);

      expect(mockOnDelete).not.toHaveBeenCalled();
    });

    it('should not trigger onSelect when delete button is clicked', async () => {
      const user = userEvent.setup();

      render(
        <SessionItem
          session={mockSession}
          isActive={false}
          onSelect={mockOnSelect}
          onDelete={mockOnDelete}
        />
      );

      const deleteButton = screen.getByRole('button', { name: /delete/i });
      await user.click(deleteButton);

      expect(mockOnSelect).not.toHaveBeenCalled();
    });
  });

  describe('Edge cases', () => {
    it('should handle long session titles', () => {
      const longTitleSession = createMockSession({
        title: 'This is a very long session title that might need truncation',
      });

      render(
        <SessionItem
          session={longTitleSession}
          isActive={false}
          onSelect={mockOnSelect}
          onDelete={mockOnDelete}
        />
      );

      expect(screen.getByText(/This is a very long session title/)).toBeInTheDocument();
    });

    it('should handle special characters in title', () => {
      const specialSession = createMockSession({
        title: 'Session with special chars: <>&"\'',
      });

      render(
        <SessionItem
          session={specialSession}
          isActive={false}
          onSelect={mockOnSelect}
          onDelete={mockOnDelete}
        />
      );

      expect(screen.getByText(/Session with special chars/)).toBeInTheDocument();
    });

    it('should handle many messages', () => {
      const manyMessagesSession = createMockSession({
        messages: Array(100).fill(null).map((_, i) => ({
          id: `msg-${i}`,
          role: 'user' as const,
          content: 'Message',
          timestamp: Date.now(),
          status: 'success' as const,
        })),
      });

      render(
        <SessionItem
          session={manyMessagesSession}
          isActive={false}
          onSelect={mockOnSelect}
          onDelete={mockOnDelete}
        />
      );

      expect(screen.getByText(/100 条消息/)).toBeInTheDocument();
    });
  });
});
